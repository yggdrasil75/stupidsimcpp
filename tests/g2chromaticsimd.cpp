#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include "../util/grid/grid2.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/timing_decorator.cpp"

struct AnimationConfig {
    int width = 512;
    int height = 512;
    int totalFrames = 240;
    float fps = 30.0f;
    int numSeeds = 1;
};

const float PI4 = M_PI / 4.0f;
const float PI43 = PI4 * 3.0f;

struct SeedDataSoA {
    // Use alignas to ensure 16-byte alignment for SSE instructions
    std::vector<float> sx, sy; // Seed X and Y positions
    std::vector<float> sr, sg, sb, sa; // Seed R, G, B, A colors
    size_t count = 0;

    void reserve(size_t n) {
        sx.reserve(n); sy.reserve(n);
        sr.reserve(n); sg.reserve(n); sb.reserve(n); sa.reserve(n);
    }
};

SeedDataSoA convertSeedsToSoA(const std::vector<Vec2>& points, const std::vector<Vec4>& colors) {
    TIME_FUNCTION;
    SeedDataSoA soaData;
    size_t numSeeds = points.size();
    // Pad to the nearest multiple of 4 for clean SIMD loops
    size_t paddedSize = (numSeeds + 3) & ~3; 
    soaData.count = paddedSize;
    soaData.reserve(paddedSize);

    for (size_t i = 0; i < numSeeds; ++i) {
        soaData.sx.push_back(points[i].x);
        soaData.sy.push_back(points[i].y);
        soaData.sr.push_back(colors[i].r);
        soaData.sg.push_back(colors[i].g);
        soaData.sb.push_back(colors[i].b);
        soaData.sa.push_back(colors[i].a);
    }

    // Add padding elements
    for (size_t i = numSeeds; i < paddedSize; ++i) {
        soaData.sx.push_back(0); soaData.sy.push_back(0);
        soaData.sr.push_back(0); soaData.sg.push_back(0);
        soaData.sb.push_back(0); soaData.sa.push_back(0);
    }
    std::cout << "Converted " << numSeeds << " seeds to SoA format (padded to " << paddedSize << ")" << std::endl;
    return soaData;
}

// Helper for a horizontal add of a __m128 register
inline float horizontal_add(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v); // {v.z, v.z, v.w, v.w}
    __m128 sums = _mm_add_ps(v, shuf);    // {v.x+v.z, v.y+v.z, v.z+v.w, v.w+v.w}
    shuf = _mm_movehl_ps(shuf, sums); // {v.z+v.w, v.w+v.w, ...}
    sums = _mm_add_ss(sums, shuf);    // adds lowest float
    return _mm_cvtss_f32(sums);
}

// Non-optimized atan2 for a SIMD vector. For a real high-performance scenario,
// you would use a library like SLEEF or a polynomial approximation.
inline __m128 _mm_atan2_ps(__m128 y, __m128 x) {
    float y_lanes[4], x_lanes[4];
    _mm_storeu_ps(y_lanes, y);
    _mm_storeu_ps(x_lanes, x);
    for(int i = 0; i < 4; ++i) {
        y_lanes[i] = std::atan2(y_lanes[i], x_lanes[i]);
    }
    return _mm_loadu_ps(y_lanes);
}

bool initializeGrid(Grid2& grid, const AnimationConfig& config) {
    TIME_FUNCTION;
    std::cout << "Initializing grayscale grid..." << std::endl;
    
    for (int y = 1; y < config.height; ++y) {
        for (int x = 1; x < config.width; ++x) {
            float gradient = (x + y) / float(config.width + config.height - 2);
            Vec2 position(static_cast<float>(x), static_cast<float>(y));
            Vec4 color(gradient, gradient, gradient, 1.0f);
            grid.addObject(position, color, 1.0f);
        }
    }
    
    std::cout << "Grayscale grid created with " << config.width * config.height << " objects" << std::endl;
    return true;
}


std::pair<std::vector<Vec2>, std::vector<Vec4>> generateSeedPoints(const AnimationConfig& config) {
    TIME_FUNCTION;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDist(0, config.width - 1);
    std::uniform_int_distribution<> yDist(0, config.height - 1);
    std::uniform_real_distribution<> colorDist(0.2f, 0.8f);
    
    std::vector<Vec2> seedPoints;
    std::vector<Vec4> seedColors;
    
    for (int i = 0; i < config.numSeeds; ++i) {
        seedPoints.emplace_back(xDist(gen), yDist(gen));
        seedColors.emplace_back(colorDist(gen), colorDist(gen), colorDist(gen), 1.0f); // Alpha fixed to 1.0
    }
    std::cout << "Generated " << config.numSeeds << " seed points for color propagation" << std::endl;
    return {seedPoints, seedColors};
}

Vec4 calculateInfluencedColor(const Vec2& position, const Vec4& originalColor, float progress,
                                   const SeedDataSoA& seeds, const AnimationConfig& config) {
    const __m128 v_max_dist = _mm_set1_ps(std::max(config.width, config.height) * 0.6f);
    const __m128 v_progress = _mm_set1_ps(progress);
    const __m128 v_one = _mm_set1_ps(1.0f);
    const __m128 v_zero = _mm_setzero_ps();
    const __m128 v_pi4 = _mm_set1_ps(PI4);
    const __m128 v_pi43 = _mm_set1_ps(PI43);
    const __m128 v_neg_zero = _mm_set1_ps(-0.0f); // For fast absolute value

    // Broadcast pixel position into SIMD registers
    const __m128 v_pos_x = _mm_set1_ps(position.x);
    const __m128 v_pos_y = _mm_set1_ps(position.y);

    // Accumulators for color channel updates
    __m128 r_updates = _mm_setzero_ps();
    __m128 g_updates = _mm_setzero_ps();
    __m128 b_updates = _mm_setzero_ps();
    __m128 a_updates = _mm_setzero_ps();

    // Process 4 seeds at a time
    for (size_t s = 0; s < seeds.count; s += 4) {
        // --- Load data for 4 seeds ---
        const __m128 seed_x = _mm_load_ps(&seeds.sx[s]);
        const __m128 seed_y = _mm_load_ps(&seeds.sy[s]);
        
        // --- Calculate distance and influence ---
        const __m128 dir_x = _mm_sub_ps(v_pos_x, seed_x);
        const __m128 dir_y = _mm_sub_ps(v_pos_y, seed_y);
        const __m128 dist_sq = _mm_add_ps(_mm_mul_ps(dir_x, dir_x), _mm_mul_ps(dir_y, dir_y));
        const __m128 dist = _mm_sqrt_ps(dist_sq);
        __m128 influence = _mm_sub_ps(v_one, _mm_div_ps(dist, v_max_dist));
        influence = _mm_max_ps(v_zero, influence); // clamp to 0

        // --- Calculate full potential color contribution ---
        const __m128 influence_progress = _mm_mul_ps(influence, v_progress);
        const __m128 contrib_r = _mm_mul_ps(_mm_load_ps(&seeds.sr[s]), influence_progress);
        const __m128 contrib_g = _mm_mul_ps(_mm_load_ps(&seeds.sg[s]), influence_progress);
        const __m128 contrib_b = _mm_mul_ps(_mm_load_ps(&seeds.sb[s]), influence_progress);
        const __m128 contrib_a = _mm_mul_ps(_mm_load_ps(&seeds.sa[s]), influence_progress);

        // --- Branchless Masking based on Angle ---
        const __m128 angle = _mm_atan2_ps(dir_y, dir_x);
        const __m128 abs_angle = _mm_andnot_ps(v_neg_zero, angle); // fast abs

        // Create masks for each condition
        const __m128 mask_right = _mm_cmplt_ps(abs_angle, v_pi4);  // abs(angle) < PI4
        const __m128 mask_left = _mm_cmpgt_ps(abs_angle, v_pi43); // abs(angle) > PI43
        
        // The "else if" logic is tricky. We need to ensure mutual exclusivity.
        // mask_below is true if (angle > 0) AND NOT (right OR left)
        const __m128 is_not_rl = _mm_andnot_ps(mask_right, _mm_andnot_ps(mask_left, v_one));
        const __m128 mask_below = _mm_and_ps(_mm_cmpgt_ps(angle, v_zero), is_not_rl);

        // The "else" case (above) is whatever is left over.
        // mask_above is true if NOT (right OR left OR below)
        const __m128 mask_above = _mm_andnot_ps(mask_below, is_not_rl);

        // --- Accumulate updates using masks ---
        // Add contribution only where mask is active (all 1s)
        r_updates = _mm_add_ps(r_updates, _mm_and_ps(contrib_r, mask_above));
        g_updates = _mm_add_ps(g_updates, _mm_and_ps(contrib_g, mask_below));
        b_updates = _mm_add_ps(b_updates, _mm_and_ps(contrib_b, mask_left));
        a_updates = _mm_add_ps(a_updates, _mm_and_ps(contrib_a, mask_right));
    }

    // --- Horizontal Reduction: Sum the updates from all lanes ---
    Vec4 newColor = originalColor;
    newColor.r += horizontal_add(r_updates);
    newColor.g += horizontal_add(g_updates);
    newColor.b += horizontal_add(b_updates);
    newColor.a += horizontal_add(a_updates);

    // --- Apply fmod(x, 1.0) which is x - floor(x) for positive numbers ---
    // Can do this with SIMD as well for the final color vector
    __m128 final_color_v = _mm_loadu_ps(&newColor.r);
    final_color_v = _mm_sub_ps(final_color_v, _mm_floor_ps(final_color_v));
    _mm_storeu_ps(&newColor.r, final_color_v);
    
    return newColor.clampColor();
}

void updateColorsForFrame(Grid2& grid, float progress, const SeedDataSoA& seeds, const AnimationConfig& config) {
    TIME_FUNCTION;
    
    grid.bulkUpdateColors([&](size_t id, const Vec2& pos, const Vec4& currentColor) {
        return calculateInfluencedColor(pos, currentColor, progress, seeds, config);
    });
}

std::vector<uint8_t> convertFrameToBGR(Grid2& grid, const AnimationConfig& config) {
    TIME_FUNCTION;
    int frameWidth, frameHeight;
    std::vector<int> bgrData;
    grid.getGridRegionAsBGR(Vec2(0,0), Vec2(config.width, config.height), frameWidth, frameHeight, bgrData);
    std::vector<uint8_t> bgrFrame(frameWidth * frameHeight * 3);
    #pragma omp parallel for
    for (int i = 0; i < frameWidth * frameHeight; ++i) {
        bgrFrame[i * 3] = static_cast<uint8_t>(std::clamp(bgrData[i * 3], 0, 255));
        bgrFrame[i * 3 + 1] = static_cast<uint8_t>(std::clamp(bgrData[i * 3 + 1], 0, 255));
        bgrFrame[i * 3 + 2] = static_cast<uint8_t>(std::clamp(bgrData[i * 3 + 2], 0, 255));
    }
    return bgrFrame;
}


std::vector<std::vector<uint8_t>> createAnimationFrames(Grid2& grid, 
                                                       const SeedDataSoA& seeds,
                                                       const AnimationConfig& config) {
    TIME_FUNCTION;
    std::vector<std::vector<uint8_t>> frames;
    
    for (int frame = 0; frame < config.totalFrames; ++frame) {
        std::cout << "Processing frame " << frame + 1 << "/" << config.totalFrames << std::endl;
        
        float progress = static_cast<float>(frame) / (config.totalFrames - 1);
        updateColorsForFrame(grid, progress, seeds, config);
        
        std::vector<u_int8_t> bgrFrame = convertFrameToBGR(grid, config);
        frames.push_back(bgrFrame);
    }
    
    return frames;
}
void printSuccessMessage(const std::string& filename, 
                        const std::vector<Vec2>& seedPoints,
                        const std::vector<Vec4>& seedColors,
                        const AnimationConfig& config) {
    std::cout << "\nSuccessfully saved chromatic transformation animation to: " << filename << std::endl;
    std::cout << "Video details:" << std::endl;
    std::cout << "  - Dimensions: " << config.width << " x " << config.height << std::endl;
    std::cout << "  - Frames: " << config.totalFrames << " (" 
              << config.totalFrames/config.fps << " seconds at " << config.fps << "fps)" << std::endl;
    std::cout << "  - Seed points: " << config.numSeeds << std::endl;
    
    std::cout << "\nSeed points used:" << std::endl;
    for (int i = 0; i < config.numSeeds; ++i) {
        std::cout << "  Seed " << i + 1 << ": Position " << seedPoints[i] 
                  << ", Color " << seedColors[i].toColorString() << std::endl;
    }
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
}

void printErrorMessage(const std::vector<std::vector<uint8_t>>& frames, const AnimationConfig& config) {
    std::cerr << "Failed to save AVI file!" << std::endl;
    std::cerr << "Debug info:" << std::endl;
    std::cerr << "  - Frames count: " << frames.size() << std::endl;
    if (!frames.empty()) {
        std::cerr << "  - First frame size: " << frames[0].size() << std::endl;
        std::cerr << "  - Expected frame size: " << config.width * config.height * 3 << std::endl;
    }
    std::cerr << "  - Width: " << config.width << ", Height: " << config.height << std::endl;
}

bool saveAnimation(const std::vector<std::vector<uint8_t>>& frames, const std::vector<Vec2>& seedPoints, 
                  const std::vector<Vec4>& seedColors, const AnimationConfig& config) {
    TIME_FUNCTION;
    std::string filename = "output/chromatic_transformation.avi";
    std::cout << "Attempting to save AVI file: " << filename << std::endl;
    
    bool success = AVIWriter::saveAVI(filename, frames, config.width, config.height, config.fps);
    
    if (success) {
        printSuccessMessage(filename, seedPoints, seedColors, config);
        return true;
    } else {
        printErrorMessage(frames, config);
        return false;
    }
}


int main() {
    std::cout << "Creating chromatic transformation animation..." << std::endl;
    
    AnimationConfig config;
    
    Grid2 grid;
    if (!initializeGrid(grid, config)) return 1;
    
    auto [seedPoints, seedColors] = generateSeedPoints(config);
    auto seeds_SoA = convertSeedsToSoA(seedPoints, seedColors);
    
    // Create animation frames using the SIMD-friendly data
    auto frames = createAnimationFrames(grid, seeds_SoA, config);
    
    if (!saveAnimation(frames, seedPoints, seedColors, config)) return 1;
    
    return 0;
}