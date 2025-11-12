#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
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
        seedColors.emplace_back(colorDist(gen), colorDist(gen), colorDist(gen), colorDist(gen));
    }
    
    std::cout << "Generated " << config.numSeeds << " seed points for color propagation" << std::endl;
    return {seedPoints, seedColors};
}

Vec4 calculateInfluencedColor(const Vec2& position, const Vec4& originalColor, float progress,
                             const std::vector<Vec2>& seedPoints, const std::vector<Vec4>& seedColors,
                             const AnimationConfig& config) {
    //TIME_FUNCTION;
    Vec4 newColor = originalColor;
    
    float maxDistance = std::max(config.width, config.height) * 0.6f;

    for (int s = 0; s < config.numSeeds; ++s) {
        float distance = position.distance(seedPoints[s]);
        float influence = std::max(0.0f, 1.0f - (distance / maxDistance));
        
        Vec2 direction = position - seedPoints[s];
        float angle = std::atan2(direction.y, direction.x);
        auto SC = seedColors[s];
        // applyDirectionalColorInfluence(newColor, seedColors[s], influence, progress, angle);
        float absAngle = std::abs(angle);
        if (absAngle < PI4) { // Right - affect alpha
            newColor.a = std::fmod(newColor.a + SC.a * influence * progress, 1.0f);
        } else if (absAngle > PI43) { // Left - affect blue
            newColor.b = std::fmod(newColor.b + SC.b * influence * progress, 1.0f);
        } else if (angle > 0) { // Below - affect green
            newColor.g = std::fmod(newColor.g + SC.g * influence * progress, 1.0f);
        } else { // Above - affect red
            newColor.r = std::fmod(newColor.r + SC.r * influence * progress, 1.0f);
        }
    }
    
    return newColor.clampColor();
}

void updateColorsForFrame(Grid2& grid, float progress, const std::vector<Vec2>& seedPoints, 
                         const std::vector<Vec4>& seedColors, const AnimationConfig& config) {
    TIME_FUNCTION;
    
    grid.bulkUpdateColors([&](size_t id, const Vec2& pos, const Vec4& currentColor) {
        return calculateInfluencedColor(pos, currentColor, progress, seedPoints, seedColors, config);
    });
}

std::vector<uint8_t> convertFrameToBGR(Grid2& grid, const AnimationConfig& config) {
    TIME_FUNCTION;
    int frameWidth, frameHeight;
    std::vector<int> bgrData;
    grid.getGridRegionAsBGR(Vec2(0,0),Vec2(512,512), frameWidth, frameHeight, bgrData);
    //grid.getGridRegionAsRGB(0.0f,0.0f,512.0f,512.0f,frameWidth,frameHeight,rgbData);
    
    std::vector<uint8_t> bgrFrame(frameWidth * frameHeight * 3);
    #pragma omp parallel for
    for (int i = 0; i < frameWidth * frameHeight; ++i) {
        int r = std::clamp(bgrData[i * 3 + 2], 0, 255);
        int g = std::clamp(bgrData[i * 3 + 1], 0, 255);
        int b = std::clamp(bgrData[i * 3], 0, 255);
        
        bgrFrame[i * 3] = static_cast<uint8_t>(b);
        bgrFrame[i * 3 + 1] = static_cast<uint8_t>(g);
        bgrFrame[i * 3 + 2] = static_cast<uint8_t>(r);
    }
    
    return bgrFrame;
}

bool validateFrameSize(const std::vector<uint8_t>& frame, const AnimationConfig& config) {
    return frame.size() == config.width * config.height * 3;
}

std::vector<std::vector<uint8_t>> createAnimationFrames(Grid2& grid, 
                                                       const std::vector<Vec2>& seedPoints,
                                                       const std::vector<Vec4>& seedColors,
                                                       const AnimationConfig& config) {
    TIME_FUNCTION;
    std::vector<std::vector<uint8_t>> frames;
    
    for (int frame = 0; frame < config.totalFrames; ++frame) {
        std::cout << "Processing frame " << frame + 1 << "/" << config.totalFrames << std::endl;
        
        float progress = static_cast<float>(frame) / (config.totalFrames - 1);
        updateColorsForFrame(grid, progress, seedPoints, seedColors, config);
        
        auto bgrFrame = convertFrameToBGR(grid, config);
        
        // if (!validateFrameSize(bgrFrame, config)) {
        //     std::cerr << "ERROR: Frame size mismatch in frame " << frame << std::endl;
        //     continue;
        // }
        
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
    
    // Configuration
    AnimationConfig config;
    
    // Initialize grid
    Grid2 grid;
    if (!initializeGrid(grid, config)) {
        return 1;
    }
    
    // Generate seed points
    auto [seedPoints, seedColors] = generateSeedPoints(config);
    
    // Create animation frames
    auto frames = createAnimationFrames(grid, seedPoints, seedColors, config);
    
    // Save animation
    if (!saveAnimation(frames, seedPoints, seedColors, config)) {
        return 1;
    }
    
    return 0;
}