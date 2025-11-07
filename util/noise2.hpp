#ifndef NOISE2_HPP
#define NOISE2_HPP

#include "grid2.hpp"
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <array>
#include <vector>
#include <unordered_map>

struct Grad { float x, y; };

class Noise2 {
public:
    enum NoiseType {
        PERLIN,
        SIMPLEX,
        VALUE,
        WORLEY,
        GABOR,
        POISSON_DISK,
        FRACTAL,
        WAVELET,
        GAUSSIAN,
        CELLULAR
    };
    
    enum GradientType {
        HASH_BASED,
        SIN_BASED,
        DOT_BASED,
        PRECOMPUTED
    };

    Noise2(uint32_t seed = 0, NoiseType type = PERLIN, GradientType gradType = PRECOMPUTED) : 
            rng(seed), dist(0.0f, 1.0f), currentType(type), gradType(gradType),
            currentSeed(seed), gaborFrequency(4.0f), gaborBandwidth(0.5f) 
    {
        initializePermutationTable(seed);
        initializeFeaturePoints(64, seed); // Default 64 feature points
        initializeWaveletCoefficients(32, seed); // 32x32 wavelet coefficients
    }
    
    // Set random seed and reinitialize dependent structures
    void setSeed(uint32_t seed) {
        currentSeed = seed;
        rng.seed(seed);
        initializePermutationTable(seed);
        initializeFeaturePoints(featurePoints.size(), seed);
        initializeWaveletCoefficients(static_cast<int>(std::sqrt(waveletCoefficients.size())), seed);
    }
    
    // Set noise type
    void setNoiseType(NoiseType type) {
        currentType = type;
    }
    
    // Set gradient type
    void setGradientType(GradientType type) {
        gradType = type;
    }
    
    // Main noise function that routes to the selected algorithm
    float noise(float x, float y, int octaves = 1, float persistence = 0.5f, float lacunarity = 2.0f) {
        switch (currentType) {
            case PERLIN:
                return perlinNoise(x, y, octaves, persistence, lacunarity);
            case SIMPLEX:
                return simplexNoise(x, y, octaves, persistence, lacunarity);
            case VALUE:
                return valueNoise(x, y, octaves, persistence, lacunarity);
            case WORLEY:
                return worleyNoise(x, y);
            case GABOR:
                return gaborNoise(x, y);
            case POISSON_DISK:
                return poissonDiskNoise(x, y);
            case FRACTAL:
                return fractalNoise(x, y, octaves, persistence, lacunarity);
            case WAVELET:
                return waveletNoise(x, y);
            case GAUSSIAN:
                return gaussianNoise(x, y);
            case CELLULAR:
                return cellularNoise(x, y);
            default:
                return perlinNoise(x, y, octaves, persistence, lacunarity);
        }
    }
    
    // Generate simple value noise
    float valueNoise(float x, float y, int octaves = 1, float persistence = 0.5f, float lacunarity = 2.0f) {
        float total = 0.0f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float maxValue = 0.0f;
        
        for (int i = 0; i < octaves; i++) {
            total += rawNoise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        
        return total / maxValue;
    }
    
    // Generate Perlin-like noise
    float perlinNoise(float x, float y, int octaves = 1, float persistence = 0.5f, float lacunarity = 2.0f) {
        float total = 0.0f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float maxValue = 0.0f;
        
        for (int i = 0; i < octaves; i++) {
            total += improvedNoise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        
        return (total / maxValue + 1.0f) * 0.5f; // Normalize to [0,1]
    }
    
    float simplexNoise(float x, float y, int octaves = 1, float persistence = 0.5f, float lacunarity = 2.0f) {
        float total = 0.0f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float maxValue = 0.0f;
        
        for (int i = 0; i < octaves; i++) {
            total += rawSimplexNoise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        
        return (total / maxValue + 1.0f) * 0.5f;
    }
    
    // Worley (cellular) noise
    float worleyNoise(float x, float y) {
        if (featurePoints.empty()) return 0.0f;
        
        // Find the closest and second closest feature points
        float minDist1 = std::numeric_limits<float>::max();
        float minDist2 = std::numeric_limits<float>::max();
        
        for (const auto& point : featurePoints) {
            float dx = x - point.x;
            float dy = y - point.y;
            float dist = dx * dx + dy * dy; // Squared distance for performance
            
            if (dist < minDist1) {
                minDist2 = minDist1;
                minDist1 = dist;
            } else if (dist < minDist2) {
                minDist2 = dist;
            }
        }
        
        // Return distance to closest feature point (normalized)
        return std::sqrt(minDist1);
    }
    
    // Cellular noise variation
    float cellularNoise(float x, float y) {
        if (featurePoints.empty()) return 0.0f;
        
        float minDist1 = std::numeric_limits<float>::max();
        float minDist2 = std::numeric_limits<float>::max();
        
        for (const auto& point : featurePoints) {
            float dx = x - point.x;
            float dy = y - point.y;
            float dist = dx * dx + dy * dy;
            
            if (dist < minDist1) {
                minDist2 = minDist1;
                minDist1 = dist;
            } else if (dist < minDist2) {
                minDist2 = dist;
            }
        }
        
        // Cellular pattern: second closest minus closest
        return std::sqrt(minDist2) - std::sqrt(minDist1);
    }
    
    // Gabor noise
    float gaborNoise(float x, float y) {
        // Simplified Gabor noise - in practice this would be more complex
        float gaussian = std::exp(-(x*x + y*y) / (2.0f * gaborBandwidth * gaborBandwidth));
        float cosine = std::cos(2.0f * M_PI * gaborFrequency * (x + y));
        
        return gaussian * cosine;
    }
    
    // Poisson disk noise
    float poissonDiskNoise(float x, float y) {
        // Sample Poisson disk distribution
        // This is a simplified version - full implementation would use more sophisticated sampling
        float minDist = std::numeric_limits<float>::max();
        
        for (const auto& point : featurePoints) {
            float dx = x - point.x;
            float dy = y - point.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            minDist = std::min(minDist, dist);
        }
        
        return 1.0f - std::min(minDist * 10.0f, 1.0f); // Invert and scale
    }
    
    // Fractal noise (fractional Brownian motion)
    float fractalNoise(float x, float y, int octaves = 8, float persistence = 0.5f, float lacunarity = 2.0f) {
        float total = 0.0f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float maxValue = 0.0f;
        
        for (int i = 0; i < octaves; i++) {
            total += improvedNoise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        
        // Fractal noise often has wider range, so we don't normalize as strictly
        return total;
    }
    
    // Wavelet noise
    float waveletNoise(float x, float y) {
        // Simplified wavelet noise using precomputed coefficients
        int ix = static_cast<int>(std::floor(x * 4)) % 32;
        int iy = static_cast<int>(std::floor(y * 4)) % 32;
        
        if (ix < 0) ix += 32;
        if (iy < 0) iy += 32;
        
        return waveletCoefficients[iy * 32 + ix];
    }
    
    // Gaussian noise
    float gaussianNoise(float x, float y) {
        // Use coordinates to seed RNG for deterministic results
        rng.seed(static_cast<uint32_t>(x * 1000 + y * 1000 + currentSeed));
        
        // Box-Muller transform for Gaussian distribution
        float u1 = dist(rng);
        float u2 = dist(rng);
        float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        
        // Normalize to [0,1] range
        return (z0 + 3.0f) / 6.0f; // Assuming 3 sigma covers most of the distribution
    }
    
    // Generate a grayscale noise grid using current noise type
    Grid2 generateGrayNoise(int width, int height, 
                           float scale = 1.0f, 
                           int octaves = 1, 
                           float persistence = 0.5f, 
                           uint32_t seed = 0,
                           const Vec2& offset = Vec2(0, 0)) {
        if (seed != 0) setSeed(seed);
        
        Grid2 grid(width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = (x + offset.x) / width * scale;
                float ny = (y + offset.y) / height * scale;
                
                float noiseValue = noise(nx, ny, octaves, persistence);
                
                // Convert to position and grayscale color
                Vec2 position(x, y);
                Vec4 color(noiseValue, noiseValue, noiseValue, 1.0f);
                
                grid.positions[y * width + x] = position;
                grid.colors[y * width + x] = color;
            }
        }
        
        return grid;
    }

    float pascalTri(const float& a, const float& b) {
        TIME_FUNCTION;
        int result = 1;
        for (int i = 0; i < b; ++i){
            result *= (a - 1) / (i + 1);
        }
        return result;
    }

    float genSmooth(int N, float x) {
        TIME_FUNCTION;
        x = clamp(x, 0, 1);
        float result = 0;
        for (int n = 0; n <= N; ++n){
            result += pascalTri(-N - 1, n) * pascalTri(2 * N + 1, N-1) * pow(x, N + n + 1);
        }
        return result;
    }

    float inverse_smoothstep(float x) {
        TIME_FUNCTION;
        return 0.5 - sin(asin(1.0 - 2.0 * x) / 3.0);
    }
    
    // Generate multi-layered RGBA noise
    Grid2 generateRGBANoise(int width, int height,
                           const Vec4& scale = Vec4(1.0f, 1.0f, 1.0f, 1.0f),
                           const Vec4& octaves = Vec4(1.0f, 1.0f, 1.0f, 1.0f),
                           const Vec4& persistence = Vec4(0.5f, 0.5f, 0.5f, 0.5f),
                           uint32_t seed = 0,
                           const Vec2& offset = Vec2(0, 0)) {
        if (seed != 0) setSeed(seed);
        
        Grid2 grid(width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = (x + offset.x) / width;
                float ny = (y + offset.y) / height;
                
                // Generate separate noise for each channel using current noise type
                float r = noise(nx * scale.x, ny * scale.x, 
                               static_cast<int>(octaves.x), persistence.x);
                float g = noise(nx * scale.y, ny * scale.y, 
                               static_cast<int>(octaves.y), persistence.y);
                float b = noise(nx * scale.z, ny * scale.z, 
                               static_cast<int>(octaves.z), persistence.z);
                float a = noise(nx * scale.w, ny * scale.w, 
                               static_cast<int>(octaves.w), persistence.w);
                
                Vec2 position(x, y);
                Vec4 color(r, g, b, a);
                
                grid.positions[y * width + x] = position;
                grid.colors[y * width + x] = color;
            }
        }
        
        return grid;
    }
    
    // Generate terrain-like noise (useful for heightmaps)
    Grid2 generateTerrainNoise(int width, int height, float scale = 1.0f, int octaves = 4,
                              float persistence = 0.5f, uint32_t seed = 0, const Vec2& offset = Vec2(0, 0)) {
        if (seed != 0) setSeed(seed);
        
        Grid2 grid(width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = (x + offset.x) / width * scale;
                float ny = (y + offset.y) / height * scale;
                
                // Use multiple octaves for more natural terrain
                float heightValue = noise(nx, ny, octaves, persistence);
                
                // Apply some curve to make it more terrain-like
                heightValue = std::pow(heightValue, 1.5f);
                
                Vec2 position(x, y);
                Vec4 color(heightValue, heightValue, heightValue, 1.0f);
                
                grid.positions[y * width + x] = position;
                grid.colors[y * width + x] = color;
            }
        }
        
        return grid;
    }
    
    // Generate cloud-like noise
    Grid2 generateCloudNoise(int width, int height,
                            float scale = 2.0f,
                            int octaves = 3,
                            float persistence = 0.7f,
                            uint32_t seed = 0,
                            const Vec2& offset = Vec2(0, 0)) {
        auto grid = generateGrayNoise(width, height, scale, octaves, persistence, seed, offset);
        
        // Apply soft threshold for cloud effect
        for (auto& color : grid.colors) {
            float value = color.x; // Assuming grayscale in red channel
            // Soft threshold: values below 0.3 become 0, above 0.7 become 1, smooth in between
            if (value < 0.3f) value = 0.0f;
            else if (value > 0.7f) value = 1.0f;
            else value = (value - 0.3f) / 0.4f; // Linear interpolation
            
            color = Vec4(value, value, value, 1.0f);
        }
        
        return grid;
    }
    
    // Generate specific noise type directly
    Grid2 generateSpecificNoise(NoiseType type, int width, int height,
                               float scale = 1.0f, int octaves = 1,
                               float persistence = 0.5f, uint32_t seed = 0) {
        NoiseType oldType = currentType;
        currentType = type;
        
        auto grid = generateGrayNoise(width, height, scale, octaves, persistence, seed);
        
        currentType = oldType;
        return grid;
    }

private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
    // Precomputed gradient directions for 8 directions
    static constexpr std::array<Grad, 8> grads = {
        Grad{1.0f, 0.0f}, 
        Grad{0.707f, 0.707f}, 
        Grad{0.0f, 1.0f}, 
        Grad{-0.707f, 0.707f},
        Grad{-1.0f, 0.0f}, 
        Grad{-0.707f, -0.707f}, 
        Grad{0.0f, -1.0f}, 
        Grad{0.707f, -0.707f}
    };

    NoiseType currentType;
    GradientType gradType;
    uint32_t currentSeed;
    
    // Permutation table for Simplex noise
    std::array<int, 512> perm;
    
    // For Worley noise
    std::vector<Vec2> featurePoints;
    
    // For Gabor noise
    float gaborFrequency;
    float gaborBandwidth;
    
    // For wavelet noise
    std::vector<float> waveletCoefficients;
    
    // Initialize permutation table for Simplex noise
    void initializePermutationTable(uint32_t seed) {
        std::mt19937 localRng(seed);
        std::uniform_int_distribution<int> intDist(0, 255);
        
        // Create initial permutation
        std::array<int, 256> p;
        for (int i = 0; i < 256; i++) {
            p[i] = i;
        }
        
        // Shuffle using Fisher-Yates
        for (int i = 255; i > 0; i--) {
            int j = intDist(localRng) % (i + 1);
            std::swap(p[i], p[j]);
        }
        
        // Duplicate for overflow
        for (int i = 0; i < 512; i++) {
            perm[i] = p[i & 255];
        }
    }
    
    // Initialize feature points for Worley/Poisson noise
    void initializeFeaturePoints(int numPoints, uint32_t seed) {
        std::mt19937 localRng(seed);
        std::uniform_real_distribution<float> localDist(0.0f, 1.0f);
        
        featurePoints.clear();
        featurePoints.reserve(numPoints);
        
        for (int i = 0; i < numPoints; i++) {
            featurePoints.emplace_back(localDist(localRng), localDist(localRng));
        }
    }
    
    // Initialize wavelet coefficients
    void initializeWaveletCoefficients(int size, uint32_t seed) {
        std::mt19937 localRng(seed);
        std::uniform_real_distribution<float> localDist(-1.0f, 1.0f);
        
        waveletCoefficients.resize(size * size);
        for (int i = 0; i < size * size; i++) {
            waveletCoefficients[i] = (localDist(localRng) + 1.0f) * 0.5f; // Normalize to [0,1]
        }
    }
    
    // Raw Simplex noise implementation
    float rawSimplexNoise(float x, float y) {
        // Skewing factors for 2D
        const float F2 = 0.5f * (std::sqrt(3.0f) - 1.0f);
        const float G2 = (3.0f - std::sqrt(3.0f)) / 6.0f;
        
        // Skew the input space
        float s = (x + y) * F2;
        int i = fastFloor(x + s);
        int j = fastFloor(y + s);
        
        float t = (i + j) * G2;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        
        // Determine which simplex we're in
        int i1, j1;
        if (x0 > y0) {
            i1 = 1; j1 = 0;
        } else {
            i1 = 0; j1 = 1;
        }
        
        // Calculate other corners
        float x1 = x0 - i1 + G2;
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2;
        float y2 = y0 - 1.0f + 2.0f * G2;
        
        // Calculate contributions from each corner
        float n0, n1, n2;
        float t0 = 0.5f - x0*x0 - y0*y0;
        if (t0 < 0) n0 = 0.0f;
        else {
            t0 *= t0;
            n0 = t0 * t0 * grad(perm[i + perm[j]], x0, y0);
        }
        
        float t1 = 0.5f - x1*x1 - y1*y1;
        if (t1 < 0) n1 = 0.0f;
        else {
            t1 *= t1;
            n1 = t1 * t1 * grad(perm[i + i1 + perm[j + j1]], x1, y1);
        }
        
        float t2 = 0.5f - x2*x2 - y2*y2;
        if (t2 < 0) n2 = 0.0f;
        else {
            t2 *= t2;
            n2 = t2 * t2 * grad(perm[i + 1 + perm[j + 1]], x2, y2);
        }
        
        return 70.0f * (n0 + n1 + n2);
    }
    
    // Fast floor function
    int fastFloor(float x) {
        int xi = static_cast<int>(x);
        return x < xi ? xi - 1 : xi;
    }
    
    // Gradient function for Simplex noise
    float grad(int hash, float x, float y) {
        int h = hash & 7;
        float u = h < 4 ? x : y;
        float v = h < 4 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
    }
    
    // Raw noise function (simple hash-based)
    float rawNoise(float x, float y) {
        // Simple hash function for deterministic noise
        int xi = static_cast<int>(std::floor(x));
        int yi = static_cast<int>(std::floor(y));
        
        // Use the RNG to generate consistent noise based on grid position
        rng.seed(xi * 1619 + yi * 31337 + currentSeed);
        return dist(rng);
    }
    
    // Improved noise function (Perlin-like) using selected gradient type
    float improvedNoise(float x, float y) {
        // Integer part
        int xi = static_cast<int>(std::floor(x));
        int yi = static_cast<int>(std::floor(y));
        
        // Fractional part
        float xf = x - xi;
        float yf = y - yi;
        
        // Smooth interpolation
        float u = fade(xf);
        float v = fade(yf);
        
        // Gradient noise from corners using selected gradient calculation
        float n00 = gradNoise(xi, yi, xf, yf);
        float n01 = gradNoise(xi, yi + 1, xf, yf - 1);
        float n10 = gradNoise(xi + 1, yi, xf - 1, yf);
        float n11 = gradNoise(xi + 1, yi + 1, xf - 1, yf - 1);
        
        // Bilinear interpolation
        float x1 = lerp(n00, n10, u);
        float x2 = lerp(n01, n11, u);
        return lerp(x1, x2, v);
    }
    
    // Gradient noise function using selected gradient type
    float gradNoise(int xi, int yi, float xf, float yf) {
        switch (gradType) {
            case HASH_BASED:
                return hashGradNoise(xi, yi, xf, yf);
            case SIN_BASED:
                return sinGradNoise(xi, yi, xf, yf);
            case DOT_BASED:
                return dotGradNoise(xi, yi, xf, yf);
            case PRECOMPUTED:
            default:
                return precomputedGradNoise(xi, yi, xf, yf);
        }
    }
    
    // Fast gradient noise function using precomputed gradient directions
    float precomputedGradNoise(int xi, int yi, float xf, float yf) {
        // Generate deterministic hash from integer coordinates
        int hash = (xi * 1619 + yi * 31337 + currentSeed);
        
        // Use hash to select from 8 precomputed gradient directions
        int gradIndex = hash & 7; // 8 directions (0-7)
        
        // Dot product between distance vector and gradient
        return xf * grads[gradIndex].x + yf * grads[gradIndex].y;
    }
    
    // Hash-based gradient noise
    float hashGradNoise(int xi, int yi, float xf, float yf) {
        // Generate hash from coordinates
        uint32_t hash = (xi * 1619 + yi * 31337 + currentSeed);
        
        // Use hash to generate gradient angle
        hash = (hash << 13) ^ hash;
        hash = (hash * (hash * hash * 15731 + 789221) + 1376312589);
        float angle = (hash & 0xFFFF) / 65535.0f * 2.0f * M_PI;
        
        // Gradient vector
        float gx = std::cos(angle);
        float gy = std::sin(angle);
        
        // Dot product
        return xf * gx + yf * gy;
    }
    
    // Sine-based gradient noise
    float sinGradNoise(int xi, int yi, float xf, float yf) {
        // Use sine of coordinates to generate gradient
        float angle = std::sin(xi * 12.9898f + yi * 78.233f + currentSeed) * 43758.5453f;
        angle = angle - std::floor(angle); // Fractional part
        angle *= 2.0f * M_PI;
        
        float gx = std::cos(angle);
        float gy = std::sin(angle);
        
        return xf * gx + yf * gy;
    }
    
    // Dot product based gradient noise
    float dotGradNoise(int xi, int yi, float xf, float yf) {
        // Simple dot product with random vector based on coordinates
        float random = std::sin(xi * 127.1f + yi * 311.7f) * 43758.5453123f;
        random = random - std::floor(random);
        
        Vec2 grad(std::cos(random * 2.0f * M_PI), std::sin(random * 2.0f * M_PI));
        Vec2 dist(xf, yf);
        
        return grad.dot(dist);
    }
    
    // Fade function for smooth interpolation
    float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    // Linear interpolation
    float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }
    
    float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
        TIME_FUNCTION;
        if (x < lowerlimit) return lowerlimit;
        if (x > upperlimit) return upperlimit;
        return x;
    }
};

#endif