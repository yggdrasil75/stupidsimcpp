#ifndef NOISE2_HPP
#define NOISE2_HPP

#include "grid2.hpp"
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>

struct Grad { float x, y; };
std::array<Grad, 256> gradients;

class Noise2 {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
public:
    Noise2(uint32_t seed = 0) : rng(seed), dist(0.0f, 1.0f) {}
    
    // Set random seed
    void setSeed(uint32_t seed) {
        rng.seed(seed);
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
    
    // Generate a grayscale noise grid
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
                
                float noiseValue = perlinNoise(nx, ny, octaves, persistence);
                
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
                
                // Generate separate noise for each channel
                float r = perlinNoise(nx * scale.x, ny * scale.x, 
                                     static_cast<int>(octaves.x), persistence.x);
                float g = perlinNoise(nx * scale.y, ny * scale.y, 
                                     static_cast<int>(octaves.y), persistence.y);
                float b = perlinNoise(nx * scale.z, ny * scale.z, 
                                     static_cast<int>(octaves.z), persistence.z);
                float a = perlinNoise(nx * scale.w, ny * scale.w, 
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
    Grid2 generateTerrainNoise(int width, int height,
                              float scale = 1.0f,
                              int octaves = 4,
                              float persistence = 0.5f,
                              uint32_t seed = 0,
                              const Vec2& offset = Vec2(0, 0)) {
        if (seed != 0) setSeed(seed);
        
        Grid2 grid(width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = (x + offset.x) / width * scale;
                float ny = (y + offset.y) / height * scale;
                
                // Use multiple octaves for more natural terrain
                float heightValue = perlinNoise(nx, ny, octaves, persistence);
                
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

private:
    // Raw noise function (simple hash-based)
    float rawNoise(float x, float y) {
        // Simple hash function for deterministic noise
        int xi = static_cast<int>(std::floor(x));
        int yi = static_cast<int>(std::floor(y));
        
        // Use the RNG to generate consistent noise based on grid position
        rng.seed(xi * 1619 + yi * 31337);
        return dist(rng);
    }
    
    // Improved noise function (Perlin-like)
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
        
        // Gradient noise from corners
        float n00 = gradNoise(xi, yi, xf, yf);
        float n01 = gradNoise(xi, yi + 1, xf, yf - 1);
        float n10 = gradNoise(xi + 1, yi, xf - 1, yf);
        float n11 = gradNoise(xi + 1, yi + 1, xf - 1, yf - 1);
        
        // Bilinear interpolation
        float x1 = lerp(n00, n10, u);
        float x2 = lerp(n01, n11, u);
        return lerp(x1, x2, v);
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

    // float grad(const int& hash, const float& b, const float& c, const float& d) {
    //     TIME_FUNCTION;
    //     int h = hash & 15;
    //     float u = (h < 8) ? c : b;
    //     float v = (h < 4) ? b : ((h == 12 || h == 14) ? c : d);
    //     return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
    // }
    float gradNoise(int xi, int yi, float xf, float yf) {
        // Generate deterministic "random" unit vector using hash
        int hash = (xi * 1619 + yi * 31337);
        
        // Use hash to generate angle in fixed steps (faster than trig)
        float angle = (hash & 255) * (2.0f * 3.14159265f / 256.0f);
        
        // Or even faster: use gradient table with 8 or 16 precomputed directions
        int gradIndex = hash & 7; // 8 directions
        static constexpr std::array<Grad, 8> grads = {
            {1,0}, {0.707f,0.707f}, {0,1}, {-0.707f,0.707f},
            {-1,0}, {-0.707f,-0.707f}, {0,-1}, {0.707f,-0.707f}
        };
        
        return xf * grads[gradIndex].x + yf * grads[gradIndex].y;
    }

    // Gradient noise function
    float slowGradNoise(int xi, int yi, float xf, float yf) {
        // Generate consistent random gradient from integer coordinates
        rng.seed(xi * 1619 + yi * 31337);
        float angle = dist(rng) * 2.0f * 3.14159265f;
        
        // Gradient vector
        float gx = std::cos(angle);
        float gy = std::sin(angle);
        
        // Dot product
        return xf * gx + yf * gy;
    }
};

#endif