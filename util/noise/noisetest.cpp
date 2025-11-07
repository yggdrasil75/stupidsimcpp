// noisetest.cpp
#include "pnoise.hpp"
#include "../bmpwriter.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// Convert noise value to grayscale color
Vec3 noiseToColor(double noiseValue) {
    float value = static_cast<float>(noiseValue);
    return Vec3(value, value, value);
}

// Convert noise value to color using a blue-to-red colormap
Vec3 noiseToHeatmap(double noiseValue) {
    // Blue (0.0) -> Cyan -> Green -> Yellow -> Red (1.0)
    float value = static_cast<float>(noiseValue);
    
    if (value < 0.25f) {
        // Blue to Cyan
        float t = value / 0.25f;
        return Vec3(0.0f, t, 1.0f);
    } else if (value < 0.5f) {
        // Cyan to Green
        float t = (value - 0.25f) / 0.25f;
        return Vec3(0.0f, 1.0f, 1.0f - t);
    } else if (value < 0.75f) {
        // Green to Yellow
        float t = (value - 0.5f) / 0.25f;
        return Vec3(t, 1.0f, 0.0f);
    } else {
        // Yellow to Red
        float t = (value - 0.75f) / 0.25f;
        return Vec3(1.0f, 1.0f - t, 0.0f);
    }
}

// Convert noise value to terrain-like colors
Vec3 noiseToTerrain(double noiseValue) {
    float value = static_cast<float>(noiseValue);
    
    if (value < 0.3f) {
        // Deep water to shallow water
        return Vec3(0.0f, 0.0f, 0.3f + value * 0.4f);
    } else if (value < 0.4f) {
        // Sand
        return Vec3(0.76f, 0.70f, 0.50f);
    } else if (value < 0.6f) {
        // Grass
        float t = (value - 0.4f) / 0.2f;
        return Vec3(0.0f, 0.4f + t * 0.3f, 0.0f);
    } else if (value < 0.8f) {
        // Forest
        return Vec3(0.0f, 0.3f, 0.0f);
    } else {
        // Mountain to snow
        float t = (value - 0.8f) / 0.2f;
        return Vec3(0.8f + t * 0.2f, 0.8f + t * 0.2f, 0.8f + t * 0.2f);
    }
}

// Generate basic 2D noise map
void generateBasicNoise(const std::string& filename, int width, int height, 
                       double scale = 0.01, unsigned int seed = 42) {
    std::cout << "Generating basic noise: " << filename << std::endl;
    
    PerlinNoise pn(seed);
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.noise(x * scale, y * scale);
            pixels[y][x] = noiseToColor(noise);
        }
    }
    
    BMPWriter::saveBMP(filename, pixels);
}

// Generate fractal Brownian motion noise
void generateFBMNoise(const std::string& filename, int width, int height,
                     size_t octaves, double scale = 0.01, unsigned int seed = 42) {
    std::cout << "Generating FBM noise (" << octaves << " octaves): " << filename << std::endl;
    
    PerlinNoise pn(seed);
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.fractal(octaves, x * scale, y * scale);
            pixels[y][x] = noiseToColor(noise);
        }
    }
    
    BMPWriter::saveBMP(filename, pixels);
}

// Generate turbulence noise
void generateTurbulenceNoise(const std::string& filename, int width, int height,
                           size_t octaves, double scale = 0.01, unsigned int seed = 42) {
    std::cout << "Generating turbulence noise (" << octaves << " octaves): " << filename << std::endl;
    
    PerlinNoise pn(seed);
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.turbulence(octaves, x * scale, y * scale);
            pixels[y][x] = noiseToColor(noise);
        }
    }
    
    BMPWriter::saveBMP(filename, pixels);
}

// Generate ridged multi-fractal noise
void generateRidgedNoise(const std::string& filename, int width, int height,
                        size_t octaves, double scale = 0.01, unsigned int seed = 42,
                        double lacunarity = 2.0, double gain = 0.5, double offset = 1.0) {
    std::cout << "Generating ridged multi-fractal noise (" << octaves << " octaves): " << filename << std::endl;
    
    PerlinNoise pn(seed);
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.ridgedMultiFractal(octaves, x * scale, y * scale, 
                                               0.0, lacunarity, gain, offset);
            pixels[y][x] = noiseToColor(noise);
        }
    }
    
    BMPWriter::saveBMP(filename, pixels);
}

// Generate noise with different color mappings
void generateColoredNoise(const std::string& filename, int width, int height,
                         double scale = 0.01, unsigned int seed = 42,
                         const std::string& colorMap = "heatmap") {
    std::cout << "Generating colored noise (" << colorMap << "): " << filename << std::endl;
    
    PerlinNoise pn(seed);
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.noise(x * scale, y * scale);
            
            if (colorMap == "heatmap") {
                pixels[y][x] = noiseToHeatmap(noise);
            } else if (colorMap == "terrain") {
                pixels[y][x] = noiseToTerrain(noise);
            } else {
                pixels[y][x] = noiseToColor(noise);
            }
        }
    }
    
    BMPWriter::saveBMP(filename, pixels);
}

// Generate multi-octave comparison
void generateOctaveComparison(const std::string& baseFilename, int width, int height,
                            double scale = 0.01, unsigned int seed = 42) {
    for (size_t octaves = 1; octaves <= 6; ++octaves) {
        std::string filename = baseFilename + "_octaves_" + std::to_string(octaves) + ".bmp";
        generateFBMNoise(filename, width, height, octaves, scale, seed);
    }
}

// Generate scale comparison
void generateScaleComparison(const std::string& baseFilename, int width, int height,
                           unsigned int seed = 42) {
    std::vector<double> scales = {0.002, 0.005, 0.01, 0.02, 0.05, 0.1};
    
    for (double scale : scales) {
        std::string filename = baseFilename + "_scale_" + std::to_string(scale) + ".bmp";
        generateBasicNoise(filename, width, height, scale, seed);
    }
}

// Generate seed comparison
void generateSeedComparison(const std::string& baseFilename, int width, int height,
                          double scale = 0.01) {
    std::vector<unsigned int> seeds = {42, 123, 456, 789, 1000};
    
    for (unsigned int seed : seeds) {
        std::string filename = baseFilename + "_seed_" + std::to_string(seed) + ".bmp";
        generateBasicNoise(filename, width, height, scale, seed);
    }
}

// Generate combined effects (FBM with different color maps)
void generateCombinedEffects(const std::string& baseFilename, int width, int height,
                           double scale = 0.01, unsigned int seed = 42) {
    PerlinNoise pn(seed);
    
    // FBM with grayscale
    std::vector<std::vector<Vec3>> pixels1(height, std::vector<Vec3>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.fractal(4, x * scale, y * scale);
            pixels1[y][x] = noiseToColor(noise);
        }
    }
    BMPWriter::saveBMP(baseFilename + "_fbm_grayscale.bmp", pixels1);
    
    // FBM with heatmap
    std::vector<std::vector<Vec3>> pixels2(height, std::vector<Vec3>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.fractal(4, x * scale, y * scale);
            pixels2[y][x] = noiseToHeatmap(noise);
        }
    }
    BMPWriter::saveBMP(baseFilename + "_fbm_heatmap.bmp", pixels2);
    
    // FBM with terrain
    std::vector<std::vector<Vec3>> pixels3(height, std::vector<Vec3>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double noise = pn.fractal(4, x * scale, y * scale);
            pixels3[y][x] = noiseToTerrain(noise);
        }
    }
    BMPWriter::saveBMP(baseFilename + "_fbm_terrain.bmp", pixels3);
}

// Generate 3D slice noise (showing different Z slices)
void generate3DSlices(const std::string& baseFilename, int width, int height,
                     double scale = 0.01, unsigned int seed = 42) {
    PerlinNoise pn(seed);
    
    std::vector<double> zSlices = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    
    for (size_t i = 0; i < zSlices.size(); ++i) {
        std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width));
        double z = zSlices[i] * 10.0; // Scale Z for meaningful variation
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double noise = pn.noise(x * scale, y * scale, z);
                pixels[y][x] = noiseToColor(noise);
            }
        }
        
        std::string filename = baseFilename + "_zslice_" + std::to_string(i) + ".bmp";
        BMPWriter::saveBMP(filename, pixels);
        std::cout << "Generated 3D slice " << i << ": " << filename << std::endl;
    }
}

int main() {
    const int WIDTH = 512;
    const int HEIGHT = 512;
    
    std::cout << "Generating Perlin noise variations..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // 1. Basic noise variations
    std::cout << "\n1. Basic Noise Variations:" << std::endl;
    generateBasicNoise("output/basic_noise.bmp", WIDTH, HEIGHT, 0.01, 42);
    
    // 2. Fractal Brownian Motion with different octaves
    std::cout << "\n2. FBM Noise (Multiple Octaves):" << std::endl;
    generateOctaveComparison("output/fbm", WIDTH, HEIGHT, 0.01, 42);
    
    // 3. Turbulence noise
    std::cout << "\n3. Turbulence Noise:" << std::endl;
    generateTurbulenceNoise("output/turbulence_4oct.bmp", WIDTH, HEIGHT, 4, 0.01, 42);
    generateTurbulenceNoise("output/turbulence_6oct.bmp", WIDTH, HEIGHT, 6, 0.01, 42);
    
    // 4. Ridged multi-fractal noise
    std::cout << "\n4. Ridged Multi-Fractal Noise:" << std::endl;
    generateRidgedNoise("output/ridged_4oct.bmp", WIDTH, HEIGHT, 4, 0.01, 42, 2.0, 0.5, 1.0);
    generateRidgedNoise("output/ridged_6oct.bmp", WIDTH, HEIGHT, 6, 0.01, 42, 2.0, 0.5, 1.0);
    
    // 5. Different color mappings
    std::cout << "\n5. Color Mappings:" << std::endl;
    generateColoredNoise("output/heatmap_noise.bmp", WIDTH, HEIGHT, 0.01, 42, "heatmap");
    generateColoredNoise("output/terrain_noise.bmp", WIDTH, HEIGHT, 0.01, 42, "terrain");
    
    // 6. Scale variations
    std::cout << "\n6. Scale Variations:" << std::endl;
    generateScaleComparison("output/scale_test", WIDTH, HEIGHT, 42);
    
    // 7. Seed variations
    std::cout << "\n7. Seed Variations:" << std::endl;
    generateSeedComparison("output/seed_test", WIDTH, HEIGHT, 0.01);
    
    // 8. Combined effects
    std::cout << "\n8. Combined Effects:" << std::endl;
    generateCombinedEffects("output/combined", WIDTH, HEIGHT, 0.01, 42);
    
    // 9. 3D slices
    std::cout << "\n9. 3D Slices:" << std::endl;
    generate3DSlices("output/3d_slice", WIDTH, HEIGHT, 0.01, 42);
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "All noise maps generated successfully!" << std::endl;
    std::cout << "Check the 'output' directory for BMP files." << std::endl;
    
    return 0;
}