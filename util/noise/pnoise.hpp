
#ifndef PERLIN_NOISE_HPP
#define PERLIN_NOISE_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

class PerlinNoise {
private:
    std::vector<int> permutation;

    // Fade function as defined by Ken Perlin
    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    // Linear interpolation
    static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    // Gradient function
    static double grad(int hash, double x, double y, double z) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    // Constructor with optional seed
    PerlinNoise(unsigned int seed = 0) {
        permutation.resize(256);
        // Initialize with values 0-255
        for (int i = 0; i < 256; ++i) {
            permutation[i] = i;
        }
        
        // Shuffle using the seed
        std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine(seed));
        
        // Duplicate the permutation vector
        permutation.insert(permutation.end(), permutation.begin(), permutation.end());
    }

    // 1D Perlin noise
    double noise(double x) const {
        return noise(x, 0.0, 0.0);
    }

    // 2D Perlin noise
    double noise(double x, double y) const {
        return noise(x, y, 0.0);
    }

    // 3D Perlin noise (main implementation)
    double noise(double x, double y, double z) const {
        // Find unit cube that contains the point
        int X = (int)floor(x) & 255;
        int Y = (int)floor(y) & 255;
        int Z = (int)floor(z) & 255;

        // Find relative x, y, z of point in cube
        x -= floor(x);
        y -= floor(y);
        z -= floor(z);

        // Compute fade curves for x, y, z
        double u = fade(x);
        double v = fade(y);
        double w = fade(z);

        // Hash coordinates of the 8 cube corners
        int A = permutation[X] + Y;
        int AA = permutation[A] + Z;
        int AB = permutation[A + 1] + Z;
        int B = permutation[X + 1] + Y;
        int BA = permutation[B] + Z;
        int BB = permutation[B + 1] + Z;

        // Add blended results from 8 corners of cube
        double res = lerp(w, lerp(v, lerp(u, grad(permutation[AA], x, y, z),
                                            grad(permutation[BA], x - 1, y, z)),
                                    lerp(u, grad(permutation[AB], x, y - 1, z),
                                            grad(permutation[BB], x - 1, y - 1, z))),
                            lerp(v, lerp(u, grad(permutation[AA + 1], x, y, z - 1),
                                            grad(permutation[BA + 1], x - 1, y, z - 1)),
                                    lerp(u, grad(permutation[AB + 1], x, y - 1, z - 1),
                                            grad(permutation[BB + 1], x - 1, y - 1, z - 1))));
        return (res + 1.0) / 2.0; // Normalize to [0,1]
    }

    // Fractal Brownian Motion (fBm) - multiple octaves of noise
    double fractal(size_t octaves, double x, double y = 0.0, double z = 0.0) const {
        double value = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double maxValue = 0.0;

        for (size_t i = 0; i < octaves; ++i) {
            value += amplitude * noise(x * frequency, y * frequency, z * frequency);
            maxValue += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }

        return value / maxValue;
    }

    // Turbulence - absolute value of noise for more dramatic effects
    double turbulence(size_t octaves, double x, double y = 0.0, double z = 0.0) const {
        double value = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double maxValue = 0.0;

        for (size_t i = 0; i < octaves; ++i) {
            value += amplitude * std::abs(noise(x * frequency, y * frequency, z * frequency));
            maxValue += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }

        return value / maxValue;
    }

    // Ridged multi-fractal - creates ridge-like patterns
    double ridgedMultiFractal(size_t octaves, double x, double y = 0.0, double z = 0.0, 
                             double lacunarity = 2.0, double gain = 0.5, double offset = 1.0) const {
        double value = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double prev = 1.0;
        double weight;

        for (size_t i = 0; i < octaves; ++i) {
            double signal = offset - std::abs(noise(x * frequency, y * frequency, z * frequency));
            signal *= signal;
            signal *= prev;
            
            weight = std::clamp(signal * gain, 0.0, 1.0);
            value += signal * amplitude;
            prev = weight;
            amplitude *= weight;
            frequency *= lacunarity;
        }

        return value;
    }
};

// Utility functions for common noise operations
namespace PerlinUtils {
    // Create a 1D noise array
    static std::vector<double> generate1DNoise(int width, double scale = 1.0, unsigned int seed = 0) {
        PerlinNoise pn(seed);
        std::vector<double> result(width);
        
        for (int x = 0; x < width; ++x) {
            result[x] = pn.noise(x * scale);
        }
        
        return result;
    }

    // Create a 2D noise array
    static std::vector<std::vector<double>> generate2DNoise(int width, int height, 
                                                           double scale = 1.0, unsigned int seed = 0) {
        PerlinNoise pn(seed);
        std::vector<std::vector<double>> result(height, std::vector<double>(width));
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                result[y][x] = pn.noise(x * scale, y * scale);
            }
        }
        
        return result;
    }

    // Create a 3D noise array
    static std::vector<std::vector<std::vector<double>>> generate3DNoise(int width, int height, int depth,
                                                                        double scale = 1.0, unsigned int seed = 0) {
        PerlinNoise pn(seed);
        std::vector<std::vector<std::vector<double>>> result(
            depth, std::vector<std::vector<double>>(
                height, std::vector<double>(width)));
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    result[z][y][x] = pn.noise(x * scale, y * scale, z * scale);
                }
            }
        }
        
        return result;
    }
}

#endif