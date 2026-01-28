#ifndef PNOISE2_HPP
#define PNOISE2_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include "../../eigen/Eigen/Core"
#include "../timing_decorator.hpp"

class PNoise2 {
private:
    std::vector<int> permutation;
    std::default_random_engine rng;
    
    using Vector2f = Eigen::Vector2f;
    using Vector3f = Eigen::Vector3f;
    using Vector4f = Eigen::Vector4f;
    
    /// @brief Linear interpolation between two values
    /// @param t Interpolation factor [0,1]
    /// @param a1 First value
    /// @param a2 Second value
    /// @return Interpolated value between a1 and a2
    /// @note Changing interpolation method affects noise smoothness
    float lerp(float t, float a1, float a2) {
        return a1 + t * (a2 - a1);
    }

    /// @brief Fade function for smooth interpolation
    /// @param t Input parameter
    /// @return Smoothed t value using 6t^5 - 15t^4 + 10t^3
    /// @note Critical for gradient continuity; changes affect noise smoothness
    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    /// @brief Get constant gradient vector for 2D Perlin noise
    /// @param v Hash value (0-3)
    /// @return One of four 2D gradient vectors
    /// @note Changing vectors affects noise pattern orientation
    Vector2f GetConstantVector(int v) {
        int h = v & 3;
        if (h == 0) return Vector2f(1,1);
        else if (h == 1) return Vector2f(-1,1);
        else if (h == 2) return Vector2f(-1,-1);
        else return Vector2f(1,-1);
    }

    /// @brief Get constant gradient vector for 3D Perlin noise
    /// @param v Hash value (0-11)
    /// @return One of twelve 3D gradient vectors
    /// @note Vector selection affects 3D noise patterns
    Vector3f GetConstantVector3(int v) {
        int h = v & 11;
        switch(h) {
            case 0: return Vector3f( 1, 1, 0);
            case 1: return Vector3f(-1, 1, 0);
            case 2: return Vector3f( 1,-1, 0);
            case 3: return Vector3f(-1,-1, 0);
            case 4: return Vector3f( 1, 0, 1);
            case 5: return Vector3f(-1, 0, 1);
            case 6: return Vector3f( 1, 0,-1);
            case 7: return Vector3f(-1, 0,-1);
            case 8: return Vector3f( 0, 1, 1);
            case 9: return Vector3f( 0,-1, 1);
            case 10: return Vector3f( 0, 1,-1);
            case 11: return Vector3f( 0,-1,-1);
            default: return Vector3f(0,0,0);
        }
    }

    /// @brief Gradient function for 2D/3D Perlin noise
    /// @param hash Hash value for gradient selection
    /// @param x X coordinate
    /// @param y Y coordinate
    /// @param z Z coordinate (default 0 for 2D)
    /// @return Dot product of gradient vector and distance vector
    /// @note Core of Perlin noise; changes affect basic noise character
    static double grad(int hash, double x, double y, double z = 0.0) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    /// @brief Initialize permutation table with shuffled values
    /// @note Called on construction; changing seed or shuffle affects all noise patterns
    void initializePermutation() {
        permutation.clear();
        std::vector<int> permutationt;
        permutationt.reserve(256);
        for (int i = 0; i < 256; i++){
            permutationt.push_back(i);
        }
        std::ranges::shuffle(permutationt, rng);
        permutation.insert(permutation.end(), permutationt.begin(), permutationt.end());
        permutation.insert(permutation.end(), permutationt.begin(), permutationt.end());
    }

    /// @brief Normalize noise value from [-1,1] to [0,1]
    /// @param point Input coordinate
    /// @return Normalized noise value in [0,1] range
    /// @note Useful for texture generation; changes affect output range
    float normalizedNoise(const Vector2f& point) {
        return (permute(point) + 1.0f) * 0.5f;
    }

    /// @brief Normalize 3D noise value from [-1,1] to [0,1]
    /// @param point Input coordinate
    /// @return Normalized noise value in [0,1] range
    float normalizedNoise(const Vector3f& point) {
        return (permute(point) + 1.0f) * 0.5f;
    }

    /// @brief Map value from one range to another
    /// @param value Input value
    /// @param inMin Original range minimum
    /// @param inMax Original range maximum
    /// @param outMin Target range minimum
    /// @param outMax Target range maximum
    /// @return Value mapped to new range
    /// @note Useful for post-processing; changes affect output scaling
    float mapRange(float value, float inMin, float inMax, float outMin, float outMax) {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }

    /// @brief Blend two noise values
    /// @param noise1 First noise value
    /// @param noise2 Second noise value
    /// @param blendFactor Blending factor [0,1]
    /// @return Blended noise value
    /// @note Changes affect multi-layer noise combinations
    float blendNoises(float noise1, float noise2, float blendFactor) {
        return lerp(blendFactor, noise1, noise2);
    }

    /// @brief Add two noise values with clamping
    /// @param noise1 First noise value
    /// @param noise2 Second noise value
    /// @return Sum clamped to [-1,1]
    /// @note Clamping prevents overflow; changes affect combined noise magnitude
    float addNoises(float noise1, float noise2) {
        return std::clamp(noise1 + noise2, -1.0f, 1.0f);
    }

    /// @brief Multiply two noise values
    /// @param noise1 First noise value
    /// @param noise2 Second noise value
    /// @return Product of noise values
    /// @note Creates modulation effects; changes affect combined noise character
    float multiplyNoises(float noise1, float noise2) {
        return noise1 * noise2;
    }

    /// @brief Hash function for 2D coordinates
    /// @param x X coordinate integer
    /// @param y Y coordinate integer
    /// @return Hash value in [-1,1] range
    /// @note Core of value noise; changes affect random distribution
    float hash(int x, int y) {
        return (permutation[(x + permutation[y & 255]) & 255] / 255.0f) * 2.0f - 1.0f;
    }
    
    /// @brief Hash function for 3D coordinates
    /// @param x X coordinate integer
    /// @param y Y coordinate integer
    /// @param z Z coordinate integer
    /// @return Hash value in [-1,1] range
    /// @note 3D version of hash function
    float hash(int x, int y, int z) {
        return (permutation[(z + permutation[(y + permutation[x & 255]) & 255]) & 255] / 255.0f) * 2.0f - 1.0f;
    }

public:
    /// @brief Default constructor with random seed
    /// @note Uses random_device for seed; different runs produce different noise
    PNoise2() : rng(std::random_device{}()) {
        initializePermutation();
    }
    
    /// @brief Constructor with specified seed
    /// @param seed Random seed value
    /// @note Same seed produces identical noise patterns across runs
    PNoise2(unsigned int seed) : rng(seed) {
        initializePermutation();
    }
    
    /// @brief Generate 2D Perlin noise at given point
    /// @param point 2D coordinate
    /// @return Noise value in [-1,1] range
    /// @note Core 2D noise function; changes affect all 2D noise outputs
    float permute(const Vector2f& point) {
        TIME_FUNCTION;
        float x = point.x();
        float y = point.y();
        int X = (int)floor(x);
        int xmod = X & 255;
        int Y = (int)floor(y);
        int ymod = Y & 255;
        float xf = x - X;
        float yf = y - Y;

        Vector2f BL(xf-0, yf-0);
        Vector2f BR(xf-1, yf-0);
        Vector2f TL(xf-0, yf-1);
        Vector2f TR(xf-1, yf-1);

        int vBL = permutation[permutation[xmod+0]+ymod+0];
        int vBR = permutation[permutation[xmod+1]+ymod+0];
        int vTL = permutation[permutation[xmod+0]+ymod+1];
        int vTR = permutation[permutation[xmod+1]+ymod+1];

        // float dBL = BL.dot(GetConstantVector(vBL));
        // float dBR = BR.dot(GetConstantVector(vBR));
        // float dTL = TL.dot(GetConstantVector(vTL));
        // float dTR = TR.dot(GetConstantVector(vTR));
        
        float u = fade(xf);
        float v = fade(yf);

        float x1 = lerp(u, grad(vBL, xf, yf),     grad(vBR, xf - 1, yf));
        float x2 = lerp(u, grad(vTL, xf, yf - 1), grad(vTR, xf - 1, yf - 1));
        float retval = lerp(v, x1, x2);
        
        return retval;
    }

    /// @brief Generate 3D Perlin noise at given point
    /// @param point 3D coordinate
    /// @return Noise value in [-1,1] range
    float permute(const Vector3f& point) {
        TIME_FUNCTION;
        float x = point.x();
        float y = point.y();
        float z = point.z();
        
        int X = (int)floor(x) & 255;
        int Y = (int)floor(y) & 255;
        int Z = (int)floor(z) & 255;
        float xf = x - X;
        float yf = y - Y;
        float zf = z - Z;

        // Distance vectors from corners
        Vector3f FBL(xf-0, yf-0, zf-0);
        Vector3f FBR(xf-1, yf-0, zf-0);
        Vector3f FTL(xf-0, yf-1, zf-0);
        Vector3f FTR(xf-1, yf-1, zf-0);

        Vector3f RBL(xf-0, yf-0, zf-1);
        Vector3f RBR(xf-1, yf-0, zf-1);
        Vector3f RTL(xf-0, yf-1, zf-1);
        Vector3f RTR(xf-1, yf-1, zf-1);

        int vFBL = permutation[permutation[permutation[Z+0]+X+0]+Y+0];
        int vFBR = permutation[permutation[permutation[Z+0]+X+1]+Y+0];
        int vFTL = permutation[permutation[permutation[Z+0]+X+0]+Y+1];
        int vFTR = permutation[permutation[permutation[Z+0]+X+1]+Y+1];

        int vRBL = permutation[permutation[permutation[Z+1]+X+0]+Y+0];
        int vRBR = permutation[permutation[permutation[Z+1]+X+1]+Y+0];
        int vRTL = permutation[permutation[permutation[Z+1]+X+0]+Y+1];
        int vRTR = permutation[permutation[permutation[Z+1]+X+1]+Y+1];

        // float dFBL = FBL.dot(GetConstantVector3(vFBL));
        // float dFBR = FBR.dot(GetConstantVector3(vFBR));
        // float dFTL = FTL.dot(GetConstantVector3(vFTL));
        // float dFTR = FTR.dot(GetConstantVector3(vFTR));

        // float dRBL = RBL.dot(GetConstantVector3(vRBL));
        // float dRBR = RBR.dot(GetConstantVector3(vRBR));
        // float dRTL = RTL.dot(GetConstantVector3(vRTL));
        // float dRTR = RTR.dot(GetConstantVector3(vRTR));
        
        float u = fade(xf);
        float v = fade(yf);
        float w = fade(zf);

        float x1 = lerp(u, grad(vFBL, xf, yf + 0, zf + 0), grad(vFBR, xf - 1, yf + 0, zf + 0));
        float x2 = lerp(u, grad(vFTL, xf, yf - 1, zf + 0), grad(vFTR, xf - 1, yf - 1, zf + 0));
        float y1 = lerp(v, x1, x2);

        float x3 = lerp(u, grad(vRBL, xf, yf - 1, zf + 1), grad(vRBR, xf - 1, yf - 1, zf + 1));
        float x4 = lerp(u, grad(vRTL, xf, yf - 1, zf + 1), grad(vRTR, xf - 1, yf - 1, zf + 1));
        float y2 = lerp(v, x3, x4);

        float retval = lerp(w, y1, y2);

        return retval;
    }

    /// @brief Generate 2D value noise (simpler alternative to Perlin)
    /// @param point 2D coordinate
    /// @return Noise value in [-1,1] range
    /// @note Different character than Perlin; changes affect value-based textures
    float valueNoise(const Vector2f& point) {
        int xi = (int)std::floor(point.x());
        int yi = (int)std::floor(point.y());
        
        float tx = point.x() - xi;
        float ty = point.y() - yi;
        
        int rx0 = xi & 255;
        int rx1 = (xi + 1) & 255;
        int ry0 = yi & 255;
        int ry1 = (yi + 1) & 255;
        
        // Random values at corners
        float c00 = hash(rx0, ry0);
        float c10 = hash(rx1, ry0);
        float c01 = hash(rx0, ry1);
        float c11 = hash(rx1, ry1);
        
        // Interpolation
        float sx = fade(tx);
        float sy = fade(ty);
        
        float nx0 = lerp(c00, c10, sx);
        float nx1 = lerp(c01, c11, sx);
        
        return lerp(nx0, nx1, sy);
    }

    /// @brief Generate 3D value noise
    /// @param point 3D coordinate
    /// @return Noise value in [-1,1] range
    float valueNoise(const Vector3f& point) {
        int xi = (int)std::floor(point.x());
        int yi = (int)std::floor(point.y());
        int zi = (int)std::floor(point.z());
        
        float tx = point.x() - xi;
        float ty = point.y() - yi;
        float tz = point.z() - zi;
        
        int rx0 = xi & 255;
        int rx1 = (xi + 1) & 255;
        int ry0 = yi & 255;
        int ry1 = (yi + 1) & 255;
        int rz0 = zi & 255;
        int rz1 = (zi + 1) & 255;
        
        // Random values at corners
        float c000 = hash(rx0, ry0, rz0);
        float c100 = hash(rx1, ry0, rz0);
        float c010 = hash(rx0, ry1, rz0);
        float c110 = hash(rx1, ry1, rz0);
        float c001 = hash(rx0, ry0, rz1);
        float c101 = hash(rx1, ry0, rz1);
        float c011 = hash(rx0, ry1, rz1);
        float c111 = hash(rx1, ry1, rz1);
        
        // Interpolation
        float sx = fade(tx);
        float sy = fade(ty);
        float sz = fade(tz);
        
        float nx00 = lerp(c000, c100, sx);
        float nx10 = lerp(c010, c110, sx);
        float nx01 = lerp(c001, c101, sx);
        float nx11 = lerp(c011, c111, sx);
        
        float ny0 = lerp(nx00, nx10, sy);
        float ny1 = lerp(nx01, nx11, sy);
        
        return lerp(ny0, ny1, sz);
    }

    /// @brief Generate RGBA color from 3D noise with offset channels
    /// @param point 3D coordinate
    /// @return Vector4f containing RGBA noise values
    /// @note Each channel uses different offset; changes affect color patterns
    Vector4f permuteColor(const Vector3f& point) {
        TIME_FUNCTION;
        float noiseR = permute(point);
        float noiseG = permute(Vector3f(point + Vector3f(100.0f, 100.0f, 100.0f)));
        float noiseB = permute(Vector3f(point + Vector3f(200.0f, 200.0f, 200.0f)));
        float noiseA = permute(Vector3f(point + Vector3f(300.0f, 300.0f, 300.0f)));
        
        float rNormalized = (noiseR + 1.0f) * 0.5f;
        float gNormalized = (noiseG + 1.0f) * 0.5f;
        float bNormalized = (noiseB + 1.0f) * 0.5f;
        float aNormalized = (noiseA + 1.0f) * 0.5f;
        
        rNormalized = std::clamp(rNormalized, 0.0f, 1.0f);
        gNormalized = std::clamp(gNormalized, 0.0f, 1.0f);
        bNormalized = std::clamp(bNormalized, 0.0f, 1.0f);
        aNormalized = std::clamp(aNormalized, 0.0f, 1.0f);
        
        return Vector4f(rNormalized, gNormalized, bNormalized, aNormalized);
    }

    /// @brief Generate fractal (octave) noise for natural-looking patterns
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @param persistence Amplitude multiplier per octave
    /// @param lacunarity Frequency multiplier per octave
    /// @return Combined noise value
    /// @note Parameters control noise character: octaves=detail, persistence=roughness, lacunarity=frequency change
    float fractalNoise(const Vector2f& point, int octaves, float persistence, float lacunarity) {
        float total = 0.0f;
        float frequency = 1.f;
        float amplitude = 1.f;
        float maxV = 0.f;
        Vector2f scaledPoint = point * frequency;
        for (int i = 0; i < octaves; i++) {
            total += permute(scaledPoint) * amplitude;
            maxV += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return total / maxV;
    }

    /// @brief Generate 3D fractal (octave) noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @param persistence Amplitude multiplier per octave
    /// @param lacunarity Frequency multiplier per octave
    /// @return Combined noise value
    float fractalNoise(const Vector3f& point, int octaves, float persistence, float lacunarity) {
        float total = 0.0f;
        float frequency = 1.f;
        float amplitude = 1.f;
        float maxV = 0.f;
        Vector3f scaledPoint = point * frequency;
        for (int i = 0; i < octaves; i++) {
            total += permute(scaledPoint) * amplitude;
            maxV += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return total / maxV;
    }

    /// @brief Generate turbulence noise (absolute value of octaves)
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @return Turbulence noise value
    /// @note Creates swirling, turbulent patterns; changes affect visual complexity
    float turbulence(const Vector2f& point, int octaves) {
        float value = 0.0f;
        Vector2f tempPoint = point;
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(tempPoint)) / (1 << i);
            tempPoint *= 2.f;
        }
        return value;
    }

    /// @brief Generate 3D turbulence noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @return Turbulence noise value
    float turbulence(const Vector3f& point, int octaves) {
        float value = 0.0f;
        Vector3f tempPoint = point;
        
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(tempPoint)) / (1 << i);
            tempPoint *= 2.f;
        }
        return value;
    }

    /// @brief Generate ridged (ridged multifractal) noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @param offset Weighting offset for ridge formation
    /// @return Ridged noise value
    /// @note Creates sharp ridge-like patterns; offset controls ridge prominence
    float ridgedNoise(const Vector2f& point, int octaves, float offset = 1.0f) {
        float result = 0.f;
        float weight = 1.f;
        Vector2f p = point;
        
        for (int i = 0; i < octaves; i++) {
            float signal = 1.f - std::abs(permute(p));
            signal *= signal;
            signal *= weight;
            weight = signal * offset;
            result += signal;
            p *= 2.f;
        }
        
        return result;
    }

    /// @brief Generate 3D ridged noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @param offset Weighting offset for ridge formation
    /// @return Ridged noise value
    float ridgedNoise(const Vector3f& point, int octaves, float offset = 1.0f) {
        float result = 0.f;
        float weight = 1.f;
        Vector3f p = point;
        
        for (int i = 0; i < octaves; i++) {
            float signal = 1.f - std::abs(permute(p));
            signal *= signal;
            signal *= weight;
            weight = signal * offset;
            result += signal;
            p *= 2.f;
        }
        
        return result;
    }

    /// @brief Generate billow (cloud-like) noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @return Billow noise value
    /// @note Creates soft, billowy patterns like clouds
    float billowNoise(const Vector2f& point, int octaves) {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        Vector2f scaledPoint = point * frequency;
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(scaledPoint)) * amplitude;
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }
        
        return value;
    }

    /// @brief Generate 3D billow noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @return Billow noise value
    float billowNoise(const Vector3f& point, int octaves) {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        Vector3f scaledPoint = point * frequency;
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(scaledPoint)) * amplitude;
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }
        
        return value;
    }
};

#endif