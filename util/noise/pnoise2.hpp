#ifndef PNOISE2_HPP
#define PNOISE2_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <limits>
#include "../../eigen/Eigen/Core"
#include "../timing_decorator.hpp"
#include "../basicdefines.hpp"

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

    /// @brief Gradient function for 2D only Perlin noise
    /// @param hash Hash value for gradient selection
    /// @param x X coordinate
    /// @param y Y coordinate
    /// @return Dot product of gradient vector and distance vector
    /// @note Core of Perlin noise; changes affect basic noise character
    inline static float grad2(int hash, float x, float y) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : ((h == 12 || h == 14) ? x : 0.0f);
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

    /// @brief Pseudo-random vector for Worley noise (2D)
    Vector2f hashVector(const Vector2f& gridPoint) {
        int x = (int)gridPoint.x() & 255;
        int y = (int)gridPoint.y() & 255;
        // Generate pseudo-random float [0,1] for x and y offsets
        float hx = permutation[(x + permutation[y]) & 255] / 255.0f;
        float hy = permutation[(y + permutation[(x + 1) & 255]) & 255] / 255.0f;
        return Vector2f(hx, hy);
    }

    /// @brief Pseudo-random vector for Worley noise (3D)
    Vector3f hashVector(const Vector3f& gridPoint) {
        int x = (int)gridPoint.x() & 255;
        int y = (int)gridPoint.y() & 255;
        int z = (int)gridPoint.z() & 255;
        
        int h_xy = permutation[(x + permutation[y]) & 255];
        float hx = permutation[(h_xy + z) & 255] / 255.0f;
        float hy = permutation[(h_xy + permutation[(z + 1) & 255]) & 255] / 255.0f;
        float hz = permutation[(permutation[(x+1)&255] + permutation[(y+1)&255] + z) & 255] / 255.0f;
        
        return Vector3f(hx, hy, hz);
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
    
    /// @brief Generate 2D Perlin noise at given point
    /// @param point 2D coordinate
    /// @return Noise value in [-1,1] range
    /// @note Core 2D noise function; changes affect all 2D noise outputs
    float permute(const Vector2f& point) {
        // TIME_FUNCTION;
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
        // TIME_FUNCTION;
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
        
        float nx0 = lerp(sy, c00, c10);
        float nx1 = lerp(sx, c01, c11);
        
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
        
        float nx00 = lerp(sx, c000, c100);
        float nx10 = lerp(sx, c010, c110);
        float nx01 = lerp(sx, c001, c101);
        float nx11 = lerp(sx, c011, c111);
        
        float ny0 = lerp(sy, nx00, nx10);
        float ny1 = lerp(sy, nx01, nx11);
        
        return lerp(sz, ny0, ny1);
    }

    /// @brief Generate RGBA color from 3D noise with offset channels
    /// @param point 3D coordinate
    /// @return Vector4f containing RGBA noise values
    /// @note Each channel uses different offset; changes affect color patterns
    Vector4f permuteColor(const Vector3f& point) {
        // TIME_FUNCTION;
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
    float fractalNoise(const Vector2f& point, int octaves, float persistence = 0.5, float lacunarity = 2.0) {
        float frequency = 1.f;
        float amplitude = 1.f;
        float maxV = 0.f;
        for (int i = 0; i < octaves; i++) {
            Vector2f scaledPoint = point * frequency;
            maxV += permute(scaledPoint) * amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return maxV;
    }

    /// @brief Generate 3D fractal (octave) noise
    /// @param point Input coordinate
    /// @param octaves Number of noise layers
    /// @param persistence Amplitude multiplier per octave
    /// @param lacunarity Frequency multiplier per octave
    /// @return Combined noise value
    float fractalNoise(const Vector3f& point, int octaves, float persistence = 0.5, float lacunarity = 2.0) {
        float frequency = 1.f;
        float amplitude = 1.f;
        float maxV = 0.f;
        for (int i = 0; i < octaves; i++) {
            Vector3f scaledPoint = point * frequency;
            maxV += permute(scaledPoint) * amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return maxV;
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

    /// @brief Pure White Noise
    /// @param point Input coordinate
    /// @return Random value [-1, 1] based solely on integer coordinate hashing
    float whiteNoise(const Vector2f& point) {
        return hash((int)floor(point.x()), (int)floor(point.y()));
    }

    /// @brief Pure White Noise 3D
    float whiteNoise(const Vector3f& point) {
        return hash((int)floor(point.x()), (int)floor(point.y()), (int)floor(point.z()));
    }

    /// @brief Worley (Cellular) Noise 2D
    /// @param point Input coordinate
    /// @return Distance to the nearest feature point [0, 1+]
    /// @note Used for stone, water caustics, biological cells
    float worleyNoise(const Vector2f& point) {
        Vector2f p = Vector2f(floor(point.x()), floor(point.y()));
        Vector2f f = Vector2f(point.x() - p.x(), point.y() - p.y());

        float minDist = 1.0f;

        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vector2f neighbor(x, y);
                Vector2f pointInCell = hashVector(Vector2f(p + neighbor));
                Vector2f diff = neighbor + pointInCell - f;
                
                float dist = diff.norm();
                if (dist < minDist) minDist = dist;
            }
        }
        return minDist;
    }

    /// @brief Worley Noise 3D
    /// @param point Input coordinate
    /// @return Distance to nearest feature point
    float worleyNoise(const Vector3f& point) {
        Vector3f p = Vector3f(floor(point.x()), floor(point.y()), floor(point.z()));
        Vector3f f = Vector3f(point.x() - p.x(), point.y() - p.y(), point.z() - p.z());

        float minDist = 1.0f;

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    Vector3f neighbor(x, y, z);
                    Vector3f pointInCell = hashVector(Vector3f(p + neighbor));
                    Vector3f diff = neighbor + pointInCell - f;
                    float dist = diff.norm();
                    if (dist < minDist) minDist = dist;
                }
            }
        }
        return minDist;
    }

    /// @brief Voronoi Noise 2D (Cell ID)
    /// @param point Input coordinate
    /// @return Random hash value [-1, 1] unique to the closest cell
    float voronoiNoise(const Vector2f& point) {
        Vector2f p = Vector2f(floor(point.x()), floor(point.y()));
        Vector2f f = Vector2f(point.x() - p.x(), point.y() - p.y());

        float minDist = 100.0f;
        Vector2f cellID = p;

        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vector2f neighbor(x, y);
                Vector2f pointInCell = hashVector(Vector2f(p + neighbor));
                Vector2f diff = neighbor + pointInCell - f;
                float dist = diff.squaredNorm();
                if (dist < minDist) {
                    minDist = dist;
                    cellID = p + neighbor;
                }
            }
        }
        return hash((int)cellID.x(), (int)cellID.y());
    }

    /// @brief "Crystals" Noise (Variant of Worley)
    /// @param point Input coordinate
    /// @return F2 - F1 (Distance to 2nd closest - Distance to closest)
    /// @note Creates cell-like borders, cracks, or crystal facets
    float crystalNoise(const Vector2f& point) {
        Vector2f p = Vector2f(floor(point.x()), floor(point.y()));
        Vector2f f = Vector2f(point.x() - p.x(), point.y() - p.y());

        float d1 = 10.0f;
        float d2 = 10.0f;

        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vector2f neighbor(x, y);
                Vector2f pointInCell = hashVector(Vector2f(p + neighbor));
                Vector2f diff = neighbor + pointInCell - f;
                float dist = diff.norm();

                if (dist < d1) {
                    d2 = d1;
                    d1 = dist;
                } else if (dist < d2) {
                    d2 = dist;
                }
            }
        }
        return d2 - d1;
    }

    /// @brief Domain Warping
    /// @param point Input coordinate
    /// @param strength Magnitude of the warp
    /// @return Warped Perlin noise value
    /// @note Calculates noise(p + noise(p)) for marble/fluid effects
    float domainWarp(const Vector2f& point, float strength = 1.0f) {
        Vector2f q(
            permute(point),
            permute(Vector2f(point + Vector2f(5.2f, 1.3f)))
        );
        return permute(Vector2f(point + q * strength));
    }

    /// @brief 3D Domain Warping
    float domainWarp(const Vector3f& point, float strength = 1.0f) {
        Vector3f q(
            permute(point),
            permute(Vector3f(point + Vector3f(5.2f, 1.3f, 2.8f))),
            permute(Vector3f(point + Vector3f(1.1f, 8.4f, 5.5f)))
        );
        return permute(Vector3f(point + q * strength));
    }

    /// @brief Curl Noise 2D
    /// @param point Input coordinate
    /// @return Divergence-free vector field (useful for particle simulation)
    /// @note Calculated via finite difference curl of a potential field
    Vector2f curlNoise(const Vector2f& point) {
        float n1 = permute(Vector2f(point + Vector2f(0, EPSILON)));
        float n2 = permute(Vector2f(point + Vector2f(0, -EPSILON)));
        float n3 = permute(Vector2f(point + Vector2f(EPSILON, 0)));
        float n4 = permute(Vector2f(point + Vector2f(-EPSILON, 0)));

        float dx = (n3 - n4) / (2.0f * EPSILON);
        float dy = (n1 - n2) / (2.0f * EPSILON);
        return Vector2f(dy, -dx).normalized();
    }

    /// @brief Curl Noise 3D
    /// @param point Input coordinate
    /// @return Divergence-free vector field
    /// @note Uses 3 offsets of Perlin noise as Vector Potential
    Vector3f curlNoise(const Vector3f& point) {
        Vector3f dx(EPSILON, 0.0f, 0.0f);
        Vector3f dy(0.0f, EPSILON, 0.0f);
        Vector3f dz(0.0f, 0.0f, EPSILON);

        auto potential = [&](const Vector3f& p) -> Vector3f {
            return Vector3f(
                permute(p),
                permute(Vector3f(p + Vector3f(123.4f, 129.1f, 827.0f))), 
                permute(Vector3f(p + Vector3f(492.5f, 991.2f, 351.4f))) 
            );
        };

        Vector3f p_dx_p = potential(point + dx);
        Vector3f p_dx_m = potential(point - dx);
        Vector3f p_dy_p = potential(point + dy);
        Vector3f p_dy_m = potential(point - dy);
        Vector3f p_dz_p = potential(point + dz);
        Vector3f p_dz_m = potential(point - dz);

        // Finite difference
        float dFz_dy = (p_dy_p.z() - p_dy_m.z()) / (2.0f * EPSILON);
        float dFy_dz = (p_dz_p.y() - p_dz_m.y()) / (2.0f * EPSILON);
        float dFx_dz = (p_dz_p.x() - p_dz_m.x()) / (2.0f * EPSILON);
        float dFz_dx = (p_dx_p.z() - p_dx_m.z()) / (2.0f * EPSILON);
        float dFy_dx = (p_dx_p.y() - p_dx_m.y()) / (2.0f * EPSILON);
        float dFx_dy = (p_dy_p.x() - p_dy_m.x()) / (2.0f * EPSILON);

        return Vector3f(
            dFz_dy - dFy_dz,
            dFx_dz - dFz_dx,
            dFy_dx - dFx_dy
        ).normalized();
    }
};

#endif