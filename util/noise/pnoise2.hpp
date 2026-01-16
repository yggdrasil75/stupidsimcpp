#ifndef PNOISE2_HPP
#define PNOISE2_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"

class PNoise2 {
private:
    std::vector<int> permutation;
    std::default_random_engine rng;
    
    float lerp(float t, float a1, float a2) {
        return a1 + t * (a2 - a1);
    }

    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    Vec2f GetConstantVector(int v) {
        int h = v & 3;
        if (h == 0) return Vec2f(1,1);
        else if (h == 1) return Vec2f(-1,1);
        else if (h == 2) return Vec2f(-1,-1);
        else return Vec2f(1,-1);
    }

    Vec3ui8 GetConstantVector3(int v) {
        int h = v & 7;
        if (h == 0) return Vec3ui8(1,1,1);
        else if (h == 1) return Vec3ui8(-1,1, 1);
        else if (h == 2) return Vec3ui8(-1,-1, 1);
        else if (h == 3) return Vec3ui8(-1,-1, 1);
        else if (h == 4) return Vec3ui8(-1,-1,-1);
        else if (h == 5) return Vec3ui8(-1,-1, -1);
        else if (h == 6) return Vec3ui8(-1,-1, -1);
        else return Vec3ui8(1,-1, -1);
    }

    static double grad(int hash, double x, double y, double z = 0.0) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

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

    float normalizedNoise(const Vec2<float>& point) {
        return (permute(point) + 1.0f) * 0.5f;
    }

    float mapRange(float value, float inMin, float inMax, float outMin, float outMax) {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }

    float blendNoises(float noise1, float noise2, float blendFactor) {
        return lerp(noise1, noise2, blendFactor);
    }

    float addNoises(float noise1, float noise2) {
        return std::clamp(noise1 + noise2, -1.0f, 1.0f);
    }

    float multiplyNoises(float noise1, float noise2) {
        return noise1 * noise2;
    }

    float hash(int x, int y) {
        return (permutation[(x + permutation[y & 255]) & 255] / 255.0f) * 2.0f - 1.0f;
    }
public:
    PNoise2() : rng(std::random_device{}()) {
        initializePermutation();
    }
    
    PNoise2(unsigned int seed) : rng(seed) {
        initializePermutation();
    }
    
    template<typename T>
    float permute(Vec2<T> point) {
        TIME_FUNCTION;
        float x = static_cast<float>(point.x);
        float y = static_cast<float>(point.y);
        int X = (int)floor(x);
        int xmod = X & 255;
        int Y = (int)floor(point.y);
        int ymod = Y & 255;
        float xf = point.x - X;
        float yf = point.y - Y;

        Vec2 BL = Vec2(xf-0, yf-0);
        Vec2 BR = Vec2(xf-1, yf-0);
        Vec2 TL = Vec2(xf-0, yf-1);
        Vec2 TR = Vec2(xf-1, yf-1);

        int vBL = permutation[permutation[xmod+0]+ymod+0];
        int vBR = permutation[permutation[xmod+1]+ymod+0];
        int vTL = permutation[permutation[xmod+0]+ymod+1];
        int vTR = permutation[permutation[xmod+1]+ymod+1];

        float dBL = BL.dot(GetConstantVector(vBL));
        float dBR = BR.dot(GetConstantVector(vBR));
        float dTL = TL.dot(GetConstantVector(vTL));
        float dTR = TR.dot(GetConstantVector(vTR));
        
        float u = fade(xf);
        float v = fade(yf);

        float x1 = lerp(u, grad(vBL, xf, yf),     grad(vBR, xf - 1, yf));
        float x2 = lerp(u, grad(vTL, xf, yf - 1), grad(vTR, xf - 1, yf - 1));
        float retval = lerp(v, x1, x2);
        //std::cout << "returning: " << retval << std::endl;
        return retval;
    }

    template<typename T>
    float permute(Vec3<T> point) {
        TIME_FUNCTION;
        int X = (int)floor(point.x) & 255;
        int Y = (int)floor(point.y) & 255;
        int Z = (int)floor(point.z) & 255;
        float xf = point.x - X;
        float yf = point.y - Y;
        float zf = point.z - Z;

        Vec3ui8 FBL = Vec3ui8(xf-0, yf-0, zf-0);
        Vec3ui8 FBR = Vec3ui8(xf-1, yf-0, zf-0);
        Vec3ui8 FTL = Vec3ui8(xf-0, yf-1, zf-0);
        Vec3ui8 FTR = Vec3ui8(xf-1, yf-1, zf-0);

        Vec3ui8 RBL = Vec3ui8(xf-0, yf-0, zf-1);
        Vec3ui8 RBR = Vec3ui8(xf-1, yf-0, zf-1);
        Vec3ui8 RTL = Vec3ui8(xf-0, yf-1, zf-1);
        Vec3ui8 RTR = Vec3ui8(xf-1, yf-1, zf-1);

        int vFBL = permutation[permutation[permutation[Z+0]+X+0]+Y+0];
        int vFBR = permutation[permutation[permutation[Z+0]+X+1]+Y+0];
        int vFTL = permutation[permutation[permutation[Z+0]+X+0]+Y+1];
        int vFTR = permutation[permutation[permutation[Z+0]+X+1]+Y+1];

        int vRBL = permutation[permutation[permutation[Z+1]+X+0]+Y+0];
        int vRBR = permutation[permutation[permutation[Z+1]+X+1]+Y+0];
        int vRTL = permutation[permutation[permutation[Z+1]+X+0]+Y+1];
        int vRTR = permutation[permutation[permutation[Z+1]+X+1]+Y+1];

        float dFBL = FBL.dot(GetConstantVector3(vFBL));
        float dFBR = FBR.dot(GetConstantVector3(vFBR));
        float dFTL = FTL.dot(GetConstantVector3(vFTL));
        float dFTR = FTR.dot(GetConstantVector3(vFTR));

        float dRBL = RBL.dot(GetConstantVector3(vRBL));
        float dRBR = RBR.dot(GetConstantVector3(vRBR));
        float dRTL = RTL.dot(GetConstantVector3(vRTL));
        float dRTR = RTR.dot(GetConstantVector3(vRTR));
        
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

        //std::cout << "returning: " << retval << std::endl;
        return retval;
    }

    float valueNoise(const Vec2<float>& point) {
        int xi = (int)std::floor(point.x);
        int yi = (int)std::floor(point.y);
        
        float tx = point.x - xi;
        float ty = point.y - yi;
        
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

    Vec4ui8 permuteColor(const Vec3<float>& point) {
        TIME_FUNCTION;
        float noiseR = permute(point);
        float noiseG = permute(point + Vec3<float>(100.0f, 100.0f, 100.0f));
        float noiseB = permute(point + Vec3<float>(200.0f, 200.0f, 200.0f));
        float noiseA = permute(point + Vec3<float>(300.0f, 300.0f, 300.0f));
        // float rNormalized = (noiseR + 1.0f) * 0.5f;
        // float gNormalized = (noiseG + 1.0f) * 0.5f;
        // float bNormalized = (noiseB + 1.0f) * 0.5f;
        // float aNormalized = (noiseA + 1.0f) * 0.5f;
        // rNormalized = std::clamp(rNormalized, 0.0f, 1.0f);
        // gNormalized = std::clamp(gNormalized, 0.0f, 1.0f);
        // bNormalized = std::clamp(bNormalized, 0.0f, 1.0f);
        // aNormalized = std::clamp(aNormalized, 0.0f, 1.0f);
        uint8_t r = static_cast<uint8_t>(noiseR * 255.0f);
        uint8_t g = static_cast<uint8_t>(noiseG * 255.0f);
        uint8_t b = static_cast<uint8_t>(noiseB * 255.0f);
        uint8_t a = static_cast<uint8_t>(noiseA * 255.0f);
        
        return Vec4ui8(r, g, b, a);
    }

    template<typename T>
    float fractalNoise(const Vec2<T>& point, int octaves, float persistence, float lacunarity) {
        float total = 0.0f;
        float frequency = 1.f;
        float amplitude = 1.f;
        float maxV = 0.f;
        for (int i = 0; i < octaves; i++) {
            total += permute(point*frequency) * amplitude;
            maxV += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return total / maxV;
    }

    template<typename T>
    float turbulence(const Vec2<T>& point, int octaves) {
        float value = 0.0f;
        Vec2<float> tempPoint = point;
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(tempPoint));
            tempPoint *= 2.f;
        }
        return value;
    }

    float ridgedNoise(const Vec2<float>& point, int octaves, float offset = 1.0f) {
        float result = 0.f;
        float weight = 1.f;
        Vec2<float> p = point;
        
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

    float billowNoise(const Vec2<float>& point, int octaves) {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        
        for (int i = 0; i < octaves; i++) {
            value += std::abs(permute(point * frequency)) * amplitude;
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }
        
        return value;
    }


};

#endif