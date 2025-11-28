#ifndef PNOISE2_HPP
#define PNOISE2_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
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

    Vec2 GetConstantVector(int v) {
        int h = v & 3;
        if (h == 0) return Vec2(1,1);
        else if (h == 1) return Vec2(-1,1);
        else if (h == 2) return Vec2(-1,-1);
        else return Vec2(1,-1);
    }

    Vec3 GetConstantVector3(int v) {
        int h = v & 7;
        if (h == 0) return Vec3(1,1,1);
        else if (h == 1) return Vec3(-1,1, 1);
        else if (h == 2) return Vec3(-1,-1, 1);
        else if (h == 3) return Vec3(-1,-1, 1);
        else if (h == 4) return Vec3(-1,-1,-1);
        else if (h == 5) return Vec3(-1,-1, -1);
        else if (h == 6) return Vec3(-1,-1, -1);
        else return Vec3(1,-1, -1);
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
public:
    PNoise2() : rng(std::random_device{}()) {
        initializePermutation();
    }
    
    PNoise2(unsigned int seed) : rng(seed) {
        initializePermutation();
    }
    
    float permute(Vec2 point) {
        TIME_FUNCTION;
        float x = point.x;
        float y = point.y;
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

    float permute(Vec3 point) {
        TIME_FUNCTION;
        int X = (int)floor(point.x) & 255;
        int Y = (int)floor(point.y) & 255;
        int Z = (int)floor(point.z) & 255;
        float xf = point.x - X;
        float yf = point.y - Y;
        float zf = point.z - Z;

        Vec3 FBL = Vec3(xf-0, yf-0, zf-0);
        Vec3 FBR = Vec3(xf-1, yf-0, zf-0);
        Vec3 FTL = Vec3(xf-0, yf-1, zf-0);
        Vec3 FTR = Vec3(xf-1, yf-1, zf-0);

        Vec3 RBL = Vec3(xf-0, yf-0, zf-1);
        Vec3 RBR = Vec3(xf-1, yf-0, zf-1);
        Vec3 RTL = Vec3(xf-0, yf-1, zf-1);
        Vec3 RTR = Vec3(xf-1, yf-1, zf-1);

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

        std::cout << "returning: " << retval << std::endl;
        return retval;
    }

};

#endif
//https://rtouti.github.io/graphics/perlin-noise-algorithm
