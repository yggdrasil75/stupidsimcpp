#ifndef PNOISE2_HPP
#define PNOISE2_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include "../vectorlogic/vec2.hpp"

class PNoise2 {
private:
    std::vector<float> permutation;
    std::default_random_engine rng;
    float TR,TL,BR,BL;
public:
    PNoise2() : rng(std::random_device{}()){
        //permutation.reserve(256);
        std::vector<float> permutationt;
        for (int i = 0; i < 255; i++){
            //permutation[i] = i;
            permutationt.push_back(i);
        }
        std::ranges::shuffle(permutationt, rng);
        permutation.insert(permutation.end(),permutationt.begin(),permutationt.end());
        permutation.insert(permutation.end(),permutationt.begin(),permutationt.end());
    }

    float permute(Vec2 point) {
        float x = point.x;
        float y = point.y;
        int X = (int)floor(x);
        int xmod = X & 255;
        int Y = (int)floor(point.y);
        int ymod = Y & 255;
        float xf = point.x - X;
        float yf = point.y - Y;

        Vec2 TR = Vec2(xf-1, yf-1);
        Vec2 TL = Vec2(xf-0, yf-1);
        Vec2 BR = Vec2(xf-1, yf-0);
        Vec2 BL = Vec2(xf-0, yf-0);

        int vTR = permutation[permutation[xmod+1]+ymod+1];
        int vTL = permutation[permutation[xmod+0]+ymod+1];
        int vBR = permutation[permutation[xmod+1]+ymod+0];
        int vBL = permutation[permutation[xmod+0]+ymod+0];

        float dTR = TR.dot(GetConstantVector(vTR));
        float dTL = TL.dot(GetConstantVector(vTL));
        float dBR = BR.dot(GetConstantVector(vBR));
        float dBL = BL.dot(GetConstantVector(vBL));
        
        float u = Fade(xf);
        float v = Fade(yf);

        return lerp(u,lerp(v,dBL,dTL),lerp(v,dBR,dTR));
        
    }

    float lerp(float t, float a1, float a2) {
        return a1 + t * (a2 - a1);
    }

    float Fade(float t) {
        return (((6 * t - 15)* t + 10) * t * t * t);
    }

    Vec2 GetConstantVector(float v) {
        int h = (int)v & 3;
        if (h == 0) return Vec2(1,1);
        else if (h == 1) return Vec2(-1,1);
        else if (h == 2) return Vec2(-1,-1);
        else return Vec2(1,-1);
    }

};

#endif
//https://rtouti.github.io/graphics/perlin-noise-algorithm
