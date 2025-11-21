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
    PNoise2() {
        permutation.reserve(256);
        for (int i = 0; i < 256; i++){
            permutation[i] = i;
        }
        std::ranges::shuffle(permutation, rng);
        TR = permutation[permutation[1]+1];
        TR = permutation[permutation[0]+1];
        TR = permutation[permutation[1]+0];
        TR = permutation[permutation[0]+0];
    }

    Vec2 GetConstantVector(Vec2 v) {
        Vec2 h = v & 3;
    }

};

#endif
//https://rtouti.github.io/graphics/perlin-noise-algorithm