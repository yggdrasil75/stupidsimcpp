#ifndef SIM2_HPP
#define SIM2_HPP

#include "../util/noise2.hpp"
#include "../util/grid2.hpp"
#include "../util/vec2.hpp"
#include "../util/vec4.hpp"
#include <memory>
#include <string>
#include <unordered_map>

class Sim2 {
private:
    Noise2 noise;
    Grid2 terrainGrid;
    int width;
    int height;

    float scale;
    int octaves;
    float lacunarity;
    int seed;
    float elevationMult;
    float waterLevel;
    Vec4 landColor;
    Vec4 waterColor;
    float erosion;
public:
    Sim2(int width = 512, int height = 512, int seed = 42, float scale = 4) : 
        width(width), height(height), scale(scale), octaves(4), seed(seed)
    { }
};

#endif