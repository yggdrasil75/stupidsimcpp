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
}

#endif