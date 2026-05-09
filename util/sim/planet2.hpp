#ifndef PLANET_HPP
#define PLANET_HPP

#include "../grid/grid3eigen.hpp"
#include "../timing_decorator.cpp"

using v3 = Eigen::Vector3f;
using v3half = Eigen::Matrix<Eigen::half, 3, 1>;

// struct AltPos {
//     v3 originalPos;
//     v3 noisePos;
//     v3 tectonicPos;
//     v3 erosionPos;
// };

struct Point {
    int plateID = -1;
    int16_t altDeltas[12];
    v3 currentPos;
    v3half velocity;
    v3half acceleration;
    v3half forceAccumulator;
    v3half pressureForce;
    v3half viscosityForce;
    Eigen::SparseMatrix<float> materials;
};

struct world {
    
}

#endif