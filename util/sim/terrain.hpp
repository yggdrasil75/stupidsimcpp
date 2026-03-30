#ifndef TERRAIN_HPP
#define TERRAIN_HPP

#include <memory>
#include <vector>
#include <fstream>
#include "../../eigen/Eigen/Dense"
#include "../grid/grid2eigen.hpp"
#include "../util/noise/pnoise.cpp"

using v2 = Eigen::Vector2f;

struct TerrainData {
    float elevation = 0.0f;
    float water = 0.0f;
    float clay = 0.0f;
    float sand = 0.0f;
    float loam = 0.0f;
    float silt = 0.0f;
    float root = 0.0f;
    float organics = 0.0f;
    // generic "rock" will probably later become something better.
    float rocktype1 = 0.0f;
    float rocktype2 = 0.0f;
    float rocktype3 = 0.0f;
    float rocktype4 = 0.0f;
    float rocktype5 = 0.0f;
    float igneous = 0.0f;
    float metamorphic = 0.0f;
    
    TerrainData() = default;
    
    void serialize(std::ofstream& out) const {
    ///TODO
    }
    
    static std::shared_ptr<TerrainData> deserialize(std::ifstream& in) {
    ///TODO
    }
};

class TerrainSim {
public:
    Quadtree<std::shared_ptr<TerrainData>> grid;
    
    TerrainSim() {
        grid = Quadtree<std::shared_ptr<TerrainData>>(v2(-1000, -1000), v2(1000, 1000), 16, 12);
        grid.setBackgroundColor(Eigen::Vector3f(0.1f, 0.1f, 0.1f));
    }
    
    void generateTerrain(const NoisePreviewState& noiseState) {
        ///TODO: uses noise to create the elevation map
        generateRegion(Eigen::Vector3f(1024.0f, 0, 0), 1000.0f, 10.0f, noiseState);
    }

    void generaterockMap1(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each rock map
    }

    void generaterockMap2(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each rock map
    }

    void generaterockMap3(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each rock map
    }

    void generaterockMap4(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each rock map
    }

    void generaterockMap5(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each rock map
    }
    
    void generatesiltMap(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each map
    }

    void generatesandMap(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each map
    }

    void generateclayMap(const NoisePreviewState& noiseState) {
        ///TODO: use the noise map to generate each map
    }
};

#endif