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
        out.write(reinterpret_cast<const char*>(&elevation), sizeof(elevation));
        out.write(reinterpret_cast<const char*>(&water), sizeof(water));
        out.write(reinterpret_cast<const char*>(&clay), sizeof(clay));
        out.write(reinterpret_cast<const char*>(&sand), sizeof(sand));
        out.write(reinterpret_cast<const char*>(&loam), sizeof(loam));
        out.write(reinterpret_cast<const char*>(&silt), sizeof(silt));
        out.write(reinterpret_cast<const char*>(&root), sizeof(root));
        out.write(reinterpret_cast<const char*>(&organics), sizeof(organics));
        out.write(reinterpret_cast<const char*>(&rocktype1), sizeof(rocktype1));
        out.write(reinterpret_cast<const char*>(&rocktype2), sizeof(rocktype2));
        out.write(reinterpret_cast<const char*>(&rocktype3), sizeof(rocktype3));
        out.write(reinterpret_cast<const char*>(&rocktype4), sizeof(rocktype4));
        out.write(reinterpret_cast<const char*>(&rocktype5), sizeof(rocktype5));
        out.write(reinterpret_cast<const char*>(&igneous), sizeof(igneous));
        out.write(reinterpret_cast<const char*>(&metamorphic), sizeof(metamorphic));
    }
    
    static std::shared_ptr<TerrainData> deserialize(std::ifstream& in) {
        auto t = std::make_shared<TerrainData>();
        in.read(reinterpret_cast<char*>(&t->elevation), sizeof(elevation));
        in.read(reinterpret_cast<char*>(&t->water), sizeof(water));
        in.read(reinterpret_cast<char*>(&t->clay), sizeof(clay));
        in.read(reinterpret_cast<char*>(&t->sand), sizeof(sand));
        in.read(reinterpret_cast<char*>(&t->loam), sizeof(loam));
        in.read(reinterpret_cast<char*>(&t->silt), sizeof(silt));
        in.read(reinterpret_cast<char*>(&t->root), sizeof(root));
        in.read(reinterpret_cast<char*>(&t->organics), sizeof(organics));
        in.read(reinterpret_cast<char*>(&t->rocktype1), sizeof(rocktype1));
        in.read(reinterpret_cast<char*>(&t->rocktype2), sizeof(rocktype2));
        in.read(reinterpret_cast<char*>(&t->rocktype3), sizeof(rocktype3));
        in.read(reinterpret_cast<char*>(&t->rocktype4), sizeof(rocktype4));
        in.read(reinterpret_cast<char*>(&t->rocktype5), sizeof(rocktype5));
        in.read(reinterpret_cast<char*>(&t->igneous), sizeof(igneous));
        in.read(reinterpret_cast<char*>(&t->metamorphic), sizeof(metamorphic));
        return t;
    }
};

class TerrainSim {
public:
    Quadtree<std::shared_ptr<TerrainData>> grid;
    
    v2 cornerMin = v2(-500.0f, -500.0f);
    v2 cornerMax = v2(500.0f, 500.0f);
    float resolution = 10.0f;

    TerrainSim() {
        grid = Quadtree<std::shared_ptr<TerrainData>>(cornerMin, cornerMax, 16, 12);
        grid.setBackgroundColor(Eigen::Vector3f(0.1f, 0.1f, 0.1f));
    }
    
    float evaluate2DStack(const Eigen::Vector2f& pos, const NoisePreviewState& state, PNoise2& gen) {
        float totalVal = 0.0f;
        for (const auto& layer : state.layers) {
            float val = 0.0f;
            float freq = layer.scale;
            float amp = 1.0f;
            float max = 0.0f;

            for (int i = 0; i < layer.octaves; ++i) {
                val += gen.noise(pos.x() * freq + layer.offsetX, pos.y() * freq + layer.offsetY) * amp;
                max += amp;
                amp *= layer.persistence;
                freq *= layer.lacunarity;
            }
            if (max > 0) val /= max;
            
            val *= layer.strength;
            totalVal += val;
        }
        return totalVal;
    }
    
    void generateRegion(Eigen::Vector3f center3D, float size, float res, const NoisePreviewState& noiseState) {
        ///TODO: create the initial region based on the factors provided from the planet (mostly terrain height)
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

    void Erosion() {
        ///TODO: using the elevation map and the various rocks and clay, erode from higher points to lower, transferring the rocks downstream.
        /// igneous and metamorphic will break down into sand silt and clay. 
        /// the 5 rock types will remain (ie: sand from granite type idea)
        /// root will slow the erosion slightly less than rocks
    }
};

#endif