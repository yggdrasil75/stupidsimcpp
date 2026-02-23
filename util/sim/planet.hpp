#ifndef PLANET_HPP
#define PLANET_HPP

#include <map>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <map>

#include "../grid/grid3eigen.hpp"
#include "../timing_decorator.cpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

using v3 = Eigen::Vector3f;
const float Φ = M_PI * (3.0f - std::sqrt(5.0f));

enum class PlateType {
    CONTINENTAL,
    OCEANIC,
    MIXED
};

struct Particle {
    float noiseDisplacement = 0.0f;
    int plateID = -1;
    Eigen::Vector3f basePos;
    Eigen::Vector3f currentPos;
    float plateDisplacement = 0.0f;
    float temperature = -1;
    float water = -1;
    Eigen::Vector3f originColor;
    bool surface = false;

    //gravity factors:
    Eigen::Matrix<float, 3, 1> velocity;
    Eigen::Matrix<float, 3, 1> acceleration;
    Eigen::Matrix<float, 3, 1> forceAccumulator;
    float density = 0.0f;
    float pressure = 0.0f;
    Eigen::Matrix<float, 3, 1> pressureForce;
    float viscosity = 0.5f;
    Eigen::Matrix<float, 3, 1> viscosityForce;
    float restitution = 5.0f;
    float mass;
    bool isStatic = false;
    float soundSpeed = 100.0f;

    std::unordered_map<int, float> neighbors;
    std::map<float, int> nearNeighbors;
};

struct planetConfig {
    Eigen::Vector3f center = Eigen::Vector3f(0,0,0);
    float radius = 1024.0f;
    Eigen::Vector3f color = Eigen::Vector3f(0, 1, 0);
    
    float voxelSize = 10.0f;
    int surfacePoints = 50000;
    
    int currentStep = 0;
    
    float displacementStrength = 200.0f;
    std::vector<Particle> surfaceNodes;
    float noiseStrength = 1.0f;
    int numPlates = 15;
    float plateRandom = 0.6f;
    int smoothingPasses = 3;
    float mountHeight = 250.0f;
    float valleyDepth = -150.0f;
    float transformRough = 80.0f;
    int stressPasses = 5;
    float maxElevationRatio = 0.25f;
    float gridSizeCube = 16384;
    float SMOOTHING_RADIUS = 1024.0f;
    float REST_DENSITY = 0.00005f;
    float TIMESTEP = 0.016f;
    float G_ATTRACTION = 50.0f;
    float gravitySoftening = 10.0f;
    float pressureStiffness = 50000.0f;
    float coreRepulsionRadius = 1000.0f; 
    float coreRepulsionStiffness = 100000.0f;
    float dampingFactor = 0.98f;
};

struct PlateConfig {
    int plateId = -1;
    Eigen::Vector3f plateEulerPole;
    Eigen::Vector3f direction;
    float angularVelocity = 0;
    float thickness = 0;
    float density = 0;
    float rigidity = 0;
    float temperature = 0;
    Eigen::Vector3f debugColor;
    PlateType ptype = PlateType::MIXED;
    std::vector<int> assignedNodes;
};

class planetsim {
public:
    planetConfig config;
    Octree<Particle> grid;
    std::vector<PlateConfig> plates;
    std::mt19937 rng = std::mt19937(42);

    planetsim() {
        config = planetConfig();
        grid = Octree<Particle>({-config.gridSizeCube,-config.gridSizeCube,-config.gridSizeCube,},{config.gridSizeCube,config.gridSizeCube,config.gridSizeCube});
    }

    float evaluate2DStack(const Eigen::Vector2f& point, const NoisePreviewState& state, PNoise2& gen) {
        float finalValue = 0.0f;
        Eigen::Vector2f p = point;

        for (const auto& layer : state.layers) {
            if (!layer.enabled) continue;

            Eigen::Vector2f samplePoint = p * layer.scale;
            
            samplePoint += Eigen::Vector2f((float)layer.seedOffset * 10.5f, (float)layer.seedOffset * -10.5f);

            if (layer.blend == BlendMode::DomainWarp) {
                if (layer.type == NoiseType::CurlNoise) {
                        Eigen::Vector2f flow = gen.curlNoise(samplePoint);
                        p += flow * layer.strength * 100.0f; 
                } else {
                    float warpX = sampleNoiseLayer(gen, layer.type, samplePoint, layer);
                    float warpY = sampleNoiseLayer(gen, layer.type, samplePoint + Eigen::Vector2f(5.2f, 1.3f), layer);
                    p += Eigen::Vector2f(warpX, warpY) * layer.strength * 100.0f;
                }
                continue; 
            }

            float nVal = sampleNoiseLayer(gen, layer.type, samplePoint, layer);
            
            switch (layer.blend) {
                case BlendMode::Replace:   finalValue = nVal * layer.strength; break;
                case BlendMode::Add:       finalValue += nVal * layer.strength; break;
                case BlendMode::Subtract:  finalValue -= nVal * layer.strength; break;
                case BlendMode::Multiply:  finalValue *= (nVal * layer.strength); break;
                case BlendMode::Max:       finalValue = std::max(finalValue, nVal * layer.strength); break;
                case BlendMode::Min:       finalValue = std::min(finalValue, nVal * layer.strength); break;
            }
        }
        
        float norm = std::tanh(finalValue);
        return norm;
    }

    void generateFibSphere() {
        TIME_FUNCTION;
        grid.clear();
        config.surfaceNodes.clear();
        for (int i = 0; i < config.surfacePoints; i++) {
            float y = 1.0f - (i * 2.0f) / (config.surfacePoints - 1);
            float radiusY = std::sqrt(1.0f- y * y);
            float Θ = Φ * i;
            float x = std::cos(Θ) * radiusY;
            float z = std::sin(Θ) * radiusY;

            v3 dir(x, y, z);
            v3 pos = config.center + dir * config.radius;
            Particle pt;
            pt.basePos = pos;
            pt.currentPos = pos;
            pt.originColor = config.color;
            pt.noiseDisplacement = 0.0f;
            pt.surface = true;
            config.surfaceNodes.emplace_back(pt);
            grid.set(pt, pt.currentPos, true, pt.originColor, config.voxelSize, true, 1, 0, false, 0.0f, 0.0f, 0.0f);
        }
        config.currentStep = 1;
        std::cout << "Step 1 done. base sphere generated" << std::endl;
        buildAdjacencyList();
        grid.save("output/fibSphere");
    }

    inline void _applyNoise(std::function<float(const Eigen::Vector3f&)> noiseFunc) {
        for (auto& p : config.surfaceNodes) {
            Eigen::Vector3f oldPos = p.currentPos;
            float displacementValue = noiseFunc(p.basePos);
            p.noiseDisplacement = displacementValue;
            Eigen::Vector3f normal = p.basePos.normalized();
            p.currentPos = p.basePos + (normal * displacementValue * config.displacementStrength);

            grid.update(oldPos, p.currentPos, p, true, p.originColor, config.voxelSize, true, -2, false, 0.0f, 0.0f, 0.0f);
        }
    }

    void assignSeeds() {
        plates.clear();
        plates.resize(config.numPlates);
        float sphereSurfaceArea = 4.0f * M_PI * config.radius * config.radius;
        float averageAreaPerPlate = sphereSurfaceArea / config.numPlates;
        float minDistance = std::sqrt(averageAreaPerPlate) * 0.4f;
        std::vector<int> selectedSeedIndices;
        std::uniform_int_distribution<int> distNode(0, config.surfaceNodes.size() - 1);
        for (int i = 0; i < config.numPlates; ++i) {
            int attempts = 1000;
            bool foundValidSeed = false;
            int seedid = distNode(rng);
            plates[i].plateId = i;

            
            while (!foundValidSeed && attempts > 0) {
                int seedIndex = distNode(rng);
                
                bool tooClose = false;
                for (int selectedIndex : selectedSeedIndices) {
                    const auto& existingSeed = config.surfaceNodes[selectedIndex];
                    const auto& candidateSeed = config.surfaceNodes[seedIndex];
                    
                    float dot = existingSeed.basePos.normalized().dot(candidateSeed.basePos.normalized());
                    float angle = std::acos(std::clamp(dot, -1.0f, 1.0f));
                    float distanceOnSphere = angle * config.radius;
                    
                    if (distanceOnSphere < minDistance) {
                        tooClose = true;
                        break;
                    }
                }
                
                if (!tooClose || selectedSeedIndices.empty()) {
                    selectedSeedIndices.push_back(seedIndex);
                    plates[i].plateId = i;
                    config.surfaceNodes[seedIndex].plateID = i; 
                    float colorVal = static_cast<float>(seedid) / config.surfaceNodes.size();
                    if (i % 3 == 0) {
                        float r = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                        plates[i].debugColor = v3(r, colorVal, colorVal);   
                    } else if (i % 3 == 1) {
                        float g = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                        plates[i].debugColor = v3(colorVal, g, colorVal);
                    } else {
                        float b = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                        plates[i].debugColor = v3(colorVal, colorVal, b);
                    }
                    
                    foundValidSeed = true;
                }
                
                attempts--;
            }
            if (!foundValidSeed) {
                int seedIndex = distNode(rng);
                selectedSeedIndices.push_back(seedIndex);
                plates[i].plateId = i;
                float colorVal = static_cast<float>(seedIndex) / config.surfaceNodes.size();
                if (i % 3 == 0) {
                    float r = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                    plates[i].debugColor = v3(r, colorVal, colorVal);   
                } else if (i % 3 == 1) {
                    float g = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                    plates[i].debugColor = v3(colorVal, g, colorVal);
                } else {
                    float b = static_cast<float>(seedid * seedid) / config.surfaceNodes.size();
                    plates[i].debugColor = v3(colorVal, colorVal, b);
                }
                config.surfaceNodes[seedIndex].plateID = i; 
            }
        }
        
    }

    void buildAdjacencyList() {
        TIME_FUNCTION;
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            Particle& in = config.surfaceNodes[i];
            int test_idx = 0;
            if (test_idx == i) test_idx++;
            float current_radius = (in.basePos - config.surfaceNodes[test_idx].basePos).norm();
            in.neighbors[test_idx] = current_radius; 
            std::vector<std::pair<float, int>> valid_results;

            int safety_counter = 0;
            while (safety_counter++ < 1000) { 
                auto results = grid.findInRadius(in.basePos, current_radius);
                valid_results.clear();
                valid_results.reserve(results.size());
            
                for (const auto& node : results) {
                    int j = node->subId;
                    if (i == j) continue;
                    float dist = (in.basePos - node->position).norm();
                    in.neighbors[j] = dist; 
                    valid_results.push_back({dist, j});
                }
                
                int count = valid_results.size();
                
                if (count < 8) {
                    test_idx++;
                    if (test_idx == i) test_idx++;
                    if (test_idx >= config.surfaceNodes.size()) break;
                    
                    current_radius = (in.basePos - config.surfaceNodes[test_idx].basePos).norm();
                    in.neighbors[test_idx] = current_radius;
                } 
                else if (count > 16) {
                    float new_radius = valid_results[0].first;
                    if (new_radius >= current_radius || new_radius == 0.0f) {
                        current_radius *= 0.9f; 
                    } else {
                        current_radius = new_radius;
                    }
                }
            }
            
            std::sort(valid_results.begin(), valid_results.end());
            
            int max_neighbors = std::min(8, (int)valid_results.size());
            for (int k = 0; k < max_neighbors; k++) {
                in.nearNeighbors[valid_results[k].first] = valid_results[k].second;
            }
        }
    }

    void growPlatesRandom() {
        TIME_FUNCTION;
        std::uniform_int_distribution<int> distPlate(0, config.numPlates);
        std::vector<int> unassigned;
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            if (config.surfaceNodes[i].plateID != -1)
                unassigned.emplace_back(i);
            else plates[config.surfaceNodes[i].plateID].assignedNodes.emplace_back(i);
        }

        while (!unassigned.empty()) {
            int selPlate = distPlate(rng);
            
        }
    }

    void growPlatesCellular() {
        TIME_FUNCTION;

    }

    void fixBoundaries() {
        TIME_FUNCTION;
    }

    void extraplateste() {

    }

    void boundaryStress() {
        TIME_FUNCTION;

    }

    void finalizeApplyResults() {
        TIME_FUNCTION;
        float maxAllowedDisp = config.radius * config.maxElevationRatio;

        for (auto& p : config.surfaceNodes) {
            Eigen::Vector3f oldPos = p.currentPos;
            grid.update(oldPos, p.currentPos, p, true, plates[p.plateID].debugColor, config.voxelSize, true, -2, false, 0.0f, 0.0f, 0.0f);
        }
        std::cout << "Finalize apply results completed." << std::endl;
    }

};

#endif