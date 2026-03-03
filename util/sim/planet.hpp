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
#include <set>
#include <memory>

#include "../grid/grid3eigen.hpp"
#include "../timing_decorator.cpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

using v3 = Eigen::Vector3f;
using v3half = Eigen::Matrix<Eigen::half, 3, 1>; // 16-bit vector alias
const float Φ = M_PI * (3.0f - std::sqrt(5.0f));

enum class PlateType {
    CONTINENTAL,
    OCEANIC,
    MIXED
};

struct AltPositions {
    v3half originalPos;
    v3half noisePos;
    v3half tectonicPos;
};

struct NeighborData {
    int index = -1;
    float distance = 0.0f;
};

struct Particle {
    float noiseDisplacement = 0.0f;
    int plateID = -1;
    
    std::unique_ptr<AltPositions> altPos = nullptr;

    Eigen::Vector3f currentPos;
    
    float plateDisplacement = 0.0f;
    // float temperature = -1;
    // float water = -1;
    v3half originColor;
    bool surface = false;

    //gravity factors:
    // Eigen::Matrix<float, 3, 1> velocity;
    // Eigen::Matrix<float, 3, 1> acceleration;
    // Eigen::Matrix<float, 3, 1> forceAccumulator;
    // float density = 0.0f;
    // float pressure = 0.0f;
    // Eigen::Matrix<float, 3, 1> pressureForce;
    // float viscosity = 0.5f;
    // Eigen::Matrix<float, 3, 1> viscosityForce;
    // float restitution = 5.0f;
    // float mass;
    // bool isStatic = false;
    // float soundSpeed = 100.0f;
    // float sandcontent = 0.0f;
    // float siltcontent = 0.0f;
    // float claycontent = 0.0f;
    // float rockcontent = 0.0f;
    // float metalcontent = 0.0f;

    NeighborData nearNeighbors[8];

    Particle() = default;

    Particle(const Particle& other) {
        noiseDisplacement = other.noiseDisplacement;
        plateID = other.plateID;
        currentPos = other.currentPos;
        plateDisplacement = other.plateDisplacement;
        originColor = other.originColor;
        surface = other.surface;
        
        for(int i = 0; i < 8; ++i) {
            nearNeighbors[i] = other.nearNeighbors[i];
        }
        
        if (other.altPos) {
            altPos = std::make_unique<AltPositions>(*other.altPos);
        }
    }

    Particle& operator=(const Particle& other) {
        if (this != &other) {
            noiseDisplacement = other.noiseDisplacement;
            plateID = other.plateID;
            currentPos = other.currentPos;
            plateDisplacement = other.plateDisplacement;
            originColor = other.originColor;
            surface = other.surface;
            
            for(int i = 0; i < 8; ++i) {
                nearNeighbors[i] = other.nearNeighbors[i];
            }
            
            if (other.altPos) {
                altPos = std::make_unique<AltPositions>(*other.altPos);
            } else {
                altPos.reset();
            }
        }
        return *this;
    }

    Particle(Particle&&) noexcept = default;
    Particle& operator=(Particle&&) noexcept = default;
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
    std::vector<Particle> interpolatedNodes;
    float noiseStrength = 10.0f;
    int numPlates = 15;
    int smoothingPasses = 3;
    float mountHeight = 250.0f;
    float valleyDepth = -150.0f;
    float transformRough = 80.0f;
    int stressPasses = 5;
    float maxElevationRatio = 0.25f;

    float gridSizeCube = 65536; //absolute max size for all nodes
    float gridSizeCubeMin = 16384; //max size, if something leaves this, then it probably needs to be purged before it leaves the grid and becomes lost
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
    Particle plateEulerPole;
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
        grid = Octree<Particle>(v3(-config.gridSizeCube,-config.gridSizeCube,-config.gridSizeCube),v3(config.gridSizeCube,config.gridSizeCube,config.gridSizeCube), 16, 32);
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
            
            pt.altPos = std::make_unique<AltPositions>();
            pt.altPos->originalPos = pos.cast<Eigen::half>();
            pt.altPos->noisePos = pos.cast<Eigen::half>();
            pt.altPos->tectonicPos = pos.cast<Eigen::half>();
            
            pt.currentPos = pos;
            pt.originColor = config.color.cast<Eigen::half>();
            pt.noiseDisplacement = 0.0f;
            pt.surface = true;
            config.surfaceNodes.emplace_back(pt);
            grid.set(pt, pt.currentPos, true, pt.originColor.cast<float>(), config.voxelSize, true, 1, 0, false, 0.0f, 0.0f, 0.0f);
        }
        config.currentStep = 1;
        std::cout << "Step 1 done. base sphere generated" << std::endl;
        grid.save("output/fibSphere");
    }

    inline void _applyNoise(std::function<float(const Eigen::Vector3f&)> noiseFunc) {
        for (auto& p : config.surfaceNodes) {
            Eigen::Vector3f oldPos = p.currentPos;
            float displacementValue = noiseFunc(p.altPos->originalPos.cast<float>());
            p.noiseDisplacement = displacementValue;
            Eigen::Vector3f normal = p.altPos->originalPos.cast<float>().normalized();
            p.altPos->noisePos = (p.altPos->originalPos.cast<float>() + (normal * displacementValue * config.noiseStrength)).cast<Eigen::half>();
            p.currentPos = p.altPos->noisePos.cast<float>();
            grid.move(oldPos, p.currentPos);
            grid.update(p.currentPos, p);
        }
        grid.optimize();
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
                    
                    float dot = existingSeed.altPos->originalPos.cast<float>().normalized().dot(candidateSeed.altPos->originalPos.cast<float>().normalized());
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
                    plates[i].plateEulerPole = config.surfaceNodes[seedIndex];

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
                plates[i].plateEulerPole = config.surfaceNodes[seedIndex];

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
        int numNodes = config.surfaceNodes.size();
        std::vector<v3> normPos(numNodes);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; i++) {
            normPos[i] = config.surfaceNodes[i].altPos->originalPos.cast<float>().normalized();
        }
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            Particle& in = config.surfaceNodes[i];
            v3 inn = normPos[i];
            std::priority_queue<std::pair<float, int>> top8;
            
            for (int j = 0; j < numNodes; j++) {
                if (i == j) {
                    continue;
                }
                float cosangle = std::clamp(inn.dot(normPos[j]), -1.0f, 1.0f);
                float angle = std::acos(cosangle);

                if (top8.size() < 8) {
                    top8.push({angle, j});
                } else if (angle < top8.top().first) {
                    top8.pop();
                    top8.push({angle, j});
                }
            }
            
            int nIdx = 0;
            while (!top8.empty() && nIdx < 8) {
                in.nearNeighbors[nIdx].index = top8.top().second;
                in.nearNeighbors[nIdx].distance = top8.top().first;
                nIdx++;
                top8.pop();
            }
        }
    }

    void growPlatesRandom() {
        TIME_FUNCTION;
        int unassignedCount = 0;
        std::vector<int> plateWeights(config.numPlates, 1);
        std::vector<std::vector<int>> frontiers(config.numPlates);
        
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            int pID = config.surfaceNodes[i].plateID;
            if (pID == -1) {
                unassignedCount++;
            } else {
                plates[pID].assignedNodes.push_back(i);
                for (int n = 0; n < 8; n++) {
                    int nIdx = config.surfaceNodes[i].nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    if (config.surfaceNodes[nIdx].plateID == -1) {
                        frontiers[pID].push_back(nIdx);
                    }
                }
            }
        }
        
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        std::cout << "have " << unassignedCount << " remaining nodes" << std::endl;
        
        while (unassignedCount > 0) {
            int totalWeight = 0;
            for (int i = 0; i < config.numPlates; i++) {
                totalWeight += plateWeights[i];
            }
            
            if (totalWeight <= 0) {
                std::cout << "something probably broke." << std::endl;
                break;
            }
            
            int randVal = distFloat(rng) * totalWeight;
            int selPlate = -1;
            float accum = 0.0f;
            for (int i = 0; i < config.numPlates; i++) {
                if (plateWeights[i] > 0) {
                    accum += plateWeights[i];
                    if (randVal <= accum) {
                        selPlate = i;
                        break;
                    }
                }
            }

            bool successfulGrowth = false;
            if (!frontiers[selPlate].empty()) {
                std::uniform_int_distribution<int> fDist(0, frontiers[selPlate].size() - 1);
                int fIdx = fDist(rng);
                int candIdx = frontiers[selPlate][fIdx];

                frontiers[selPlate][fIdx] = frontiers[selPlate].back();
                frontiers[selPlate].pop_back();

                if (config.surfaceNodes[candIdx].plateID == -1) {
                    config.surfaceNodes[candIdx].plateID = selPlate;
                    plates[selPlate].assignedNodes.push_back(candIdx);
                    unassignedCount--;
                    successfulGrowth = true;

                    for (int n = 0; n < 8; n++) {
                        int nIdx = config.surfaceNodes[candIdx].nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        if (config.surfaceNodes[nIdx].plateID == -1) {
                            frontiers[selPlate].push_back(nIdx);
                        }
                    }
                }
            }

            if (successfulGrowth) {
                plateWeights[selPlate] = 1;
                for (int i = 0; i < config.numPlates; i++) {
                    if (i != selPlate && plateWeights[i] > 0) {
                        plateWeights[i] += 1; 
                    }
                }
            }
        }
    }

    void growPlatesCellular() {
        TIME_FUNCTION;
        int unassignedCount = 0;
        for (const auto& p : config.surfaceNodes) {
            if (p.plateID == -1) unassignedCount++;
        }

        while (unassignedCount > 0) {
            std::vector<int> nextState(config.surfaceNodes.size(), -1);
            int assignedThisRound = 0;
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                if (config.surfaceNodes[i].plateID != -1) {
                    nextState[i] = config.surfaceNodes[i].plateID;
                } else {
                    std::unordered_map<int, int> counts;
                    int bestPlate = -1;
                    int maxCount = 0;
                    
                    for (int n = 0; n < 8; n++) {
                        int nIdx = config.surfaceNodes[i].nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        int pID = config.surfaceNodes[nIdx].plateID;
                        if (pID != -1) {
                            counts[pID]++;
                            if (counts[pID] > maxCount || (counts[pID] == maxCount && (rng() % 2 == 0))) {
                                maxCount = counts[pID];
                                bestPlate = pID;
                            }
                        }
                    }
                    if (bestPlate != -1) {
                        nextState[i] = bestPlate;
                        assignedThisRound++;
                    }
                }
            }
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                if (config.surfaceNodes[i].plateID == -1 && nextState[i] != -1) {
                    config.surfaceNodes[i].plateID = nextState[i];
                    plates[nextState[i]].assignedNodes.push_back(i);
                    unassignedCount--;
                }
            }
            
            if (assignedThisRound == 0 && unassignedCount > 0) {
                for (int i = 0; i < config.surfaceNodes.size(); i++) {
                    if (config.surfaceNodes[i].plateID == -1) {
                        int closestPlate = 0;
                        float minDist = std::numeric_limits<float>::max();
                        for (int p = 0; p < config.numPlates; p++) {
                            float d = (config.surfaceNodes[i].altPos->originalPos.cast<float>() - plates[p].plateEulerPole.altPos->originalPos.cast<float>()).norm();
                            if (d < minDist) {
                                minDist = d;
                                closestPlate = p;
                            }
                        }
                        config.surfaceNodes[i].plateID = closestPlate;
                        plates[closestPlate].assignedNodes.push_back(i);
                        unassignedCount--;
                    }
                }
            }
        }
    }

    void fixBoundaries() {
        TIME_FUNCTION;
        for (int pass = 0; pass < config.smoothingPasses; pass++) {
            std::vector<int> nextPlateID(config.surfaceNodes.size());
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                std::unordered_map<int, int> counts;
                counts[config.surfaceNodes[i].plateID]++;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = config.surfaceNodes[i].nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    counts[config.surfaceNodes[nIdx].plateID]++;
                }
                
                int bestPlate = config.surfaceNodes[i].plateID;
                int maxCount = 0;
                for (auto& pair : counts) {
                    if (pair.second > maxCount) {
                        maxCount = pair.second;
                        bestPlate = pair.first;
                    }
                }
                nextPlateID[i] = bestPlate;
            }
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                config.surfaceNodes[i].plateID = nextPlateID[i];
            }
        }
        
        for (auto& plate : plates) {
            plate.assignedNodes.clear();
        }
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            if (config.surfaceNodes[i].plateID != -1) {
                plates[config.surfaceNodes[i].plateID].assignedNodes.push_back(i);
            }
        }
    }

    void extraplateste() {
        TIME_FUNCTION;
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        std::vector<std::pair<int, float>> plateStats;
        for (int i = 0; i < config.numPlates; i++) {
            plateStats.emplace_back(std::make_pair(i, 0.0f));
        }

        for (int i = 0; i < config.numPlates; i++) {
            float sumElevation = 0.0f;
            Eigen::Vector3f centroid(0,0,0);
            
            for (int nIdx : plates[i].assignedNodes) {
                sumElevation += config.surfaceNodes[nIdx].currentPos.norm();
                centroid += config.surfaceNodes[nIdx].altPos->originalPos.cast<float>();
            }
            
            if (!plates[i].assignedNodes.empty()) {
                plateStats[i].second = sumElevation / plates[i].assignedNodes.size();
                centroid /= plates[i].assignedNodes.size();

                float maxSpread = 0.0f;
                for (int nIdx : plates[i].assignedNodes) {
                    float d = (config.surfaceNodes[nIdx].altPos->originalPos.cast<float>() - centroid).norm();
                    if (d > maxSpread) maxSpread = d;
                }

                float distToCentroid = (plates[i].plateEulerPole.altPos->originalPos.cast<float>() - centroid).norm();
                
                if (distToCentroid > maxSpread * 0.6f) {
                    int bestNodeIdx = plates[i].assignedNodes[0];
                    float minDistToCentroid = std::numeric_limits<float>::max();
                    
                    for (int nIdx : plates[i].assignedNodes) {
                        float d = (config.surfaceNodes[nIdx].altPos->originalPos.cast<float>() - centroid).norm();
                        if (d < minDistToCentroid) {
                            minDistToCentroid = d;
                            bestNodeIdx = nIdx;
                        }
                    }
                    plates[i].plateEulerPole = config.surfaceNodes[bestNodeIdx]; 
                }
            } else {
                plateStats[i].second = config.radius;
            }
            
            Eigen::Vector3f randomDir(distFloat(rng) - 0.5f, distFloat(rng) - 0.5f, distFloat(rng) - 0.5f);
            randomDir.normalize();
            
            Eigen::Vector3f poleDir = plates[i].plateEulerPole.altPos->originalPos.cast<float>().normalized();
            plates[i].direction = (randomDir - poleDir * randomDir.dot(poleDir)).normalized();
            
            plates[i].angularVelocity = distFloat(rng) * 0.1f + 0.02f;
            plates[i].rigidity = distFloat(rng) * 100.0f;
            plates[i].temperature = distFloat(rng) * 1000.0f;
        }
        
        std::sort(plateStats.begin(), plateStats.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second < b.second;
        });
        
        int oneThird = config.numPlates / 3;
        int twoThirds = (2 * config.numPlates) / 3;
        
        for (int i = 0; i < config.numPlates; i++) {
            int pID = plateStats[i].first;
            if (i < oneThird) {
                plates[pID].ptype = PlateType::OCEANIC;
                plates[pID].thickness = distFloat(rng) * 10.0f + 5.0f;
                plates[pID].density = distFloat(rng) * 500.0f + 3000.0f;
            } else if (i < twoThirds) {
                plates[pID].ptype = PlateType::MIXED;
                plates[pID].thickness = distFloat(rng) * 20.0f + 15.0f;
                plates[pID].density = distFloat(rng) * 500.0f + 2500.0f;
            } else {
                plates[pID].ptype = PlateType::CONTINENTAL;
                plates[pID].thickness = distFloat(rng) * 30.0f + 35.0f;
                plates[pID].density = distFloat(rng) * 500.0f + 2000.0f;
            }
        }
    }

    void boundaryStress() {
        TIME_FUNCTION;
        int numNodes = config.surfaceNodes.size();
        std::vector<float> nodeStress(numNodes, 0.0f);
        std::vector<float> nodeNoise(numNodes, 0.0f);
        
        std::vector<Eigen::Vector3f> ω(config.numPlates);
        for (int i = 0; i < config.numPlates; i++) {
            ω[i] = plates[i].plateEulerPole.altPos->originalPos.cast<float>().normalized().cross(plates[i].direction) * plates[i].angularVelocity;
        }

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int pass = 0; pass < config.stressPasses; pass++) {
            std::vector<float> newStress = nodeStress;
            std::vector<float> newNoise = nodeNoise;
            
            for (int i = 0; i < numNodes; i++) {
                int myPlate = config.surfaceNodes[i].plateID;
                if (myPlate == -1) continue;
                
                Eigen::Vector3f myPos = config.surfaceNodes[i].altPos->originalPos.cast<float>().normalized();
                Eigen::Vector3f myVel = ω[myPlate].cross(myPos);
                
                float localStress = 0.0f;
                float localNoise = 0.0f;
                int boundaryCount = 0;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = config.surfaceNodes[i].nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    
                    int nPlate = config.surfaceNodes[nIdx].plateID;
                    if (nPlate != -1 && myPlate != nPlate) {
                        boundaryCount++;
                        Eigen::Vector3f nPos = config.surfaceNodes[nIdx].altPos->originalPos.cast<float>().normalized();
                        Eigen::Vector3f nVel = ω[nPlate].cross(nPos);
                        
                        Eigen::Vector3f relVel = nVel - myVel;
                        Eigen::Vector3f dirToNeighbor = (nPos - myPos).normalized();
                        
                        float convergence = -relVel.dot(dirToNeighbor);
                        
                        PlateType myType = plates[myPlate].ptype;
                        PlateType nType = plates[nPlate].ptype;
                        
                        if (convergence > 0) { 
                            if (myType == PlateType::CONTINENTAL && nType == PlateType::OCEANIC) {
                                localStress += convergence * config.mountHeight;
                            } else if (myType == PlateType::OCEANIC && nType == PlateType::CONTINENTAL) {
                                localStress += convergence * config.valleyDepth;
                            } else {
                                localStress += convergence * config.mountHeight * 0.5f;
                            }
                            localNoise += convergence * config.transformRough;
                        } else { 
                            localStress += convergence * std::abs(config.valleyDepth) * 0.5f;
                            localNoise += std::abs(convergence) * config.transformRough * 0.5f;
                        }
                    }
                }
                
                if (boundaryCount > 0) {
                    newStress[i] = localStress / boundaryCount; 
                    newNoise[i] = localNoise / boundaryCount;
                } else {
                    float sumS = 0.0f;
                    float sumN = 0.0f;
                    int validNeighbors = 0;
                    
                    for (int n = 0; n < 8; n++) {
                        int nIdx = config.surfaceNodes[i].nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        sumS += nodeStress[nIdx];
                        sumN += nodeNoise[nIdx];
                        validNeighbors++;
                    }
                    if (validNeighbors > 0) {
                        float decay = 0.95f; 
                        newStress[i] = (sumS / validNeighbors) * decay;
                        newNoise[i] = (sumN / validNeighbors) * decay;
                    }
                }
            }
            nodeStress = newStress;
            nodeNoise = newNoise;
        }
        
        for (int i = 0; i < numNodes; i++) {
            Particle& p = config.surfaceNodes[i];
            p.plateDisplacement = nodeStress[i];
            
            float noiseVal = dist(rng) * nodeNoise[i];
            
            Eigen::Vector3f normal = p.altPos->originalPos.cast<float>().normalized();
            p.altPos->tectonicPos = (p.altPos->noisePos.cast<float>() + (normal * (p.plateDisplacement + noiseVal))).cast<Eigen::half>();
        }
    }

    void finalizeApplyResults() {
        TIME_FUNCTION;
        float maxAllowedDisp = config.radius * config.maxElevationRatio;

        for (auto& p : config.surfaceNodes) {
            Eigen::Vector3f oldPos = p.currentPos;
            p.currentPos = p.altPos->tectonicPos.cast<float>();
            grid.move(oldPos, p.currentPos);
            grid.update(p.currentPos, p);
        }
        grid.optimize();
        std::cout << "Finalize apply results completed." << std::endl;
    }

    void addStar() {
        ///TODO: add a star at roughly earth distance scaled based on planet radius.
    }

    void addMoon() {
        ///TODO: using planetConfig, add moon(s).
    }
    
    void stretchPlanet() {
        ///TODO: simulate millenia of gravitational stretching by nearby celestial bodies by squeezing the planet slightly at its poles
    }

    void interpolateSurface() {
        TIME_FUNCTION;
        
        config.interpolatedNodes.clear();

        std::set<std::tuple<int, int, int>> uniqueTriangles;

        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            Particle& p1 = config.surfaceNodes[i];
            for (int n1 = 0; n1 < 8; n1++) {
                int j = p1.nearNeighbors[n1].index;
                if (j == -1) break;
                if (j >= i) continue; 
                
                Particle& p2 = config.surfaceNodes[j];
                for (int n2 = 0; n2 < 8; n2++) {
                    int k = p2.nearNeighbors[n2].index;
                    if (k == -1) break;
                    if (k <= j) continue;
                    
                    bool isNeighbor = false;
                    for (int n3 = 0; n3 < 8; n3++) {
                        int nIdx = config.surfaceNodes[k].nearNeighbors[n3].index;
                        if (nIdx == -1) break;
                        if (nIdx == i) { isNeighbor = true; break; }
                    }
                    if (isNeighbor) {
                        uniqueTriangles.insert({i, j, k});
                    }
                }
            }
        }

        std::cout << "Identified " << uniqueTriangles.size() << " surface triangles. Filling..." << std::endl;

        size_t counter = 0;
        
        for (const auto& tri : uniqueTriangles) {
            int idx1 = std::get<0>(tri);
            int idx2 = std::get<1>(tri);
            int idx3 = std::get<2>(tri);

            const Particle& p1 = config.surfaceNodes[idx1];
            const Particle& p2 = config.surfaceNodes[idx2];
            const Particle& p3 = config.surfaceNodes[idx3];

            float d1 = (p2.currentPos - p1.currentPos).norm();
            float d2 = (p3.currentPos - p1.currentPos).norm();
            float d3 = (p3.currentPos - p2.currentPos).norm();
            float maxDist = std::max({d1, d2, d3});

            int steps = static_cast<int>(maxDist / config.voxelSize);
            if (steps < 1) steps = 1;

            for (int u = 0; u <= steps; u++) {
                for (int v = 0; v <= steps - u; v++) {
                    float w2 = (float)u / steps;
                    float w3 = (float)v / steps;
                    float w1 = 1.0f - w2 - w3;
                    
                    if (w1 > 0.99f || w2 > 0.99f || w3 > 0.99f) continue;

                    v3 interpNormal = (p1.altPos->originalPos.cast<float>().normalized() * w1 + 
                                       p2.altPos->originalPos.cast<float>().normalized() * w2 + 
                                       p3.altPos->originalPos.cast<float>().normalized() * w3);
                    interpNormal.normalize(); 
                    
                    float r1 = p1.currentPos.norm();
                    float r2 = p2.currentPos.norm();
                    float r3 = p3.currentPos.norm();
                    float interpRadius = (r1 * w1) + (r2 * w2) + (r3 * w3);

                    v3 smoothPos = interpNormal * interpRadius;

                    if (grid.find(smoothPos, config.voxelSize * 0.1f) != nullptr) {
                        continue; 
                    }

                    Particle newPt;
                    newPt.surface = true;
                    newPt.currentPos = smoothPos;
                    // Note: originalPos, noisePos, tectonicPos remain null for these lightweight models!
                    
                    if (w1 > w2 && w1 > w3) {
                        newPt.plateID = p1.plateID;
                        newPt.originColor = p1.originColor;
                    } else if (w2 > w3) {
                        newPt.plateID = p2.plateID;
                        newPt.originColor = p2.originColor;
                    } else {
                        newPt.plateID = p3.plateID;
                        newPt.originColor = p3.originColor;
                    }

                    grid.set(newPt, newPt.currentPos, true, newPt.originColor.cast<float>(), config.voxelSize, true, 1, 2, false, 0.0f, 0.0f, 0.0f);
                    
                    config.interpolatedNodes.push_back(newPt);

                    counter++;
                }
            }
        }
        grid.optimize();
        std::cout << "Interpolated " << counter << " surface gaps." << std::endl;
    }

    void fillPlanet() {
        TIME_FUNCTION;
        if (config.interpolatedNodes.empty()) {
            std::cout << "Please run interpolate surface first." << std::endl;
            return;
        }
        std::cout << "Starting Volume Fill..." << std::endl;

        float safeRadius = config.radius - std::abs(config.valleyDepth) - (config.noiseStrength * 2.0f) - config.voxelSize;
        if (safeRadius <= 0) safeRadius = config.radius * 0.5f;

        int maxSteps = std::ceil(safeRadius / config.voxelSize);
        size_t fillCount = 0;

        for (int x = -maxSteps; x <= maxSteps; ++x) {
            for (int y = -maxSteps; y <= maxSteps; ++y) {
                for (int z = -maxSteps; z <= maxSteps; ++z) {
                    v3 pos = config.center + v3(x, y, z) * config.voxelSize;
                    float dist = (pos - config.center).norm();

                    if (dist <= safeRadius) {
                        if (grid.find(pos, config.voxelSize * 0.5f) == nullptr) {
                            Particle ip;
                            ip.surface = false;
                            ip.plateID = -1;
                            ip.currentPos = pos;
                            // Alternate memory-heavy positions stay natively cleanly null!

                            float depthRatio = dist / safeRadius;
                            Eigen::Vector3f coreColor(1.0f, 0.9f, 0.4f);
                            Eigen::Vector3f mantleColor(0.8f, 0.15f, 0.0f);
                            Eigen::Vector3f finalColor = mantleColor;

                            if (depthRatio < 0.5f) {
                                float blend = depthRatio * 2.0f;
                                finalColor = coreColor * (1.0f - blend) + mantleColor * blend;
                            }

                            ip.originColor = finalColor.cast<Eigen::half>();

                            grid.set(ip, pos, true, finalColor, config.voxelSize, true, 1, 3, false, 0.0f, 0.0f, 0.0f);
                            fillCount++;
                        }
                    }
                }
            }
        }

        grid.optimize();
        std::cout << "Volume Fill Complete. Inserted " << fillCount << " interior nodes directly into the grid." << std::endl;
    }

    void simulateImpacts() {
        TIME_FUNCTION;
        ///TODO: this needs to be run on a separate thread to allow visuals to continue.
        // apply data required for gravity to all nodes, including the ability to "clump" to prevent explosions or implosions of the planet upon reaching this step (perhaps include static core)
        // randomly spawn large clumps of nodes to simulate "asteroids" and create stuff like impact craters on the surface
        // they should be spawned going in random directions that are roughly towards the planet.
        //the gravity portion should be turned off when this is done.
    }

    void erosion() {
        ///TODO: simulate erosion by spawning many nodes all over the surface one at a time and then pulling them towards the lowest neighboring points. reducing height from source as it flows downhill and increasing at bottom.
        // this needs to be run on a separate thread to allow visuals to continue.
    }

    void storms() {
        ///TODO: generate weather patterns to determine stuff like rock vs dirt vs sand vs clay, etc. 
        //this will probably require putting a lot more into individual particle data to be able to simulate heat and such.
        // this needs to be run on a separate thread to allow visuals to continue.
    }

};

#endif