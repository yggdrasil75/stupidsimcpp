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
#include <iostream>

#include "../grid/grid3eigen.hpp"
#include "../timing_decorator.cpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"
#include <fstream>

// Generic binary helpers for trivial structures and fixed-size Eigen Vectors
template <typename T>
inline void writeBin(std::ofstream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
inline void readBin(std::ifstream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

using v3 = Eigen::Vector3f;
using v3half = Eigen::Matrix<Eigen::half, 3, 1>;
const float Φ = M_PI * (3.0f - std::sqrt(5.0f));

enum class PlateType {
    CONTINENTAL,
    OCEANIC,
    MIXED
};

struct AltPositions {
    v3 originalPos;
    v3 noisePos;
    v3 tectonicPos;
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
    v3 originColor;
    bool surface = false;

    //gravity factors:
    Eigen::Matrix<float, 3, 1> velocity = Eigen::Matrix<float, 3, 1>::Zero();
    Eigen::Matrix<float, 3, 1> acceleration = Eigen::Matrix<float, 3, 1>::Zero();
    Eigen::Matrix<float, 3, 1> forceAccumulator = Eigen::Matrix<float, 3, 1>::Zero();
    float density = 0.0f;
    float pressure = 0.0f;
    Eigen::Matrix<float, 3, 1> pressureForce = Eigen::Matrix<float, 3, 1>::Zero();
    float viscosity = 0.5f;
    Eigen::Matrix<float, 3, 1> viscosityForce = Eigen::Matrix<float, 3, 1>::Zero();
    float restitution = 5.0f;
    float mass = 1.0f;
    bool isStatic = false;
    float soundSpeed = 100.0f;
    float sandcontent = 0.0f;
    float siltcontent = 0.0f;
    float claycontent = 0.0f;
    float rockcontent = 1.0f;
    float metalcontent = 0.0f;

    float impactShock = 0.0f;
    float impactHeat = 0.0f;
    float impactDebris = 0.0f;

    NeighborData nearNeighbors[8];

    Particle() = default;

    Particle(const Particle& other) {
        noiseDisplacement = other.noiseDisplacement;
        plateID = other.plateID;
        currentPos = other.currentPos;
        plateDisplacement = other.plateDisplacement;
        originColor = other.originColor;
        surface = other.surface;
        
        velocity = other.velocity;
        acceleration = other.acceleration;
        forceAccumulator = other.forceAccumulator;
        density = other.density;
        pressure = other.pressure;
        pressureForce = other.pressureForce;
        viscosity = other.viscosity;
        viscosityForce = other.viscosityForce;
        restitution = other.restitution;
        mass = other.mass;
        isStatic = other.isStatic;
        soundSpeed = other.soundSpeed;
        sandcontent = other.sandcontent;
        siltcontent = other.siltcontent;
        claycontent = other.claycontent;
        rockcontent = other.rockcontent;
        metalcontent = other.metalcontent;
        
        impactShock = other.impactShock;
        impactHeat = other.impactHeat;
        impactDebris = other.impactDebris;
        
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
            
            velocity = other.velocity;
            acceleration = other.acceleration;
            forceAccumulator = other.forceAccumulator;
            density = other.density;
            pressure = other.pressure;
            pressureForce = other.pressureForce;
            viscosity = other.viscosity;
            viscosityForce = other.viscosityForce;
            restitution = other.restitution;
            mass = other.mass;
            isStatic = other.isStatic;
            soundSpeed = other.soundSpeed;
            sandcontent = other.sandcontent;
            siltcontent = other.siltcontent;
            claycontent = other.claycontent;
            rockcontent = other.rockcontent;
            metalcontent = other.metalcontent;
            
            impactShock = other.impactShock;
            impactHeat = other.impactHeat;
            impactDebris = other.impactDebris;
            
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

    void serialize(std::ofstream& out) const {
        writeBin(out, noiseDisplacement);
        writeBin(out, plateID);
        
        // Handle unique_ptr safely
        bool hasAltPos = (altPos != nullptr);
        writeBin(out, hasAltPos);
        if (hasAltPos) {
            writeBin(out, altPos->originalPos);
            writeBin(out, altPos->noisePos);
            writeBin(out, altPos->tectonicPos);
        }
        
        writeBin(out, currentPos);
        writeBin(out, plateDisplacement);
        writeBin(out, originColor);
        writeBin(out, surface);
        writeBin(out, velocity);
        writeBin(out, acceleration);
        writeBin(out, forceAccumulator);
        writeBin(out, density);
        writeBin(out, pressure);
        writeBin(out, pressureForce);
        writeBin(out, viscosity);
        writeBin(out, viscosityForce);
        writeBin(out, restitution);
        writeBin(out, mass);
        writeBin(out, isStatic);
        writeBin(out, soundSpeed);
        writeBin(out, sandcontent);
        writeBin(out, siltcontent);
        writeBin(out, claycontent);
        writeBin(out, rockcontent);
        writeBin(out, metalcontent);
        
        writeBin(out, impactShock);
        writeBin(out, impactHeat);
        writeBin(out, impactDebris);
        
        for(int i = 0; i < 8; ++i) {
            writeBin(out, nearNeighbors[i].index);
            writeBin(out, nearNeighbors[i].distance);
        }
    }

    static Particle deserialize(std::ifstream& in) {
        Particle p;
        readBin(in, p.noiseDisplacement);
        readBin(in, p.plateID);
        
        bool hasAltPos;
        readBin(in, hasAltPos);
        if (hasAltPos) {
            p.altPos = std::make_unique<AltPositions>();
            readBin(in, p.altPos->originalPos);
            readBin(in, p.altPos->noisePos);
            readBin(in, p.altPos->tectonicPos);
        }
        
        readBin(in, p.currentPos);
        readBin(in, p.plateDisplacement);
        readBin(in, p.originColor);
        readBin(in, p.surface);
        readBin(in, p.velocity);
        readBin(in, p.acceleration);
        readBin(in, p.forceAccumulator);
        readBin(in, p.density);
        readBin(in, p.pressure);
        readBin(in, p.pressureForce);
        readBin(in, p.viscosity);
        readBin(in, p.viscosityForce);
        readBin(in, p.restitution);
        readBin(in, p.mass);
        readBin(in, p.isStatic);
        readBin(in, p.soundSpeed);
        readBin(in, p.sandcontent);
        readBin(in, p.siltcontent);
        readBin(in, p.claycontent);
        readBin(in, p.rockcontent);
        readBin(in, p.metalcontent);
        
        readBin(in, p.impactShock);
        readBin(in, p.impactHeat);
        readBin(in, p.impactDebris);
        
        for(int i = 0; i < 8; ++i) {
            readBin(in, p.nearNeighbors[i].index);
            readBin(in, p.nearNeighbors[i].distance);
        }
        return p;
    }

    // friend ostream& operator<<(ostream& os) {

    //     return os;
    // }
};

struct planetConfig {
    Eigen::Vector3f center = Eigen::Vector3f(0,0,0);
    float radius = 1024.0f;
    Eigen::Vector3f color = Eigen::Vector3f(0, 1, 0);
    
    float voxelSize = 10.0f;
    int surfacePoints = 50000;
    
    int currentStep = 0;
    
    float displacementStrength = 200.0f;
    std::vector<v3> surfaceNodes;
    std::vector<v3> interpolatedNodes;
    float noiseStrength = 10.0f;
    int numPlates = 15;
    int smoothingPasses = 3;
    float mountHeight = 250.0f;
    float valleyDepth = -150.0f;
    float transformRough = 80.0f;
    int stressPasses = 5;
    float maxElevationRatio = 0.25f;

    float gridSizeCube = 65536; //absolute max size for all nodes
    float gridSizeCubeMin = 4096; //max size, if something leaves this, then it probably needs to be purged before it leaves the grid and becomes lost
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
    v3 plateEulerPolep;
    // Particle plateEulerPole;
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

struct ImpactEvent {
    v3 position;
    float radius;
};

class planetsim {
public:
    planetConfig config;
    Octree<Particle, int16_t, "output/fibSphere"> grid;
    std::vector<PlateConfig> plates;
    std::mt19937 rng = std::mt19937(42);
    bool starAdded = false;
    bool coreFilled = false;

    std::vector<ImpactEvent> impactHistory;
    int dynMoonId = 10;

    planetsim() {
        config = planetConfig();
        grid = Octree<Particle, int16_t, "output/fibSphere">(v3(-config.gridSizeCubeMin,-config.gridSizeCubeMin,-config.gridSizeCubeMin),v3(config.gridSizeCubeMin,config.gridSizeCubeMin,config.gridSizeCubeMin), 4, 64);
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
        grid.load("output/fibSphere.yggs");
        grid.clear();
        config.surfaceNodes.clear();
        config.surfaceNodes.resize(config.surfacePoints);
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
            pt.altPos->originalPos = pos;
            pt.altPos->noisePos = pos;
            pt.altPos->tectonicPos = pos;
            
            pt.currentPos = pos;
            pt.originColor = config.color;
            pt.noiseDisplacement = 0.0f;
            pt.surface = true;
            config.surfaceNodes[i] = pt.currentPos;
            grid.set(pt, pt.currentPos, true, pt.originColor, config.voxelSize, true, 1, false, 0.0f, 0.0f, 0.0f);
        }
        config.currentStep = 1;
        std::cout << "Step 1 done. base sphere generated" << std::endl;
        grid.waitForIdle();
        grid.save("output/fibSphere.yggs");
    }

    inline void _applyNoise(std::function<float(const Eigen::Vector3f&)> noiseFunc) {
        std::vector<v3> newPos(config.surfaceNodes.size());
        // grid.waitForIdle();
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            v3 pos = config.surfaceNodes[i];
        // for (auto& pos : config.surfaceNodes) {
            
            auto node = grid.find(pos, config.voxelSize * 0.5f);
            // if (!node) continue;
            if (!node) {
                std::cout << "something broke as early as applynoise!" << std::endl;
            }
            Particle p = node->data;
            Eigen::Vector3f oldPos = p.currentPos;
            float displacementValue = noiseFunc(p.altPos->originalPos);
            p.noiseDisplacement = displacementValue;
            Eigen::Vector3f normal = p.altPos->originalPos.normalized();
            p.altPos->noisePos = (p.altPos->originalPos + (normal * displacementValue * config.noiseStrength));
            p.currentPos = p.altPos->noisePos;
            newPos[i] = p.currentPos;
            grid.move(oldPos, p.currentPos);
            grid.update(p.currentPos, p);
            // grid.queuedupdate(oldPos, p.currentPos, p);
            // grid.queuedupdate(p.currentPos, p);
        }
        config.surfaceNodes = newPos;
        std::cout << "applied noise" << std::endl;
    }

    void assignSeeds() {
        grid.waitForIdle();
        std::cout << "assigning seeds" << std::endl;
        int asdf = 0;
        plates.clear();
        plates.resize(config.numPlates);
        float sphereSurfaceArea = 4.0f * M_PI * config.radius * config.radius;
        float averageAreaPerPlate = sphereSurfaceArea / config.numPlates;
        float minDistance = std::sqrt(averageAreaPerPlate) * 0.4f;
        std::vector<int> selectedSeedIndices;
        std::uniform_int_distribution<int> distNode(0, config.surfacePoints - 1);
        // std::cout << asdf++ << std::endl;
        for (int i = 0; i < config.numPlates; ++i) {
            int attempts = 10;
            bool foundValidSeed = false;
            // std::cout << asdf++ << std::endl;
            int seedid = distNode(rng);
            plates[i].plateId = i;
            // std::cout << asdf++ << std::endl;

            while (!foundValidSeed && attempts > 0) {
                int seedIndex = distNode(rng);
                
                bool tooClose = false;
                for (int selectedIndex : selectedSeedIndices) {
                    // std::cout << asdf++ << std::endl;
                    auto existingNode = grid.find(config.surfaceNodes[selectedIndex], config.voxelSize * 0.5f);
                    auto candidateNode = grid.find(config.surfaceNodes[seedIndex], config.voxelSize * 0.5f);
                    // std::cout << asdf++ << std::endl;
                    if (!existingNode || !candidateNode) {
                        std::cout << "no nodes" << std::endl;
                        continue;
                    }

                    const auto& existingSeed = existingNode->data;
                    const auto& candidateSeed = candidateNode->data;
                    // std::cout << asdf++ << std::endl;

                    float dot = existingSeed.altPos->originalPos.normalized().dot(candidateSeed.altPos->originalPos.normalized());
                    float angle = std::acos(std::clamp(dot, -1.0f, 1.0f));
                    float distanceOnSphere = angle * config.radius;
                    
                    if (distanceOnSphere < minDistance) {
                        tooClose = true;
                        std::cout << "too close" << std::endl;
                        break;
                    }
                }
                
                if (!tooClose || selectedSeedIndices.empty()) {
                    selectedSeedIndices.push_back(seedIndex);
                    plates[i].plateId = i;
                    
                    auto sNode = grid.find(config.surfaceNodes[seedIndex], config.voxelSize * 0.5f);
                    if (sNode) {
                        Particle p = sNode->data;
                        p.plateID = i; 
                        grid.updateData(config.surfaceNodes[seedIndex], p);
                        plates[i].plateEulerPolep = sNode->position;
                        // plates[i].plateEulerPole = p;
                    }

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

                auto sNode = grid.find(config.surfaceNodes[seedIndex], config.voxelSize * 0.5f);
                if (sNode) {
                    Particle p = sNode->data;
                    p.plateID = i;
                    grid.updateData(config.surfaceNodes[seedIndex], p);
                    plates[i].plateEulerPolep = sNode->position;
                    // plates[i].plateEulerPole = p;
                }

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
            }
        }
        std::cout << "finished assigning seeds" << std::endl;
    }

    void buildAdjacencyList() {
        TIME_FUNCTION;
        grid.waitForIdle();
        std::cout << "building an adjacency list" << std::endl;
        int numNodes = config.surfaceNodes.size();
        std::vector<v3> normPos(numNodes);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; i++) {
            auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (node) {
                normPos[i] = node->data.altPos->originalPos.normalized();
            } else {
                normPos[i] = config.surfaceNodes[i].normalized();
            }
        }
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (!node) continue;
            Particle in = node->data;
            
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
            grid.updateData(config.surfaceNodes[i], in);
            // grid.queuedupdate(config.surfaceNodes[i], in);
        }
        grid.waitForIdle();
    }

    void growPlatesRandom() {
        TIME_FUNCTION;
        grid.waitForIdle();
        std::cout << "growing randomly" << std::endl;
        int unassignedCount = 0;
        std::vector<int> plateWeights(config.numPlates, 1);
        std::vector<std::vector<int>> frontiers(config.numPlates);
        
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (!node) {
                std::cout << "gpr couldnt find" << std::endl;
                continue;
            }
            int pID = node->data.plateID;
            if (pID == -1) {
                unassignedCount++;
            } else {
                std::cout << "found seed for " << pID << std::endl;
                plates[pID].assignedNodes.push_back(i);
                for (int n = 0; n < 8; n++) {
                    int nIdx = node->data.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    auto nNode = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                    if (nNode && nNode->data.plateID == -1) {
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
            
            // std::cout << "have " << unassignedCount << " remaining nodes" << std::endl;
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

                auto candNode = grid.find(config.surfaceNodes[candIdx], config.voxelSize * 0.5f);
                if (candNode && candNode->data.plateID == -1) {
                    Particle p = candNode->data;
                    p.plateID = selPlate;
                    grid.updateData(config.surfaceNodes[candIdx], p);
                    
                    plates[selPlate].assignedNodes.push_back(candIdx);
                    unassignedCount--;
                    successfulGrowth = true;

                    for (int n = 0; n < 8; n++) {
                        int nIdx = p.nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        auto nNode = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                        if (nNode && nNode->data.plateID == -1) {
                            frontiers[selPlate].push_back(nIdx);
                        }
                    }
                }
            } else {
                plateWeights[selPlate] = 0;
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
        grid.waitForIdle();
        std::cout << "growing using cellular automata" << std::endl;
        int unassignedCount = 0;
        for (const auto& pos : config.surfaceNodes) {
            auto node = grid.find(pos, config.voxelSize * 0.5f);
            if (node && node->data.plateID == -1) unassignedCount++;
        }

        while (unassignedCount > 0) {
            std::vector<int> nextState(config.surfaceNodes.size(), -1);
            int assignedThisRound = 0;
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                if (!node) continue;
                
                if (node->data.plateID != -1) {
                    nextState[i] = node->data.plateID;
                } else {
                    std::unordered_map<int, int> counts;
                    int bestPlate = -1;
                    int maxCount = 0;
                    
                    for (int n = 0; n < 8; n++) {
                        int nIdx = node->data.nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        auto nNode = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                        if (!nNode) continue;
                        
                        int pID = nNode->data.plateID;
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
                auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                if (node && node->data.plateID == -1 && nextState[i] != -1) {
                    Particle p = node->data;
                    p.plateID = nextState[i];
                    grid.updateData(config.surfaceNodes[i], p);
                    plates[nextState[i]].assignedNodes.push_back(i);
                    unassignedCount--;
                }
            }
            
            if (assignedThisRound == 0 && unassignedCount > 0) {
                for (int i = 0; i < config.surfaceNodes.size(); i++) {
                    auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                    if (node && node->data.plateID == -1) {
                        int closestPlate = 0;
                        float minDist = std::numeric_limits<float>::max();
                        for (int p = 0; p < config.numPlates; p++) {
                            auto asdfsdfsdfsdfs = grid.find(plates[p].plateEulerPolep);
                            float d = (node->data.altPos->originalPos - asdfsdfsdfsdfs->data.altPos->originalPos).norm();
                            if (d < minDist) {
                                minDist = d;
                                closestPlate = p;
                            }
                        }
                        Particle pt = node->data;
                        pt.plateID = closestPlate;
                        grid.updateData(config.surfaceNodes[i], pt);
                        plates[closestPlate].assignedNodes.push_back(i);
                        unassignedCount--;
                    }
                }
            }
        }
    }

    void fixBoundaries() {
        TIME_FUNCTION;
        grid.waitForIdle();
        std::cout << "fixing boundaries" << std::endl;
        for (int pass = 0; pass < config.smoothingPasses; pass++) {
            std::vector<int> nextPlateID(config.surfaceNodes.size(), -1);
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                if (!node) continue;
                
                std::unordered_map<int, int> counts;
                counts[node->data.plateID]++;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = node->data.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    auto nNode = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                    if (nNode) counts[nNode->data.plateID]++;
                }
                
                int bestPlate = node->data.plateID;
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
                auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                if (node && nextPlateID[i] != -1) {
                    Particle p = node->data;
                    p.plateID = nextPlateID[i];
                    grid.updateData(config.surfaceNodes[i], p);
                }
            }
        }
        
        for (auto& plate : plates) {
            plate.assignedNodes.clear();
        }
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (node && node->data.plateID != -1) {
                plates[node->data.plateID].assignedNodes.push_back(i);
            }
        }
    }

    void extraplateste() {
        TIME_FUNCTION;
        grid.waitForIdle();
        std::cout << "plate setting stuff" << std::endl;
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        std::vector<std::pair<int, float>> plateStats;
        for (int i = 0; i < config.numPlates; i++) {
            plateStats.emplace_back(std::make_pair(i, 0.0f));
        }

        for (int i = 0; i < config.numPlates; i++) {
            float sumElevation = 0.0f;
            Eigen::Vector3f centroid(0,0,0);
            
            for (int nIdx : plates[i].assignedNodes) {
                auto node = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                if (!node) continue;
                sumElevation += node->data.currentPos.norm();
                centroid += node->data.altPos->originalPos;
            }
            
            if (!plates[i].assignedNodes.empty()) {
                plateStats[i].second = sumElevation / plates[i].assignedNodes.size();
                centroid /= plates[i].assignedNodes.size();

                float maxSpread = 0.0f;
                for (int nIdx : plates[i].assignedNodes) {
                    auto node = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                    if (!node) continue;
                    float d = (node->data.altPos->originalPos - centroid).norm();
                    if (d > maxSpread) maxSpread = d;
                }

                auto asdfasdflkj = grid.find(plates[i].plateEulerPolep);
                if (asdfasdflkj) {
                    Particle pasdf = asdfasdflkj->data;
                    float distToCentroid = (pasdf.altPos->originalPos - centroid).norm();
                    
                    if (distToCentroid > maxSpread * 0.6f) {
                        int bestNodeIdx = plates[i].assignedNodes[0];
                        float minDistToCentroid = std::numeric_limits<float>::max();
                        
                        for (int nIdx : plates[i].assignedNodes) {
                            auto node = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                            if (!node) continue;
                            float d = (node->data.altPos->originalPos - centroid).norm();
                            if (d < minDistToCentroid) {
                                minDistToCentroid = d;
                                bestNodeIdx = nIdx;
                            }
                        }
                        auto bestNode = grid.find(config.surfaceNodes[bestNodeIdx], config.voxelSize * 0.5f);
                        if (bestNode) {
                            plates[i].plateEulerPolep = bestNode->position;
                        }
                        // grid.update(plates[i].plateEulerPolep, pasdf);
                    }
                }
            } else {
                plateStats[i].second = config.radius;
            }
            
            Eigen::Vector3f randomDir(distFloat(rng) - 0.5f, distFloat(rng) - 0.5f, distFloat(rng) - 0.5f);
            randomDir.normalize();
            
            auto pasd = grid.find(plates[i].plateEulerPolep);
            if (pasd) {
                Particle pasddata = pasd->data;
                Eigen::Vector3f poleDir = pasddata.altPos->originalPos;
                poleDir.normalize();
                plates[i].direction = (randomDir - poleDir * randomDir.dot(poleDir));
                plates[i].direction.normalize();
            }
            
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
        grid.waitForIdle();
        std::cout << "applying boundary stresses" << std::endl;
        int numNodes = config.surfaceNodes.size();
        std::vector<float> nodeStress(numNodes, 0.0f);
        std::vector<float> nodeNoise(numNodes, 0.0f);
        
        std::vector<Eigen::Vector3f> ω(config.numPlates);
        for (int i = 0; i < config.numPlates; i++) {
            auto asdfasdflkj = grid.find(plates[i].plateEulerPolep);
            ω[i] = asdfasdflkj->data.altPos->originalPos.normalized().cross(plates[i].direction) * plates[i].angularVelocity;
        }

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int pass = 0; pass < config.stressPasses; pass++) {
            std::vector<float> newStress = nodeStress;
            std::vector<float> newNoise = nodeNoise;
            
            for (int i = 0; i < numNodes; i++) {
                auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
                if (!node) continue;
                Particle p = node->data;

                int myPlate = p.plateID;
                if (myPlate == -1) continue;
                
                Eigen::Vector3f myPos = p.altPos->originalPos.normalized();
                Eigen::Vector3f myVel = ω[myPlate].cross(myPos);
                
                float localStress = 0.0f;
                float localNoise = 0.0f;
                int boundaryCount = 0;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = p.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    
                    auto nNode = grid.find(config.surfaceNodes[nIdx], config.voxelSize * 0.5f);
                    if (!nNode) continue;

                    int nPlate = nNode->data.plateID;
                    if (nPlate != -1 && myPlate != nPlate) {
                        boundaryCount++;
                        Eigen::Vector3f nPos = nNode->data.altPos->originalPos.normalized();
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
                        int nIdx = p.nearNeighbors[n].index;
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
            auto node = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (!node) continue;
            Particle p = node->data;
            p.plateDisplacement = nodeStress[i];
            
            float noiseVal = dist(rng) * nodeNoise[i];
            
            Eigen::Vector3f normal = p.altPos->originalPos.normalized();
            p.altPos->tectonicPos = (p.altPos->noisePos + (normal * (p.plateDisplacement + noiseVal)));
            
            grid.updateData(config.surfaceNodes[i], p);
            // grid.queuedupdate(config.surfaceNodes[i], p);
        }
    }

    void finalizeApplyResults() {
        TIME_FUNCTION;
        std::cout << "finalizing results" << std::endl;
        ///TODO: fix this not being used.
        float maxAllowedDisp = config.radius * config.maxElevationRatio;

        grid.waitForIdle();
        std::vector<v3> newPos(config.surfaceNodes.size());
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            v3 pos = config.surfaceNodes[i];
        // for (auto& pos : config.surfaceNodes) {
            auto node = grid.find(pos, config.voxelSize * 0.5f);
            if (!node || !node->isActive()) {
                newPos[i] = pos;
                continue;
            }
            Particle p = node->data;
            Eigen::Vector3f oldPos = p.currentPos;
            p.currentPos = p.altPos->tectonicPos;
            newPos[i] = p.currentPos;
            grid.updateData(oldPos, p);
            grid.move(oldPos, p.currentPos);
            // grid.queuedupdate(oldPos, p.currentPos, p);
            // grid.waitForIdle();
            // grid.update(p.currentPos, p);
        }
        config.surfaceNodes = newPos;
        std::cout << "Finalize apply results completed." << std::endl;
    }

    void addStar() {
        if (starAdded) return;
        TIME_FUNCTION;

        const float realEarthRadiusKm = 6371.0f;
        const float realSunRadiusKm = 696340.0f;
        const float realAuKm = 149597870.0f;
        float simScale = config.radius / realEarthRadiusKm;

        float starRadius = realSunRadiusKm * simScale;
        float orbitDistance = realAuKm * simScale;

        std::cout << "--- STAR GENERATION ---" << std::endl;
        std::cout << "Sim Scale: " << simScale << " units/km" << std::endl;
        std::cout << "Star Radius: " << starRadius << " units" << std::endl;
        std::cout << "Orbit Distance: " << orbitDistance << " units" << std::endl;
        std::cout << "Registering Star directly to Octree Skybox..." << std::endl;

        v3 starDir = v3(1.0f, 0.0f, 0.0f); 
        float angularRadius = std::asin(starRadius / orbitDistance);

        grid.addSkyBody(0, starDir, angularRadius, 255, 242, 204, 255);
        
        config.currentStep = 1;
        std::cout << "Star generation complete via Skybox." << std::endl;
        starAdded = true;
    }

    void addMoon() {
        TIME_FUNCTION;
        
        const float realMoonRadiusKm = 1737.0f;
        const float realMoonDistKm = 384400.0f;
        float simScale = config.radius / 6371.0f;
        
        float moonRadius = realMoonRadiusKm * simScale;
        float orbitDistance = realMoonDistKm * simScale;
        
        std::cout << "--- MOON GENERATION ---" << std::endl;
        std::cout << "Moon Radius: " << moonRadius << " units" << std::endl;
        std::cout << "Orbit Distance: " << orbitDistance << " units" << std::endl;
        std::cout << "Registering Moon directly to Octree Skybox..." << std::endl;

        v3 moonDir = v3(-1.0f, 0.5f, 0.0f).normalized();
        float angularRadius = std::asin(moonRadius / orbitDistance);
        
        grid.addSkyBody(1, moonDir, angularRadius, 200, 200, 200, 100);
        
        std::cout << "Moon added successfully." << std::endl;
    }
    
    void stretchPlanet() {
        ///TODO: simulate millenia of gravitational stretching by nearby celestial bodies by squeezing the planet slightly at its poles
    }

    void interpolateSurface() {
        TIME_FUNCTION;
        
        config.interpolatedNodes.clear();
        grid.waitForIdle();

        std::set<std::tuple<int, int, int>> uniqueTriangles;

        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto pt1 = grid.find(config.surfaceNodes[i], config.voxelSize * 0.5f);
            if (!pt1) {
                std::cout << "something broke interpolate find" << std::endl;
                continue;
            }
            Particle p1 = pt1->data;
            if (!p1.altPos) {
                std::cout << "missing alt positions" << std::endl;
            }

            for (int n1 = 0; n1 < 8; n1++) {
                int j = p1.nearNeighbors[n1].index;
                if (j == -1) break;
                if (j >= i) continue; 
                
                auto pt2 = grid.find(config.surfaceNodes[j], config.voxelSize * 0.5f);
                if (!pt2) continue;
                Particle p2 = pt2->data;

                for (int n2 = 0; n2 < 8; n2++) {
                    int k = p2.nearNeighbors[n2].index;
                    if (k == -1) break;
                    if (k <= j) continue;
                    
                    auto pt3 = grid.find(config.surfaceNodes[k], config.voxelSize * 0.5f);
                    if (!pt3) continue;

                    bool isNeighbor = false;
                    for (int n3 = 0; n3 < 8; n3++) {
                        int nIdx = pt3->data.nearNeighbors[n3].index;
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

            auto pt1 = grid.find(config.surfaceNodes[idx1], config.voxelSize * 0.5f);
            // if (!pt1) continue;
            Particle p1 = pt1->data;
            auto pt2 = grid.find(config.surfaceNodes[idx2], config.voxelSize * 0.5f);
            // if (!pt2) continue;
            Particle p2 = pt2->data;
            auto pt3 = grid.find(config.surfaceNodes[idx3], config.voxelSize * 0.5f);
            // if (!pt3) continue;
            Particle p3 = pt3->data;

            if (!p1.altPos) {
                std::cout << "broken p1" << std::endl;
                std::cout << p1.altPos << std::endl;
                continue;
            }
            if (!p2.altPos) {
                std::cout << "broken p2" << std::endl;
                // std::cout << "data: " << p2 << std::endl;
                std::cout << "pos: " << p2.altPos << std::endl;
                continue;
            }
            if (!p3.altPos) {
                std::cout << "broken p3" << std::endl;
                std::cout << p3.altPos << std::endl;
                continue;
            }

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

                    v3 interpNormal = (p1.altPos->originalPos * w1 + 
                                       p2.altPos->originalPos * w2 + 
                                       p3.altPos->originalPos * w3);
                    interpNormal.normalize(); 
                    
                    float r1 = p1.currentPos.norm();
                    float r2 = p2.currentPos.norm();
                    float r3 = p3.currentPos.norm();
                    float interpRadius = (r1 * w1) + (r2 * w2) + (r3 * w3);

                    v3 smoothPos = interpNormal * interpRadius;

                    Particle newPt;
                    newPt.surface = true;
                    newPt.currentPos = smoothPos;
                    
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

                    grid.queuedset(newPt, newPt.currentPos, true, newPt.originColor, config.voxelSize, true, 1, false, 0.0f, 0.0f, 0.0f);
                    
                    config.interpolatedNodes.push_back(newPt.currentPos);

                    counter++;
                }
            }
        }
        grid.waitForIdle();
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

        const int LON_RES = 720;
        const int LAT_RES = 360;
        std::vector<float> surfaceMap(LON_RES * LAT_RES, safeRadius);

        for (const auto& pos : config.surfaceNodes) {
            auto node = grid.find(pos, config.voxelSize * 0.5f);
            if (!node) continue;
            v3 d = node->data.currentPos - config.center;
            float r = d.norm();
            if (r < 0.1f) continue;
            
            float phi = std::asin(std::clamp(d.y() / r, -1.0f, 1.0f));
            float theta = std::atan2(d.z(), d.x());
            
            int u = std::clamp(static_cast<int>((theta + M_PI) / (2.0f * M_PI) * LON_RES), 0, LON_RES - 1);
            int v = std::clamp(static_cast<int>((phi + M_PI / 2.0f) / M_PI * LAT_RES), 0, LAT_RES - 1);
            
            float angular_radius = (config.voxelSize * 1.5f) / r;
            int u_radius = std::ceil(angular_radius / (2.0f * M_PI / LON_RES));
            int v_radius = std::ceil(angular_radius / (M_PI / LAT_RES));
            
            for (int dv = -v_radius; dv <= v_radius; ++dv) {
                for (int du = -u_radius; du <= u_radius; ++du) {
                    if (du*du + dv*dv <= u_radius*u_radius) {
                        int nu = ((u + du) % LON_RES + LON_RES) % LON_RES;
                        int nv = std::clamp(v + dv, 0, LAT_RES - 1);
                        int idx = nv * LON_RES + nu;
                        if (r > surfaceMap[idx]) {
                            surfaceMap[idx] = r;
                        }
                    }
                }
            }
        }

        for (const auto& pos : config.interpolatedNodes) {
            auto node = grid.find(pos, config.voxelSize * 0.5f);
            if (!node) continue;
            v3 d = node->data.currentPos - config.center;
            float r = d.norm();
            if (r < 0.1f) continue;
            
            float phi = std::asin(std::clamp(d.y() / r, -1.0f, 1.0f));
            float theta = std::atan2(d.z(), d.x());
            
            int u = std::clamp(static_cast<int>((theta + M_PI) / (2.0f * M_PI) * LON_RES), 0, LON_RES - 1);
            int v = std::clamp(static_cast<int>((phi + M_PI / 2.0f) / M_PI * LAT_RES), 0, LAT_RES - 1);
            
            float angular_radius = (config.voxelSize * 1.5f) / r;
            int u_radius = std::ceil(angular_radius / (2.0f * M_PI / LON_RES));
            int v_radius = std::ceil(angular_radius / (M_PI / LAT_RES));
            
            for (int dv = -v_radius; dv <= v_radius; ++dv) {
                for (int du = -u_radius; du <= u_radius; ++du) {
                    if (du*du + dv*dv <= u_radius*u_radius) {
                        int nu = ((u + du) % LON_RES + LON_RES) % LON_RES;
                        int nv = std::clamp(v + dv, 0, LAT_RES - 1);
                        int idx = nv * LON_RES + nu;
                        if (r > surfaceMap[idx]) {
                            surfaceMap[idx] = r;
                        }
                    }
                }
            }
        }

        float maxPlanetRadius = config.radius + config.mountHeight + config.noiseStrength * 2.0f + config.voxelSize;

        size_t fillCount = 0;

        float d_hcp = config.voxelSize;
        float z_step = d_hcp * std::sqrt(2.0f / 3.0f);
        float y_step = d_hcp * std::sqrt(3.0f) / 2.0f;
        float x_step = d_hcp;

        int maxK = std::ceil(maxPlanetRadius / z_step);
        int maxJ = std::ceil(maxPlanetRadius / y_step);
        int maxI = std::ceil(maxPlanetRadius / x_step);

        #pragma omp parallel for collapse(3)
        for (int k = -maxK; k <= maxK; ++k) {
            for (int j = -maxJ; j <= maxJ; ++j) {
                for (int i = -maxI; i <= maxI; ++i) {
                    float z = k * z_step;
                    int k_mod2 = std::abs(k % 2); 
                    int j_mod2 = std::abs(j % 2);
                    float y = j * y_step + k_mod2 * d_hcp * std::sqrt(3.0f) / 6.0f;
                    float x = i * x_step + j_mod2 * (d_hcp / 2.0f) + k_mod2 * (d_hcp / 2.0f);

                    v3 pos = config.center + v3(x, y, z);
                    v3 dir = pos - config.center;
                    float dist = dir.norm();

                    if (dist > maxPlanetRadius) continue;

                    bool inside = (dist <= safeRadius);
                    float localMaxRadius = config.radius;
                    
                    if (!inside && dist > 0.1f) {
                        float phi = std::asin(std::clamp(dir.y() / dist, -1.0f, 1.0f));
                        float theta = std::atan2(dir.z(), dir.x());
                        int u = std::clamp(static_cast<int>((theta + M_PI) / (2.0f * M_PI) * LON_RES), 0, LON_RES - 1);
                        int v = std::clamp(static_cast<int>((phi + M_PI / 2.0f) / M_PI * LAT_RES), 0, LAT_RES - 1);
                        
                        localMaxRadius = surfaceMap[v * LON_RES + u];
                        
                        if (dist <= localMaxRadius - config.voxelSize * 0.7f) {
                            inside = true;
                        }
                    }

                    if (inside) {
                        Particle ip;
                        ip.surface = false;
                        ip.plateID = -1;
                        ip.currentPos = pos;

                        float depthRatio = dist / localMaxRadius;
                        Eigen::Vector3f coreColor(1.0f, 0.9f, 0.4f);
                        Eigen::Vector3f mantleColor(0.8f, 0.15f, 0.0f);
                        Eigen::Vector3f finalColor = mantleColor;

                        if (depthRatio < 0.5f) {
                            float blend = depthRatio * 2.0f;
                            finalColor = coreColor * (1.0f - blend) + mantleColor * blend;
                        }

                        ip.originColor = finalColor;
                        ip.mass = 100.0f;

                        grid.queuedset(ip, pos, true, finalColor, config.voxelSize, true, 1, false, 0.0f, 0.0f, 0.0f);
                        #pragma omp atomic
                        fillCount++;
                    }
                }
            }
        }

        grid.waitForIdle();

        coreFilled = true;
        std::cout << "Volume Fill Complete. Inserted " << fillCount << " interior nodes directly into the grid." << std::endl;
    }

    void applyDirectImpact(v3 targetPos, v3 velocity, float astRadius, float incidenceDot) {
        float depth = astRadius * (1.0f + incidenceDot * 2.0f);
        v3 impactCenter = targetPos + velocity * (depth * 0.5f);
        
        float affectRadius = astRadius * 1.5f;
        auto affected = grid.findInRadius(impactCenter, affectRadius);
        
        std::vector<v3> toRemove;
        std::vector<std::pair<v3, Particle>> toUpdate;
        
        for (auto& n : affected) {
            if (!n->isActive()) continue;
            
            float dist = (n->position - impactCenter).norm();
            if (dist < astRadius) {
                toRemove.push_back(n->position);
            } else {
                Particle p = n->data;
                float falloff = 1.0f - ((dist - astRadius) / (affectRadius - astRadius));
                
                p.impactHeat = std::clamp(p.impactHeat + falloff, 0.0f, 1.0f);
                p.impactShock = std::clamp(p.impactShock + falloff, 0.0f, 1.0f);
                p.metalcontent = std::clamp(p.metalcontent + falloff * 0.5f, 0.0f, 1.0f);
                
                v3 origCol = p.originColor;
                v3 burnCol(0.1f, 0.05f, 0.05f); // Reddish dark scorch
                v3 newCol = origCol * (1.0f - falloff) + burnCol * falloff;
                p.originColor = newCol;
                
                // Push nodes out slightly to form a crater rim
                v3 pushDir = (n->position - impactCenter).normalized();
                p.currentPos = n->position + pushDir * (falloff * config.voxelSize * 2.0f);
                
                toUpdate.push_back({n->position, p});
            }
        }
        
        for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
        for (auto& up : toUpdate) {
            grid.move(up.first, up.second.currentPos);
            grid.updateData(up.second.currentPos, up.second);
            grid.setColor(up.second.currentPos, up.second.originColor);
        }
        
        impactHistory.push_back({targetPos, affectRadius});
        std::cout << "  - Direct Impact processed. Radius: " << astRadius << std::endl;
    }

    void applyScrapeImpact(v3 targetPos, v3 velocity, float astRadius) {
        float trenchLength = astRadius * (4.0f + 4.0f * (static_cast<float>(rng()) / RAND_MAX));
        
        int steps = std::max(1, static_cast<int>(trenchLength / (astRadius * 0.5f)));
        for (int i = 0; i <= steps; i++) {
            float t = static_cast<float>(i) / steps;
            v3 currentPos = targetPos + velocity * (trenchLength * t);
            
            float currentRadius = astRadius * (1.0f - t * 0.5f);
            float affectRadius = currentRadius * 1.5f;
            
            auto affected = grid.findInRadius(currentPos, affectRadius);
            
            std::vector<v3> toRemove;
            std::vector<std::pair<v3, Particle>> toUpdate;
            
            for (auto& n : affected) {
                if (!n->isActive()) continue;
                float dist = (n->position - currentPos).norm();
                if (dist < currentRadius) {
                    toRemove.push_back(n->position);
                } else {
                    Particle p = n->data;
                    float falloff = 1.0f - ((dist - currentRadius) / (affectRadius - currentRadius));
                    
                    p.impactHeat = std::clamp(p.impactHeat + falloff * 0.5f, 0.0f, 1.0f);
                    p.impactShock = std::clamp(p.impactShock + falloff * 0.5f, 0.0f, 1.0f);
                    
                    v3 origCol = p.originColor;
                    v3 burnCol(0.2f, 0.15f, 0.1f);
                    p.originColor = (origCol * (1.0f - falloff) + burnCol * falloff);
                    
                    toUpdate.push_back({n->position, p});
                }
            }
            
            for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
            for (auto& up : toUpdate) {
                grid.updateData(up.first, up.second);
                grid.setColor(up.first, up.second.originColor);
            }
        }
        
        impactHistory.push_back({targetPos, astRadius * 3.0f});
        std::cout << "  - Scrape Impact processed. Length: " << trenchLength << std::endl;
    }

    void applyAirburstImpact(v3 targetPos, v3 normal, float astRadius) {
        int numFragments = 5 + (rng() % 10);
        
        for (int i = 0; i < numFragments; i++) {
            v3 offset = v3(
                (static_cast<float>(rng()) / RAND_MAX) - 0.5f,
                (static_cast<float>(rng()) / RAND_MAX) - 0.5f,
                (static_cast<float>(rng()) / RAND_MAX) - 0.5f
            ).normalized() * (astRadius * 1.5f * (static_cast<float>(rng()) / RAND_MAX));
            
            v3 fragPos = targetPos + offset;
            float fragRadius = astRadius * (0.1f + 0.3f * (static_cast<float>(rng()) / RAND_MAX));
            
            auto affected = grid.findInRadius(fragPos, fragRadius * 1.5f);
            
            std::vector<v3> toRemove;
            std::vector<std::pair<v3, Particle>> toUpdate;
            
            for (auto& n : affected) {
                if (!n->isActive()) continue;
                float dist = (n->position - fragPos).norm();
                if (dist < fragRadius) {
                    toRemove.push_back(n->position);
                } else {
                    Particle p = n->data;
                    float falloff = 1.0f - ((dist - fragRadius) / (fragRadius * 0.5f));
                    
                    p.impactDebris = std::clamp(p.impactDebris + falloff, 0.0f, 1.0f);
                    
                    v3 origCol = p.originColor;
                    v3 dustCol(0.6f, 0.5f, 0.4f);
                    p.originColor = (origCol * (1.0f - falloff) + dustCol * falloff);
                    
                    toUpdate.push_back({n->position, p});
                }
            }
            
            for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
            for (auto& up : toUpdate) {
                grid.updateData(up.first, up.second);
                grid.setColor(up.first, up.second.originColor);
            }
        }
        
        impactHistory.push_back({targetPos, astRadius * 2.0f});
        std::cout << "  - Airburst Impact processed. Fragments: " << numFragments << std::endl;
    }

    void generateAsteroidImpacts(int count) {
        TIME_FUNCTION;
        if (!coreFilled) {
            std::cout << "ERROR: Core must be filled before spawning asteroids!" << std::endl;
            return;
        }

        std::uniform_real_distribution<float> rand(0.0f, 1.0f);
        std::cout << "Generating " << count << " Asteroid Impacts..." << std::endl;

        for (int i = 0; i < count; i++) {
            v3 up = v3(rand(rng) - 0.5f, rand(rng) - 0.5f, rand(rng) - 0.5f).normalized();
            v3 targetPos = config.center + up * config.radius;
            
            auto possibleNodes = grid.findInRadius(targetPos, config.radius * 0.5f);
            float maxDist = 0.0f;
            v3 actualPos = targetPos;
            bool found = false;
            
            for(auto& n : possibleNodes) {
                if(!n->isActive()) continue;
                float d = (n->position - config.center).norm();
                if(d > maxDist) {
                    maxDist = d;
                    actualPos = n->position;
                    found = true;
                }
            }
            if (found) {
                targetPos = actualPos;
                up = (targetPos - config.center).normalized();
            }

            v3 randomTangent = up.cross(v3(rand(rng) - 0.5f, rand(rng) - 0.5f, rand(rng) - 0.5f)).normalized();
            float incidenceDot = rand(rng); 
            v3 velocity = (-up * incidenceDot + randomTangent * (1.0f - incidenceDot)).normalized();

            float sizeClass = rand(rng);
            float astRadius = config.radius * (0.02f + sizeClass * 0.08f); 

            if (incidenceDot < 0.2f) {
                applyScrapeImpact(targetPos, velocity, astRadius);
            } else if (sizeClass < 0.2f) {
                applyAirburstImpact(targetPos, up, astRadius);
            } else {
                applyDirectImpact(targetPos, velocity, astRadius, incidenceDot);
            }
        }
        
        grid.optimize();
        std::cout << "Completed Asteroid Impacts." << std::endl;
    }

    void quickSmoothSurface() {
        TIME_FUNCTION;
        auto allNodes = grid.findInRadius(config.center, config.radius * 1.5f);
        
        std::vector<std::pair<std::shared_ptr<Octree<Particle, int16_t, "output/fibSphere">::NodeData>, v3>> smoothMoves;
        
        for (auto& n : allNodes) {
            if (!n->isActive() || n->objectId == 2) continue;
            
            auto neighbors = grid.findInRadius(n->position, config.voxelSize * 1.5f);
            if (neighbors.size() > 20) continue;
            
            v3 avgPos = v3::Zero();
            int count = 0;
            for (auto& neighbor : neighbors) {
                if (neighbor->isActive() && neighbor->objectId != 2) {
                    avgPos += neighbor->position;
                    count++;
                }
            }
            if (count > 0) {
                avgPos /= static_cast<float>(count);
                v3 newP = n->position * 0.5f + avgPos * 0.5f;
                smoothMoves.push_back({n, newP});
            }
        }
        
        for (auto& m : smoothMoves) {
            v3 oldP = m.first->position;
            m.first->data.currentPos = m.second;
            grid.move(oldP, m.second);
        }
        grid.optimize();
        std::cout << "Applied Quick Smoothing to " << smoothMoves.size() << " surface nodes." << std::endl;
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