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
    
    std::unique_ptr<AltPositions> altPos = nullptr;

    Eigen::Vector3f currentPos;
    
    float plateDisplacement = 0.0f;
    float temperature = 0.0f;
    float water = 0.0f;
    float moisture = 0.0f;
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

    // Matrix rows: 0: Sand, 1: Silt, 2: Clay, 3: Rock, 4: Metal
    // Columns are 10 sub-variants for each type.
    Eigen::Matrix<Eigen::half, 5, 10> materials = Eigen::Matrix<Eigen::half, 5, 10>::Zero();

    float impactShock = 0.0f;
    float impactHeat = 0.0f;
    float impactDebris = 0.0f;

    NeighborData nearNeighbors[8];

    Particle() {
        materials(3, 0) = Eigen::half(1.0f);
    }


    Particle(const Particle& other) {
        noiseDisplacement = other.noiseDisplacement;
        currentPos = other.currentPos;
        plateDisplacement = other.plateDisplacement;
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
        
        materials = other.materials;
        water = other.water;
        temperature = other.temperature;
        moisture = other.moisture;
        
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
            currentPos = other.currentPos;
            plateDisplacement = other.plateDisplacement;
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
            
            materials = other.materials;
            water = other.water;
            temperature = other.temperature;
            moisture = other.moisture;
            
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
        
        bool hasAltPos = (altPos != nullptr);
        writeBin(out, hasAltPos);
        if (hasAltPos) {
            writeBin(out, altPos->originalPos);
            writeBin(out, altPos->noisePos);
            writeBin(out, altPos->tectonicPos);
        }
        
        writeBin(out, currentPos);
        writeBin(out, plateDisplacement);
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
        
        writeBin(out, materials);
        writeBin(out, water);
        writeBin(out, temperature);
        writeBin(out, moisture);
        
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
        
        readBin(in, p.materials);
        readBin(in, p.water);
        readBin(in, p.temperature);
        readBin(in, p.moisture);
        
        readBin(in, p.impactShock);
        readBin(in, p.impactHeat);
        readBin(in, p.impactDebris);
        
        for(int i = 0; i < 8; ++i) {
            readBin(in, p.nearNeighbors[i].index);
            readBin(in, p.nearNeighbors[i].distance);
        }
        return p;
    }
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
    float noiseStrength = 25.0f;
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
    
    int erosionDrops = 500000;
    int weatherIterations = 100;
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
    Grid::Octree<Particle> grid{v3(-config.gridSizeCubeMin,-config.gridSizeCubeMin,-config.gridSizeCubeMin),v3(config.gridSizeCubeMin,config.gridSizeCubeMin,config.gridSizeCubeMin), "output/Planet", 16};
    std::vector<PlateConfig> plates;
    std::mt19937 rng = std::mt19937(42);
    bool starAdded = false;
    bool coreFilled = false;

    std::vector<ImpactEvent> impactHistory;
    int dynMoonId = 10;

    using NodeType = std::shared_ptr<typename Grid::Octree<Particle>::NodeData>;

    planetsim() {
        config = planetConfig();
        grid = Grid::Octree<Particle>(v3(-config.gridSizeCubeMin,-config.gridSizeCubeMin,-config.gridSizeCubeMin),v3(config.gridSizeCubeMin,config.gridSizeCubeMin,config.gridSizeCubeMin), "output/Planet", 16);
    }

    v3 getOriginColor(const NodeType& pt) {
        if (!pt) return v3(1,1,1);
        return pt->color.template head<3>();
    }

    void setOriginColor(const NodeType& pt, const v3& color) {
        grid.setColor(pt->position, color);
    }

    void changeNodeObject(const v3& pos, int newObjectId) {
        auto pt = grid.find(pos, -2, config.voxelSize * 0.5f);
        if (pt && pt->objectId != newObjectId) {
            
            auto newObj = grid.getOrCreateObject(newObjectId);
            pt->objectId = newObjectId;
        }
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
                case BlendMode::Replace:
                    finalValue = nVal * layer.strength;
                    break;
                case BlendMode::Add:
                    finalValue += nVal * layer.strength;
                    break;
                case BlendMode::Subtract:
                    finalValue -= nVal * layer.strength;
                    break;
                case BlendMode::Multiply:
                    finalValue *= (nVal * layer.strength);
                    break;
                case BlendMode::Max:
                    finalValue = std::max(finalValue, nVal * layer.strength);
                    break;
                case BlendMode::Min:
                    finalValue = std::min(finalValue, nVal * layer.strength);
                    break;
            }
        }
        
        float norm = std::tanh(finalValue);
        return norm;
    }

    void generateFibSphere() {
        TIME_FUNCTION;
        grid.load("output/fibSphere.yggs");
        grid.clear(v3(-config.gridSizeCubeMin,-config.gridSizeCubeMin,-config.gridSizeCubeMin),v3(config.gridSizeCubeMin,config.gridSizeCubeMin,config.gridSizeCubeMin));
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
            pt.noiseDisplacement = 0.0f;
            pt.surface = true;
            config.surfaceNodes[i] = pt.currentPos;
            grid.insert(pt, pt.currentPos, true, config.color, config.voxelSize, true, -1, 0.0f, 1.0f, 0.0f, 0.0f, 1.45f, config.color);
        }
        config.currentStep = 1;
        std::cout << "Step 1 done. base sphere generated" << std::endl;
        grid.waitForIdle();
        grid.save("output/fibSphere.yggs");
    }

    inline void _applyNoise(std::function<float(const Eigen::Vector3f&)> noiseFunc) {
        std::vector<v3> newPos(config.surfaceNodes.size());
        grid.waitForIdle();
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            v3 pos = config.surfaceNodes[i];
            
            auto node = grid.find(pos, -2, config.voxelSize * 0.5f);
            if (!node) {
                std::cout << "something broke as early as applynoise!" << std::endl;
                continue;
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
            // grid.update(p.currentPos, p);
            grid.updateData(p.currentPos, p);
            // grid.queuedupdate(oldPos, p.currentPos, p);
            // grid.queuedupdate(p.currentPos, p);
        }
        config.surfaceNodes = newPos;
        std::cout << "applied noise" << std::endl;
    }

    void assignSeeds() {
        grid.waitForIdle();
        std::cout << "assigning seeds" << std::endl;
        plates.clear();
        plates.resize(config.numPlates);
        float sphereSurfaceArea = 4.0f * M_PI * config.radius * config.radius;
        float averageAreaPerPlate = sphereSurfaceArea / config.numPlates;
        float minDistance = std::sqrt(averageAreaPerPlate) * 0.4f;
        std::vector<int> selectedSeedIndices;
        std::uniform_int_distribution<int> distNode(0, config.surfacePoints - 1);
        for (int i = 0; i < config.numPlates; ++i) {
            int attempts = 10;
            bool foundValidSeed = false;
            int seedid = distNode(rng);
            plates[i].plateId = i;

            while (!foundValidSeed && attempts > 0) {
                int seedIndex = distNode(rng);
                
                bool tooClose = false;
                for (int selectedIndex : selectedSeedIndices) {
                    auto existingNode = grid.find(config.surfaceNodes[selectedIndex], -2, config.voxelSize * 0.5f);
                    auto candidateNode = grid.find(config.surfaceNodes[seedIndex], -2, config.voxelSize * 0.5f);
                    if (!existingNode || !candidateNode) {
                        std::cout << "no nodes" << std::endl;
                        continue;
                    }

                    const auto& existingSeed = existingNode->data;
                    const auto& candidateSeed = candidateNode->data;

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
                    
                    auto sNode = grid.find(config.surfaceNodes[seedIndex], -2, config.voxelSize * 0.5f);
                    if (sNode) {
                        changeNodeObject(config.surfaceNodes[seedIndex], i);
                        plates[i].plateEulerPolep = sNode->position;
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

                auto sNode = grid.find(config.surfaceNodes[seedIndex], -2, config.voxelSize * 0.5f);
                if (sNode) {
                    changeNodeObject(config.surfaceNodes[seedIndex], i);
                    plates[i].plateEulerPolep = sNode->position;
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
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
            if (node) {
                normPos[i] = node->data.altPos->originalPos.normalized();
            } else {
                normPos[i] = config.surfaceNodes[i].normalized();
            }
        }
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
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
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
            if (!node) {
                std::cout << "gpr couldnt find" << std::endl;
                continue;
            }
            int pID = node->objectId;
            if (pID == -1) {
                unassignedCount++;
            } else {
                plates[pID].assignedNodes.push_back(i);
                for (int n = 0; n < 8; n++) {
                    int nIdx = node->data.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    auto nNode = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                    if (nNode && nNode->objectId == -1) {
                        frontiers[pID].push_back(nIdx);
                    }
                }
            }
        }
        
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        
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

                auto candNode = grid.find(config.surfaceNodes[candIdx], -2, config.voxelSize * 0.5f);
                if (candNode && candNode->objectId == -1) {
                    changeNodeObject(config.surfaceNodes[candIdx], selPlate);
                    
                    plates[selPlate].assignedNodes.push_back(candIdx);
                    unassignedCount--;
                    successfulGrowth = true;

                    for (int n = 0; n < 8; n++) {
                        int nIdx = candNode->data.nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        auto nNode = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                        if (nNode && nNode->objectId == -1) {
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
            auto node = grid.find(pos, -2, config.voxelSize * 0.5f);
            if (node && node->objectId == -1) unassignedCount++;
        }

        while (unassignedCount > 0) {
            std::vector<int> nextState(config.surfaceNodes.size(), -1);
            int assignedThisRound = 0;
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                if (!node) continue;
                
                if (node->objectId != -1) {
                    nextState[i] = node->objectId;
                } else {
                    std::unordered_map<int, int> counts;
                    int bestPlate = -1;
                    int maxCount = 0;
                    
                    for (int n = 0; n < 8; n++) {
                        int nIdx = node->data.nearNeighbors[n].index;
                        if (nIdx == -1) break;
                        auto nNode = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                        if (!nNode) continue;
                        
                        int pID = nNode->objectId;
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
                auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                if (node && node->objectId == -1 && nextState[i] != -1) {
                    changeNodeObject(config.surfaceNodes[i], nextState[i]);
                    plates[nextState[i]].assignedNodes.push_back(i);
                    unassignedCount--;
                }
            }
            
            if (assignedThisRound == 0 && unassignedCount > 0) {
                for (int i = 0; i < config.surfaceNodes.size(); i++) {
                    auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                    if (node && node->objectId == -1) {
                        int closestPlate = 0;
                        float minDist = std::numeric_limits<float>::max();
                        for (int p = 0; p < config.numPlates; p++) {
                            auto asdfsdfsdfsdfs = grid.find(plates[p].plateEulerPolep);
                            if (!asdfsdfsdfsdfs) continue;
                            float d = (node->data.altPos->originalPos - asdfsdfsdfsdfs->data.altPos->originalPos).norm();
                            if (d < minDist) {
                                minDist = d;
                                closestPlate = p;
                            }
                        }
                        changeNodeObject(config.surfaceNodes[i], closestPlate);
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
                auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                if (!node) continue;
                
                std::unordered_map<int, int> counts;
                counts[node->objectId]++;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = node->data.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    auto nNode = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                    if (nNode) counts[nNode->objectId]++;
                }
                
                int bestPlate = node->objectId;
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
                auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                if (node && nextPlateID[i] != -1 && node->objectId != nextPlateID[i]) {
                    changeNodeObject(config.surfaceNodes[i], nextPlateID[i]);
                }
            }
        }
        
        for (auto& plate : plates) {
            plate.assignedNodes.clear();
        }
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
            if (node && node->objectId != -1) {
                plates[node->objectId].assignedNodes.push_back(i);
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
                auto node = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                if (!node) continue;
                sumElevation += node->data.currentPos.norm();
                centroid += node->data.altPos->originalPos;
            }
            
            if (!plates[i].assignedNodes.empty()) {
                plateStats[i].second = sumElevation / plates[i].assignedNodes.size();
                centroid /= plates[i].assignedNodes.size();

                float maxSpread = 0.0f;
                for (int nIdx : plates[i].assignedNodes) {
                    auto node = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
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
                            auto node = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                            if (!node) continue;
                            float d = (node->data.altPos->originalPos - centroid).norm();
                            if (d < minDistToCentroid) {
                                minDistToCentroid = d;
                                bestNodeIdx = nIdx;
                            }
                        }
                        auto bestNode = grid.find(config.surfaceNodes[bestNodeIdx], -2, config.voxelSize * 0.5f);
                        if (bestNode) {
                            plates[i].plateEulerPolep = bestNode->position;
                        }
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
                auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
                if (!node) continue;
                Particle p = node->data;

                int myPlate = node->objectId;
                if (myPlate == -1) continue;
                
                Eigen::Vector3f myPos = p.altPos->originalPos.normalized();
                Eigen::Vector3f myVel = ω[myPlate].cross(myPos);
                
                float localStress = 0.0f;
                float localNoise = 0.0f;
                int boundaryCount = 0;
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = p.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    
                    auto nNode = grid.find(config.surfaceNodes[nIdx], -2, config.voxelSize * 0.5f);
                    if (!nNode) continue;

                    int nPlate = nNode->objectId;
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
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
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

        grid.waitForIdle();
        std::vector<v3> newPos(config.surfaceNodes.size());
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            v3 pos = config.surfaceNodes[i];
            auto node = grid.find(pos, -2, config.voxelSize * 0.5f);
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
            auto pt1 = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
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
                
                auto pt2 = grid.find(config.surfaceNodes[j], -2, config.voxelSize * 0.5f);
                if (!pt2) continue;
                Particle p2 = pt2->data;

                for (int n2 = 0; n2 < 8; n2++) {
                    int k = p2.nearNeighbors[n2].index;
                    if (k == -1) break;
                    if (k <= j) continue;
                    
                    auto pt3 = grid.find(config.surfaceNodes[k], -2, config.voxelSize * 0.5f);
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

            auto pt1 = grid.find(config.surfaceNodes[idx1], -2, config.voxelSize * 0.5f);
            Particle p1 = pt1->data;
            auto pt2 = grid.find(config.surfaceNodes[idx2], -2, config.voxelSize * 0.5f);
            Particle p2 = pt2->data;
            auto pt3 = grid.find(config.surfaceNodes[idx3], -2, config.voxelSize * 0.5f);
            Particle p3 = pt3->data;

            if (!p1.altPos || !p2.altPos || !p3.altPos) continue;

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
                    
                    int newPtObjectId = -1;
                    v3 newPtColor(1,1,1);
                    if (w1 > w2 && w1 > w3) {
                        newPtObjectId = pt1->objectId;
                        newPtColor = getOriginColor(pt1);
                    } else if (w2 > w3) {
                        newPtObjectId = pt2->objectId;
                        newPtColor = getOriginColor(pt2);
                    } else {
                        newPtObjectId = pt3->objectId;
                        newPtColor = getOriginColor(pt3);
                    }

                    grid.queuedset(newPt, newPt.currentPos, true, newPtColor, config.voxelSize, true, newPtObjectId, 0.0f, 1.0f, 0.0f, 0.0f, 1.45f, newPtColor);
                    
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
            auto node = grid.find(pos, -2, config.voxelSize * 0.5f);
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
            auto node = grid.find(pos, -2, config.voxelSize * 0.5f);
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
                        ip.currentPos = pos;

                        float depthRatio = dist / localMaxRadius;
                        Eigen::Vector3f coreColor(1.0f, 0.9f, 0.4f);
                        Eigen::Vector3f mantleColor(0.8f, 0.15f, 0.0f);
                        Eigen::Vector3f finalColor = mantleColor;

                        if (depthRatio < 0.5f) {
                            float blend = depthRatio * 2.0f;
                            float quantizedBlend = std::round(blend * 32.0f) / 32.0f;
                            finalColor = coreColor * (1.0f - quantizedBlend) + mantleColor * quantizedBlend;
                        }

                        ip.mass = 100.0f;

                        int interiorObjectId = config.numPlates; 
                        grid.queuedset(ip, pos, true, finalColor, config.voxelSize, true, interiorObjectId, 0.0f, 1.0f, 0.0f, 0.0f, 1.45f, finalColor);
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
        std::vector<std::tuple<v3, Particle, v3>> toUpdate;
        
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
                p.materials(4, 0) = Eigen::half(std::clamp(static_cast<float>(p.materials(4, 0)) + falloff * 0.5f, 0.0f, 1.0f));
                
                v3 origCol = getOriginColor(n);
                v3 burnCol(0.1f, 0.05f, 0.05f); // Reddish dark scorch
                v3 newCol = origCol * (1.0f - falloff) + burnCol * falloff;
                setOriginColor(n, newCol);
                
                // Push nodes out slightly to form a crater rim
                v3 pushDir = (n->position - impactCenter).normalized();
                p.currentPos = n->position + pushDir * (falloff * config.voxelSize * 2.0f);
                
                toUpdate.push_back({n->position, p, newCol});
            }
        }
        
        for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
        for (auto& up : toUpdate) {
            grid.move(std::get<0>(up), std::get<1>(up).currentPos);
            grid.updateData(std::get<1>(up).currentPos, std::get<1>(up));
            grid.setColor(std::get<1>(up).currentPos, std::get<2>(up));
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
            std::vector<std::tuple<v3, Particle, v3>> toUpdate;
            
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
                    
                    v3 origCol = getOriginColor(n);
                    v3 burnCol(0.2f, 0.15f, 0.1f);
                    v3 newCol = (origCol * (1.0f - falloff) + burnCol * falloff);
                    setOriginColor(n, newCol);
                    
                    toUpdate.push_back({n->position, p, newCol});
                }
            }
            
            for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
            for (auto& up : toUpdate) {
                grid.updateData(std::get<0>(up), std::get<1>(up));
                grid.setColor(std::get<0>(up), std::get<2>(up));
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
            std::vector<std::tuple<v3, Particle, v3>> toUpdate;
            
            for (auto& n : affected) {
                if (!n->isActive()) continue;
                float dist = (n->position - fragPos).norm();
                if (dist < fragRadius) {
                    toRemove.push_back(n->position);
                } else {
                    Particle p = n->data;
                    float falloff = 1.0f - ((dist - fragRadius) / (fragRadius * 0.5f));
                    
                    p.impactDebris = std::clamp(p.impactDebris + falloff, 0.0f, 1.0f);
                    
                    v3 origCol = getOriginColor(n);
                    v3 dustCol(0.6f, 0.5f, 0.4f);
                    v3 newCol = (origCol * (1.0f - falloff) + dustCol * falloff);
                    setOriginColor(n, newCol);
                    
                    toUpdate.push_back({n->position, p, newCol});
                }
            }
            
            for (auto& pos : toRemove) grid.remove(pos, config.voxelSize);
            for (auto& up : toUpdate) {
                grid.updateData(std::get<0>(up), std::get<1>(up));
                grid.setColor(std::get<0>(up), std::get<2>(up));
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
        
        std::vector<std::pair<std::shared_ptr<Grid::Octree<Particle>::NodeData>, v3>> smoothMoves;
        
        for (auto& n : allNodes) {
            if (!n->isActive() || n->objectId == config.numPlates) continue;
            
            auto neighbors = grid.findInRadius(n->position, config.voxelSize * 1.5f);
            if (neighbors.size() > 20) continue;
            
            v3 avgPos = v3::Zero();
            int count = 0;
            for (auto& neighbor : neighbors) {
                if (neighbor->isActive() && neighbor->objectId != config.numPlates) {
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
        TIME_FUNCTION;
        std::cout << "Starting hydraulic erosion with " << config.erosionDrops << " drops." << std::endl;
        grid.waitForIdle();

        std::vector<Particle> surfData(config.surfaceNodes.size());
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
            if (node) surfData[i] = node->data;
            else {
                surfData[i] = Particle(); 
                surfData[i].currentPos = config.surfaceNodes[i];
            }
        }

        std::uniform_int_distribution<int> distNode(0, config.surfaceNodes.size() - 1);
        
        float evapRate = 0.05f;
        float depositionRate = 0.1f;
        float minVol = 0.01f;
        float friction = 0.1f;
        
        for (int d = 0; d < config.erosionDrops; d++) {
            int idx = distNode(rng);
            float water = 1.0f;
            float vel = 1.0f;
            float sediment = 0.0f;
            
            int maxSteps = 50;
            for (int step = 0; step < maxSteps; step++) {
                Particle& p = surfData[idx];
                
                int lowestIdx = idx;
                float minHeight = p.currentPos.norm() + p.water; 
                
                for (int n = 0; n < 8; n++) {
                    int nIdx = p.nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    float nh = surfData[nIdx].currentPos.norm() + surfData[nIdx].water;
                    if (nh < minHeight) {
                        minHeight = nh;
                        lowestIdx = nIdx;
                    }
                }
                
                if (lowestIdx == idx) {
                    p.water += water;
                    float heightIncrease = sediment;
                    p.currentPos += p.currentPos.normalized() * heightIncrease;
                    p.materials(0, 0) = Eigen::half(static_cast<float>(p.materials(0,0)) + sediment);
                    break;
                }
                
                Particle& nP = surfData[lowestIdx];
                float hDiff = (p.currentPos.norm() + p.water) - (nP.currentPos.norm() + nP.water);
                
                float capacity = std::max(0.01f, vel * water * hDiff * 2.0f);
                
                if (sediment > capacity) {
                    float amount = (sediment - capacity) * depositionRate;
                    sediment -= amount;
                    p.currentPos += p.currentPos.normalized() * amount;
                    p.materials(0, 0) = Eigen::half(static_cast<float>(p.materials(0,0)) + amount);
                } else {
                    float amount = std::min((capacity - sediment) * depositionRate, hDiff * 0.9f); 
                    sediment += amount;
                    p.currentPos -= p.currentPos.normalized() * amount;
                    float currentRock = static_cast<float>(p.materials(3, 0));
                    if(currentRock > amount) {
                        p.materials(3, 0) = Eigen::half(currentRock - amount);
                    } else {
                        float currentSand = static_cast<float>(p.materials(0, 0));
                        if (currentSand > amount) {
                            p.materials(0, 0) = Eigen::half(currentSand - amount);
                        }
                    }
                }
                
                vel = std::sqrt(std::max(0.01f, vel * vel + hDiff * 10.0f));
                vel *= (1.0f - friction);
                water *= (1.0f - evapRate);
                
                idx = lowestIdx;
                
                if (water < minVol) {
                    p.currentPos += p.currentPos.normalized() * sediment;
                    p.materials(0, 0) = Eigen::half(static_cast<float>(p.materials(0,0)) + sediment);
                    break;
                }
            }
        }

        std::vector<v3> newPos(config.surfaceNodes.size());
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            v3 oldP = config.surfaceNodes[i];
            Particle p = surfData[i];
            newPos[i] = p.currentPos;
            grid.move(oldP, p.currentPos);
            grid.updateData(p.currentPos, p);
        }
        config.surfaceNodes = newPos;
        grid.optimize();
        std::cout << "Erosion completed." << std::endl;
    }

    void storms() {
        TIME_FUNCTION;
        std::cout << "Starting weather simulation for " << config.weatherIterations << " iterations..." << std::endl;
        grid.waitForIdle();

        std::vector<Particle> surfData(config.surfaceNodes.size());
        std::vector<v3> normals(config.surfaceNodes.size());
        std::vector<float> heights(config.surfaceNodes.size());
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            auto node = grid.find(config.surfaceNodes[i], -2, config.voxelSize * 0.5f);
            if (node) {
                surfData[i] = node->data;
                normals[i] = node->data.currentPos.normalized();
                heights[i] = node->data.currentPos.norm();
            } else {
                surfData[i] = Particle();
                normals[i] = config.surfaceNodes[i].normalized();
                heights[i] = config.surfaceNodes[i].norm();
            }
        }

        v3 starDir = v3(1.0f, 0.0f, 0.0f);
        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            float latitude = std::asin(std::clamp(normals[i].y(), -1.0f, 1.0f));
            float insolation = std::max(0.0f, normals[i].dot(starDir)); 
            
            float baseTemp = std::cos(latitude) * 30.0f - 10.0f; 
            
            float altitude = heights[i] - config.radius;
            baseTemp -= std::max(0.0f, altitude) * 0.02f; 

            surfData[i].temperature = baseTemp;
        }

        std::vector<float> moisture(config.surfaceNodes.size(), 0.0f);
        std::vector<float> clouds(config.surfaceNodes.size(), 0.0f);
        
        for (int it = 0; it < config.weatherIterations; it++) {
            std::vector<float> nextClouds = clouds;
            
            for (int i = 0; i < config.surfaceNodes.size(); i++) {
                if (surfData[i].water > 0.0f) {
                    float evap = std::max(0.0f, surfData[i].temperature + 10.0f) * 0.005f * std::min(1.0f, surfData[i].water);
                    nextClouds[i] += evap;
                }

                float lat = std::asin(std::clamp(normals[i].y(), -1.0f, 1.0f));
                v3 up = normals[i];
                v3 N = v3(0, 1, 0);
                v3 east = N.cross(up);
                if (east.norm() < 1e-4f) {
                    east = v3(1, 0, 0); 
                } else {
                    east.normalize();
                }
                v3 north = up.cross(east).normalized();
                
                v3 windDir;
                float absLat = std::abs(lat);
                if (absLat < M_PI / 6.0f) { 
                    windDir = -east + (lat > 0 ? -north : north) * 0.2f;
                } else if (absLat < M_PI / 3.0f) { 
                    windDir = east + (lat > 0 ? north : -north) * 0.2f;
                } else { 
                    windDir = -east + (lat > 0 ? -north : north) * 0.2f;
                }
                windDir.normalize();

                int bestNeighbor = -1;
                float maxDot = -1.0f;
                for (int n = 0; n < 8; n++) {
                    int nIdx = surfData[i].nearNeighbors[n].index;
                    if (nIdx == -1) break;
                    v3 toNeighbor = (normals[nIdx] - normals[i]).normalized();
                    float d = toNeighbor.dot(windDir);
                    if (d > maxDot) {
                        maxDot = d;
                        bestNeighbor = nIdx;
                    }
                }

                if (bestNeighbor != -1 && clouds[i] > 0.0f) {
                    float transfer = clouds[i] * 0.8f; 
                    nextClouds[i] -= transfer;
                    nextClouds[bestNeighbor] += transfer;
                    
                    float hDiff = heights[bestNeighbor] - heights[i];
                    if (hDiff > 0) {
                        float rain = std::min(transfer, hDiff * 0.05f);
                        nextClouds[bestNeighbor] -= rain;
                        moisture[bestNeighbor] += rain;
                    }
                }
                
                float capacity = std::max(0.1f, (surfData[i].temperature + 10.0f) * 0.2f);
                if (nextClouds[i] > capacity) {
                    float rain = (nextClouds[i] - capacity) * 0.5f;
                    nextClouds[i] -= rain;
                    moisture[i] += rain;
                }
            }
            clouds = nextClouds;
        }

        for (int i = 0; i < config.surfaceNodes.size(); i++) {
            surfData[i].moisture = moisture[i];
            grid.updateData(surfData[i].currentPos, surfData[i]);
        }
        std::cout << "Weather simulation completed." << std::endl;
    }

};

#endif