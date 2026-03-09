#ifndef PLANT_HPP
#define PLANT_HPP

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include "../grid/grid3eigen.hpp"

using v3 = Eigen::Vector3f;

enum class ParticleType {
    AIR,
    SUN,
    DIRT,
    PLANT,
    WATER,
    SNOW
};

enum class PlantPart {
    SEED,
    ROOT,
    LEAF
};

struct PlantsimParticle {
    ParticleType pt;
    float energy;
    v3 velocity = v3(0, 0, 0);
    float timeOutOfBounds = 0.0f;
    bool anchored = false;
    float strength = 15.0f;

    PlantsimParticle() : pt(ParticleType::AIR), energy(0.0) {}
    PlantsimParticle(ParticleType t, float e) : pt(t), energy(e) {}
    virtual ~PlantsimParticle() = default;
};

struct AirParticle : public PlantsimParticle {
    float co2;
    float temperature;

    AirParticle(float c = 400.0f, float t = 20.0f) 
        : PlantsimParticle(ParticleType::AIR, 0.0f), co2(c), temperature(t) {}
};

struct SunParticle : public PlantsimParticle {
    SunParticle() : PlantsimParticle(ParticleType::SUN, 1000.0f) {}
};

struct DirtParticle : public PlantsimParticle {
    float nitrogen;
    float phosphorus;
    float potassium;
    float carbon;
    float magnesium;
    
    float hydration;
    float temperature;

    DirtParticle(float n = 100.0f, float p = 100.0f, float k = 100.0f, float c = 100.0f, float mg = 100.0f) 
        : PlantsimParticle(ParticleType::DIRT, 0.0f), nitrogen(n), phosphorus(p), potassium(k), carbon(c), magnesium(mg),
          hydration(0.0f), temperature(20.0f) {}
};

struct PlantParticle : public PlantsimParticle {
    PlantPart part;
    float age;
    float water;
    
    float nitrogen;
    float phosphorus;
    float potassium;
    float carbon;
    float magnesium;

    PlantParticle(PlantPart p = PlantPart::SEED) : PlantsimParticle(ParticleType::PLANT, 10.0f), part(p), age(0.0f),
                  water(10.0f), nitrogen(0.0f), phosphorus(0.0f), potassium(0.0f), carbon(0.0f), magnesium(0.0f) {}
};

struct WaterParticle : public PlantsimParticle {
    WaterParticle() : PlantsimParticle(ParticleType::WATER, 5.0f) {}
};

struct SnowParticle : public PlantsimParticle {
    float meltProgress = 0.0f;
    SnowParticle() : PlantsimParticle(ParticleType::SNOW, 2.0f) {}
};

struct PlantConfig {
    // World Settings
    float voxelSize = 0.5f;
    float groundSize = 20.0f;
    
    // Sun Settings
    float dayDuration = 60.0f;
    float sunInclination = 25.0f;
    float timeOfDay = 0.3f;
    float season = 0.0f;
    int currentDay = 0;
    int daysPerYear = 24;
    float latitude = 45.0f;
    float axialTilt = 23.5f;
    float sunDistance = 60.0f;
    float sunSize = 25.0f;
    Eigen::Vector3f sunColor = Eigen::Vector3f(1.0f, 0.95f, 0.8f);
    float sunIntensity = 50.0f;
    float precipRate = 80.0f;
};

class PlantSim {
public:
    enum class WeatherState { CLEAR, RAIN, SNOW };

    PlantConfig config;
    Octree<std::shared_ptr<PlantsimParticle>> grid;
    std::mt19937 rng;
    v3 currentSunPos = v3(0, 0, 0);
    bool worldInitialized = false;
    bool sunExists = false;
    
    float precipAccumulator = 0.0f;
    float currentTemperature = 20.0f;
    float cloudCover = 0.0f;
    WeatherState currentWeather = WeatherState::CLEAR;
    float weatherTimer = 0.0f;
    float atmosphericMoisture = 0.0f; 
    float extendedHeatTimer = 0.0f;
    
    float hourlyTimer = 0.0f;
    std::vector<uint16_t> dirtColorPalette;

    // Plant Stats
    int leafCount = 0;
    int rootCount = 0;
    float totalPlantEnergy = 0.0f;
    float totalPlantWater = 0.0f;

    PlantSim() : rng(std::random_device{}()) {
        grid = Octree<std::shared_ptr<PlantsimParticle>>(v3(-100, -100, -100), v3(100, 100, 100), 8, 16);
        
        // Default Sky settings for the octree renderer
        grid.setSkylight(v3(0.05f, 0.05f, 0.1f)); 
        grid.setBackgroundColor(v3(0.02f, 0.02f, 0.05f)); 
    }

    void buildDirtPalette() {
        dirtColorPalette.resize(11 * 11);
        for (int h = 0; h <= 10; ++h) {
            for (int w = 0; w <= 10; ++w) {
                float dryHeat = h / 10.0f;
                float wetness = w / 10.0f;
                v3 dryColor(0.36f, 0.25f, 0.20f);
                v3 wetColor(0.15f, 0.10f, 0.08f);
                v3 bakedColor(0.65f, 0.55f, 0.40f); 
                v3 baseColor = dryColor * (1.0f - dryHeat) + bakedColor * dryHeat;
                v3 newColor = baseColor * (1.0f - wetness) + wetColor * wetness;
                dirtColorPalette[h * 11 + w] = grid.getColorIndex(newColor);
            }
        }
    }

    void initWorld() {
        grid.clear();
        sunExists = false;
        precipAccumulator = 0.0f;
        weatherTimer = 0.0f;
        currentWeather = WeatherState::CLEAR;
        atmosphericMoisture = 0.0f;
        extendedHeatTimer = 0.0f;
        hourlyTimer = 0.0f;

        buildDirtPalette();

        float vSize = config.voxelSize;
        int gridRadius = static_cast<int>(config.groundSize / (2.0f * vSize));
        int depth = 4; // Generate multiple layers of dirt
        v3 dirtColor(0.36f, 0.25f, 0.20f);
        
        v3 seedPos(0, config.voxelSize / 2.0f + 0.01f, 0); 
        
        for (int x = -gridRadius; x <= gridRadius; ++x) {
            for (int z = -gridRadius; z <= gridRadius; ++z) {
                for (int y = 0; y < depth; ++y) {
                    auto dirt = std::make_shared<DirtParticle>();
                    dirt->hydration = 10.0f;
                    v3 pos(x * vSize, -(y + 0.5f) * vSize, z * vSize);
                    grid.set(dirt, pos, true, dirtColor, vSize, true, 0, 0, 0.0f, 1.0f);
                }
                
                // Air Layers (y >= 0)
                int airHeight = static_cast<int>(config.groundSize / vSize);
                for (int y = 0; y < airHeight; ++y) {
                    v3 pos(x * vSize, (y + 0.5f) * vSize, z * vSize);
                    
                    // Don't spawn air directly on top of where the seed will be
                    if (std::abs(pos.x() - seedPos.x()) < 0.001f && 
                        std::abs(pos.y() - seedPos.y()) < 0.001f && 
                        std::abs(pos.z() - seedPos.z()) < 0.001f) {
                        continue;
                    }

                    auto air = std::make_shared<AirParticle>(400.0f, 20.0f);
                    grid.set(air, pos, false, v3(0,0,0), vSize, true, 5, 0);
                }
            }
        }

        auto seed = std::make_shared<PlantParticle>(PlantPart::SEED);
        seed->energy = 25.0f;
        seed->water = 25.0f;
        v3 plantColor(0.8f, 0.7f, 0.2f);
        
        grid.set(seed, seedPos, true, plantColor, config.voxelSize * 0.5f, true, 1, 0, 0.0f, 0.6f);

        worldInitialized = true;
        
        updateSunPosition(true);
        updateEnvironment(0.016f);
    }

    void spawnPrecipitation() {
        std::uniform_real_distribution<float> xzDist(-config.groundSize / 2.0f, config.groundSize / 2.0f);
        std::uniform_real_distribution<float> yDist(config.groundSize, config.groundSize * 1.5f);
        
        v3 pos(xzDist(rng), yDist(rng), xzDist(rng));
        
        if (currentWeather == WeatherState::RAIN) {
            auto rain = std::make_shared<WaterParticle>();
            rain->velocity = v3(0, -2.0f, 0); 
            float rainSize = config.voxelSize * 0.2f;
            v3 rainColor(0.3f, 0.5f, 0.9f); 
            grid.set(rain, pos, true, rainColor, rainSize, true, 3, 0, 0.0f, 0.6f); 
        } else if (currentWeather == WeatherState::SNOW) {
            auto snow = std::make_shared<SnowParticle>();
            snow->velocity = v3(0, -0.5f, 0); 
            float snowSize = config.voxelSize * 0.3f;
            v3 snowColor(0.9f, 0.95f, 1.0f); 
            grid.set(snow, pos, true, snowColor, snowSize, true, 4, 0, 0.0f, 0.9f); 
        }
    }

    void updateWeather(float dt) {
        weatherTimer -= dt;
        
        if (weatherTimer <= 0.0f) {
            if (currentWeather != WeatherState::CLEAR) {
                // Stop Precipitating
                currentWeather = WeatherState::CLEAR;
                weatherTimer = std::uniform_real_distribution<float>(0.2f, 0.8f)(rng) * config.dayDuration;
            } else {
                float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                float precipChance = 0.2f;
                float durationDays = 0.1f;
                
                float delta = config.axialTilt * M_PI / 180.0f * std::sin(config.season * 2.0f * M_PI);
                
                if (std::abs(config.latitude) < 20.0f) {
                    float latRad = config.latitude * M_PI / 180.0f;
                    float diff = std::abs(latRad - delta);
                    precipChance = (diff < 0.2f) ? 0.6f : 0.1f;
                    durationDays = std::uniform_real_distribution<float>(0.1f, 0.4f)(rng);
                } else {
                    float springFallFactor = std::abs(std::sin(config.season * 4.0f * M_PI)); 
                    precipChance = 0.15f + 0.25f * springFallFactor; // More precip in spring/autumn transition
                    
                    if (currentTemperature > 25.0f) {
                        durationDays = std::uniform_real_distribution<float>(0.02f, 0.08f)(rng); // Short summer showers
                    } else if (currentTemperature < 0.0f) {
                        durationDays = std::uniform_real_distribution<float>(0.2f, 0.6f)(rng); // Long winter snows
                    } else {
                        durationDays = std::uniform_real_distribution<float>(0.1f, 0.3f)(rng); // Normal rain
                    }
                }
                
                float moistureBonus = std::min(atmosphericMoisture * 0.005f, 0.4f);
                
                if (extendedHeatTimer > 3.0f * config.dayDuration) { // Heat wave
                    moistureBonus = 0.0f;
                    precipChance *= 0.3f; // Desertification stops rain severely
                }
                
                precipChance += moistureBonus;

                if (r < precipChance) {
                    currentWeather = (currentTemperature < 0.0f) ? WeatherState::SNOW : WeatherState::RAIN;
                    weatherTimer = durationDays * config.dayDuration;
                    atmosphericMoisture *= 0.5f;
                } else {
                    currentWeather = WeatherState::CLEAR;
                    weatherTimer = std::uniform_real_distribution<float>(0.2f, 0.8f)(rng) * config.dayDuration;
                }
            }
        }
        
        float targetCloudCover = (currentWeather != WeatherState::CLEAR) ? 0.8f : 0.0f;
        cloudCover += (targetCloudCover - cloudCover) * dt * 0.1f; // Lerp clouds
    }

    void updatePlants(float dt) {
        float searchRadius = std::max(config.groundSize, config.groundSize * 2.0f) * 2.0f;
        auto plantNodes = grid.findInRadius(v3(0,0,0), searchRadius, 1); // objectId 1 = Plant
        
        leafCount = 0;
        rootCount = 0;
        totalPlantEnergy = 0.0f;
        totalPlantWater = 0.0f;
        
        if (plantNodes.empty()) return;
        
        std::vector<std::shared_ptr<Octree<std::shared_ptr<PlantsimParticle>>::NodeData>> leaves;
        std::vector<std::shared_ptr<Octree<std::shared_ptr<PlantsimParticle>>::NodeData>> roots;
        
        for (auto& node : plantNodes) {
            auto p = std::static_pointer_cast<PlantParticle>(node->data);
            if (p->part == PlantPart::LEAF) leaves.push_back(node);
            else roots.push_back(node); 
            
            totalPlantEnergy += p->energy;
            totalPlantWater += p->water;
            p->age += dt;
        }

        leafCount = leaves.size();
        rootCount = roots.size();
        
        double timeBudgetPerLeaf = 0.0;
        if (!leaves.empty()) {
            timeBudgetPerLeaf = static_cast<double>(dt) / leaves.size(); 
        }
        
        // 1. Light gathering & Photosynthesis for leaves
        #pragma omp parallel for
        for (size_t i = 0; i < leaves.size(); ++i) {
            auto& leafNode = leaves[i];
            auto p = std::static_pointer_cast<PlantParticle>(leafNode->data);
            
            int res = std::clamp(static_cast<int>((leafNode->size / config.voxelSize) * 16.0f), 4, 16);
            
            Camera leafCam;
            leafCam.origin = leafNode->position + v3(0, leafNode->size * 0.51f, 0);
            
            v3 sunDir = currentSunPos.normalized();
            if (sunDir.y() < 0.1f) sunDir.y() = 0.1f; // Avoid rendering straight into the ground

            leafCam.direction = sunDir;
            if (std::abs(leafCam.direction.y()) > 0.99f) {
                leafCam.up = v3(1, 0, 0);
            } else {
                leafCam.up = v3(0, 1, 0);
            }
            leafCam.fov = 90.0f; // Broad FOV to collect ambient scattering
            
            // Generate the small frame to detect incoming sunlight using grid3eigen's existing renderer
            frame lightFrame = grid.renderFrameTimed(leafCam, res, res, frame::colormap::RGB, timeBudgetPerLeaf, 1, false, true);
            
            float totalLight = 0.0f;
            const auto& frameData = lightFrame.getData();
            int pixelCount = res * res;
            
            for (int p_idx = 0; p_idx < pixelCount * 3; p_idx += 3) {
                float r = frameData[p_idx];
                float g = frameData[p_idx+1];
                float b = frameData[p_idx+2];
                // Approximate perceived luminance representing energy capture
                totalLight += (r * 0.299f + g * 0.587f + b * 0.114f);
            }
            float lightIntensity = totalLight / static_cast<float>(pixelCount);
            
            // Photosynthesis: Convert water & light -> energy
            float maxWaterUsable = dt * 10.0f;
            float waterUsed = std::min(p->water, lightIntensity * maxWaterUsable);
            p->water -= waterUsed;
            p->energy += waterUsed * 2.5f; // Produce more energy than water consumed
            
            // Passive drain to stay alive
            p->water -= dt * 0.5f;
            p->energy -= dt * 0.5f;
        }
        
        // 2. Roots absorb water from dirt
        #pragma omp parallel for
        for (size_t i = 0; i < roots.size(); ++i) {
            auto& rootNode = roots[i];
            auto p = std::static_pointer_cast<PlantParticle>(rootNode->data);
            
            auto dirtNodes = grid.findInRadius(rootNode->position, config.voxelSize * 1.5f, 0);
            float waterAbsorbed = 0.0f;
            for (auto& dirt : dirtNodes) {
                auto dp = std::static_pointer_cast<DirtParticle>(dirt->data);
                if (dp->hydration > 0.5f) {
                    float absorb = std::min(dp->hydration, dt * 5.0f);
                    #pragma omp atomic
                    dp->hydration -= absorb;
                    waterAbsorbed += absorb;
                }
            }
            p->water += waterAbsorbed;
            
            // Passive root maintenance drain
            p->water -= dt * 0.1f;
            p->energy -= dt * 0.5f;
        }
        
        // 3. Resource Diffusion (Homogenizes resources, effectively carrying water up and energy down naturally)
        std::vector<float> dWater(plantNodes.size(), 0.0f);
        std::vector<float> dEnergy(plantNodes.size(), 0.0f);
        
        for (size_t i = 0; i < plantNodes.size(); ++i) {
            auto p1 = std::static_pointer_cast<PlantParticle>(plantNodes[i]->data);
            auto neighbors = grid.findInRadius(plantNodes[i]->position, config.voxelSize * 1.2f, 1);
            for (auto& n : neighbors) {
                if (n == plantNodes[i]) continue;
                auto p2 = std::static_pointer_cast<PlantParticle>(n->data);
                
                float diffW = (p1->water - p2->water) * 2.0f * dt;
                float diffE = (p1->energy - p2->energy) * 2.0f * dt;
                
                dWater[i] -= diffW;
                dEnergy[i] -= diffE;
            }
        }

        // Apply diffusions & evaluate survival/growth
        struct NewPlantData {
            v3 pos;
            PlantPart part;
            float initW, initE;
        };
        std::vector<NewPlantData> newPlants;
        std::vector<v3> toRemove;

        v3 upDir(0, config.voxelSize, 0);
        v3 downDir(0, -config.voxelSize, 0);
        v3 sides[4] = { v3(config.voxelSize,0,0), v3(-config.voxelSize,0,0), v3(0,0,config.voxelSize), v3(0,0,-config.voxelSize) };

        for (size_t i = 0; i < plantNodes.size(); ++i) {
            auto p = std::static_pointer_cast<PlantParticle>(plantNodes[i]->data);
            p->water = std::max(0.0f, p->water + dWater[i]);
            p->energy = std::max(0.0f, p->energy + dEnergy[i]);
            
            if (p->energy <= 0.0f && p->water <= 0.0f) {
                toRemove.push_back(plantNodes[i]->position);
                continue;
            }

            // Update Visuals depending on health
            v3 newColor;
            if (p->part == PlantPart::LEAF) {
                v3 healthyGreen(0.1f, 0.8f, 0.1f);
                v3 dryYellow(0.7f, 0.7f, 0.2f);
                float health = std::clamp(p->water / 15.0f, 0.0f, 1.0f);
                newColor = dryYellow * (1.0f - health) + healthyGreen * health;
            } else if (p->part == PlantPart::ROOT) {
                newColor = v3(0.6f, 0.5f, 0.4f);
            } else { // SEED
                newColor = v3(0.8f, 0.7f, 0.2f);
            }

            // --- Growth & Split Logic ---
            float growThreshold = 20.0f;
            
            if (p->part == PlantPart::SEED && p->energy > 15.0f && p->water > 15.0f) {
                // Morph to root, spawn leaf directly above
                p->part = PlantPart::ROOT;
                p->energy -= 10.0f;
                p->water -= 10.0f;
                newPlants.push_back({plantNodes[i]->position + upDir, PlantPart::LEAF, 10.0f, 10.0f});
            }
            else if (p->energy > growThreshold && p->water > growThreshold) {
                
                if (plantNodes[i]->size < config.voxelSize * 0.95f) {
                    // Grow in place
                    float growth = config.voxelSize * 0.2f * dt;
                    float newSize = std::min(config.voxelSize, plantNodes[i]->size + growth);
                    p->energy -= 5.0f * dt;
                    p->water -= 5.0f * dt;
                    grid.update(plantNodes[i]->position, plantNodes[i]->position, plantNodes[i]->data, true, newColor, newSize);
                } else {
                    // Maximum size reached, try to replicate / split
                    bool split = false;
                    v3 basePos = plantNodes[i]->position;
                    
                    if (p->part == PlantPart::LEAF) {
                        if (!grid.find(basePos + upDir, config.voxelSize * 0.1f)) {
                            newPlants.push_back({basePos + upDir, PlantPart::LEAF, 10.0f, 10.0f});
                            split = true;
                        } else {
                            for (auto& s : sides) {
                                if (!grid.find(basePos + s + upDir * 0.5f, config.voxelSize * 0.1f)) {
                                    newPlants.push_back({basePos + s, PlantPart::LEAF, 10.0f, 10.0f});
                                    split = true; break;
                                }
                            }
                        }
                    } 
                    else if (p->part == PlantPart::ROOT) {
                        if (!grid.find(basePos + downDir, config.voxelSize * 0.1f)) {
                            newPlants.push_back({basePos + downDir, PlantPart::ROOT, 10.0f, 10.0f});
                            split = true;
                        } else {
                            for (auto& s : sides) {
                                if (!grid.find(basePos + s + downDir * 0.5f, config.voxelSize * 0.1f)) {
                                    newPlants.push_back({basePos + s + downDir, PlantPart::ROOT, 10.0f, 10.0f});
                                    split = true; break;
                                }
                            }
                        }
                    }
                    
                    if (split) {
                        p->energy -= 15.0f;
                        p->water -= 15.0f;
                    }
                }
            } else {
                // Just update color
                grid.update(plantNodes[i]->position, plantNodes[i]->position, plantNodes[i]->data, true, newColor, plantNodes[i]->size);
            }
        }

        // Apply structural changes
        for (auto& pos : toRemove) grid.remove(pos);
        for (auto& np : newPlants) {
            auto plantCell = std::make_shared<PlantParticle>(np.part);
            plantCell->water = np.initW;
            plantCell->energy = np.initE;
            v3 initColor = (np.part == PlantPart::LEAF) ? v3(0.1f, 0.8f, 0.1f) : v3(0.6f, 0.5f, 0.4f);
            // Spawn at half size and grow into the space
            grid.set(plantCell, np.pos, true, initColor, config.voxelSize * 0.5f, true, 1, 0, 0.0f, 0.8f);
        }
    }

    void applyPhysics(float dt) {
        float searchRadius = std::max(config.groundSize, config.groundSize * 2.0f) * 2.0f;
        float boundMinX = -config.groundSize / 2.0f;
        float boundMaxX =  config.groundSize / 2.0f;
        float boundMinZ = -config.groundSize / 2.0f;
        float boundMaxZ =  config.groundSize / 2.0f;
        float boundMinY = -config.groundSize - 10.0f; 
        float boundMaxY =  config.groundSize * 2.0f; 
        float floorY = 0.0f;
        v3 gravity(0, -9.8f, 0);

        auto dirtNodes = grid.findInRadius(v3(0,0,0), searchRadius, 0);
        float totalEvaporation = 0.0f;
        if (!dirtNodes.empty()) {
            #pragma omp parallel
            {
                float localEvap = 0.0f;
                #pragma omp for
                for (size_t i = 0; i < dirtNodes.size(); ++i) {
                    auto& node = dirtNodes[i];
                    auto dp = std::static_pointer_cast<DirtParticle>(node->data);
                    
                    // Depth logic (Layer 0 is at -config.voxelSize/2)
                    float depth = std::abs(node->position.y() + (config.voxelSize / 2.0f));
                    float depthFactor = std::max(0.0f, 1.0f - (depth / (config.voxelSize * 4.0f)));
                    
                    // 1. Local Temperature
                    float targetTemp = currentTemperature;
                    if (depthFactor > 0.5f && cloudCover < 0.5f && currentSunPos.y() > 0.0f) { // Exposed Surface
                        float sunHeat = std::max(0.0f, currentSunPos.normalized().y()) * 15.0f * (1.0f - cloudCover);
                        targetTemp += sunHeat;
                    } else {
                        // Ground regulates to a cool average under the surface
                        targetTemp = 15.0f + (currentTemperature - 15.0f) * 0.2f;
                    }
                    dp->temperature += (targetTemp - dp->temperature) * dt * 0.05f;
                    
                    // 2. Evaporation
                    if (dp->temperature > 20.0f && dp->hydration > 0.0f && depthFactor > 0.1f) {
                        float evap = (dp->temperature - 20.0f) * 0.1f * dt * depthFactor;
                        evap = std::min(evap, dp->hydration);
                        dp->hydration -= evap;
                        localEvap += evap;
                    }
                    
                    // 3. Water Table Reduction (Flow downwards)
                    if (dp->hydration > 1.0f) {
                        v3 belowPos = node->position - v3(0, config.voxelSize, 0);
                        auto lowerNodes = grid.findInRadius(belowPos, config.voxelSize * 0.2f, 0);
                        for (auto& lowerNode : lowerNodes) {
                            if (lowerNode->position.y() < node->position.y() - config.voxelSize * 0.5f) {
                                auto lowerDp = std::static_pointer_cast<DirtParticle>(lowerNode->data);
                                if (dp->hydration > lowerDp->hydration) {
                                    float flow = (dp->hydration - lowerDp->hydration) * 0.5f * dt;
                                    float transfer = std::min(flow, dp->hydration);
                                    #pragma omp atomic
                                    dp->hydration -= transfer;
                                    #pragma omp atomic
                                    lowerDp->hydration += transfer;
                                }
                                break; 
                            }
                        }
                    }
                    
                    float wetness = std::clamp(dp->hydration / 300.0f, 0.0f, 1.0f);
                    float dryHeat = std::clamp((dp->temperature - 25.0f) / 15.0f, 0.0f, 1.0f);
                    int hIdx = std::clamp(static_cast<int>(dryHeat * 10.0f), 0, 10);
                    int wIdx = std::clamp(static_cast<int>(wetness * 10.0f), 0, 10);
                    node->colorIdx = dirtColorPalette[hIdx * 11 + wIdx];
                }
                #pragma omp atomic
                totalEvaporation += localEvap;
            }
            
            // Desertification Logic
            atmosphericMoisture += totalEvaporation;
            if (currentTemperature > 28.0f && cloudCover < 0.3f) {
                extendedHeatTimer += dt;
            } else if (currentTemperature < 22.0f) {
                extendedHeatTimer = std::max(0.0f, extendedHeatTimer - dt * 2.0f);
            }
            
            if (extendedHeatTimer > 3.0f * config.dayDuration) {
                // Persistent heat scorches wind out of atmospheric moisture
                atmosphericMoisture = std::max(0.0f, atmosphericMoisture - (atmosphericMoisture * 0.05f * dt));
            }
        }

        updatePlants(dt); 

        auto plantNodes = grid.findInRadius(v3(0,0,0), searchRadius, 1);
        if (!plantNodes.empty()) {
            for (auto& node : plantNodes) node->data->anchored = false;

            std::queue<std::pair<std::shared_ptr<Octree<std::shared_ptr<PlantsimParticle>>::NodeData>, float>> q;
            float floorY = 0.0f; 
            float tol = config.voxelSize * 0.1f;

            for (auto& node : plantNodes) {
                if (node->position.y() <= floorY + config.voxelSize / 2.0f + tol || std::static_pointer_cast<PlantParticle>(node->data)->part == PlantPart::ROOT) {
                    node->data->anchored = true;
                    q.push({node, 0.0f});
                }
            }

            while (!q.empty()) {
                auto currPair = q.front();
                auto curr = currPair.first;
                float stress = currPair.second;
                q.pop();

                auto neighbors = grid.findInRadius(curr->position, config.voxelSize * 1.5f, 1);
                for (auto& neighbor : neighbors) {
                    if (!neighbor->data->anchored && neighbor->objectId == 1) {
                        v3 diff = neighbor->position - curr->position;
                        float addedStress = (std::abs(diff.x()) > tol || std::abs(diff.z()) > tol) ? 2.5f : 1.0f;
                        float newStress = stress + addedStress;

                        if (newStress <= neighbor->data->strength) {
                            neighbor->data->anchored = true;
                            q.push({neighbor, newStress});
                        }
                    }
                }
            }


            std::vector<v3> toRemove;
            std::vector<std::pair<v3, v3>> toMove;

            #pragma omp parallel
            {
                std::vector<v3> localRemove;
                std::vector<std::pair<v3, v3>> localMove;

                #pragma omp for
                for (size_t i = 0; i < plantNodes.size(); ++i) {
                    auto& node = plantNodes[i];
                    auto p = node->data;
                    if (p->anchored) {
                        p->velocity = v3(0,0,0);
                        p->timeOutOfBounds = 0.0f;
                        continue;
                    }

                    p->velocity += gravity * dt;
                    v3 newPos = node->position + p->velocity * dt;

                    if (newPos.y() < boundMinY) {
                        localRemove.push_back(node->position);
                        continue;
                    }

                    if (newPos.y() - config.voxelSize / 2.0f <= floorY) {
                        newPos.y() = floorY + config.voxelSize / 2.0f;
                        p->velocity = v3(0,0,0);
                    }

                    localMove.push_back({node->position, newPos});
                }

                #pragma omp critical
                {
                    toRemove.insert(toRemove.end(), localRemove.begin(), localRemove.end());
                    toMove.insert(toMove.end(), localMove.begin(), localMove.end());
                }
            }

            for (auto& pos : toRemove) grid.remove(pos);
            for (auto& movePair : toMove) grid.move(movePair.first, movePair.second);
        }

        std::vector<v3> dirtToHydrate;
        auto rainNodes = grid.findInRadius(v3(0,0,0), searchRadius, 3);
        if (!rainNodes.empty()) {
            float rainRadius = config.voxelSize * 0.1f;

            std::vector<v3> rainToRemove;
            std::vector<std::pair<v3, v3>> rainToMove;

            #pragma omp parallel
            {
                std::vector<v3> localRemove;
                std::vector<std::pair<v3, v3>> localMove;
                std::vector<v3> localHydrate;

                static thread_local std::mt19937 local_rng(std::random_device{}());
                std::uniform_real_distribution<float> bounceDist(-1.5f, 1.5f);
                std::uniform_real_distribution<float> probDist(0.0f, 1.0f);

                #pragma omp for
                for (size_t i = 0; i < rainNodes.size(); ++i) {
                    auto& node = rainNodes[i];
                    auto p = node->data;
                    p->velocity += gravity * dt;
                    v3 newPos = node->position + p->velocity * dt;
                    
                    bool overDirt = (newPos.x() >= boundMinX && newPos.x() <= boundMaxX &&
                                     newPos.z() >= boundMinZ && newPos.z() <= boundMaxZ);

                    if (overDirt && newPos.y() - rainRadius <= floorY) {
                        newPos.y() = floorY + rainRadius;

                        // Bounce logic
                        if (p->velocity.y() < -1.5f) {
                            p->velocity.y() = -p->velocity.y() * 0.3f;
                            p->velocity.x() += bounceDist(local_rng);
                            p->velocity.z() += bounceDist(local_rng);
                        } else {
                            p->velocity.y() = 0.0f;
                            v3 centerDir = newPos; centerDir.y() = 0.0f;
                            if (centerDir.norm() > 0.1f) {
                                p->velocity += centerDir.normalized() * (dt * 3.0f);
                            }
                            p->velocity.x() *= 0.95f;
                            p->velocity.z() *= 0.95f;
                        }

                        if (probDist(local_rng) < dt * 1.5f) { 
                            localRemove.push_back(node->position);
                            localHydrate.push_back(newPos);
                            continue;
                        }
                    }

                    if (newPos.y() < boundMinY) {
                        localRemove.push_back(node->position);
                        continue;
                    }

                    localMove.push_back({node->position, newPos});
                }

                #pragma omp critical
                {
                    rainToRemove.insert(rainToRemove.end(), localRemove.begin(), localRemove.end());
                    rainToMove.insert(rainToMove.end(), localMove.begin(), localMove.end());
                    dirtToHydrate.insert(dirtToHydrate.end(), localHydrate.begin(), localHydrate.end());
                }
            }

            for (auto& pos : rainToRemove) grid.remove(pos);
            for (auto& movePair : rainToMove) grid.move(movePair.first, movePair.second);
        }
            
        auto snowNodes = grid.findInRadius(v3(0,0,0), searchRadius, 4);
        if (!snowNodes.empty()) {
            float snowRadius = config.voxelSize * 0.15f;
            std::vector<v3> snowToRemove;
            std::vector<std::pair<v3, v3>> snowToMove;

            #pragma omp parallel
            {
                std::vector<v3> localRemove;
                std::vector<std::pair<v3, v3>> localMove;
                std::vector<v3> localHydrate;

                #pragma omp for
                for (size_t i = 0; i < snowNodes.size(); ++i) {
                    auto& node = snowNodes[i];
                    auto p = std::static_pointer_cast<SnowParticle>(node->data);
                    if (!p->anchored) {
                        p->velocity += gravity * dt * 0.15f; // Falls much slower
                        if (p->velocity.y() < -1.5f) p->velocity.y() = -1.5f; // Terminal velocity
                        v3 newPos = node->position + p->velocity * dt;
                        
                        bool overDirt = (newPos.x() >= boundMinX && newPos.x() <= boundMaxX &&
                                         newPos.z() >= boundMinZ && newPos.z() <= boundMaxZ);

                        if (overDirt && newPos.y() - snowRadius <= floorY + 0.02f) { 
                            newPos.y() = floorY + snowRadius;
                            p->anchored = true;
                            p->velocity = v3(0,0,0);
                        } else if (newPos.y() < boundMinY) {
                            localRemove.push_back(node->position);
                            continue;
                        }

                        localMove.push_back({node->position, newPos});
                    } else {
                        // Melting logic for snow sitting on ground
                        if (currentTemperature > 0.0f) {
                            p->meltProgress += currentTemperature * dt;
                            if (p->meltProgress > 15.0f) { 
                                localRemove.push_back(node->position);
                                localHydrate.push_back(node->position);
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    snowToRemove.insert(snowToRemove.end(), localRemove.begin(), localRemove.end());
                    snowToMove.insert(snowToMove.end(), localMove.begin(), localMove.end());
                    dirtToHydrate.insert(dirtToHydrate.end(), localHydrate.begin(), localHydrate.end());
                }
            }
            
            for (auto& pos : snowToRemove) grid.remove(pos);
            for (auto& movePair : snowToMove) grid.move(movePair.first, movePair.second);
        }
           
        if (!dirtToHydrate.empty()) {
            #pragma omp parallel for
            for (size_t i = 0; i < dirtToHydrate.size(); ++i) {
                auto localDirtNodes = grid.findInRadius(dirtToHydrate[i], config.voxelSize * 1.5f, 0);
                float highestY = -9999.0f;
                std::shared_ptr<DirtParticle> highestDirt = nullptr;

                for (auto& dirtNode : localDirtNodes) {
                    if (dirtNode->position.y() > highestY) {
                        highestY = dirtNode->position.y();
                        highestDirt = std::static_pointer_cast<DirtParticle>(dirtNode->data);
                    }
                }

                if (highestDirt) {
                    #pragma omp atomic
                    highestDirt->hydration += 15.0f;
                }
            }
        }
        
        // Air Update
        auto airNodes = grid.findInRadius(v3(0,0,0), searchRadius, 5);
        if (!airNodes.empty()) {
            #pragma omp parallel for
            for (size_t i = 0; i < airNodes.size(); ++i) {
                auto p = std::static_pointer_cast<AirParticle>(airNodes[i]->data);
                // Slowly normalize air temperature to global temperature
                p->temperature += (currentTemperature - p->temperature) * dt * 0.01f;
            }
        }
    }

    void update(float dt) {
        if (!worldInitialized) return;

        // Advance Time
        if (config.dayDuration > 0.0f) {
            float timeStep = dt / config.dayDuration;
            config.timeOfDay += timeStep;
            while (config.timeOfDay >= 1.0f) {
                config.timeOfDay -= 1.0f;
                config.currentDay++;
                if (config.currentDay >= config.daysPerYear) {
                    config.currentDay = 0;
                }
            }
            config.season = (static_cast<float>(config.currentDay) + config.timeOfDay) / config.daysPerYear;
        }

        updateEnvironment(dt);
        
        hourlyTimer += dt;
        float hourlyThreshold = config.dayDuration / 24.0f;
        if (hourlyTimer >= hourlyThreshold) {
            updateWeather(hourlyTimer);
            hourlyTimer = std::fmod(hourlyTimer, hourlyThreshold);
        }

        if (currentWeather != WeatherState::CLEAR) {
            precipAccumulator += config.precipRate * dt;
            while (precipAccumulator >= 1.0f) {
                precipAccumulator -= 1.0f;
                spawnPrecipitation();
            }
        } else {
            precipAccumulator = 0.0f;
        }

        updateSunPosition(false);
        applyPhysics(dt);
    }

    void updateEnvironment(float dt) {
        float delta = config.axialTilt * M_PI / 180.0f * std::sin(config.season * 2.0f * M_PI);
        float phi = config.latitude * M_PI / 180.0f;
        float H = (config.timeOfDay - 0.5f) * 2.0f * M_PI;
        float sunHeight = std::sin(phi) * std::sin(delta) + std::cos(phi) * std::cos(delta) * std::cos(H);

        // Compute local temperature
        float baseTemp = 30.0f - (std::abs(config.latitude) / 90.0f) * 50.0f; 
        float seasonalSwing = (std::abs(config.latitude) / 90.0f) * 30.0f;
        float latSign = (config.latitude >= 0) ? 1.0f : -1.0f;
        float seasonOffset = (delta * 180.0f / M_PI) / 23.5f * latSign * seasonalSwing;
        float diurnalOffset = (sunHeight - 0.0f) * 10.0f; 
        currentTemperature = baseTemp + seasonOffset + diurnalOffset;

        float targetCloudCover = (currentWeather != WeatherState::CLEAR) ? 0.8f : 0.0f;
        cloudCover += (targetCloudCover - cloudCover) * dt * 0.1f;

        v3 nightColor(0.02f, 0.02f, 0.05f);
        v3 dawnColor(0.8f, 0.4f, 0.2f);
        v3 dayColor(0.53f, 0.81f, 0.92f);

        v3 currentBg;
        v3 currentLight;
        
        float overcastFactor = 1.0f - cloudCover * 0.5f;

        if (sunHeight > 0.2f) {
            currentBg = dayColor * overcastFactor;
            float brightness = (0.3f + (sunHeight * 0.7f)) * overcastFactor; 
            currentLight = v3(brightness, brightness, brightness);
        } else if (sunHeight > 0.0f) {
            float fade = sunHeight / 0.2f;
            currentBg = ((1.0f - fade) * dawnColor + fade * dayColor) * overcastFactor;
            float brightness = (0.1f + (fade * 0.2f)) * overcastFactor;
            currentLight = v3(brightness, brightness, brightness);
        } else if (sunHeight > -0.2f) {
            float fade = (sunHeight + 0.2f) / 0.2f; 
            currentBg = (1.0f - fade) * nightColor + fade * dawnColor * overcastFactor;
            currentLight = v3(0.02f, 0.02f, 0.05f) * (1.0f - fade) + v3(0.1f, 0.1f, 0.15f) * fade * overcastFactor;
        } else {
            currentBg = nightColor;
            currentLight = v3(0.02f, 0.02f, 0.05f);
        }

        grid.setBackgroundColor(currentBg);
        grid.setSkylight(currentLight);
    }

    void updateSunPosition(bool forceRebuild) {
        float delta = config.axialTilt * M_PI / 180.0f * std::sin(config.season * 2.0f * M_PI);
        float phi = config.latitude * M_PI / 180.0f;
        float H = (config.timeOfDay - 0.5f) * 2.0f * M_PI;
        
        float x = -std::cos(delta) * std::sin(H);
        float y = std::sin(phi) * std::sin(delta) + std::cos(phi) * std::cos(delta) * std::cos(H);
        float z = std::sin(phi) * std::cos(delta) * std::cos(H) - std::cos(phi) * std::sin(delta);

        v3 newSunPos = v3(x, y, z).normalized() * config.sunDistance;

        if (!forceRebuild && (newSunPos - currentSunPos).norm() < (config.sunSize * 0.1f)) {
            return; 
        }

        v3 deltaMove = newSunPos - currentSunPos;
        currentSunPos = newSunPos;

        if (sunExists && !forceRebuild) {
            if (grid.moveObject(2, deltaMove)) {
                return;
            } else {
                sunExists = false; 
            }
        }
        
        if (!sunExists || forceRebuild) {
            int count = 25; 
            float step = config.sunSize / count;
            float offset = config.sunSize / 2.0f;

            v3 forward = -currentSunPos.normalized();
            v3 right = v3(0,1,0).cross(forward).normalized();
            v3 up = forward.cross(right).normalized();

            float t = config.timeOfDay;
            v3 sunColor(1.0f, 0.95f, 0.8f);
            if (t > 0.7f || t < 0.3f) sunColor = v3(1.0f, 0.4f, 0.1f);

            for(int i=0; i<count; ++i) {
                for(int j=0; j<count; ++j) {
                    float u = (i * step) - offset;
                    float v = (j * step) - offset;
                    v3 pos = currentSunPos + (right * u) + (up * v);
                    
                    grid.set(std::make_shared<SunParticle>(), pos, true, sunColor, step, true, 2, 0, config.sunIntensity, 0.0f, 0.0f, 1.0f, 1.0f);
                }
            }
            sunExists = true;
        }
    }

    void grow() {
        std::cout << "Simulating growth step at day " << config.currentDay + 1 << ", time " << config.timeOfDay << std::endl;
    }
};

#endif