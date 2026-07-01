#ifndef PLANT_CPP
#define PLANT_CPP

#include "../util/sim/plant.hpp"
#include "../util/grid/camera.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <cstring>
#include <string>
#include <chrono>
#include <algorithm>

class PlantSimUI {
private:
    PlantSim sim;
    Camera cam;
    
    // Rendering / Texture vars
    GLuint textu = 0;
    std::mutex PreviewMutex;
    bool textureInitialized = false;
    frame currentPreviewFrame;
    
    // Render Settings
    int outWidth = 512;
    int outHeight = 512;
    bool slowRender = false;
    bool globalIllumination = true;
    float deltaTime = 0.016f;
    int simulationSpeed = 1;
    int reflectCount = 2;
    int spp = 5;
    float maxDist = 200.0f;
    bool keyStates[GLFW_KEY_LAST + 1] = {false};

    // Dynamic Distance & LOD Settings
    bool useLod = true;
    bool autoAdjustDistance = true;
    int targetMemoryMB = 1024;
    float minAutoDistance = 50.0f;
    float maxAutoDistance = 2000.0f;
    float lodDist = 100.0f;
    float lodDropoff = 0.05f;
    size_t currentMemoryMB = 0;
    size_t currentLoadedPoints = 0;
    std::chrono::steady_clock::time_point lastAutoAdjustTime = std::chrono::steady_clock::now();

    std::shared_ptr<PlantDNA> currentDesignerDNA;
    std::vector<std::shared_ptr<PlantDNA>> speciesLibrary;
    int selectedSpeciesIndex = -1;
    char nameBuffer[128] = "Emergent Plant";

    const char* getSeasonName(float season, float latitude) {
        bool north = latitude >= 0;
        if (season < 0.25f) return north ? "Spring" : "Autumn";
        if (season < 0.50f) return north ? "Summer" : "Winter";
        if (season < 0.75f) return north ? "Autumn" : "Spring";
        return north ? "Winter" : "Summer";
    }

    const char* getWeatherName(PlantSim::WeatherState state) {
        switch(state) {
            case PlantSim::WeatherState::RAIN: return "Raining";
            case PlantSim::WeatherState::SNOW: return "Snowing";
            default: return "Clear";
        }
    }
public:
    PlantSimUI() {
        cam.origin = v3(0, 5, 30);
        cam.lookAt(v3(0, 2, 0));
        cam.fov = 45;
    }

    ~PlantSimUI() {
        if (textu != 0) glDeleteTextures(1, &textu);
    }

    void renderUI(GLFWwindow* window) {
        handleCameraControls(window);

        ImGui::Begin("Plant Simulation Laboratory", nullptr, ImGuiWindowFlags_NoCollapse);
        
        if (ImGui::BeginTable("PlantLayout", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthStretch, 0.35f);
            ImGui::TableSetupColumn("Viewport", ImGuiTableColumnFlags_WidthStretch, 0.65f);
            ImGui::TableNextColumn();
            
            if (ImGui::BeginTabBar("ControlTabs")) {
                if (ImGui::BeginTabItem("Simulation")) {
                    renderControls();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Genetics")) {
                    renderGeneticsDesigner();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Species Library")) {
                    renderSpeciesLibrary();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("World Creation")) {
                    renderWorldControls();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            
            ImGui::TableNextColumn();
            
            renderPreview();
            
            ImGui::EndTable();
        }

        for(int i = 0; i < simulationSpeed; ++i) {
            sim.update(deltaTime);
        }
        
        ImGui::End();
        sim.grid.waitForIdle();
    }

private:
    void handleCameraControls(GLFWwindow* window) {
        glfwPollEvents();
        for (int i = GLFW_KEY_SPACE; i <= GLFW_KEY_LAST; i++) {
            keyStates[i] = (glfwGetKey(window, i) == GLFW_PRESS);
        }
        
        float speed = 15.0f * deltaTime;
        if (keyStates[GLFW_KEY_W]) cam.moveForward(speed);
        if (keyStates[GLFW_KEY_S]) cam.moveBackward(speed);
        if (keyStates[GLFW_KEY_A]) cam.moveLeft(speed);
        if (keyStates[GLFW_KEY_D]) cam.moveRight(speed);
        if (keyStates[GLFW_KEY_Q]) cam.rotateYaw(-deltaTime * 0.5f);
        if (keyStates[GLFW_KEY_E]) cam.rotateYaw(deltaTime * 0.5f);
        if (keyStates[GLFW_KEY_SPACE]) cam.moveUp(speed);
        if (keyStates[GLFW_KEY_LEFT_SHIFT]) cam.moveUp(-speed);
    }

    void renderControls() {
        if (ImGui::CollapsingHeader("World State", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("Simulation Speed Multiplier", &simulationSpeed, 1, 100);
            ImGui::SliderFloat("Plant Growth Speed", &sim.config.growthSpeedMultiplier, 0.1f, 100.0f);

            ImGui::Text("Day: %d / %d", sim.config.currentDay + 1, sim.config.daysPerYear);
            ImGui::Text("Season: %s", getSeasonName(sim.config.season, sim.config.latitude));
            ImGui::Text("Global Temperature: %.1f °C", sim.currentTemperature);
            
            ImVec4 weatherColor = ImVec4(1, 1, 1, 1);
            if (sim.currentWeather == PlantSim::WeatherState::RAIN) weatherColor = ImVec4(0.3f, 0.5f, 1.0f, 1.0f);
            else if (sim.currentWeather == PlantSim::WeatherState::SNOW) weatherColor = ImVec4(0.8f, 0.9f, 1.0f, 1.0f);
            
            ImGui::TextColored(weatherColor, "Current Weather: %s", getWeatherName(sim.currentWeather));
            if (sim.weatherTimer > 0.0f) {
                ImGui::TextColored(weatherColor, "(Time Remaining: %.1fs)", sim.weatherTimer);
            }
            ImGui::BulletText("Active Water Voxels: %zu", sim.getActiveWaterVoxelCount());
            
            ImGui::Separator();
            ImGui::Text("Atmospheric Moisture: %.1f", sim.atmosphericMoisture);
        }
        
        if (ImGui::CollapsingHeader("Plant Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BulletText("Active Plants: %zu", sim.plantStates.size());
            ImGui::BulletText("Active Meristems (Buds): %zu", sim.activeMeristems.size());
            
            int totalLeaves = 0;
            int totalRoots = 0;
            float avgEnergy = 0.0f;
            float avgWater = 0.0f;
            for (auto& pair : sim.plantStates) {
                totalLeaves += pair.second->leafCount;
                totalRoots += pair.second->rootCount;
                avgEnergy += pair.second->energy;
                avgWater += pair.second->water;
            }
            if (!sim.plantStates.empty()) {
                avgEnergy /= sim.plantStates.size();
                avgWater /= sim.plantStates.size();
        }
            
            ImGui::BulletText("Total Leaves: %d", totalLeaves);
            ImGui::BulletText("Total Roots: %d", totalRoots);
            
            float eCap = std::max(100.0f, static_cast<float>(totalLeaves + totalRoots) * 20.0f);
            if (sim.plantStates.size() > 0) eCap /= sim.plantStates.size();

            ImGui::ProgressBar(std::min(avgEnergy / eCap, 1.0f), ImVec2(-1, 0), 
                ("Avg Energy: " + std::to_string((int)avgEnergy)).c_str());
            ImGui::ProgressBar(std::min(avgWater / eCap, 1.0f), ImVec2(-1, 0), 
                ("Avg Water: " + std::to_string((int)avgWater)).c_str());
        }

        if (ImGui::CollapsingHeader("Sun & Seasons", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Time of Day", &sim.config.timeOfDay, 0.0f, 1.0f);
            if (ImGui::SliderInt("Current Day", &sim.config.currentDay, 0, sim.config.daysPerYear - 1)) {
                sim.config.season = (static_cast<float>(sim.config.currentDay) + sim.config.timeOfDay) / sim.config.daysPerYear;
            }
            ImGui::SliderFloat("Day Duration (s)", &sim.config.dayDuration, 1.0f, 600.0f);
            if (ImGui::SliderInt("Days per Year", &sim.config.daysPerYear, 4, 365)) {
                if (sim.config.currentDay >= sim.config.daysPerYear) sim.config.currentDay = 0;
            }
            
            ImGui::Separator();
            ImGui::SliderFloat("Latitude", &sim.config.latitude, -90.0f, 90.0f);
            ImGui::SliderFloat("Axial Tilt", &sim.config.axialTilt, 0.0f, 90.0f);
            
            ImGui::Separator();
            if (ImGui::ColorEdit3("Sun Color", sim.config.sunColor.data())) sim.updateSkyBodies();
            ImGui::DragFloat("Sun Intensity", &sim.config.sunIntensity, 0.1f, 0.0f, 200.0f);
            if (ImGui::DragFloat("Sun Size", &sim.config.sunSize, 0.1f, 1.0f, 90.0f)) sim.updateSkyBodies();

            ImGui::Separator();
            if (ImGui::ColorEdit3("Moon Color", sim.config.moonColor.data())) sim.updateSkyBodies();
            if (ImGui::DragFloat("Moon Size", &sim.config.moonSize, 0.1f, 1.0f, 90.0f)) sim.updateSkyBodies();
            
            ImGui::Separator();
            ImGui::DragFloat("Precipitation Rate", &sim.config.precipRate, 1.0f, 0.0f, 1000.0f);
        }

        if (ImGui::CollapsingHeader("Render Settings")) {
            ImGui::Checkbox("High Quality Render (Slower)", &slowRender);
            if (slowRender) {
                ImGui::Checkbox("Global Illumination", &globalIllumination);
                ImGui::DragInt("Bounces", &reflectCount, 1, 1, 8);
                ImGui::DragInt("Samples Per Pixel", &spp, 1, 2, 128);
            }
            
            ImGui::Separator();
            ImGui::Text("Streaming & LOD Settings");
            ImGui::Checkbox("Use LOD", &useLod);
            ImGui::Checkbox("Auto-Adjust View Distance", &autoAdjustDistance);
            
            if (autoAdjustDistance) {
                ImGui::InputInt("Target Memory Budget (MB)", &targetMemoryMB);
                ImGui::DragFloat("Min Auto Distance", &minAutoDistance, 10.0f, 10.0f, maxAutoDistance);
                ImGui::DragFloat("Max Auto Distance", &maxAutoDistance, 10.0f, minAutoDistance, 65536.0f);
                
                ImGui::BeginDisabled();
                ImGui::DragFloat("Render Distance", &maxDist, 10.0f, 10.0f, 65536.0f);
                ImGui::DragFloat("LOD Distance", &lodDist, 10.0f, 10.0f, 5000.0f);
                ImGui::EndDisabled();
            } else {
                ImGui::DragFloat("Render Distance", &maxDist, 10.0f, 10.0f, 65536.0f);
                ImGui::DragFloat("LOD Distance", &lodDist, 10.0f, 10.0f, 5000.0f);
            }
            ImGui::DragFloat("LOD Dropoff", &lodDropoff, 0.01f, 0.01f, 1.0f);
            
            ImGui::Separator();
            ImGui::Text("Est. Memory Usage: %zu MB", currentMemoryMB);
            ImGui::Text("Currently Loaded Points: %zu", currentLoadedPoints);
        }
    }

    void renderGeneticsDesigner() {
        ImGui::Text("Species Designer");
        ImGui::InputText("Species Name", nameBuffer, sizeof(nameBuffer));
        if (ImGui::Button("Save to Library")) {
            currentDesignerDNA->speciesName = std::string(nameBuffer);
            speciesLibrary.push_back(std::make_shared<PlantDNA>(*currentDesignerDNA));
            selectedSpeciesIndex = static_cast<int>(speciesLibrary.size()) - 1;
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("Stem Morphology")) {
            ImGui::SliderFloat("Woodiness", &currentDesignerDNA->stem.woodiness, 0.0f, 1.0f);
            ImGui::SliderFloat("Flexibility", &currentDesignerDNA->stem.flexibility, 0.0f, 1.0f);
            ImGui::SliderFloat("Max Height", &currentDesignerDNA->stem.maxHeight, 1.0f, 100.0f);
            ImGui::SliderFloat("Max Girth", &currentDesignerDNA->stem.maxGirth, 0.01f, 10.0f);
            ImGui::SliderInt("Max Branch Depth", &currentDesignerDNA->stem.maxBranchDepth, 0, 6);
            ImGui::SliderFloat("Apical Dominance", &currentDesignerDNA->stem.apicalDominance, 0.0f, 1.0f);
            
            ImGui::Text("Tropisms (Behavioral growth)");
            ImGui::SliderFloat("Gravitropism", &currentDesignerDNA->stem.gravitropism, -1.0f, 1.0f, "%.2f ( -1=Up )");
            ImGui::SliderFloat("Phototropism", &currentDesignerDNA->stem.phototropism, 0.0f, 1.0f, "%.2f ( Seeks Light )");
            ImGui::SliderFloat("Skototropism", &currentDesignerDNA->stem.skototropism, 0.0f, 1.0f, "%.2f ( Seeks Shade )");
            ImGui::SliderFloat("Thigmotropism", &currentDesignerDNA->stem.thigmotropism, 0.0f, 1.0f, "%.2f ( Climbs Solids )");
            ImGui::SliderFloat("Aerotropism", &currentDesignerDNA->stem.aerotropism, 0.0f, 1.0f, "%.2f ( Up from Water )");
            ImGui::ColorEdit3("Bark Color", currentDesignerDNA->stem.barkColor.data());
        }

        if (ImGui::CollapsingHeader("Leaf Genetics")) {
            ImGui::SliderFloat("Leaf Density", &currentDesignerDNA->leaf.leafDensity, 0.0f, 1.0f);
            ImGui::SliderFloat("Length Multiplier", &currentDesignerDNA->leaf.lengthMultiplier, 0.1f, 5.0f);
            ImGui::SliderFloat("Width Multiplier", &currentDesignerDNA->leaf.widthMultiplier, 0.1f, 5.0f);
            ImGui::Checkbox("Deciduous", &currentDesignerDNA->leaf.isDeciduous);
            ImGui::Checkbox("Evergreen", &currentDesignerDNA->leaf.evergreen);
            ImGui::ColorEdit3("Summer Color", currentDesignerDNA->leaf.color.data());
            ImGui::ColorEdit3("Autumn Color", currentDesignerDNA->leaf.autumnColor.data());
        }

        if (ImGui::CollapsingHeader("Root Genetics")) {
            ImGui::SliderFloat("Max Depth", &currentDesignerDNA->root.maxDepth, 1.0f, 50.0f);
            ImGui::SliderFloat("Spread Radius", &currentDesignerDNA->root.spreadRadius, 1.0f, 50.0f);
            ImGui::SliderFloat("Vertical Drive", &currentDesignerDNA->root.verticalDrive, 0.0f, 1.0f);
            ImGui::SliderFloat("Water Seeking (Hydrotropism)", &currentDesignerDNA->root.waterSeekStrength, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Metabolism & Resistance")) {
            ImGui::SliderFloat("Optimal Temp", &currentDesignerDNA->optimalTemp, 0.0f, 40.0f);
            ImGui::SliderFloat("Temp Tolerance", &currentDesignerDNA->tempTolerance, 5.0f, 30.0f);
            ImGui::SliderFloat("Photosynthesis Eff.", &currentDesignerDNA->photosynthesisEfficiency, 0.0f, 5.0f);
            ImGui::SliderFloat("Water Absorption", &currentDesignerDNA->waterAbsorptionRate, 1.0f, 20.0f);
            ImGui::SliderFloat("Fungal Saprotrophy", &currentDesignerDNA->special.fungalSaprotrophy, 0.0f, 1.0f, "%.2f ( Feeds on damp dirt )");
            ImGui::SliderFloat("Growth Cost (Energy)", &currentDesignerDNA->growthCostEnergy, 0.1f, 50.0f);
            ImGui::SliderFloat("Growth Cost (Water)", &currentDesignerDNA->growthCostWater, 0.1f, 50.0f);
        }
    }

    void renderSpeciesLibrary() {
        ImGui::Text("Stored Species");
        if (ImGui::BeginListBox("##species_list", ImVec2(-1, 200))) {
            for (int i = 0; i < static_cast<int>(speciesLibrary.size()); i++) {
                std::string label = speciesLibrary[i]->speciesName + " (Gen " + std::to_string(speciesLibrary[i]->generation) + ")";
                const bool is_selected = (selectedSpeciesIndex == i);
                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    selectedSpeciesIndex = i;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }

        if (selectedSpeciesIndex >= 0 && selectedSpeciesIndex < static_cast<int>(speciesLibrary.size())) {
            ImGui::Separator();
            std::shared_ptr<PlantDNA> selectedDNA = speciesLibrary[selectedSpeciesIndex];
            ImGui::Text("Selected: %s", selectedDNA->speciesName.c_str());
            
            if (ImGui::Button("Spawn Seed at Center", ImVec2(-1, 0))) {
                spawnSeed(v3(0, 0, 0), selectedDNA);
            }
            
            if (ImGui::Button("Spawn Seed at Random Soil", ImVec2(-1, 0))) {
                std::uniform_real_distribution<float> dist(-sim.config.groundSize + 1.0f, sim.config.groundSize - 1.0f);
                spawnSeed(v3(dist(sim.rng), 0, dist(sim.rng)), selectedDNA);
            }
            
            if (ImGui::Button("Load into Designer", ImVec2(-1, 0))) {
                *currentDesignerDNA = *selectedDNA;
                std::strncpy(nameBuffer, currentDesignerDNA->speciesName.c_str(), sizeof(nameBuffer));
            }
        }
    }

    void renderWorldControls() {
        ImGui::Text("Environment Generation");
        ImGui::SliderFloat("Voxel Size", &sim.config.voxelSize, 0.1f, 2.0f);
        ImGui::SliderFloat("Ground Size", &sim.config.groundSize, 10.0f, 150.0f);
        ImGui::SliderFloat("Water Level", &sim.config.waterLevel, -20.0f, 20.0f);
        
        if (ImGui::Button("Rebuild World (Keep Seeds)", ImVec2(-1, 0))) {
            sim.initWorld(false);
        }
        
        if (ImGui::Button("Rebuild World (Clear All)", ImVec2(-1, 0))) {
            sim.initWorld(true);

            currentDesignerDNA = std::make_shared<PlantDNA>();
            std::strncpy(nameBuffer, currentDesignerDNA->speciesName.c_str(), sizeof(nameBuffer));
            speciesLibrary.clear();
            speciesLibrary.push_back(std::make_shared<PlantDNA>(*currentDesignerDNA));
            selectedSpeciesIndex = 0;
            sim.activeMeristems.clear();
            sim.activeRoots.clear();
            sim.activeLeaves.clear();
            sim.activeFlowers.clear();
        }

        if (ImGui::Button("Generate Rich Ecosystem", ImVec2(-1, 0))) {
            generateEcosystem();
        }

        ImGui::Separator();
        ImGui::Text("Evolution Parameters");
        ImGui::SliderFloat("Base Mutation Rate", &sim.config.baseMutationRate, 0.0f, 0.5f);
        ImGui::SliderFloat("Somatic Mutation Rate", &sim.config.somaticMutationRate, 0.0f, 0.1f);
    }

    void generateEcosystem() {
        sim.initWorld(false); 
        
        sim.activeMeristems.clear();
        sim.activeRoots.clear();
        sim.activeLeaves.clear();
        sim.activeFlowers.clear();
        
        auto treeDNA = std::make_shared<PlantDNA>();
        treeDNA->speciesName = "Great Oak";
        treeDNA->stem.maxHeight = 25.0f;
        treeDNA->stem.maxGirth = 3.0f;
        treeDNA->stem.maxBranchDepth = 4;
        treeDNA->stem.apicalDominance = 0.85f;
        treeDNA->stem.phototropism = 0.8f;
        treeDNA->stem.barkColor = v3(0.3f, 0.18f, 0.1f);
        treeDNA->leaf.color = v3(0.15f, 0.45f, 0.15f);
        treeDNA->leaf.leafDensity = 0.6f;
        treeDNA->growthCostEnergy = 5.0f;
        treeDNA->growthCostWater = 5.0f;
        spawnSeed(v3(0, 0, 0), treeDNA);

        auto vineDNA = std::make_shared<PlantDNA>();
        vineDNA->speciesName = "Strangler Vine";
        vineDNA->stem.maxHeight = 35.0f;
        vineDNA->stem.maxGirth = 0.3f;
        vineDNA->stem.maxBranchDepth = 1;
        vineDNA->stem.apicalDominance = 0.95f;
        vineDNA->stem.thigmotropism = 1.0f;
        vineDNA->stem.skototropism = 0.5f; // Initially seeks dark tree trunks
        vineDNA->stem.gravitropism = -0.2f; // Weak upward push
        vineDNA->stem.barkColor = v3(0.1f, 0.5f, 0.2f);
        vineDNA->leaf.leafDensity = 0.3f;

        // 3. Shade Fern (Low-light loving floor plant)
        auto fernDNA = std::make_shared<PlantDNA>();
        fernDNA->speciesName = "Shade Fern";
        fernDNA->stem.maxHeight = 1.5f;
        fernDNA->stem.maxGirth = 0.2f;
        fernDNA->stem.maxBranchDepth = 2;
        fernDNA->stem.apicalDominance = 0.1f;
        fernDNA->stem.skototropism = 0.6f;
        fernDNA->leaf.color = v3(0.05f, 0.35f, 0.15f);
        fernDNA->leaf.leafDensity = 1.0f;
        fernDNA->leaf.widthMultiplier = 2.0f;

        // 4. River Kelp (Underwater Aerotropic Plant)
        auto kelpDNA = std::make_shared<PlantDNA>();
        kelpDNA->speciesName = "Abyssal Kelp";
        kelpDNA->stem.maxHeight = 30.0f;
        kelpDNA->stem.maxGirth = 0.5f;
        kelpDNA->stem.maxBranchDepth = 1;
        kelpDNA->stem.apicalDominance = 0.9f;
        kelpDNA->stem.gravitropism = 0.0f; // Neutral buoyancy
        kelpDNA->stem.aerotropism = 1.0f;  // Strong drive to the surface
        kelpDNA->stem.barkColor = v3(0.1f, 0.4f, 0.2f);
        kelpDNA->leaf.leafDensity = 0.4f;
        kelpDNA->waterAbsorptionRate = 20.0f; // Rapid water usage

        // 5. Cave Glowcap (Fungal saprotroph)
        auto shroomDNA = std::make_shared<PlantDNA>();
        shroomDNA->speciesName = "Cave Glowcap";
        shroomDNA->stem.maxHeight = 2.0f;
        shroomDNA->stem.maxGirth = 0.8f;
        shroomDNA->stem.maxBranchDepth = 0;
        shroomDNA->stem.phototropism = 0.0f;
        shroomDNA->stem.skototropism = 0.8f; // Strongly prefers dark ravines
        shroomDNA->stem.barkColor = v3(0.85f, 0.85f, 0.95f);
        shroomDNA->leaf.leafDensity = 0.0f;  // Fungi have no leaves
        shroomDNA->photosynthesisEfficiency = 0.0f;
        shroomDNA->special.fungalSaprotrophy = 0.8f; // Feeds off the dirt
        
        std::uniform_real_distribution<float> dist(-sim.config.groundSize + 2.0f, sim.config.groundSize - 2.0f);
        
        // Spawn Ecosystem Distribution
        for (int i = 0; i < 30; i++) {
            float rx = dist(sim.rng);
            float rz = dist(sim.rng);
            float ry = sim.getTerrainHeight(rx, rz);

            if (ry >= sim.config.waterLevel) {
                spawnSeed(v3(rx, ry, rz), treeDNA);
                
                // Spawn a vine directly next to the tree
                spawnSeed(v3(rx + sim.config.voxelSize, ry, rz), vineDNA);

                // Spawn some ferns around the tree base
                for(int j = 0; j < 3; j++) {
                    float fx = rx + (dist(sim.rng) * 0.1f);
                    float fz = rz + (dist(sim.rng) * 0.1f);
                    spawnSeed(v3(fx, sim.getTerrainHeight(fx, fz), fz), fernDNA);
                }
            } else {
                // Spawning underwater
                spawnSeed(v3(rx, ry, rz), kelpDNA);
            }
        }

        // Spawn Mushrooms in natural depressions/valleys
        for (int i = 0; i < 50; i++) {
            float rx = dist(sim.rng);
            float rz = dist(sim.rng);
            float ry = sim.getTerrainHeight(rx, rz);
            if (ry > sim.config.waterLevel && ry < 2.0f) { // Low but not underwater
                spawnSeed(v3(rx, ry, rz), shroomDNA);
            }
        }
        
        speciesLibrary.push_back(treeDNA);
        speciesLibrary.push_back(vineDNA);
        speciesLibrary.push_back(fernDNA);
        speciesLibrary.push_back(kelpDNA);
        speciesLibrary.push_back(shroomDNA);
    }

    void spawnSeed(v3 pos, std::shared_ptr<PlantDNA> dna) {
        float startSize = sim.config.plantVoxelSize;
        if (dna->stem.maxGirth < 1.0f) {
            startSize *= std::max(0.3f, dna->stem.maxGirth);
        }

        int pId = sim.nextPlantId++;
        auto state = std::make_shared<PlantState>();
        state->energy = 50.0f; // Give proper starter energy to sprout leaves
        state->water = 50.0f;
        sim.plantStates[pId] = state;

        auto seed = std::make_shared<PlantParticle>(PlantPart::SEED, dna, pos, v3(0.0f, 1.0f, 0.0f), 0);
        seed->plantId = pId;

        if (sim.grid.insert(seed, pos, true, v3(0.2f, 0.8f, 0.2f), startSize, true, 1)) {
            sim.activeMeristems.push_back(pos);
            sim.seeds.push_back(pos);
        } else {
            sim.plantStates.erase(pId);
        }
    }

    void renderPreview() {
        livePreview();

        if (textureInitialized) {
            float availW = ImGui::GetContentRegionAvail().x;
            float aspect = (float)outWidth / (float)outHeight;
            ImGui::Image((void*)(intptr_t)textu, ImVec2(availW, availW / aspect));
        }
    }

    void livePreview() {
        std::lock_guard<std::mutex> lock(PreviewMutex);
        
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoAdjustTime).count() > 1500) {
            lastAutoAdjustTime = now;
            
            currentMemoryMB = sim.grid.getEstimatedMemoryUsageMB();
            currentLoadedPoints = sim.grid.getLoadedPointCount();
            
            if (autoAdjustDistance) {
                if (currentMemoryMB > (size_t)targetMemoryMB) {
                    maxDist *= 0.85f;
                    lodDist *= 0.85f;
                } else if (currentMemoryMB < (size_t)targetMemoryMB * 0.8f) {
                    maxDist *= 1.05f;
                    lodDist *= 1.05f;
                }
                
                maxDist = std::clamp(maxDist, minAutoDistance, maxAutoDistance);
                lodDist = std::clamp(lodDist, 10.0f, maxDist * 0.5f);
            }
        }

        sim.grid.setLODMinDistance(lodDist);
        sim.grid.setLODFalloff(lodDropoff);
        sim.grid.setMaxDistance(maxDist);
        
        if (slowRender) {
            // #ifdef VULKAN_SUPPORT
            currentPreviewFrame = sim.grid.renderFrameVulkan(cam, outHeight, outWidth, frame::colormap::RGB, 10, reflectCount, useLod);
            // #else
            // currentPreviewFrame = sim.grid.renderFrame(cam, outHeight, outWidth, frame::colormap::RGB, 3, reflectCount, globalIllumination, useLod);
            // #endif
        } else {
            #ifdef VULKAN_SUPPORT
            currentPreviewFrame = sim.grid.fastRenderFrameVulkan(cam, outHeight, outWidth, frame::colormap::RGB);
            #else
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, outHeight, outWidth, frame::colormap::RGB);
            #endif
        }

        // Upload to GPU
        if (textu == 0) glGenTextures(1, &textu);
        
        glBindTexture(GL_TEXTURE_2D, textu);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                     0, GL_RGB, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
        
        textureInitialized = true;
    }
};

#endif