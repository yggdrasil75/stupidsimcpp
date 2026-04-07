#ifndef PLANT_CPP
#define PLANT_CPP

#include "../util/sim/plant.hpp"
#include "../util/grid/camera.hpp"

// Assuming ImGui headers are available via ptest.cpp or similar include paths
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
    int reflectCount = 2;
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
        sim.update(deltaTime);
        
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
            
            ImGui::Separator();
            ImGui::Text("Atmospheric Moisture: %.1f", sim.atmosphericMoisture);
        }
        
        if (ImGui::CollapsingHeader("Plant Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BulletText("Active Meristems (Buds): %zu", sim.activeMeristems.size());
            ImGui::BulletText("Leaves: %d", sim.leafCount);
            ImGui::BulletText("Roots: %d", sim.rootCount);
            ImGui::Text("Total Organism Resources:");
            
            float eCap = std::max(100.0f, static_cast<float>(sim.leafCount + sim.rootCount) * 20.0f);
            ImGui::ProgressBar(std::min(sim.totalPlantEnergy / eCap, 1.0f), ImVec2(-1, 0), 
                ("Energy: " + std::to_string((int)sim.totalPlantEnergy)).c_str());
            ImGui::ProgressBar(std::min(sim.totalPlantWater / eCap, 1.0f), ImVec2(-1, 0), 
                ("Water: " + std::to_string((int)sim.totalPlantWater)).c_str());
        }
            
        // if (ImGui::CollapsingHeader("Soil Environment"), ImGuiTreeNodeFlags_DefaultClosed) {
        //     float currentHydration = 0.0f;
        //     float currentTemp = 0.0f;
        //     auto dirtNodes = sim.grid.findInRadius(v3(0, -sim.config.voxelSize / 2.0f, 0), sim.config.voxelSize, 0);
        //     if (!dirtNodes.empty()) {
        //         auto dp = std::static_pointer_cast<DirtParticle>(dirtNodes[0]->data);
        //         if(dp) {
        //             currentHydration = dp->hydration;
        //             currentTemp = dp->temperature;
        //         }
        //     }
        //     ImGui::Text("Soil Temp (Center): %.1f °C", currentTemp);
        //     ImGui::ProgressBar(std::min(currentHydration / 500.0f, 1.0f), ImVec2(-1, 0), 
        //         ("Water (Center): " + std::to_string((int)currentHydration)).c_str());
        // }

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
            ImGui::ColorEdit3("Sun Color", sim.config.sunColor.data());
            ImGui::DragFloat("Sun Intensity", &sim.config.sunIntensity, 0.1f, 0.0f, 200.0f);
        }

        if (ImGui::CollapsingHeader("Render Settings")) {
            ImGui::Checkbox("High Quality Render (Slower)", &slowRender);
            if (slowRender) {
                ImGui::Checkbox("Global Illumination", &globalIllumination);
                ImGui::DragInt("Bounces", &reflectCount, 1, 0, 8);
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
            ImGui::SliderFloat("Max Girth", &currentDesignerDNA->stem.maxGirth, 0.1f, 10.0f);
            ImGui::SliderInt("Max Branch Depth", &currentDesignerDNA->stem.maxBranchDepth, 0, 6);
            ImGui::SliderFloat("Apical Dominance", &currentDesignerDNA->stem.apicalDominance, 0.0f, 1.0f);
            ImGui::SliderFloat("Branch Angle", &currentDesignerDNA->stem.branchAngle, 0.1f, 3.14f);
            ImGui::SliderFloat("Gravitropism", &currentDesignerDNA->stem.gravitropism, -1.0f, 1.0f);
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
            ImGui::SliderFloat("Horizontal Drive", &currentDesignerDNA->root.horizontalDrive, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Metabolism & Resistance")) {
            ImGui::SliderFloat("Optimal Temp", &currentDesignerDNA->optimalTemp, 0.0f, 40.0f);
            ImGui::SliderFloat("Temp Tolerance", &currentDesignerDNA->tempTolerance, 5.0f, 30.0f);
            ImGui::SliderFloat("Photosynthesis Eff.", &currentDesignerDNA->photosynthesisEfficiency, 0.1f, 5.0f);
            ImGui::SliderFloat("Water Absorption", &currentDesignerDNA->waterAbsorptionRate, 1.0f, 20.0f);
            ImGui::SliderFloat("Growth Cost (Energy)", &currentDesignerDNA->growthCostEnergy, 1.0f, 50.0f);
            ImGui::SliderFloat("Growth Cost (Water)", &currentDesignerDNA->growthCostWater, 1.0f, 50.0f);
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
        ImGui::SliderFloat("Ground Size", &sim.config.groundSize, 10.0f, 100.0f);
        
        if (ImGui::Button("Rebuild World (Keep Seeds)", ImVec2(-1, 0))) {
            sim.initWorld();
        }
        
        if (ImGui::Button("Rebuild World (Clear All)", ImVec2(-1, 0))) {// Init the simulation
            sim.initWorld();
            v3 bg = v3(0.511f, 0.625f, 0.868f);
            sim.grid.setBackgroundColor(bg);
            sim.grid.setSkylight(bg);

            currentDesignerDNA = std::make_shared<PlantDNA>();
            std::strncpy(nameBuffer, currentDesignerDNA->speciesName.c_str(), sizeof(nameBuffer));
            speciesLibrary.push_back(std::make_shared<PlantDNA>(*currentDesignerDNA));
            selectedSpeciesIndex = 0;
            sim.grid.remove(v3(0.0f, 0.0f, 0.0f), sim.config.voxelSize);
            sim.activeMeristems.clear();
            sim.activeRoots.clear();
            sim.activeLeaves.clear();
            sim.activeFlowers.clear();
            sim.leafCount = 0;
            sim.rootCount = 0;
        }

        ImGui::Separator();
        ImGui::Text("Evolution Parameters");
        ImGui::SliderFloat("Base Mutation Rate", &sim.config.baseMutationRate, 0.0f, 0.5f);
        ImGui::SliderFloat("Somatic Mutation Rate", &sim.config.somaticMutationRate, 0.0f, 0.1f);
    }

    void spawnSeed(v3 pos, std::shared_ptr<PlantDNA> dna) {
        auto seed = std::make_shared<PlantParticle>(PlantPart::SEED, dna, pos, v3(0.0f, 1.0f, 0.0f), 0);
        if (sim.grid.set(seed, pos, true, v3(0.2f, 0.8f, 0.2f), sim.config.voxelSize, true, 1)) {
            sim.activeMeristems.push_back(pos);
            sim.seeds.push_back(pos);
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
        
        // Render
        // if (slowRender) {
            // currentPreviewFrame = sim.grid.renderFrame(cam, outHeight, outWidth, frame::colormap::RGB, 10, reflectCount, globalIllumination, useLod);
        // } else {
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, outHeight, outWidth, frame::colormap::RGB);
        // }

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