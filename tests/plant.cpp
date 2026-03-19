#ifndef PLANT_CPP
#define PLANT_CPP

#include "../util/sim/plant.hpp"
#include "../util/grid/camera.hpp"

// Assuming ImGui headers are available via ptest.cpp or similar include paths
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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
    bool globalIllumination = true; // Default to true to see sun emission
    int reflectCount = 2;
    float maxDist = 200.0f;
    float framerate = 10.0f;
    
    // Input state
    std::map<int, bool> keyStates;
    float deltaTime = 0.016f; // approx 30fps

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
        // Position camera to look at the dirt
        cam.origin = v3(0, 5, 30);
        cam.lookAt(v3(0, 2, 0));
        cam.fov = 45;
        
        // Init the simulation
        sim.initWorld();
        v3 bg = v3(0.511f, 0.625f, 0.868f);
        sim.grid.setBackgroundColor(bg);
        sim.grid.setSkylight(bg);
        sim.grid.setSkyboxBackground(bg.x(), bg.y(), bg.z(), 0.1f);
    }

    ~PlantSimUI() {
        if (textu != 0) glDeleteTextures(1, &textu);
    }

    void renderUI(GLFWwindow* window) {
        handleCameraControls(window);

        ImGui::Begin("Plant Growth Lab");
        
        if (ImGui::BeginTable("PlantLayout", 2, ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthStretch, 0.3f);
            ImGui::TableSetupColumn("Viewport", ImGuiTableColumnFlags_WidthStretch, 0.7f);
            ImGui::TableNextColumn();
            
            renderControls();
            
            ImGui::TableNextColumn();
            
            renderPreview();
            
            ImGui::EndTable();
        }
        sim.update(deltaTime);
        
        ImGui::End();
    }

private:
    void handleCameraControls(GLFWwindow* window) {
        glfwPollEvents();
        for (int i = GLFW_KEY_SPACE; i <= GLFW_KEY_LAST; i++) {
            keyStates[i] = (glfwGetKey(window, i) == GLFW_PRESS);
        }
        
        float speed = 10.0f * deltaTime;
        if (keyStates[GLFW_KEY_W]) cam.moveForward(deltaTime * 10.0f);
        if (keyStates[GLFW_KEY_S]) cam.moveBackward(deltaTime * 10.0f);
        if (keyStates[GLFW_KEY_A]) cam.moveLeft(deltaTime * 10.0f);
        if (keyStates[GLFW_KEY_D]) cam.moveRight(deltaTime * 10.0f);
        if (keyStates[GLFW_KEY_Q]) cam.rotateYaw(deltaTime);
        if (keyStates[GLFW_KEY_E]) cam.rotateYaw(-deltaTime);
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
            ImGui::TextColored(weatherColor, "(Time Remaining: %.1fs)", sim.weatherTimer);
            
            ImGui::Separator();
            ImGui::Text("Atmospheric Moisture: %.1f", sim.atmosphericMoisture);
            float localCo2 = 0.0f;
            float localAirTemp = sim.currentTemperature;
            auto airNodes = sim.grid.findInRadius(v3(0, sim.config.voxelSize / 2.0f, 0), sim.config.voxelSize, 5);
            if (!airNodes.empty()) {
                auto ap = std::static_pointer_cast<AirParticle>(airNodes[0]->data);
                localCo2 = ap->co2;
                localAirTemp = ap->temperature;
            }
            ImGui::Text("Local Air Temp: %.1f °C", localAirTemp);
            ImGui::Text("Ambient CO2: %.1f ppm", localCo2);
            
            if (sim.extendedHeatTimer > 0.0f) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.0f, 1.0f), "Heat Wave Timer: %.1f days", sim.extendedHeatTimer / std::max(1.0f, sim.config.dayDuration));
            } else {
                ImGui::Text("Heat Wave Timer: 0.0 days");
            }
        }
        
        if (ImGui::CollapsingHeader("Plant Health & Structure", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Active Plant Cells:");
            ImGui::BulletText("Leaves: %d", sim.leafCount);
            ImGui::BulletText("Roots: %d", sim.rootCount);
            ImGui::Text("Total Organism Resources:");
            ImGui::ProgressBar(std::min(sim.totalPlantEnergy / (std::max(1, sim.leafCount + sim.rootCount) * 20.0f), 1.0f), ImVec2(-1, 0), 
                ("Energy: " + std::to_string((int)sim.totalPlantEnergy)).c_str());
            ImGui::ProgressBar(std::min(sim.totalPlantWater / (std::max(1, sim.leafCount + sim.rootCount) * 20.0f), 1.0f), ImVec2(-1, 0), 
                ("Water: " + std::to_string((int)sim.totalPlantWater)).c_str());
        }
            ImGui::Separator();
            
        if (ImGui::CollapsingHeader("Soil Environment")) {
            float currentHydration = 0.0f;
            float currentTemp = 0.0f;
            float n = 0;
            float p = 0;
            float k = 0;
            float c = 0;
            float mg = 0;

            auto dirtNodes = sim.grid.findInRadius(v3(0, -sim.config.voxelSize / 2.0f, 0), sim.config.voxelSize, 0);
            if (!dirtNodes.empty()) {
                auto dp = std::static_pointer_cast<DirtParticle>(dirtNodes[0]->data);
                currentHydration = dp->hydration;
                currentTemp = dp->temperature;
                n = dp->nitrogen;
                p = dp->phosphorus;
                k = dp->potassium;
                c = dp->carbon;
                mg = dp->magnesium;
            }
            ImGui::Text("Soil Temp: %.1f °C", currentTemp);
            ImGui::ProgressBar(std::min(currentHydration / 500.0f, 1.0f), ImVec2(-1, 0), 
                ("Water: " + std::to_string((int)currentHydration)).c_str());
            
            ImGui::Text("Nitrogen (N): %.1f", n);
            ImGui::Text("Phosphorus (P): %.1f", p);
            ImGui::Text("Potassium (K): %.1f", k);
            ImGui::Text("Carbon (C): %.1f", c);
            ImGui::Text("Magnesium (Mg): %.1f", mg);
        }

        if (ImGui::CollapsingHeader("Sun & Seasons", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool rebuildSun = false;
            
            ImGui::Text("Time & Season");
            ImGui::SliderFloat("Time of Day", &sim.config.timeOfDay, 0.0f, 1.0f);
            int prevDay = sim.config.currentDay;
            if (ImGui::SliderInt("Current Day", &sim.config.currentDay, 0, sim.config.daysPerYear - 1)) {
                sim.config.season = (static_cast<float>(sim.config.currentDay) + sim.config.timeOfDay) / sim.config.daysPerYear;
            }
            ImGui::SliderFloat("Day Duration (s)", &sim.config.dayDuration, 1.0f, 600.0f);
            if (ImGui::SliderInt("Days per Year", &sim.config.daysPerYear, 4, 365)) {
                if (sim.config.currentDay >= sim.config.daysPerYear) sim.config.currentDay = 0;
            }
            
            ImGui::Separator();
            ImGui::Text("Geography");
            ImGui::SliderFloat("Latitude", &sim.config.latitude, -90.0f, 90.0f);
            ImGui::SliderFloat("Axial Tilt", &sim.config.axialTilt, 0.0f, 90.0f);
            
            ImGui::Separator();
            ImGui::Text("Sun Appearance");
            rebuildSun |= ImGui::SliderFloat("Distance", &sim.config.sunDistance, 10.0f, 100.0f);
            rebuildSun |= ImGui::ColorEdit3("Color", sim.config.sunColor.data());
            rebuildSun |= ImGui::DragFloat("Intensity", &sim.config.sunIntensity, 0.1f, 0.0f, 100.0f);

            ImGui::Text("Weather Constraints");
            ImGui::SliderFloat("Precipitation Rate", &sim.config.precipRate, 10.0f, 500.0f);
        }

        if (ImGui::CollapsingHeader("Simulation")) {
            if (ImGui::Button("Reset World", ImVec2(-1, 0))) {
                sim.initWorld();
            }
        }

        if (ImGui::CollapsingHeader("Render Settings")) {
            ImGui::Checkbox("High Quality (Slow)", &slowRender);
            ImGui::Checkbox("Global Illumination", &globalIllumination);
            ImGui::DragInt("Bounces", &reflectCount, 1, 0, 8);
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
        
        // Update Grid settings based on UI
        sim.grid.setMaxDistance(maxDist);
        
        // Render
        if (slowRender) {
            currentPreviewFrame = sim.grid.renderFrame(cam, outHeight, outWidth, frame::colormap::RGB, 10, reflectCount, globalIllumination, true);
        } else {
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, outHeight, outWidth, frame::colormap::RGB);
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