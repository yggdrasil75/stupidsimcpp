#ifndef PLANT_CPP
#define PLANT_CPP

#include "../util/sim/plant2.hpp"
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
    GLuint textu = 0;
    std::mutex PreviewMutex;
    bool textureInitialized = false;
    frame currentPreviewFrame;
    int outWidth = 512;
    int outHeight = 512;
    bool slowRender = false;
    bool globalIllumination = true;
    constexpr float deltaTime = 0.016f;
    float camSpeedmult = 15.0f;
    int simulationSpeed = 1;
    int reflectCount = 2;
    float maxDist = 200.0f;
    bool keyStates[GLFW_KEY_LAST + 1] = {false};
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
    float scrollDelta = 0.0f;
    
    void handleCameraControls(GLFWwindow* window) {
        glfwPollEvents();
        for (int i = GLFW_KEY_SPACE; i <= GLFW_KEY_LAST; i++) {
            keyStates[i] = (glfwGetKey(window, i) == GLFW_PRESS);
        }
        
        float speed = camSpeedmult * deltaTime;
        if (keyStates[GLFW_KEY_W]) cam.moveForward(speed);
        if (keyStates[GLFW_KEY_S]) cam.moveBackward(speed);
        if (keyStates[GLFW_KEY_A]) cam.moveLeft(speed);
        if (keyStates[GLFW_KEY_D]) cam.moveRight(speed);
        if (keyStates[GLFW_KEY_Q] || keyStates[GLFW_KEY_E]) {
            Eigen::Vector3f orbitCenter;
            bool useHit = false;
            RayHit hit;
            Eigen::Vector3f camOrig = cam.origin;
            Eigen::Vector3f rayDir = cam.forward();
            useHit = sim.grid.raycast(camOrig, rayDir, 100, hit);
            if (useHit) {
                orbitCenter = hit.hitPoint;
            } else {
                orbitCenter = cam.origin + cam.forward() * 100;
            }
            float orbitSpeed = speed * 0.05f;
            float angle = (keyStates[GLFW_KEY_Q] ? -orbitSpeed : orbitSpeed);
            Eigen::Vector3f offset = cam.origin - orbitCenter;
            float cosA = cos(angle);
            float sinA = sin(angle);
            Eigen::Vector3f newoffset;
            newoffset = (offset.x() * cosA - offset.z() * sinA, offset.y(), offset.x() * sinA + offset.z() * cosA)
            cam.origin = orbitCenter + newoffset;
            cam.lookAt(orbitCenter);

        }
        if (keyStates[GLFW_KEY_SPACE]) cam.moveUp(speed);
        if (keyStates[GLFW_KEY_LEFT_SHIFT]) cam.moveUp(-speed);
        if (keyStates[GLFW_KEY_MINUS]) camSpeedmult += 1;
        if (keyStates[GLFW_KEY_EQUAL] && keyStates[GLFW_KEY_LEFT_SHIFT]) camSpeedmult += 1;
        if (scrollDelta != 0.0f) {
            float zoomSpeed = speed * scrollDelta * 0.1f;
            if (scrollDelta > 0) cam.moveForward(zoomSpeed);
            else cam.moveBackward(-zoomSpeed);
            scrollDelta = 0.0f;
        }
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        scrollDelta += static_cast<float>(yoffset);
    }
public:
    PlantSimUI() {
        cam.origin = v3(0, 5, 30);
        cam.lookAt(v3(0, 2, 0));
        cam.fov = 45;
        glfwSetScrollCallback(window, scroll_callback);
    }

    ~PlantSimUI() {
        if (textu != 0) glDeleteTextures(1, &textu);
    }

    void renderUI(GLFWwindow* window) {
        handleCameraControls(window);

        ImGui::Begin("Plants", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
        
        if (ImGui::BeginTable("PlantLayout", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthStretch, 0.35f);
            ImGui::TableSetupColumn("Viewport", ImGuiTableColumnFlags_WidthStretch, 0.65f);
            ImGui::TableNextColumn();
            
            if (ImGui::BeginTabBar("ControlTabs")) {
                if (ImGui::BeginTabItem("Simulation")) {
                    StateControls();
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

    void StateControls() {
        ImGui::SliderInt("Simulation Speed Multiplier", &simulationSpeed, 1, 100);
        if (ImGui::isItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            imGui::SetTooltip("will run growth pulses x times per frame, will cause lower frame rates");
        }
        ImGui::SliderFloat("Plant Growth Speed", &sim.config.growthSpeedMultiplier, 0.1f, 100.0f);
        if (ImGui::isItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            imGui::SetTooltip("will multiply growth results by x each growth pulse, will not slow down, but will cause some growth weirdness");
        }
        ImGui::Text("Temperature: %.1f °C", sim.currentTemperature);
        ImGui::Text("Air Moisture: %.1f", sim.atmosphericMoisture);
        ImGui::Text("Soil Moisture: %.1f", sim.SoilMoisture);
        ImGui::Separator();


    }
};