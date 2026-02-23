#ifndef FLUIDSIM_CPP
#define FLUIDSIM_CPP

#include "../util/sim/fluidsim.hpp"
#include "../util/timing_decorator.cpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"


struct fluidSimPreviewConfig {
    float lodDist = 4096.0f;
    float lodDropoff = 0.001f;
    bool slowRender = false;
    int outWidth = 1024;
    int outHeight = 1024;
    int rayCount = 1;
    int reflectCount = 1;
    bool globalIllumination = false;
    bool useLod = true;
};

class FluidSimUI {
private:
    fluidSim sim;
    fluidParticle baseParticle;
    fluidSimPreviewConfig viewConfig;
    Camera cam;
    
    // UI State
    bool isRunning = false;
    int particlesToSpawnPerFrame = 2;
    int totalParticleCap = 5000;
    
    // Texture Management
    GLuint textu = 0;
    std::mutex PreviewMutex;
    bool updatePreview = false;
    bool textureInitialized = false;
    frame currentPreviewFrame;

public:
    FluidSimUI() {
        cam.origin = Eigen::Vector3f(0.0f, 4000.0f, -5800.0f);
        cam.direction = (Eigen::Vector3f(0.0f, 0.0f, 0.0f) - cam.origin).normalized();
        cam.up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        cam.fov = 60.0f;

        baseParticle.velocity = Eigen::Vector3f::Zero();
        baseParticle.acceleration = Eigen::Vector3f::Zero();
        baseParticle.forceAccumulator = Eigen::Vector3f::Zero();
        baseParticle.pressure = 0.0f;
        baseParticle.viscosity = 8.0f;
        baseParticle.restitution = 200.0f;
        
        sim.grid.setLODFalloff(viewConfig.lodDropoff);
        sim.grid.setLODMinDistance(viewConfig.lodDist);
    }

    ~FluidSimUI() {
        if (textu != 0) {
            glDeleteTextures(1, &textu);
        }
    }

    void update() {
        if (isRunning) {
            if (sim.getParticleCount() < (size_t)totalParticleCap) {
                sim.spawnParticles(baseParticle, particlesToSpawnPerFrame);
            }
            sim.applyPhysics();
        }
    }

    void renderUI() {
        ImGui::Begin("Fluid Simulation Control");

        if (ImGui::Button(isRunning ? "Pause" : "Start")) {
            isRunning = !isRunning;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            sim.applyPhysics();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            std::lock_guard<std::mutex> lock(PreviewMutex);
            sim.reset();
        }

        ImGui::Separator();
        ImGui::Text("Particle Management");
        ImGui::DragInt("Spawn Rate (per frame)", &particlesToSpawnPerFrame, 1, 0, 100);
        ImGui::DragInt("Max Particles", &totalParticleCap, 10, 0, 50000);
        ImGui::Text("Current Particles: %zu", sim.getParticleCount());

        if (ImGui::CollapsingHeader("Physics Parameters")) {
            ImGui::DragFloat("Smoothing Radius", &sim.config.SMOOTHING_RADIUS, 1.0f, 100.0f, 5000.0f);
            ImGui::DragFloat("Rest Density", &sim.config.REST_DENSITY, 0.00001f, 0.0f, 0.01f, "%.6f");
            ImGui::DragFloat("Timestep", &sim.config.TIMESTEP, 0.001f, 0.001f, 0.1f);
            ImGui::DragFloat("Gravity (Attraction)", &sim.config.G_ATTRACTION, 0.1f, 0.0f, 500.0f);
            
            ImGui::Text("Base Particle Properties");
            ImGui::DragFloat("Viscosity", &baseParticle.viscosity, 0.1f, 0.0f, 50.0f);
            ImGui::DragFloat("Restitution", &baseParticle.restitution, 1.0f, 0.0f, 1000.0f);
        }

        if (ImGui::CollapsingHeader("View Settings")) {
            ImGui::DragFloat("Camera FOV", &cam.fov, 1.0f, 10.0f, 170.0f);
            
            float camPos[3] = {cam.origin.x(), cam.origin.y(), cam.origin.z()};
            if(ImGui::DragFloat3("Camera Pos", camPos, 10.0f)) {
                cam.origin = Eigen::Vector3f(camPos[0], camPos[1], camPos[2]);
                cam.direction = (Eigen::Vector3f(0.0f, 0.0f, 0.0f) - cam.origin).normalized();
            }

            ImGui::Checkbox("High Quality Render", &viewConfig.slowRender);
            if (viewConfig.slowRender) {
                ImGui::DragInt("Ray Count", &viewConfig.rayCount, 1, 1, 16);
                ImGui::Checkbox("Global Illumination", &viewConfig.globalIllumination);
            }
        }

        updatePreviewTexture();
        
        if (textureInitialized) {
            float aspect = (float)currentPreviewFrame.getWidth() / (float)currentPreviewFrame.getHeight();
            float availWidth = ImGui::GetContentRegionAvail().x;
            ImGui::Image((void*)(intptr_t)textu, ImVec2(availWidth, availWidth / aspect));
        }

        ImGui::End();
    }

private:
    void updatePreviewTexture() {
        std::lock_guard<std::mutex> lock(PreviewMutex);
        updatePreview = true;
        
        sim.grid.setLODMinDistance(viewConfig.lodDist);
        sim.grid.setLODFalloff(viewConfig.lodDropoff);
        
        if (viewConfig.slowRender) {
            currentPreviewFrame = sim.grid.renderFrame(cam, viewConfig.outWidth, viewConfig.outHeight, frame::colormap::RGB, viewConfig.rayCount, viewConfig.reflectCount, viewConfig.globalIllumination, viewConfig.useLod);
        } else {
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, viewConfig.outWidth, viewConfig.outHeight, frame::colormap::RGB);
        }
        
        if (textu == 0) {
            glGenTextures(1, &textu);
        }
        glBindTexture(GL_TEXTURE_2D, textu);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                     0, GL_RGB, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
        
        updatePreview = false;
        textureInitialized = true;
    }
};

#endif