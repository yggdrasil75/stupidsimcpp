#ifndef TERRAIN_CPP
#define TERRAIN_CPP

#include "../util/sim/terrain.hpp"
#include "../util/grid/camera.hpp"
#include "imgui.h"
#include <mutex>

class TerrainSimUI {
private:
    TerrainSim sim;
    Camera2D cam;
    
    GLuint textureId = 0;
    std::mutex renderMutex;
    bool textureReady = false;
    frame currentFrame;
    
    int previewWidth = 512;
    int previewHeight = 512;
    
    // Region context caching
    Eigen::Vector3f currentCenter = Eigen::Vector3f(1024.0f, 0.0f, 0.0f);
    float currentSize = 1000.0f;
    float currentRes = 10.0f;
    
public:
    TerrainSimUI() {
        cam.origin = Eigen::Vector2f(0.0f, 0.0f);
        cam.viewWidth = 1000.0f;
    }
    
    ~TerrainSimUI() {
        if (textureId != 0) glDeleteTextures(1, &textureId);
    }

    void applyRegion(Eigen::Vector3f center3D, float size, float res, const NoisePreviewState& noiseState) {
        currentCenter = center3D;
        currentSize = size;
        currentRes = res;
        
        sim.generateRegion(center3D, size, res, noiseState);
        
        cam.origin = Eigen::Vector2f(0.0f, 0.0f);
        cam.viewWidth = size * 1.1f;
        updatePreview();
    }
    
    void applyNoise(const NoisePreviewState& noiseState) {
        sim.generateRegion(currentCenter, currentSize, currentRes, noiseState);
        updatePreview();
    }
    
    void renderUI() {
        ImGui::Begin("2D Terrain & Erosion Lab");
        
        if (ImGui::BeginTable("TerrainLayout", 2, ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Settings", ImGuiTableColumnFlags_WidthStretch, 0.35f);
            ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthStretch, 0.65f);
            ImGui::TableNextColumn();
            
            if (ImGui::CollapsingHeader("Region Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat2("Corner Min", sim.cornerMin.data(), 1.0f);
                ImGui::DragFloat2("Corner Max", sim.cornerMax.data(), 1.0f);
                ImGui::DragFloat("Resolution", &sim.resolution, 0.1f, 0.1f, 10.0f);
                
                ImGui::Spacing();
                ImGui::TextDisabled("Planet Context:");
                ImGui::Text("Center 3D: %.1f, %.1f, %.1f", currentCenter.x(), currentCenter.y(), currentCenter.z());
                ImGui::Text("Size: %.1f m", currentSize);
            }
            
            if (ImGui::CollapsingHeader("Camera Settings")) {
                bool camMoved = false;
                camMoved |= ImGui::DragFloat2("Position", cam.origin.data(), 1.0f);
                camMoved |= ImGui::DragFloat("View Width", &cam.viewWidth, 1.0f, 10.0f, 10000.0f);
                
                if (camMoved) updatePreview();
            }
            
            ImGui::TableNextColumn();
            
            if (textureReady) {
                float availW = ImGui::GetContentRegionAvail().x;
                float aspect = (float)previewWidth / (float)previewHeight;
                ImGui::Image((void*)(intptr_t)textureId, ImVec2(availW, availW / aspect));
            } else {
                ImGui::Text("No terrain generated. Apply noise to view.");
            }
            
            ImGui::EndTable();
        }
        ImGui::End();
    }
    
private:
    void updatePreview() {
        std::lock_guard<std::mutex> lock(renderMutex);
        currentFrame = sim.grid.renderFrame(cam, previewWidth, previewHeight);
        
        if (textureId == 0) glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentFrame.getWidth(), currentFrame.getHeight(),
                     0, GL_RGB, GL_UNSIGNED_BYTE, currentFrame.getData().data());
                     
        textureReady = true;
    }
};

#endif