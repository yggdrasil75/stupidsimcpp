#ifndef PNOISE_CPP
#define PNOISE_CPP

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "./pnoise2.hpp"
#include "../jsonhelper.hpp"
#include "../timing_decorator.hpp"
#include "../../imgui/imgui.h"
#include <GLFW/glfw3.h>

enum class NoiseType {
    Perlin = 0,
    Value = 1,
    Fractal = 2,
    Turbulence = 3,
    Ridged = 4,
    Billow = 5,
    WhiteNoise = 6,
    WorleyNoise = 7,
    VoronoiNoise = 8,
    CrystalNoise = 9,
    DomainWarp = 10,
    CurlNoise = 11
};

enum class BlendMode {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Min = 3,
    Max = 4,
    Replace = 5,
    DomainWarp = 6 
};

struct NoiseLayer {
    bool enabled = true;
    char name[32] = "Layer";
    
    NoiseType type = NoiseType::Perlin;
    BlendMode blend = BlendMode::Add;
    
    int seedOffset = 0;
    float scale = 0.02f;
    float strength = 1.0f;
    int octaves = 4;
    float persistence = 0.5f;
    float lacunarity = 2.0f;
    float ridgeOffset = 1.0f;
};

struct NoisePreviewState {
    int width = 512;
    int height = 512;
    int masterSeed = 1337;
    float offset[2] = {0.0f, 0.0f}; 
    std::vector<NoiseLayer> layers;
    GLuint textureId = 0;
    std::vector<uint8_t> pixelBuffer;
    bool needsUpdate = true;
};

inline float sampleNoiseLayer(PNoise2& gen, NoiseType type, Eigen::Vector2f point, const NoiseLayer& layer) {
    float val = 0.0f;
    switch (type) {
        case NoiseType::Perlin:
            val = gen.permute(point); 
            break;
        case NoiseType::Value:
            val = gen.valueNoise(point); 
            break;
        case NoiseType::Fractal:
            val = gen.fractalNoise(point, layer.octaves, layer.persistence, layer.lacunarity);
            break;
        case NoiseType::Turbulence:
            val = gen.turbulence(point, layer.octaves);
            val = (val * 2.0f) - 1.0f; 
            break;
        case NoiseType::Ridged:
            val = gen.ridgedNoise(point, layer.octaves, layer.ridgeOffset);
            val = (val * 0.5f) - 1.0f; 
            break;
        case NoiseType::Billow:
            val = gen.billowNoise(point, layer.octaves);
            val = (val * 2.0f) - 1.0f;
            break;
        case NoiseType::WhiteNoise:
            val = gen.whiteNoise(point);
            break;
        case NoiseType::WorleyNoise:
            val = gen.worleyNoise(point);
            break;
        case NoiseType::VoronoiNoise:
            val = gen.voronoiNoise(point);
            break;
        case NoiseType::CrystalNoise:
            val = gen.crystalNoise(point);
            break;
        case NoiseType::DomainWarp:
            val = gen.domainWarp(point, 1.0f); 
            break;
        case NoiseType::CurlNoise:
            Eigen::Vector2f flow = gen.curlNoise(point);
            val = flow.norm(); 
            break;
    }
    return val;
}

inline void updateNoiseTexture(NoisePreviewState& state) {
    TIME_FUNCTION;
    if (state.textureId == 0) glGenTextures(1, &state.textureId);
    
    state.pixelBuffer.resize(state.width * state.height * 3);
    
    #pragma omp parallel for
    for (int y = 0; y < state.height; ++y) {
        PNoise2 generator(state.masterSeed); 

        for (int x = 0; x < state.width; ++x) {
            
            float nx = (x + state.offset[0]);
            float ny = (y + state.offset[1]);
            Eigen::Vector2f point(nx, ny);
            
            float finalValue = 0.0f;
            
            for (const auto& layer : state.layers) {
                if (!layer.enabled) continue;

                Eigen::Vector2f samplePoint = point * layer.scale;
                
                samplePoint += Eigen::Vector2f((float)layer.seedOffset * 10.5f, (float)layer.seedOffset * -10.5f);

                if (layer.blend == BlendMode::DomainWarp) {
                    if (layer.type == NoiseType::CurlNoise) {
                         Eigen::Vector2f flow = generator.curlNoise(samplePoint);
                         point += flow * layer.strength * 100.0f; 
                    } else {
                        float warpX = sampleNoiseLayer(generator, layer.type, samplePoint, layer);
                        float warpY = sampleNoiseLayer(generator, layer.type, samplePoint + Eigen::Vector2f(5.2f, 1.3f), layer);
                        point += Eigen::Vector2f(warpX, warpY) * layer.strength * 100.0f;
                    }
                    continue; 
                }

                float nVal = sampleNoiseLayer(generator, layer.type, samplePoint, layer);
                
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
            norm = (norm + 1.0f) * 0.5f;
            norm = std::clamp(norm, 0.0f, 1.0f);
            
            uint8_t color = static_cast<uint8_t>(norm * 255);
            int idx = (y * state.width + x) * 3;
            state.pixelBuffer[idx] = color;
            state.pixelBuffer[idx+1] = color;
            state.pixelBuffer[idx+2] = color;
        }
    }

    glBindTexture(GL_TEXTURE_2D, state.textureId);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, state.width, state.height, 
                 0, GL_RGB, GL_UNSIGNED_BYTE, state.pixelBuffer.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    state.needsUpdate = false;
}

inline void saveNoiseState(const NoisePreviewState& state, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) return;

    out << "{\n";
    out << "  \"masterSeed\": " << state.masterSeed << ",\n";
    out << "  \"offsetX\": " << state.offset[0] << ",\n";
    out << "  \"offsetY\": " << state.offset[1] << ",\n";
    out << "  \"layers\": [\n";
    
    for (size_t i = 0; i < state.layers.size(); ++i) {
        const auto& l = state.layers[i];
        out << "    {\n";
        out << "      \"enabled\": " << (l.enabled ? "true" : "false") << ",\n";
        out << "      \"name\": \"" << l.name << "\",\n";
        out << "      \"type\": " << (int)l.type << ",\n";
        out << "      \"blend\": " << (int)l.blend << ",\n";
        out << "      \"seedOffset\": " << l.seedOffset << ",\n";
        out << "      \"scale\": " << l.scale << ",\n";
        out << "      \"strength\": " << l.strength << ",\n";
        out << "      \"octaves\": " << l.octaves << ",\n";
        out << "      \"persistence\": " << l.persistence << ",\n";
        out << "      \"lacunarity\": " << l.lacunarity << ",\n";
        out << "      \"ridgeOffset\": " << l.ridgeOffset << "\n";
        out << "    }" << (i < state.layers.size() - 1 ? "," : "") << "\n";
    }
    
    out << "  ]\n";
    out << "}\n";
}

inline void loadNoiseState(NoisePreviewState& state, const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return;

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string json = buffer.str();

    state.masterSeed = JsonHelper::parseInt(json, "masterSeed", 1337);
    state.offset[0] = JsonHelper::parseFloat(json, "offsetX", 0.0f);
    state.offset[1] = JsonHelper::parseFloat(json, "offsetY", 0.0f);
    
    auto layerStrs = JsonHelper::parseArray(json, "layers");
    state.layers.clear();
    
    for (const auto& lStr : layerStrs) {
        NoiseLayer l;
        l.enabled = JsonHelper::parseBool(lStr, "enabled", true);
        std::string name = JsonHelper::parseString(lStr, "name");
        if (!name.empty()) {
            std::strncpy(l.name, name.c_str(), 31);
            l.name[31] = '\0';
        }
        l.type = (NoiseType)JsonHelper::parseInt(lStr, "type", 0);
        l.blend = (BlendMode)JsonHelper::parseInt(lStr, "blend", 0);
        l.seedOffset = JsonHelper::parseInt(lStr, "seedOffset", 0);
        l.scale = JsonHelper::parseFloat(lStr, "scale", 0.02f);
        l.strength = JsonHelper::parseFloat(lStr, "strength", 1.0f);
        l.octaves = JsonHelper::parseInt(lStr, "octaves", 4);
        l.persistence = JsonHelper::parseFloat(lStr, "persistence", 0.5f);
        l.lacunarity = JsonHelper::parseFloat(lStr, "lacunarity", 2.0f);
        l.ridgeOffset = JsonHelper::parseFloat(lStr, "ridgeOffset", 1.0f);
        
        state.layers.push_back(l);
    }
    state.needsUpdate = true;
}

inline void drawNoiseLab(NoisePreviewState& noiseState) {
    ImGui::Begin("2D Noise Lab");
    
    bool changed = false;
    
    static char filenameBuffer[128] = "output/noise_preset.json";
    ImGui::InputText("File", filenameBuffer, sizeof(filenameBuffer));
    ImGui::SameLine();
    if (ImGui::Button("Save JSON")) {
        saveNoiseState(noiseState, filenameBuffer);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load JSON")) {
        loadNoiseState(noiseState, filenameBuffer);
        changed = true;
    }

    ImGui::Separator();

    changed |= ImGui::InputInt("Master Seed", &noiseState.masterSeed);
    changed |= ImGui::DragFloat2("Pan Offset", noiseState.offset, 1.0f);
    
    ImGui::Separator();
    
    if (ImGui::Button("Add Layer")) {
        NoiseLayer l;
        sprintf(l.name, "Layer %zu", noiseState.layers.size());
        noiseState.layers.push_back(l);
        changed = true;
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Clear All")) {
        noiseState.layers.clear();
        changed = true;
    }
    
    ImGui::BeginChild("LayersScroll", ImVec2(0, 300), true);
    
    for (int i = 0; i < noiseState.layers.size(); ++i) {
        NoiseLayer& layer = noiseState.layers[i];
        
        ImGui::PushID(i);
        
        bool open = ImGui::CollapsingHeader(layer.name, ImGuiTreeNodeFlags_DefaultOpen);
        
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Move Up", nullptr, false, i > 0)) {
                std::swap(noiseState.layers[i], noiseState.layers[i-1]);
                changed = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Move Down", nullptr, false, i < noiseState.layers.size()-1)) {
                std::swap(noiseState.layers[i], noiseState.layers[i+1]);
                changed = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Duplicate")) {
                noiseState.layers.insert(noiseState.layers.begin() + i, layer);
                changed = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Delete")) {
                noiseState.layers.erase(noiseState.layers.begin() + i);
                changed = true;
                ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
                ImGui::PopID();
                continue; 
            }
            ImGui::EndPopup();
        }

        if (open) {
            if (ImGui::Checkbox("##enabled", &layer.enabled)) changed = true;
            ImGui::SameLine();
            ImGui::InputText("##name", layer.name, 32);

            const char* types[] = { "Perlin", "Value", "Fractal", "Turbulence", "Ridged", "Billow", "White", "Worley", "Voronoi", "Crystal", "Domain Warp", "Curl" };
            const char* blends[] = { "Add", "Subtract", "Multiply", "Min", "Max", "Replace", "Domain Warp (Coord)" };
            
            if (ImGui::Combo("Type", (int*)&layer.type, types, IM_ARRAYSIZE(types))) changed = true;
            if (ImGui::Combo("Blend", (int*)&layer.blend, blends, IM_ARRAYSIZE(blends))) changed = true;

            if (ImGui::SliderFloat("Scale", &layer.scale, 0.0001f, 0.2f, "%.5f")) changed = true;
            if (ImGui::SliderFloat("Strength/Weight", &layer.strength, 0.0f, 5.0f)) changed = true;
            if (ImGui::SliderInt("Seed Offset", &layer.seedOffset, 0, 100)) changed = true;

            if (layer.type == NoiseType::Fractal || layer.type == NoiseType::Turbulence || 
                layer.type == NoiseType::Ridged || layer.type == NoiseType::Billow) {
                
                if (ImGui::SliderInt("Octaves", &layer.octaves, 1, 10)) changed = true;
                if (layer.type == NoiseType::Fractal) {
                    if (ImGui::SliderFloat("Persistence", &layer.persistence, 0.0f, 1.0f)) changed = true;
                    if (ImGui::SliderFloat("Lacunarity", &layer.lacunarity, 1.0f, 4.0f)) changed = true;
                }
                if (layer.type == NoiseType::Ridged) {
                    if (ImGui::SliderFloat("Ridge Offset", &layer.ridgeOffset, 0.0f, 2.0f)) changed = true;
                }
            }
            
            ImGui::Separator();
        }
        
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::Separator();
    
    ImGui::Text("Preview Output");
    ImGui::Image((void*)(intptr_t)noiseState.textureId, ImVec2((float)noiseState.width, (float)noiseState.height));
    
    if (changed || noiseState.needsUpdate) {
        updateNoiseTexture(noiseState);
    }
    
    ImGui::End();
}

#endif