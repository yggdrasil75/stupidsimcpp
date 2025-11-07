// noisegui.cpp
#include "pnoise.hpp"
#include "../bmpwriter.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

// Convert noise value to grayscale color
Vec3 noiseToColor(double noiseValue) {
    float value = static_cast<float>(noiseValue);
    return Vec3(value, value, value);
}

// Convert noise value to color using a blue-to-red colormap
Vec3 noiseToHeatmap(double noiseValue) {
    float value = static_cast<float>(noiseValue);
    
    if (value < 0.25f) {
        float t = value / 0.25f;
        return Vec3(0.0f, t, 1.0f);
    } else if (value < 0.5f) {
        float t = (value - 0.25f) / 0.25f;
        return Vec3(0.0f, 1.0f, 1.0f - t);
    } else if (value < 0.75f) {
        float t = (value - 0.5f) / 0.25f;
        return Vec3(t, 1.0f, 0.0f);
    } else {
        float t = (value - 0.75f) / 0.25f;
        return Vec3(1.0f, 1.0f - t, 0.0f);
    }
}

// Convert noise value to terrain-like colors
Vec3 noiseToTerrain(double noiseValue) {
    float value = static_cast<float>(noiseValue);
    
    if (value < 0.3f) {
        return Vec3(0.0f, 0.0f, 0.3f + value * 0.4f);
    } else if (value < 0.4f) {
        return Vec3(0.76f, 0.70f, 0.50f);
    } else if (value < 0.6f) {
        float t = (value - 0.4f) / 0.2f;
        return Vec3(0.0f, 0.4f + t * 0.3f, 0.0f);
    } else if (value < 0.8f) {
        return Vec3(0.0f, 0.3f, 0.0f);
    } else {
        float t = (value - 0.8f) / 0.2f;
        return Vec3(0.8f + t * 0.2f, 0.8f + t * 0.2f, 0.8f + t * 0.2f);
    }
}

class NoiseTexture {
private:
    GLuint textureID;
    int width, height;
    std::vector<unsigned char> pixelData;

public:
    NoiseTexture(int w, int h) : width(w), height(h) {
        pixelData.resize(width * height * 3);
        glGenTextures(1, &textureID);
        updateTexture();
    }

    ~NoiseTexture() {
        glDeleteTextures(1, &textureID);
    }

    void generateNoise(const PerlinNoise& pn, double scale, int octaves, 
                      const std::string& noiseType, const std::string& colorMap) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double noise = 0.0;
                
                if (noiseType == "Basic") {
                    noise = pn.noise(x * scale, y * scale);
                } else if (noiseType == "FBM") {
                    noise = pn.fractal(octaves, x * scale, y * scale);
                } else if (noiseType == "Turbulence") {
                    noise = pn.turbulence(octaves, x * scale, y * scale);
                } else if (noiseType == "Ridged") {
                    noise = pn.ridgedMultiFractal(octaves, x * scale, y * scale, 0.0, 2.0, 0.5, 1.0);
                }
                
                Vec3 color;
                if (colorMap == "Grayscale") {
                    color = noiseToColor(noise);
                } else if (colorMap == "Heatmap") {
                    color = noiseToHeatmap(noise);
                } else if (colorMap == "Terrain") {
                    color = noiseToTerrain(noise);
                }
                
                int index = (y * width + x) * 3;
                pixelData[index] = static_cast<unsigned char>(color.x * 255);
                pixelData[index + 1] = static_cast<unsigned char>(color.y * 255);
                pixelData[index + 2] = static_cast<unsigned char>(color.z * 255);
            }
        }
        updateTexture();
    }

    void updateTexture() {
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixelData.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void draw(const char* label, const ImVec2& size) {
        ImGui::Image((void*)(intptr_t)textureID, size);
        ImGui::Text("%s", label);
    }

    GLuint getTextureID() const { return textureID; }
};

class NoiseComparisonApp {
private:
    GLFWwindow* window;
    int windowWidth, windowHeight;
    
    // Noise parameters
    double scale;
    int octaves;
    unsigned int seed;
    std::string noiseType;
    std::string colorMap;
    
    // Comparison views
    struct ComparisonView {
        std::unique_ptr<NoiseTexture> texture;
        double scale;
        int octaves;
        unsigned int seed;
        std::string noiseType;
        std::string colorMap;
        std::string label;
    };
    
    std::vector<ComparisonView> views;
    int textureSize;
    
    // Preset management
    struct Preset {
        std::string name;
        double scale;
        int octaves;
        std::string noiseType;
        std::string colorMap;
    };
    
    std::vector<Preset> presets;
    int selectedPreset;

public:
    NoiseComparisonApp() : windowWidth(1400), windowHeight(900), scale(0.01), octaves(4), 
                          seed(42), noiseType("FBM"), colorMap("Grayscale"), textureSize(256) {
        initializePresets();
    }

    bool initialize() {
        if (!glfwInit()) return false;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        
        window = glfwCreateWindow(windowWidth, windowHeight, "Perlin Noise Comparison Tool", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        
        ImGui::StyleColorsDark();
        
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
        
        // Initialize with some default views
        addComparisonView("FBM", "Grayscale", 0.01, 4, 42, "FBM Grayscale");
        addComparisonView("FBM", "Heatmap", 0.01, 4, 42, "FBM Heatmap");
        addComparisonView("FBM", "Terrain", 0.01, 4, 42, "FBM Terrain");
        
        return true;
    }

    void initializePresets() {
        presets = {
            {"Basic Grayscale", 0.01, 1, "Basic", "Grayscale"},
            {"FBM Grayscale", 0.01, 4, "FBM", "Grayscale"},
            {"FBM Terrain", 0.01, 4, "FBM", "Terrain"},
            {"Turbulence", 0.01, 4, "Turbulence", "Grayscale"},
            {"Ridged Multi", 0.01, 4, "Ridged", "Grayscale"},
            {"Large Scale", 0.002, 4, "FBM", "Grayscale"},
            {"Small Scale", 0.05, 4, "FBM", "Grayscale"},
            {"High Octaves", 0.01, 8, "FBM", "Grayscale"}
        };
        selectedPreset = 0;
    }

    void addComparisonView(const std::string& type, const std::string& cmap, 
                          double sc, int oct, unsigned int sd, const std::string& lbl) {
        ComparisonView view;
        view.texture = std::make_unique<NoiseTexture>(textureSize, textureSize);
        view.scale = sc;
        view.octaves = oct;
        view.seed = sd;
        view.noiseType = type;
        view.colorMap = cmap;
        view.label = lbl;
        
        PerlinNoise pn(view.seed);
        view.texture->generateNoise(pn, view.scale, view.octaves, view.noiseType, view.colorMap);
        
        views.push_back(std::move(view));
    }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            renderUI();
            
            ImGui::Render();
            
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            
            glfwSwapBuffers(window);
        }
    }

    void renderUI() {
        // Main control panel
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(300, windowHeight));
        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        
        ImGui::Text("Noise Parameters");
        ImGui::Separator();
        
        ImGui::SliderDouble("Scale", &scale, 0.001, 0.1, "%.3f");
        ImGui::SliderInt("Octaves", &octaves, 1, 8);
        ImGui::InputInt("Seed", (int*)&seed);
        
        const char* noiseTypes[] = {"Basic", "FBM", "Turbulence", "Ridged"};
        ImGui::Combo("Noise Type", [](void* data, int idx, const char** out_text) {
            *out_text = noiseTypes[idx];
            return true;
        }, nullptr, IM_ARRAYSIZE(noiseTypes));
        noiseType = noiseTypes[ImGui::GetStateStorage()->GetInt(ImGui::GetID("Noise Type"), 0)];
        
        const char* colorMaps[] = {"Grayscale", "Heatmap", "Terrain"};
        ImGui::Combo("Color Map", [](void* data, int idx, const char** out_text) {
            *out_text = colorMaps[idx];
            return true;
        }, nullptr, IM_ARRAYSIZE(colorMaps));
        colorMap = colorMaps[ImGui::GetStateStorage()->GetInt(ImGui::GetID("Color Map"), 0)];
        
        ImGui::Separator();
        ImGui::Text("Texture Size: %d", textureSize);
        ImGui::SliderInt("##TexSize", &textureSize, 64, 512);
        
        ImGui::Separator();
        if (ImGui::Button("Generate Current")) {
            std::string label = noiseType + " " + colorMap;
            addComparisonView(noiseType, colorMap, scale, octaves, seed, label);
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear All")) {
            views.clear();
        }
        
        ImGui::Separator();
        ImGui::Text("Presets");
        
        std::vector<const char*> presetNames;
        for (const auto& preset : presets) {
            presetNames.push_back(preset.name.c_str());
        }
        
        ImGui::Combo("##Presets", &selectedPreset, presetNames.data(), presetNames.size());
        
        if (ImGui::Button("Add Preset")) {
            if (selectedPreset >= 0 && selectedPreset < presets.size()) {
                const auto& preset = presets[selectedPreset];
                addComparisonView(preset.noiseType, preset.colorMap, 
                                preset.scale, preset.octaves, seed, preset.name);
            }
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Add All Presets")) {
            for (const auto& preset : presets) {
                addComparisonView(preset.noiseType, preset.colorMap, 
                                preset.scale, preset.octaves, seed, preset.name);
            }
        }
        
        ImGui::Separator();
        ImGui::Text("Quick Comparisons");
        
        if (ImGui::Button("Scale Comparison")) {
            std::vector<double> scales = {0.002, 0.005, 0.01, 0.02, 0.05};
            for (double sc : scales) {
                std::string label = "Scale " + std::to_string(sc);
                addComparisonView("FBM", "Grayscale", sc, 4, seed, label);
            }
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Octave Comparison")) {
            for (int oct = 1; oct <= 6; ++oct) {
                std::string label = oct + " Octaves";
                addComparisonView("FBM", "Grayscale", 0.01, oct, seed, label);
            }
        }
        
        ImGui::End();
        
        // Main view area
        ImGui::SetNextWindowPos(ImVec2(300, 0));
        ImGui::SetNextWindowSize(ImVec2(windowWidth - 300, windowHeight));
        ImGui::Begin("Noise Comparison", nullptr, 
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | 
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus);
        
        ImVec2 imageSize(textureSize, textureSize);
        int itemsPerRow = std::max(1, (int)((windowWidth - 320) / (textureSize + 20)));
        
        for (size_t i = 0; i < views.size(); ++i) {
            auto& view = views[i];
            
            if (i % itemsPerRow != 0) ImGui::SameLine();
            
            ImGui::BeginGroup();
            view.texture->draw(view.label.c_str(), imageSize);
            
            // Mini controls for each view
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::SmallButton("Regenerate")) {
                PerlinNoise pn(view.seed);
                view.texture->generateNoise(pn, view.scale, view.octaves, view.noiseType, view.colorMap);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove")) {
                views.erase(views.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
            
            ImGui::EndGroup();
        }
        
        ImGui::End();
    }

    ~NoiseComparisonApp() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    NoiseComparisonApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application!" << std::endl;
        return -1;
    }
    
    app.run();
    return 0;
}