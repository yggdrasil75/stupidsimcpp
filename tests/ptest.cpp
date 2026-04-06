#include <iostream>
#include <vector>
#include <string>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "../util/noise/pnoise.cpp"
#include "planet.cpp"
#include "plant.cpp"
// #include "terrain.cpp"
#include "../util/basicdefines.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return -1;

    #if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100 (WebGL 1.0)
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(IMGUI_IMPL_OPENGL_ES3)
        // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
        const char* glsl_version = "#version 300 es";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(__APPLE__)
        // GL 3.2 + GLSL 150
        const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
    #else
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    #endif

    bool application_not_closed = true;
    GLFWwindow* window = glfwCreateWindow((int)(1280), (int)(800), "StupidSim", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui_ImplGlfw_InitForOpenGL(window, true);

    #ifdef __EMSCRIPTEN__
        ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
    #endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    planetSimUI planetApp;
    NoisePreviewState noiseState;
    PlantSimUI plantApp;
    // TerrainSimUI terrainApp;
    bool doShowPlanet = false;
    bool doShowPlants = true;
    bool doShowNoiseLab = false;
    // bool doShowTerrain = true;

    if (noiseState.layers.empty()) {
        NoiseLayer defaultLayer;
        strcpy(defaultLayer.name, "Base Terrain");
        defaultLayer.type = NoiseType::Fractal;
        noiseState.layers.push_back(defaultLayer);
    }
    updateNoiseTexture(noiseState);
    
    Eigen::Vector3f selectedPlanetPos(1024.0f, 0.0f, 0.0f);
    float regionSize = 1000.0f;
    float regionResolution = 10.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::GetMainViewport();
        if (doShowNoiseLab) drawNoiseLab(noiseState);
        if (doShowPlanet) planetApp.renderUI(window);
        if (doShowPlants) plantApp.renderUI(window);
        // if (doShowTerrain) terrainApp.renderUI();

        ImGui::Begin("Main Controls");
        ImGui::Checkbox("Show Planet", &doShowPlanet);
        ImGui::Checkbox("Show Plants", &doShowPlants);
        // ImGui::Checkbox("Show 2D Terrain", &doShowTerrain);
        ImGui::Checkbox("Show Noise lab", &doShowNoiseLab);

        ImGui::Text("Bridge: Noise Lab -> Planet Sim");
        ImGui::Separator();
        
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Current Noise Layers: %zu", noiseState.layers.size());
        
        if (ImGui::Button("APPLY CURRENT NOISE TO PLANET", ImVec2(-1, 50))) {
            planetApp.applyNoise(noiseState);
        }
        
        // if (ImGui::Button("APPLY CURRENT NOISE TO 2D TERRAIN", ImVec2(-1, 30))) {
        //     terrainApp.applyNoise(noiseState);
        // }
        
        // ImGui::Separator();
        // ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f), "Detailed Terrain Region Selection");
        // ImGui::DragFloat3("Center Coord (3D)", selectedPlanetPos.data(), 1.0f);
        // ImGui::DragFloat("Region Size (m)", &regionSize, 10.0f, 100.0f, 10000.0f);
        // ImGui::DragFloat("Resolution (m/vx)", &regionResolution, 0.1f, 0.5f, 100.0f);

        // if (ImGui::Button("Extract Region to Terrain Module", ImVec2(-1, 40))) {
        //     terrainApp.applyRegion(selectedPlanetPos, regionSize, regionResolution, noiseState);
        //     doShowTerrain = true;
        // }
        
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    return 0;
}