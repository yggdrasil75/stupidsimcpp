#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

struct Voxel {
    float density;
    unsigned char material; // 0: air, 1: rock, 2: dirt, 3: grass
};

struct PlanetConfig {
    int resolution = 64;
    float radius = 10.0f;
    float noiseScale = 0.1f;
    float amplitude = 1.0f;
    int seed = 42;
};

class SphericalVoxelPlanet {
private:
    std::vector<Voxel> voxels;
    PlanetConfig config;
    std::vector<float> distanceField;
    GLuint textureID;

public:
    SphericalVoxelPlanet() : textureID(0) {
        generate();
    }

    ~SphericalVoxelPlanet() {
        if (textureID) {
            glDeleteTextures(1, &textureID);
        }
    }

    // Convert spherical to Cartesian coordinates
    static glm::vec3 sphericalToCartesian(float lat, float lon, float dist) {
        float latRad = glm::radians(lat);
        float lonRad = glm::radians(lon);
        return glm::vec3(
            dist * cos(latRad) * cos(lonRad),
            dist * cos(latRad) * sin(lonRad),
            dist * sin(latRad)
        );
    }

    // 3D noise function for spherical coordinates
    float sphericalNoise(float lat, float lon, float r, int seed) {
        glm::vec3 pos = sphericalToCartesian(lat, lon, 1.0f);
        pos *= config.noiseScale;
        pos.x += seed;

        // Simple noise function
        float fx = pos.x * 12.9898f + pos.y * 78.233f + pos.z * 137.631f;
        return glm::fract(sin(fx) * 43758.5453f);
    }

    void generate() {
        int res = config.resolution;
        voxels.resize(res * res * res);
        distanceField.resize(res * res * res);

        float latStep = 180.0f / res;
        float lonStep = 360.0f / res;
        float distStep = config.radius * 2.0f / res;

        for (int i = 0; i < res; i++) {           // latitude
            float lat = -90.0f + i * latStep;
            float latRad = glm::radians(lat);

            for (int j = 0; j < res; j++) {       // longitude
                float lon = j * lonStep;

                for (int k = 0; k < res; k++) {   // distance from center
                    float dist = k * distStep;
                    float normalizedDist = dist / (config.radius * 2.0f);

                    int idx = i * res * res + j * res + k;

                    // Base density (sphere)
                    float baseDensity = config.radius - dist;

                    // Add noise based on latitude and longitude
                    float noise = sphericalNoise(lat, lon, dist, config.seed);
                    float altitude = 1.0f - std::abs(lat / 90.0f); // Poles are higher

                    float finalDensity = baseDensity + noise * config.amplitude * altitude;

                    distanceField[idx] = finalDensity;

                    // Assign material based on density and position
                    if (finalDensity > 0.5f) {
                        if (k > res * 0.9f) {  // Surface
                            voxels[idx].material = 3; // Grass
                        } else if (k > res * 0.7f) {
                            voxels[idx].material = 2; // Dirt
                        } else {
                            voxels[idx].material = 1; // Rock
                        }
                        voxels[idx].density = 1.0f;
                    } else {
                        voxels[idx].material = 0; // Air
                        voxels[idx].density = 0.0f;
                    }
                }
            }
        }

        updateTexture();
    }

    void updateTexture() {
        int res = config.resolution;
        std::vector<unsigned char> textureData(res * res * res * 3);

        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                for (int k = 0; k < res; k++) {
                    int idx = i * res * res + j * res + k;
                    int texIdx = (i * res * res + j * res + k) * 3;

                    switch (voxels[idx].material) {
                        case 0: // Air
                            textureData[texIdx] = 0;
                            textureData[texIdx + 1] = 0;
                            textureData[texIdx + 2] = 0;
                            break;
                        case 1: // Rock
                            textureData[texIdx] = 100;
                            textureData[texIdx + 1] = 100;
                            textureData[texIdx + 2] = 100;
                            break;
                        case 2: // Dirt
                            textureData[texIdx] = 101;
                            textureData[texIdx + 1] = 67;
                            textureData[texIdx + 2] = 33;
                            break;
                        case 3: // Grass
                            textureData[texIdx] = 34;
                            textureData[texIdx + 1] = 139;
                            textureData[texIdx + 2] = 34;
                            break;
                    }
                }
            }
        }

        if (textureID == 0) {
            glGenTextures(1, &textureID);
        }

        glBindTexture(GL_TEXTURE_3D, textureID);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, res, res, res, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData.data());
    }

    void renderUI() {
        ImGui::Begin("Planet Generator");

        if (ImGui::SliderInt("Resolution", &config.resolution, 32, 128)) {
            generate();
        }

        if (ImGui::SliderFloat("Radius", &config.radius, 5.0f, 20.0f)) {
            generate();
        }

        if (ImGui::SliderFloat("Noise Scale", &config.noiseScale, 0.01f, 0.5f)) {
            generate();
        }

        if (ImGui::SliderFloat("Amplitude", &config.amplitude, 0.0f, 3.0f)) {
            generate();
        }

        if (ImGui::SliderInt("Seed", &config.seed, 0, 1000)) {
            generate();
        }

        if (ImGui::Button("Randomize Seed")) {
            config.seed = rand() % 1000;
            generate();
        }

        ImGui::End();
    }

    void renderPlanet() {
        // Simple raycasting visualization using distance field
        // This is a basic implementation - for better results, implement proper raycasting
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        ImVec2 center = ImGui::GetIO().DisplaySize * 0.5f;
        float displaySize = std::min(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y) * 0.4f;

        int slices = 32;
        float angleStep = 2.0f * 3.14159f / slices;

        // Draw planet outline
        for (int i = 0; i < slices; i++) {
            float angle1 = i * angleStep;
            float angle2 = (i + 1) % slices * angleStep;

            ImVec2 p1 = center + ImVec2(cos(angle1) * displaySize, sin(angle1) * displaySize);
            ImVec2 p2 = center + ImVec2(cos(angle2) * displaySize, sin(angle2) * displaySize);

            drawList->AddLine(p1, p2, IM_COL32(255, 255, 255, 255));
        }

        // Draw some sample voxels
        int res = config.resolution;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                float lat = -90.0f + 180.0f * (i / 7.0f);
                float lon = 360.0f * (j / 7.0f);

                // Find surface voxel
                for (int k = res - 1; k >= 0; k--) {
                    int idx = (i * res / 8) * res * res + (j * res / 8) * res + k;
                    if (voxels[idx].material != 0) {
                        // Convert spherical to screen coordinates
                        float latRad = glm::radians(lat);
                        float lonRad = glm::radians(lon);
                        float dist = (k / (float)res) * config.radius * 2.0f;

                        float screenDist = (dist / (config.radius * 2.0f)) * displaySize;
                        ImVec2 pos = center + ImVec2(
                            cos(latRad) * cos(lonRad) * screenDist,
                            sin(latRad) * screenDist
                        );

                        // Color based on material
                        ImU32 color;
                        switch (voxels[idx].material) {
                            case 1: color = IM_COL32(100, 100, 100, 255); break;
                            case 2: color = IM_COL32(101, 67, 33, 255); break;
                            case 3: color = IM_COL32(34, 139, 34, 255); break;
                            default: color = IM_COL32(0, 0, 0, 0);
                        }

                        if (color != 0) {
                            drawList->AddCircleFilled(pos, 2.0f, color);
                        }
                        break;
                    }
                }
            }
        }
    }
};

int main() {
    // GLFW initialization
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Voxel Planet Generator", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ImGui initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    SphericalVoxelPlanet planet;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render UI
        planet.renderUI();

        // Clear screen
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render planet
        planet.renderPlanet();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
