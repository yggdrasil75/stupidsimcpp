#include <map>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>

#include "../util/grid/grid3eigen.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/frame.hpp"
#include "../util/timing_decorator.cpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

struct defaults {
    int outWidth = 512;
    int outHeight = 512;
    int gridSizecube = 512;
    PNoise2 noise = PNoise2(42);
};

struct spheredefaults {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float radius = 128.0f;
    float color[3] = {0.0f, 1.0f, 0.0f};
    bool light = false;
    float emittance = 0.0f;
    float reflection = 0.0f;
    float refraction = 0.0f;
    bool fillInside = false;
    float voxelSize = 0.1f;
    int numPoints = 15000;
};

struct stardefaults {
    float x = 3000.0f;
    float y = 0.0f; 
    float z = 0.0f;
    
    float color[3] = {1.0f, 0.95f, 0.8f};
    float emittance = 1000.0f;
    float size = 1000.0f;
    bool enabled = true;
};

std::mutex PreviewMutex;
GLuint textu = 0;
bool textureInitialized = false;
bool updatePreview = false;
bool previewRequested = false;
using PointType = Eigen::Matrix<float, 3, 1>;

// Render FPS tracking variables
double renderFrameTime = 0.0;
double avgRenderFrameTime = 0.0;
double renderFPS = 0.0;
const int FRAME_HISTORY_SIZE = 60;
std::vector<double> renderFrameTimes;
int frameHistoryIndex = 0;
bool firstFrameMeasured = false;

void createSphere(const defaults& config, const spheredefaults& sconfig, Octree<int>& grid) {
    if (!grid.empty()) grid.clear();

    Eigen::Vector3f colorVec(sconfig.color[0], sconfig.color[1], sconfig.color[2]);
    Eigen::Vector3f center(sconfig.centerX, sconfig.centerY, sconfig.centerZ);
    
    float voxelSize = sconfig.voxelSize;
    float radius = sconfig.radius;
    
    // Calculate how many voxels fit in the diameter
    int voxelsPerDiameter = static_cast<int>(2.0f * radius / voxelSize);
    if (voxelsPerDiameter < 1) voxelsPerDiameter = 1;
    
    // Create a 3D grid that covers the sphere's bounding box
    for (int i = 0; i <= voxelsPerDiameter; i++) {
        for (int j = 0; j <= voxelsPerDiameter; j++) {
            for (int k = 0; k <= voxelsPerDiameter; k++) {
                // Calculate position in the grid
                float x = center.x() - radius + i * voxelSize;
                float y = center.y() - radius + j * voxelSize;
                float z = center.z() - radius + k * voxelSize;
                
                Eigen::Vector3f pos(x, y, z);
                
                // Calculate distance from center
                float dist = (pos - center).norm();
                
                // For solid sphere: include all points within radius
                if (dist <= radius + voxelSize * 0.5f) {
                    // Optional: For better surface quality, adjust surface points
                    if (dist > radius - voxelSize * 0.5f) {
                        // This is a surface voxel, adjust to exactly on surface
                        if (dist > 0.001f) {
                            pos = center + (pos - center).normalized() * radius;
                        }
                    }
                    
                    if (pos.x() >= 0 && pos.x() < config.gridSizecube &&
                        pos.y() >= 0 && pos.y() < config.gridSizecube &&
                        pos.z() >= 0 && pos.z() < config.gridSizecube) {
                        
                        grid.set(1, pos, true, colorVec, voxelSize, true, 1,
                                    sconfig.light, sconfig.emittance, sconfig.refraction, sconfig.reflection, Octree<int>::Shape::CUBE);
                    }
                }
            }
        }
    }
    
    // If we want a truly solid sphere without gaps, we need a second pass
    if (sconfig.fillInside) {
        // Scan for potential gaps in the interior
        int interiorSteps = static_cast<int>(radius / voxelSize);
        float interiorStep = voxelSize * 0.5f; // Half-step for gap checking
        
        for (int i = 0; i <= interiorSteps * 2; i++) {
            for (int j = 0; j <= interiorSteps * 2; j++) {
                for (int k = 0; k <= interiorSteps * 2; k++) {
                    Eigen::Vector3f pos(
                        center.x() - radius + i * interiorStep,
                        center.y() - radius + j * interiorStep,
                        center.z() - radius + k * interiorStep
                    );
                    
                    float dist = (pos - center).norm();
                    
                    // If deep inside the sphere
                    if (dist < radius * 0.8f) {
                        // Check if position is valid
                        if (pos.x() >= 0 && pos.x() < config.gridSizecube &&
                            pos.y() >= 0 && pos.y() < config.gridSizecube &&
                            pos.z() >= 0 && pos.z() < config.gridSizecube) {
                            
                            // Try to add the point
                            grid.set(1, pos, true, colorVec, voxelSize, true, 1,
                                     sconfig.light, sconfig.emittance, sconfig.refraction, sconfig.reflection);
                        }
                    }
                }
            }
        }
    }
}

void addStar(const defaults& config, const stardefaults& starconf, Octree<int>& grid) {
    if (!starconf.enabled) return;

    Eigen::Vector3f colorVec(starconf.color[0], starconf.color[1], starconf.color[2]);
    PointType pos(starconf.x, starconf.y, starconf.z);

    grid.set(2, pos, true, colorVec, starconf.size, true, 2, true, starconf.emittance, 0.0f, 0.0f);
}

void livePreview(Octree<int>& grid, defaults& config, const Camera& cam) {
    std::lock_guard<std::mutex> lock(PreviewMutex);
    updatePreview = true;
    
    // Measure render time
    auto renderStart = std::chrono::high_resolution_clock::now();
    
    frame currentPreviewFrame = grid.renderFrame(cam, config.outWidth, config.outHeight, frame::colormap::RGB, 3, 1, true);
    
    auto renderEnd = std::chrono::high_resolution_clock::now();
    renderFrameTime = std::chrono::duration<double>(renderEnd - renderStart).count();
    
    // Update FPS calculations
    if (!firstFrameMeasured) {
        renderFrameTimes.resize(FRAME_HISTORY_SIZE, renderFrameTime);
        firstFrameMeasured = true;
    }
    
    renderFrameTimes[frameHistoryIndex] = renderFrameTime;
    frameHistoryIndex = (frameHistoryIndex + 1) % FRAME_HISTORY_SIZE;
    
    // Calculate average frame time and FPS
    avgRenderFrameTime = 0.0;
    int validFrames = 0;
    for (int i = 0; i < FRAME_HISTORY_SIZE; i++) {
        if (renderFrameTimes[i] > 0) {
            avgRenderFrameTime += renderFrameTimes[i];
            validFrames++;
        }
    }
    if (validFrames > 0) {
        avgRenderFrameTime /= validFrames;
        renderFPS = 1.0 / avgRenderFrameTime;
    }
    
    // Update texture
    if (textu == 0) {
        glGenTextures(1, &textu);
    }
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                 0, GL_RGB, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
    
    updatePreview = false;
    textureInitialized = true;
}

void resetView(Camera& cam, float gridSize) {
    cam.origin = Vector3f(gridSize, gridSize, gridSize);
    Vector3f center(gridSize / 2.0f, gridSize / 2.0f, gridSize / 2.0f);
    cam.lookAt(center);
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "gui stuff is dumb in c++." << std::endl;
        glfwTerminate();
        return 1;
    }
    // COPIED VERBATIM FROM IMGUI.
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
    GLFWwindow* window = glfwCreateWindow((int)(1280), (int)(800), "voxelgrid live renderer", nullptr, nullptr);
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

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    defaults config;
    PointType minBound(-config.gridSizecube, -config.gridSizecube, -config.gridSizecube);
    PointType maxBound(config.gridSizecube, config.gridSizecube, config.gridSizecube);
    Octree<int> grid(minBound, maxBound, 16, 16);
    bool gridInitialized = false;
    float ghalf = config.gridSizecube / 2.f;
    
    spheredefaults sphereConf;
    stardefaults starConf;
    
    sphereConf.centerX = ghalf;
    sphereConf.centerY = ghalf;
    sphereConf.centerZ = ghalf;

    bool autoRotate = false;
    bool autoRotateView = false;
    float rotationSpeedX = 0.1f;
    float rotationSpeedY = 0.07f;
    float rotationSpeedZ = 0.05f;
    float autoRotationTime = 0.0f;
    PointType initialViewDir(0, 0, 1);
    float rotationRadius = 255.0f;
    float yawSpeed = 0.5f;
    float pitchSpeed = 0.3f;
    float rollSpeed = 0.2f;
    float autoRotationAngle = 0.0f;
    PointType initialUpDir(0, 1, 0);
    float camX = 0.0f;
    float camY = 0.0f;
    float camZ = 0.0f;
    float camvX = 0.f;
    float camvY = 0.f;
    float camvZ = 0.f;
    float camspeed = 50;
    Camera cam(PointType(400, 400, 400), PointType(0,0,1), PointType(0,1,0), 80, camspeed);

    // Keyboard state tracking
    std::map<int, bool> keyStates;
    bool mouseCaptured = false;
    double lastMouseX = 0, lastMouseY = 0;
    float deltaTime = 0.016f;
    
    // Initialize render frame times vector
    renderFrameTimes.resize(FRAME_HISTORY_SIZE, 0.0);
    
    if (grid.load("output/Treegrid.yggs")) {
        gridInitialized = true;
        grid.printStats();
        resetView(cam, config.gridSizecube);
    }
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        static double lastFrameTime = currentTime;
        deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        if (autoRotate) autoRotationTime += deltaTime;
        if (autoRotateView) autoRotationAngle += deltaTime;

        glfwPollEvents();
        
        for (int i = GLFW_KEY_SPACE; i <= GLFW_KEY_LAST; i++) {
            keyStates[i] = (glfwGetKey(window, i) == GLFW_PRESS);
        }
        
        // Camera movement - WASD + QE + ZX
        float actualMoveSpeed = deltaTime;
        float actualRotateSpeed = deltaTime;
        
        if (keyStates[GLFW_KEY_W]) {
            cam.moveForward(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_S]) {
            cam.moveBackward(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_A]) {
            cam.moveLeft(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_D]) {
            cam.moveRight(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_Z]) {
            cam.moveUp(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_X]) {
            cam.moveDown(actualMoveSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_Q]) {
            cam.rotateYaw(actualRotateSpeed);
            previewRequested = true;
        }
        if (keyStates[GLFW_KEY_R]) {
            cam.rotateYaw(-actualRotateSpeed);
            previewRequested = true;
        }
        
        // Update camera position and view direction variables for UI
        camX = cam.origin[0];
        camY = cam.origin[1];
        camZ = cam.origin[2];
        
        camvX = cam.direction[0];
        camvY = cam.direction[1];
        camvZ = cam.direction[2];
        camspeed = cam.movementSpeed;
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        {
            ImGui::Begin("Controls");
            
            ImGui::Text("Planet");
            float pos[3] = { sphereConf.centerX, sphereConf.centerY, sphereConf.centerZ };
            if (ImGui::DragFloat3("Center (X,Y,Z)", pos, 1.0f, 0.0f, (float)config.gridSizecube)) {
                sphereConf.centerX = pos[0];
                sphereConf.centerY = pos[1];
                sphereConf.centerZ = pos[2];
            }
            
            ImGui::DragFloat("Radius", &sphereConf.radius, 0.5f, 1.0f, 250.0f);
            
            // Replaced traditional voxel sizing with Point Count logic
            ImGui::DragInt("Point Count", &sphereConf.numPoints, 100, 100, 200000);
            ImGui::DragFloat("Density (Overlap)", &sphereConf.voxelSize, 0.05f, 0.1f, 5.0f);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Multiplies calculated point size. >1.0 ensures solid surface.");
            }

            ImGui::ColorEdit3("Color", sphereConf.color);
            
            ImGui::Separator();
            ImGui::Checkbox("Is Light", &sphereConf.light);
            if(sphereConf.light) {
                ImGui::DragFloat("Emittance", &sphereConf.emittance, 0.1f, 0.0f, 100.0f);
            }
            ImGui::SliderFloat("Reflection", &sphereConf.reflection, 0.0f, 1.0f);
            ImGui::SliderFloat("Refraction", &sphereConf.refraction, 0.0f, 1.0f);
            ImGui::Checkbox("Fill Inside", &sphereConf.fillInside);

            if (ImGui::CollapsingHeader("Star/Sun Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Star", &starConf.enabled);
                
                // Allow large range for position to place it "far away"
                float starPos[3] = { starConf.x, starConf.y, starConf.z };
                if (ImGui::DragFloat3("Position", starPos, 5.0f, -2000.0f, 2000.0f)) {
                    starConf.x = starPos[0];
                    starConf.y = starPos[1];
                    starConf.z = starPos[2];
                }

                ImGui::DragFloat("Size (Radius)", &starConf.size, 1.0f, 1.0f, 500.0f);
                ImGui::DragFloat("Brightness", &starConf.emittance, 1.0f, 0.0f, 1000.0f);
                ImGui::ColorEdit3("Light Color", starConf.color);
            }

            ImGui::Separator();
            
            if (ImGui::Button("Create Sphere & Render")) {
                createSphere(config, sphereConf, grid);
                grid.printStats();
                addStar(config, starConf, grid); 
                gridInitialized = true;
                
                resetView(cam, config.gridSizecube);
                
                livePreview(grid, config, cam);
                ImGui::Image((void*)(intptr_t)textu, ImVec2(config.outWidth, config.outHeight));
                
            }
            
            ImGui::End();
        }

        {
            ImGui::Begin("Preview");
            
            // Display render FPS information
            ImGui::Text("Render Performance:");
            if (renderFPS > 0) {
                // Color code based on FPS
                ImVec4 fpsColor;
                if (renderFPS >= 30.0) {
                    fpsColor = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green for good FPS
                } else if (renderFPS >= 15.0) {
                    fpsColor = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow for okay FPS
                } else {
                    fpsColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red for poor FPS
                }
                
                ImGui::TextColored(fpsColor, "FPS: %.1f", renderFPS);
                ImGui::Text("Frame time: %.1f ms", avgRenderFrameTime * 1000.0);
                
                // Simple progress bar for frame time
                ImGui::Text("%.1f/100 ms", avgRenderFrameTime * 1000.0);
                
                // Show latest frame time
                ImGui::Text("Latest: %.1f ms", renderFrameTime * 1000.0);
            } else {
                ImGui::Text("No render data yet");
            }
            
            ImGui::Separator();
            
            if (gridInitialized && textureInitialized) {
                ImGui::Image((void*)(intptr_t)textu, ImVec2(config.outWidth, config.outHeight));
            } else if (gridInitialized) {
                ImGui::Text("Preview not generated yet");
            } else {
                ImGui::Text("No grid generated");
            }
            
            ImGui::End();
        }

        {
            ImGui::Begin("controls");

            ImGui::Separator();
            ImGui::Text("Camera Controls:");
            
            float maxSliderValueX = config.gridSizecube;
            float maxSliderValueY = config.gridSizecube;
            float maxSliderValueZ = config.gridSizecube;
            float maxSliderValueRotation = 360.0f;
            
            ImGui::Text("Position (0 to grid size²):");
            if (ImGui::SliderFloat("Camera X", &camX, 0.0f, maxSliderValueX)) {
                cam.origin[0] = camX;
            }
            if (ImGui::SliderFloat("Camera Y", &camY, 0.0f, maxSliderValueY)) {
                cam.origin[1] = camY;
            }
            if (ImGui::SliderFloat("Camera Z", &camZ, 0.0f, maxSliderValueZ)) {
                cam.origin[2] = camZ;
            }
            
            ImGui::Separator();
            ImGui::Text("View Direction:");
            if (ImGui::SliderFloat("Camera View X", &camvX, -1.0f, 1.0f)) {
                cam.direction[0] = camvX;
            }
            if (ImGui::SliderFloat("Camera View Y", &camvY, -1.0f, 1.0f)) {
                cam.direction[1] = camvY;
            }
            if (ImGui::SliderFloat("Camera View Z", &camvZ, -1.0f, 1.0f)) {
                cam.direction[2] = camvZ;
            }

            if (ImGui::SliderFloat("Camera Speed", &camspeed, 1, 100)) {
                cam.movementSpeed = camspeed;
            }
            
            ImGui::Separator();
            ImGui::Text("Current Camera Position:");
            ImGui::Text("X: %.2f, Y: %.2f, Z: %.2f", 
                        cam.origin[0], 
                        cam.origin[1], 
                        cam.origin[2]);

            ImGui::Text("Auto-Rotation:");
            
            // Toggle button for auto-rotation
            if (ImGui::Button(autoRotate ? "Stop Auto-Rotation" : "Start Auto-Rotation")) {
                autoRotate = !autoRotate;
                if (autoRotate) {
                    autoRotationTime = 0.0f;
                    initialViewDir = PointType(camvX, camvY, camvZ);
                }
            }

            if (ImGui::Button(autoRotateView ? "Stop Looking Around" : "Start Looking Around")) {
                autoRotateView = !autoRotateView;
                if (autoRotateView) {
                    autoRotationAngle = 0.0f;
                    initialViewDir = PointType(camvX, camvY, camvZ);
                }
            }

            if (autoRotate) {
                ImGui::SameLine();
                ImGui::Text("(Running)");
                
                // Calculate new view direction using frame-based timing
                float angleX = autoRotationTime * rotationSpeedX;
                float angleY = autoRotationTime * rotationSpeedY;
                float angleZ = autoRotationTime * rotationSpeedZ;
                
                camvX = sinf(angleX) * cosf(angleY);
                camvY = sinf(angleY) * sinf(angleZ);
                camvZ = cosf(angleX) * cosf(angleZ);
                
                // Normalize
                float length = sqrtf(camvX * camvX + camvY * camvY + camvZ * camvZ);
                if (length > 0.001f) {
                    camvX /= length;
                    camvY /= length;
                    camvZ /= length;
                }
                
                // Update camera position
                camX = config.gridSizecube / 2.0f + rotationRadius * camvX;
                camY = config.gridSizecube / 2.0f + rotationRadius * camvY;
                camZ = config.gridSizecube / 2.0f + rotationRadius * camvZ;
                
                // Update camera
                cam.origin = PointType(camX, camY, camZ);
                cam.direction = PointType(camvX, camvY, camvZ);
                
                // Sliders to control rotation speeds
                ImGui::SliderFloat("X Speed", &rotationSpeedX, 0.01f, 1.0f);
                ImGui::SliderFloat("Y Speed", &rotationSpeedY, 0.01f, 1.0f);
                ImGui::SliderFloat("Z Speed", &rotationSpeedZ, 0.01f, 1.0f);
                
                // Slider for orbit radius
                ImGui::SliderFloat("Orbit Radius", &rotationRadius, 10.0f, 200.0f);
            }

            if (autoRotateView) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0, 1, 0, 1), " ACTIVE");
                
                // Calculate rotation angles using frame-based timing
                float yaw = autoRotationAngle * yawSpeed * (3.14159f / 180.0f);
                float pitch = sinf(autoRotationAngle * 0.7f) * pitchSpeed * (3.14159f / 180.0f);
                
                // Apply rotations
                PointType forward = initialViewDir;
                
                // Yaw rotation (around Y axis)
                float cosYaw = cosf(yaw);
                float sinYaw = sinf(yaw);
                PointType tempForward;
                tempForward[0] = forward[0] * cosYaw + forward[2] * sinYaw;
                tempForward[1] = forward[1];
                tempForward[2] = -forward[0] * sinYaw + forward[2] * cosYaw;
                forward = tempForward;
                
                // Pitch rotation (around X axis)
                float cosPitch = cosf(pitch);
                float sinPitch = sinf(pitch);
                tempForward[0] = forward[0];
                tempForward[1] = forward[1] * cosPitch - forward[2] * sinPitch;
                tempForward[2] = forward[1] * sinPitch + forward[2] * cosPitch;
                forward = tempForward;
                
                // Normalize
                float length = sqrtf(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
                if (length > 0.001f) {
                    forward[0] /= length;
                    forward[1] /= length;
                    forward[2] /= length;
                }
                
                // Update view direction
                camvX = forward[0];
                camvY = forward[1];
                camvZ = forward[2];
                
                // Update camera
                cam.direction = PointType(camvX, camvY, camvZ);
                
                // Show current view direction
                ImGui::Text("Current View: (%.3f, %.3f, %.3f)", camvX, camvY, camvZ);
                
                // Sliders to control rotation speeds
                ImGui::SliderFloat("Yaw Speed", &yawSpeed, 0.1f, 5.0f, "%.2f deg/sec");
                ImGui::SliderFloat("Pitch Speed", &pitchSpeed, 0.0f, 2.0f, "%.2f deg/sec");
            }

            ImGui::End();
        }

        if (gridInitialized) livePreview(grid, config, cam);

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    grid.save("output/Treegrid.yggs");
    
    glfwDestroyWindow(window);
    if (textu != 0) {
        glDeleteTextures(1, &textu);
        textu = 0;
    }
    glfwTerminate();
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}