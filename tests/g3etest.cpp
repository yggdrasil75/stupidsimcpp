#include <map>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>
#include <algorithm>

#include "../util/grid/grid3eigen.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/frame.hpp"
#include "../util/timing_decorator.cpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/noise/pnoise.cpp"
#include "../util/output/aviwriter.hpp"
#include "fluidsim.cpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

struct defaults {
    int outWidth = 512;
    int outHeight = 512;
    int gridSizecube = 10000;
    bool slowRender = false;
    bool globalIllumination = true;
    bool useLod = true;
    int rayCount = 3;
    int reflectCount = 3;
    int lodDist = 500;
    float lodDropoff = 0.1;
    PNoise2 noise = PNoise2(42);
    
    int meshResolution = 16;
    float meshIsoLevel = 0.4f;
};

struct spheredefaults {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float radius = 1024.0f;
    float color[3] = {0.0f, 1.0f, 0.0f};
    bool light = false;
    float emittance = 0.0f;
    float reflection = 0.0f;
    float refraction = 0.0f;
    bool fillInside = false;
    float voxelSize = 10.f;
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

Scene scene;
bool meshNeedsUpdate = false;

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
                                    sconfig.light, sconfig.emittance, sconfig.refraction, sconfig.reflection);
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
    std::cout << "voxel grid ready" << std::endl;
    meshNeedsUpdate = true;
}

void addStar(const defaults& config, const stardefaults& starconf, Octree<int>& grid) {
    if (!starconf.enabled) return;

    Eigen::Vector3f colorVec(starconf.color[0], starconf.color[1], starconf.color[2]);
    PointType pos(starconf.x, starconf.y, starconf.z);

    grid.set(2, pos, true, colorVec, starconf.size, true, 2, true, starconf.emittance, 0.0f, 0.0f);
    
    meshNeedsUpdate = true;
}


void livePreview(Octree<int>& grid, defaults& config, const Camera& cam) {
    std::lock_guard<std::mutex> lock(PreviewMutex);
    updatePreview = true;

    
    if (meshNeedsUpdate) {
        scene.clear();
        std::shared_ptr<Mesh> planetMesh = grid.generateMesh(1, config.meshIsoLevel, pow(config.meshResolution, 2));
        std::shared_ptr<Mesh> starMesh = grid.generateMesh(2, config.meshIsoLevel, config.meshResolution);

        scene.addMesh(planetMesh);
        scene.addMesh(starMesh);
        
        // planetMesh.setResolution(config.meshResolution);
        // planetMesh.setIsoLevel(config.meshIsoLevel);
        // planetMesh.update(grid);
        meshNeedsUpdate = false;
    }

    auto renderStart = std::chrono::high_resolution_clock::now();
    frame currentPreviewFrame;

    currentPreviewFrame = scene.render(cam, config.outWidth, config.outHeight, 0.1f, 10000.0f, frame::colormap::RGB);
    
    // grid.setLODMinDistance(config.lodDist);
    // grid.setLODFalloff(config.lodDropoff);
    // if (config.slowRender) {
    //     currentPreviewFrame = grid.renderFrame(cam, config.outWidth, config.outHeight, frame::colormap::RGB, config.rayCount, config.reflectCount, config.globalIllumination, config.useLod);
    // } else {
    //     currentPreviewFrame = grid.fastRenderFrame(cam, config.outWidth, config.outHeight, frame::colormap::RGB);
    // }
    
    auto renderEnd = std::chrono::high_resolution_clock::now();
    renderFrameTime = std::chrono::duration<double>(renderEnd - renderStart).count();
    
    if (!firstFrameMeasured) {
        renderFrameTimes.resize(FRAME_HISTORY_SIZE, renderFrameTime);
        firstFrameMeasured = true;
    }
    
    renderFrameTimes[frameHistoryIndex] = renderFrameTime;
    frameHistoryIndex = (frameHistoryIndex + 1) % FRAME_HISTORY_SIZE;
    
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
    
    auto now = std::chrono::steady_clock::now();
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

    defaults config;
    PointType minBound(-config.gridSizecube, -config.gridSizecube, -config.gridSizecube);
    PointType maxBound(config.gridSizecube, config.gridSizecube, config.gridSizecube);
    Octree<int> grid(minBound, maxBound, 16, 16);
    bool gridInitialized = false;
    float ghalf = config.gridSizecube / 2.f;
    
    spheredefaults sphereConf;
    stardefaults starConf;
    
    // Initialize Noise Preview State
    NoisePreviewState noiseState;
    NoiseLayer l1;
    l1.type = NoiseType::Fractal;
    l1.blend = BlendMode::Add; // Start with base
    l1.scale = 0.005f;
    strcpy(l1.name, "Continents");
    noiseState.layers.push_back(l1);

    updateNoiseTexture(noiseState);

    FluidSimUI fluidUI;
    bool showFluidSim = false;

    sphereConf.centerX = ghalf;
    sphereConf.centerY = ghalf;
    sphereConf.centerZ = ghalf;

    bool autoRotate = false;
    bool autoRotateView = false;
    bool orbitEquator = false;
    bool orbitPoles = false;
    
    float rotationSpeedX = 0.1f;
    float rotationSpeedY = 0.07f;
    float rotationSpeedZ = 0.05f;
    float autoRotationTime = 0.0f;
    PointType initialViewDir(0, 0, 1);
    float rotationRadius = 2000.0f;
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

    std::map<int, bool> keyStates;
    bool mouseCaptured = false;
    double lastMouseX = 0, lastMouseY = 0;
    float deltaTime = 0.016f;
    renderFrameTimes.resize(FRAME_HISTORY_SIZE, 0.0);
    
    bool worldPreview = false;
    
    if (grid.load("output/Treegrid.yggs")) {
        gridInitialized = true;
        resetView(cam, config.gridSizecube);
        meshNeedsUpdate = true;
    }
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        static double lastFrameTime = currentTime;
        deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        if (showFluidSim) {
            fluidUI.update();
        }
        
        if (orbitEquator || orbitPoles || autoRotate) autoRotationTime += deltaTime;
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
        
        float cx = sphereConf.centerX;
        float cy = sphereConf.centerY;
        float cz = sphereConf.centerZ;

        if (orbitEquator || orbitPoles) {
            float speed = 0.5f;
            float angle = autoRotationTime * speed;
            
            if (orbitEquator) {
                cam.origin[0] = cx + rotationRadius * cosf(angle);
                cam.origin[1] = cy;
                cam.origin[2] = cz + rotationRadius * sinf(angle);
            } else {
                cam.origin[0] = cx;
                cam.origin[1] = cy + rotationRadius * cosf(angle);
                cam.origin[2] = cz + rotationRadius * sinf(angle);
            }
            
            PointType target(cx, cy, cz);
            cam.direction = (target - cam.origin).normalized();
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
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        {
            ImGui::Begin("Sim Controls");
            
            ImGui::Text("Planet");
            float pos[3] = { sphereConf.centerX, sphereConf.centerY, sphereConf.centerZ };
            if (ImGui::DragFloat3("Center (X,Y,Z)", pos, 1.0f, 0.0f, (float)config.gridSizecube)) {
                sphereConf.centerX = pos[0];
                sphereConf.centerY = pos[1];
                sphereConf.centerZ = pos[2];
            }
            ImGui::DragFloat("Radius", &sphereConf.radius, 0.5f, 1.0f, 250.0f);
            ImGui::DragFloat("Density (Overlap)", &sphereConf.voxelSize, 0.1f, 1.0f, 100.0f);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Multiplies calculated point size. >1.0 ensures solid surface.");
            }
            ImGui::ColorEdit3("Color", sphereConf.color);
            
            ImGui::Separator();
            ImGui::Text("Marching Cubes Config");
            if (ImGui::SliderInt("Mesh Resolution", &config.meshResolution, 1, 64)) {
                meshNeedsUpdate = true;
            }
            if (ImGui::SliderFloat("Iso Level", &config.meshIsoLevel, 0.01f, 1.0f)) {
                meshNeedsUpdate = true;
            }

            ImGui::Separator();
            ImGui::Checkbox("Is Light", &sphereConf.light);
            // if(sphereConf.light) {
            //     ImGui::DragFloat("Emittance", &sphereConf.emittance, 0.1f, 0.0f, 100.0f);
            // }
            // ImGui::SliderFloat("Reflection", &sphereConf.reflection, 0.0f, 1.0f);
            // ImGui::SliderFloat("Refraction", &sphereConf.refraction, 0.0f, 1.0f);
            // ImGui::Checkbox("Fill Inside", &sphereConf.fillInside);

            if (ImGui::CollapsingHeader("Star/Sun Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Star", &starConf.enabled);
                
                // Allow large range for position to place it "far away"
                float starPos[3] = { starConf.x, starConf.y, starConf.z };
                if (ImGui::DragFloat3("Position", starPos, 5.0f, -2000.0f, 2000.0f)) {
                    starConf.x = starPos[0];
                    starConf.y = starPos[1];
                    starConf.z = starPos[2];
                }

                ImGui::DragFloat("Size (Radius)", &starConf.size, 1.0f, 1.0f, 1000.0f);
                ImGui::DragFloat("Brightness", &starConf.emittance, 1.0f, 0.0f, 1000.0f);
                ImGui::ColorEdit3("Light Color", starConf.color);
            }

            ImGui::Separator();
            
            if (ImGui::Button("Create Sphere & Render")) {
                createSphere(config, sphereConf, grid);
                addStar(config, starConf, grid); 
                gridInitialized = true;
                scene.updateStats();
                
                resetView(cam, config.gridSizecube);
                grid.generateLODs();
                livePreview(grid, config, cam);
                ImGui::Image((void*)(intptr_t)textu, ImVec2(config.outWidth, config.outHeight));
                
            }

            ImGui::Separator();
            ImGui::Text("Modules");
            ImGui::Checkbox("Fluid Simulation", &showFluidSim);
            
            ImGui::End();
        }

        if (showFluidSim) {
            fluidUI.renderUI();
        }

        scene.drawSceneWindow("Planet Preview", cam, 0.01, 1000);
        scene.drawGridStats();

        {
            ImGui::Begin("controls");

            ImGui::Separator();
            ImGui::Text("Camera Controls:");
            
            float maxSliderValueX = config.gridSizecube;
            float maxSliderValueY = config.gridSizecube;
            float maxSliderValueZ = config.gridSizecube;
            
            ImGui::Text("Position (0 to grid size):");
            if (ImGui::SliderFloat("Camera X", &camX, -maxSliderValueX, maxSliderValueX)) {
                cam.origin[0] = camX;
            }
            if (ImGui::SliderFloat("Camera Y", &camY, -maxSliderValueY, maxSliderValueY)) {
                cam.origin[1] = camY;
            }
            if (ImGui::SliderFloat("Camera Z", &camZ, -maxSliderValueZ, maxSliderValueZ)) {
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

            if (ImGui::SliderFloat("Camera Speed", &camspeed, 1, 500)) {
                cam.movementSpeed = camspeed;
            }
            
            ImGui::Separator();
            
            if (ImGui::Button("Focus on Planet")) {
                PointType target(sphereConf.centerX, sphereConf.centerY, sphereConf.centerZ);
                PointType newDir = (target - cam.origin).normalized();
                cam.direction = newDir;
                camvX = newDir[0];
                camvY = newDir[1];
                camvZ = newDir[2];
                previewRequested = true;
            }
            
            ImGui::SameLine();
            
            // Equator Orbit
            if (ImGui::Button(orbitEquator ? "Stop Equator" : "Orbit Equator")) {
                orbitEquator = !orbitEquator;
                if (orbitEquator) {
                    orbitPoles = false;
                    autoRotate = false;
                    autoRotationTime = 0.0f;
                }
            }

            ImGui::SameLine();
            
            // Polar Orbit
            if (ImGui::Button(orbitPoles ? "Stop Poles" : "Orbit Poles")) {
                orbitPoles = !orbitPoles;
                if (orbitPoles) {
                    orbitEquator = false;
                    autoRotate = false;
                    autoRotationTime = 0.0f;
                }
            }

            ImGui::Separator();
            ImGui::Text("Current Camera Position:");
            ImGui::Text("X: %.2f, Y: %.2f, Z: %.2f", cam.origin[0], cam.origin[1], cam.origin[2]);

            ImGui::Text("Auto-Rotation:");
            
            // Toggle button for auto-rotation
            if (ImGui::Button(autoRotate ? "Stop random Rotate" : "Start random Rotate")) {
                autoRotate = !autoRotate;
                if (autoRotate) {
                    orbitEquator = false;
                    orbitPoles = false;
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
                cam.lookAt(PointType(sphereConf.centerX, sphereConf.centerY, sphereConf.centerZ));

                // Sliders to control rotation speeds
                ImGui::SliderFloat("X Speed", &rotationSpeedX, 0.01f, 1.0f);
                ImGui::SliderFloat("Y Speed", &rotationSpeedY, 0.01f, 1.0f);
                ImGui::SliderFloat("Z Speed", &rotationSpeedZ, 0.01f, 1.0f);
            }
            
            // Slider for orbit radius
            ImGui::SliderFloat("Orbit Radius", &rotationRadius, 10.0f, 5000.0f);

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

            ImGui::Separator();

            ImGui::Checkbox("Continuous Preview", &worldPreview);

            // Removed Raytracing specific controls (Global Illum, LOD) as they don't apply to raster mesh
            // ImGui::Checkbox("update Preview", &worldPreview);
            // ImGui::Checkbox("Use Slower renderer", &config.slowRender);
            // if (config.slowRender) {
            //     ImGui::InputInt("Rays per pixel", &config.rayCount);
            //     ImGui::InputInt("Max reflections", &config.reflectCount);
            // }
            // ImGui::InputFloat("Lod dropoff", &config.lodDropoff);
            // ImGui::InputInt("lod minimum Distance", &config.lodDist);
            // ImGui::Checkbox("use Global illumination", &config.globalIllumination);
            // ImGui::Checkbox("use Lod", &config.useLod);

            ImGui::End();
        }
        
        drawNoiseLab(noiseState);

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
    if (noiseState.textureId != 0) {
        glDeleteTextures(1, &noiseState.textureId);
        noiseState.textureId = 0;
    }
    glfwTerminate();
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}