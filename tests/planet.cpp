#ifndef PLANET_CPP
#define PLANET_CPP

#include "../util/sim/planet.hpp"
#include "../util/grid/camera.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/noise/pnoise.cpp"

class planetSimUI {
private:
    planetsim sim;
    Camera cam;
    bool isRunning = false;
    
    // Texture Management
    GLuint textu = 0;
    std::mutex PreviewMutex;
    bool updatePreview = false;
    bool textureInitialized = false;
    frame currentPreviewFrame;
    int outWidth = 1024;
    int outHeight = 1024;
    float fps = 60;
    int rayCount = 3;
    int reflectCount = 4;
    bool slowRender = false;
    float lodDist = 1024.0f;
    float lodDropoff = 0.05f;
    float maxViewDistance = 4096;
    bool globalIllumination = false;
    bool useLod = true;
    std::map<int, bool> keyStates;
    float deltaTime = 0.16f;
    bool orbitEquator = false;
    float rotationRadius = 2500;
    float angle = 0.0f;
    const float ω = (std::pow(M_PI, 2) / 30) / 10;
    bool tectonicGenned = false;
    bool doFixPlates = true;
    bool platesUseCellular = false;
    std::chrono::steady_clock::time_point lastStatsUpdate;
    std::string cachedStats;
    bool statsNeedUpdate = true;
    float framerate = 60.0;
    bool autoAdjustDistance = true;
    int targetMemoryMB = 1024;
    float minAutoDistance = 1024.0f;
    float maxAutoDistance = 16384.0f;
    size_t currentMemoryMB = 0;
    size_t currentLoadedPoints = 0;
    std::chrono::steady_clock::time_point lastAutoAdjustTime = std::chrono::steady_clock::now();

    enum class DebugColorMode {
        BASE,
        PLATES,
        NOISE,
        IMPACTS,
        WATER,
        TEMPERATURE,
        MOISTURE,
        MATERIALS
    };
    DebugColorMode currentColorMode = DebugColorMode::BASE;

    enum class DebugMapMode {
        NONE,
        BASE,
        NOISE,
        TECTONIC,
        TECTONICCOLOR,
        CURRENT
    };
    DebugMapMode currentMapMode = DebugMapMode::NONE;
    GLuint mapTexture = 0;
    frame mapFrame;

public:
    planetSimUI() {
        cam.origin = v3(4000, 4000, 4000);
        cam.direction = (v3(0,0,0) - cam.origin).normalized();
        cam.up = v3(0,1,0);
        cam.fov = 60;
        cam.rotationSpeed = M_1_PI;
    }

    ~planetSimUI() {
        if (textu != 0) {
            glDeleteTextures(1, &textu);
        }
        if (mapTexture != 0) {
            glDeleteTextures(1, &mapTexture);
        }
        sim.grid.clear();
    }

    void renderUI(GLFWwindow* window) {
        handleCameraControls(window);
        ImGui::Begin("Planet Simulation");
        if (ImGui::BeginTable("MainLayout", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter)) {
            ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthStretch, 0.3f);
            ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthStretch, 0.7f);
            ImGui::TableNextColumn();
            renderControlsPanel();
            ImGui::TableNextColumn();
            renderPreviewPanel();
            ImGui::EndTable();
        }
        ImGui::End();
    }

    void handleCameraControls(GLFWwindow* window) {
        if (orbitEquator) {
            angle += cam.rotationSpeed * deltaTime * ω;
            
            cam.origin[0] = sim.config.center[0] + rotationRadius * cosf(angle);
            cam.origin[1] = sim.config.center[1];
            cam.origin[2] = sim.config.center[2] + rotationRadius * sinf(angle);
            
            v3 target(sim.config.center);
            cam.direction = (target - cam.origin).normalized();
        }

        glfwPollEvents();
        for (int i = GLFW_KEY_SPACE; i <= GLFW_KEY_LAST; i++) {
            keyStates[i] = (glfwGetKey(window, i) == GLFW_PRESS);
        }
        if (keyStates[GLFW_KEY_W]) cam.moveForward(deltaTime);
        if (keyStates[GLFW_KEY_S]) cam.moveBackward(deltaTime);
        if (keyStates[GLFW_KEY_A]) cam.moveLeft(deltaTime);
        if (keyStates[GLFW_KEY_D]) cam.moveRight(deltaTime);
        if (keyStates[GLFW_KEY_Z]) cam.moveUp(deltaTime);
        if (keyStates[GLFW_KEY_X]) cam.moveDown(deltaTime);
        if (keyStates[GLFW_KEY_Q]) cam.rotateYaw(deltaTime);
        if (keyStates[GLFW_KEY_R]) cam.rotateYaw(-deltaTime);
    }

    void renderControlsPanel() {
        ImGui::BeginChild("ControlsScroll", ImVec2(0, 0), true);
        
        if (ImGui::CollapsingHeader("Base Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat("Radius", &sim.config.radius, 1.0f, 10.0f, 10000.0f);
            ImGui::InputInt("Surface Points", &sim.config.surfacePoints);
            ImGui::DragFloat("Voxel Size", &sim.config.voxelSize, 0.1f, 0.1f, 100.0f);
            ImGui::ColorEdit3("Base Color", sim.config.color.data());
            
            ImGui::Separator();
            
            if (ImGui::Button("1. Generate Fib Sphere", ImVec2(-1, 40))) {
                sim.generateFibSphere();
                applyDebugColorMode();
            }
            ImGui::Text("Current Step: %d", sim.config.currentStep);
            ImGui::Text("Nodes: %zu", sim.config.surfaceNodes.size());
            ImGui::InputFloat("Noise strength", &sim.config.noiseStrength, 0.01, 1, "%.4f");
        }

        if (ImGui::CollapsingHeader("Physics Parameters")) {
            ImGui::DragFloat("Gravity (G)", &sim.config.G_ATTRACTION, 0.1f);
            ImGui::DragFloat("Time Step", &sim.config.TIMESTEP, 0.001f, 0.0001f, 0.1f);
            ImGui::DragFloat("Viscosity", &sim.config.dampingFactor, 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("Pressure Stiffness", &sim.config.pressureStiffness, 10.0f);
        }

        if (ImGui::CollapsingHeader("Tectonic Simulation")) {
            ImGui::DragInt("Num Plates", &sim.config.numPlates, 1, 1, 100);
            ImGui::DragInt("Smoothing Passes", &sim.config.smoothingPasses, 1, 0, 10);
            ImGui::DragFloat("Mountain Height", &sim.config.mountHeight, 1.0f, 0.0f, 1000.0f);
            ImGui::DragFloat("Valley Depth", &sim.config.valleyDepth, 1.0f, -1000.0f, 0.0f);
            ImGui::DragFloat("Transform Roughness", &sim.config.transformRough, 1.0f, 0.0f, 500.0f);
            ImGui::DragInt("Stress Passes", &sim.config.stressPasses, 1, 0, 20);
            ImGui::DragFloat("Max Elevation Ratio", &sim.config.maxElevationRatio, 1.0f, 0.0f, 1.0f);
            ImGui::Checkbox("Fix Boundaries", &doFixPlates);
            ImGui::Checkbox("use Cellular", &platesUseCellular);

            if (ImGui::Button("2. Simulate Tectonics", ImVec2(-1, 20))) {
                simulateTectonics();
            }
        }

        if (ImGui::CollapsingHeader("Celestial Bodies")) {
            if (ImGui::Button("Add Star", ImVec2(-1, 20))) {
                sim.addStar();
            }
            if (ImGui::Button("Add Moon", ImVec2(-1, 20))) {
                sim.addMoon();
            }
            ImGui::Separator();

            if (!sim.coreFilled) ImGui::BeginDisabled();

            static int numAsteroids = 5;
            ImGui::InputInt("Asteroids Count", &numAsteroids);
            if (numAsteroids < 1) numAsteroids = 1;

            if (ImGui::Button("Generate Asteroid Impacts", ImVec2(-1, 30))) {
                sim.generateAsteroidImpacts(numAsteroids);
                applyDebugColorMode(); 
            }
            if (ImGui::Button("Quick Smooth Surface", ImVec2(-1, 20))) {
                sim.quickSmoothSurface();
                applyDebugColorMode(); 
            }
            ImGui::Separator();
            
            ImGui::DragInt("Erosion Drops", &sim.config.erosionDrops, 1000, 1000, 1000000);
            if (ImGui::Button("Simulate Erosion", ImVec2(-1, 30))) {
                sim.erosion();
                applyDebugColorMode();
            }
            
            ImGui::DragInt("Weather Iterations", &sim.config.weatherIterations, 10, 10, 1000);
            if (ImGui::Button("Simulate Weather", ImVec2(-1, 30))) {
                sim.storms();
                applyDebugColorMode();
            }

            if (!sim.coreFilled) {
                ImGui::EndDisabled();
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Requires 'Fill Planet' to be executed first.");
            }
            
        }

        if (ImGui::CollapsingHeader("Fillings")) {
            if (ImGui::Button("Interpolate surface", ImVec2(-1, 40))) {
                interpolateSurface();
            }
            if (ImGui::Button("Fill Planet", ImVec2(-1, 40))) {
                fillPlanet();
            }
        }

        if (ImGui::CollapsingHeader("Debug Views")) {
            ImGui::Text("3D Planet Color Mode:");
            bool colorChanged = false;
            if (ImGui::RadioButton("Base Color", currentColorMode == DebugColorMode::BASE)) { 
                currentColorMode = DebugColorMode::BASE;
                colorChanged = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Plates", currentColorMode == DebugColorMode::PLATES)) { 
                currentColorMode = DebugColorMode::PLATES;
                colorChanged = true;
            }
            
            if (ImGui::RadioButton("Noise", currentColorMode == DebugColorMode::NOISE)) { 
                currentColorMode = DebugColorMode::NOISE;
                colorChanged = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Impacts", currentColorMode == DebugColorMode::IMPACTS)) { 
                currentColorMode = DebugColorMode::IMPACTS; 
                colorChanged = true; 
            }

            if (ImGui::RadioButton("Materials", currentColorMode == DebugColorMode::MATERIALS)) { 
                currentColorMode = DebugColorMode::MATERIALS; 
                colorChanged = true; 
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Water", currentColorMode == DebugColorMode::WATER)) { 
                currentColorMode = DebugColorMode::WATER; 
                colorChanged = true; 
            }
            if (ImGui::RadioButton("Temp", currentColorMode == DebugColorMode::TEMPERATURE)) { 
                currentColorMode = DebugColorMode::TEMPERATURE; 
                colorChanged = true; 
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Moisture", currentColorMode == DebugColorMode::MOISTURE)) { 
                currentColorMode = DebugColorMode::MOISTURE; 
                colorChanged = true; 
            }

            if (colorChanged) {
                applyDebugColorMode();
            }

            ImGui::Separator();
            ImGui::Text("2D Height Map Mode:");
            bool mapChanged = false;
            
            if (ImGui::RadioButton("None", currentMapMode == DebugMapMode::NONE)) { 
                currentMapMode = DebugMapMode::NONE; 
                mapChanged = true; 
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Base Pos", currentMapMode == DebugMapMode::BASE)) { 
                currentMapMode = DebugMapMode::BASE; 
                mapChanged = true; 
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Noise Pos", currentMapMode == DebugMapMode::NOISE)) { 
                currentMapMode = DebugMapMode::NOISE; 
                mapChanged = true; 
            }

            if (!tectonicGenned) ImGui::BeginDisabled();
            
            ImGui::SameLine();
            if (ImGui::RadioButton("Tectonic Pos", currentMapMode == DebugMapMode::TECTONIC)) { 
                currentMapMode = DebugMapMode::TECTONIC; 
                mapChanged = true; 
            }

            if (ImGui::RadioButton("Tectonic Color", currentMapMode == DebugMapMode::TECTONICCOLOR)) { 
                currentMapMode = DebugMapMode::TECTONICCOLOR;
                mapChanged = true; 
            }
            ImGui::SameLine();
            if (!tectonicGenned) ImGui::EndDisabled();
            
            if (ImGui::RadioButton("Current Pos", currentMapMode == DebugMapMode::CURRENT)) { 
                currentMapMode = DebugMapMode::CURRENT; 
                mapChanged = true; 
            }

            if (ImGui::Button("Refresh Map", ImVec2(-1, 24))) {
                mapChanged = true;
                generateDebugMap(currentMapMode);
            }

            if (mapChanged && currentMapMode != DebugMapMode::NONE) {
                generateDebugMap(currentMapMode);
            }

            if (currentMapMode != DebugMapMode::NONE && mapTexture != 0) {
                float availWidth = ImGui::GetContentRegionAvail().x;
                ImGui::Image((void*)(intptr_t)mapTexture, ImVec2(availWidth, availWidth * 0.5f));
            }
        }

        if (ImGui::CollapsingHeader("Camera Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Origin", cam.origin.data());
            ImGui::DragFloat3("Direction", cam.direction.data(), 0.0001f, -1.0f, 1.0f);
            ImGui::DragFloat("Movement Speed", &cam.movementSpeed, 0.1f, 1.0f, 500.0f);
            ImGui::DragFloat("Rotation Speed", &cam.rotationSpeed, M_1_PI, M_1_PI, M_PI);
            ImGui::InputFloat("Rotation Distance", &rotationRadius, 10, 100);
            ImGui::InputFloat("Max Framerate", &framerate, 1, 10);

            ImGui::Checkbox("Use Slower Render", &slowRender);

            if (ImGui::Button("Focus on Planet")) {
                v3 target(sim.config.center);
                v3 newDir = (target - cam.origin).normalized();
                cam.direction = newDir;
            }

            if (ImGui::Button(orbitEquator ? "Stop Equator" : "Orbit Equator")) orbitEquator = !orbitEquator;
        }
        
        if (ImGui::CollapsingHeader("Streaming & LOD Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Use LOD", &useLod);
            ImGui::Checkbox("Auto-Adjust View Distance", &autoAdjustDistance);
            
            if (autoAdjustDistance) {
                ImGui::InputInt("Target Memory Budget (MB)", &targetMemoryMB);
                ImGui::DragFloat("Min Auto Distance", &minAutoDistance, 10.0f, 10.0f, maxAutoDistance);
                ImGui::DragFloat("Max Auto Distance", &maxAutoDistance, 10.0f, minAutoDistance, 65536.0f);
                
                ImGui::BeginDisabled();
                ImGui::DragFloat("Max View Distance", &maxViewDistance, 10.0f, 100.0f, 65536.0f);
                ImGui::DragFloat("LOD Distance", &lodDist, 10.0f, 10.0f, 5000.0f);
                ImGui::EndDisabled();
            } else {
                ImGui::DragFloat("Max View Distance", &maxViewDistance, 10.0f, 100.0f, 65536.0f);
                ImGui::DragFloat("LOD Distance", &lodDist, 10.0f, 10.0f, 5000.0f);
            }
            ImGui::DragFloat("LOD Dropoff", &lodDropoff, 0.01f, 0.01f, 1.0f);
            
            ImGui::Separator();
            ImGui::Text("Est. Memory Usage: %zu MB", currentMemoryMB);
            ImGui::Text("Currently Loaded Points: %zu", currentLoadedPoints);
        }

        updateStatsCache();
        ImGui::TextUnformatted(cachedStats.c_str());

        ImGui::EndChild();
    }

    void renderPreviewPanel() {
        ImGui::BeginChild("PreviewChild", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        
        livePreview();
        if (textureInitialized) {
            float aspect = (float)currentPreviewFrame.getWidth() / (float)currentPreviewFrame.getHeight();
            float availWidth = ImGui::GetContentRegionAvail().x;
            ImGui::Image((void*)(intptr_t)textu, ImVec2(availWidth, availWidth / aspect));
        }

        ImGui::EndChild();
    }

    void applyDebugColorMode() {
        if (sim.config.surfaceNodes.empty()) return;

        float minNoise = std::numeric_limits<float>::max();
        float maxNoise = std::numeric_limits<float>::lowest();

        if (currentColorMode == DebugColorMode::NOISE) {
            for (const auto& pos : sim.config.surfaceNodes) {
                auto pt = sim.grid.find(pos, sim.config.voxelSize * 0.5f);
                if (!pt) continue;
                if (pt->data.noiseDisplacement < minNoise) minNoise = pt->data.noiseDisplacement;
                if (pt->data.noiseDisplacement > maxNoise) maxNoise = pt->data.noiseDisplacement;
            }
        }

        for (auto& pos : sim.config.surfaceNodes) {
            auto pt = sim.grid.find(pos, sim.config.voxelSize * 0.5f);
            if (!pt) continue;
            auto& p = pt->data;
            v3 originColor = sim.getOriginColor(pt);
            v3 color = originColor;
            
            switch (currentColorMode) {
                case DebugColorMode::PLATES:
                    if (pt->objectId != -1 && pt->objectId < sim.plates.size()) {
                        color = sim.plates[pt->objectId].debugColor;
                    } else {
                        color = v3(0.5f, 0.5f, 0.5f);
                    }
                    break;
                case DebugColorMode::NOISE: {
                    float t = 0.5f;
                    if (maxNoise > minNoise) t = (p.noiseDisplacement - minNoise) / (maxNoise - minNoise);
                    color = v3(t, t, t);
                    break;
                }
                case DebugColorMode::IMPACTS: {
                    color = v3(p.impactHeat, p.impactShock, p.impactDebris);
                    color = v3(0.2f, 0.2f, 0.2f) + color * 0.8f;
                    
                    color = color.cwiseMin(1.0f).cwiseMax(0.0f);
                    break;
                }
                case DebugColorMode::WATER: {
                    float w = std::min(1.0f, p.water * 0.1f);
                    if (w > 0.0f) color = v3(0.1f, 0.2f, 0.4f + w * 0.6f);
                    else color = v3(0.4f, 0.4f, 0.4f); 
                    break;
                }
                case DebugColorMode::TEMPERATURE: {
                    float t = (p.temperature + 20.0f) / 50.0f;
                    t = std::clamp(t, 0.0f, 1.0f);
                    color = v3(t, 0.2f, 1.0f - t); 
                    break;
                }
                case DebugColorMode::MOISTURE: {
                    float m = std::min(1.0f, p.moisture * 0.05f);
                    color = v3(1.0f - m, 1.0f, 1.0f - m); 
                    break;
                }
                case DebugColorMode::MATERIALS: {
                    float sand = static_cast<float>(p.materials(0,0));
                    float rock = static_cast<float>(p.materials(3,0));
                    float silt = static_cast<float>(p.materials(1,0));
                    float sum = sand + rock + silt + 0.001f;
                    color = v3(
                        (sand + rock * 0.5f) / sum, 
                        (silt + rock * 0.5f) / sum, 
                        (rock * 0.5f) / sum
                    );
                    break;
                }
                case DebugColorMode::BASE:
                default:
                    color = originColor;
                    break;
            }

            sim.grid.waitForIdle();
            // sim.grid.queuedsetColor(p.currentPos, color);
            sim.grid.setColor(p.currentPos, color);
                // snf++;
            // sim.grid.update(p.currentPos, p.currentPos, p, true, color, sim.config.voxelSize, true, -2, false, 0.0f, 0.0f, 0.0f);
        }
        for (auto& pos : sim.config.interpolatedNodes) {
            auto pt = sim.grid.find(pos, sim.config.voxelSize * 0.5f);
            if (!pt) continue;
            auto& p = pt->data;
            v3 originColor = sim.getOriginColor(pt);
            v3 color = originColor;
            
            switch (currentColorMode) {
                case DebugColorMode::PLATES:
                    if (pt->objectId != -1 && pt->objectId < sim.plates.size()) {
                        color = sim.plates[pt->objectId].debugColor;
                    } else {
                        color = v3(0.5f, 0.5f, 0.5f);
                    }
                    break;
                case DebugColorMode::NOISE: {
                    float t = 0.5f;
                    if (maxNoise > minNoise) t = (p.noiseDisplacement - minNoise) / (maxNoise - minNoise);
                    color = v3(t, t, t);
                    break;
                }
                case DebugColorMode::IMPACTS: {
                    color = v3(p.impactHeat, p.impactShock, p.impactDebris);
                    color = v3(0.2f, 0.2f, 0.2f) + color * 0.8f;
                    
                    color = color.cwiseMin(1.0f).cwiseMax(0.0f);
                    break;
                }
                case DebugColorMode::WATER: {
                    float w = std::min(1.0f, p.water * 0.1f);
                    if (w > 0.0f) color = v3(0.1f, 0.2f, 0.4f + w * 0.6f);
                    else color = v3(0.4f, 0.4f, 0.4f); 
                    break;
                }
                case DebugColorMode::TEMPERATURE: {
                    float t = (p.temperature + 20.0f) / 50.0f;
                    t = std::clamp(t, 0.0f, 1.0f);
                    color = v3(t, 0.2f, 1.0f - t); 
                    break;
                }
                case DebugColorMode::MOISTURE: {
                    float m = std::min(1.0f, p.moisture * 0.05f);
                    color = v3(1.0f - m, 1.0f, 1.0f - m); 
                    break;
                }
                case DebugColorMode::MATERIALS: {
                    float sand = static_cast<float>(p.materials(0,0));
                    float rock = static_cast<float>(p.materials(3,0));
                    float silt = static_cast<float>(p.materials(1,0));
                    float sum = sand + rock + silt + 0.001f;
                    color = v3(
                        (sand + rock * 0.5f) / sum, 
                        (silt + rock * 0.5f) / sum, 
                        (rock * 0.5f) / sum
                    );
                    break;
                }
                case DebugColorMode::BASE:
                default:
                    color = originColor;
                    break;
            }

            sim.grid.waitForIdle();
            // sim.grid.queuedsetColor(p.currentPos, color);
            sim.grid.setColor(p.currentPos, color);
            // sim.grid.update(p.currentPos, p.currentPos, p, true, color, sim.config.voxelSize, true, -2, false, 0.0f, 0.0f, 0.0f);
        }
        // if (snf > 0 || inf > 0) {
        //     std::cout << snf << " original nodes failed to set" << std::endl;
        //     std::cout << inf << " interpolated nodes failed to set" << std::endl;
        // }
    }

    void generateDebugMap(DebugMapMode mode) {
        if (mode == DebugMapMode::NONE || sim.config.surfaceNodes.empty()) return;

        int w = 512;
        int h = 348;
        std::vector<float> depths(w * h, -1.0f);
        float minD = std::numeric_limits<float>::max();
        float maxD = std::numeric_limits<float>::lowest();

        for (const auto& pos : sim.config.surfaceNodes) {
            auto pt = sim.grid.find(pos, sim.config.voxelSize * 0.5f);
            if (!pt || !pt->isActive()) continue;
            auto& p = pt->data;

            v3 p_pos;
            switch(mode) {
                case DebugMapMode::BASE: 
                    p_pos = p.altPos->originalPos;
                    break;
                case DebugMapMode::NOISE: 
                    p_pos = p.altPos->noisePos;
                    break;
                case DebugMapMode::TECTONIC: 
                    p_pos = p.altPos->tectonicPos;
                    break;
                case DebugMapMode::CURRENT:
                default: 
                    p_pos = p.currentPos;
                    break;
            }
            float d = p_pos.norm();
            if (d < minD) minD = d;
            if (d > maxD) maxD = d;
        }

        for (const auto& pos : sim.config.surfaceNodes) {
            auto pt = sim.grid.find(pos, sim.config.voxelSize * 0.5f);
            if (!pt || !pt->isActive()) continue;
            auto& p = pt->data;

            v3 p_pos;
            switch(mode) {
                case DebugMapMode::BASE: 
                    p_pos = p.altPos->originalPos;
                    break;
                case DebugMapMode::NOISE: 
                    p_pos = p.altPos->noisePos;
                    break;
                case DebugMapMode::TECTONIC: 
                    p_pos = p.altPos->tectonicPos;
                    break;
                case DebugMapMode::TECTONICCOLOR: 
                    if (pt->objectId != -1 && pt->objectId < sim.plates.size()) {
                        p_pos = sim.plates[pt->objectId].debugColor;
                    } else {
                        p_pos = v3(0.5f, 0.5f, 0.5f);
                    }
                    break;
                case DebugMapMode::CURRENT:
                default:
                    p_pos = p.currentPos;
                    break;
            }

            float d = p_pos.norm();
            v3 n = p.altPos->originalPos.normalized();
            
            float u = 0.5f + std::atan2(n.z(), n.x()) / (2.0f * static_cast<float>(M_PI));
            float v = 0.5f - std::asin(n.y()) / static_cast<float>(M_PI);

            int px = std::clamp(static_cast<int>(u * w), 0, w - 1);
            int py = std::clamp(static_cast<int>(v * h), 0, h - 1);

            float normalizedD = (maxD > minD) ? (d - minD) / (maxD - minD) : 0.5f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = px + dx;
                    int ny = py + dy;
                    
                    if (nx < 0) nx += w;
                    if (nx >= w) nx -= w;
                    
                    if (ny >= 0 && ny < h) {
                        int idx = ny * w + nx;
                        if (depths[idx] < 0.0f || normalizedD > depths[idx]) {
                            depths[idx] = normalizedD;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < w * h; i++) {
            if (depths[i] < 0.0f) depths[i] = 0.0f;
        }

        std::vector<uint8_t> pixels(w * h * 3);
        for (int i = 0; i < w * h; i++) {
            uint8_t val = static_cast<uint8_t>(depths[i] * 255.0f);
            pixels[i * 3 + 0] = val;
            pixels[i * 3 + 1] = val;
            pixels[i * 3 + 2] = val;
        }

        mapFrame = frame(w, h, frame::colormap::RGB);
        mapFrame.setData(pixels);

        if (mapTexture == 0) {
            glGenTextures(1, &mapTexture);
        }
        
        glBindTexture(GL_TEXTURE_2D, mapTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, mapFrame.getData().data());
    }

    void applyNoise(const NoisePreviewState& noiseState) {
        TIME_FUNCTION;
        auto triplanarNoise = [&](const Eigen::Vector3f& pos) -> float {
            PNoise2 gen(noiseState.masterSeed);
            
            Eigen::Vector3f n = pos.normalized();
            Eigen::Vector3f blend = n.cwiseAbs();
            float sum = blend.x() + blend.y() + blend.z();
            blend /= sum;

            Eigen::Vector3f offsetPos = pos + Eigen::Vector3f(noiseState.offset[0], noiseState.offset[1], 0.0f);
            float vXY = sim.evaluate2DStack(Eigen::Vector2f(offsetPos.x(), offsetPos.y()), noiseState, gen);
            float vXZ = sim.evaluate2DStack(Eigen::Vector2f(offsetPos.x(), offsetPos.z()), noiseState, gen);
            float vYZ = sim.evaluate2DStack(Eigen::Vector2f(offsetPos.y(), offsetPos.z()), noiseState, gen);

            // Blend results
            return vYZ * blend.x() + vXZ * blend.y() + vXY * blend.z();
        };
        sim._applyNoise(triplanarNoise);
        applyDebugColorMode();
    }

    void livePreview() {
        std::lock_guard<std::mutex> lock(PreviewMutex);
        updatePreview = true;
        
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoAdjustTime).count() > 1500) {
            lastAutoAdjustTime = now;
            
            currentMemoryMB = sim.grid.getEstimatedMemoryUsageMB();
            currentLoadedPoints = sim.grid.getLoadedPointCount();
            
            if (autoAdjustDistance) {
                if (currentMemoryMB > (size_t)targetMemoryMB) {
                    maxViewDistance *= 0.85f;
                    lodDist *= 0.85f;
                } else if (currentMemoryMB < (size_t)targetMemoryMB * 0.8f) {
                    maxViewDistance *= 1.05f;
                    lodDist *= 1.05f;
                }
                
                maxViewDistance = std::clamp(maxViewDistance, minAutoDistance, maxAutoDistance);
                lodDist = std::clamp(lodDist, 10.0f, maxViewDistance * 0.5f);
            }
        }

        sim.grid.setLODMinDistance(lodDist);
        sim.grid.setLODFalloff(lodDropoff);
        sim.grid.setMaxDistance(maxViewDistance);
        
        float invFrameRate = 1 / framerate;
        
        if (slowRender) {
            #ifdef VULKAN_SUPPORT
            currentPreviewFrame = sim.grid.renderFrameVulkan(cam, outHeight, outWidth, frame::colormap::RGB, 3, reflectCount, useLod);
            #else
            currentPreviewFrame = sim.grid.renderFrame(cam, outHeight, outWidth, frame::colormap::RGB, 3, reflectCount, globalIllumination, useLod);
            #endif
        } else {
            #ifdef VULKAN_SUPPORT
            currentPreviewFrame = sim.grid.fastRenderFrameVulkan(cam, outHeight, outWidth, frame::colormap::RGB);
            #else
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, outHeight, outWidth, frame::colormap::RGB);
            #endif
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

    void resetView() {
        cam.origin = Vector3f(sim.config.gridSizeCube, sim.config.gridSizeCube, sim.config.gridSizeCube);
        Vector3f center(sim.config.gridSizeCube / 2.0f, sim.config.gridSizeCube / 2.0f, sim.config.gridSizeCube / 2.0f);
        cam.lookAt(center);
    }

    void simulateTectonics() {
        sim.grid.waitForIdle();
        
        currentColorMode = DebugColorMode::PLATES;
        
        sim.grid.waitForIdle();
        sim.assignSeeds();
        sim.buildAdjacencyList();
        if (platesUseCellular) {
            sim.growPlatesCellular();
        } else sim.growPlatesRandom();
        if (doFixPlates) sim.fixBoundaries();
        sim.extraplateste();
        sim.boundaryStress();
        sim.finalizeApplyResults();

        applyDebugColorMode();
        tectonicGenned = true;
        if(currentMapMode != DebugMapMode::NONE) generateDebugMap(currentMapMode);
    }

    void interpolateSurface() {
        sim.interpolateSurface();
        applyDebugColorMode();
    }

    void fillPlanet() {
        sim.fillPlanet();
    }

    void updateStatsCache() {
        std::stringstream gridstats;
        sim.grid.printStats(gridstats);
        cachedStats = gridstats.str();
        lastStatsUpdate = std::chrono::steady_clock::now();
        statsNeedUpdate = false;
    }
};

#endif