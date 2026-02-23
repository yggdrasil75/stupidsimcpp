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
    int outWidth = 512;
    int outHeight = 512;
    float fps = 60;
    int rayCount = 3;
    int reflectCount = 4;
    bool slowRender = false;
    float lodDist = 4096.0f;
    float lodDropoff = 0.001f;
    bool globalIllumination = false;
    bool useLod = true;
    std::map<int, bool> keyStates;
    float deltaTime = 0.16f;
    bool orbitEquator = false;
    float rotationRadius = 2500;
    float angle = 0.0f;
    const float ω = (std::pow(M_PI, 2) / 30) / 10;

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
        sim.grid.clear();
    }

    void renderUI(GLFWwindow* window) {
        ImGui::Begin("Planet Simulation");
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
        
        if (ImGui::CollapsingHeader("Base Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat("Radius", &sim.config.radius, 1.0f, 10.0f, 10000.0f);
            ImGui::InputInt("Surface Points", &sim.config.surfacePoints);
            ImGui::DragFloat("Voxel Size", &sim.config.voxelSize, 0.1f, 0.1f, 100.0f);
            ImGui::ColorEdit3("Base Color", sim.config.color.data());
            
            ImGui::Separator();
            
            if (ImGui::Button("1. Generate Fib Sphere", ImVec2(-1, 40))) {
                sim.generateFibSphere();
            }
            ImGui::Text("Current Step: %d", sim.config.currentStep);
            ImGui::Text("Nodes: %zu", sim.config.surfaceNodes.size());
        }
        if (ImGui::Button("Apply Tectonics")) {
            simulateTectonics();
        }

        if (ImGui::CollapsingHeader("Physics Parameters")) {
            ImGui::DragFloat("Gravity (G)", &sim.config.G_ATTRACTION, 0.1f);
            ImGui::DragFloat("Time Step", &sim.config.TIMESTEP, 0.001f, 0.0001f, 0.1f);
            ImGui::DragFloat("Viscosity", &sim.config.dampingFactor, 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("Pressure Stiffness", &sim.config.pressureStiffness, 10.0f);
            ImGui::DragInt("Num Plates", &sim.config.numPlates, 1, 1, 50);
        }

        if (ImGui::CollapsingHeader("Tectonic Simulation")) {
            ImGui::DragInt("Num Plates", &sim.config.numPlates, 1, 1, 100);
            ImGui::DragFloat("Plate Randomness", &sim.config.plateRandom, 0.01f, 0.0f, 2.0f);
            ImGui::DragInt("Smoothing Passes", &sim.config.smoothingPasses, 1, 0, 10);
            ImGui::DragFloat("Mountain Height", &sim.config.mountHeight, 1.0f, 0.0f, 1000.0f);
            ImGui::DragFloat("Valley Depth", &sim.config.valleyDepth, 1.0f, -1000.0f, 0.0f);
            ImGui::DragFloat("Transform Roughness", &sim.config.transformRough, 1.0f, 0.0f, 500.0f);
            ImGui::DragInt("Stress Passes", &sim.config.stressPasses, 1, 0, 20);

            if (ImGui::Button("2. Simulate Tectonics", ImVec2(-1, 40))) {
                simulateTectonics();
            }
        }

        if (ImGui::CollapsingHeader("Camera Controls")) {
            ImGui::DragFloat3("Origin", cam.origin.data());
            ImGui::DragFloat3("Direction", cam.direction.data(), 0.0001f, -1.0f, 1.0f);
            ImGui::DragFloat("Movement Speed", &cam.movementSpeed, 0.1f, 1.0f, 500.0f);
            ImGui::DragFloat("Rotation Speed", &cam.rotationSpeed, M_1_PI, M_1_PI, M_PI);
            ImGui::InputFloat("Rotation Distance", &rotationRadius, 10, 100);

            ImGui::Checkbox("Use Slower Render", &slowRender);

            if (ImGui::Button("Focus on Planet")) {
                v3 target(sim.config.center);
                v3 newDir = (target - cam.origin).normalized();
                cam.direction = newDir;
            }

            if (ImGui::Button(orbitEquator ? "Stop Equator" : "Orbit Equator")) orbitEquator = !orbitEquator;
        }

        livePreview();
        if (textureInitialized) {
            float aspect = (float)currentPreviewFrame.getWidth() / (float)currentPreviewFrame.getHeight();
            float availWidth = ImGui::GetContentRegionAvail().x;
            ImGui::Image((void*)(intptr_t)textu, ImVec2(availWidth, availWidth / aspect));
        }

        ImGui::End();
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
    }

    void livePreview() {
        std::lock_guard<std::mutex> lock(PreviewMutex);
        updatePreview = true;
        
        sim.grid.setLODMinDistance(lodDist);
        sim.grid.setLODFalloff(lodDropoff);
        
        if (slowRender) {
            currentPreviewFrame = sim.grid.renderFrame(cam, outWidth, outHeight, frame::colormap::RGB, rayCount, reflectCount, globalIllumination, useLod);
        } else {
            currentPreviewFrame = sim.grid.fastRenderFrame(cam, outWidth, outHeight, frame::colormap::RGB);
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
        sim.assignSeeds();
        // sim.growPlatesCellular();
        sim.growPlatesRandom();
        //sim.fixBoundaries();
        sim.extraplateste();
        sim.finalizeApplyResults();
    }
};

#endif