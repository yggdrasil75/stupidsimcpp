#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_set>
#include <iomanip>
#include <filesystem>
#include "../util/grid/grid2.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/timing_decorator.cpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

#include <thread>
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::mutex m;
std::atomic<bool> isGenerating{false};
std::future<void> generationFuture;

std::mutex previewMutex;
std::atomic<bool> updatePreview{false};
frame currentPreviewFrame;
GLuint textu = 0;
std::string previewText;

struct Shared {
    std::mutex mutex;
    Grid2 grid;
    bool hasNewFrame = false;
    int currentFrame = 0;
    float currentTime = 0.0f;
};

struct SimulationConfig {
    int width = 1024;
    int height = 1024;
    int totalFrames = 480;
    float fps = 30.0f;
    float atomDensity = 0.3f;
    int simulationStepsPerFrame = 1;
    
    // Physics parameters
    float timeStep = 0.016f;
    float coulombStrength = 1.0f;
    float gravityStrength = 1.0e-6f;
    float dampingFactor = 0.99f;
    float temperature = 300.0f;
    
    // Element distribution
    float hydrogenProb = 0.4f;
    float heliumProb = 0.2f;
    float carbonProb = 0.15f;
    float oxygenProb = 0.15f;
    float ironProb = 0.1f;
    
    // Interaction settings
    bool enableCoulomb = true;
    bool enableGravity = true;
    bool enableFusion = true;
    bool enableElectronTransfer = true;
};

Grid2 setupAtomicGrid(SimulationConfig config) {
    TIME_FUNCTION;
    Grid2 grid;
    
    // Set physics parameters
    grid.setPhysicsParameters(config.timeStep, config.coulombStrength, 
                            config.gravityStrength, config.dampingFactor);
    
    // Set simulation bounds
    Vec2 minBound(0, 0);
    Vec2 maxBound(config.width, config.height);
    grid.setSimulationBounds(minBound, maxBound);
    
    // Set default background (dark space)
    grid.setDefault(0.0f, 0.0f, 0.1f, 1.0f);
    
    // Generate random atoms
    grid.generateRandomAtoms(0, 0, config.width, config.height, config.atomDensity);
    
    std::cout << "Generated atomic grid with size: " << config.width << "x" << config.height << std::endl;
    std::cout << "Atom density: " << config.atomDensity << std::endl;
    
    return grid;
}

void Preview(Grid2& grid) {
    TIME_FUNCTION;
    frame rgbData = grid.getGridAsFrame(frame::colormap::RGB);
    std::cout << "Frame size: " << rgbData.getWidth() << "x" << rgbData.getHeight() << std::endl;
    bool success = BMPWriter::saveBMP("output/atomic_preview.bmp", rgbData);
    if (!success) {
        std::cout << "Failed to save preview image!" << std::endl;
    }
}

void livePreview(const Grid2& grid) {
    std::lock_guard<std::mutex> lock(previewMutex);
    
    currentPreviewFrame = grid.getGridAsFrame(frame::colormap::RGBA);
    
    if (textu == 0) {
        glGenTextures(1, &textu);
    }
    
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
    
    updatePreview = true;
}

bool exportavi(std::vector<frame> frames, SimulationConfig config) {
    TIME_FUNCTION;
    std::string filename = "output/atomic_simulation.avi";
    
    std::cout << "Frame count: " << frames.size() << std::endl;
    
    // Create output directory
    std::filesystem::path dir = "output";
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cout << "Failed to create output directory!" << std::endl;
            return false;
        }
    }
    
    bool success = AVIWriter::saveAVIFromCompressedFrames(filename, frames, 
                                                         frames[0].getWidth(), 
                                                         frames[0].getHeight(), 
                                                         config.fps);
    
    if (success) {
        if (std::filesystem::exists(filename)) {
            auto file_size = std::filesystem::file_size(filename);
            std::cout << "\nAVI file created successfully: " << filename 
                      << " (" << file_size << " bytes, " 
                      << std::fixed << std::setprecision(2) << (file_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        } 
    } else {
        std::cout << "Failed to save AVI file!" << std::endl;
    }
    
    return success;
}

void atomicSimulationLoop(const SimulationConfig& config, Shared& state) {
    TIME_FUNCTION;
    isGenerating = true;
    
    try {
        // Setup atomic grid
        Grid2 grid = setupAtomicGrid(config);
        
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.grid = grid;
            state.hasNewFrame = true;
            state.currentFrame = 0;
            state.currentTime = 0.0f;
        }
        
        std::cout << "Generated atomic grid" << std::endl;
        
        // Get initial statistics
        size_t totalAtoms, totalProtons, totalNeutrons, totalElectrons;
        float totalCharge, totalMass;
        grid.getStatistics(totalAtoms, totalProtons, totalNeutrons, 
                          totalElectrons, totalCharge, totalMass);
        
        std::cout << "Initial Statistics:" << std::endl;
        std::cout << "  Total atoms: " << totalAtoms << std::endl;
        std::cout << "  Total protons: " << totalProtons << std::endl;
        std::cout << "  Total neutrons: " << totalNeutrons << std::endl;
        std::cout << "  Total electrons: " << totalElectrons << std::endl;
        std::cout << "  Total charge: " << totalCharge << " e" << std::endl;
        std::cout << "  Total mass: " << totalMass << " kg" << std::endl;
        
        // Save initial preview
        Preview(grid);
        std::cout << "Saved initial preview" << std::endl;
        
        std::vector<frame> frames;
        
        for (int frameNum = 0; frameNum < config.totalFrames; frameNum++) {
            // Check if we should stop the generation
            if (!isGenerating) {
                std::cout << "Simulation cancelled at frame " << frameNum << std::endl;
                return;
            }
            
            // Multiple simulation steps per frame for smoother physics
            for (int step = 0; step < config.simulationStepsPerFrame; step++) {
                // Update physics
                grid.updatePhysics(config.timeStep);
                
                // Simulate atomic interactions
                grid.simulateAtomicInteractions();
                
                // Update time
                state.currentTime += config.timeStep;
            }
            
            // Update shared state
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.grid = grid;
                state.hasNewFrame = true;
                state.currentFrame = frameNum;
            }
            
            // Save frame every 5 frames or at the end
            if (frameNum % 5 == 0 || frameNum == config.totalFrames - 1) {
                frame currentFrame = grid.getGridAsFrame(frame::colormap::BGR);
                frames.push_back(currentFrame);
                
                // Optional: Save individual frames for debugging
                if (frameNum % 50 == 0) {
                    std::string frameFilename = "output/atomic_frame_" + 
                                               std::to_string(frameNum) + ".bmp";
                    BMPWriter::saveBMP(frameFilename, currentFrame);
                }
                
                std::cout << "Processed frame " << frameNum + 1 << "/" 
                         << config.totalFrames << std::endl;
            }
            
            // Update statistics periodically
            if (frameNum % 100 == 0) {
                grid.getStatistics(totalAtoms, totalProtons, totalNeutrons, 
                                  totalElectrons, totalCharge, totalMass);
                
                std::cout << "\nFrame " << frameNum << " Statistics:" << std::endl;
                std::cout << "  Total atoms: " << totalAtoms << std::endl;
                std::cout << "  Net charge: " << totalCharge << " e" << std::endl;
            }
        }
        
        // Export final video
        exportavi(frames, config);
        
        std::cout << "\nSimulation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in atomic simulation: " << e.what() << std::endl;
    }
    
    isGenerating = false;
}

void cancelGeneration() {
    if (isGenerating) {
        isGenerating = false;
        if (generationFuture.valid()) {
            auto status = generationFuture.wait_for(std::chrono::milliseconds(500));
            if (status != std::future_status::ready) {
                std::cout << "Waiting for simulation thread to finish..." << std::endl;
            }
        }
    }
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    // GL version setup
    #if defined(IMGUI_IMPL_OPENGL_ES2)
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(IMGUI_IMPL_OPENGL_ES3)
        const char* glsl_version = "#version 300 es";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    #elif defined(__APPLE__)
        const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #else
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    #endif
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Atomic Simulation", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    // Setup ImGui
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Simulation parameters
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.1f, 0.1f, 0.15f, 1.00f);
    
    // Configuration variables
    static SimulationConfig config;
    static int simulationMode = 0; // 0: Random atoms, 1: Custom setup
    static float previewScale = 0.5f;
    static bool showStatistics = true;
    static bool showPhysicsControls = true;
    
    Shared state;
    std::future<void> simulationThread;
    previewText = "Ready to simulate";
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        {
            ImGui::Begin("Atomic Simulation Controls");
            
            ImGui::Text("Simulation Settings");
            ImGui::Separator();
            
            ImGui::SliderInt("Width", &config.width, 256, 4096);
            ImGui::SliderInt("Height", &config.height, 256, 4096);
            ImGui::SliderInt("Frame Count", &config.totalFrames, 10, 5000);
            ImGui::SliderFloat("FPS", &config.fps, 10.0f, 60.0f);
            ImGui::SliderFloat("Atom Density", &config.atomDensity, 0.01f, 0.8f);
            ImGui::SliderInt("Steps/Frame", &config.simulationStepsPerFrame, 1, 10);
            
            ImGui::Separator();
            ImGui::Text("Physics Parameters");
            
            ImGui::SliderFloat("Time Step", &config.timeStep, 0.001f, 0.1f, "%.3f");
            ImGui::SliderFloat("Coulomb Strength", &config.coulombStrength, 0.0f, 10.0f);
            ImGui::SliderFloat("Gravity Strength", &config.gravityStrength, 0.0f, 1.0e-3f, "%.6f");
            ImGui::SliderFloat("Damping", &config.dampingFactor, 0.9f, 1.0f);
            ImGui::SliderFloat("Temperature", &config.temperature, 100.0f, 10000.0f);
            
            ImGui::Separator();
            ImGui::Text("Element Distribution");
            
            ImGui::SliderFloat("Hydrogen %", &config.hydrogenProb, 0.0f, 1.0f);
            ImGui::SliderFloat("Helium %", &config.heliumProb, 0.0f, 1.0f);
            ImGui::SliderFloat("Carbon %", &config.carbonProb, 0.0f, 1.0f);
            ImGui::SliderFloat("Oxygen %", &config.oxygenProb, 0.0f, 1.0f);
            ImGui::SliderFloat("Iron %", &config.ironProb, 0.0f, 1.0f);
            
            // Normalize probabilities
            float total = config.hydrogenProb + config.heliumProb + 
                         config.carbonProb + config.oxygenProb + config.ironProb;
            if (total > 0) {
                config.hydrogenProb /= total;
                config.heliumProb /= total;
                config.carbonProb /= total;
                config.oxygenProb /= total;
                config.ironProb /= total;
            }
            
            ImGui::Separator();
            ImGui::Text("Interaction Settings");
            
            ImGui::Checkbox("Coulomb Forces", &config.enableCoulomb);
            ImGui::Checkbox("Gravity", &config.enableGravity);
            ImGui::Checkbox("Fusion Reactions", &config.enableFusion);
            ImGui::Checkbox("Electron Transfer", &config.enableElectronTransfer);
            
            ImGui::Separator();
            ImGui::SliderFloat("Preview Scale", &previewScale, 0.1f, 2.0f);
            
            // Simulation controls
            if (isGenerating) {
                ImGui::BeginDisabled();
            }
            
            if (ImGui::Button("Start Atomic Simulation")) {
                simulationThread = std::async(std::launch::async, atomicSimulationLoop, 
                                            config, std::ref(state));
                previewText = "Starting atomic simulation...";
            }
            
            if (isGenerating) {
                ImGui::EndDisabled();
                
                ImGui::SameLine();
                if (ImGui::Button("Cancel Simulation")) {
                    cancelGeneration();
                    previewText = "Cancelling simulation...";
                }
                
                // Update preview from simulation thread
                bool hasNewFrame = false;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (state.hasNewFrame) {
                        livePreview(state.grid);
                        state.hasNewFrame = false;
                        previewText = "Simulating... Frame: " + 
                                     std::to_string(state.currentFrame) + 
                                     ", Time: " + std::to_string(state.currentTime) + "s";
                    }
                }
            }
            
            ImGui::Separator();
            ImGui::Text("Status: %s", previewText.c_str());
            
            // Display preview image
            if (textu != 0) {
                ImVec2 imageSize = ImVec2(config.width * previewScale, 
                                         config.height * previewScale);
                ImVec2 uv_min = ImVec2(0.0f, 0.0f);
                ImVec2 uv_max = ImVec2(1.0f, 1.0f);
                ImGui::Image((void*)(intptr_t)textu, imageSize, uv_min, uv_max);
            } else {
                ImGui::Text("No preview available");
                ImGui::Text("Start simulation to see atomic interactions");
            }
            
            // Display statistics if available
            if (showStatistics) {
                ImGui::Separator();
                ImGui::Text("Simulation Statistics");
                // Statistics would be updated from the simulation thread
            }
            
            ImGui::End();
        }
        
        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, 
                    clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    cancelGeneration();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    if (textu != 0) {
        glDeleteTextures(1, &textu);
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}