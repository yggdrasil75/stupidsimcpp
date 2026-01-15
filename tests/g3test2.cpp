#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

#include "../util/grid/grid3.hpp"
#include "../util/grid/g3_serialization.hpp"
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
    int gridWidth = 64;
    int gridHeight = 64;
    int gridDepth = 64;
    float fps = 30.0f;
    PNoise2 noise = PNoise2(42);
};

std::mutex PreviewMutex;
GLuint textu = 0;
bool textureInitialized = false;
bool updatePreview = false;
bool previewRequested = false;

// Add AVI recording variables
std::atomic<bool> isRecordingAVI{false};
std::atomic<bool> recordingRequested{false};
std::atomic<int> recordingFramesRemaining{0};
std::vector<frame> recordedFrames;
std::mutex recordingMutex;

struct Shared {
    std::mutex mutex;
    VoxelGrid grid;
};

void setup(defaults config, VoxelGrid& grid) {
    uint8_t threshold = 0.1 * 255;
    grid.resize(config.gridWidth, config.gridHeight, config.gridDepth);
    std::cout << "Generating grid of size " << config.gridWidth << "x" << config.gridHeight << "x" << config.gridDepth << std::endl;
    for (int z = 0; z < config.gridDepth; ++z) {
        if (z % 64 == 0) {
            std::cout << "Processing layer " << z << " of " << config.gridDepth << std::endl;
        }

        for (int y = 0; y < config.gridHeight; ++y) {
            for (int x = 0; x < config.gridWidth; ++x) {
                uint8_t r = config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 64, static_cast<float>(y) / config.gridHeight / 64, static_cast<float>(z) / config.gridDepth / 64)) * 255;
                uint8_t g = config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 32, static_cast<float>(y) / config.gridHeight / 32, static_cast<float>(z) / config.gridDepth / 32)) * 255;
                uint8_t b = config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 16, static_cast<float>(y) / config.gridHeight / 16, static_cast<float>(z) / config.gridDepth / 16)) * 255;
                uint8_t a = config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 8 , static_cast<float>(y) / config.gridHeight / 8 , static_cast<float>(z) / config.gridDepth / 8)) * 255;
                //Vec4ui8 noisecolor = config.noise.permuteColor(Vec3f( static_cast<float>(x) / 64, static_cast<float>(y) / 64, static_cast<float>(z) / 64));
                if (a > threshold) {
                    //std::cout << "setting a position" << std::endl;
                    grid.set(Vec3i(x, y, z), true, Vec3ui8(r,g,b));
                }
            }
        }
    }
    std::cout << "Noise grid generation complete!" << std::endl;
    grid.serializeToFile("output/gridsave.ygg3");
    grid.printStats();
}

void livePreview(VoxelGrid& grid, defaults& config, const Camera& cam) {
    std::lock_guard<std::mutex> lock(PreviewMutex);
    updatePreview = true;
    frame currentPreviewFrame = grid.renderFrame(cam, Vec2i(config.outWidth, config.outHeight), frame::colormap::RGB);
    
    glGenTextures(1, &textu);
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                 0, GL_RGB, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
    
    updatePreview = false;
    textureInitialized = true;
    if (isRecordingAVI) {
        std::lock_guard<std::mutex> recLock(recordingMutex);
        currentPreviewFrame.compressFrameLZ78();
        recordedFrames.push_back(currentPreviewFrame);
    }
}

bool savePreview(VoxelGrid& grid, defaults& config, const Camera& cam) {
    TIME_FUNCTION;
    
    // Render the view
    frame output = grid.renderFrame(cam, Vec2i(config.outWidth, config.outHeight), frame::colormap::RGB);
    //grid.renderOut(renderBuffer, width, height, cam);
    
    // Save to BMP
    bool success = BMPWriter::saveBMP("output/save.bmp", output);
    //bool success = BMPWriter::saveBMP(filename, renderBuffer, width, height);
    
    // if (success) {
    //     std::cout << "Saved: " << filename << std::endl;
    // } else {
    //     std::cout << "Failed to save: " << filename << std::endl;
    // }
    
    return success;
}

void startAVIRecording(int frameCount) {
    std::lock_guard<std::mutex> lock(recordingMutex);
    recordedFrames.clear();
    recordedFrames.reserve(frameCount);
    recordingFramesRemaining = frameCount;
    recordingRequested = true;
}

void stopAndSaveAVI(defaults& config, const std::string& filename) {
    std::lock_guard<std::mutex> lock(recordingMutex);
    
    if (!recordedFrames.empty()) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        std::string finalFilename = "output/recording_" + std::to_string(timestamp) + ".avi";
        
        
        std::cout << "Saving AVI with " << recordedFrames.size() << " frames..." << std::endl;
        
        // Save using the new streaming method
        bool success = AVIWriter::saveAVIFromCompressedFrames(finalFilename, recordedFrames, config.outWidth, config.outHeight, config.fps);
        
        if (success) {
            std::cout << "AVI saved to: " << finalFilename << std::endl;
        } else {
            std::cout << "Failed to save AVI: " << finalFilename << std::endl;
        }
        
        recordedFrames.clear();
    }
    
    isRecordingAVI = false;
    recordingFramesRemaining = 0;
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
    VoxelGrid grid;
    bool gridInitialized = false;
    auto supposedGrid = VoxelGrid::deserializeFromFile("output/gridsave.ygg3");
    if (supposedGrid) {
        grid = std::move(*supposedGrid);
        gridInitialized = true;
        config.gridDepth = grid.getDepth();
        config.gridHeight = grid.getHeight();
        config.gridWidth = grid.getWidth();
    }
    
    
    
    Camera cam(Vec3f(config.gridWidth/2.0f, config.gridHeight/2.0f, config.gridDepth/2.0f), Vec3f(0,0,1), Vec3f(0,1,0), 80);
    
    // Variables for camera sliders
    float camX = 0.0f;
    float camY = 0.0f;
    float camZ = 0.0f;
    float camvX = 0.f;
    float camvY = 0.f;
    float camvZ = 0.f;
    //float camYaw = 0.0f;
    //float camPitch = 0.0f;
    bool autoRotate = false;  // Toggle for auto-rotation
    bool autoRotateView = false; // Toggle for auto-rotation of the view only
    float rotationSpeedX = 0.1f;  // Speed for X rotation
    float rotationSpeedY = 0.07f;  // Speed for Y rotation
    float rotationSpeedZ = 0.05f;  // Speed for Z rotation
    float autoRotationTime = 0.0f;  // Timer for auto-rotation
    Vec3f initialViewDir = Vec3f(0, 0, 1);  // Initial view direction
    float rotationRadius = 50.0f;  // Distance from center for rotation
    float yawSpeed = 0.5f;                  // Horizontal rotation speed (degrees per second)
    float pitchSpeed = 0.3f;                // Vertical rotation speed
    float rollSpeed = 0.2f;                 // Roll rotation speed (optional)
    float autoRotationAngle = 0.0f;         // Accumulated rotation angle
    Vec3f initialUpDir = Vec3f(0, 1, 0);    // Initial up direction
    
    // Variables for framerate limiting
    const double targetFrameTime = 1.0 / config.fps; // 30 FPS
    double lastFrameTime = glfwGetTime();
    double accumulator = 0.0;
    
    // For camera movement
    bool cameraMoved = false;
    double lastUpdateTime = glfwGetTime();
    
    // AVI recording variables
    int recordingDurationFrames = 300; // 10 seconds at 30fps
    std::string aviFilename = "output/recording.avi";
    
    // For frame-based timing (not real time)
    int frameCounter = 0;
    float animationTime = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        // Accumulate time
        accumulator += deltaTime;
        
        // Limit framerate
        if (accumulator < targetFrameTime) {
            std::this_thread::sleep_for(std::chrono::duration<double>(targetFrameTime - accumulator));
            currentTime = glfwGetTime();
            accumulator = targetFrameTime;
        }
        
        // Frame-based timing for animations (independent of real time)
        frameCounter++;
        animationTime = frameCounter / config.fps; // Time in seconds based on frame count
        
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        {
            ImGui::Begin("settings");
            
            if(ImGui::CollapsingHeader("output", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderInt("Width", &config.outWidth, 256, 4096);
                ImGui::SliderInt("Height", &config.outHeight, 256, 4096);
                ImGui::SliderFloat("FPS", &config.fps, 1.0f, 120.0f);
            }

            if (ImGui::CollapsingHeader("Grid Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderInt("#Width", &config.gridWidth, 64, 512);
                ImGui::SliderInt("#Height", &config.gridHeight, 64, 512);
                ImGui::SliderInt("#Depth", &config.gridDepth, 64, 512);
            }

            ImGui::Separator();

            if (ImGui::Button("Generate Grid")) {
                setup(config, grid);
                gridInitialized = true;
                // Reset camera to center of grid
                camX = config.gridWidth / 2.0f;
                camY = config.gridHeight / 2.0f;
                camZ = config.gridDepth / 2.0f;
                //camYaw = 0.0f;
                //camPitch = 0.0f;
                
                // Update camera position
                cam.posfor.origin = Vec3f(camX, camY, camZ);
                cam.posfor.direction = Vec3f(camvX, camvY, camvZ);
                // cam.rotateYaw(camYaw);
                // cam.rotatePitch(camPitch);
                
                savePreview(grid, config, cam);
                cameraMoved = true;
            }
            
            // AVI Recording Controls
            ImGui::Separator();
            ImGui::Text("AVI Recording:");
            
            if (!isRecordingAVI) {
                ImGui::InputInt("Frames to Record", &recordingDurationFrames, 30, 300);
                recordingDurationFrames = std::max(30, recordingDurationFrames); // Minimum 1 second at 30fps
                
                if (ImGui::Button("Start AVI Recording")) {
                    startAVIRecording(recordingDurationFrames);
                    ImGui::OpenPopup("Recording Started");
                }
            } else {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "RECORDING");
                ImGui::Text("Frames captured: %d / %d", 
                           recordedFrames.size(), 
                           recordingDurationFrames);
                
                if (ImGui::Button("Stop Recording Early")) {
                    isRecordingAVI = false;
                }
            }
            
            // Display camera controls
            if (gridInitialized) {
                ImGui::Separator();
                ImGui::Text("Camera Controls:");
                
                // Calculate max slider values based on grid size squared
                float maxSliderValueX = config.gridWidth;
                float maxSliderValueY = config.gridHeight;
                float maxSliderValueZ = config.gridDepth;
                float maxSliderValueRotation = 360.0f; // Degrees
                
                ImGui::Text("Position (0 to grid size²):");
                if (ImGui::SliderFloat("Camera X", &camX, 0.0f, maxSliderValueX)) {
                    cameraMoved = true;
                    cam.posfor.origin.x = camX;
                }
                if (ImGui::SliderFloat("Camera Y", &camY, 0.0f, maxSliderValueY)) {
                    cameraMoved = true;
                    cam.posfor.origin.y = camY;
                }
                if (ImGui::SliderFloat("Camera Z", &camZ, 0.0f, maxSliderValueZ)) {
                    cameraMoved = true;
                    cam.posfor.origin.z = camZ;
                }
                
                ImGui::Separator();
                // ImGui::Text("Rotation (degrees):");
                // if (ImGui::SliderFloat("Yaw", &camYaw, 0.0f, maxSliderValueRotation)) {
                //     cameraMoved = true;
                //     // Reset and reapply rotation
                //     // You might need to adjust this based on your Camera class implementation
                //     cam = Camera(config.gridWidth, Vec3f(camX, camY, camZ), Vec3f(0,1,0), 80);
                //     cam.rotateYaw(camYaw);
                //     cam.rotatePitch(camPitch);
                // }
                // if (ImGui::SliderFloat("Pitch", &camPitch, 0.0f, maxSliderValueRotation)) {
                //     cameraMoved = true;
                //     // Reset and reapply rotation
                //     cam = Camera(config.gridWidth, Vec3f(camX, camY, camZ), Vec3f(0,1,0), 80);
                //     cam.rotateYaw(camYaw);
                //     cam.rotatePitch(camPitch);
                // }
                ImGui::Text("View Direction:");
                if (ImGui::SliderFloat("Camera View X", &camvX, -1.0f, 1.0f)) {
                    cameraMoved = true;
                    cam.posfor.direction.x = camvX;
                }
                if (ImGui::SliderFloat("Camera View Y", &camvY, -1.0f, 1.0f)) {
                    cameraMoved = true;
                    cam.posfor.direction.y = camvY;
                }
                if (ImGui::SliderFloat("Camera View Z", &camvZ, -1.0f, 1.0f)) {
                    cameraMoved = true;
                    cam.posfor.direction.z = camvZ;
                }
                
                ImGui::Separator();
                ImGui::Text("Current Camera Position:");
                ImGui::Text("X: %.2f, Y: %.2f, Z: %.2f", 
                           cam.posfor.origin.x, 
                           cam.posfor.origin.y, 
                           cam.posfor.origin.z);
                // ImGui::Text("Yaw: %.2f°, Pitch: %.2f°", camYaw, camPitch);
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Preview");
            
            if (gridInitialized && textureInitialized) {
                ImGui::Image((void*)(intptr_t)textu, ImVec2(config.outWidth, config.outHeight));
            } else if (gridInitialized) {
                ImGui::Text("Preview not generated yet");
            } else {
                ImGui::Text("No grid generated");
            }
            
            ImGui::End();
        }

        // Auto-rotation controls
        {
            ImGui::Begin("Animation Controls");
            
            ImGui::Text("Auto-Rotation:");
            
            // Toggle button for auto-rotation
            if (ImGui::Button(autoRotate ? "Stop Auto-Rotation" : "Start Auto-Rotation")) {
                autoRotate = !autoRotate;
                if (autoRotate) {
                    autoRotationTime = 0.0f;
                    initialViewDir = Vec3f(camvX, camvY, camvZ);
                }
            }

            if (ImGui::Button(autoRotateView ? "Stop Looking Around" : "Start Looking Around")) {
                autoRotateView = !autoRotateView;
                if (autoRotateView) {
                    autoRotationAngle = 0.0f;
                    initialViewDir = Vec3f(camvX, camvY, camvZ);
                }
            }

            if (autoRotate) {
                ImGui::SameLine();
                ImGui::Text("(Running)");
                
                // Use frame-based timing for animation
                float frameTime = 1.0f / config.fps;
                autoRotationTime += frameTime; // Use constant frame time, not real time
                
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
                camX = config.gridWidth / 2.0f + rotationRadius * camvX;
                camY = config.gridHeight / 2.0f + rotationRadius * camvY;
                camZ = config.gridDepth / 2.0f + rotationRadius * camvZ;
                
                // Update camera
                cam.posfor.origin = Vec3f(camX, camY, camZ);
                cam.posfor.direction = Vec3f(camvX, camvY, camvZ);
                
                cameraMoved = true;
                
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
                
                // Use frame-based timing
                float frameTime = 1.0f / config.fps;
                autoRotationAngle += frameTime;
                
                // Calculate rotation angles using frame-based timing
                float yaw = autoRotationAngle * yawSpeed * (3.14159f / 180.0f);
                float pitch = sinf(autoRotationAngle * 0.7f) * pitchSpeed * (3.14159f / 180.0f);
                
                // Apply rotations
                Vec3f forward = initialViewDir;
                
                // Yaw rotation (around Y axis)
                float cosYaw = cosf(yaw);
                float sinYaw = sinf(yaw);
                Vec3f tempForward;
                tempForward.x = forward.x * cosYaw + forward.z * sinYaw;
                tempForward.y = forward.y;
                tempForward.z = -forward.x * sinYaw + forward.z * cosYaw;
                forward = tempForward;
                
                // Pitch rotation (around X axis)
                float cosPitch = cosf(pitch);
                float sinPitch = sinf(pitch);
                tempForward.x = forward.x;
                tempForward.y = forward.y * cosPitch - forward.z * sinPitch;
                tempForward.z = forward.y * sinPitch + forward.z * cosPitch;
                forward = tempForward;
                
                // Normalize
                float length = sqrtf(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
                if (length > 0.001f) {
                    forward.x /= length;
                    forward.y /= length;
                    forward.z /= length;
                }
                
                // Update view direction
                camvX = forward.x;
                camvY = forward.y;
                camvZ = forward.z;
                
                // Update camera
                cam.posfor.direction = Vec3f(camvX, camvY, camvZ);
                
                cameraMoved = true;
                
                // Show current view direction
                ImGui::Text("Current View: (%.3f, %.3f, %.3f)", camvX, camvY, camvZ);
                
                // Sliders to control rotation speeds
                ImGui::SliderFloat("Yaw Speed", &yawSpeed, 0.1f, 5.0f, "%.2f deg/sec");
                ImGui::SliderFloat("Pitch Speed", &pitchSpeed, 0.0f, 2.0f, "%.2f deg/sec");
            }
            
            // Record button during animations
            if ((autoRotate || autoRotateView) && !isRecordingAVI) {
                ImGui::Separator();
                if (ImGui::Button("Record Animation to AVI")) {
                    startAVIRecording(recordingDurationFrames);
                }
            }
            
            ImGui::End();
        }

        // Handle AVI recording start request
        if (recordingRequested) {
            isRecordingAVI = true;
            recordingRequested = false;
        }
        
        // Check if recording should stop
        if (isRecordingAVI && recordedFrames.size() >= recordingDurationFrames) {
            stopAndSaveAVI(config, aviFilename);
            ImGui::OpenPopup("Recording Complete");
        }

        // Update preview if camera moved or enough time has passed
        if (gridInitialized && !updatePreview) {
            double timeSinceLastUpdate = currentTime - lastUpdateTime;
            
            // Update preview if needed
            if (cameraMoved || timeSinceLastUpdate > 0.1) {
                livePreview(grid, config, cam);
                lastUpdateTime = currentTime;
                cameraMoved = false;
            }
        }
        
        // Reset accumulator for next frame
        accumulator -= targetFrameTime;

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
    
    glfwDestroyWindow(window);
    if (textu != 0) {
        glDeleteTextures(1, &textu);
        textu = 0;
    }
    glfwTerminate();
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}