#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_set>
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
#define M_PI = 3.1415
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
};

struct AnimationConfig {
    int width = 1024;
    int height = 1024;
    int totalFrames = 480;
    float fps = 30.0f;
    int numSeeds = 8;
};

Grid2 setup(AnimationConfig config) {
    TIME_FUNCTION;
    Grid2 grid;
    std::vector<Vec2> pos;
    std::vector<Vec4> colors;
    std::vector<float> sizes;
    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            float gradient = (x + y) / float(config.width + config.height - 2);
            pos.push_back(Vec2(x,y));
            colors.push_back(Vec4(gradient, gradient, gradient, 1.0f));
            sizes.push_back(1.0f);
        }
    }
    grid.bulkAddObjects(pos,colors,sizes);
    return grid;
}

void Preview(Grid2& grid) {
    TIME_FUNCTION;
    int width;
    int height;
    //std::vector<uint8_t> rgbData;

    frame rgbData = grid.getGridAsFrame(frame::colormap::RGB);
    std::cout << "Frame looks like: " << rgbData << std::endl;
    bool success = BMPWriter::saveBMP("output/grayscalesource.bmp", rgbData);
    if (!success) {
        std::cout << "yo! this failed in Preview" << std::endl;
    }
}

void livePreview(const Grid2& grid) {
    std::lock_guard<std::mutex> lock(previewMutex);
    
    currentPreviewFrame = grid.getGridAsFrame(frame::colormap::RGBA);

    glGenTextures(1, &textu);
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    glBindTexture(GL_TEXTURE_2D, textu);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
    
    updatePreview = true;
}

std::vector<std::tuple<size_t, Vec2, Vec4>> pickSeeds(Grid2 grid, AnimationConfig config) {
    TIME_FUNCTION;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDist(0, config.width - 1);
    std::uniform_int_distribution<> yDist(0, config.height - 1);
    std::uniform_real_distribution<> colorDist(0.2f, 0.8f);

    std::vector<std::tuple<size_t, Vec2, Vec4>> seeds;

    for (int i = 0; i < config.numSeeds; ++i) {
        Vec2 point(xDist(gen), yDist(gen));
        Vec4 color(colorDist(gen), colorDist(gen), colorDist(gen), 255);
        size_t id = grid.getOrCreatePositionVec(point, 0.0, true);
        grid.setColor(id, color);
        seeds.push_back(std::make_tuple(id,point, color));
    }
    return seeds;
}

void expandPixel(Grid2& grid, AnimationConfig config, std::vector<std::tuple<size_t, Vec2, Vec4>>& seeds) {
    TIME_FUNCTION;
    std::vector<std::tuple<size_t, Vec2, Vec4>> newseeds; 

    std::unordered_set<size_t> visitedThisFrame; 
    for (const auto& seed : seeds) {
        visitedThisFrame.insert(std::get<0>(seed));
    }

    //#pragma omp parallel for
    for (const std::tuple<size_t, Vec2, Vec4>& seed : seeds) {
        size_t id = std::get<0>(seed);
        Vec2 seedPOS = std::get<1>(seed);
        Vec4 seedColor = std::get<2>(seed);
        std::vector<size_t> neighbors = grid.getNeighbors(id);
        //grid.setSize(id, grid.getSize(id)+4);
        for (size_t neighbor : neighbors) {
            if (visitedThisFrame.count(neighbor)) {
                continue;
            }
            visitedThisFrame.insert(neighbor);

            Vec2 neipos = grid.getPositionID(neighbor);
            Vec4 neighborColor = grid.getColor(neighbor);
            float distance = seedPOS.distance(neipos);
            float angle = seedPOS.directionTo(neipos);

            float normalizedAngle = (angle + M_PI) / (2.0f * M_PI);
            float blendFactor = 0.3f + 0.4f * std::sin(normalizedAngle * 2.0f * M_PI);
            blendFactor = std::clamp(blendFactor, 0.1f, 0.9f);
            
            Vec4 newcolor = Vec4(
                seedColor.r * blendFactor + neighborColor.r * (1.0f - blendFactor),
                seedColor.g * (1.0f - blendFactor) + neighborColor.g * blendFactor,
                seedColor.b * (0.5f + 0.5f * std::sin(normalizedAngle * 4.0f * M_PI)),
                1.0f
            );
            
            newcolor = newcolor.clamp(0.0f, 1.0f);

            grid.setColor(neighbor, newcolor);
            newseeds.emplace_back(neighbor, neipos, newcolor);
        }
    }
    seeds.clear();
    seeds.shrink_to_fit();
    seeds = std::move(newseeds);
}

//bool exportavi(std::vector<std::vector<uint8_t>> frames, AnimationConfig config) {
bool exportavi(std::vector<frame> frames, AnimationConfig config) {
    TIME_FUNCTION;
    std::string filename = "output/chromatic_transformation.avi";
    
    std::cout << "Frame count: " << frames.size() << std::endl;
    
    // Log compression statistics for all frames
    std::cout << "\n=== Frame Compression Statistics ===" << std::endl;
    size_t totalOriginalSize = 0;
    size_t totalCompressedSize = 0;
    
    for (int i = 0; i < frames.size(); ++i) {    
        totalOriginalSize += frames[i].getSourceSize();
        totalCompressedSize += frames[i].getTotalCompressedSize();
    }
    
    double overallRatio = static_cast<double>(totalOriginalSize) / totalCompressedSize;
    double overallSavings = (1.0 - 1.0/overallRatio) * 100.0;
    
    std::cout << "\n=== Overall Compression Summary ===" << std::endl;
    std::cout << "Total frames: " << frames.size() << std::endl;
    std::cout << "Compressed frames: " << frames.size() << std::endl;
    std::cout << "Total original size: " << totalOriginalSize << " bytes (" 
                << std::fixed << std::setprecision(2) << (totalOriginalSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Total compressed size: " << totalCompressedSize << " bytes (" 
                << std::fixed << std::setprecision(2) << (totalCompressedSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Overall compression ratio: " << std::fixed << std::setprecision(2) << overallRatio << ":1" << std::endl;
    std::cout << "Overall space savings: " << std::fixed << std::setprecision(1) << overallSavings << "%" << std::endl;
    
    std::filesystem::path dir = "output";
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cout << "Failed to create output directory!" << std::endl;
            return false;
        }
    }
    
    bool success = AVIWriter::saveAVIFromCompressedFrames(filename, frames, frames[0].getWidth(), frames[0].getHeight(), config.fps);
    
    if (success) {
        // Check if file actually exists
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

void mainLogic(const AnimationConfig& config, Shared& state, int gradnoise) {
    isGenerating = true;
    try {
        Grid2 grid;
        if (gradnoise == 0) {
            grid = setup(config);
        } else if (gradnoise == 1) {
            grid = grid.noiseGenGrid(0,0,config.height, config.width);
        }
        grid.setDefault(Vec4(0,0,0,0));
        {
            std:: lock_guard<std::mutex> lock(state.mutex);
            state.grid = grid;
            state.hasNewFrame = true;
            state.currentFrame = 0;
        }
        Preview(grid);
        std::vector<std::tuple<size_t, Vec2, Vec4>> seeds = pickSeeds(grid, config);
        std::vector<frame> frames;

        for (int i = 0; i < config.totalFrames; ++i){
            // Check if we should stop the generation
            if (!isGenerating) {
                std::cout << "Generation cancelled at frame " << i << std::endl;
                return;
            }
            
            expandPixel(grid,config,seeds);
            
            std::lock_guard<std::mutex> lock(state.mutex);
            state.grid = grid;
            state.hasNewFrame = true;
            state.currentFrame = i;

            // Print compression info for this frame
            if (i % 10 == 0 ) {
                frame bgrframe;
                //std::cout << "Processing frame " << i + 1 << "/" << config.totalFrames << std::endl;
                bgrframe = grid.getGridAsFrame(frame::colormap::BGR);
                frames.push_back(bgrframe);
                //bgrframe.decompress();
                //BMPWriter::saveBMP(std::format("output/grayscalesource.{}.bmp", i), bgrframe);
                bgrframe.compressFrameLZ78();
                //bgrframe.printCompressionStats();
            }
        }
        exportavi(frames,config);
    }
    catch (const std::exception& e) {
        std::cerr << "errored at: " << e.what() << std::endl;
    }
    isGenerating = false;
}

// Function to cancel ongoing generation
void cancelGeneration() {
    if (isGenerating) {
        isGenerating = false;
        // Wait for the thread to finish (with timeout to avoid hanging)
        if (generationFuture.valid()) {
            auto status = generationFuture.wait_for(std::chrono::milliseconds(100));
            if (status != std::future_status::ready) {
                std::cout << "Waiting for generation thread to finish..." << std::endl;
            }
        }
    }
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    //static bool window = true;
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
    //ImGui::SetNextWindowSize(ImVec2(1110,667));
    //auto beg = ImGui::Begin("Gradient thing", &window);
    //if (beg) {
    // std::cout << "stuff breaks at 223" << std::endl;
    bool application_not_closed = true;
    //float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
    GLFWwindow* window = glfwCreateWindow((int)(1280), (int)(800), "Chromatic gradient generator thing", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    //IMGUI_CHECKVERSION(); //this might be more important than I realize. but cant run with it so currently ignoring. 
    ImGui::CreateContext();
    // std::cout << "context created" << std::endl;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    //style.ScaleAllSizes(1);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    //style.FontScaleDpi = 1; //will need to implement my own scaling at some point. currently just ignoring it.
    ImGui_ImplGlfw_InitForOpenGL(window, true);


    #ifdef __EMSCRIPTEN__
        ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
    #endif
        ImGui_ImplOpenGL3_Init(glsl_version);


    // std::cout << "created glfw window" << std::endl;


    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 30.0f;
    static int i1 = 1024;
    static int i2 = 1024;
    static int i3 = 480;
    static int i4 = 8;
    static float fs = 1.0;

    std::future<void> mainlogicthread;
    Shared state;
    Grid2 grid;
    AnimationConfig config;
    previewText = "Please generate";
    int gradnoise = true;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {

            ImGui::Begin("settings");

            ImGui::SliderFloat("fps", &f, 20.0f, 60.0f);
            ImGui::SliderInt("width", &i1, 256, 4096);
            ImGui::SliderInt("height", &i2, 256, 4096);
            ImGui::SliderInt("framecount", &i3, 10, 5000);
            ImGui::SliderInt("numSeeds", &i4, 0, 10);
            ImGui::SliderFloat("ScalePreview", &fs, 0.0, 2.0);
            ImGui::RadioButton("Gradient", &gradnoise, 0);
            ImGui::RadioButton("Perlin Noise", &gradnoise, 1);

            if (isGenerating) {
                ImGui::BeginDisabled();
            }
            
            if (ImGui::Button("Generate Animation")) {
                config = AnimationConfig(i1, i2, i3, f, i4);
                mainlogicthread = std::async(std::launch::async, mainLogic, config, std::ref(state), gradnoise);
            }

            if (isGenerating && textu != 0) {
                ImGui::EndDisabled();
                
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) {
                    cancelGeneration();
                }
                // Check for new frames from the generation thread
                bool hasNewFrame = false;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (state.hasNewFrame) {
                        livePreview(state.grid);
                        state.hasNewFrame = false;
                        previewText = "Generating... Frame: " + std::to_string(state.currentFrame);
                    }
                }
                
                ImGui::Text(previewText.c_str());
                
                if (textu != 0) {
                    ImVec2 imageSize = ImVec2(config.width * fs, config.height * fs);
                    ImVec2 uv_min = ImVec2(0.0f, 0.0f);
                    ImVec2 uv_max = ImVec2(1.0f, 1.0f);
                    ImGui::Image((void*)(intptr_t)textu, imageSize, uv_min, uv_max);
                } else {
                    ImGui::Text("Generating preview...");
                }
                
            } else if (isGenerating) {
                ImGui::EndDisabled();
                
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) {
                    cancelGeneration();
                }
                // Check for new frames from the generation thread
                bool hasNewFrame = false;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (state.hasNewFrame) {
                        livePreview(state.grid);
                        state.hasNewFrame = false;
                        previewText = "Generating... Frame: " + std::to_string(state.currentFrame);
                    }
                }
                
                ImGui::Text(previewText.c_str());

            } else if (textu != 0){
                //ImGui::EndDisabled();
                                
                ImGui::Text(previewText.c_str());
                
                if (textu != 0) {
                    ImVec2 imageSize = ImVec2(config.width * 0.5f, config.height * 0.5f);
                    ImVec2 uv_min = ImVec2(0.0f, 0.0f);
                    ImVec2 uv_max = ImVec2(1.0f, 1.0f);
                    ImGui::Image((void*)(intptr_t)textu, imageSize, uv_min, uv_max);
                } else {
                    ImGui::Text("Generating preview...");
                }
                
            } else {
                ImGui::Text("No preview available");
                ImGui::Text("Start generation to see live preview");
            }
            //std::cout << "sleeping" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //std::cout << "ending" << std::endl;
            ImGui::End();
        }

        
        // std::cout << "ending frame" << std::endl;
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        
        // std::cout << "rendering" << std::endl;
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        //mainlogicthread.join();
        
        // std::cout << "swapping buffers" << std::endl;
    }
    cancelGeneration();

    
    // std::cout << "shutting down" << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // std::cout << "destroying" << std::endl;
    glfwDestroyWindow(window);
    if (textu != 0) {
        glDeleteTextures(1, &textu);
        textu = 0;
    }
    glfwTerminate();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    // std::cout << "printing" << std::endl;
    return 0;
}
//I need this: https://raais.github.io/ImStudio/
// g++ -std=c++23 -O3 -march=native -o ./bin/g2gradc ./tests/g2chromatic2.cpp -I./imgui -L./imgui -limgui -lstb `pkg-config --cflags --libs glfw3` && ./bin/g2gradc