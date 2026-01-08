#include <iostream>
#include <vector>
#include "../util/grid/grid3.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/frame.hpp"
#include "../util/timing_decorator.cpp"

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

struct Shared {
    std::mutex mutex;
    VoxelGrid grid;
};

void setup(defaults config, VoxelGrid& grid) {
    float threshold = 0.3 * 255;
    grid.resize(config.gridWidth, config.gridHeight, config.gridDepth);
    std::cout << "Generating grid of size " << config.gridWidth << "x" << config.gridHeight << "x" << config.gridDepth << std::endl;
    for (int z = 0; z < config.gridDepth; ++z) {
        if (z % 64 == 0) {
            std::cout << "Processing layer " << z << " of " << config.gridDepth << std::endl;
        }

        for (int y = 0; y < config.gridHeight; ++y) {
            for (int x = 0; x < config.gridWidth; ++x) {
                Vec4ui8 noisecolor = config.noise.permuteColor(Vec3f( static_cast<float>(x) / 64, static_cast<float>(y) / 64, static_cast<float>(z) / 64));
                if (noisecolor.a > threshold) {
                    grid.set(Vec3i(x, y, z), true, Vec3ui8(noisecolor.xyz()));
                }
            }
        }
    }
    std::cout << "Noise grid generation complete!" << std::endl;
}

void livePreview(VoxelGrid& grid, defaults config, Camera cam) {
    updatePreview = true;
    //std::lock_guard<std::mutex> lock(PreviewMutex);
    //frame currentPreviewFrame = 
    grid.renderFrame(cam.posfor.origin, cam.posfor.direction, cam.up, cam.fov, config.outWidth, config.outHeight, frame::colormap::BGRA);

    // glGenTextures(1, &textu);
    // glBindTexture(GL_TEXTURE_2D, textu);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    // glBindTexture(GL_TEXTURE_2D, textu);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, currentPreviewFrame.getWidth(), currentPreviewFrame.getHeight(), 
    //              0, GL_RGBA, GL_UNSIGNED_BYTE, currentPreviewFrame.getData().data());
    
    std::cout << "freeing previous frame" << std::endl;
    //currentPreviewFrame.free();
    updatePreview = false;
    textureInitialized = true;
}

bool savePreview(VoxelGrid& grid, defaults config, Camera cam) {
    TIME_FUNCTION;
    
    std::vector<uint8_t> renderBuffer;
    size_t width = config.outWidth;
    size_t height = config.outHeight;
    
    // Render the view
    frame output = grid.renderFrame(cam.posfor.origin, cam.posfor.direction, cam.up, cam.fov, config.outWidth, config.outHeight, frame::colormap::RGB);
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

    Camera cam(config.gridWidth, Vec3f(0,0,0), Vec3f(0,1,0), 80);

    while (!glfwWindowShouldClose(window)) {
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
                //livePreview(grid,config,cam);
            }

            ImGui::End();
        }

        // {
        //     ImGui::Begin("Preview");
            
        //     if (gridInitialized && textureInitialized) {
        //         ImGui::Image((void*)(intptr_t)textu,ImVec2(config.outWidth, config.outHeight));
        //     } else if (gridInitialized) {
        //         ImGui::Text("Preview not generated yet");
        //     } else {
        //         ImGui::Text("No grid generated");
        //     }
            
        //     ImGui::End();
        // }

        // if (gridInitialized && updatePreview == false) {
        //     livePreview(grid, config, cam);
        // }

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

    
    // std::cout << "shutting down" << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    savePreview(grid, config, cam);

    // std::cout << "destroying" << std::endl;
    glfwDestroyWindow(window);
    if (textu != 0) {
        glDeleteTextures(1, &textu);
        textu = 0;
    }
    glfwTerminate();
    // std::cout << "printing" << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}