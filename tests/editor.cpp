#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "../eigen/Eigen/Dense"
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3render.cpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/grid/grid3edit.cpp"
#include "../util/timing_decorator.cpp"

using Vec3 = Eigen::Vector3f;
using Grid3 = Grid::Octree<int>;
using Grid::VoxelMat;

static const int ANCHOR_OBJ = 1000000;

enum class RenderMode { FAST_CPU, GAMESTYLE, FAST_VK, BLENDED_VK, SUPERBLENDED_VK, OFFLINE_VK };
enum class PrimType { CUBE, SPHERE, PYRAMID, CYLINDER };

struct RenderSettings {
    RenderMode mode = RenderMode::FAST_VK;
    int width = 640;
    int height = 640;
    float pbrScale = 0.5f;
    float ptScale = 0.25f;
    int samplesPerPixel = 2;
    int minSamplesPerPixel = 4;
    int maxBounces = 4;
    bool globalIllumination = false;
    bool useLod = true;
};

struct PrimSettings {
    PrimType type = PrimType::CUBE;
    float voxel = 0.1f;
    float dims[3] = {1.0f, 1.0f, 1.0f};
    float radius = 0.5f;
    float height = 1.0f;
    float baseSize = 1.0f;
    float center[3] = {0.0f, 0.0f, 0.0f};
    float color[3] = {0.7f, 0.7f, 0.7f};
    bool hollow = false;
};

struct MatUI {
    float albedo[3] = {0.7f, 0.7f, 0.7f};
    float emittance = 0.0f;
    float roughness = 1.0f;
    float metallic = 0.0f;
    float transmission = 0.0f;
    float ior = 1.45f;
    float absorption[3] = {0.0f, 0.0f, 0.0f};
    bool useSellmeier = false;
    float sellB[3] = {0.0f, 0.0f, 0.0f};
    float sellC[3] = {0.0f, 0.0f, 0.0f};
    int bodyType = 0;
    float mass = 1.0f;
};

class EditorUI {
private:
    Grid3 grid{Vec3(-64,-64,-64),Vec3(64,64,64), "/tmp/editorstore", 8};
    Camera cam;
    RenderSettings rs;
    PrimSettings ps;
    MatUI mat;

    GLuint textu = 0;
    frame previewFrame;
    ImVec2 lastImageMin{0, 0};
    ImVec2 lastImageSize{0, 0};

    int editObjectId = -1;
    float editRadius = 0.5f;
    int bevelLayers = 1;
    int extrudeLayers = 1;
    int roundPasses = 1;
    float addColor[3] = {0.8f, 0.2f, 0.2f};
    char savePath[256] = "object.bin";

    float moveSpeed = 4.0f;
    float orbitSpeed = 1.5f;

    bool orbiting = false;
    Vec3 orbitTarget = Vec3::Zero();
    float orbitAzimuth = 0.0f;
    float orbitElevation = 0.0f;
    float orbitRadius = 6.0f;
    bool orbitInit = false;

public:
    EditorUI() {
        grid = Grid3(Vec3(-64, -64, -64), Vec3(64, 64, 64), "/tmp/editorstore", 8);
        grid.insert(0, Vec3(0, 0, 0), true, Vec3(0.9f, 0.9f, 0.2f), 0.1f, true, ANCHOR_OBJ);
        cam.origin = Vec3(0, 2, -6);
        cam.up = Vec3(0, 1, 0);
        cam.fov = 60.0f;
        cam.direction = (Vec3(0, 0, 0) - cam.origin).normalized();
        cam.up = Vec3(0, 1, 0);
    }

    ~EditorUI() {
        if (textu != 0) glDeleteTextures(1, &textu);
    }

    void renderUI(GLFWwindow* window) {
        handleKeyboardCamera();
        drawToolPanel();
        drawViewport(window);
    }

private:
    using RayHit = Grid3::RayHit;

    VoxelMat buildMat() const {
        VoxelMat m;
        m.albedo = Vec3(mat.albedo[0], mat.albedo[1], mat.albedo[2]);
        m.emittance = mat.emittance;
        m.roughness = mat.roughness;
        m.metallic = mat.metallic;
        m.transmission = mat.transmission;
        m.ior = mat.ior;
        m.absorption = Vec3(mat.absorption[0], mat.absorption[1], mat.absorption[2]);
        m.useSellmeier = mat.useSellmeier;
        if (mat.useSellmeier) {
            m.sellB = Grid::v3half(Eigen::half(mat.sellB[0]), Eigen::half(mat.sellB[1]), Eigen::half(mat.sellB[2]));
            m.sellC = Grid::v3half(Eigen::half(mat.sellC[0]), Eigen::half(mat.sellC[1]), Eigen::half(mat.sellC[2]));
        }
        m.bType = (Grid::BodyType)mat.bodyType;
        m.mass = mat.mass;
        return m;
    }

    void handleKeyboardCamera() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput) return;
        float dt = io.DeltaTime > 0.0f ? io.DeltaTime : 0.016f;
        float d = moveSpeed * dt;
        Vec3 fwd = cam.direction.normalized();
        Vec3 right = fwd.cross(cam.up).normalized();
        Vec3 up = cam.up.normalized();

        if (ImGui::IsKeyDown(ImGuiKey_W)) cam.origin += fwd * d;
        if (ImGui::IsKeyDown(ImGuiKey_S)) cam.origin -= fwd * d;
        if (ImGui::IsKeyDown(ImGuiKey_A)) cam.origin -= right * d;
        if (ImGui::IsKeyDown(ImGuiKey_D)) cam.origin += right * d;
        if (ImGui::IsKeyDown(ImGuiKey_Z)) cam.origin += up * d;
        if (ImGui::IsKeyDown(ImGuiKey_X)) cam.origin -= up * d;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) rotateAroundUp(orbitSpeed * dt);
        if (ImGui::IsKeyDown(ImGuiKey_E)) rotateAroundUp(-orbitSpeed * dt);
    }

    void rotateAroundUp(float angle) {
        Eigen::Matrix3f r;
        r = Eigen::AngleAxisf(angle, cam.up.normalized());
        cam.direction = (r * cam.direction).normalized();
    }

    void panCamera(float dx, float dy) {
        Vec3 fwd = cam.direction.normalized();
        Vec3 right = fwd.cross(cam.up).normalized();
        Vec3 up = right.cross(fwd).normalized();
        cam.origin += right * (-dx) + up * dy;
    }

    void beginOrbit() {
        Vec3 offset = cam.origin - orbitTarget;
        orbitRadius = offset.norm();
        if (orbitRadius < 1e-4f) orbitRadius = 1e-4f;
        Vec3 n = offset / orbitRadius;
        orbitElevation = std::asin(std::clamp(n.y(), -1.0f, 1.0f));
        orbitAzimuth = std::atan2(n.x(), n.z());
        orbitInit = true;
    }

    void orbitCamera(float dx, float dy) {
        if (!orbitInit) beginOrbit();
        orbitAzimuth += dx;
        orbitElevation += dy;

        float ce = std::cos(orbitElevation);
        float se = std::sin(orbitElevation);
        float ca = std::cos(orbitAzimuth);
        float sa = std::sin(orbitAzimuth);

        Vec3 offset(ce * sa, se, ce * ca);
        cam.origin = orbitTarget + offset * orbitRadius;
        cam.direction = (-offset).normalized();
        Vec3 right(ca, 0.0f, -sa);
        cam.up = right.cross(cam.direction).normalized();
    }

    static bool numF(const char* label, float* v, float step = 0.01f) {
        return ImGui::InputFloat(label, v, step, step * 10.0f, "%.4f");
    }
    static bool numI(const char* label, int* v, int step = 1) {
        return ImGui::InputInt(label, v, step, step * 10);
    }

    void drawMaterialControls() {
        ImGui::ColorEdit3("Albedo", mat.albedo);
        numF("Emittance (chroma mult)", &mat.emittance);
        numF("Roughness", &mat.roughness);
        float gloss = 1.0f - mat.roughness;
        if (numF("Glossiness", &gloss)) mat.roughness = 1.0f - gloss;
        numF("Metallic", &mat.metallic);
        numF("Transmission", &mat.transmission);
        numF("IOR", &mat.ior);
        ImGui::ColorEdit3("Absorption", mat.absorption);
        ImGui::Checkbox("Use raw Sellmeier B/C", &mat.useSellmeier);
        if (mat.useSellmeier) {
            ImGui::InputFloat3("Sellmeier B", mat.sellB, "%.4f");
            ImGui::InputFloat3("Sellmeier C", mat.sellC, "%.4f");
        }
        const char* bodies[] = {"Static", "Fluid", "Rigid"};
        ImGui::Combo("Body type", &mat.bodyType, bodies, 3);
        numF("Mass", &mat.mass);
    }

    void drawPrimTab() {
        const char* names[] = {"Cube", "Sphere", "Pyramid", "Cylinder"};
        int t = (int)ps.type;
        if (ImGui::Combo("Primitive", &t, names, 4)) ps.type = (PrimType)t;
        numF("Voxel size", &ps.voxel);
        ImGui::InputFloat3("Center", ps.center, "%.3f");
        ImGui::Checkbox("Hollow", &ps.hollow);
        switch (ps.type) {
            case PrimType::CUBE:
                ImGui::InputFloat3("Dimensions", ps.dims, "%.3f");
                break;
            case PrimType::SPHERE:
                numF("Radius", &ps.radius);
                break;
            case PrimType::PYRAMID:
                numF("Base size", &ps.baseSize);
                numF("Height", &ps.height);
                break;
            case PrimType::CYLINDER:
                numF("Radius", &ps.radius);
                numF("Height", &ps.height);
                break;
        }
        ImGui::SeparatorText("Material");
        drawMaterialControls();
        ImGui::Separator();
        if (ImGui::Button("Insert Primitive")) insertPrimitive();
        if (editObjectId >= 0) ImGui::Text("Active object id: %d", editObjectId);
    }

    void insertPrimitive() {
        Vec3 c(ps.center[0], ps.center[1], ps.center[2]);
        VoxelMat m = buildMat();
        int id = -1;
        switch (ps.type) {
            case PrimType::CUBE:
                id = grid.insertCube(c, Vec3(ps.dims[0], ps.dims[1], ps.dims[2]), ps.voxel, m, -1, ps.hollow);
                break;
            case PrimType::SPHERE:
                id = grid.insertSphere(c, ps.radius, ps.voxel, m, -1, ps.hollow);
                break;
            case PrimType::PYRAMID:
                id = grid.insertPyramid(c, ps.baseSize, ps.height, ps.voxel, m, -1, ps.hollow);
                break;
            case PrimType::CYLINDER:
                id = grid.insertCylinder(c, ps.radius, ps.height, ps.voxel, m, -1, ps.hollow);
                break;
        }
        if (id >= 0) editObjectId = id;
    }

    void drawToolsTab() {
        ImGui::Text("Active object id");
        numI("##objid", &editObjectId);
        ImGui::Separator();

        ImGui::ColorEdit3("Add color", addColor);
        ImGui::TextWrapped("Left-click viewport: add voxel. Right-click: remove.");
        ImGui::Separator();
        numI("Bevel layers", &bevelLayers);
        if (ImGui::Button("Bevel") && editObjectId >= 0) grid.bevelObject(editObjectId, bevelLayers);
        ImGui::SameLine();
        if (ImGui::Button("Chamfer") && editObjectId >= 0) grid.chamferObject(editObjectId);
        numI("Round passes", &roundPasses);
        if (ImGui::Button("Round") && editObjectId >= 0) grid.roundObject(editObjectId, roundPasses);
        ImGui::SameLine();
        if (ImGui::Button("Fillet") && editObjectId >= 0) grid.filletObject(editObjectId, bevelLayers);

        if (ImGui::Button("Subdivide") && editObjectId >= 0) grid.subdivideObject(editObjectId);
        ImGui::SameLine();
        if (ImGui::Button("Smooth") && editObjectId >= 0) grid.smoothObject(editObjectId);

        ImGui::Separator();
        numI("Extrude layers", &extrudeLayers);
        ImGui::TextWrapped("Middle-click viewport: extrude face.");
        numF("Flatten radius", &editRadius);
        ImGui::TextWrapped("Shift+Left-click viewport: flatten surface.");
    }

    void drawRenderTab() {
        ImGui::Text("Render Mode");
        int m = (int)rs.mode;
        ImGui::RadioButton("Fast (CPU)", &m, (int)RenderMode::FAST_CPU);
        ImGui::RadioButton("Game Style", &m, (int)RenderMode::GAMESTYLE);
        ImGui::RadioButton("Fast (Vulkan)", &m, (int)RenderMode::FAST_VK);
        ImGui::RadioButton("Blended (Vulkan)", &m, (int)RenderMode::BLENDED_VK);
        ImGui::RadioButton("Super Blended (Vulkan)", &m, (int)RenderMode::SUPERBLENDED_VK);
        ImGui::RadioButton("Offline PT (Vulkan)", &m, (int)RenderMode::OFFLINE_VK);
        rs.mode = (RenderMode)m;

        ImGui::Separator();
        numI("Width", &rs.width, 8);
        numI("Height", &rs.height, 8);
        switch (rs.mode) {
            case RenderMode::FAST_CPU:
            case RenderMode::GAMESTYLE:
            case RenderMode::FAST_VK:
                ImGui::TextDisabled("No extra settings.");
                break;
            case RenderMode::BLENDED_VK:
                numF("PBR scale", &rs.pbrScale);
                numI("Samples/pixel", &rs.samplesPerPixel);
                numI("Max bounces", &rs.maxBounces);
                ImGui::Checkbox("Global illumination", &rs.globalIllumination);
                ImGui::Checkbox("Use LOD", &rs.useLod);
                break;
            case RenderMode::SUPERBLENDED_VK:
                numF("PT scale", &rs.ptScale);
                numI("Samples/pixel", &rs.samplesPerPixel);
                numI("Min samples/pixel", &rs.minSamplesPerPixel);
                numI("Max bounces", &rs.maxBounces);
                ImGui::Checkbox("Global illumination", &rs.globalIllumination);
                ImGui::Checkbox("Use LOD", &rs.useLod);
                break;
            case RenderMode::OFFLINE_VK:
                numI("Samples/pixel", &rs.samplesPerPixel);
                numI("Max bounces", &rs.maxBounces);
                ImGui::Checkbox("Global illumination", &rs.globalIllumination);
                ImGui::Checkbox("Use LOD", &rs.useLod);
                break;
        }
    }

    void drawCameraTab() {
        float pos[3] = {cam.origin.x(), cam.origin.y(), cam.origin.z()};
        if (ImGui::InputFloat3("Position", pos, "%.3f"))
            cam.origin = Vec3(pos[0], pos[1], pos[2]);
        float dir[3] = {cam.direction.x(), cam.direction.y(), cam.direction.z()};
        if (ImGui::InputFloat3("Direction", dir, "%.3f"))
            cam.direction = Vec3(dir[0], dir[1], dir[2]).normalized();
        numF("FOV", &cam.fov, 1.0f);
        numF("Move speed", &moveSpeed, 0.5f);
        numF("Orbit speed", &orbitSpeed, 0.1f);
        if (ImGui::Button("Look at origin"))
            cam.direction = (Vec3(0, 0, 0) - cam.origin).normalized();
        ImGui::SeparatorText("Controls");
        ImGui::TextWrapped("WASD move, Z/X up/down, Q/E orbit-turn. MB3 drag: pan. Shift+MB3: orbit screen center. Scroll: forward/back.");
    }

    void drawFileTab() {
        ImGui::InputText("Path", savePath, sizeof(savePath));
        if (ImGui::Button("Save object") && editObjectId >= 0) grid.saveObject(editObjectId, savePath);
        ImGui::SameLine();
        if (ImGui::Button("Export (drop anchor)")) exportObject();
        if (ImGui::Button("Load object")) {
            int id = grid.loadObject(savePath, Vec3(0, 0, 0), -1);
            if (id >= 0) editObjectId = id;
        }
        ImGui::Separator();
        if (ImGui::Button("Save full scene")) grid.save(savePath);
        ImGui::SameLine();
        if (ImGui::Button("Load full scene")) grid.load(savePath);
    }

    void exportObject() {
        if (editObjectId < 0) return;
        grid.remove(Vec3(0, 0, 0), 0.06f);
        grid.saveObject(editObjectId, savePath);
        grid.insert(0, Vec3(0, 0, 0), true, Vec3(0.9f, 0.9f, 0.2f), 0.1f, true, ANCHOR_OBJ);
    }

    void drawToolPanel() {
        ImGui::Begin("Editor");
        if (ImGui::BeginTabBar("tabs")) {
            if (ImGui::BeginTabItem("Primitives")) { drawPrimTab(); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("Tools")) { drawToolsTab(); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("Render")) { drawRenderTab(); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("Camera")) { drawCameraTab(); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("File")) { drawFileTab(); ImGui::EndTabItem(); }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

    frame renderCurrent() {
        switch (rs.mode) {
            case RenderMode::GAMESTYLE:
                return grid.GameStyleRenderFrame(cam, rs.height, rs.width);
            case RenderMode::FAST_VK:
                return grid.fastRenderFrameVulkan(cam, rs.height, rs.width);
            case RenderMode::BLENDED_VK:
                return grid.blendedRenderFrameVulkan(cam, rs.height, rs.width, rs.pbrScale,
                    frame::colormap::RGB, rs.samplesPerPixel, rs.maxBounces, rs.globalIllumination, rs.useLod);
            case RenderMode::SUPERBLENDED_VK:
                return grid.superBlendedRenderFrameVulkan(cam, rs.height, rs.width, rs.ptScale,
                    frame::colormap::RGB, rs.samplesPerPixel, rs.maxBounces, rs.globalIllumination, rs.useLod, rs.minSamplesPerPixel);
            case RenderMode::OFFLINE_VK:
                return grid.renderFrameVulkan(cam, rs.height, rs.width, frame::colormap::RGB,
                    rs.samplesPerPixel, rs.maxBounces, rs.globalIllumination, rs.useLod);
            case RenderMode::FAST_CPU:
            default:
                return grid.fastRenderFrame(cam, rs.height, rs.width);
        }
    }

    void drawViewport(GLFWwindow* window) {
        ImGui::Begin("Viewport");
        previewFrame = renderCurrent();

        if (textu == 0) glGenTextures(1, &textu);
        glBindTexture(GL_TEXTURE_2D, textu);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, previewFrame.getWidth(), previewFrame.getHeight(),
                     0, GL_RGB, GL_UNSIGNED_BYTE, previewFrame.getData().data());

        float aspect = (float)previewFrame.getWidth() / (float)previewFrame.getHeight();
        float availW = ImGui::GetContentRegionAvail().x;
        ImVec2 imgSize(availW, availW / aspect);
        lastImageMin = ImGui::GetCursorScreenPos();
        lastImageSize = imgSize;
        ImGui::Image((void*)(intptr_t)textu, imgSize);

        handleViewportInput();
        ImGui::End();
    }

    bool pixelUnderMouse(int& px, int& py) {
        ImVec2 mouse = ImGui::GetIO().MousePos;
        float u = (mouse.x - lastImageMin.x) / lastImageSize.x;
        float v = (mouse.y - lastImageMin.y) / lastImageSize.y;
        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return false;
        px = (int)(u * rs.width);
        py = (int)(v * rs.height);
        return true;
    }

    void handleViewportInput() {
        bool hovered = ImGui::IsItemHovered();
        ImGuiIO& io = ImGui::GetIO();

        if (hovered && io.MouseWheel != 0.0f) {
            cam.origin += cam.direction.normalized() * io.MouseWheel * (moveSpeed * 0.15f);
            orbitInit = false;
        }

        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            if (io.KeyShift) {
                RayHit hit;
                if (grid.raycastFromCamera(cam, rs.width / 2, rs.height / 2, rs.width, rs.height, hit))
                    orbitTarget = hit.hitPoint;
                else
                    orbitTarget = cam.origin + cam.direction.normalized() * 6.0f;
                orbiting = true;
                beginOrbit();
            }
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) && (hovered || orbiting)) {
            ImVec2 dv = io.MouseDelta;
            if (io.KeyShift && orbiting) {
                orbitCamera(dv.x * 0.01f, dv.y * 0.01f);
            } else {
                panCamera(dv.x * (moveSpeed * 0.004f), dv.y * (moveSpeed * 0.004f));
            }
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) orbiting = false;

        if (!hovered) return;

        int px, py;
        if (!pixelUnderMouse(px, py)) return;
        bool shift = io.KeyShift;
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (shift) {
                grid.flattenSurface(cam, px, py, rs.width, rs.height, editRadius);
            } else {
                Vec3 col(addColor[0], addColor[1], addColor[2]);
                grid.addVoxelAtPixel(cam, px, py, rs.width, rs.height, col, editObjectId < 0 ? -2 : editObjectId);
            }
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            grid.removeVoxelAtPixel(cam, px, py, rs.width, rs.height);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle) && io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Middle] < 9.0f && !shift) {
            grid.extrudeFace(cam, px, py, rs.width, rs.height, extrudeLayers);
        }
    }
};

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return -1;
    const char* glsl_version = "#version 450";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1600, 900, "StupidSim Editor", nullptr, nullptr);
    if (window == nullptr) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    EditorUI editor;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        editor.renderUI(window);

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