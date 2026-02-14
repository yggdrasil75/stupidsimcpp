#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <array>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include "../../eigen/Eigen/Dense"
#include "../../eigen/Eigen/Geometry"
#include "./camera.hpp"
#include "../output/frame.hpp"
#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

using Vector2f = Eigen::Vector2f;
using Vector3f = Eigen::Vector3f;
using Vector4f = Eigen::Vector4f;
using Matrix4f = Eigen::Matrix4f;
using Color = Eigen::Vector3f;

struct Triangle2D {
    Vector2f a, b, c;
    Color color;
    float depth;
};

class Mesh {
private:
    int id;
    std::vector<Vector3f> _vertices;
    std::vector<std::vector<int>> _polys;
    std::vector<Color> _colors;
    mutable std::vector<Eigen::Vector3i> _tris;
    mutable bool _needs_triangulation = true;
    
    inline static float edgeFunction(const Vector2f& a, const Vector2f& b, const Vector2f& c) {
        return (c.x() - a.x()) * (b.y() - a.y()) - (c.y() - a.y()) * (b.x() - a.x());
    }
public:
    Mesh(int id, const std::vector<Vector3f>& verts, const std::vector<std::vector<int>>& polys, const std::vector<Color>& colors)
        : id(id), _vertices(verts), _polys(polys), _colors(colors) {}

    std::vector<Vector3f> vertices() {
        return _vertices;
    }

    bool vertices(std::vector<Vector3f> verts) {
        if (verts.size() != _colors.size()) {
            if (_colors.size() == 1) {
                _vertices = verts;
                return true;
            }
            return false;
        } else {
            _vertices = verts;
            return true;
        }
    }

    std::vector<std::vector<int>> polys() {
        return _polys;
    }

    void triangulate() {
        if (!_needs_triangulation) return;
        std::vector<Eigen::Vector3i> newPols;
        for (auto& pol : _polys) {
            if (pol.size() > 3) {
                auto v0 = pol[0];
                for (size_t i = 1; i < pol.size() - 1; i++) {
                    newPols.emplace_back(Eigen::Vector3i{v0, pol[i], pol[i+1]});
                }
            } else if (pol.size() == 3) {
                Eigen::Vector3i tri{pol[0], pol[1], pol[2]};
                newPols.emplace_back(tri);
            }
        }
        _tris = newPols;
        _needs_triangulation = false; 
    }

    std::vector<Color> colors() {
        return _colors;
    }

    bool colors(std::vector<Color> colorlist) {
        if (colorlist.size() == 1 || colorlist.size() == _vertices.size()) {
            _colors = colorlist;
            return true;
        }
        return false;
    }

    std::vector<Triangle2D> project_2d(Camera cam, int height, int width, float near, float far) {
        triangulate();

        std::vector<Triangle2D> renderList;

        Vector3f forward = cam.forward();
        Vector3f right = cam.right();
        Vector3f up = cam.up;

        Matrix4f viewMatrixa = Matrix4f::Identity();
        viewMatrixa.block<3,1>(0,0) = right;
        viewMatrixa.block<3,1>(0,1) = up;
        viewMatrixa.block<3,1>(0,2) = -forward; 
        viewMatrixa.block<3,1>(0,3) = cam.origin;
        
        Matrix4f viewMatrix = viewMatrixa.inverse();

        float aspect = float(width) / float(height);
        float fovrad = cam.fovRad();
        float tanHalfFov = std::tan(fovrad / 2.0f);
        
        Matrix4f projMatrix = Matrix4f::Zero();
        projMatrix(0,0) = 1.0f / (aspect * tanHalfFov);
        projMatrix(1,1) = 1.0f / tanHalfFov;
        projMatrix(2,2) = -(far + near) / (far - near);
        projMatrix(2,3) = -(2.0f * far * near) / (far - near);
        projMatrix(3,2) = -1.0f;

        int n = _vertices.size();
        
        std::vector<Vector2f> screenCoords(n);
        std::vector<float> linearDepths(n);
        std::vector<bool> validVerts(n, false);

        Eigen::MatrixXf homogeneousVerts(n, 4);
        for (int i = 0; i < n; ++i) {
            homogeneousVerts.row(i).head<3>() = _vertices[i];
            homogeneousVerts(i, 3) = 1.0f;
        }

        Eigen::MatrixXf viewVerts = homogeneousVerts * viewMatrix.transpose();
        Eigen::MatrixXf clipVerts = viewVerts * projMatrix.transpose();

        for (int i = 0; i < n; ++i) {
            float w = clipVerts(i, 3);
            
            if (w <= near) {
                validVerts[i] = false;
                continue; 
            }

            float x_ndc = clipVerts(i, 0) / w;
            float y_ndc = clipVerts(i, 1) / w;

            screenCoords[i].x() = (x_ndc + 1.0f) * 0.5f * width;
            screenCoords[i].y() = (1.0f - (y_ndc + 1.0f) * 0.5f) * height;
            
            linearDepths[i] = w; 
            validVerts[i] = true;
        }

        for (const auto& triIdx : _tris) {
            int i0 = triIdx.x();
            int i1 = triIdx.y();
            int i2 = triIdx.z();

            if (!validVerts[i0] || !validVerts[i1] || !validVerts[i2]) {
                continue;
            }

            Triangle2D t2d;
            t2d.a = screenCoords[i0];
            t2d.b = screenCoords[i1];
            t2d.c = screenCoords[i2];

            t2d.depth = (linearDepths[i0] + linearDepths[i1] + linearDepths[i2]) / 3.0f;

            // Handle Coloring
            if (_colors.size() == n) {
                t2d.color = (_colors[i0] + _colors[i1] + _colors[i2]) / 3.0f;
            } else if (_colors.size() == 1) {
                t2d.color = _colors[0];
            } else {
                t2d.color = {1.0f, 0.0f, 1.0f};
            }

            renderList.push_back(t2d);
        }

        std::sort(renderList.begin(), renderList.end(), [](const Triangle2D& a, const Triangle2D& b) {
            return a.depth > b.depth;
        });

        return renderList;
    }
    
    frame renderFrame(Camera cam, int height, int width, float near, float far, frame::colormap colorformat = frame::colormap::RGB) {
        std::vector<Triangle2D> triangles = project_2d(cam, height, width, near, far);
        return rasterizeTriangles(triangles, width, height, colorformat);
    }

    static frame rasterizeTriangles(const std::vector<Triangle2D>& triangles, int width, int height, frame::colormap colorformat) {
        frame outFrame(width, height, colorformat);
        std::vector<float> buffer(width * height * 3);
        
        for (const auto& tri : triangles) {
            int minX = std::max(0, (int)std::floor(std::min({tri.a.x(), tri.b.x(), tri.c.x()})));
            int maxX = std::min(width - 1, (int)std::ceil(std::max({tri.a.x(), tri.b.x(), tri.c.x()})));
            int minY = std::max(0, (int)std::floor(std::min({tri.a.y(), tri.b.y(), tri.c.y()})));
            int maxY = std::min(height - 1, (int)std::ceil(std::max({tri.a.y(), tri.b.y(), tri.c.y()})));

            float area = edgeFunction(tri.a, tri.b, tri.c);

            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    Vector2f p(x + 0.5f, y + 0.5f);

                    float w0 = edgeFunction(tri.b, tri.c, p);
                    float w1 = edgeFunction(tri.c, tri.a, p);
                    float w2 = edgeFunction(tri.a, tri.b, p);
                    bool inside = false;
                    
                    if (area >= 0) {
                        if (w0 >= 0 && w1 >= 0 && w2 >= 0) inside = true;
                    } else {
                        if (w0 <= 0 && w1 <= 0 && w2 <= 0) inside = true;
                    }

                    if (inside) {
                        int index = (y * width + x) * 3;
                        if (index >= 0 && index < buffer.size() - 2) {
                            buffer[index    ] = tri.color.x();
                            buffer[index + 1] = tri.color.y();
                            buffer[index + 2] = tri.color.z();
                        }
                    }
                }
            }
        }

        outFrame.setData(buffer, colorformat);
        return outFrame;
    }

    void printStats(std::ostream& os = std::cout) const {
        os << "========================================\n";
        os << "              Mesh STATS                \n";
        os << "========================================\n";
        os << "Structure:\n";
        os << "  Total Vertices    : " << _vertices.size() << "\n";
        os << "  Total Tris        : " << _tris.size() << "\n";
        os << "  Total Polys       : " << _polys.size() << "\n";
        os << "  colors            : " << _colors.size() << "\n";
    }
};

class Scene {
private:
    std::vector<std::shared_ptr<Mesh>> _meshes;
    std::string cachedStats;
    std::chrono::steady_clock::time_point lastStatsUpdate;
    bool statsNeedUpdate = true;
    const std::chrono::seconds STATS_UPDATE_INTERVAL{60};
public:
    Scene() {}

    void addMesh(std::shared_ptr<Mesh> mesh) {
        _meshes.push_back(mesh);
    }

    void clear() {
        _meshes.clear();
    }

    frame render(Camera cam, int height, int width, float near, float far, frame::colormap colorformat = frame::colormap::RGB) {
        std::vector<Triangle2D> allTriangles;

        for (auto& mesh : _meshes) {
            std::vector<Triangle2D> meshTris = mesh->project_2d(cam, height, width, near, far);
            allTriangles.insert(allTriangles.end(), meshTris.begin(), meshTris.end());
        }

        std::sort(allTriangles.begin(), allTriangles.end(), [](const Triangle2D& a, const Triangle2D& b) {
            return a.depth > b.depth;
        });

        return Mesh::rasterizeTriangles(allTriangles, width, height, colorformat);
    }

    void drawSceneWindow(const char* windowTitle, Camera& cam, float nearClip = 0.1f, float farClip = 1000.0f, bool showFps = true) {
        ImGui::Begin(windowTitle);

        // Get the position and size of the content area within the ImGui window
        ImVec2 canvas_p = ImGui::GetCursorScreenPos();
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();

        if (canvas_sz.x < 1.0f || canvas_sz.y < 1.0f) {
            ImGui::End();
            return;
        }

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        // Clip drawing to the specific window area
        draw_list->PushClipRect(canvas_p, ImVec2(canvas_p.x + canvas_sz.x, canvas_p.y + canvas_sz.y), true);

        // Draw background
        draw_list->AddRectFilled(canvas_p, ImVec2(canvas_p.x + canvas_sz.x, canvas_p.y + canvas_sz.y), IM_COL32(20, 20, 20, 255));
        
        // Collect and project triangles
        std::vector<Triangle2D> allTriangles;
        for (auto& mesh : _meshes) {
            std::vector<Triangle2D> meshTris = mesh->project_2d(cam, (int)canvas_sz.y, (int)canvas_sz.x, nearClip, farClip);
            allTriangles.insert(allTriangles.end(), meshTris.begin(), meshTris.end());
        }

        // Sort by depth (Painter's Algorithm)
        std::sort(allTriangles.begin(), allTriangles.end(), [](const Triangle2D& a, const Triangle2D& b) {
            return a.depth > b.depth;
        });

        // Draw triangles
        for (const auto& tri : allTriangles) {
            ImVec2 p1(canvas_p.x + tri.a.x(), canvas_p.y + tri.a.y());
            ImVec2 p2(canvas_p.x + tri.b.x(), canvas_p.y + tri.b.y());
            ImVec2 p3(canvas_p.x + tri.c.x(), canvas_p.y + tri.c.y());

            // Simple culling optimization
            // float minX = std::min({p1.x, p2.x, p3.x});
            // float minY = std::min({p1.y, p2.y, p3.y});
            // float maxX = std::max({p1.x, p2.x, p3.x});
            // float maxY = std::max({p1.y, p2.y, p3.y});

            // if (maxX < canvas_p.x || minX > canvas_p.x + canvas_sz.x || 
            //     maxY < canvas_p.y || minY > canvas_p.y + canvas_sz.y) {
            //     continue;
            // }

            // float r = std::clamp(tri.color.x(), 0.0f, 1.0f);
            // float g = std::clamp(tri.color.y(), 0.0f, 1.0f);
            // float b = std::clamp(tri.color.z(), 0.0f, 1.0f);
            
            ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(tri.color.x(), tri.color.y(), tri.color.z(), 1.0f));
            draw_list->AddTriangleFilled(p1, p2, p3, col);
        }

        // Draw FPS Counter
        if (showFps) {
            char fpsText[32];
            snprintf(fpsText, sizeof(fpsText), "FPS: %.1f", ImGui::GetIO().Framerate);
            
            ImVec2 txtSz = ImGui::CalcTextSize(fpsText);
            // Position: Top Right with 10px padding
            ImVec2 txtPos = ImVec2(canvas_p.x + canvas_sz.x - txtSz.x - 10.0f, canvas_p.y + 10.0f);
            
            // Semi-transparent background for text
            draw_list->AddRectFilled(
                ImVec2(txtPos.x - 5.0f, txtPos.y - 2.0f), 
                ImVec2(txtPos.x + txtSz.x + 5.0f, txtPos.y + txtSz.y + 2.0f), 
                IM_COL32(0, 0, 0, 150), 
                4.0f
            );
            
            draw_list->AddText(txtPos, IM_COL32(255, 255, 255, 255), fpsText);
        }

        draw_list->PopClipRect();

        // Invisible button to handle inputs over the viewport area
        ImGui::InvisibleButton("viewport_btn", canvas_sz);

        ImGui::End();
    }

    void printStats(std::ostream& os = std::cout) const {
        os << "========================================\n";
        os << "             Scene STATS                \n";
        os << "========================================\n";
        for (auto m : _meshes) {
            m->printStats(os);
        }
    }

    void drawGridStats() {
        auto now = std::chrono::steady_clock::now();
        if ((now - lastStatsUpdate) > STATS_UPDATE_INTERVAL || statsNeedUpdate) {
            std::stringstream meshStats;
            printStats(meshStats);
            cachedStats = meshStats.str();
            lastStatsUpdate = std::chrono::steady_clock::now();
            statsNeedUpdate = false;
        }
        ImGui::TextUnformatted(cachedStats.c_str());
    }
    void updateStats() {
        statsNeedUpdate = true;
    }
};

#endif
