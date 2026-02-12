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
#include "../../eigen/Eigen/Dense"
#include "../../eigen/Eigen/Geometry"
#include "./camera.hpp"
#include "../output/frame.hpp"

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
};


class Scene {
private:
    std::vector<std::shared_ptr<Mesh>> _meshes;

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
};

#endif
