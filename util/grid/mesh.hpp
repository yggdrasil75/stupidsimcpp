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

using Vector2f = Eigen::Vector2f;
using Vector3f = Eigen::Vector3f;
using Vector4f = Eigen::Vector4f;
using Matrix4f = Eigen::Matrix4f;
using Color = Eigen::Vector3f;

class Mesh {
private:
    int id;
    std::vector<Vector3f> _vertices;
    std::vector<std::vector<int>> _polys;
    std::vector<Color> _colors;
    mutable std::vector<Eigen::Vector3i> _tris;
    mutable bool _needs_triangulation = true;
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
        std::vector<Eigen::Vector3i> newPols;
        if (!_needs_triangulation) return;
        for (auto& pol : _polys) {
            if (pol.size() > 3) {
                auto v0 = pol[0];
                for (int i = 0; i < pol.size() - 1; i++) {
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
        if (colorlist.size() == 1) {
            _colors = colorlist;
        } else if (colorlist.size() == _vertices.size()) {
            _colors = colorlist;
        }
    }

    void project_2d(Camera cam, int height, int width, float near, float far) {
        Vector3f zaxis = cam.direction - cam.origin;
        zaxis = zaxis / zaxis.norm();
        Vector3f xAxis = cam.up.cross(zaxis);
        xAxis = xAxis / xAxis.norm();
        Vector3f yAxis = zaxis.cross(xAxis);

        Matrix4f viewMatrix = Matrix4f::Identity();
        viewMatrix.block<3,1>(0,0) = xAxis;
        viewMatrix.block<3,1>(0,1) = yAxis;
        viewMatrix.block<3,1>(0,2) = -zaxis;
        viewMatrix.block<3,1>(0,3) = cam.origin;

        viewMatrix = viewMatrix.inverse();

        float aspect = float(height) / float(width);
        float fovrad = cam.fovRad();
        float tanh = std::tan(fovrad / 2);
        
        Matrix4f projMatrix = Matrix4f::Zero();
        projMatrix(0,0) = 1.0 / (aspect * tanh);
        projMatrix(1,1) = 1.0 / tanh;
        projMatrix(2,2) = -(far + near) / (near - far);
        projMatrix(2,3) = (-2.0f * far * near) / (near - far);
        projMatrix(3,2) = -1.0f;

        Vector3f viewDir = (cam.direction - cam.origin).normalized();

        Vector3f mesh_center = _vertices.colwise().mean(); 

    }

};

#endif