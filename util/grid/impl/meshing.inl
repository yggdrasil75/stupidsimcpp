#pragma once
#include "../../basicdefines.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace Grid {
struct MeshFaceDir {
    int nx, ny, nz;
    int axis;
    int sign;
};

inline constexpr MeshFaceDir kFaces[6] = {
    { 1, 0, 0, 0, +1}, {-1, 0, 0, 0, -1},
    { 0, 1, 0, 1, +1}, { 0,-1, 0, 1, -1},
    { 0, 0, 1, 2, +1}, { 0, 0,-1, 2, -1},
};

inline Eigen::Vector3i meshFaceNormal(const MeshFaceDir& f) {
    return Eigen::Vector3i(f.nx, f.ny, f.nz);
}

inline constexpr int kMCcornerOff[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};

inline Eigen::Vector3i kMCcorner(int i) {
    return Eigen::Vector3i(kMCcornerOff[i][0], kMCcornerOff[i][1], kMCcornerOff[i][2]);
}

inline constexpr int kMCedge[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

template<typename Obj>
inline size_t meshAddVertex(Obj& obj, uint32_t colorIdx, const Eigen::Vector3f& p) {
    size_t i = obj.objMesh.vertices.size();
    obj.objMesh.vertices.push_back({static_cast<decltype(obj.objMesh.vertices[0].cIdx)>(colorIdx), p});
    return i;
}

template<typename Obj>
inline void meshAddQuad(Obj& obj, size_t a, size_t b, size_t c, size_t d) {
    obj.objMesh.tris.push_back({a, b, c});
    obj.objMesh.tris.push_back({a, c, d});
}

}
