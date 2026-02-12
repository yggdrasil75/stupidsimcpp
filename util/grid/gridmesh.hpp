#ifndef GRIDMESH_HPP
#define GRIDMESH_HPP

#include "grid3eigen.hpp"
#include "mesh.hpp"
#include "../basicdefines.hpp"
#include <vector>
#include <limits>
#include <memory>
#include <unordered_map>
#include <map>
#include <cmath>

struct GridPoint {
    Eigen::Vector3f pos;
    float val;
    Eigen::Vector3f color; // Interpolated color field
};

struct GridCell {
    GridPoint p[8];
};

template<typename T>
class MetaballMesher {
private:
    float _resolution; 
    float _isoLevel;
    float _influenceRadius;

    void vertexInterp(float isolevel, GridPoint p1, GridPoint p2, Eigen::Vector3f& outPos, Eigen::Vector3f& outColor) {
        if (std::abs(p1.val - p2.val) < 0.00001f) {
            outPos = p1.pos;
            outColor = p1.color;
            return;
        }

        float mu = (isolevel - p1.val) / (p2.val - p1.val);
        outPos = p1.pos + mu * (p2.pos - p1.pos);
        outColor = p1.color + mu * (p2.color - p1.color);
    }

    void sampleField(const Eigen::Vector3f& pos, const std::vector<std::shared_ptr<typename Octree<T>::NodeData>>& nodes, float& outVal, Eigen::Vector3f& outColor) {
        float sumVal = 0.0f;
        Eigen::Vector3f sumColor = Eigen::Vector3f::Zero();
        float totalWeight = 0.0f;
        
        for (const auto& node : nodes) {
            float r = node->size * _influenceRadius;
            float distSq = (pos - node->position).squaredNorm();
            
            if (distSq < r * r) {
                float dist = std::sqrt(distSq);
                float x = dist / r;
                float val = std::pow(1.0f - x*x, 3.0f);
                
                sumVal += val;
                sumColor += node->color * val;
                totalWeight += val;
            }
        }

        outVal = sumVal;
        if (totalWeight > 0.0001f) {
            outColor = sumColor / totalWeight;
        } else {
            outColor = Eigen::Vector3f(1.0f, 0.0f, 1.0f);
        }
    }

public:
    MetaballMesher(float resolution = 0.5f, float isoLevel = 0.5f, float influenceMult = 2.5f) 
        : _resolution(resolution), _isoLevel(isoLevel), _influenceRadius(influenceMult) {}

    void polygonize(const std::vector<std::shared_ptr<typename Octree<T>::NodeData>>& nodes, 
                   std::vector<Eigen::Vector3f>& outVerts, 
                   std::vector<Eigen::Vector3f>& outColors,
                   std::vector<int>& outIndices,
                   int& startIndex) {
        
        if (nodes.empty()) return;

        Eigen::Vector3f minBounds(1e9, 1e9, 1e9);
        Eigen::Vector3f maxBounds(-1e9, -1e9, -1e9);
        float maxNodeSize = 0;

        for (const auto& node : nodes) {
            minBounds = minBounds.cwiseMin(node->position);
            maxBounds = maxBounds.cwiseMax(node->position);
            if (node->size > maxNodeSize) maxNodeSize = node->size;
        }

        float padding = maxNodeSize * _influenceRadius;
        minBounds -= Eigen::Vector3f(padding, padding, padding);
        maxBounds += Eigen::Vector3f(padding, padding, padding);

        float step = maxNodeSize * _resolution;
        if (step <= 0.0001f) step = 0.1f;

        int stepsX = static_cast<int>((maxBounds.x() - minBounds.x()) / step);
        int stepsY = static_cast<int>((maxBounds.y() - minBounds.y()) / step);
        int stepsZ = static_cast<int>((maxBounds.z() - minBounds.z()) / step);
        
        for (int x = 0; x < stepsX; ++x) {
            for (int y = 0; y < stepsY; ++y) {
                for (int z = 0; z < stepsZ; ++z) {
                    
                    Eigen::Vector3f basePos = minBounds + Eigen::Vector3f(x*step, y*step, z*step);
                    GridCell cell;
                    
                    Eigen::Vector3f offsets[8] = {
                        {0,0,0}, {step,0,0}, {step,0,step}, {0,0,step},
                        {0,step,0}, {step,step,0}, {step,step,step}, {0,step,step}
                    };

                    int cubeIndex = 0;
                    for (int i = 0; i < 8; ++i) {
                        cell.p[i].pos = basePos + offsets[i];
                        sampleField(cell.p[i].pos, nodes, cell.p[i].val, cell.p[i].color);
                        if (cell.p[i].val < _isoLevel) cubeIndex |= (1 << i);
                    }

                    // Look up edges
                    if (edgeTable[cubeIndex] == 0) continue;
                    if (edgeTable[cubeIndex] == 255) continue; // Inside or Outside completely

                    Eigen::Vector3f vertList[12];
                    Eigen::Vector3f colorList[12];

                    if (edgeTable[cubeIndex] & 1) vertexInterp(_isoLevel, cell.p[0], cell.p[1], vertList[0], colorList[0]);
                    if (edgeTable[cubeIndex] & 2) vertexInterp(_isoLevel, cell.p[1], cell.p[2], vertList[1], colorList[1]);
                    if (edgeTable[cubeIndex] & 4) vertexInterp(_isoLevel, cell.p[2], cell.p[3], vertList[2], colorList[2]);
                    if (edgeTable[cubeIndex] & 8) vertexInterp(_isoLevel, cell.p[3], cell.p[0], vertList[3], colorList[3]);
                    if (edgeTable[cubeIndex] & 16) vertexInterp(_isoLevel, cell.p[4], cell.p[5], vertList[4], colorList[4]);
                    if (edgeTable[cubeIndex] & 32) vertexInterp(_isoLevel, cell.p[5], cell.p[6], vertList[5], colorList[5]);
                    if (edgeTable[cubeIndex] & 64) vertexInterp(_isoLevel, cell.p[6], cell.p[7], vertList[6], colorList[6]);
                    if (edgeTable[cubeIndex] & 128) vertexInterp(_isoLevel, cell.p[7], cell.p[4], vertList[7], colorList[7]);
                    if (edgeTable[cubeIndex] & 256) vertexInterp(_isoLevel, cell.p[0], cell.p[4], vertList[8], colorList[8]);
                    if (edgeTable[cubeIndex] & 512) vertexInterp(_isoLevel, cell.p[1], cell.p[5], vertList[9], colorList[9]);
                    if (edgeTable[cubeIndex] & 1024) vertexInterp(_isoLevel, cell.p[2], cell.p[6], vertList[10], colorList[10]);
                    if (edgeTable[cubeIndex] & 2048) vertexInterp(_isoLevel, cell.p[3], cell.p[7], vertList[11], colorList[11]);

                    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                        outVerts.push_back(vertList[triTable[cubeIndex][i]]);
                        outVerts.push_back(vertList[triTable[cubeIndex][i+1]]);
                        outVerts.push_back(vertList[triTable[cubeIndex][i+2]]);
                        
                        outColors.push_back(colorList[triTable[cubeIndex][i]]);
                        outColors.push_back(colorList[triTable[cubeIndex][i+1]]);
                        outColors.push_back(colorList[triTable[cubeIndex][i+2]]);

                        outIndices.push_back(startIndex++);
                        outIndices.push_back(startIndex++);
                        outIndices.push_back(startIndex++);
                    }
                }
            }
        }
    }
};

template<typename T>
class CachedGridMesh {
private:
    std::unique_ptr<Mesh> mesh_;
    int id_;
    
    float _resolution = 0.5f;
    float _isoLevel = 0.4f;

public:
    CachedGridMesh(int id = 0) : id_(id) {
        mesh_ = std::make_unique<Mesh>(id_, std::vector<Vector3f>{}, std::vector<std::vector<int>>{}, std::vector<Color>{});
    }

    void setResolution(float res) { _resolution = res; }
    void setIsoLevel(float iso) { _isoLevel = iso; }

    void update(Octree<T>& grid, const Eigen::Vector3f& center = Eigen::Vector3f::Zero(), float radius = std::numeric_limits<float>::max()) {
        std::vector<Vector3f> allVertices;
        std::vector<std::vector<int>> allPolys;
        std::vector<Color> allColors;
        
        auto nodes = grid.findInRadius(center, radius);
        if (nodes.empty()) {
            mesh_ = std::make_unique<Mesh>(id_, allVertices, allPolys, allColors);
            return;
        }
        std::unordered_map<int, std::vector<std::shared_ptr<typename Octree<T>::NodeData>>> objectGroups;
        
        for (const auto& node : nodes) {
            if (!node->active || !node->visible) continue;
            objectGroups[node->objectId].push_back(node);
        }
        std::cout << "object map made" << std::endl;

        MetaballMesher<T> mesher(_resolution, _isoLevel);
        
        int globalVertIndex = 0;
        
        for (auto& [objId, groupNodes] : objectGroups) {
            std::vector<Eigen::Vector3f> objVerts;
            std::vector<Eigen::Vector3f> objColors;
            std::vector<int> objIndices;
            int startIndex = globalVertIndex;
            int localVertCounter = 0; 
            
            std::vector<Eigen::Vector3f> mVerts;
            std::vector<Eigen::Vector3f> mColors;
            std::vector<int> mIndices;
            int mIdxStart = 0;

            mesher.polygonize(groupNodes, mVerts, mColors, mIndices, mIdxStart);

            for (size_t i = 0; i < mVerts.size(); ++i) {
                allVertices.push_back(mVerts[i]);
                allColors.push_back(mColors[i]);
            }

            for (size_t i = 0; i < mIndices.size(); i += 3) {
                std::vector<int> tri = {
                    mIndices[i] + globalVertIndex, 
                    mIndices[i+1] + globalVertIndex, 
                    mIndices[i+2] + globalVertIndex
                };
                allPolys.push_back(tri);
            }

            globalVertIndex += mVerts.size();
            std::cout << "object done" << std::endl;
        }

        mesh_ = std::make_unique<Mesh>(id_, allVertices, allPolys, allColors);
        mesh_->triangulate(); 
    }

    frame render(Camera cam, int height, int width, float near = 0.1f, float far = 1000.0f, frame::colormap format = frame::colormap::RGB) {
        if (!mesh_) {
            return frame(width, height, format);
        }
        return mesh_->renderFrame(cam, height, width, near, far, format);
    }

    Mesh* getMesh() {
        return mesh_.get();
    }
    
    size_t getVertexCount() const {
        return mesh_ ? mesh_->vertices().size() : 0;
    }
};

#endif