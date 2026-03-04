#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
#include "mesh.hpp"
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <cstdint>

#ifdef SSE
#include <immintrin.h>
#endif

constexpr int Dim = 3;

template<typename T, typename IndexType = uint16_t>
class Octree {
public:
    using PointType = Eigen::Matrix<float, Dim, 1>;
    using BoundingBox = std::pair<PointType, PointType>;

    struct Vector3fCompare {
        bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
            return std::tie(a.x(), a.y(), a.z()) < std::tie(b.x(), b.y(), b.z());
        }
    };

    struct Material {
        float emittance;
        float roughness;
        float metallic;
        float transmission;
        float ior;

        Material(float e = 0.0f, float r = 1.0f, float m = 0.0f, float t = 0.0f, float i = 1.45f)
            : emittance(e), roughness(r), metallic(m), transmission(t), ior(i) {}

        bool operator<(const Material& o) const {
            if (emittance != o.emittance) return emittance < o.emittance;
            if (roughness != o.roughness) return roughness < o.roughness;
            if (metallic != o.metallic) return metallic < o.metallic;
            if (transmission != o.transmission) return transmission < o.transmission;
            return ior < o.ior;
        }
    };
    
    struct NodeData {
        T data;
        PointType position;
        int objectId;
        int subId;
        float size;
        IndexType colorIdx;
        IndexType materialIdx;
        bool active;
        bool visible;

        NodeData(const T& data, const PointType& pos, bool visible, IndexType colorIdx, float size = 0.01f,
                 bool active = true, int objectId = -1, int subId = 0, IndexType materialIdx = 0) 
                : data(data), position(pos), objectId(objectId), subId(subId), size(size), 
                  colorIdx(colorIdx), materialIdx(materialIdx), active(active), visible(visible) {}
        
        NodeData() : objectId(-1), subId(0), size(0.0f), colorIdx(0), materialIdx(0), 
                     active(false), visible(false) {}
        
        PointType getHalfSize() const {
            return PointType(size * 0.5f, size * 0.5f, size * 0.5f);
        }
        
        BoundingBox getCubeBounds() const {
            PointType halfSize = getHalfSize();
            return {position - halfSize, position + halfSize};
        }
    };

    struct OctreeNode {
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::unique_ptr<OctreeNode>, 8> children;
        PointType center;
        float nodeSize;
        bool isLeaf;
        
        mutable std::shared_ptr<NodeData> lodData;
        mutable std::mutex lodMutex; 

        OctreeNode(const PointType& min, const PointType& max) : bounds(min,max), isLeaf(true), lodData(nullptr) {
            for (std::unique_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5;
            nodeSize = (bounds.second - bounds.first).norm();
        }

        bool contains(const PointType& point) const {
            return (point[0] >= bounds.first[0] && point[0] <= bounds.second[0] &&
                    point[1] >= bounds.first[1] && point[1] <= bounds.second[1] &&
                    point[2] >= bounds.first[2] && point[2] <= bounds.second[2]);
        }

        bool isEmpty() const {
            return points.empty();
        }
    };

private:
    std::unique_ptr<OctreeNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    Eigen::Vector3f skylight_ = {0.1f, 0.1f, 0.1f};
    Eigen::Vector3f backgroundColor_ = {0.53f, 0.81f, 0.92f};
    
    // Addressable Maps
    std::unique_ptr<std::mutex> mapMutex_;
    std::vector<Eigen::Vector3f> colorMap_;
    std::map<Eigen::Vector3f, IndexType, Vector3fCompare> colorToIndex_;
    
    std::vector<Material> materialMap_;
    std::map<Material, IndexType> materialToIndex_;

    std::map<std::pair<int, int>, std::shared_ptr<Mesh>> meshCache_;
    std::set<std::pair<int, int>> dirtyMeshes_;
    int nextSubIdGenerator_ = 1;
    
public:
    inline IndexType getColorIndex(const Eigen::Vector3f& color) {
        std::lock_guard<std::mutex> lock(*mapMutex_);
        auto it = colorToIndex_.find(color);
        if (it != colorToIndex_.end()) return it->second;
        
        if (colorMap_.size() >= std::numeric_limits<IndexType>::max()) {
            IndexType bestIdx = 0;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < colorMap_.size(); ++i) {
                float dist = (colorMap_[i] - color).squaredNorm();
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<IndexType>(i);
                }
            }
            return bestIdx;
        }

        IndexType idx = static_cast<IndexType>(colorMap_.size());
        colorMap_.push_back(color);
        colorToIndex_[color] = idx;
        return idx;
    }

    inline const Eigen::Vector3f& getColor(IndexType idx) const {
        if (idx < colorMap_.size()) return colorMap_[idx];
        static const Eigen::Vector3f fallback = Eigen::Vector3f::Zero();
        return fallback;
    }

    inline IndexType getMaterialIndex(const Material& mat) {
        std::lock_guard<std::mutex> lock(*mapMutex_);
        auto it = materialToIndex_.find(mat);
        if (it != materialToIndex_.end()) return it->second;

        if (materialMap_.size() >= std::numeric_limits<IndexType>::max()) {
            IndexType bestIdx = 0;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < materialMap_.size(); ++i) {
                float d_e = materialMap_[i].emittance - mat.emittance;
                float d_r = materialMap_[i].roughness - mat.roughness;
                float d_m = materialMap_[i].metallic - mat.metallic;
                float d_t = materialMap_[i].transmission - mat.transmission;
                float d_i = materialMap_[i].ior - mat.ior;
                float dist = d_e*d_e + d_r*d_r + d_m*d_m + d_t*d_t + d_i*d_i;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<IndexType>(i);
                }
            }
            return bestIdx;
        }

        IndexType idx = static_cast<IndexType>(materialMap_.size());
        materialMap_.push_back(mat);
        materialToIndex_[mat] = idx;
        return idx;
    }

    inline const Material& getMaterial(IndexType idx) const {
        if (idx < materialMap_.size()) return materialMap_[idx];
        static const Material fallback;
        return fallback;
    }

private:
    void invalidateMesh(int objectId, int subId) {
        if (objectId < 0) return;
        dirtyMeshes_.insert({objectId, subId});
    }

    void collectNodesBySubIdRecursive(OctreeNode* node, int objId, int subId, std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) const {
        if (!node) return;
        for (const auto& pt : node->points) {
            if (pt->active && pt->objectId == objId && pt->subId == subId) {
                if (seen.insert(pt).second) {
                    results.push_back(pt);
                }
            }
        }
        if (!node->isLeaf) {
            for (const auto& child : node->children) {
                if (child) collectNodesBySubIdRecursive(child.get(), objId, subId, results, seen);
            }
        }
    }
    
    void collectNodesBySubId(OctreeNode* node, int objId, int subId, std::vector<std::shared_ptr<NodeData>>& results) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        collectNodesBySubIdRecursive(node, objId, subId, results, seen);
    }

    float lodFalloffRate_ = 0.1f; // Lower = better, higher = worse. 0-1
    float lodMinDistance_ = 100.0f;
    float maxDistance_ = size * size;
    
    struct Ray {
        PointType origin;
        PointType dir;
        PointType invDir;
        uint8_t sign[3];
        uint8_t signMask;
        Ray(const PointType& orig, const PointType& dir) : origin(orig), dir(dir) {
            invDir = dir.cwiseInverse();
            sign[0] = (invDir[0] < 0);
            sign[1] = (invDir[1] < 0);
            sign[2] = (invDir[2] < 0);
            signMask = (sign[0] | sign[1] << 1 | sign[2] << 2);
        }
    };

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
        // uint8_t octant = 0;
        // if (point[0] >= center[0]) octant |= 1;
        // if (point[1] >= center[1]) octant |= 2;
        // if (point[2] >= center[2]) octant |= 4;
        // return octant;
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint8_t octant) const {
        PointType childMin, childMax;
        const PointType& center = node->center;
        
        childMin[0] = (octant & 1) ? center[0] : node->bounds.first[0];
        childMax[0] = (octant & 1) ? node->bounds.second[0] : center[0];
        
        childMin[1] = (octant & 2) ? center[1] : node->bounds.first[1];
        childMax[1] = (octant & 2) ? node->bounds.second[1] : center[1];
        
        childMin[2] = (octant & 4) ? center[2] : node->bounds.first[2];
        childMax[2] = (octant & 4) ? node->bounds.second[2] : center[2];

        return {childMin, childMax};
    }

    bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const {
        return (a.first[0] <= b.second[0] && a.second[0] >= b.first[0] &&
                a.first[1] <= b.second[1] && a.second[1] >= b.first[1] &&
                a.first[2] <= b.second[2] && a.second[2] >= b.first[2]);
    }

    void splitNode(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
        }

        std::vector<std::shared_ptr<NodeData>> keep;
        for (const auto& pointData : node->points) {
            // Keep massive objects in the parent
            if (pointData->size >= node->nodeSize) {
                keep.emplace_back(pointData);
                continue;
            }

            BoundingBox cubeBounds = pointData->getCubeBounds();
            for (int i = 0; i < 8; ++i) {
                if (boxIntersectsBox(node->children[i]->bounds, cubeBounds)) {
                    node->children[i]->points.emplace_back(pointData);
                }
            }
        }

        node->points = std::move(keep);
        node->isLeaf = false;

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        {
            std::lock_guard<std::mutex> lock(node->lodMutex);
            node->lodData = nullptr;
        }

        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxIntersectsBox(node->bounds, cubeBounds)) return false;

        if (!node->isLeaf && pointData->size >= node->nodeSize) {
            node->points.emplace_back(pointData);
            return true;
        }

        if (node->isLeaf) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            return true;
        } else {
            bool inserted = false;
            for (int i = 0; i < 8; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                if (boxIntersectsBox(childBounds, cubeBounds)) {
                    if (!node->children[i]) {
                        node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
                    }
                    inserted |= insertRecursive(node->children[i].get(), pointData, depth + 1);
                }
            }
            return inserted;
        }
    }

    bool invalidateNodeLODRecursive(OctreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        {
            std::lock_guard<std::mutex> lock(node->lodMutex);
            node->lodData = nullptr;
        }
        if (!node->isLeaf) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    invalidateNodeLODRecursive(node->children[i].get(), bounds);
                }
            }
        }
        return true;
    }

    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        if (root_ && pointData) {
            invalidateNodeLODRecursive(root_.get(), pointData->getCubeBounds());
        }
    }
    
    void ensureBounds(const BoundingBox& targetBounds) {
        if (!root_) {
            PointType center = (targetBounds.first + targetBounds.second) * 0.5f;
            PointType size = targetBounds.second - targetBounds.first;
            float maxDim = size.maxCoeff();
            if (maxDim <= 0.0f) maxDim = 1.0f;
            PointType halfSize = PointType::Constant(maxDim * 0.5f);
            root_ = std::make_unique<OctreeNode>(center - halfSize, center + halfSize);
            return;
        }

        while (true) {
            bool xInside = root_->bounds.first.x() <= targetBounds.first.x() && root_->bounds.second.x() >= targetBounds.second.x();
            bool yInside = root_->bounds.first.y() <= targetBounds.first.y() && root_->bounds.second.y() >= targetBounds.second.y();
            bool zInside = root_->bounds.first.z() <= targetBounds.first.z() && root_->bounds.second.z() >= targetBounds.second.z();

            if (xInside && yInside && zInside) {
                break;
            }

            PointType min = root_->bounds.first;
            PointType max = root_->bounds.second;
            PointType size = max - min;
            
            int expandX = (targetBounds.first.x() < min.x()) ? -1 : 1;
            int expandY = (targetBounds.first.y() < min.y()) ? -1 : 1;
            int expandZ = (targetBounds.first.z() < min.z()) ? -1 : 1;
            
            PointType newMin = min;
            PointType newMax = max;
            
            if (expandX < 0) newMin.x() -= size.x(); else newMax.x() += size.x();
            if (expandY < 0) newMin.y() -= size.y(); else newMax.y() += size.y();
            if (expandZ < 0) newMin.z() -= size.z(); else newMax.z() += size.z();
            
            auto newRoot = std::make_unique<OctreeNode>(newMin, newMax);
            newRoot->isLeaf = false;
            
            uint8_t oldOctant = 0;
            if (expandX < 0) oldOctant |= 1;
            if (expandY < 0) oldOctant |= 2;
            if (expandZ < 0) oldOctant |= 4;
            
            newRoot->children[oldOctant] = std::move(root_);
            root_ = std::move(newRoot);
            
            maxDepth++; 
        }
    }

    void ensureLOD(OctreeNode* node) {
        std::lock_guard<std::mutex> lock(node->lodMutex);
        if (node->lodData != nullptr) return;

        PointType avgPos = PointType::Zero();
        Eigen::Vector3f avgColor = Eigen::Vector3f::Zero();
        float avgEmittance = 0.0f;
        float avgRoughness = 0.0f;
        float avgMetallic = 0.0f;
        float avgTransmission = 0.0f;
        float avgIor = 0.0f;
        int count = 0;
        
        if (node->isLeaf && node->points.size() == 1) {
            node->lodData = node->points[0];
            return;
        } else if (node->isLeaf && node->points.empty()) {
            return;
        }

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!item || !item->active || !item->visible) return;
            avgColor += getColor(item->colorIdx);
            Material mat = getMaterial(item->materialIdx);
            avgEmittance += mat.emittance;
            avgRoughness += mat.roughness;
            avgMetallic += mat.metallic;
            avgTransmission += mat.transmission;
            avgIor += mat.ior;
            count++;
        };

        for(const auto& pt : node->points) accumulate(pt);

        for (const auto& child : node->children) {
            if (child) {
                ensureLOD(child.get());
                if (child->lodData) {
                    accumulate(child->lodData);
                }
            }
        }

        if (count > 0) {
            float invCount = 1.0f / count;
            
            auto lod = std::make_shared<NodeData>();
            lod->position = node->center;
            
            PointType nodeDims = node->bounds.second - node->bounds.first;
            lod->size = nodeDims.maxCoeff();

            lod->colorIdx = getColorIndex(avgColor * invCount);
            Material avgMat(avgEmittance * invCount, avgRoughness * invCount, 
                            avgMetallic * invCount, avgTransmission * invCount, avgIor * invCount);
            lod->materialIdx = getMaterialIndex(avgMat);
            
            lod->active = true;
            lod->visible = true;
            lod->objectId = -1; 

            node->lodData = lod;
        }
    }

    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, const PointType& pos, float tolerance) const {
        if (!node->contains(pos)) return nullptr;
        
        for (const auto& pointData : node->points) {
            if (!pointData->active) continue;
            
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        if (!node->isLeaf) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                return findRecursive(node->children[octant].get(), pos, tolerance);
            }
        }
        return nullptr;
    }

    bool removeRecursive(OctreeNode* node, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
        {
            std::lock_guard<std::mutex> lock(node->lodMutex);
            node->lodData = nullptr; 
        }

        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        
        bool foundAny = false;
        
        auto it = std::remove_if(node->points.begin(), node->points.end(),
            [&](const std::shared_ptr<NodeData>& pointData) {
                return pointData == targetPt;
            });
        
        if (it != node->points.end()) {
            node->points.erase(it, node->points.end());
            foundAny = true;
        }

        if (!node->isLeaf) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    foundAny |= removeRecursive(node->children[i].get(), bounds, targetPt);
                }
            }
        }
        return foundAny;
    }

    void searchNodeRecursive(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) const {
        PointType closestPoint;
        for (int i = 0; i < Dim; ++i) {
            closestPoint[i] = std::max(node->bounds.first[i], std::min(center[i], node->bounds.second[i]));
        }
        
        float distSq = (closestPoint - center).squaredNorm();
        if (distSq > radiusSq) {
            return;
        }
        
        for (const auto& pointData : node->points) {
            if (!pointData->active) continue;
            
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid == -1)) {
                if (seen.insert(pointData).second) results.emplace_back(pointData);
            }
        }
        
        if (!node->isLeaf) {
            for (const auto& child : node->children) {
                if (child) searchNodeRecursive(child.get(), center, radiusSq, objectid, results, seen);
            }
        }
    }

    void searchNode(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        searchNodeRecursive(node, center, radiusSq, objectid, results, seen);
    }

    void voxelTraverseRecursive(OctreeNode* node, float tMin, float tMax, float& maxDist, 
                                bool enableLOD, const Ray& ray, std::shared_ptr<NodeData>& hit, float invLodf) const {
        if (enableLOD && !node->isLeaf) {
            float dist = (node->center - ray.origin).norm();
            float ratio = dist / node->nodeSize;
            
            if (dist > lodMinDistance_ && ratio > invLodf && node->lodData) {
                float t;
                PointType n;
                PointType h;
                if (rayCubeIntersect(ray, node->lodData.get(), t, n, h)) {
                    if (t >= 0 && t <= maxDist) {
                        hit = node->lodData;
                        maxDist = t;
                    }
                }
                return;
            }
        }

        for (const auto& pointData : node->points) {
            if (!pointData->active) continue;
            
            float t;
            PointType normal, hitPoint;
            if (rayCubeIntersect(ray, pointData.get(), t, normal, hitPoint)) {
                if (t >= 0 && t <= maxDist && t <= tMax + 0.001f) {
                    maxDist = t;
                    hit = pointData;
                }
            }
        }

        // DDA Traversal
        PointType center = node->center;
        Eigen::Vector3f ttt = (center - ray.origin).cwiseProduct(ray.invDir);

        int currIdx = 0;
        currIdx = ((tMin >= ttt.x()) ? 1 : 0) | ((tMin >= ttt.y()) ? 2 : 0) | ((tMin >= ttt.z()) ? 4 : 0);
        
        float tNext;

        while(tMin < tMax && tMin <= maxDist) {
            Eigen::Vector3f next_t;
            next_t[0] = (currIdx & 1) ? tMax : ttt[0];
            next_t[1] = (currIdx & 2) ? tMax : ttt[1];
            next_t[2] = (currIdx & 4) ? tMax : ttt[2];

            tNext = next_t.minCoeff();

            int physIdx = currIdx ^ ray.signMask;

            if (node->children[physIdx]) {
                voxelTraverseRecursive(node->children[physIdx].get(), tMin, tNext, maxDist, enableLOD, ray, hit, invLodf);
            }

            tMin = tNext;
            currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
        }
    }

    PointType sampleGGX(const PointType& n, float roughness, uint32_t& state) const {
        float alpha = std::max(EPSILON, roughness * roughness);
        float r1 = float(rand_r(&state)) / float(RAND_MAX);
        float r2 = float(rand_r(&state)) / float(RAND_MAX);
        
        float phi = 2.0f * M_PI * r1;
        float denom = 1.0f + (alpha * alpha - 1.0f) * r2;
        denom = std::max(denom, EPSILON);
        float cosTheta = std::sqrt(std::max(0.0f, (1.0f - r2) / denom));
        float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        
        PointType h;
        h[0] = sinTheta * std::cos(phi);
        h[1] = sinTheta * std::sin(phi);
        h[2] = cosTheta;
        
        PointType up = std::abs(n.z()) < 0.999f ? PointType(0,0,1) : PointType(1,0,0);
        PointType tangent = up.cross(n).normalized();
        PointType bitangent = n.cross(tangent);
        
        return (tangent * h[0] + bitangent * h[1] + n * h[2]).normalized();
    }

    PointType sampleCosineHemisphere(const PointType& n, uint32_t& state) const {
        float r1 = float(rand_r(&state)) / float(RAND_MAX);
        float r2 = float(rand_r(&state)) / float(RAND_MAX);
        float phi = 2.0f * M_PI * r1;
        float r = std::sqrt(r2);
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = std::sqrt(std::max(0.0f, 1.0f - x*x - y*y));
        
        PointType up = std::abs(n.z()) < 0.999f ? PointType(0,0,1) : PointType(1,0,0);
        PointType tangent = up.cross(n).normalized();
        PointType bitangent = n.cross(tangent);
        
        return (tangent * x + bitangent * y + n * z).normalized();
    }

    Eigen::Vector3f traceRay(const PointType& rayOrig, const PointType& rayDir, int bounces, uint32_t& rngState,
                             int maxBounces, bool globalIllumination, bool useLod) {
        if (bounces > maxBounces) return globalIllumination ? skylight_ : Eigen::Vector3f::Zero();

        auto hit = voxelTraverse(rayOrig, rayDir, std::numeric_limits<float>::max(), useLod);
        if (!hit) {
            if (bounces == 0) return backgroundColor_;
            return globalIllumination ? skylight_ : Eigen::Vector3f::Zero();
        }

        auto obj = hit;
        
        PointType hitPoint;
        PointType normal;
        float t = 0.0f;
        Ray ray(rayOrig, rayDir);
        rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
        
        Eigen::Vector3f objColor = getColor(obj->colorIdx);
        Material objMat = getMaterial(obj->materialIdx);

        Eigen::Vector3f finalColor = globalIllumination ? skylight_ : Eigen::Vector3f::Zero();
        if (objMat.emittance > 0.0f) {
            finalColor += objColor * objMat.emittance;
        }

        float roughness = std::clamp(objMat.roughness, 0.01f, 1.0f);
        float metallic = std::clamp(objMat.metallic, 0.0f, 1.0f);
        float transmission = std::clamp(objMat.transmission, 0.0f, 1.0f);
        
        PointType V = -rayDir;
        float cosThetaI = normal.dot(V);
        bool isInside = cosThetaI < 0.0f;
        PointType n_eff = isInside ? -normal : normal;
        cosThetaI = std::max(0.001f, n_eff.dot(V));

        float coordMax = hitPoint.cwiseAbs().maxCoeff();
        float rayOffset = std::max(1e-4f, 1e-5f * coordMax);

        Eigen::Vector3f F0 = Eigen::Vector3f::Constant(0.04f);
        F0 = F0 * (1.0f - metallic) + objColor * metallic;

        PointType H = sampleGGX(n_eff, roughness, rngState);
        float VdotH = std::max(0.001f, V.dot(H));
        
        Eigen::Vector3f F_spec = F0 + (Eigen::Vector3f::Constant(1.0f) - F0) * std::pow(std::max(0.0f, 1.0f - VdotH), 5.0f);
        
        PointType specDir = (2.0f * VdotH * H - V).normalized();
        Eigen::Vector3f W_spec = Eigen::Vector3f::Zero();
        
        if (specDir.dot(n_eff) > 0.0f) {
            float NdotV = cosThetaI;
            float NdotL = std::max(0.001f, n_eff.dot(specDir));
            float NdotH = std::max(0.001f, n_eff.dot(H));
            
            float k_smith = (roughness * roughness) / 2.0f;
            float G = (NdotV / (NdotV * (1.0f - k_smith) + k_smith)) * (NdotL / (NdotL * (1.0f - k_smith) + k_smith));
            
            W_spec = F_spec * G * VdotH / (NdotV * NdotH);
        }

        Eigen::Vector3f W_second = Eigen::Vector3f::Zero();
        PointType secondDir;
        PointType secondOrigin;

        float transmissionWeight = transmission * (1.0f - metallic);
        float diffuseWeight = (1.0f - transmission) * (1.0f - metallic);

        if (transmissionWeight > 0.0f) {
            float eta = isInside ? objMat.ior : (1.0f / objMat.ior);
            float k = 1.0f - eta * eta * (1.0f - VdotH * VdotH);
            
            if (k >= 0.0f) {
                secondDir = ((eta * VdotH - std::sqrt(k)) * H - eta * V).normalized();
                secondOrigin = hitPoint - n_eff * rayOffset;
                W_second = (Eigen::Vector3f::Constant(1.0f) - F_spec) * transmissionWeight;
                W_second = W_second.cwiseProduct(objColor);
            } else {
                Eigen::Vector3f tirWeight = (Eigen::Vector3f::Constant(1.0f) - F_spec) * transmissionWeight;
                W_spec += tirWeight.cwiseProduct(objColor);
            }
        } else if (diffuseWeight > 0.0f) {
            secondDir = sampleCosineHemisphere(n_eff, rngState);
            secondOrigin = hitPoint + n_eff * rayOffset;
            W_second = (Eigen::Vector3f::Constant(1.0f) - F_spec) * diffuseWeight;
            W_second = W_second.cwiseProduct(objColor);
        }

        W_spec = W_spec.cwiseMin(Eigen::Vector3f::Constant(4.0f));
        W_second = W_second.cwiseMin(Eigen::Vector3f::Constant(4.0f));

        float lumSpec = W_spec.maxCoeff();
        float lumSecond = W_second.maxCoeff();
        
        bool doSplit = (bounces <= 1);

        if (doSplit) {
            Eigen::Vector3f specColor = Eigen::Vector3f::Zero();
            if (lumSpec > 0.001f) {
                specColor = W_spec.cwiseProduct(traceRay(hitPoint + n_eff * rayOffset, specDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod));
            }
            
            Eigen::Vector3f secondColor = Eigen::Vector3f::Zero();
            if (lumSecond > 0.001f) {
                secondColor = W_second.cwiseProduct(traceRay(secondOrigin, secondDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod));
            }
            
            return finalColor + specColor + secondColor;
        } else {
            float totalLum = lumSpec + lumSecond;
            if (totalLum < 0.0001f) return finalColor;
            
            float pSpec = lumSpec / totalLum;
            float roll = float(rand_r(&rngState)) / float(RAND_MAX);
            
            if (roll < pSpec) {
                Eigen::Vector3f sample = traceRay(hitPoint + n_eff * rayOffset, specDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod);
                return finalColor + (W_spec / std::max(EPSILON, pSpec)).cwiseProduct(sample);
            } else {
                Eigen::Vector3f sample = traceRay(secondOrigin, secondDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod);
                return finalColor + (W_second / std::max(EPSILON, 1.0f - pSpec)).cwiseProduct(sample);
            }
        }
    }
    
    void clearNode(OctreeNode* node) {
        if (!node) return;
        
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                clearNode(node->children[i].get());
                node->children[i].reset(nullptr);
            }
        }
        
        node->isLeaf = true;
    }

    void printStatsRecursive(const OctreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes) const {
        if (!node) return;
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (node->lodData) lodGeneratedNodes++;
        
        size_t pts = node->points.size();
        actualPoints += pts;

        if (node->isLeaf) {
            leafNodes++;
            maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
            minPointsInLeaf = std::min(minPointsInLeaf, pts);
        } else {
            for (const auto& child : node->children) {
                printStatsRecursive(child.get(), depth + 1, totalNodes, leafNodes, actualPoints, 
                                    maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes);
            }
        }
    }

    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;

        if (node->isLeaf) {
            return;
        }

        bool childrenAreLeaves = true;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                optimizeRecursive(node->children[i].get());
                if (!node->children[i]->isLeaf) {
                    childrenAreLeaves = false;
                }
            }
        }

        if (childrenAreLeaves) {
            std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    allPoints.insert(allPoints.end(), node->children[i]->points.begin(), node->children[i]->points.end());
                }
            }

            std::sort(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                return a.get() < b.get();
            });
            allPoints.erase(std::unique(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                return a.get() == b.get();
            }), allPoints.end());

            if (allPoints.size() <= maxPointsPerNode) {
                node->points = std::move(allPoints);
                for (int i = 0; i < 8; ++i) {
                    node->children[i].reset(nullptr);
                }
                node->isLeaf = true;
                
                {
                    std::lock_guard<std::mutex> lock(node->lodMutex);
                    node->lodData = nullptr;
                }
            }
        }
    }

    template<typename V>
    inline void writeVal(std::ofstream& out, const V& val) const {
        out.write(reinterpret_cast<const char*>(&val), sizeof(V));
    }

    template<typename V>
    inline void readVal(std::ifstream& in, V& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(V));
    }

    inline void writeVec3(std::ofstream& out, const Eigen::Vector3f& vec) const {
        writeVal(out, vec.x());
        writeVal(out, vec.y());
        writeVal(out, vec.z());
    }

    inline void readVec3(std::ifstream& in, Eigen::Vector3f& vec) {
        float x, y, z;
        readVal(in, x); readVal(in, y); readVal(in, z);
        vec = Eigen::Vector3f(x, y, z);
    }

    void serializeNode(std::ofstream& out, const OctreeNode* node) const {
        writeVal(out, node->isLeaf);

        // ALWAYS serialize points, unconditionally
        size_t pointCount = node->points.size();
        writeVal(out, pointCount);
        for (const auto& pt : node->points) {
            writeVal(out, pt->data);
            writeVec3(out, pt->position);
            writeVal(out, pt->objectId);
            writeVal(out, pt->active);
            writeVal(out, pt->visible);
            writeVal(out, pt->size);
            writeVal(out, pt->colorIdx);
            writeVal(out, pt->materialIdx);
        }

        if (!node->isLeaf) {
            // Write bitmask of active children
            uint8_t childMask = 0;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i] != nullptr) {
                    childMask |= (1 << i);
                }
            }
            writeVal(out, childMask);

            // Recursively write only existing children
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    serializeNode(out, node->children[i].get());
                }
            }
        }
    }

    void deserializeNode(std::ifstream& in, OctreeNode* node) {
        bool isLeaf;
        readVal(in, isLeaf);
        node->isLeaf = isLeaf;
        node->lodData = nullptr;

        size_t pointCount;
        readVal(in, pointCount);
        node->points.reserve(pointCount);
        
        for (size_t i = 0; i < pointCount; ++i) {
            auto pt = std::make_shared<NodeData>();
            readVal(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            readVal(in, pt->active);
            readVal(in, pt->visible);
            readVal(in, pt->size);
            readVal(in, pt->colorIdx);
            readVal(in, pt->materialIdx);
            node->points.push_back(pt);
        }

        if (!isLeaf) {
            uint8_t childMask;
            readVal(in, childMask);
            
            PointType center = node->center;

            for (int i = 0; i < 8; ++i) {
                if ((childMask >> i) & 1) {
                    // Reconstruct bounds for child
                    PointType childMin, childMax;
                    for (int d = 0; d < Dim; ++d) {
                        bool high = (i >> d) & 1;
                        childMin[d] = high ? center[d] : node->bounds.first[d];
                        childMax[d] = high ? node->bounds.second[d] : center[d];
                    }
                    
                    node->children[i] = std::make_unique<OctreeNode>(childMin, childMax);
                    deserializeNode(in, node->children[i].get());
                } else {
                    node->children[i] = nullptr;
                }
            }
        }
    }

    void bitonic_sort_8(std::array<std::pair<int, float>, 8>& arr) const noexcept {
        auto a0 = arr[0], a1 = arr[1], a2 = arr[2], a3 = arr[3];
        auto a4 = arr[4], a5 = arr[5], a6 = arr[6], a7 = arr[7];
        
        if (a0.second > a1.second) std::swap(a0, a1);
        if (a2.second < a3.second) std::swap(a2, a3);
        if (a4.second > a5.second) std::swap(a4, a5);
        if (a6.second < a7.second) std::swap(a6, a7);
        
        if (a0.second > a2.second) std::swap(a0, a2);
        if (a1.second > a3.second) std::swap(a1, a3);
        if (a0.second > a1.second) std::swap(a0, a1);
        if (a2.second > a3.second) std::swap(a2, a3);
        
        if (a4.second < a6.second) std::swap(a4, a6);
        if (a5.second < a7.second) std::swap(a5, a7);
        if (a4.second < a5.second) std::swap(a4, a5);
        if (a6.second < a7.second) std::swap(a6, a7);
        
        if (a0.second > a4.second) std::swap(a0, a4);
        if (a1.second > a5.second) std::swap(a1, a5);
        if (a2.second > a6.second) std::swap(a2, a6);
        if (a3.second > a7.second) std::swap(a3, a7);
        
        if (a0.second > a2.second) std::swap(a0, a2);
        if (a1.second > a3.second) std::swap(a1, a3);
        if (a4.second > a6.second) std::swap(a4, a6);
        if (a5.second > a7.second) std::swap(a5, a7);
        
        if (a0.second > a1.second) std::swap(a0, a1);
        if (a2.second > a3.second) std::swap(a2, a3);
        if (a4.second > a5.second) std::swap(a4, a5);
        if (a6.second > a7.second) std::swap(a6, a7);
        
        arr[0] = a0; arr[1] = a1; arr[2] = a2; arr[3] = a3;
        arr[4] = a4; arr[5] = a5; arr[6] = a6; arr[7] = a7;
    }

    bool rayBoxIntersect(const Ray& ray, const BoundingBox& box, float& tMin, float& tMax) const {
        float tx1 = (box.first[0] - ray.origin[0]) * ray.invDir[0];
        float tx2 = (box.second[0] - ray.origin[0]) * ray.invDir[0];

        tMin = std::min(tx1, tx2);
        tMax = std::max(tx1, tx2);

        float ty1 = (box.first[1] - ray.origin[1]) * ray.invDir[1];
        float ty2 = (box.second[1] - ray.origin[1]) * ray.invDir[1];

        tMin = std::max(tMin, std::min(ty1, ty2));
        tMax = std::min(tMax, std::max(ty1, ty2));

        float tz1 = (box.first[2] - ray.origin[2]) * ray.invDir[2];
        float tz2 = (box.second[2] - ray.origin[2]) * ray.invDir[2];

        tMin = std::max(tMin, std::min(tz1, tz2));
        tMax = std::min(tMax, std::max(tz1, tz2));

        return tMax >= std::max(0.0f, tMin);
    }

    bool rayCubeIntersect(const Ray& ray, const NodeData* cube, float& t, PointType& normal, PointType& hitPoint) const {
        BoundingBox bounds = cube->getCubeBounds();
        
        float t0x = (bounds.first[0] - ray.origin[0]) * ray.invDir[0];
        float t1x = (bounds.second[0] - ray.origin[0]) * ray.invDir[0];
        if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

        float t0y = (bounds.first[1] - ray.origin[1]) * ray.invDir[1];
        float t1y = (bounds.second[1] - ray.origin[1]) * ray.invDir[1];
        if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

        float tMin = std::max(t0x, t0y);
        float tMax = std::min(t1x, t1y);

        float t0z = (bounds.first[2] - ray.origin[2]) * ray.invDir[2];
        float t1z = (bounds.second[2] - ray.origin[2]) * ray.invDir[2];
        if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

        tMin = std::max(tMin, t0z);
        tMax = std::min(tMax, t1z);

        if (tMax < std::max(0.0f, tMin) || tMax < 0.0f) {
            return false;
        }

        if (tMin < 0.0f) {
            t = tMax;
        } else {
            t = tMin;
        }
        
        hitPoint = ray.origin + ray.dir * t;
        
        PointType dMin = (hitPoint - bounds.first).cwiseAbs();
        PointType dMax = (hitPoint - bounds.second).cwiseAbs();
        
        float minDist = std::numeric_limits<float>::max();
        int minAxis = 0;
        float sign = 1.0f;

        for (int i = 0; i < Dim; ++i) {
            if (dMin[i] < minDist) {
                minDist = dMin[i];
                minAxis = i;
                sign = -1.0f;
            }
            if (dMax[i] < minDist) {
                minDist = dMax[i];
                minAxis = i;
                sign = 1.0f;
            }
        }
        
        normal = PointType::Zero();
        normal[minAxis] = sign;
        return true;
    }

    float randomValueNormalDistribution(uint32_t& state) {
        std::mt19937 gen(state);
        state = gen();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float θ = 2 * M_PI * dist(gen);
        float ρ = sqrt(-2 * log(dist(gen)));
        return ρ * cos(θ);
    }

    PointType randomInHemisphere(const PointType& normal, uint32_t& state) {
        float x = randomValueNormalDistribution(state);
        float y = randomValueNormalDistribution(state);
        float z = randomValueNormalDistribution(state);
        PointType randomDir(x, y, z);
        randomDir.normalize();
        
        if (randomDir.dot(normal) < 0.0f) {
            return -randomDir;
        }
        return randomDir;
    }

    void collectNodesByObjectIdRecursive(OctreeNode* node, int id, std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) const {
        if (!node) return;
        
        for (const auto& pt : node->points) {
            if (pt->active && (id == -1 || pt->objectId == id)) {
                if (seen.insert(pt).second) {
                    results.push_back(pt);
                }
            }
        }
        if (!node->isLeaf) {
            for (const auto& child : node->children) {
                if (child) {
                    collectNodesByObjectIdRecursive(child.get(), id, results, seen);
                }
            }
        }
    }

    void collectNodesByObjectId(OctreeNode* node, int id, std::vector<std::shared_ptr<NodeData>>& results) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        collectNodesByObjectIdRecursive(node, id, results, seen);
    }

    struct GridCell {
        std::array<PointType, 8> p;
        std::array<float, 8> val;
        std::array<PointType, 8> color;
    };

    std::pair<PointType, PointType> vertexInterpolate(float isolevel, const PointType& p1, const PointType& p2, float val1, float val2, const PointType& c1, const PointType& c2) {
        if (std::abs(isolevel - val1) < EPSILON) return {p1, c1};
        if (std::abs(isolevel - val2) < EPSILON) return {p2, c2};
        if (std::abs(val1 - val2) < EPSILON) return {p1, c1};
        float mu = (isolevel - val1) / (val2 - val1);
        PointType pos = p1 + mu * (p2 - p1);
        PointType color = c1 + mu * (c2 - c1);
        return {pos, color};
    }

    void polygonise(GridCell& grid, float isolevel, std::vector<PointType>& vertices, std::vector<Eigen::Vector3f>& colors, std::vector<Eigen::Vector3i>& triangles) {
        int cubeindex = 0;
        if (grid.val[0] < isolevel) cubeindex |= 1;
        if (grid.val[1] < isolevel) cubeindex |= 2;
        if (grid.val[2] < isolevel) cubeindex |= 4;
        if (grid.val[3] < isolevel) cubeindex |= 8;
        if (grid.val[4] < isolevel) cubeindex |= 16;
        if (grid.val[5] < isolevel) cubeindex |= 32;
        if (grid.val[6] < isolevel) cubeindex |= 64;
        if (grid.val[7] < isolevel) cubeindex |= 128;

        if (edgeTable[cubeindex] == 0) return;

        std::array<PointType, 12> vertlist_pos;
        std::array<PointType, 12> vertlist_color;

        auto interpolate = [&](int edge_idx, int p_idx1, int p_idx2) {
            auto [pos, col] = vertexInterpolate(isolevel, grid.p[p_idx1], grid.p[p_idx2], grid.val[p_idx1], grid.val[p_idx2], grid.color[p_idx1], grid.color[p_idx2]);
            vertlist_pos[edge_idx] = pos;
            vertlist_color[edge_idx] = col;
        };

        if (edgeTable[cubeindex] & 1)    interpolate(0, 0, 1);
        if (edgeTable[cubeindex] & 2)    interpolate(1, 1, 2);
        if (edgeTable[cubeindex] & 4)    interpolate(2, 2, 3);
        if (edgeTable[cubeindex] & 8)    interpolate(3, 3, 0);
        if (edgeTable[cubeindex] & 16)   interpolate(4, 4, 5);
        if (edgeTable[cubeindex] & 32)   interpolate(5, 5, 6);
        if (edgeTable[cubeindex] & 64)   interpolate(6, 6, 7);
        if (edgeTable[cubeindex] & 128)  interpolate(7, 7, 4);
        if (edgeTable[cubeindex] & 256)  interpolate(8, 0, 4);
        if (edgeTable[cubeindex] & 512)  interpolate(9, 1, 5);
        if (edgeTable[cubeindex] & 1024) interpolate(10, 2, 6);
        if (edgeTable[cubeindex] & 2048) interpolate(11, 3, 7);

        int currentVertCount = vertices.size();
        
        for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
            Eigen::Vector3i tri;
            for(int j=0; j<3; ++j) {
                int table_idx = triTable[cubeindex][i+j];
                tri[j] = currentVertCount + j; 
                vertices.push_back(vertlist_pos[table_idx]);
                colors.push_back(vertlist_color[table_idx]);
            }
            triangles.push_back(tri);
            currentVertCount += 3;
        }
    }

    std::pair<float, PointType> getScalarFieldValue(const PointType& pos, int objectId, float radius) {
        auto neighbors = findInRadius(pos, radius, objectId);
        float density = 0.0f;
        PointType accumulatedColor = PointType::Zero();
        float totalWeight = 0.0f;

        if (neighbors.empty()) {
            return {0.0f, {0.0f, 0.0f, 0.0f}};
        }

        for (auto& neighbor : neighbors) {
            float distSq = (neighbor->position - pos).squaredNorm();
            float rSq = radius * radius;
            if (distSq < rSq) {
                float x = distSq / rSq;
                float w = (1.0f - x) * (1.0f - x);
                density += w;
                accumulatedColor += getColor(neighbor->colorIdx) * w;
                totalWeight += w;
            }
        }

        if (totalWeight > 1e-6) {
            return {density, accumulatedColor / totalWeight};
        }
        
        return {0.0f, {0.0f, 0.0f, 0.0f}};
    }

    std::unique_ptr<Mesh> mergeMeshes(std::vector<Mesh*>& meshes, int objectId) {
        if (meshes.empty()) return nullptr;

        std::vector<Eigen::Vector3f> mergedVertices;
        std::vector<std::vector<int>> mergedPolys;
        std::map<Eigen::Vector3f, int, Vector3fCompare> vertexMap;
        
        for (auto& mesh_ptr : meshes) {
            if (!mesh_ptr) continue;
            Mesh& mesh = *mesh_ptr;

            auto mesh_verts = mesh.vertices();
            auto mesh_tris = mesh.polys();
            std::vector<int> index_map(mesh_verts.size());

            for (size_t i = 0; i < mesh_verts.size(); ++i) {
                auto it = vertexMap.find(mesh_verts[i]);
                if (it == vertexMap.end()) {
                    int new_idx = mergedVertices.size();
                    vertexMap[mesh_verts[i]] = new_idx;
                    mergedVertices.push_back(mesh_verts[i]);
                    index_map[i] = new_idx;
                } else {
                    index_map[i] = it->second;
                }
            }

            for (auto& poly : mesh_tris) {
                mergedPolys.push_back({index_map[poly[0]], index_map[poly[1]], index_map[poly[2]]});
            }
        }
        
        if (mergedVertices.empty()) return nullptr;
        
        return std::make_unique<Mesh>(objectId, mergedVertices, mergedPolys, std::vector<Eigen::Vector3f>{ {1.f, 1.f, 1.f} });
    }

public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16) :
    root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
    maxDepth(maxDepth), size(0), mapMutex_(std::make_unique<std::mutex>()) {}

    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), mapMutex_(std::make_unique<std::mutex>()) {}

    void setSkylight(const Eigen::Vector3f& skylight) { 
        skylight_ = skylight; 
    }

    Eigen::Vector3f getSkylight() const { 
        return skylight_; 
    }

    void setBackgroundColor(const Eigen::Vector3f& color) { 
        backgroundColor_ = color; 
    }

    Eigen::Vector3f getBackgroundColor() const { 
        return backgroundColor_; 
    }

    void setLODFalloff(float rate) { lodFalloffRate_ = rate; }
    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }
    void setMaxDistance(float dist) { maxDistance_ = dist; }

    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, int subId = 0, float emittance = 0.0f, float roughness = 1.0f, 
             float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f) {
        
        IndexType cIdx = getColorIndex(color);
        Material mat(emittance, roughness, metallic, transmission, ior);
        IndexType mIdx = getMaterialIndex(mat);
        
        auto pointData = std::make_shared<NodeData>(data, pos, visible, cIdx, size, active, objectId, subId, mIdx);
        
        ensureBounds(pointData->getCubeBounds());
        
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        }
        return false;
    }

    bool save(const std::string& filename) const {
        if (!root_) return false;
        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x79676733;
        writeVal(out, magic);
        writeVal(out, maxDepth);
        writeVal(out, maxPointsPerNode);
        writeVal(out, size);
        
        writeVec3(out, skylight_);
        writeVec3(out, backgroundColor_);

        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            size_t cMapSize = colorMap_.size();
            writeVal(out, cMapSize);
            for (const auto& c : colorMap_) {
                writeVec3(out, c);
            }

            size_t mMapSize = materialMap_.size();
            writeVal(out, mMapSize);
            for (const auto& m : materialMap_) {
                writeVal(out, m.emittance);
                writeVal(out, m.roughness);
                writeVal(out, m.metallic);
                writeVal(out, m.transmission);
                writeVal(out, m.ior);
            }
        }
        
        writeVec3(out, root_->bounds.first);
        writeVec3(out, root_->bounds.second);

        serializeNode(out, root_.get());
        
        out.close();
        std::cout << "successfully saved grid to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        uint32_t magic;
        readVal(in, magic);
        if (magic != 0x79676733) {
            std::cerr << "Invalid Octree file format" << std::endl;
            return false;
        }

        readVal(in, maxDepth);
        readVal(in, maxPointsPerNode);
        readVal(in, size);
        
        readVec3(in, skylight_);
        readVec3(in, backgroundColor_);

        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            colorMap_.clear();
            colorToIndex_.clear();
            materialMap_.clear();
            materialToIndex_.clear();

            size_t cMapSize;
            readVal(in, cMapSize);
            colorMap_.resize(cMapSize);
            for (size_t i = 0; i < cMapSize; ++i) {
                readVec3(in, colorMap_[i]);
                colorToIndex_[colorMap_[i]] = static_cast<IndexType>(i);
            }

            size_t mMapSize;
            readVal(in, mMapSize);
            materialMap_.resize(mMapSize);
            for (size_t i = 0; i < mMapSize; ++i) {
                readVal(in, materialMap_[i].emittance);
                readVal(in, materialMap_[i].roughness);
                readVal(in, materialMap_[i].metallic);
                readVal(in, materialMap_[i].transmission);
                readVal(in, materialMap_[i].ior);
                materialToIndex_[materialMap_[i]] = static_cast<IndexType>(i);
            }
        }

        PointType minBound, maxBound;
        readVec3(in, minBound);
        readVec3(in, maxBound);

        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        deserializeNode(in, root_.get());

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = EPSILON) {
        return findRecursive(root_.get(), pos, tolerance);
    }

    bool inGrid(PointType pos) {
        return root_->contains(pos);
    }

    bool remove(const PointType& pos, float tolerance = EPSILON) {
        auto pt = find(pos, tolerance);
        if (!pt) return false;
        if (removeRecursive(root_.get(), pt->getCubeBounds(), pt)) {
            size--;
            return true;
        }
        return false;
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius, int objectid = -1) const {
        std::vector<std::shared_ptr<NodeData>> results;
        if (!root_) return results;
        
        float radiusSq = radius * radius;
        searchNode(root_.get(), center, radiusSq, objectid, results);
        
        return results;
    }

    bool update(const PointType& pos, const T& newData) {
        auto pointData = find(pos);
        if (!pointData) return false;
        else pointData->data = newData;
        return true;
    }

    bool update(const PointType& oldPos, const PointType& newPos, const T& newData, bool newVisible = true, 
                Eigen::Vector3f newColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, float newEmittance = -1.0f, float newRoughness = -1.0f, 
                float newMetallic = -1.0f, float newTransmission = -1.0f, float newIor = -1.0f, float tolerance = EPSILON) {

        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;

        int oldSubId = pointData->subId;
        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        int finalSubId = oldSubId;
        
        if (oldSubId == 0 && targetObjId == pointData->objectId) {
            finalSubId = nextSubIdGenerator_++; 
            auto neighbors = findInRadius(oldPos, newSize * 3.0f, targetObjId);
            for(auto& n : neighbors) {
                if(n->subId == 0) {
                    n->subId = finalSubId;
                    invalidateMesh(targetObjId, 0);
                }
            }
        }
        
        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->visible = newVisible;
        
        if (newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f)) pointData->colorIdx = getColorIndex(newColor);
        if (newSize > 0) pointData->size = newSize;
        pointData->active = newActive;
        pointData->objectId = targetObjId;
        pointData->subId = finalSubId;
        
        Material mat = getMaterial(pointData->materialIdx);
        bool matChanged = false;
        if (newEmittance >= 0) {
            mat.emittance = newEmittance;
            matChanged = true;
        }
        if (newRoughness >= 0) { 
            mat.roughness = newRoughness;
            matChanged = true;
        }
        if (newMetallic >= 0) { 
            mat.metallic = newMetallic;
            matChanged = true;
        }
        if (newTransmission >= 0) { 
            mat.transmission = newTransmission;
            matChanged = true;
        }
        if (newIor >= 0) { 
            mat.ior = newIor;
            matChanged = true;
        }
        if (matChanged) pointData->materialIdx = getMaterialIndex(mat);
        
        ensureBounds(pointData->getCubeBounds());
        bool res = insertRecursive(root_.get(), pointData, 0);
        
        if(res) {
            invalidateMesh(targetObjId, finalSubId);
            if(finalSubId != oldSubId) invalidateMesh(targetObjId, oldSubId);
        } else {
            size--;
        }

        return res;
    }

    bool move(const PointType& pos, const PointType& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        pointData->position = newPos;
        ensureBounds(pointData->getCubeBounds());

        if (insertRecursive(root_.get(), pointData, 0)) {
            return true;
        }
        size--;
        return false;
    }

    bool setObjectId(const PointType& pos, int objectId, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->objectId = objectId;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool updateData(const PointType& pos, const T& newData, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setActive(const PointType& pos, bool active, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->active = active;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setVisible(const PointType& pos, bool visible, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->visible = visible;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setColor(const PointType& pos, Eigen::Vector3f color, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->colorIdx = getColorIndex(color);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setEmittance(const PointType& pos, float emittance, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        Material mat = getMaterial(pointData->materialIdx);
        mat.emittance = emittance;
        pointData->materialIdx = getMaterialIndex(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setRoughness(const PointType& pos, float roughness, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        Material mat = getMaterial(pointData->materialIdx);
        mat.roughness = roughness;
        pointData->materialIdx = getMaterialIndex(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setMetallic(const PointType& pos, float metallic, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        Material mat = getMaterial(pointData->materialIdx);
        mat.metallic = metallic;
        pointData->materialIdx = getMaterialIndex(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setTransmission(const PointType& pos, float transmission, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        Material mat = getMaterial(pointData->materialIdx);
        mat.transmission = transmission;
        pointData->materialIdx = getMaterialIndex(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    std::shared_ptr<NodeData> voxelTraverse(const PointType& origin, const PointType& direction,
                                        float maxDist, bool enableLOD = false) const {
        std::shared_ptr<NodeData> hit;
        float invLodf = 1.0f / lodFalloffRate_;
        Ray oray(origin, direction);
        
        float tMin, tMax;
        if (rayBoxIntersect(oray, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            voxelTraverseRecursive(root_.get(), tMin, tMax, currentMaxDist, enableLOD, oray, hit, invLodf);
        }
        return hit;
    }

    frame renderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 2,
                    int maxBounces = 4, bool globalIllumination = false, bool useLod = true) {
        generateLODs();
        PointType origin = cam.origin;
        PointType dir = cam.direction.normalized();
        PointType up = cam.up.normalized();
        PointType right = cam.right();
        
        frame outFrame(width, height, colorformat);
        std::vector<float> colorBuffer;
        int channels = 3;
        colorBuffer.resize(width * height * channels);

        float aspect = static_cast<float>(width) / height;
        float fovRad = cam.fovRad();
        float tanHalfFov = tan(fovRad * 0.5f);
        float tanfovy = tanHalfFov;
        float tanfovx = tanHalfFov * aspect;

        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pidx = (y * width + x);
                uint32_t seed = pidx * 1973 + 9277;
                int idx = pidx * channels;

                float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                
                PointType rayDir = dir + (right * px) + (up * py);
                rayDir.normalize();

                Eigen::Vector3f accumulatedColor(0.0f, 0.0f, 0.0f);
                
                for(int s = 0; s < samplesPerPixel; ++s) {
                    accumulatedColor += traceRay(origin, rayDir, 0, seed, maxBounces, globalIllumination, useLod);
                }
                
                Eigen::Vector3f color = accumulatedColor / static_cast<float>(samplesPerPixel);
                
                color = color.cwiseMax(0.0f).cwiseMin(1.0f);

                colorBuffer[idx    ] = color[0];
                colorBuffer[idx + 1] = color[1];
                colorBuffer[idx + 2] = color[2];
            }
        }
        
        outFrame.setData(colorBuffer, frame::colormap::RGB);
        return outFrame;
    }

    std::shared_ptr<NodeData> fastVoxelTraverse(const Ray& ray, float maxDist, bool enableLOD = false) const {
        std::shared_ptr<NodeData> hit;
        if (empty()) return hit;
        float lodRatio = 1.0f / lodFalloffRate_;
        float tMin, tMax;
        if (rayBoxIntersect(ray, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            voxelTraverseRecursive(root_.get(), tMin, tMax, currentMaxDist, enableLOD, ray, hit, lodRatio);
        }
        return hit;
    }

    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB) {
        //TIME_FUNCTION;
        generateLODs();
        PointType origin = cam.origin;
        PointType dir = cam.direction.normalized();
        PointType up = cam.up.normalized();
        PointType right = cam.right();
        
        frame outFrame(width, height, colorformat);
        std::vector<float> colorBuffer;
        colorBuffer.resize(width * height * 3);

        const float aspect = static_cast<float>(width) / height;
        const float fovRad = cam.fovRad();
        const float tanHalfFov = tan(fovRad * 0.5f);
        const float tanfovy = tanHalfFov;
        const float tanfovx = tanHalfFov * aspect;
        
        const PointType globalLightDir = (-cam.direction * 0.2f).normalized();
        const float fogStart = 1000.0f;
        const float minVisibility = 0.2f; 
        
        #pragma omp parallel for schedule(dynamic, 128) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pidx = (y * width + x);
                int idx = pidx * 3;

                float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                
                PointType rayDir = dir + (right * px) + (up * py);
                rayDir.normalize();

                Eigen::Vector3f color = backgroundColor_;
                Ray ray(origin, rayDir);
                auto hit = fastVoxelTraverse(ray, maxDistance_, true);
                if (hit != nullptr) {
                    auto obj = hit;
                    
                    float t = 0.0f;
                    PointType normal, hitPoint;

                    rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
                    color = getColor(obj->colorIdx);
                    Material objMat = getMaterial(obj->materialIdx);
                    
                    if (objMat.emittance > 0.0f) {
                        color = color * objMat.emittance;
                    } else {
                        float diffuse = std::max(0.0f, normal.dot(globalLightDir));
                        float ambient = 0.35f;
                        float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                        color = color * intensity;
                    }
                    
                    float fogFactor = std::clamp((maxDistance_ - t) / (maxDistance_ - fogStart), minVisibility, 1.0f);
                    
                    color = color * fogFactor + backgroundColor_ * (1.0f - fogFactor);
                }

                color = color.cwiseMax(0.0f).cwiseMin(1.0f);

                colorBuffer[idx    ] = color[0];
                colorBuffer[idx + 1] = color[1];
                colorBuffer[idx + 2] = color[2];
            }
        }
        
        outFrame.setData(colorBuffer, frame::colormap::RGB);
        return outFrame;
    }

    frame renderFrameTimed(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, 
                           double maxTimeSeconds = 0.16, int maxBounces = 4, bool globalIllumination = false, bool useLod = true) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        generateLODs();
        PointType origin = cam.origin;
        PointType dir = cam.direction.normalized();
        PointType up = cam.up.normalized();
        PointType right = cam.right();
        
        frame outFrame(width, height, colorformat);
        std::vector<float> colorBuffer(width * height * 3, 0.0f);
        std::vector<Eigen::Vector3f> accumColor(width * height, Eigen::Vector3f::Zero());
        std::vector<int> sampleCount(width * height, 0);

        const float aspect = static_cast<float>(width) / height;
        const float fovRad = cam.fovRad();
        const float tanHalfFov = std::tan(fovRad * 0.5f);
        const float tanfovy = tanHalfFov;
        const float tanfovx = tanHalfFov * aspect;
        
        const PointType globalLightDir = (-cam.direction * 0.2f).normalized();
        const float fogStart = 1000.0f;
        const float minVisibility = 0.2f; 
        
        #pragma omp parallel for schedule(dynamic, 128) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pidx = (y * width + x);

                float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                
                PointType rayDir = dir + (right * px) + (up * py);
                rayDir.normalize();

                Eigen::Vector3f color = backgroundColor_;
                Ray ray(origin, rayDir);
                auto hit = fastVoxelTraverse(ray, maxDistance_, true);
                if (hit != nullptr) {
                    auto obj = hit;
                    
                    float t = 0.0f;
                    PointType normal, hitPoint;

                    rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
                    color = getColor(obj->colorIdx);
                    Material objMat = getMaterial(obj->materialIdx);
                    
                    if (objMat.emittance > 0.0f) {
                        color = color * objMat.emittance;
                    } else {
                        float diffuse = std::max(0.0f, normal.dot(globalLightDir));
                        float ambient = 0.35f;
                        float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                        color = color * intensity;
                    }
                    
                    float fogFactor = std::clamp((maxDistance_ - t) / (maxDistance_ - fogStart), minVisibility, 1.0f);
                    color = color * fogFactor + backgroundColor_ * (1.0f - fogFactor);
                }
                
                accumColor[pidx] = color;
                sampleCount[pidx] = 1;
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - startTime;

        if (elapsed.count() < maxTimeSeconds) {
            std::atomic<bool> timeUp(false);
            std::atomic<uint64_t> counter(0);
            uint64_t totalPixels = static_cast<uint64_t>(width) * height;

            std::vector<uint64_t> pixelIndices(totalPixels);
            std::iota(pixelIndices.begin(), pixelIndices.end(), 0); 
            
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(pixelIndices.begin(), pixelIndices.end(), g);

            #pragma omp parallel
            {
                uint32_t localSeed = omp_get_thread_num() * 1973 + 9277;
                int chunkSize = 64;
                
                while (!timeUp.load(std::memory_order_relaxed)) {
                    uint64_t startIdx = counter.fetch_add(chunkSize, std::memory_order_relaxed);
                    
                    if (omp_get_thread_num() == 0) {
                        auto checkTime = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> checkElapsed = checkTime - startTime;
                        if (checkElapsed.count() >= maxTimeSeconds) {
                            timeUp.store(true, std::memory_order_relaxed);
                            break;
                        }
                    }

                    if (timeUp.load(std::memory_order_relaxed)) break;

                    for (int i = 0; i < chunkSize; ++i) {
                        uint64_t currentOffset = startIdx + i;
                        uint64_t pidx = pixelIndices[currentOffset % totalPixels];
                        
                        int y = pidx / width;
                        int x = pidx % width;

                        float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                        float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                        
                        PointType rayDir = dir + (right * px) + (up * py);
                        rayDir.normalize();

                        uint32_t pass = currentOffset / totalPixels;
                        uint32_t seed = pidx * 1973 + pass * 12345 + localSeed;

                        Eigen::Vector3f pbrColor = traceRay(origin, rayDir, 0, seed, maxBounces, globalIllumination, useLod);
                        
                        accumColor[pidx] += pbrColor;
                        sampleCount[pidx] += 1;
                    }
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int pidx = 0; pidx < width * height; ++pidx) {
            Eigen::Vector3f finalColor = accumColor[pidx];
            int count = sampleCount[pidx];
            
            if (count > 0) {
                finalColor /= static_cast<float>(count);
            }
            
            finalColor = finalColor.cwiseMax(0.0f).cwiseMin(1.0f);
            
            int idx = pidx * 3;
            colorBuffer[idx]     = finalColor[0];
            colorBuffer[idx + 1] = finalColor[1];
            colorBuffer[idx + 2] = finalColor[2];
        }

        outFrame.setData(colorBuffer, frame::colormap::RGB);
        return outFrame;
    }

    std::vector<std::shared_ptr<NodeData>> getExternalNodes(int targetObjectId) {
        std::vector<std::shared_ptr<NodeData>> candidates;
        std::vector<std::shared_ptr<NodeData>> surfaceNodes;
        
        collectNodesByObjectId(root_.get(), targetObjectId, candidates);

        if (candidates.empty()) return surfaceNodes;

        surfaceNodes.reserve(candidates.size());

        const std::array<PointType, 6> directions = {
            PointType(1, 0, 0), PointType(-1, 0, 0), PointType(0, 1, 0), 
            PointType(0, -1, 0), PointType(0, 0, 1), PointType(0, 0, -1)
        };

        for (const auto& node : candidates) {
            bool isExposed = false;
            float step = node->size; 
            for (const auto& dir : directions) {
                PointType probePos = node->position + (dir * step);
                auto neighbor = find(probePos, step * 0.25f);

                if (neighbor == nullptr || !neighbor->active || neighbor->objectId != node->objectId) {
                    isExposed = true;
                    break;
                }
            }

            if (isExposed) {
                surfaceNodes.push_back(node);
            }
        }
        surfaceNodes.shrink_to_fit();

        return surfaceNodes;
    }

    std::shared_ptr<Mesh> generateMesh(int objectId, float isolevel = 0.5f, int resolution = 32) {
        TIME_FUNCTION;
        if (!root_) return nullptr;

        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if(nodes.empty()) return nullptr;

        PointType minB = nodes[0]->position;
        PointType maxB = nodes[0]->position;
        float maxRadius = 0.0f;

        for(auto& n : nodes) {
            minB = minB.cwiseMin(n->position);
            maxB = maxB.cwiseMax(n->position);
            maxRadius = std::max(maxRadius, n->size);
        }
        
        float padding = maxRadius * 2.0f; 
        minB -= PointType::Constant(padding);
        maxB += PointType::Constant(padding);

        PointType size = maxB - minB;
        PointType step = size / static_cast<float>(resolution);

        std::vector<PointType> vertices;
        std::vector<Eigen::Vector3f> vertexColors;
        std::vector<Eigen::Vector3i> triangles;

        for(int z = 0; z < resolution; ++z) {
            for(int y = 0; y < resolution; ++y) {
                for(int x = 0; x < resolution; ++x) {
                    
                    PointType currPos(
                        minB.x() + x * step.x(),
                        minB.y() + y * step.y(),
                        minB.z() + z * step.z()
                    );

                    GridCell cell;
                    
                    PointType offsets[8] = {
                        {0,0,0}, {step.x(),0,0}, {step.x(),step.y(),0}, {0,step.y(),0},
                        {0,0,step.z()}, {step.x(),0,step.z()}, {step.x(),step.y(),step.z()}, {0,step.y(),step.z()}
                    };

                    bool hasInside = false;
                    bool hasOutside = false;
                    bool activeCell = false;
                    for(int i=0; i<8; ++i) {
                        cell.p[i] = currPos + offsets[i];
                        auto [density, color] = getScalarFieldValue(cell.p[i], objectId, maxRadius * 2.0f);
                        cell.val[i] = density;
                        cell.color[i] = color;
                        if(cell.val[i] >= isolevel) hasInside = true;
                        else hasOutside = true;
                    }

                    if(hasInside && hasOutside) {
                        polygonise(cell, isolevel, vertices, vertexColors, triangles);
                    }
                }
            }
        }

        if(vertices.empty()) return nullptr;

        std::vector<std::vector<int>> polys;
        for(auto& t : triangles) {
            polys.push_back({t[0], t[1], t[2]});
        }
        
        return std::make_shared<Mesh>(objectId, vertices, polys, vertexColors);
    }

    std::shared_ptr<Mesh> generateSubMesh(int objectId, int subId, float isolevel = 0.5f, int resolution = 32) {
        if (subId == -999) return nullptr;

        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesBySubId(root_.get(), objectId, subId, nodes);
        if (nodes.empty()) return nullptr;

        PointType minB = nodes[0]->position;
        PointType maxB = nodes[0]->position;
        float maxRadius = 0.0f;

        for(auto& n : nodes) {
            minB = minB.cwiseMin(n->position);
            maxB = maxB.cwiseMax(n->position);
            maxRadius = std::max(maxRadius, n->size);
        }
        
        float padding = maxRadius * 2.5f; 
        minB -= PointType::Constant(padding);
        maxB += PointType::Constant(padding);

        PointType size = maxB - minB;
        PointType step = size / static_cast<float>(resolution);

        std::vector<PointType> vertices;
        std::vector<Eigen::Vector3f> vertexColors;
        std::vector<Eigen::Vector3i> triangles;

        for(int z = 0; z < resolution; ++z) {
            for(int y = 0; y < resolution; ++y) {
                for(int x = 0; x < resolution; ++x) {
                    PointType currPos(minB.x() + x * step.x(), minB.y() + y * step.y(), minB.z() + z * step.z());

                    GridCell cell;
                    PointType offsets[8] = {{0,0,0}, {step.x(),0,0}, {step.x(),step.y(),0}, {0,step.y(),0},
                                          {0,0,step.z()}, {step.x(),0,step.z()}, {step.x(),step.y(),step.z()}, {0,step.y(),step.z()}};

                    bool activeCell = false;
                    for(int i=0; i<8; ++i) {
                        cell.p[i] = currPos + offsets[i];
                        auto [density, color] = getScalarFieldValue(cell.p[i], objectId, maxRadius * 2.0f);
                        cell.val[i] = density;
                        cell.color[i] = color;
                        if(cell.val[i] >= isolevel) activeCell = true;
                    }

                    if(activeCell) {
                        auto owner = find(currPos + step*0.5f, maxRadius * 2.0f);
                        if (owner && owner->objectId == objectId && owner->subId == subId) {
                            polygonise(cell, isolevel, vertices, vertexColors, triangles);
                        }
                    }
                }
            }
        }

        if(vertices.empty()) return nullptr;

        std::vector<std::vector<int>> polys;
        for(auto& t : triangles) polys.push_back({t[0], t[1], t[2]});
        
        return std::make_shared<Mesh>(objectId, vertices, polys, vertexColors, subId);
    }

    std::vector<std::shared_ptr<Mesh>> getMeshes(int objectId) {
        std::vector<std::shared_ptr<Mesh>> result;
        
        std::set<int> relevantSubIds;
        for(auto const& [key, val] : meshCache_) if(key.first == objectId) relevantSubIds.insert(key.second);
        for(auto const& key : dirtyMeshes_) if(key.first == objectId) relevantSubIds.insert(key.second);

        for(int subId : relevantSubIds) {
            if(dirtyMeshes_.count({objectId, subId})) {
                auto newMesh = generateSubMesh(objectId, subId);
                if(newMesh) meshCache_[{objectId, subId}] = newMesh;
                else meshCache_.erase({objectId, subId});
                dirtyMeshes_.erase({objectId, subId});
            }
        }

        for(auto const& [key, mesh] : meshCache_) {
            if(key.first == objectId && mesh) result.push_back(mesh);
        }
        return result;
    }

    void isolateInternalNodes(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes); 

        if(nodes.empty()) return;
        float checkRad = nodes[0]->size * 1.5f;

        for(auto& node : nodes) {
            int hiddenSides = 0;
            PointType dirs[6] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
            for(int i=0; i<6; ++i) {
                auto neighbor = find(node->position + dirs[i] * node->size, checkRad);
                if(neighbor && neighbor->objectId == objectId && neighbor->active) {
                    Material nMat = getMaterial(neighbor->materialIdx);
                    if (nMat.transmission < 0.01f) {
                        hiddenSides++;
                    }
                }
            }

            if(hiddenSides == 6) {
                if(node->subId != -999) {
                    invalidateMesh(objectId, node->subId);
                    node->subId = -999; 
                }
            }
        }
    }

    bool moveObject(int objectId, const PointType& offset) {
        if (!root_) return false;
        
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if(nodes.empty()) return false;

        for(auto& n : nodes) {
            if (removeRecursive(root_.get(), n->getCubeBounds(), n)) {
                size--;
            }
        }
        for(auto& n : nodes) {
            n->position += offset;
            ensureBounds(n->getCubeBounds());
            if (insertRecursive(root_.get(), n, 0)) {
                size++;
            }
        }

        for (auto& [key, mesh] : meshCache_) {
            if (key.first == objectId && mesh) {
                mesh->translate(offset); 
            }
        }
        return true;
    }

    void optimize() {
        if (root_) {
            optimizeRecursive(root_.get());
        }
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) {
            os << "[Octree Stats] Tree is null/empty." << std::endl;
            return;
        }

        size_t totalNodes = 0;
        size_t leafNodes = 0;
        size_t actualPoints = 0;
        size_t maxTreeDepth = 0;
        size_t maxPointsInLeaf = 0;
        size_t minPointsInLeaf = std::numeric_limits<size_t>::max();
        size_t lodGeneratedNodes = 0;

        printStatsRecursive(root_.get(), 0, totalNodes, leafNodes, actualPoints, 
                            maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes);

        if (leafNodes == 0) minPointsInLeaf = 0;
        double avgPointsPerLeaf = leafNodes > 0 ? (double)actualPoints / leafNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16); 
        size_t mapMem = colorMap_.size() * sizeof(Eigen::Vector3f) + materialMap_.size() * sizeof(Material);
        size_t maxSize = ((1 << (sizeof(IndexType)*8 - 2) - 1) * 2) + 1;

        os << "========================================\n";
        os << "             OCTREE STATS               \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
        os << "  LOD Falloff Rate  : " << lodFalloffRate_ << "\n";
        os << "  LOD Min Distance  : " << lodMinDistance_ << "\n";
        os << "Structure:\n";
        os << "  Total Nodes       : " << totalNodes << "\n";
        os << "  Leaf Nodes        : " << leafNodes << "\n";
        os << "  Non-Leaf Nodes    : " << (totalNodes - leafNodes) << "\n";
        os << "  LODs Generated    : " << lodGeneratedNodes << "\n";
        os << "  Tree Height       : " << maxTreeDepth << "\n";
        os << "Data:\n";
        os << "  Total Points      : " << size << " (Tracked) / " << actualPoints << " (Counted)\n";
        os << "  Points/Leaf (Avg) : " << std::fixed << std::setprecision(2) << avgPointsPerLeaf << "\n";
        os << "  Points/Leaf (Min) : " << minPointsInLeaf << "\n";
        os << "  Points/Leaf (Max) : " << maxPointsInLeaf << "\n";
        os << "Maps:\n";
        os << "  Unique Colors     : " << colorMap_.size() << "/" << maxSize << "\n";
        os << "  Unique Materials  : " << materialMap_.size() << "/" << maxSize << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "  Dictionary Maps   : " << (mapMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }

    void clear() {
        if (root_) {   
            clearNode(root_.get());
        }
        PointType minBound = root_->bounds.first;
        PointType maxBound = root_->bounds.second;
        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        
        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            colorMap_.clear();
            colorToIndex_.clear();
            materialMap_.clear();
            materialToIndex_.clear();
        }
        
        size = 0;
    }
};

#endif
