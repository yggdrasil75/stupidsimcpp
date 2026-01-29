#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
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

#ifdef SSE
#include <immintrin.h>
#endif

constexpr int Dim = 3;

template<typename T>
class Octree {
public:
    using PointType = Eigen::Matrix<float, Dim, 1>;
    using BoundingBox = std::pair<PointType, PointType>;
    
    enum class Shape {
        SPHERE,
        CUBE
    };
    
    struct NodeData {
        T data;
        PointType position;
        int objectId;
        bool active;
        bool visible;
        float size;
        Eigen::Vector3f color;
        bool light;
        float emittance;
        float refraction; 
        float reflection;
        Shape shape;

        NodeData(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f,
                 bool active = true, int objectId = -1, bool light = false, float emittance = 0.0f, 
                 float refraction = 0.0f, float reflection = 0.0f, Shape shape = Shape::SPHERE) 
                : data(data), position(pos), objectId(objectId), active(active), visible(visible), 
                color(color), size(size), light(light), emittance(emittance), refraction(refraction), 
                reflection(reflection), shape(shape) {}
        
        NodeData() : objectId(-1), active(false), visible(false), size(0.0f), light(false), 
                     emittance(0.0f), refraction(0.0f), reflection(0.0f), shape(Shape::SPHERE) {}
        
        // Helper method to get half-size for cube
        PointType getHalfSize() const {
            return PointType(size * 0.5f, size * 0.5f, size * 0.5f);
        }
        
        // Helper method to get bounding box for cube
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
        bool isLeaf;

        OctreeNode(const PointType& min, const PointType& max) : bounds(min,max), isLeaf(true) {
            for (std::unique_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5;
        }

        bool contains(const PointType& point) const {
            for (int i = 0; i < Dim; ++i) {
                if (point[i] < bounds.first[i] || point[i] > bounds.second[i]) {
                    return false;
                }
            }
            return true;
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

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        int octant = 0;
        for (int i = 0; i < Dim; ++i) {
            if (point[i] >= center[i]) octant |= (1 << i);
        }
        return octant;
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint8_t octant) const {
        PointType childMin, childMax;
        PointType center = node->center;
        for (int i = 0; i < Dim; ++i) {
            bool high = (octant >> i) & 1;
            childMin[i] = high ? center[i] : node->bounds.first[i];
            childMax[i] = high ? node->bounds.second[i] : center[i];
        }
        return {childMin, childMax};
    }

    void splitNode(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
        }

        for (const auto& pointData : node->points) {
            int octant = getOctant(pointData->position, node->center);
            node->children[octant]->points.emplace_back(pointData);
        }

        node->points.clear();
        node->isLeaf = false;

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        if (!node->contains(pointData->position)) return false;

        if (node->isLeaf) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            return true;
        } else {
            int octant = getOctant(pointData->position, node->center);
            if (node->children[octant]) {
                return insertRecursive(node->children[octant].get(), pointData, depth + 1);
            }
        }
        return false;
    }

    template<typename V>
    void writeVal(std::ofstream& out, const V& val) const {
        out.write(reinterpret_cast<const char*>(&val), sizeof(V));
    }

    template<typename V>
    void readVal(std::ifstream& in, V& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(V));
    }

    void writeVec3(std::ofstream& out, const Eigen::Vector3f& vec) const {
        writeVal(out, vec.x());
        writeVal(out, vec.y());
        writeVal(out, vec.z());
    }

    void readVec3(std::ifstream& in, Eigen::Vector3f& vec) {
        float x, y, z;
        readVal(in, x); readVal(in, y); readVal(in, z);
        vec = Eigen::Vector3f(x, y, z);
    }

    void serializeNode(std::ofstream& out, const OctreeNode* node) const {
        writeVal(out, node->isLeaf);

        if (node->isLeaf) {
            size_t pointCount = node->points.size();
            writeVal(out, pointCount);
            for (const auto& pt : node->points) {
                // Write raw data T (Must be POD)
                writeVal(out, pt->data);
                // Write properties
                writeVec3(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->active);
                writeVal(out, pt->visible);
                writeVal(out, pt->size);
                writeVec3(out, pt->color);
                writeVal(out, pt->light);
                writeVal(out, pt->emittance);
                writeVal(out, pt->refraction);
                writeVal(out, pt->reflection);
                writeVal(out, static_cast<int>(pt->shape));
            }
        } else {
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

        if (isLeaf) {
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
                readVec3(in, pt->color);
                readVal(in, pt->light);
                readVal(in, pt->emittance);
                readVal(in, pt->refraction);
                readVal(in, pt->reflection);
                int shapeInt;
                readVal(in, shapeInt);
                pt->shape = static_cast<Shape>(shapeInt);
                node->points.push_back(pt);
            }
        } else {
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

    bool rayBoxIntersect(const PointType& origin, const PointType& dir, const BoundingBox& box,
                            float& tMin, float& tMax) const {
        tMin = 0.0f;
        tMax = std::numeric_limits<float>::max();

        for (int i = 0; i < Dim; ++i) {
            if (std::abs(dir[i]) < std::numeric_limits<float>::epsilon()) {
                if (origin[i] < box.first[i] || origin[i] > box.second[i]) return false;
            } else {
                float invD = 1.f / dir[i];
                float t1 = (box.first[i] - origin[i]) * invD;
                float t2 = (box.second[i] - origin[i]) * invD;
                
                if (t1 > t2) std::swap(t1, t2);
                tMin = std::max(tMin, t1);
                tMax = std::min(tMax, t2);
                if (tMin > tMax) return false;
            }
        }
        return true;
    }

    bool raySphereIntersect(const PointType& origin, const PointType& dir, const PointType& center,
                           float radius, float& t) const {
        PointType oc = origin - center;
        float a = dir.dot(dir);
        float b = 2.0f * oc.dot(dir);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) return false;
        
        float sqrtDisc = sqrt(discriminant);
        float t0 = (-b - sqrtDisc) / (2.0f * a);
        float t1 = (-b + sqrtDisc) / (2.0f * a);
        
        t = t0;
        if (t0 < 0.001f) {
            t = t1;
            if (t1 < 0.001f) return false;
        }
        return true;
    }

    // Ray-cube intersection
    bool rayCubeIntersect(const PointType& origin, const PointType& dir, const NodeData* cube,
                         float& t, PointType& normal, PointType& hitPoint) const {
        // Use the cube's bounds for intersection
        BoundingBox bounds = cube->getCubeBounds();
        
        float tMin, tMax;
        if (!rayBoxIntersect(origin, dir, bounds, tMin, tMax)) {
            return false;
        }
        
        if (tMin < 0.001f) {
            if (tMax < 0.001f) return false;
            t = tMax;
        } else {
            t = tMin;
        }
        
        hitPoint = origin + dir * t;
        
        const float epsilon = 0.0001f;
        normal = PointType::Zero();
        
        for (int i = 0; i < Dim; ++i) {
            if (std::abs(hitPoint[i] - bounds.first[i]) < epsilon) {
                normal[i] = -1.0f;
            } else if (std::abs(hitPoint[i] - bounds.second[i]) < epsilon) {
                normal[i] = 1.0f;
            }
        }
        
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
            randomDir = -randomDir;
        }
        return randomDir;
    }

    float rgbToGrayscale(const Eigen::Vector3f& color) const {
        return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
    }

public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=16, size_t maxDepth = 16) :
        root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
        maxDepth(maxDepth), size(0) {}
    
    Octree() : root_(nullptr), maxPointsPerNode(16), maxDepth(16), size(0) {}

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size, bool active,
             int objectId = -1, bool light = false, float emittance = 0.0f, float refraction = 0.0f, 
             float reflection = 0.0f, Shape shape = Shape::SPHERE) {
        auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId,
                                                    light, emittance, refraction, reflection, shape);
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

        PointType minBound, maxBound;
        readVec3(in, minBound);
        readVec3(in, maxBound);

        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        deserializeNode(in, root_.get());

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = 0.0001f) {
        std::function<std::shared_ptr<NodeData>(OctreeNode*)> searchNode = [&](OctreeNode* node) -> std::shared_ptr<NodeData> {
            if (!node->contains(pos)) return nullptr;
            
            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    if (!pointData->active) continue;
                    
                    float distSq = (pointData->position - pos).squaredNorm();
                    if (distSq <= tolerance * tolerance) {
                        return pointData;
                    }
                }
                return nullptr;
            } else {
                int octant = getOctant(pos, node->center);
                if (node->children[octant]) {
                    return searchNode(node->children[octant].get());
                }
            }
            return nullptr;
        };
        
        return searchNode(root_.get());
    }

    bool remove(const PointType& pos, float tolerance = 0.0001f) {
        bool found = false;
        std::function<bool(OctreeNode*)> removeNode = [&](OctreeNode* node) -> bool {
            if (!node->contains(pos)) return false;
            
            if (node->isLeaf) {
                auto it = std::remove_if(node->points.begin(), node->points.end(),
                    [&](const std::shared_ptr<NodeData>& pointData) {
                        if (!pointData->active) return false;
                        
                        float distSq = (pointData->position - pos).squaredNorm();
                        if (distSq <= tolerance * tolerance) {
                            found = true;
                            return true;
                        }
                        return false;
                    });
                
                if (found) {
                    node->points.erase(it, node->points.end());
                    size--;
                    return true;
                }
                return false;
            } else {
                int octant = getOctant(pos, node->center);
                if (node->children[octant]) {
                    return removeNode(node->children[octant].get());
                }
            }
            return false;
        };
        
        return removeNode(root_.get());
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius) const {
        std::vector<std::shared_ptr<NodeData>> results;
        if (!root_) return results;
        
        float radiusSq = radius * radius;
        
        std::function<void(OctreeNode*)> searchNode = [&](OctreeNode* node) {
            // Check if node's bounding box intersects with search sphere
            PointType boxCenter = (node->bounds.first + node->bounds.second) * 0.5f;
            PointType boxHalfSize = (node->bounds.second - node->bounds.first) * 0.5f;
            
            PointType closestPoint;
            for (int i = 0; i < Dim; ++i) {
                closestPoint[i] = std::max(node->bounds.first[i], 
                                        std::min(center[i], node->bounds.second[i]));
            }
            
            float distSq = (closestPoint - center).squaredNorm();
            if (distSq > radiusSq) {
                return;
            }
            
            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    if (!pointData->active) continue;
                    
                    float pointDistSq = (pointData->position - center).squaredNorm();
                    if (pointDistSq <= radiusSq) {
                        results.emplace_back(pointData);
                    }
                }
            } else {
                for (const auto& child : node->children) {
                    if (child) {
                        searchNode(child.get());
                    }
                }
            }
        };
        
        searchNode(root_.get());
        return results;
    }

    bool update(const PointType& oldPos, const PointType& newPos, const T& newData = T(), bool newVisible = true, 
                Eigen::Vector3f newColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, bool newLight = false, float newEmittance = 0.0f, float newRefraction = 0.0f, 
                float newReflection = 0.0f, Shape newShape = Shape::SPHERE, float tolerance = 0.0001f) {
    
        // Find the existing point
        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;
        
        // If position changed, we need to remove and reinsert
        float moveDistSq = (newPos - oldPos).squaredNorm();
        if (moveDistSq > tolerance * tolerance) {
            // Save the data
            T dataCopy = pointData->data;
            bool activeCopy = pointData->active;
            bool visibleCopy = pointData->visible;
            Eigen::Vector3f colorCopy = pointData->color;
            float sizeCopy = pointData->size;
            int objectIdCopy = pointData->objectId;
            bool lightCopy = pointData->light;
            float emittanceCopy = pointData->emittance;
            float refractionCopy = pointData->refraction;
            float reflectionCopy = pointData->reflection;
            Shape shapeCopy = pointData->shape;
            
            // Remove the old point
            if (!remove(oldPos, tolerance)) {
                return false;
            }
            
            // Insert at new position with updated properties
            return set(newData != T() ? newData : dataCopy, newPos, 
                    newVisible ? newVisible : visibleCopy,
                    newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f) ? newColor : colorCopy,
                    newSize > 0 ? newSize : sizeCopy,
                    newActive ? newActive : activeCopy,
                    newObjectId != -2 ? newObjectId : objectIdCopy,
                    newLight ? newLight : lightCopy,
                    newEmittance > 0 ? newEmittance : emittanceCopy,
                    newRefraction >= 0 ? newRefraction : refractionCopy,
                    newReflection >= 0 ? newReflection : reflectionCopy,
                    newShape);
        } else {
            // Just update properties in place
            pointData->data = newData;
            pointData->position = newPos;  // Minor adjustment within tolerance
            pointData->visible = newVisible;
            pointData->color = newColor;
            pointData->size = newSize;
            if (newObjectId != -2) pointData->objectId = newObjectId;
            pointData->active = newActive;
            pointData->light = newLight;
            pointData->emittance = newEmittance;
            pointData->refraction = newRefraction;
            pointData->reflection = newReflection;
            pointData->shape = newShape;
            return true;
        }
    }

    bool setObjectId(const PointType& pos, int objectId, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->objectId = objectId;
        return true;
    }

    bool updateData(const PointType& pos, const T& newData, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->data = newData;
        return true;
    }

    bool setActive(const PointType& pos, bool active, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->active = active;
        return true;
    }

    bool setVisible(const PointType& pos, bool visible, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->visible = visible;
        return true;
    }

    bool setLight(const PointType& pos, bool light, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->light = light;
        return true;
    }

    bool setColor(const PointType& pos, Eigen::Vector3f color, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->color = color;
        return true;
    }

    bool setReflection(const PointType& pos, float reflection, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->reflection = reflection;
        return true;
    }

    bool setEmittance(const PointType& pos, float emittance, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->emittance = emittance;
        return true;
    }

    bool setShape(const PointType& pos, Shape shape, float tolerance = 0.0001f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        
        pointData->shape = shape;
        return true;
    }

    std::vector<std::shared_ptr<NodeData>> voxelTraverse(const PointType& origin, const PointType& direction,
                                         float maxDist, bool stopAtFirstHit) const {
        std::vector<std::shared_ptr<NodeData>> hits;
        
        if (empty()) return hits;

        std::function<void(OctreeNode*, const PointType&, const PointType&, float, float)> traverseNode = 
                        [&](OctreeNode* node, const PointType& origin, const PointType& dir, float tMin, float tMax) {
            if (!node || tMin > tMax) return;
            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    if (!pointData->active) continue;
                    
                    if (pointData->shape == Shape::SPHERE) {
                        PointType center = pointData->position;
                        float radius = pointData->size;
                        float t;
                        
                        if (raySphereIntersect(origin, dir, center, radius, t)) {
                            if (t >= 0 && t <= maxDist) {
                                hits.emplace_back(pointData);
                                if (stopAtFirstHit) return;
                            }
                        }
                    } else {
                        float t;
                        PointType normal, hitPoint;
                        if (rayCubeIntersect(origin, dir, pointData.get(), t, normal, hitPoint)) {
                            if (t >= 0 && t <= maxDist) {
                                hits.emplace_back(pointData);
                                if (stopAtFirstHit) return;
                            }
                        }
                    }
                }
            } else {
                std::array<std::pair<int, float>, 8> childOrder;
                PointType center = node->center;
                for (int i = 0; i < 8; ++i) {
                    if (node->children[i]) {
                        PointType childCenter = node->children[i]->center;
                        float dist = (childCenter - origin).dot(dir);
                        childOrder[i] = {i, dist};
                    } else {
                        childOrder[i] = {i, std::numeric_limits<float>::lowest()};
                    }
                }

                bitonic_sort_8(childOrder);

                for (int i = 0; i < 8; ++i) {
                    int childIdx = childOrder[i].first;
                    if (node->children[childIdx]) {
                        const auto& childBounds = node->children[childIdx]->bounds;
                        
                        float childtMin = tMin;
                        float childtMax = tMax;
                        if (rayBoxIntersect(origin, dir, childBounds, childtMin, childtMax)) {
                            traverseNode(node->children[childIdx].get(), origin, dir, childtMin, childtMax);
                            if (stopAtFirstHit && !hits.empty()) return;
                        }
                    }
                }
            }
        };
        float tMin, tMax;
        if (rayBoxIntersect(origin, direction, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            if (tMin <= tMax) {
                traverseNode(root_.get(), origin, direction, tMin, tMax);
            }
        }
        return hits;
    }
    
    frame renderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 2,
                      int maxBounces = 4, bool globalIllumination = false) {
        PointType origin = cam.origin;
        PointType dir = cam.direction.normalized();
        PointType up = cam.up.normalized();
        PointType right = cam.right();
        
        frame outFrame(width, height, colorformat);
        std::vector<uint8_t> colorBuffer;
        int channels;
        if (colorformat == frame::colormap::B) {
            channels = 1;
        } else if (colorformat == frame::colormap::RGB || colorformat == frame::colormap::BGR) {
            channels = 3;
        } else { //BGRA and RGBA
            channels = 4;
        }
        colorBuffer.resize(width * height * channels);

        float aspect = static_cast<float>(width) / height;
        float fovRad = cam.fovRad();
        float tanHalfFov = tan(fovRad * 0.5f);
        float tanfovy = tanHalfFov;
        float tanfovx = tanHalfFov * aspect;
        PointType space(0,0,0);
        if (globalIllumination) space = {0.1f, 0.1f, 0.1f};

        const Eigen::Vector3f defaultColor(0.01f, 0.01f, 0.01f); 
        float rayLength = std::numeric_limits<float>::max();
        std::function<Eigen::Vector3f(const PointType&, const PointType&, int, uint32_t&)> traceRay = 
            [&](const PointType& rayOrig, const PointType& rayDir, int bounces, uint32_t& rngState) -> Eigen::Vector3f {

            if (bounces > maxBounces) return space;

            auto hits = voxelTraverse(rayOrig, rayDir, rayLength, true);
            if (hits.empty() && bounces == 0) {
                return defaultColor; 
            } else if (hits.empty()) {
                return space;
            }

            auto obj = hits[0];
            
            PointType hitPoint;
            PointType normal;
            float t = 0.0f;
            
            // Calculate intersection based on shape
            if (obj->shape == Shape::SPHERE) {
                // Sphere intersection
                PointType center = obj->position;
                float radius = obj->size;
                PointType L_vec = center - rayOrig;
                float tca = L_vec.dot(rayDir);
                float d2 = L_vec.dot(L_vec) - tca * tca;
                float radius2 = radius * radius;

                if (d2 <= radius2) {
                    float thc = std::sqrt(radius2 - d2);
                    t = tca - thc; 
                    if (t < 0.001f) t = tca + thc;
                }
                
                hitPoint = rayOrig + rayDir * t;
                normal = (hitPoint - center).normalized();
            } else {
                // Cube intersection
                PointType cubeNormal;
                if (!rayCubeIntersect(rayOrig, rayDir, obj.get(), t, normal, hitPoint)) {
                    return space;
                }
            }

            Eigen::Vector3f finalColor = space;

            if (obj->light) {
                return obj->color * obj->emittance;
            }

            float refl = obj->reflection;
            float refr = obj->refraction;
            float diffuseProb = 1.0f - refl - refr;

            if (diffuseProb > 0.001f) {
                PointType scatterDir = randomInHemisphere(normal, rngState);
                
                Eigen::Vector3f incomingLight = traceRay(hitPoint + normal * 0.002f, scatterDir, bounces + 1, rngState);
                
                finalColor += obj->color.cwiseProduct(incomingLight) * diffuseProb;
            }

            if (refl > 0.001f) {
                PointType rDir = (rayDir - 2.0f * rayDir.dot(normal) * normal).normalized();
                finalColor += traceRay(hitPoint + normal * 0.002f, rDir, bounces + 1, rngState) * refl;
            }

            if (refr > 0.001f) {
                float ior = 1.45f;
                float η = 1.0f / ior;
                float cosI = -normal.dot(rayDir);
                PointType n_eff = normal;

                if (cosI < 0) {
                    cosI = -cosI;
                    n_eff = -normal;
                    η = ior;
                }

                float k = 1.0f - η * η * (1.0f - cosI * cosI);
                PointType nextDir;
                if (k >= 0.0f) {
                    nextDir = (η * rayDir + (η * cosI - std::sqrt(k)) * n_eff).normalized();
                    finalColor += traceRay(hitPoint - n_eff * 0.002f, nextDir, bounces + 1, rngState) * refr;
                } else {
                    nextDir = (rayDir - 2.0f * rayDir.dot(n_eff) * n_eff).normalized();
                    finalColor += traceRay(hitPoint + n_eff * 0.002f, nextDir, bounces + 1, rngState) * refr;
                }
            }

            return finalColor;
        };
        
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
                    accumulatedColor += traceRay(origin, rayDir, 0, seed);
                }
                
                Eigen::Vector3f color = accumulatedColor / static_cast<float>(samplesPerPixel);
                
                color = color.cwiseMax(0.0f).cwiseMin(1.0f);
                
                switch(colorformat) {
                    case frame::colormap::B:
                        colorBuffer[idx    ] = static_cast<uint8_t>(rgbToGrayscale(color) * 255.0f);
                        break;
                    case frame::colormap::RGB:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[2] * 255.0f);
                        break;
                    case frame::colormap::BGR:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[0] * 255.0f);
                        break;
                    case frame::colormap::RGBA:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 3] = 255;
                        break;
                    case frame::colormap::BGRA:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 3] = 255;
                        break;
                }
            }
        }
        
        outFrame.setData(colorBuffer);
        return outFrame;
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
        
        std::function<void(const OctreeNode*, size_t)> traverse = 
            [&](const OctreeNode* node, size_t depth) {
            if (!node) return;
            
            totalNodes++;
            maxTreeDepth = std::max(maxTreeDepth, depth);

            if (node->isLeaf) {
                leafNodes++;
                size_t pts = node->points.size();
                actualPoints += pts;
                maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
                minPointsInLeaf = std::min(minPointsInLeaf, pts);
            } else {
                for (const auto& child : node->children) {
                    traverse(child.get(), depth + 1);
                }
            }
        };

        traverse(root_.get(), 0);

        if (leafNodes == 0) minPointsInLeaf = 0;
        double avgPointsPerLeaf = leafNodes > 0 ? (double)actualPoints / leafNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16); 

        os << "========================================\n";
        os << "             OCTREE STATS               \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
        os << "Structure:\n";
        os << "  Total Nodes       : " << totalNodes << "\n";
        os << "  Leaf Nodes        : " << leafNodes << "\n";
        os << "  Non-Leaf Nodes    : " << (totalNodes - leafNodes) << "\n";
        os << "  Tree Height       : " << maxTreeDepth << "\n";
        os << "Data:\n";
        os << "  Total Points      : " << size << " (Tracked) / " << actualPoints << " (Counted)\n";
        os << "  Points/Leaf (Avg) : " << std::fixed << std::setprecision(2) << avgPointsPerLeaf << "\n";
        os << "  Points/Leaf (Min) : " << minPointsInLeaf << "\n";
        os << "  Points/Leaf (Max) : " << maxPointsInLeaf << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }

    void clear() {
        if (!root_) return;
        
        std::function<void(OctreeNode*)> clearNode = [&](OctreeNode* node) {
            if (!node) return;
            
            node->points.clear();
            node->points.shrink_to_fit();
            
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    clearNode(node->children[i].get());
                    node->children[i].reset(nullptr);
                }
            }
            
            node->isLeaf = true;
        };
        
        clearNode(root_.get());
        
        PointType minBound = root_->bounds.first;
        PointType maxBound = root_->bounds.second;
        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        
        size = 0;
    }
};

#endif
