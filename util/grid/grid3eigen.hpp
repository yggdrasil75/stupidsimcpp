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
#include <sstream>
#include <string>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <cstdint>

#ifdef SSE
#include <immintrin.h>
#endif

constexpr int Dim = 3;
// Flags for NodeData
static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;

// Flags for OctreeNode
static constexpr uint8_t NODE_LOADED_BIT = 1 << 0;
static constexpr uint8_t NODE_DIRTY_BIT = 1 << 1;
static constexpr uint8_t NODE_LEAF_BIT = 1 << 2;
static constexpr uint8_t NODE_LOD_VALID_BIT = 1 << 3;

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
        float size;
        IndexType colorIdx;
        IndexType materialIdx;
        std::atomic<uint8_t> flags;

        NodeData(const T& data, const PointType& pos, bool visible, IndexType colorIdx, float size = 0.01f,
                 bool active = true, int objectId = -1, IndexType materialIdx = 0) 
                : data(data), position(pos), objectId(objectId), size(size), 
                  colorIdx(colorIdx), materialIdx(materialIdx), flags(0) {
                    uint8_t f = 0;
                    if (active) f |= ACTIVE_BIT;
                    if (visible) f |= VISIBLE_BIT;
                    flags.store(f, std::memory_order_release);
                  }
        
        NodeData() : objectId(-1), size(0.0f), colorIdx(0), materialIdx(0), flags(0) {}

        NodeData(const NodeData& other) : data(other.data), position(other.position),
                objectId(other.objectId), size(other.size), colorIdx(other.colorIdx),
                materialIdx(other.materialIdx), flags(other.flags.load(std::memory_order_acquire)) {}

        NodeData& operator=(const NodeData& other) {
            if (this != &other) {
                data = other.data;
                position = other.position;
                objectId = other.objectId;
                size = other.size;
                colorIdx = other.colorIdx;
                materialIdx = other.materialIdx;
                flags.store(other.flags.load(std::memory_order_acquire), std::memory_order_release);
            }
            return *this;
        }

        NodeData(NodeData&& other) noexcept : data(std::move(other.data)), position(std::move(other.position)),
                objectId(other.objectId), size(other.size), colorIdx(other.colorIdx),
                materialIdx(other.materialIdx), flags(other.flags.load(std::memory_order_acquire)) {}

        NodeData& operator=(NodeData&& other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                position = std::move(other.position);
                objectId = other.objectId;
                size = other.size;
                colorIdx = other.colorIdx;
                materialIdx = other.materialIdx;
                flags.store(other.flags.load(std::memory_order_acquire), std::memory_order_release);
            }
            return *this;
        }
        
        bool isActive() const {
            return flags.load(std::memory_order_acquire) & ACTIVE_BIT;
        }
        bool isVisible() const {
            return flags.load(std::memory_order_acquire) & VISIBLE_BIT;
        }
        void setActive(bool val) {
            if (val) flags.fetch_or(ACTIVE_BIT, std::memory_order_release);
            else flags.fetch_and(~ACTIVE_BIT, std::memory_order_release);
        }
        void setVisible(bool val) {
            if (val) flags.fetch_or(VISIBLE_BIT, std::memory_order_release);
            else flags.fetch_and(~VISIBLE_BIT, std::memory_order_release);
        }
        
        PointType getHalfSize() const {
            return PointType(size * 0.5f, size * 0.5f, size * 0.5f);
        }
        
        BoundingBox getCubeBounds() const {
            PointType halfSize = getHalfSize();
            return {position - halfSize, position + halfSize};
        }

        void serialize(std::ostream& os) const {
            os.write(reinterpret_cast<const char*>(&data), sizeof(T));
            os.write(reinterpret_cast<const char*>(&position), sizeof(PointType));
            os.write(reinterpret_cast<const char*>(&objectId), sizeof(int));
            os.write(reinterpret_cast<const char*>(&size), sizeof(float));
            os.write(reinterpret_cast<const char*>(&colorIdx), sizeof(IndexType));
            os.write(reinterpret_cast<const char*>(&materialIdx), sizeof(IndexType));
            uint8_t f = flags.load(std::memory_order_acquire);
            os.write(reinterpret_cast<const char*>(&f), sizeof(uint8_t));
        }

        void deserialize(std::istream& is) {
            is.read(reinterpret_cast<char*>(&data), sizeof(T));
            is.read(reinterpret_cast<char*>(&position), sizeof(PointType));
            is.read(reinterpret_cast<char*>(&objectId), sizeof(int));
            is.read(reinterpret_cast<char*>(&size), sizeof(float));
            is.read(reinterpret_cast<char*>(&colorIdx), sizeof(IndexType));
            is.read(reinterpret_cast<char*>(&materialIdx), sizeof(IndexType));
            uint8_t f = 0;
            is.read(reinterpret_cast<char*>(&f), sizeof(uint8_t));
            flags.store(f, std::memory_order_release);
        }
    };

    struct Frustum {
        struct Plane {
            PointType normal;
            float d;
            bool intersects(const BoundingBox& box) const {
                PointType p = box.first;
                if (normal.x() >= 0) p.x() = box.second.x();
                if (normal.y() >= 0) p.y() = box.second.y();
                if (normal.z() >= 0) p.z() = box.second.z();
                return normal.dot(p) + d >= 0;
            }
        };

        Plane planes[6];

        Frustum(const Camera& cam, int width, int height, float maxDist) {
            float aspect = static_cast<float>(width) / height;
            float fovRad = cam.fovRad();
            float tanfovy = std::tan(fovRad * 0.5f);
            float tanfovx = tanfovy * aspect;

            PointType dir = cam.direction.normalized();
            PointType right = cam.right().normalized();
            PointType up = cam.up.normalized();

            planes[0] = {dir, -dir.dot(cam.origin)}; // Near
            planes[1] = {-dir, -(-dir).dot(cam.origin + dir * maxDist)}; // Far

            PointType n_left = up.cross(dir - right * tanfovx).normalized();
            planes[2] = {n_left, -n_left.dot(cam.origin)};

            PointType n_right = (dir + right * tanfovx).cross(up).normalized();
            planes[3] = {n_right, -n_right.dot(cam.origin)};

            PointType n_bottom = (dir - up * tanfovy).cross(right).normalized();
            planes[4] = {n_bottom, -n_bottom.dot(cam.origin)};

            PointType n_top = right.cross(dir + up * tanfovy).normalized();
            planes[5] = {n_top, -n_top.dot(cam.origin)};
        }

        bool intersects(const BoundingBox& box) const {
            for (int i = 0; i < 6; ++i) {
                if (!planes[i].intersects(box)) return false;
            }
            return true;
        }

        float distanceToAABB(const BoundingBox& box, const PointType& point) const {
            float distSq = 0.0f;
            for (int i = 0; i < Dim; ++i) {
                if (point[i] < box.first[i]) {
                    distSq += (box.first[i] - point[i]) * (box.first[i] - point[i]);
                } else if (point[i] > box.second[i]) {
                    distSq += (point[i] - box.second[i]) * (point[i] - box.second[i]);
                }
            }
            return std::sqrt(distSq);
        }
    };

    struct OctreeNode {
        private:
            std::atomic<uint8_t> flags;
        public:
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::shared_ptr<OctreeNode>, 8> children;
        PointType center;
        float nodeSize;
        
        mutable std::shared_ptr<NodeData> lodData;
        mutable std::shared_mutex nodeMutex; 

        OctreeNode(const PointType& min, const PointType& max) 
            : bounds(min,max), lodData(nullptr) {
            flags.store(NODE_LEAF_BIT | NODE_LOADED_BIT | NODE_DIRTY_BIT, std::memory_order_release);
            for (std::shared_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5;
            nodeSize = (bounds.second - bounds.first).norm();
        }

        OctreeNode(const OctreeNode& other) : bounds(other.bounds), points(other.points),
                  children(other.children), center(other.center), nodeSize(other.nodeSize),
                  lodData(other.lodData), flags(other.flags.load(std::memory_order_acquire)) {}

        OctreeNode& operator=(const OctreeNode& other) {
            if (this != &other) {
                std::unique_lock<std::shared_mutex> lockSelf(nodeMutex, std::defer_lock);
                std::shared_lock<std::shared_mutex> lockOther(other.nodeMutex, std::defer_lock);
                std::lock(lockSelf, lockOther);

                bounds = other.bounds;
                points = other.points;
                children = other.children;
                center = other.center;
                nodeSize = other.nodeSize;
                lodData = other.lodData;
                flags.store(other.flags.load(std::memory_order_acquire), std::memory_order_release);
            }
            return *this;
        }

        OctreeNode(OctreeNode&& other) noexcept : bounds(std::move(other.bounds)),
                   points(std::move(other.points)), children(std::move(other.children)),
                   center(std::move(other.center)), nodeSize(other.nodeSize),
                   lodData(std::move(other.lodData)), flags(other.flags.load(std::memory_order_acquire)) {}

        OctreeNode& operator=(OctreeNode&& other) noexcept {
            if (this != &other) {
                std::unique_lock<std::shared_mutex> lockSelf(nodeMutex, std::defer_lock);
                std::unique_lock<std::shared_mutex> lockOther(other.nodeMutex, std::defer_lock);
                std::lock(lockSelf, lockOther);

                bounds = std::move(other.bounds);
                points = std::move(other.points);
                children = std::move(other.children);
                center = std::move(other.center);
                nodeSize = other.nodeSize;
                lodData = std::move(other.lodData);
                flags.store(other.flags.load(std::memory_order_acquire), std::memory_order_release);
            }
            return *this;
        }

        bool isLeaf() const {
            return flags.load(std::memory_order_acquire) & NODE_LEAF_BIT;
        }
        void setLeaf(bool val) {
            if (val) flags.fetch_or(NODE_LEAF_BIT, std::memory_order_release);
            else flags.fetch_and(~NODE_LEAF_BIT, std::memory_order_release);
        }

        bool isLoaded() const {
            return flags.load(std::memory_order_acquire) & NODE_LOADED_BIT;
        }
        
        void setLoaded(bool val) {
            if (val) flags.fetch_or(NODE_LOADED_BIT, std::memory_order_release);
            else flags.fetch_and(~NODE_LOADED_BIT, std::memory_order_release);
        }

        bool isDirty() const {
            return flags.load(std::memory_order_acquire) & NODE_DIRTY_BIT;
        }
        
        void setDirty(bool val) {
            if (val) flags.fetch_or(NODE_DIRTY_BIT, std::memory_order_release);
            else flags.fetch_and(~NODE_DIRTY_BIT, std::memory_order_release);
        }

        bool isLodValid() const {
            return flags.load(std::memory_order_acquire) & NODE_LOD_VALID_BIT;
        }
        
        void setLodValid(bool val) {
            if (val) flags.fetch_or(NODE_LOD_VALID_BIT, std::memory_order_release);
            else flags.fetch_and(~NODE_LOD_VALID_BIT, std::memory_order_release);
        }
        
        void invalidateLod() {
            setLodValid(false);
        }

        bool isDirtyRecursive(bool lockSelf = true) const {
            if (isDirty()) return true;
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            if (!isLeaf()) {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (const auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (const auto& child : validChildren) {
                    if (child->isDirtyRecursive(true)) return true;
                }
            }
            return false;
        }

        void clearDirtyRecursive(bool lockSelf = true) {
            setDirty(false);
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            if (!isLeaf()) {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (auto& child : validChildren) {
                    child->clearDirtyRecursive(true);
                }
            }
        }
        
        void setLoadedRecursive(bool val, bool lockSelf = true) {
            setLoaded(val);
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            if (!isLeaf()) {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (auto& child : validChildren) {
                    child->setLoadedRecursive(val, true);
                }
            }
        }

        void collectAllPoints(std::vector<std::shared_ptr<NodeData>>& allPoints, bool lockSelf = true) const {
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            for (const auto& pt : points) {
                allPoints.push_back(pt);
            }
            if (!isLeaf()) {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (const auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (const auto& child : validChildren) {
                    child->collectAllPoints(allPoints, true);
                }
            }
        }

        bool isFullyUnloaded(bool lockSelf = true) const {
            if (isLoaded()) return false;
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            if (!isLeaf()) {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (const auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (const auto& child : validChildren) {
                    if (!child->isFullyUnloaded(true)) return false;
                }
            }
            return true;
        }

        bool contains(const PointType& point) const {
            return (point[0] >= bounds.first[0] && point[0] <= bounds.second[0] &&
                    point[1] >= bounds.first[1] && point[1] <= bounds.second[1] &&
                    point[2] >= bounds.first[2] && point[2] <= bounds.second[2]);
        }

        bool intersectsBounds(const PointType& bMin, const PointType& bMax) const {
            return (bounds.first[0] <= bMax[0] && bounds.second[0] >= bMin[0]) &&
                   (bounds.first[1] <= bMax[1] && bounds.second[1] >= bMin[1]) &&
                   (bounds.first[2] <= bMax[2] && bounds.second[2] >= bMin[2]);
        }

        bool intersectsBounds(const BoundingBox& other) const {
            return (bounds.first[0] <= other.second[0] && bounds.second[0] >= other.first[0]) &&
                   (bounds.first[1] <= other.second[1] && bounds.second[1] >= other.first[1]) &&
                   (bounds.first[2] <= other.second[2] && bounds.second[2] >= other.first[2]);
        }

        bool isEmpty() const {
            std::shared_lock<std::shared_mutex> lock(nodeMutex);
            if (!points.empty()) return false;
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && !children[i]->isEmpty()) return false;
                }
            }
            return true;
        }

        std::string getRegionName() const {
            std::ostringstream oss;
            oss << static_cast<long long>(std::floor(center.x())) << "_"
                << static_cast<long long>(std::floor(center.y())) << "_"
                << static_cast<long long>(std::floor(center.z()));
            return oss.str();
        }
        
        void saveStructure(std::ostream& os) const {
            os.write(reinterpret_cast<const char*>(&bounds), sizeof(bounds));
            uint8_t f = flags.load(std::memory_order_acquire);
            os.write(reinterpret_cast<const char*>(&f), sizeof(uint8_t));

            if (!isLeaf()) {
                uint8_t childMask = 0;
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) childMask |= (1 << i);
                }
                os.write(reinterpret_cast<const char*>(&childMask), sizeof(childMask));
                
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->saveStructure(os);
                }
            }
        }

        void loadStructure(std::istream& is) {
            is.read(reinterpret_cast<char*>(&bounds), sizeof(bounds));
            uint8_t f = 0;
            is.read(reinterpret_cast<char*>(&f), sizeof(uint8_t));
            flags.store(f, std::memory_order_release);

            center = (bounds.first + bounds.second) * 0.5;
            nodeSize = (bounds.second - bounds.first).norm();

            setLoaded(false);
            setDirty(false);

            if (!isLeaf()) {
                uint8_t childMask = 0;
                is.read(reinterpret_cast<char*>(&childMask), sizeof(childMask));
                
                for (int i = 0; i < 8; ++i) {
                    if (childMask & (1 << i)) {
                        children[i] = std::make_shared<OctreeNode>(PointType::Zero(), PointType::Zero());
                        children[i]->loadStructure(is);
                    }
                }
            }
        }

        void saveData(int currentDepth, int chunkDepth, const std::filesystem::path& currentDir, bool lockSelf = true) {
            std::unique_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::unique_lock<std::shared_mutex>(nodeMutex);
            
            if (currentDepth >= chunkDepth || isLeaf()) {
                if (isLoaded() && isDirtyRecursive(false)) {
                std::filesystem::create_directories(currentDir);
                    std::filesystem::path filePath = currentDir / (getRegionName() + ".chunkdata");
                    
                    std::vector<std::shared_ptr<NodeData>> allPoints;
                    collectAllPoints(allPoints, false);
                    
                    std::sort(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                        return a.get() < b.get();
                    });
                    allPoints.erase(std::unique(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                        return a.get() == b.get();
                    }), allPoints.end());
                    
                std::ofstream os(filePath, std::ios::binary);
                    if (os) {
                        size_t numPoints = allPoints.size();
                        os.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
                        for (const auto& pt : allPoints) {
                            pt->serialize(os);
                        }
                    }
                    clearDirtyRecursive(false);
                }
            } else {
                if (isLoaded() && isDirty()) {
                    if (!points.empty()) {
                        std::filesystem::create_directories(currentDir);
                        std::filesystem::path filePath = currentDir / (getRegionName() + ".nodedata");
                        
                        std::vector<std::shared_ptr<NodeData>> allPoints = points;
                        std::sort(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                            return a.get() < b.get();
                        });
                        allPoints.erase(std::unique(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                            return a.get() == b.get();
                        }), allPoints.end());
                        
                        std::ofstream os(filePath, std::ios::binary);
                        if (os) {
                            size_t numPoints = allPoints.size();
                            os.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
                            for (const auto& pt : allPoints) {
                                pt->serialize(os);
                            }
                        }
                    }
                    setDirty(false);
                }
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                std::filesystem::path subDir = currentDir / getRegionName();
                for (auto& child : validChildren) {
                    child->saveData(currentDepth + 1, chunkDepth, subDir, true);
                }
            }
        }

        void loadData(int currentDepth, int chunkDepth, const std::filesystem::path& currentDir, void* treePtr) {
            std::unique_lock<std::shared_mutex> lock(nodeMutex);
            if (isLoaded()) return;
            
            Octree* tree = static_cast<Octree*>(treePtr);

            if (currentDepth >= chunkDepth || isLeaf()) {
                std::filesystem::path filePath = currentDir / (getRegionName() + ".chunkdata");
                
                for (auto& child : children) child.reset();
                setLeaf(true);
                points.clear();
                
                // setLoaded(true);
                
                if (std::filesystem::exists(filePath)) {
                    std::ifstream is(filePath, std::ios::binary);
                    if (is) {
                        size_t numPoints = 0;
                        is.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
                        for (size_t i = 0; i < numPoints; ++i) {
                            auto pt = std::make_shared<NodeData>();
                            pt->deserialize(is);
                            tree->insertRecursive(this, currentDepth, pt, currentDir, true, false);
                        }
                    }
                } else {
                    std::filesystem::path oldFilePath = currentDir / (getRegionName() + ".leafdata");
                    if (std::filesystem::exists(oldFilePath)) {
                        std::ifstream is(oldFilePath, std::ios::binary);
                        if (is) {
                            size_t numPoints = 0;
                            is.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
                            for (size_t i = 0; i < numPoints; ++i) {
                                auto pt = std::make_shared<NodeData>();
                                pt->deserialize(is);
                                tree->insertRecursive(this, currentDepth, pt, currentDir, true, false);
                            }
                        }
                    }
                }
                setLoadedRecursive(true, false);
                clearDirtyRecursive(false);
            } else {
                std::filesystem::path filePath = currentDir / (getRegionName() + ".nodedata");
                points.clear();
                if (std::filesystem::exists(filePath)) {
                    std::ifstream is(filePath, std::ios::binary);
                    if (is) {
                        size_t numPoints = 0;
                        is.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
                        for (size_t i = 0; i < numPoints; ++i) {
                            auto pt = std::make_shared<NodeData>();
                            pt->deserialize(is);
                            points.push_back(pt);
                        }
                    }
                }
                setLoaded(true);
                setDirty(false);
                
                std::filesystem::path subDir = currentDir / getRegionName();
                for (auto& child : children) {
                    if (child) child->loadData(currentDepth + 1, chunkDepth, subDir, tree);
                }
            }
        }

        void offloadRegion(int currentDepth, int chunkDepth, const std::filesystem::path& currentDir, const PointType& minB, const PointType& maxB) {
            if (!intersectsBounds(minB, maxB)) return;
            
            std::unique_lock<std::shared_mutex> lock(nodeMutex);
            if (currentDepth >= chunkDepth || isLeaf()) {
                if (isLoaded()) {
                    if (isDirtyRecursive(false)) {
                        saveData(currentDepth, chunkDepth, currentDir, false); 
                    }
                    for (auto& child : children) child.reset();
                    setLeaf(true);
                    points.clear();      
                    points.shrink_to_fit();
                    setLoaded(false);
                }
            } else {
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                std::filesystem::path subDir = currentDir / getRegionName();
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                lock.unlock();
                
                for (auto& child : validChildren) {
                        child->offloadRegion(currentDepth + 1, chunkDepth, subDir, minB, maxB);
                    }
                
                lock.lock();
                if (isFullyUnloaded(false)) {
                    if (isLoaded() && isDirty()) {
                        saveData(currentDepth, chunkDepth, currentDir, false);
                    }
                    points.clear();
                    points.shrink_to_fit();
                    setLoaded(false);
                }
            }
        }

        void loadRegion(int currentDepth, int chunkDepth, const std::filesystem::path& currentDir, const PointType& minB, const PointType& maxB, Octree* tree) {
            if (!intersectsBounds(minB, maxB)) return;
            
            if (currentDepth >= chunkDepth || isLeaf()) {
                if (!isLoaded()) {
                    loadData(currentDepth, chunkDepth, currentDir, tree);
                }
            } else {
                if (!isLoaded()) {
                    loadData(currentDepth, chunkDepth, currentDir, tree);
                }
                std::shared_lock<std::shared_mutex> lock(nodeMutex);
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                lock.unlock();
                
                std::filesystem::path subDir = currentDir / getRegionName();
                for (auto& child : validChildren) {
                    child->loadRegion(currentDepth + 1, chunkDepth, subDir, minB, maxB, tree);
                }
            }
        }
        
        void saveAllData(int currentDepth, int chunkDepth, const std::filesystem::path& currentDir, bool lockSelf = true) {
            std::shared_lock<std::shared_mutex> lock;
            if (lockSelf) lock = std::shared_lock<std::shared_mutex>(nodeMutex);
            if (currentDepth >= chunkDepth || isLeaf()) {
                if (isLoaded() && isDirtyRecursive(false)) {
                    if (lockSelf) lock.unlock();
                    saveData(currentDepth, chunkDepth, currentDir, true);
                    if (lockSelf) lock.lock();
                }
            } else {
                if (isLoaded() && isDirty()) {
                    if (lockSelf) lock.unlock();
                    saveData(currentDepth, chunkDepth, currentDir, true);
                    if (lockSelf) lock.lock();
                }
                std::vector<std::shared_ptr<OctreeNode>> validChildren;
                std::filesystem::path subDir = currentDir / getRegionName();
                for (auto& child : children) {
                    if (child) validChildren.push_back(child);
                }
                if (lockSelf) lock.unlock();
                for (auto& child : validChildren) {
                    child->saveAllData(currentDepth + 1, chunkDepth, subDir, true);
                }
                if (lockSelf) lock.lock();
            }
        }
    };

    struct CelestialBody {
        PointType direction;
        float angularRadius;
        float cosAngularRadius;
        uint8_t r, g, b, emittance;
        bool baked;
        struct PixelBackup {
            size_t x, y;
            std::vector<uint8_t> data;
        };
        std::vector<PixelBackup> backup;

        CelestialBody() : angularRadius(0), cosAngularRadius(1), r(255), g(255), b(255), emittance(255), baked(false) {}
    };

    struct Skybox {
        frame skybox;
        std::map<int, CelestialBody> bodies;
        Eigen::Quaternion<float> skyRotation;

        Skybox(size_t w = 1024, size_t h = 1024) : skybox(w, h, frame::colormap::RGBA), skyRotation(Eigen::Quaternion<float>::Identity()) { }

        void dirToUV(const PointType& dir, float& u, float& v) const {
            PointType d = dir.normalized();
            u = 0.5f + (std::atan2(d.z(), d.x()) / (2.0f * M_PI));
            v = 0.5f - (std::asin(d.y()) / M_PI);
        }

        PointType uvToDir(float u, float v) const {
            float theta = (u - 0.5f) * 2.0f * M_PI;
            float phi = (0.5f - v) * M_PI;
            float y = std::sin(phi);
            float cosphi = std::cos(phi);
            float x = std::cos(theta) * cosphi;
            float z = std::sin(theta) * cosphi;
            return PointType(x, y, z);
        }
        
        std::vector<uint8_t> sample(const PointType& dir) {
            PointType rotatedDir = skyRotation * dir;
            for (auto it = bodies.rbegin(); it != bodies.rend(); ++it) {
                if (!it->second.baked) {
                    if (rotatedDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                        return {it->second.r, it->second.g, it->second.b, it->second.emittance};
                    }
                }
            }

            float u, v;
            dirToUV(rotatedDir, u, v);

            u = std::clamp(u, 0.0f, 0.9999f);
            v = std::clamp(v, 0.0f, 0.9999f);
            size_t x = static_cast<size_t>(u * skybox.getWidth());
            size_t y = static_cast<size_t>(v * skybox.getHeight());

            return skybox.getPixel(x, y);
        }

        void setBackground(float r, float g, float b, float e) {
            size_t w = skybox.getWidth();
            size_t h = skybox.getHeight();
            std::vector<float> data(w * h * 4);

            for (size_t i = 0; i < data.size(); i += 4) {
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = e;
            }
            skybox.setData(data);
        }
        
        void addBody(int id, const PointType& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance) {
            removeBody(id);
            CelestialBody body;
            body.direction = dir.normalized();
            body.angularRadius = angularRadius;
            body.cosAngularRadius = std::cos(angularRadius);
            body.r = r;
            body.g = g;
            body.b = b;
            body.emittance = emittance;
            body.baked = false;
            bodies[id] = std::move(body);
        }

        void removeBody(int id) {
            auto it = bodies.find(id);
            if (it != bodies.end()) {
                if (it->second.baked) {
                    resetBody(id);
                }
                bodies.erase(it);
            }
        }

        void moveBody(int id, const PointType& newDir) {
            auto it = bodies.find(id);
            if (it != bodies.end()) {
                bool wasBaked = it->second.baked;
                if (wasBaked) resetBody(id);
                
                it = bodies.find(id);
                it->second.direction = newDir.normalized();
                
                if (wasBaked) bakeBody(id);
            }
        }

        void bakeBody(int id) {
            auto it = bodies.find(id);
            if (it == bodies.end() || it->second.baked) return;

            if (skybox.getCompressionType() != frame::compresstype::RAW) {
                skybox.decompress();
            }

            size_t w = skybox.getWidth();
            size_t h = skybox.getHeight();
            std::vector<uint8_t> newColor = {it->second.r, it->second.g, it->second.b, it->second.emittance};
            
            it->second.backup.clear();

            for (size_t y = 0; y < h; ++y) {
                float v = (static_cast<float>(y) + 0.5f) / h; 
                for (size_t x = 0; x < w; ++x) {
                    float u = (static_cast<float>(x) + 0.5f) / w;
                    PointType pixelDir = uvToDir(u, v);
                    
                    if (pixelDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                        typename CelestialBody::PixelBackup backup;
                        backup.x = x;
                        backup.y = y;
                        backup.data = skybox.getPixel(x, y);
                        it->second.backup.push_back(std::move(backup));
                        
                        skybox.setPixel(x, y, newColor);
                    }
                }
            }
            it->second.baked = true;
        }

        void resetBody(int id) {
            auto it = bodies.find(id);
            if (it == bodies.end() || !it->second.baked) return;

            if (skybox.getCompressionType() != frame::compresstype::RAW) {
                skybox.decompress();
            }

            for (const auto& backup : it->second.backup) {
                skybox.setPixel(backup.x, backup.y, backup.data);
            }
            
            it->second.backup.clear();
            it->second.backup.shrink_to_fit();
            it->second.baked = false;
        }
    };

private:
    std::shared_ptr<OctreeNode> root_;
    mutable std::shared_mutex rootMutex_;
    size_t maxDepth;
    std::atomic<size_t> size; 
    size_t maxPointsPerNode;
    size_t chunkDepth_ = 4;
    
    Eigen::Vector3f skylight_ = {0.1f, 0.1f, 0.1f};
    Eigen::Vector3f backgroundColor_ = {0.53f, 0.81f, 0.92f};
    Skybox skybox;
    
    // Addressable Maps
    std::unique_ptr<std::mutex> mapMutex_;
    std::vector<Eigen::Vector3f> colorMap_;
    std::map<Eigen::Vector3f, IndexType, Vector3fCompare> colorToIndex_;
    
    std::vector<Material> materialMap_;
    std::map<Material, IndexType> materialToIndex_;
    std::filesystem::path storageBasePath_;

    int getChunkDepth() const {
        return static_cast<int>(chunkDepth_);
    }
    
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
    float lodFalloffRate_ = 0.1f; // Lower = better, higher = worse. 0-1
    float invLodf;
    float lodMinDistance_ = 1000.0f;
    float maxDistance_ = lodMinDistance_ * lodMinDistance_;
    
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

    void splitNode(OctreeNode* node, int currentDepth, const std::filesystem::path& currentDir, bool isLoading = false) {
        if (currentDepth >= maxDepth) return;
        // if (!isLoading && !storageBasePath_.empty() && !node->isLoaded()) {
        //     node->loadData(currentDepth, getChunkDepth(), currentDir, this);
        // }
        node->setLeaf(false);
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_shared<OctreeNode>(childBounds.first, childBounds.second);
        }

        std::vector<std::shared_ptr<NodeData>> keep;
        auto pointsToMove = std::move(node->points);
        node->points.clear();

        for (const auto& pointData : pointsToMove) {
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
        node->setDirty(true);

        std::filesystem::path subDir = currentDir / node->getRegionName();
        for (int i = 0; i < 8; ++i) {
            node->children[i]->setDirty(true);
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), currentDepth + 1, subDir, isLoading);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, int currentDepth, const std::shared_ptr<NodeData>& pointData, const std::filesystem::path& currentDir, bool isLoading = false, bool lockNode = true) {
        std::unique_lock<std::shared_mutex> lock;
        if (lockNode) lock = std::unique_lock<std::shared_mutex>(node->nodeMutex);

        node->invalidateLod();

        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxIntersectsBox(node->bounds, cubeBounds)) return false;
        
        if (!isLoading && !storageBasePath_.empty() && !node->isLoaded()) {
            if (lockNode) lock.unlock();
            const_cast<OctreeNode*>(node)->loadData(currentDepth, getChunkDepth(), currentDir, this);
            if (lockNode) lock.lock();
        }
        node->setDirty(true);

        if (!node->isLeaf() && pointData->size >= node->nodeSize) {
            node->points.emplace_back(pointData);
            return true;
        }

        if (node->isLeaf()) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && currentDepth < maxDepth) {
                splitNode(node, currentDepth, currentDir, isLoading);
            }
            return true;
        } else {
            std::vector<std::shared_ptr<OctreeNode>> childrenToVisit;
            for (int i = 0; i < 8; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                if (boxIntersectsBox(childBounds, cubeBounds)) {
                    if (!node->children[i]) {
                        node->children[i] = std::make_shared<OctreeNode>(childBounds.first, childBounds.second);
                    }
                    childrenToVisit.push_back(node->children[i]);
                }
            }
            
            if (lockNode) lock.unlock();
            bool inserted = false;
            std::filesystem::path subDir = currentDir / node->getRegionName();
            for (auto child : childrenToVisit) {
                inserted |= insertRecursive(child.get(), currentDepth + 1, pointData, subDir, isLoading, true);
            }
            return inserted;
        }
    }

    bool invalidateNodeLODRecursive(OctreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        node->invalidateLod();
        
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) validChildren.push_back(node->children[i]);
                }
            }
        lock.unlock();
        
        for (auto child : validChildren) {
            invalidateNodeLODRecursive(child.get(), bounds);
        }
        return true;
    }

    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy && pointData) {
            invalidateNodeLODRecursive(rootCopy.get(), pointData->getCubeBounds());
        }
    }
    
    void ensureBounds(const BoundingBox& targetBounds) {
        std::unique_lock<std::shared_mutex> lock(rootMutex_);
        if (!root_) {
            PointType center = (targetBounds.first + targetBounds.second) * 0.5f;
            PointType size = targetBounds.second - targetBounds.first;
            float maxDim = size.maxCoeff();
            if (maxDim <= 0.0f) maxDim = 1.0f;
            PointType halfSize = PointType::Constant(maxDim * 0.5f);
            root_ = std::make_shared<OctreeNode>(center - halfSize, center + halfSize);
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
            
            auto newRoot = std::make_shared<OctreeNode>(newMin, newMax);
            newRoot->setLeaf(false);
            
            uint8_t oldOctant = 0;
            if (expandX < 0) oldOctant |= 1;
            if (expandY < 0) oldOctant |= 2;
            if (expandZ < 0) oldOctant |= 4;
            
            newRoot->children[oldOctant] = std::move(root_);
            root_ = std::move(newRoot);
            
            maxDepth++; 
        }
    }

    void ensureLOD(OctreeNode* node, int currentDepth, const std::filesystem::path& currentDir) {
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        if (node->isLodValid()) return;

        if (!storageBasePath_.empty() && !node->isLoaded()) {
            lock.unlock();
            node->loadData(currentDepth, getChunkDepth(), currentDir, this);
            lock.lock();
            if (node->isLodValid()) return;
        }

        PointType avgPos = PointType::Zero();
        Eigen::Vector3f avgColor = Eigen::Vector3f::Zero();
        float avgEmittance = 0.0f;
        float avgRoughness = 0.0f;
        float avgMetallic = 0.0f;
        float avgTransmission = 0.0f;
        float avgIor = 0.0f;
        int count = 0;
        
        if (node->isLeaf() && node->points.size() == 1) {
            node->lodData = node->points[0];
            node->setLodValid(true);
            return;
        } else if (node->isLeaf() && node->points.empty()) {
             node->setLodValid(true);
            return;
        }

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!item || !item->isActive() || !item->isVisible()) return;
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

        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        for (const auto& child : node->children) {
            if (child) validChildren.push_back(child);
        }
        
        lock.unlock();
        std::filesystem::path subDir = currentDir / node->getRegionName();
        for (auto child : validChildren) {
            ensureLOD(child.get(), currentDepth + 1, subDir);
        }
        lock.lock();

        for (auto child : validChildren) {
            std::shared_lock<std::shared_mutex> childLock(child->nodeMutex);
            if (child->lodData) {
                accumulate(child->lodData);
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
            
            lod->setActive(true);
            lod->setVisible(true);
            lod->objectId = -1; 

            node->lodData = lod;
        }
        node->setLodValid(true);
    }

    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, int currentDepth, const PointType& pos, float tolerance, const std::filesystem::path& currentDir) const {
        if (!node->contains(pos)) return nullptr;
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!storageBasePath_.empty() && !node->isLoaded()) {
            lock.unlock();
            const_cast<OctreeNode*>(node)->loadData(currentDepth, getChunkDepth(), currentDir, const_cast<Octree*>(this));
            lock.lock();
        }
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        std::shared_ptr<OctreeNode> child;
        if (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                child = node->children[octant];
            }
        }
        lock.unlock();
        
        if (child) {
            return findRecursive(child.get(), currentDepth + 1, pos, tolerance, currentDir / node->getRegionName());
        }
        return nullptr;
    }

    bool removeRecursive(OctreeNode* node, int currentDepth, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt, const std::filesystem::path& currentDir) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;

        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!storageBasePath_.empty() && !node->isLoaded()) {
            lock.unlock();
            node->loadData(currentDepth, getChunkDepth(), currentDir, this);
            lock.lock();
        }

        node->invalidateLod();
        node->setDirty(true);
        
        bool foundAny = false;
        
        auto it = std::remove_if(node->points.begin(), node->points.end(),
            [&](const std::shared_ptr<NodeData>& pointData) {
                return pointData == targetPt;
            });
        
        if (it != node->points.end()) {
            node->points.erase(it, node->points.end());
            foundAny = true;
        }

        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) validChildren.push_back(node->children[i]);
                }
            }
        lock.unlock();

        std::filesystem::path subDir = currentDir / node->getRegionName();
        for (auto child : validChildren) {
            foundAny |= removeRecursive(child.get(), currentDepth + 1, bounds, targetPt, subDir);
        }
        
        return foundAny;
    }

    void searchNodeRecursive(OctreeNode* node, int currentDepth, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen, const std::filesystem::path& currentDir) const {
        PointType closestPoint;
        for (int i = 0; i < Dim; ++i) {
            closestPoint[i] = std::max(node->bounds.first[i], std::min(center[i], node->bounds.second[i]));
        }
        
        float distSq = (closestPoint - center).squaredNorm();
        if (distSq > radiusSq) {
            return;
        }
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!storageBasePath_.empty() && !node->isLoaded()) {
            lock.unlock();
            const_cast<OctreeNode*>(node)->loadData(currentDepth, getChunkDepth(), currentDir, const_cast<Octree*>(this));
            lock.lock();
        }
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid == -1)) {
                if (seen.insert(pointData).second) results.emplace_back(pointData);
            }
        }
        
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) validChildren.push_back(child);
            }
        }
        lock.unlock();
        
        std::filesystem::path subDir = currentDir / node->getRegionName();
        for (auto child : validChildren) {
            searchNodeRecursive(child.get(), currentDepth + 1, center, radiusSq, objectid, results, seen, subDir);
        }
    }

    void searchNode(OctreeNode* node, int currentDepth, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, const std::filesystem::path& currentDir) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        searchNodeRecursive(node, currentDepth, center, radiusSq, objectid, results, seen, currentDir);
    }

    void voxelTraverseRecursive(OctreeNode* node, int currentDepth, float tMin, float tMax, float& maxDist, 
                                bool enableLOD, const Ray& ray, std::shared_ptr<NodeData>& hit, const std::filesystem::path& currentDir) const {
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        
        if (enableLOD && !node->isLeaf()) {
            float dist = (node->center - ray.origin).norm();
            float ratio = dist / node->nodeSize;
            
            if (dist > lodMinDistance_ && ratio > invLodf) {
                 if (!node->isLodValid()) {
                    lock.unlock();
                    const_cast<Octree*>(this)->ensureLOD(const_cast<OctreeNode*>(node), currentDepth, currentDir);
                    lock.lock();
                }
                if(node->lodData) {
                    float t;
                    PointType n, h;
                    if (rayCubeIntersect(ray, node->lodData.get(), t, n, h)) {
                        if (t >= 0 && t <= maxDist) {
                            hit = node->lodData;
                            maxDist = t;
                        }
                    }
                }
                return;
            }
        }
        
        if (!storageBasePath_.empty() && !node->isLoaded()) {
            lock.unlock();
            const_cast<OctreeNode*>(node)->loadData(currentDepth, getChunkDepth(), currentDir, const_cast<Octree*>(this));
            lock.lock();
        }

        for (const auto& pointData : node->points) {
            if (!pointData->isActive() || !pointData->isVisible()) continue;
            
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
        std::filesystem::path subDir = currentDir / node->getRegionName();

        while(tMin < tMax && tMin <= maxDist) {
            Eigen::Vector3f next_t;
            next_t[0] = (currIdx & 1) ? tMax : ttt[0];
            next_t[1] = (currIdx & 2) ? tMax : ttt[1];
            next_t[2] = (currIdx & 4) ? tMax : ttt[2];

            tNext = next_t.minCoeff();

            int physIdx = currIdx ^ ray.signMask;

            std::shared_ptr<OctreeNode> child;
            if (node->children[physIdx]) {
                child = node->children[physIdx];
            }
            if (child) {
                voxelTraverseRecursive(child.get(), currentDepth + 1, tMin, tNext, maxDist, enableLOD, ray, hit, subDir);
            }

            tMin = tNext;
            currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
        }
    }

    void updateNodeVisibility(OctreeNode* node, int currentDepth, const std::filesystem::path& currentDir, const Frustum& f, const PointType& camPos) {
        float dist = f.distanceToAABB(node->bounds, camPos);
        bool visible = f.intersects(node->bounds);

        if (!visible || dist > maxDistance_) {
            node->offloadRegion(currentDepth, getChunkDepth(), currentDir, node->bounds.first, node->bounds.second);
            return;
        }

        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);

        if (dist > lodMinDistance_) {
            if (!node->isLodValid()) {
                lock.unlock();
                ensureLOD(node, currentDepth, currentDir);
                lock.lock();
            }
            
            std::vector<std::shared_ptr<OctreeNode>> validChildren;
            if (!node->isLeaf()) {
                for (auto& child : node->children) {
                    if (child) validChildren.push_back(child);
                }
            }
            lock.unlock();
            
            std::filesystem::path subDir = currentDir / node->getRegionName();
            for (auto& child : validChildren) {
                child->offloadRegion(currentDepth + 1, getChunkDepth(), subDir, child->bounds.first, child->bounds.second);
            }
        } else {
            if (!node->isLoaded()) {
                lock.unlock();
                node->loadData(currentDepth, getChunkDepth(), currentDir, this);
                lock.lock();
            }
            
            std::vector<std::shared_ptr<OctreeNode>> validChildren;
            if (!node->isLeaf()) {
                for (auto& child : node->children) {
                    if (child) validChildren.push_back(child);
                    }
                }
            lock.unlock();
            
            std::filesystem::path subDir = currentDir / node->getRegionName();
            for (auto& child : validChildren) {
                updateNodeVisibility(child.get(), currentDepth + 1, subDir, f, camPos);
            }
        }
    }

    void updateCameraVisibility(const Camera& cam, int width, int height) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy || storageBasePath_.empty()) return;
        Frustum f(cam, width, height, maxDistance_);
        updateNodeVisibility(rootCopy.get(), 0, storageBasePath_, f, cam.origin);
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
        if (bounces > maxBounces) return Eigen::Vector3f::Zero();

        auto hit = voxelTraverse(rayOrig, rayDir, std::numeric_limits<float>::max(), useLod);
        if (!hit) {
            std::vector<uint8_t> skyColor = skybox.sample(rayDir);
            Eigen::Vector3f skyEmittance(skyColor[0] / 255.0f, skyColor[1] / 255.0f, skyColor[2] / 255.0f);
            float emitPower = (skyColor.size() == 4) ? (skyColor[3] / 25.5f) : 1.0f;
            return skyEmittance * emitPower;
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
        
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                validChildren.push_back(node->children[i]);
                node->children[i].reset();
            }
        }
        node->setLeaf(true);
        lock.unlock();
        
        for (auto child : validChildren) {
            clearNode(child.get());
        }
    }

    void printStatsRecursive(const OctreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes) const {
        if (!node) return;
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (node->isLodValid() && node->lodData) lodGeneratedNodes++;
        
        size_t pts = node->points.size();
        actualPoints += pts;

        if (node->isLeaf()) {
            leafNodes++;
            maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
            minPointsInLeaf = std::min(minPointsInLeaf, pts);
        }
        
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) validChildren.push_back(child);
            }
        }
        lock.unlock();
        
        for (auto child : validChildren) {
                printStatsRecursive(child.get(), depth + 1, totalNodes, leafNodes, actualPoints, 
                                    maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes);
        }
    }

    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;
        
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        if (node->isLeaf()) return;

        bool childrenAreLeaves = true;
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                validChildren.push_back(node->children[i]);
            }
        }
        
        lock.unlock();
        for (auto child : validChildren) {
            optimizeRecursive(child.get());
            if (!child->isLeaf()) {
                childrenAreLeaves = false;
            }
        }
        lock.lock();

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
                    node->children[i].reset();
                }
                node->setLeaf(true);
                node->lodData = nullptr;
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
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        
        for (const auto& pt : node->points) {
            if (pt->isActive() && (id == -1 || pt->objectId == id)) {
                if (seen.insert(pt).second) {
                    results.push_back(pt);
                }
            }
        }
        
        std::vector<std::shared_ptr<OctreeNode>> validChildren;
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) validChildren.push_back(child);
                }
            }
        lock.unlock();
        
        for (auto child : validChildren) {
            collectNodesByObjectIdRecursive(child.get(), id, results, seen);
        }
    }

    void collectNodesByObjectId(OctreeNode* node, int id, std::vector<std::shared_ptr<NodeData>>& results) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        collectNodesByObjectIdRecursive(node, id, results, seen);
    }

public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16) :
     root_(std::make_shared<OctreeNode>(minBound, maxBound)), maxDepth(maxDepth), size(0), maxPointsPerNode(maxPointsPerNode), mapMutex_(std::make_unique<std::mutex>()) {}

    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), mapMutex_(std::make_unique<std::mutex>()) {}

    Octree(const Octree& other)
        : root_(other.root_), maxDepth(other.maxDepth), maxPointsPerNode(other.maxPointsPerNode),
          chunkDepth_(other.chunkDepth_), skylight_(other.skylight_), backgroundColor_(other.backgroundColor_),
          skybox(other.skybox), colorMap_(other.colorMap_), colorToIndex_(other.colorToIndex_),
          materialMap_(other.materialMap_), materialToIndex_(other.materialToIndex_),
          storageBasePath_(other.storageBasePath_), lodFalloffRate_(other.lodFalloffRate_),
          invLodf(other.invLodf), lodMinDistance_(other.lodMinDistance_), maxDistance_(other.maxDistance_),
          mapMutex_(std::make_unique<std::mutex>()) {
        size.store(other.size.load(std::memory_order_acquire), std::memory_order_release);
    }

    Octree& operator=(const Octree& other) {
        if (this != &other) {
            std::unique_lock<std::shared_mutex> lockSelf(rootMutex_, std::defer_lock);
            std::shared_lock<std::shared_mutex> lockOther(other.rootMutex_, std::defer_lock);
            std::lock(lockSelf, lockOther);

            std::unique_lock<std::mutex> mapLockSelf(*mapMutex_, std::defer_lock);
            std::unique_lock<std::mutex> mapLockOther(*other.mapMutex_, std::defer_lock);
            std::lock(mapLockSelf, mapLockOther);

            root_ = other.root_;
            maxDepth = other.maxDepth;
            maxPointsPerNode = other.maxPointsPerNode;
            chunkDepth_ = other.chunkDepth_;
            skylight_ = other.skylight_;
            backgroundColor_ = other.backgroundColor_;
            skybox = other.skybox;
            colorMap_ = other.colorMap_;
            colorToIndex_ = other.colorToIndex_;
            materialMap_ = other.materialMap_;
            materialToIndex_ = other.materialToIndex_;
            storageBasePath_ = other.storageBasePath_;
            lodFalloffRate_ = other.lodFalloffRate_;
            invLodf = other.invLodf;
            lodMinDistance_ = other.lodMinDistance_;
            maxDistance_ = other.maxDistance_;
            size.store(other.size.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    Octree(Octree&& other) noexcept
        : root_(std::move(other.root_)), maxDepth(other.maxDepth),
          maxPointsPerNode(other.maxPointsPerNode), chunkDepth_(other.chunkDepth_),
          skylight_(std::move(other.skylight_)), backgroundColor_(std::move(other.backgroundColor_)),
          skybox(std::move(other.skybox)), mapMutex_(std::make_unique<std::mutex>()),
          colorMap_(std::move(other.colorMap_)), colorToIndex_(std::move(other.colorToIndex_)),
          materialMap_(std::move(other.materialMap_)), materialToIndex_(std::move(other.materialToIndex_)),
          storageBasePath_(std::move(other.storageBasePath_)), lodFalloffRate_(other.lodFalloffRate_),
          invLodf(other.invLodf), lodMinDistance_(other.lodMinDistance_), maxDistance_(other.maxDistance_) {
        size.store(other.size.load(std::memory_order_acquire), std::memory_order_release);
    }

    Octree& operator=(Octree&& other) noexcept {
        if (this != &other) {
            std::unique_lock<std::shared_mutex> lockSelf(rootMutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lockOther(other.rootMutex_, std::defer_lock);
            std::lock(lockSelf, lockOther);

            std::unique_lock<std::mutex> mapLockSelf(*mapMutex_, std::defer_lock);
            std::unique_lock<std::mutex> mapLockOther(*other.mapMutex_, std::defer_lock);
            std::lock(mapLockSelf, mapLockOther);

            root_ = std::move(other.root_);
            maxDepth = other.maxDepth;
            maxPointsPerNode = other.maxPointsPerNode;
            chunkDepth_ = other.chunkDepth_;
            skylight_ = std::move(other.skylight_);
            backgroundColor_ = std::move(other.backgroundColor_);
            skybox = std::move(other.skybox);
            colorMap_ = std::move(other.colorMap_);
            colorToIndex_ = std::move(other.colorToIndex_);
            materialMap_ = std::move(other.materialMap_);
            materialToIndex_ = std::move(other.materialToIndex_);
            storageBasePath_ = std::move(other.storageBasePath_);
            lodFalloffRate_ = other.lodFalloffRate_;
            invLodf = other.invLodf;
            lodMinDistance_ = other.lodMinDistance_;
            maxDistance_ = other.maxDistance_;
            
            size.store(other.size.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    void setStoragePath(const std::string& path) {
        storageBasePath_ = path;
    }

    void setChunkDepth(size_t h) {
        chunkDepth_ = h;
    }
    size_t getChunkDepthConfig() const {
        return chunkDepth_;
    }

    void offloadRegion(const PointType& minBounds, const PointType& maxBounds) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy && !storageBasePath_.empty()) {
            rootCopy->offloadRegion(0, getChunkDepth(), storageBasePath_, minBounds, maxBounds);
        }
    }

    void loadRegion(const PointType& minBounds, const PointType& maxBounds) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy && !storageBasePath_.empty()) {
            rootCopy->loadRegion(0, getChunkDepth(), storageBasePath_, minBounds, maxBounds, this);
        }
    }

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

    void setLODFalloff(float rate) { 
        lodFalloffRate_ = rate;
        invLodf = 1 / rate;
    }
    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }
    void setMaxDistance(float dist) { maxDistance_ = dist; }

    void generateLODs() {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) return;
        ensureLOD(rootCopy.get(), 0, storageBasePath_);
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, 
             float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f) {
        
        IndexType cIdx = getColorIndex(color);
        Material mat(emittance, roughness, metallic, transmission, ior);
        IndexType mIdx = getMaterialIndex(mat);
        
        auto pointData = std::make_shared<NodeData>(data, pos, visible, cIdx, size, active, objectId, mIdx);
        
        ensureBounds(pointData->getCubeBounds());
        
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        if (insertRecursive(root_.get(), 0, pointData, storageBasePath_, false, true)) {
            this->size.fetch_add(1);
            return true;
        }
        return false;
    }

    bool remove(const PointType& pos, float tolerance = EPSILON) {
        auto pt = find(pos, tolerance);
        if (!pt) return false;
        
        auto ptBounds = pt->getCubeBounds();
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy && removeRecursive(rootCopy.get(), 0, ptBounds, pt, storageBasePath_)) {
            size.fetch_sub(1);
            return true;
        }
        return false;
    }
    
    void saveMaps(const std::filesystem::path& path) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) return;

        std::lock_guard<std::mutex> lock(*mapMutex_);
        size_t cSize = colorMap_.size();
        os.write(reinterpret_cast<const char*>(&cSize), sizeof(cSize));
        for (const auto& c : colorMap_) {
            os.write(reinterpret_cast<const char*>(c.data()), sizeof(float) * 3);
        }

        size_t mSize = materialMap_.size();
        os.write(reinterpret_cast<const char*>(&mSize), sizeof(mSize));
        for (const auto& m : materialMap_) {
            os.write(reinterpret_cast<const char*>(&m), sizeof(Material));
        }
        os.close();
    }

    void loadMaps(const std::filesystem::path& path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) return;
        
        std::lock_guard<std::mutex> lock(*mapMutex_);
        colorMap_.clear();
        colorToIndex_.clear();
        materialMap_.clear();
        materialToIndex_.clear();

        size_t cSize = 0;
        is.read(reinterpret_cast<char*>(&cSize), sizeof(cSize));
        colorMap_.resize(cSize);
        for(size_t i = 0; i < cSize; ++i) {
            is.read(reinterpret_cast<char*>(colorMap_[i].data()), sizeof(float) * 3);
            colorToIndex_[colorMap_[i]] = static_cast<IndexType>(i);
        }

        size_t mSize = 0;
        is.read(reinterpret_cast<char*>(&mSize), sizeof(mSize));
        materialMap_.resize(mSize);
        for(size_t i = 0; i < mSize; ++i) {
            is.read(reinterpret_cast<char*>(&materialMap_[i]), sizeof(Material));
            materialToIndex_[materialMap_[i]] = static_cast<IndexType>(i);
        }
        is.close();
    }

    bool save(const std::string& path) {
        setStoragePath(path);
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (storageBasePath_.empty() || !rootCopy) return false;
        std::filesystem::create_directories(storageBasePath_);
        
        saveMaps(storageBasePath_ / "maps.bin");
        
        std::ofstream os(storageBasePath_ / "tree.struct", std::ios::binary);
        rootCopy->saveStructure(os);
        os.close();
        
        rootCopy->saveAllData(0, getChunkDepth(), storageBasePath_);
        
        saveSkybox();
        
        std::cout << "Successfully saved octree to " << path << std::endl;
        return true;
    }

    bool load(const std::string& path, bool loadAllData = false) {
        setStoragePath(path);
        if (storageBasePath_.empty() || !std::filesystem::exists(storageBasePath_)) return false;

        loadMaps(storageBasePath_ / "maps.bin");
        
        std::unique_lock<std::shared_mutex> lock(rootMutex_);
        std::ifstream is(storageBasePath_ / "tree.struct", std::ios::binary);
        if (is) {
            root_ = std::make_shared<OctreeNode>(PointType::Zero(), PointType::Zero());
            root_->loadStructure(is);
            is.close();
            if (loadAllData) {
                root_->loadRegion(0, getChunkDepth(), storageBasePath_, root_->bounds.first, root_->bounds.second, this);
            }
        } else {
            return false;
        }

        loadSkybox();
        
        std::cout << "Successfully loaded octree from " << path << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = EPSILON) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy) return findRecursive(rootCopy.get(), 0, pos, tolerance, storageBasePath_);
        return nullptr;
    }

    bool inGrid(PointType pos) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        return rootCopy && rootCopy->contains(pos);
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius, int objectid = -1) const {
        std::vector<std::shared_ptr<NodeData>> results;
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) return results;
        
        float radiusSq = radius * radius;
        searchNode(rootCopy.get(), 0, center, radiusSq, objectid, results, storageBasePath_);
        
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

        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy) removeRecursive(rootCopy.get(), 0, pointData->getCubeBounds(), pointData, storageBasePath_);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->setVisible(newVisible);
        
        if (newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f)) pointData->colorIdx = getColorIndex(newColor);
        if (newSize > 0) pointData->size = newSize;
        pointData->setActive(newActive);
        pointData->objectId = targetObjId;

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
        
        lock.unlock();
        ensureBounds(pointData->getCubeBounds());
        
        lock.lock();
        bool res = false;
        rootCopy = root_;
        if (rootCopy) res = insertRecursive(rootCopy.get(), 0, pointData, storageBasePath_, false, true);
        
        if(!res) {
            size.fetch_sub(1);
        }

        return res;
    }

    bool move(const PointType& pos, const PointType& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy) removeRecursive(rootCopy.get(), 0, pointData->getCubeBounds(), pointData, storageBasePath_);
        pointData->position = newPos;
        lock.unlock();
        
        ensureBounds(pointData->getCubeBounds());

        lock.lock();
        rootCopy = root_;
        if (rootCopy && insertRecursive(rootCopy.get(), 0, pointData, storageBasePath_, false, true)) {
            return true;
        }
        size.fetch_sub(1);
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
        pointData->setActive(active);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setVisible(const PointType& pos, bool visible, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->setVisible(visible);
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
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) return hit;

        float invLodf = 1.0f / lodFalloffRate_;
        Ray oray(origin, direction);
        
        float tMin, tMax;
        if (rayBoxIntersect(oray, rootCopy->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            voxelTraverseRecursive(rootCopy.get(), 0, tMin, tMax, currentMaxDist, enableLOD, oray, hit, storageBasePath_);
        }
        return hit;
    }

    frame renderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 2,
                    int maxBounces = 4, bool globalIllumination = false, bool useLod = true) {
        updateCameraVisibility(cam, width, height);
        
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
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) return hit;
        float tMin, tMax;
        if (rayBoxIntersect(ray, rootCopy->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            voxelTraverseRecursive(rootCopy.get(), 0, tMin, tMax, currentMaxDist, enableLOD, ray, hit, storageBasePath_);
        }
        return hit;
    }

    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB) {
        //TIME_FUNCTION;
        // generateLODs();
        updateCameraVisibility(cam, width, height);

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
        // generateLODs();
        updateCameraVisibility(cam, width, height);

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
        
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy) collectNodesByObjectId(rootCopy.get(), targetObjectId, candidates);

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

    bool moveObject(int objectId, const PointType& offset) {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) return false;
        
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(rootCopy.get(), objectId, nodes);
        if(nodes.empty()) return false;

        for(auto& n : nodes) {
            removeRecursive(rootCopy.get(), 0, n->getCubeBounds(), n, storageBasePath_);
        }
        lock.unlock();

        for(auto& n : nodes) {
            n->position += offset;
            ensureBounds(n->getCubeBounds());
            lock.lock();
            rootCopy = root_; // get updated root if bounds altered
            insertRecursive(rootCopy.get(), 0, n, storageBasePath_, false, true);
            lock.unlock();
        }

        return true;
    }
    
    bool saveSkybox() {
        if (storageBasePath_.empty()) return false;
        std::filesystem::path p = storageBasePath_ / "skybox.bin";
        std::ofstream os(p, std::ios::binary);
        if (!os) return false;

        size_t numBodies = skybox.bodies.size();
        os.write(reinterpret_cast<const char*>(&numBodies), sizeof(numBodies));
        for (const auto& kv : skybox.bodies) {
            os.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
            os.write(reinterpret_cast<const char*>(&kv.second.direction), sizeof(PointType));
            os.write(reinterpret_cast<const char*>(&kv.second.angularRadius), sizeof(float));
            os.write(reinterpret_cast<const char*>(&kv.second.r), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.g), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.b), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.emittance), sizeof(uint8_t));
        }
        os.write(reinterpret_cast<const char*>(&skybox.skyRotation), sizeof(skybox.skyRotation));
        return true;
    }

    bool loadSkybox() {
        if (storageBasePath_.empty()) return false;
        std::filesystem::path p = storageBasePath_ / "skybox.bin";
        if (!std::filesystem::exists(p)) return false;
        
        std::ifstream is(p, std::ios::binary);
        if (!is) return false;

        size_t numBodies = 0;
        is.read(reinterpret_cast<char*>(&numBodies), sizeof(numBodies));
        skybox.bodies.clear();
        for (size_t i = 0; i < numBodies; i++) {
            int id;
            PointType dir;
            float angRad;
            uint8_t r, g, b, e;
            is.read(reinterpret_cast<char*>(&id), sizeof(id));
            is.read(reinterpret_cast<char*>(&dir), sizeof(PointType));
            is.read(reinterpret_cast<char*>(&angRad), sizeof(float));
            is.read(reinterpret_cast<char*>(&r), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&g), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&b), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&e), sizeof(uint8_t));
            skybox.addBody(id, dir, angRad, r, g, b, e);
        }
        is.read(reinterpret_cast<char*>(&skybox.skyRotation), sizeof(skybox.skyRotation));
        return true;
    }

    void addSkyboxBody(int id, const PointType& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance) {
        skybox.addBody(id, dir, angularRadius, r, g, b, emittance);
    }
    
    void removeSkyboxBody(int id) {
        skybox.removeBody(id);
    }
    
    void moveSkyboxBody(int id, const PointType& newDir) {
        skybox.moveBody(id, newDir);
    }
    
    void bakeSkyboxBody(int id) {
        skybox.bakeBody(id);
    }
    
    void setSkyboxBackground(float r, float g, float b, float e) {
        skybox.setBackground(r, g, b, e);
    }

    void optimize() {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (rootCopy) {
            lock.unlock();
            optimizeRecursive(rootCopy.get());
        }
        generateLODs();
    }

    void printStats(std::ostream& os = std::cout) const {
        std::shared_lock<std::shared_mutex> lock(rootMutex_);
        std::shared_ptr<OctreeNode> rootCopy = root_;
        if (!rootCopy) {
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

        printStatsRecursive(rootCopy.get(), 0, totalNodes, leafNodes, actualPoints, 
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
        os << "  Min               : [" << rootCopy->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << rootCopy->bounds.second.transpose() << "]\n";
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
        PointType minBound = root_ ? root_->bounds.first : PointType::Zero();
        PointType maxBound = root_ ? root_->bounds.second : PointType::Zero();
        root_ = std::make_shared<OctreeNode>(minBound, maxBound);
        
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
