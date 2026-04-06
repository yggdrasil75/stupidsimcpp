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
#include <mutex>
#include <shared_mutex>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <random>
#include <chrono>
#include <cstdint>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <type_traits>
#include <future>

#ifdef SSE
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

constexpr int Dim = 3;

#ifndef gridheaders
#define gridheaders
static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
static constexpr uint8_t STATIC_BIT = 1 << 7;

static constexpr uint8_t LEAF_BIT = 1 << 0;
static constexpr uint8_t LOADED_BIT = 1 << 1;
static constexpr uint8_t DIRTY_BIT = 1 << 2;
static constexpr uint8_t LOADQUEUED = 1 << 3;
static constexpr uint8_t SAVEDQUEUED = 1 << 4;
static constexpr uint8_t KEEPLOADED_BIT = 1 << 5;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
#endif

template<size_t N>
struct GridStoragePath {
    char value[N];
    constexpr GridStoragePath(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            value[i] = str[i];
        }
    }
};

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
class Octree {
public:
    using PointType = Eigen::Matrix<float, Dim, 1>;
    using BoundingBox = std::pair<PointType, PointType>;

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
        Eigen::Vector3f color;
        Material material;
        std::atomic<uint8_t> flags;

        NodeData(const T& data, const PointType& pos, bool visible, const Eigen::Vector3f& color, float size = 0.01f,
                 bool active = true, int objectId = -1, const Material& material = Material(), bool staticbit = 0) 
                : data(data), position(pos), objectId(objectId), size(size), 
                  color(color), material(material), flags(0) {
            setActive(active);
            setVisible(visible);
            setStatic(staticbit);
        }
        
        NodeData() : objectId(-1), size(0.0f), color(Eigen::Vector3f::Zero()), material(), flags(0) {}

        NodeData(const NodeData& other)
            : data(other.data), position(other.position), objectId(other.objectId), size(other.size),
              color(other.color), material(other.material), flags(other.flags.load(std::memory_order_relaxed)) {}

        NodeData& operator=(const NodeData& other) {
            if (this != &other) {
                data = other.data;
                position = other.position;
                objectId = other.objectId;
                size = other.size;
                color = other.color;
                material = other.material;
                flags.store(other.flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
            return *this;
        }

        bool isActive() const {
            return flags.load(std::memory_order_relaxed) & ACTIVE_BIT;
        }
        bool isVisible() const {
            return flags.load(std::memory_order_relaxed) & VISIBLE_BIT;
        }
        bool isStatic() const {
            return flags.load(std::memory_order_relaxed) & STATIC_BIT;
        }
        bool isActiveAndVisible() const {
            return (flags.load(std::memory_order_relaxed) & (ACTIVE_BIT | VISIBLE_BIT)) != (ACTIVE_BIT | VISIBLE_BIT);
        }

        void setActive(bool v) {
            if (v) flags.fetch_or(ACTIVE_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~ACTIVE_BIT, std::memory_order_relaxed);
        }
        void setVisible(bool v) {
            if (v) flags.fetch_or(VISIBLE_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~VISIBLE_BIT, std::memory_order_relaxed);
        }
        void setStatic(bool v) {
            if (v) flags.fetch_or(STATIC_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~STATIC_BIT, std::memory_order_relaxed);
        }
        
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
        std::atomic<uint8_t> flags;
        
        mutable std::shared_ptr<NodeData> lodData;
        mutable std::shared_mutex nodeMutex;

        OctreeNode(const PointType& min, const PointType& max) : bounds(min,max), flags(0), lodData(nullptr) {
            setLeaf(true);
            setLoaded(true);
            setDirty(true);
            setLoadQueued(false);
            setSaveQueued(false);
            setKeepLoaded(false);
            for (std::unique_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5;
            nodeSize = (bounds.second - bounds.first).norm();
        }

        std::unique_ptr<OctreeNode> clone() const {
            auto newNode = std::make_unique<OctreeNode>(bounds.first, bounds.second);
            newNode->flags.store(flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
            
            newNode->points = points; 
            newNode->center = center;
            newNode->nodeSize = nodeSize;
            newNode->lodData = lodData;
            
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) {
                        newNode->children[i] = children[i]->clone();
                    }
                }
            }
            return newNode;
        }

        bool isLeaf() const {
            return flags.load(std::memory_order_relaxed) & LEAF_BIT;
        }
        bool isLoaded() const {
            return flags.load(std::memory_order_relaxed) & LOADED_BIT;
        }
        bool isDirty() const {
            return flags.load(std::memory_order_relaxed) & DIRTY_BIT;
        }
        bool isQueued() const {
            return flags.load(std::memory_order_relaxed) & LOADQUEUED;
        }
        bool isSaveQueued() const {
            return flags.load(std::memory_order_relaxed) & SAVEDQUEUED;
        }
        bool isKeepLoaded() const {
            return flags.load(std::memory_order_relaxed) & KEEPLOADED_BIT;
        }

        void setLeaf(bool v) {
            if (v) flags.fetch_or(LEAF_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~LEAF_BIT, std::memory_order_relaxed);
        }
        void setLoaded(bool v) {
            if (v) flags.fetch_or(LOADED_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~LOADED_BIT, std::memory_order_relaxed);
        }
        void setDirty(bool v) {
            if (v) flags.fetch_or(DIRTY_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~DIRTY_BIT, std::memory_order_relaxed);
        }
        void setLoadQueued(bool v) {
            if (v) flags.fetch_or(LOADQUEUED, std::memory_order_relaxed);
            else flags.fetch_and(~LOADQUEUED, std::memory_order_relaxed);
        }
        void setSaveQueued(bool v) {
            if (v) flags.fetch_or(SAVEDQUEUED, std::memory_order_relaxed);
            else flags.fetch_and(~SAVEDQUEUED, std::memory_order_relaxed);
        }
        void setKeepLoaded(bool v) {
            if (v) flags.fetch_or(KEEPLOADED_BIT, std::memory_order_relaxed);
            else flags.fetch_and(~KEEPLOADED_BIT, std::memory_order_relaxed);
        }

        bool contains(const PointType& point) const {
            return (point[0] >= bounds.first[0] && point[0] <= bounds.second[0] &&
                    point[1] >= bounds.first[1] && point[1] <= bounds.second[1] &&
                    point[2] >= bounds.first[2] && point[2] <= bounds.second[2]);
        }

        bool isEmpty() const {
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
            oss << static_cast<int>(std::floor(center.x())) << "." 
                << static_cast<int>(std::floor(center.y())) << "." 
                << static_cast<int>(std::floor(center.z()));
            return oss.str();
        }

        std::string getRegionPath() const {
            int64_t cx = static_cast<int64_t>(std::floor(center.x()));
            int64_t cy = static_cast<int64_t>(std::floor(center.y()));
            int64_t cz = static_cast<int64_t>(std::floor(center.z()));
            int64_t s = static_cast<int64_t>(std::floor(nodeSize));
            
            fs::path p(StoragePath.value);
            p /= std::to_string(s);
            p /= std::to_string(cx);
            p /= std::to_string(cy);
            p /= std::to_string(cz);
            
            std::error_code ec;
            fs::create_directories(p, ec);
            
            p /= "data.region";
            return p.string();
        }

        template<typename V>
        static void writeVal(std::ofstream& out, const V& val) {
            out.write(reinterpret_cast<const char*>(&val), sizeof(V));
        }

        template<typename V>
        static void readVal(std::ifstream& in, V& val) {
            in.read(reinterpret_cast<char*>(&val), sizeof(V));
        }

        static void writeVec3(std::ofstream& out, const Eigen::Vector3f& vec) {
            writeVal(out, vec.x());
            writeVal(out, vec.y());
            writeVal(out, vec.z());
        }

        static void readVec3(std::ifstream& in, Eigen::Vector3f& vec) {
            float x, y, z;
            readVal(in, x);
            readVal(in, y);
            readVal(in, z);
            vec = Eigen::Vector3f(x, y, z);
        }

        static void serializeData(std::ofstream& out, const T& data) {
            if constexpr (is_shared_ptr<T>::value) {
                bool hasData = (data != nullptr);
                writeVal(out, hasData);
                if (hasData) data->serialize(out);
            } else if constexpr (std::is_pointer_v<T>) {
                bool hasData = (data != nullptr);
                writeVal(out, hasData);
                if (hasData) data->serialize(out);
            } else if constexpr (std::is_class_v<T>) {
                data.serialize(out);
            } else {
                writeVal(out, data);
            }
        }

        static void deserializeData(std::ifstream& in, T& data) {
            if constexpr (is_shared_ptr<T>::value) {
                bool hasData;
                readVal(in, hasData);
                if (hasData) {
                    using ElemType = typename T::element_type;
                    data = ElemType::deserialize(in);
                } else {
                    data = nullptr;
                }
            } else if constexpr (std::is_pointer_v<T>) {
                bool hasData;
                readVal(in, hasData);
                if (hasData) {
                    using ElemType = std::remove_pointer_t<T>;
                    data = ElemType::deserialize(in);
                } else {
                    data = nullptr;
                }
            } else if constexpr (std::is_class_v<T>) {
                data = T::deserialize(in);
            } else {
                readVal(in, data);
            }
        }

        size_t getSubtreePointCount() const {
            if (!isLoaded()) return 0;
            size_t count = points.size();
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) {
                        count += children[i]->getSubtreePointCount();
                    }
                }
            }
            return count;
        }

        bool isSubtreeFullyLoaded() const {
            if (!isLoaded()) return false;
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && !children[i]->isSubtreeFullyLoaded()) return false;
                }
            }
            return true;
        }

        void serializeSubtree(std::ofstream& out) const {
            writeVal(out, isLeaf());
            writeVal(out, points.size());
            for (const auto& pt : points) {
                serializeData(out, pt->data);
                writeVec3(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->flags.load(std::memory_order_relaxed));
                writeVal(out, pt->size);
                writeVec3(out, pt->color);
                writeVal(out, pt->material);
            }

            if (!isLeaf()) {
                uint8_t childMask = 0;
                for (int i = 0; i < 8; ++i) if (children[i]) childMask |= (1 << i);
                writeVal(out, childMask);
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->serializeSubtree(out);
                }
            }
        }

        void deserializeSubtree(std::ifstream& in) {
            bool leaf;
            readVal(in, leaf);
            setLeaf(leaf);

            size_t pointCount;
            readVal(in, pointCount);
            points.reserve(pointCount);
            for (size_t i = 0; i < pointCount; ++i) {
                auto pt = std::make_shared<NodeData>();
                deserializeData(in, pt->data);
                readVec3(in, pt->position);
                readVal(in, pt->objectId);
                uint8_t f;
                readVal(in, f);
                pt->flags.store(f, std::memory_order_relaxed);
                readVal(in, pt->size);
                readVec3(in, pt->color);
                readVal(in, pt->material);
                points.push_back(pt);
            }

            if (!isLeaf()) {
                uint8_t childMask;
                readVal(in, childMask);
                for (int i = 0; i < 8; ++i) {
                    if ((childMask >> i) & 1) {
                        PointType childMin, childMax;
                        for (int d = 0; d < Dim; ++d) {
                            bool high = (i >> d) & 1;
                            childMin[d] = high ? center[d] : bounds.first[d];
                            childMax[d] = high ? bounds.second[d] : center[d];
                        }
                        children[i] = std::make_unique<OctreeNode>(childMin, childMax);
                        std::lock_guard<std::shared_mutex> lock(children[i]->nodeMutex);
                        children[i]->deserializeSubtree(in);
                    } else {
                        children[i] = nullptr;
                    }
                }
            }
            setLoaded(true);
            setDirty(false);
        }

        void clearDirtySubtree() {
            setDirty(false);
            if (!isLeaf()) {
                for (auto& child : children) if (child) child->clearDirtySubtree();
            }
        }

        bool saveRegion() {
            std::string path = getRegionPath();
            std::ofstream out(path, std::ios::binary);
            if (!out) return false;
            serializeSubtree(out);
            clearDirtySubtree();
            return true;
        }

        bool loadRegion() {
            std::string path = getRegionPath();
            std::ifstream in(path, std::ios::binary);
            if (in) {
                deserializeSubtree(in);
                setLoaded(true);
                return true;
            }
            return false;
        }

        void offload() {
            std::lock_guard<std::shared_mutex> lock(nodeMutex);
            if (isDirty()) return;
            setLoaded(false);
            for (int i = 0; i < 8; ++i) {
                children[i].reset();
            }
            points.clear();
            points.shrink_to_fit();
        }

        void serialize(std::ofstream& out, size_t regionTargetPoints) {
            bool offloaded = !isLoaded();
            size_t subPoints = offloaded ? 0 : getSubtreePointCount();

            bool isRegion = offloaded || (subPoints > 0 && (subPoints <= regionTargetPoints || isLeaf()) && isSubtreeFullyLoaded());

            writeVal(out, isRegion);

            if (isRegion) {
                if (!offloaded && isDirty()) saveRegion();
                return;
            }

            writeVal(out, isLeaf());
            writeVal(out, points.size());
            for (const auto& pt : points) {
                serializeData(out, pt->data);
                writeVec3(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->flags.load(std::memory_order_relaxed));
                writeVal(out, pt->size);
                writeVec3(out, pt->color);
                writeVal(out, pt->material);
            }

            if (!isLeaf()) {
                uint8_t childMask = 0;
                for (int i = 0; i < 8; ++i) if (children[i]) childMask |= (1 << i);
                writeVal(out, childMask);
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->serialize(out, regionTargetPoints);
                }
            }
        }

        void deserialize(std::ifstream& in, size_t regionTargetPoints) {
            bool isRegion;
            readVal(in, isRegion);

            if (isRegion) {
                setLoaded(false);
                setDirty(false);
                setLeaf(false);
                return;
            }

            bool leaf;
            readVal(in, leaf);
            setLeaf(leaf);

            size_t pointCount;
            readVal(in, pointCount);
            points.reserve(pointCount);
            for (size_t i = 0; i < pointCount; ++i) {
                auto pt = std::make_shared<NodeData>();
                deserializeData(in, pt->data);
                readVec3(in, pt->position);
                readVal(in, pt->objectId);
                uint8_t f;
                readVal(in, f);
                pt->flags.store(f, std::memory_order_relaxed);
                readVal(in, pt->size);
                readVec3(in, pt->color);
                readVal(in, pt->material);
                points.push_back(pt);
            }

            if (!isLeaf()) {
                uint8_t childMask;
                readVal(in, childMask);
                for (int i = 0; i < 8; ++i) {
                    if ((childMask >> i) & 1) {
                        PointType childMin, childMax;
                        for (int d = 0; d < Dim; ++d) {
                            bool high = (i >> d) & 1;
                            childMin[d] = high ? center[d] : bounds.first[d];
                            childMax[d] = high ? bounds.second[d] : center[d];
                        }
                        children[i] = std::make_unique<OctreeNode>(childMin, childMax);
                        children[i]->deserialize(in, regionTargetPoints);
                    } else {
                        children[i] = nullptr;
                    }
                }
            }
            setLoaded(true);
            setDirty(false);
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
        Eigen::Quaternionf skyRotation;

        Skybox(size_t w = 1024, size_t h = 1024) : skybox(w, h, frame::colormap::RGBA), skyRotation(Eigen::Quaternionf::Identity()) { }

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

        Eigen::Vector3f sampleVector(const PointType& dir) {
            std::vector<uint8_t> px = sample(dir);
            if (px.size() >= 3) {
                float r = px[0] / 255.0f;
                float g = px[1] / 255.0f;
                float b = px[2] / 255.0f;
                float e = px.size() >= 4 ? (px[3] / 255.0f) : 1.0f;
                return Eigen::Vector3f(r, g, b) * e;
            }
            return Eigen::Vector3f::Zero();
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
            skybox.setData(data, frame::colormap::RGBA);
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
                
                it->second.direction = newDir.normalized();
                
                if (wasBaked) bakeBody(id);
            }
        }

        void bakeBody(int id) {
            auto it = bodies.find(id);
            if (it == bodies.end() || it->second.baked) return;

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

            for (const auto& backup : it->second.backup) {
                skybox.setPixel(backup.x, backup.y, backup.data);
            }
            
            it->second.backup.clear();
            it->second.backup.shrink_to_fit();
            it->second.baked = false;
        }
    };

private:
    std::unique_ptr<OctreeNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    Skybox skybox_;
    Eigen::Vector3f skylight_ = {0.1f, 0.1f, 0.1f};
    Eigen::Vector3f backgroundColor_ = {0.53f, 0.81f, 0.92f};

    mutable std::queue<std::function<void()>> taskQueue_;
    mutable std::mutex taskMutex_;
    mutable std::condition_variable taskCV_;
    std::thread workerThread_;
    std::atomic<bool> stopWorker_{false};
    std::atomic<bool> autoOptimize_{true};
    std::atomic<bool> streamingQueued_{false};

    size_t regionTargetPoints_ = 4096;

    void lazilyOffload(OctreeNode* node) {
        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            if (!node->isLoaded() || node->isSaveQueued()) return;

            node->setSaveQueued(true);
            node->setLoadQueued(false);
        }

        enqueueTask([this, node]() {
            {
                std::shared_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (node->isLoaded() && node->isSaveQueued()) {
                    if (node->isDirty()) {
                        node->saveRegion();
                    }
                }
            }
            node->offload();
            {
                std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                node->setSaveQueued(false);
            }
        });
    }

    void ensureLoaded(OctreeNode* node, bool asyncLoad = false) {
        {
            // std::unique_lock<std::shared_mutex> lock(node->nodeMutex); // using atomics so dont need this?
            if (node->isLoaded() || node->isQueued()) return; 
            else {
                node->setLoadQueued(true);
                node->setSaveQueued(false);
            }
        }

        if (asyncLoad) {
            enqueueTask([this, node]() {
                bool justLoaded = false;
                {
                    std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                    if (!node->isLoaded()) {
                        node->loadRegion();
                        justLoaded = node->isLoaded();
                    }
                    node->setLoadQueued(false);
                }
                if (justLoaded) {
                    ensureLOD(node);
                }
                
            });
        } else {
            {
                std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (!node->isLoaded()) node->loadRegion();
                node->setLoadQueued(false);
            }
            if (node->isLoaded()) {
                ensureLOD(node);
            }
        }
    }

    void startWorkerThread() {
        stopWorker_.store(false);
        workerThread_ = std::thread([this]() {
            auto lastOptimize = std::chrono::steady_clock::now();
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(taskMutex_);
                    auto nextOptimize = lastOptimize + std::chrono::seconds(60);
                    
                    bool timedOut = !taskCV_.wait_until(lock, nextOptimize, [this] { 
                        return stopWorker_ || !taskQueue_.empty(); 
                    });
                    
                    if (stopWorker_ && taskQueue_.empty()) return;
                    
                    if (!taskQueue_.empty()) {
                        task = std::move(taskQueue_.front());
                        taskQueue_.pop();
                    } else if (timedOut && autoOptimize_) {
                        task = [this]() { this->optimize(); };
                        lastOptimize = std::chrono::steady_clock::now();
                    }
                }
                if (task) {
                    task();
                    lastOptimize = std::chrono::steady_clock::now();
                }
            }
        });
    }

    void stopWorkerThread() {
        stopWorker_.store(true);
        taskCV_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    uint32_t expandBits10(uint32_t v) const noexcept {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v <<  8)) & 0x0300F00F;
        v = (v | (v <<  4)) & 0x030C30C3;
        v = (v | (v <<  2)) & 0x09249249;
        return v;
    }

    uint32_t morton3D_10(float x, float y, float z) const noexcept {
        x = std::max(0.0f, std::min(1.0f, x));
        y = std::max(0.0f, std::min(1.0f, y));
        z = std::max(0.0f, std::min(1.0f, z));
        uint32_t xx = static_cast<uint32_t>(x * 1023.0f);
        uint32_t yy = static_cast<uint32_t>(y * 1023.0f);
        uint32_t zz = static_cast<uint32_t>(z * 1023.0f);
        return (expandBits10(xx) << 2) | (expandBits10(yy) << 1) | expandBits10(zz);
    }
    
public:
    void addSkyBody(int id, const PointType& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance = 255) {
        skybox_.addBody(id, dir, angularRadius, r, g, b, emittance);
    }

    void moveSkyBody(int id, const PointType& newDir) {
        skybox_.moveBody(id, newDir);
    }

    void removeSkyBody(int id) {
        skybox_.removeBody(id);
    }

    void bakeSkyBody(int id) {
        skybox_.bakeBody(id);
    }

    void waitForIdle() {
        if (std::this_thread::get_id() == workerThread_.get_id()) {
            return;
        }
        std::promise<void> p;
        auto f = p.get_future();
        enqueueTask([&p]{ p.set_value(); });
        f.wait();
    }
private:

    float lodFalloffRate_ = 0.1f; // Lower = better, higher = worse. 0-1
    float invLodf = 1 / lodFalloffRate_;
    float lodMinDistance_ = 100.0f;
    float maxDistance_ = lodMinDistance_ * lodMinDistance_;
    float keepDistance_ = maxDistance_ * 1.2;
    
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

    bool boxContainsBox(const BoundingBox& outer, const BoundingBox& inner) const {
        return (inner.first[0] >= outer.first[0] && inner.second[0] <= outer.second[0] &&
                inner.first[1] >= outer.first[1] && inner.second[1] <= outer.second[1] &&
                inner.first[2] >= outer.first[2] && inner.second[2] <= outer.second[2]);
    }

    void splitNode(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
        }

        std::vector<std::shared_ptr<NodeData>> keep;
        for (const auto& pointData : node->points) {
            BoundingBox cubeBounds = pointData->getCubeBounds();
            bool placedInChild = false;
            for (int i = 0; i < 8; ++i) {
                if (boxContainsBox(node->children[i]->bounds, cubeBounds)) {
                    node->children[i]->points.emplace_back(pointData);
                    placedInChild = true;
                    break;
                }
            }
            if (!placedInChild) {
                keep.emplace_back(pointData);
            }
        }

        node->points = std::move(keep);
        node->setLeaf(false);

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        ensureLoaded(node);

        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxIntersectsBox(node->bounds, cubeBounds)) return false;

        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        if (node->isLeaf()) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            node->setDirty(true);
            return true;
        } else {
            bool insertedInChild = false;
            for (int i = 0; i < 8; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                if (boxContainsBox(childBounds, cubeBounds)) {
                    if (!node->children[i]) {
                        node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
                    }
                    insertedInChild = insertRecursive(node->children[i].get(), pointData, depth + 1);
                    break;
                }
            }
            
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            if (!insertedInChild) {
                node->points.emplace_back(pointData);
            }
            node->setDirty(true);
            return true;
        }
    }

    bool invalidateNodeLODRecursive(OctreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        ensureLoaded(node);
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        if (!node->isLeaf()) {
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
            
            if (expandX < 0) newMin.x() -= size.x();
            else newMax.x() += size.x();
            if (expandY < 0) newMin.y() -= size.y();
            else newMax.y() += size.y();
            if (expandZ < 0) newMin.z() -= size.z();
            else newMax.z() += size.z();
            
            auto newRoot = std::make_unique<OctreeNode>(newMin, newMax);
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

    void ensureLOD(OctreeNode* node) {
        ensureLoaded(node);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        if (node->lodData != nullptr) return;

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
            return;
        } else if (node->isLeaf() && node->points.empty()) {
            return;
        }

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!item || !item->isActive() || !item->isVisible()) return;
            avgColor += item->color;
            Material mat = item->material;
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

            lod->color = avgColor * invCount;
            Material avgMat(avgEmittance * invCount, avgRoughness * invCount, 
                            avgMetallic * invCount, avgTransmission * invCount, avgIor * invCount);
            lod->material = avgMat;
            
            lod->setActive(true);
            lod->setVisible(true);
            lod->objectId = -1; 

            node->lodData = lod;
        }
    }

    void loadSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLoaded(node, true);
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                loadSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void loadAndLodSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLOD(node);
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                loadAndLodSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void updateStreamingRecursive(OctreeNode* node, const PointType& camPos, const PointType& camDir) {
        if (!node) return;
        
        float minDistSq = 0.0f;
        float maxDistSq = 0.0f;

        // if (!node->contains(camPos)) {
        for(int i = 0; i < Dim; ++i) {
            float v = camPos[i];
            float minBound = node->bounds.first[i];
            float maxBound = node->bounds.second[i];
            
            float d1 = v - minBound;
            float d2 = v - maxBound;
            
            if(v < minBound) {
                minDistSq += d1 * d1;
            } else if(v > maxBound) {
                minDistSq += d2 * d2;
            }
            
            float maxD = std::max(std::abs(d1), std::abs(d2));
            maxDistSq += maxD * maxD;
        }

        bool isBehind = false;
        PointType maxPoint;
        maxPoint.x() = (camDir.x() >= 0) ? node->bounds.second.x() : node->bounds.first.x();
        maxPoint.y() = (camDir.y() >= 0) ? node->bounds.second.y() : node->bounds.first.y();
        maxPoint.z() = (camDir.z() >= 0) ? node->bounds.second.z() : node->bounds.first.z();
        
        if ((maxPoint - camPos).dot(camDir) < -0.05f) {
            isBehind = true;
        }
        
        float lodDistSq = lodMinDistance_ * lodMinDistance_;
        float maxDistSq_max = maxDistance_ * maxDistance_;
        float keepDistSq = keepDistance_ * keepDistance_;
        
        if (maxDistSq <= lodDistSq) {
            loadSubtreeRecursive(node);
            return;
        }

        if (maxDistSq <= maxDistSq_max && minDistSq > lodDistSq) {
            loadAndLodSubtreeRecursive(node);
            return;
        }
        
        if (minDistSq > keepDistSq) {
            if (!node->isLoaded()) return;
            size_t subPoints = node->getSubtreePointCount();
            bool fullyLoaded = node->isSubtreeFullyLoaded();

            if ((subPoints > regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
                if (subPoints > 0) lazilyOffload(node);
                return;
            }
            if (!node->isLeaf()){
                for (int i = 0; i < 8; ++i) {
                    updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                }
            }
            return;
        }

        if (minDistSq > lodDistSq) {
            ensureLOD(node);
        } else {
            ensureLoaded(node, true);
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                }
            }
        }
    }

    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, const PointType& pos, float tolerance) {
        if (!node->contains(pos)) return nullptr;
        ensureLoaded(node, false);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        
        for (const auto& pointData : node->points) {
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        if (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                return findRecursive(node->children[octant].get(), pos, tolerance);
            }
        }
        return nullptr;
    }

    bool removeRecursive(OctreeNode* node, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        ensureLoaded(node, false);
        bool foundAny = false;
        
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            
            auto it = std::remove_if(node->points.begin(), node->points.end(),
                [&](const std::shared_ptr<NodeData>& pointData) {
                    return pointData == targetPt;
                });
            
            if (it != node->points.end()) {
                node->points.erase(it, node->points.end());
                foundAny = true;
            }
            if (foundAny) {
                node->lodData = nullptr; 
                node->setDirty(true);
            }
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    foundAny |= removeRecursive(node->children[i].get(), bounds, targetPt);
                }
            }
            if (foundAny) {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                node->lodData = nullptr;
                node->setDirty(true);
            }
        }
        return foundAny;
    }

    void searchNodeRecursive(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) {
        PointType closestPoint;
        for (int i = 0; i < Dim; ++i) {
            closestPoint[i] = std::max(node->bounds.first[i], std::min(center[i], node->bounds.second[i]));
        }
        
        float distSq = (closestPoint - center).squaredNorm();
        if (distSq > radiusSq) return;
        
        ensureLoaded(node, false);
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid == -1)) {
                if (seen.insert(pointData).second) results.emplace_back(pointData);
            }
        }
        
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) searchNodeRecursive(child.get(), center, radiusSq, objectid, results, seen);
            }
        }
    }

    void searchNode(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results) {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        searchNodeRecursive(node, center, radiusSq, objectid, results, seen);
    }

    void voxelTraverseRecursive(OctreeNode* node, float tMin, float tMax, float& maxDist, bool enableLOD,
          const Ray& ray, std::shared_ptr<NodeData>& hit, bool asyncLoad = false) {
        if (!node->isLoaded()) {
            ensureLoaded(node, asyncLoad);
            if (!node->isLoaded()) {
                if (enableLOD && node->lodData) {
                    float dist = (node->center - ray.origin).norm();
                    float ratio = dist / node->nodeSize;
                    if (dist > lodMinDistance_ && ratio > invLodf) {
                        float t;
                        PointType n, h;
                        if (rayCubeIntersect(ray, node->lodData.get(), t, n, h)) {
                            if (t >= 0 && t <= maxDist) {
                                hit = node->lodData;
                                maxDist = t;
                            }
                        }
                    }
                }
                return;
            }
        }

        if (enableLOD && !node->isLeaf()) {
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
            if (!pointData->isActive()) continue;
            
            float t;
            PointType normal, hitPoint;
            if (rayCubeIntersect(ray, pointData.get(), t, normal, hitPoint) && pointData->isVisible()) {
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
                voxelTraverseRecursive(node->children[physIdx].get(), tMin, tNext, maxDist, enableLOD, ray, hit, asyncLoad);
            }

            tMin = tNext;
            currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
        }
    }
    
    void fastVoxelTraverseRecursive(OctreeNode* node, float tMin, float tMax, float& maxDist,
          const Ray& ray, std::shared_ptr<NodeData>& hit) {
        bool nodeloaded = node->isLoaded();
        ensureLoaded(node, true);

        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLeaf() && node->lodData) {
            float dist = (node->center - ray.origin).norm();
            if (dist > lodMinDistance_ && (dist / node->nodeSize) > invLodf) {
                float t;
                PointType n, h;
                if (rayCubeIntersect(ray, node->lodData.get(), t, n, h)) {
                    if (t >= 0 && t <= maxDist) {
                        hit = node->lodData;
                        maxDist = t;
                    }
                }
                return;
            }
        }

        for (const auto& pt : node->points) {
            if (pt->isActiveAndVisible()) continue;
            float t;
            PointType n, h;
            if (rayCubeIntersect(ray, pt.get(), t, n, h)) {
                if (t >= 0 && t <= maxDist && t <= tMax + 0.001f) {
                    maxDist = t;
                    hit = pt;
                }
            }
        }

        if (node->isLeaf()) return;

        Eigen::Vector3f ttt = (node->center - ray.origin).cwiseProduct(ray.invDir);
        int currIdx = ((tMin >= ttt.x()) ? 1 : 0) | ((tMin >= ttt.y()) ? 2 : 0) | ((tMin >= ttt.z()) ? 4 : 0);
        
        while(tMin < tMax && tMin <= maxDist) {
            float tNextX = (currIdx & 1) ? tMax : ttt.x();
            float tNextY = (currIdx & 2) ? tMax : ttt.y();
            float tNextZ = (currIdx & 4) ? tMax : ttt.z();
            float tNext = std::min({tNextX, tNextY, tNextZ});
            
            int physIdx = currIdx ^ ray.signMask;
            if (node->children[physIdx]) {
                fastVoxelTraverseRecursive(node->children[physIdx].get(), tMin, tNext, maxDist, ray, hit);
            }
            
            tMin = tNext;
            currIdx |= ((tNextX <= tNext) ? 1 : 0) | ((tNextY <= tNext) ? 2 : 0) | ((tNextZ <= tNext) ? 4 : 0);
        }
    }
    
    void fastBypassVoxelTraverseRecursive(OctreeNode* node, float tMin, float tMax, float& maxDist,
          const Ray& ray, std::shared_ptr<NodeData>& hit) {

        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLeaf() && node->lodData) {
            float dist = (node->center - ray.origin).norm();
            if (dist > lodMinDistance_ && (dist / node->nodeSize) > invLodf) {
                float t;
                PointType n, h;
                if (rayCubeIntersect(ray, node->lodData.get(), t, n, h)) {
                    if (t >= 0 && t <= maxDist) {
                        hit = node->lodData;
                        maxDist = t;
                    }
                }
                return;
            }
        }

        for (const auto& pt : node->points) {
            if (pt->isActiveAndVisible()) continue;
            float t;
            PointType n, h;
            if (rayCubeIntersect(ray, pt.get(), t, n, h)) {
                if (t >= 0 && t <= maxDist && t <= tMax + 0.001f) {
                    maxDist = t;
                    hit = pt;
                }
            }
        }

        if (node->isLeaf()) return;

        Eigen::Vector3f ttt = (node->center - ray.origin).cwiseProduct(ray.invDir);
        int currIdx = ((tMin >= ttt.x()) ? 1 : 0) | ((tMin >= ttt.y()) ? 2 : 0) | ((tMin >= ttt.z()) ? 4 : 0);
        
        while(tMin < tMax && tMin <= maxDist) {
            float tNextX = (currIdx & 1) ? tMax : ttt.x();
            float tNextY = (currIdx & 2) ? tMax : ttt.y();
            float tNextZ = (currIdx & 4) ? tMax : ttt.z();
            float tNext = std::min({tNextX, tNextY, tNextZ});
            
            int physIdx = currIdx ^ ray.signMask;
            if (node->children[physIdx]) {
                fastBypassVoxelTraverseRecursive(node->children[physIdx].get(), tMin, tNext, maxDist, ray, hit);
            }
            
            tMin = tNext;
            currIdx |= ((tNextX <= tNext) ? 1 : 0) | ((tNextY <= tNext) ? 2 : 0) | ((tNextZ <= tNext) ? 4 : 0);
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
                             int maxBounces, bool globalIllumination, bool useLod, bool asyncLoad = false) {
        if (bounces > maxBounces) return globalIllumination ? skylight_ : Eigen::Vector3f::Zero();

        auto hit = voxelTraverse(rayOrig, rayDir, std::numeric_limits<float>::max(), useLod, asyncLoad);
        if (!hit) {
            if (bounces == 0) return skybox_.sampleVector(rayDir);
            return globalIllumination ? skybox_.sampleVector(rayDir) : Eigen::Vector3f::Zero();
        }

        auto obj = hit;
        
        PointType hitPoint;
        PointType normal;
        float t = 0.0f;
        Ray ray(rayOrig, rayDir);
        rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
        
        Eigen::Vector3f objColor = obj->color;
        Material objMat = obj->material;

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
                specColor = W_spec.cwiseProduct(traceRay(hitPoint + n_eff * rayOffset, specDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod, asyncLoad));
            }
            
            Eigen::Vector3f secondColor = Eigen::Vector3f::Zero();
            if (lumSecond > 0.001f) {
                secondColor = W_second.cwiseProduct(traceRay(secondOrigin, secondDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod, asyncLoad));
            }
            
            return finalColor + specColor + secondColor;
        } else {
            float totalLum = lumSpec + lumSecond;
            if (totalLum < 0.0001f) return finalColor;
            
            float pSpec = lumSpec / totalLum;
            float roll = float(rand_r(&rngState)) / float(RAND_MAX);
            
            if (roll < pSpec) {
                Eigen::Vector3f sample = traceRay(hitPoint + n_eff * rayOffset, specDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod, asyncLoad);
                return finalColor + (W_spec / std::max(EPSILON, pSpec)).cwiseProduct(sample);
            } else {
                Eigen::Vector3f sample = traceRay(secondOrigin, secondDir, bounces + 1, rngState, maxBounces, globalIllumination, useLod, asyncLoad);
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
        
        node->setLeaf(true);
    }

    void printStatsRecursive(const OctreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes, size_t& unloaded) const {
        if (!node) return;
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (!node->isLoaded()) {
            unloaded++;
            return;
        }

        if (node->lodData) lodGeneratedNodes++;
        
        size_t pts = node->points.size();
        actualPoints += pts;

        if (node->isLeaf()) {
            leafNodes++;
            maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
            minPointsInLeaf = std::min(minPointsInLeaf, pts);
        } else {
            for (const auto& child : node->children) {
                printStatsRecursive(child.get(), depth + 1, totalNodes, leafNodes, actualPoints, 
                                    maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes, unloaded);
            }
        }
    }

    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;
        if (!node->isLoaded()) return; 

        if (node->isLeaf()) {
            return;
        }

        bool childrenAreLeaves = true;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                optimizeRecursive(node->children[i].get());
                if (!node->children[i]->isLeaf()) {
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
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                node->points = std::move(allPoints);
                for (int i = 0; i < 8; ++i) {
                    node->children[i].reset(nullptr);
                }
                node->setLeaf(true);
                node->setDirty(true);
                
                node->lodData = nullptr;
            }
        }
    }

    void offloadRecursive(OctreeNode* node) {
        if (!node->isLoaded()) return;
        
        size_t subPoints = node->getSubtreePointCount();
        bool fullyLoaded = node->isSubtreeFullyLoaded();
        
        if (subPoints > 0 && (subPoints <= regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
            if (node->isDirty()) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                node->saveRegion();
            }
            node->offload();
            return;
        }

        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) offloadRecursive(node->children[i].get());
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
        
        arr[0] = a0;
        arr[1] = a1;
        arr[2] = a2;
        arr[3] = a3;
        arr[4] = a4;
        arr[5] = a5;
        arr[6] = a6;
        arr[7] = a7;
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

        t = tMin < 0.0f ? tMax : tMin;
        
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

    void collectNodesByObjectIdRecursive(OctreeNode* node, int id,
          std::vector<std::shared_ptr<NodeData>>& results,
          std::unordered_set<std::shared_ptr<NodeData>>& seen) {
        if (!node) return;
        ensureLoaded(node);
        
        for (const auto& pt : node->points) {
            if (pt->isActive() && (id == -1 || pt->objectId == id)) {
                if (seen.insert(pt).second) {
                    results.push_back(pt);
                }
            }
        }
        if (!node->isLeaf()) {
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
public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16) :
            root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
            maxDepth(maxDepth), size(0), skybox_(1024, 1024),
            streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), skybox_(1024, 1024), streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    ~Octree() {
        stopWorkerThread();
    }
    
    Octree(const Octree& other) : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(other.skylight_), backgroundColor_(other.backgroundColor_), autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(other.skybox_), regionTargetPoints_(other.regionTargetPoints_) {
        if (other.root_) root_ = other.root_->clone();
        startWorkerThread();
    }

    Octree(Octree&& other) noexcept : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(std::move(other.skylight_)), backgroundColor_(std::move(other.backgroundColor_)),
            autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(std::move(other.skybox_)), regionTargetPoints_(other.regionTargetPoints_) {
        other.stopWorkerThread();
        root_ = std::move(other.root_);
        
        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
    }

    Octree& operator=(const Octree& other) {
        if (this == &other) return *this;
        
        stopWorkerThread();
        clear();
        
        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        skylight_ = other.skylight_;
        backgroundColor_ = other.backgroundColor_;
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        skybox_ = other.skybox_;
        regionTargetPoints_ = other.regionTargetPoints_;

        if (other.root_) root_ = other.root_->clone();
        
        startWorkerThread();
        return *this;
    }

    Octree& operator=(Octree&& other) noexcept {
        if (this == &other) return *this;

        stopWorkerThread();
        other.stopWorkerThread();

        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        skylight_ = std::move(other.skylight_);
        backgroundColor_ = std::move(other.backgroundColor_);
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        skybox_ = std::move(other.skybox_);
        regionTargetPoints_ = std::move(other.regionTargetPoints_);
        
        root_ = std::move(other.root_);
        
        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
        return *this;
    }

    void enqueueTask(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(taskMutex_);
            taskQueue_.push(std::move(task));
        }
        taskCV_.notify_one();
    }

    void offloadRegions() {
        if (root_) offloadRecursive(root_.get());
    }

    void setAutoOptimize(bool v) { 
        autoOptimize_.store(v); 
    }

    void setSkylight(const Eigen::Vector3f& skylight) { 
        skylight_ = skylight; 
    }

    Eigen::Vector3f getSkylight() const { 
        return skylight_; 
    }

    void setBackgroundColor(const Eigen::Vector3f& color) { 
        backgroundColor_ = color; 
        skybox_.setBackground(color.x(), color.y(), color.z(), 1.0f);
    }

    Eigen::Vector3f getBackgroundColor() const { 
        return backgroundColor_; 
    }

    void setLODFalloff(float rate) {
        lodFalloffRate_ = rate;
        invLodf = 1 / rate;
    }
    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }
    void setMaxDistance(float dist) {
        maxDistance_ = dist;
        keepDistance_ = dist * 1.2;
    }
    void setRegionTargetPoints(size_t points) { regionTargetPoints_ = points; }
    size_t getRegionTargetPoints() const { return regionTargetPoints_; }

    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, 
             float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f) {
        
        Material mat(emittance, roughness, metallic, transmission, ior);
        auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId, mat);
        
        ensureBounds(pointData->getCubeBounds());
        
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        }
        return false;
    }

    void queuedset(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, 
             float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f) {
        enqueueTask([this, data, pos, visible, color, size, active, objectId, emittance, roughness, metallic, transmission, ior]() {
            Material mat(emittance, roughness, metallic, transmission, ior);
            auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId, mat);
            
            ensureBounds(pointData->getCubeBounds());
            
            if (insertRecursive(root_.get(), pointData, 0)) {
                this->size++;
                return;
            }
            return;
        });
    }

    void updateStreaming(const Camera& cam) {
        if (streamingQueued_.exchange(true, std::memory_order_acquire)) return;
        PointType camPos = cam.origin;
        PointType camDir = cam.direction.normalized();
        enqueueTask([this, camPos, camDir]() {
            if (root_) {
                updateStreamingRecursive(root_.get(), camPos, camDir);
            }
            streamingQueued_.store(false, std::memory_order_release);
        });
    }

    bool save(const std::string& filename) {
        if (!root_) return false;

        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x79676733;
        OctreeNode::writeVal(out, magic);
        OctreeNode::writeVal(out, maxDepth);
        OctreeNode::writeVal(out, maxPointsPerNode);
        OctreeNode::writeVal(out, size);
        OctreeNode::writeVal(out, regionTargetPoints_);
        
        OctreeNode::writeVec3(out, skylight_);
        OctreeNode::writeVec3(out, backgroundColor_);
        
        OctreeNode::writeVec3(out, root_->bounds.first);
        OctreeNode::writeVec3(out, root_->bounds.second);

        root_->serialize(out, regionTargetPoints_);
        
        out.close();
        std::cout << "successfully saved grid to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        uint32_t magic;
        OctreeNode::readVal(in, magic);
        if (magic != 0x79676733) {
            std::cerr << "Invalid Octree file format" << std::endl;
            return false;
        }

        OctreeNode::readVal(in, maxDepth);
        OctreeNode::readVal(in, maxPointsPerNode);
        OctreeNode::readVal(in, size);
        OctreeNode::readVal(in, regionTargetPoints_);
        
        OctreeNode::readVec3(in, skylight_);
        OctreeNode::readVec3(in, backgroundColor_);

        PointType minBound, maxBound;
        OctreeNode::readVec3(in, minBound);
        OctreeNode::readVec3(in, maxBound);

        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        root_->deserialize(in, regionTargetPoints_);

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = EPSILON) {
        return findRecursive(root_.get(), pos, tolerance);
    }

    std::shared_ptr<NodeData> findwNode(const PointType& pos, OctreeNode* node, float tolerance = EPSILON) {
        // node = root_.get();
        return findRecursive(node, pos, tolerance);
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

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius, int objectid = -1) {
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
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedupdate(const PointType pos, const T newData) {
        enqueueTask([this, pos, newData]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node);
            if (!pointData) return;
            else {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->data = newData;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool update(const PointType& oldPos, const PointType& newPos, const T& newData, bool newVisible = true, 
                Eigen::Vector3f newColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, float newEmittance = -1.0f, float newRoughness = -1.0f, 
                float newMetallic = -1.0f, float newTransmission = -1.0f, float newIor = -1.0f, float tolerance = EPSILON) {

        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;

        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        
        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->setVisible(newVisible);
        
        if (newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f)) pointData->color = newColor;
        if (newSize > 0) pointData->size = newSize;
        pointData->setActive(newActive);
        pointData->objectId = targetObjId;
        
        if (newEmittance >= 0) {
            pointData->material.emittance = newEmittance;
        }
        if (newRoughness >= 0) { 
            pointData->material.roughness = newRoughness;
        }
        if (newMetallic >= 0) { 
            pointData->material.metallic = newMetallic;
        }
        if (newTransmission >= 0) { 
            pointData->material.transmission = newTransmission;
        }
        if (newIor >= 0) { 
            pointData->material.ior = newIor;
        }
        
        ensureBounds(pointData->getCubeBounds());
        bool res = insertRecursive(root_.get(), pointData, 0);
        
        if(!res) {
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

    void queuedmove(const PointType pos, const PointType newPos) {
        enqueueTask([this, pos, newPos]() {
            auto pointData = find(pos);
            if (!pointData) return;

            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            pointData->position = newPos;
            ensureBounds(pointData->getCubeBounds());

            if (insertRecursive(root_.get(), pointData, 0)) {
                return;
            }
            size--;
            return;
        });
    }

    void queuedupdate(const PointType pos, const PointType newPos, const T newData) {
        enqueueTask([this, pos, newPos, newData]() {
            auto pointData = find(pos);
            if (!pointData) return;
            
            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            
            auto newPointData = std::make_shared<NodeData>(*pointData);
            newPointData->position = newPos;
            newPointData->data = newData;
            
            ensureBounds(newPointData->getCubeBounds());
            if (!insertRecursive(root_.get(), newPointData, 0)) {
                size--;
            }
        });
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
        pointData->color = color;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedsetColor(const PointType& pos, Eigen::Vector3f color, float tolerance = EPSILON) {
        enqueueTask([this, pos, color, tolerance]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node, tolerance);
            if (!pointData) return;
            {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->color = color;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool setEmittance(const PointType& pos, float emittance, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->material.emittance = emittance;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setRoughness(const PointType& pos, float roughness, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->material.roughness = roughness;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setMetallic(const PointType& pos, float metallic, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->material.metallic = metallic;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setTransmission(const PointType& pos, float transmission, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->material.transmission = transmission;
        invalidateLODForPoint(pointData);
        return true;
    }

    std::shared_ptr<NodeData> voxelTraverse(const PointType& origin, const PointType& direction,
                                        float maxDist, bool enableLOD = false, bool asyncLoad = false) {
        std::shared_ptr<NodeData> hit;
        Ray oray(origin, direction);
        
        float tMin, tMax;
        if (rayBoxIntersect(oray, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            voxelTraverseRecursive(root_.get(), tMin, tMax, currentMaxDist, enableLOD, oray, hit, asyncLoad);
        }
        return hit;
    }

    frame renderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 2,
                    int maxBounces = 4, bool globalIllumination = false, bool useLod = true) {
        updateStreaming(cam);
        optimize();
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
                    accumulatedColor += traceRay(origin, rayDir, 0, seed, maxBounces, globalIllumination, useLod, false);
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

    std::shared_ptr<NodeData> fastVoxelTraverse(const Ray& ray, float maxDist) {
        std::shared_ptr<NodeData> hit;
        if (empty()) return hit;
        float tMin, tMax;
        if (rayBoxIntersect(ray, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            fastVoxelTraverseRecursive(root_.get(), tMin, tMax, currentMaxDist, ray, hit);
        }
        return hit;
    }

    std::shared_ptr<NodeData> fastBypassVoxelTraverse(const Ray& ray, float maxDist) {
        std::shared_ptr<NodeData> hit;
        if (empty()) return hit;
        float tMin, tMax;
        if (rayBoxIntersect(ray, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            float currentMaxDist = maxDist;
            fastBypassVoxelTraverseRecursive(root_.get(), tMin, tMax, currentMaxDist, ray, hit);
        }
        return hit;
    }

    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB) {
        updateStreaming(cam);
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
        const float minVisibility = 0.1f;
        float maxmin = (maxDistance_ - lodMinDistance_);
        float invMaxMin = 1 / maxmin;
        
        #pragma omp parallel for schedule(dynamic, 128) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pidx = (y * width + x);
                int idx = pidx * 3;

                float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                
                PointType rayDir = dir + (right * px) + (up * py);
                rayDir.normalize();

                Eigen::Vector3f color = skybox_.sampleVector(rayDir);
                Ray ray(origin, rayDir);
                std::shared_ptr<NodeData> hit;
                if (x % 10 == 0 && y % 10 == 0) hit = fastVoxelTraverse(ray, maxDistance_);
                else hit = fastBypassVoxelTraverse(ray, maxDistance_);
                if (hit != nullptr) {
                    auto obj = hit;
                    
                    float t = 0.0f;
                    PointType normal, hitPoint;

                    rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
                    color = obj->color;
                    Material objMat = obj->material;
                    
                    if (objMat.emittance > 0.0f) {
                        color = color * objMat.emittance;
                    } else {
                        float diffuse = std::max(0.0f, normal.dot(globalLightDir));
                        float ambient = 0.35f;
                        float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                        color = color * intensity;
                    }
                    
                    float fogFactor = std::clamp((maxDistance_ - t) * invMaxMin, minVisibility, 1.0f);
                    
                    color = color * fogFactor + skybox_.sampleVector(rayDir) * (1.0f - fogFactor);
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
        updateStreaming(cam);
        auto startTime = std::chrono::high_resolution_clock::now();
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

                Eigen::Vector3f color = skybox_.sampleVector(rayDir);
                Ray ray(origin, rayDir);
                auto hit = fastVoxelTraverse(ray, maxDistance_);
                if (hit != nullptr) {
                    auto obj = hit;
                    
                    float t = 0.0f;
                    PointType normal, hitPoint;

                    rayCubeIntersect(ray, obj.get(), t, normal, hitPoint);
                    color = obj->color;
                    Material objMat = obj->material;
                    
                    if (objMat.emittance > 0.0f) {
                        color = color * objMat.emittance;
                    } else {
                        float diffuse = std::max(0.0f, normal.dot(globalLightDir));
                        float ambient = 0.35f;
                        float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                        color = color * intensity;
                    }
                    
                    float fogFactor = std::clamp((maxDistance_ - t) / (maxDistance_ - fogStart), minVisibility, 1.0f);
                    color = color * fogFactor + skybox_.sampleVector(rayDir) * (1.0f - fogFactor);
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

                        Eigen::Vector3f pbrColor = traceRay(origin, rayDir, 0, seed, maxBounces, globalIllumination, useLod, true);
                        
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

                if (neighbor == nullptr || !neighbor->isActive() || neighbor->objectId != node->objectId) {
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
                if(neighbor && neighbor->objectId == objectId && neighbor->isActive()) {
                    Material nMat = neighbor->material;
                    if (nMat.transmission < 0.01f) {
                        hiddenSides++;
                    }
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
        return true;
    }

    void optimize() {
        if (root_) {
            optimizeRecursive(root_.get());
            generateLODs();
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
        size_t unloaded = 0;

        printStatsRecursive(root_.get(), 0, totalNodes, leafNodes, actualPoints, 
                            maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes, unloaded);

        if (leafNodes == 0) minPointsInLeaf = 0;
        double avgPointsPerLeaf = totalNodes > 0 ? (double)actualPoints / totalNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16); 

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
        os << "  Unloaded Nodes    : " << unloaded << "\n";
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
        if (root_) {
            clearNode(root_.get());
        }
        PointType minBound = root_ ? root_->bounds.first : PointType::Zero();
        PointType maxBound = root_ ? root_->bounds.second : PointType::Zero();
        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        
        size = 0;
    }
    
    void getLoadedStatsSafe(const OctreeNode* node, size_t& loadedNodes, size_t& loadedPoints) const {
        if (!node) return;
        loadedNodes++;
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLoaded()) return;
        
        loadedPoints += node->points.size();
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    getLoadedStatsSafe(node->children[i].get(), loadedNodes, loadedPoints);
                }
            }
        }
    }

    size_t getEstimatedMemoryUsageMB() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_.get(), loadedNodes, loadedPoints);
        
        size_t nodeMem = loadedNodes * sizeof(OctreeNode);
        size_t pointMem = loadedPoints * (sizeof(NodeData) + sizeof(std::shared_ptr<NodeData>));
        
        return (nodeMem + pointMem) / (1024 * 1024);
    }

    size_t getLoadedPointCount() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_.get(), loadedNodes, loadedPoints);
        return loadedPoints;
    }
};

#endif
