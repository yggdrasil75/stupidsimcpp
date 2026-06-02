#pragma once
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <memory>

namespace Grid{

constexpr int Dim = 3;

static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
static constexpr uint8_t STATIC_BIT = 1 << 7;

static constexpr uint8_t LEAF_BIT = 1 << 0;
static constexpr uint8_t LOADED_BIT = 1 << 1;
static constexpr uint8_t DIRTY_BIT = 1 << 2;
static constexpr uint8_t LOADQUEUED = 1 << 3;
static constexpr uint8_t SAVEDQUEUED = 1 << 4;
static constexpr uint8_t KEEPLOADED_BIT = 1 << 5;
static constexpr uint8_t FAT_BIT = 1 << 6;

static constexpr uint8_t OBJ_ALLOW_PARTIAL_UNLOAD_BIT = 1 << 0;

static constexpr uint8_t WORKER_ON = 1 << 0;
static constexpr uint8_t AUTO_OPTIMIZE = 1 << 1;
static constexpr uint8_t QUEUE_STREAMING = 1 << 2;
static constexpr uint8_t PHYSICS_COLLIDER_DIRTY = 1 << 3;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
using PointType = Eigen::Matrix<float, Dim, 1>;
using BoundingBox = std::pair<PointType, PointType>;
namespace fs = std::filesystem;


template<typename T, typename IndexType = uint16_t>
struct PhysicsState_ {
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f force{0.0f, 0.0f, 0.0f};
    float density = 1.0f;
    float pressure = 0.0f;
};

struct SPHIntegratePC {
    float dt;
    float velocityDamping;
    uint32_t numParticles;
};

struct SPHDensityPC {
    float h;
    float h2;
    float poly6_k;
    float restDensity;
    float gasConstant;
    uint32_t numParticles;
};

struct SPHForcePC {
    float h;
    float spiky_k;
    float visc_l_k;
    float viscosity;
    float gravX;
    float gravY;
    float gravZ;
    float gravStrength;
    float gravCX;
    float gravCY;
    float gravCZ;
    uint32_t useGravityPoint;
    uint32_t numParticles;
    float airDensity;
};

template<typename T, typename IndexType = uint16_t>
struct NodeData_ {
    T data;
    PointType position;
    uint16_t objectId;
    float size;
    Eigen::Vector3f color;
    uint16_t renderMatIdx;
    uint16_t physMatIdx;
    std::atomic<uint8_t> flags;
    PhysicsState_<T, IndexType> physics;

    NodeData_(const T& data, const PointType& pos, bool visible, const Eigen::Vector3f& color, float size = 0.01f,
                bool active = true, int objectId = -1, uint16_t rIdx = 0, uint16_t pIdx = 0, bool staticbit = 0) 
            : data(data), position(pos), objectId(objectId), size(size), 
                color(color), renderMatIdx(rIdx), physMatIdx(pIdx), flags(0) {
        setActive(active);
        setVisible(visible);
        setStatic(staticbit);
    }
    
    NodeData_() : objectId(-1), size(0.0f), color(Eigen::Vector3f::Zero()), renderMatIdx(0), physMatIdx(0), flags(0) {}

    NodeData_(const NodeData_& other) : data(other.data), position(other.position), objectId(other.objectId), size(other.size),
            color(other.color), renderMatIdx(other.renderMatIdx), physMatIdx(other.physMatIdx), flags(other.flags.load(std::memory_order_relaxed)), physics(other.physics) {}

    NodeData_& operator=(const NodeData_& other) {
        if (this != &other) {
            data = other.data;
            position = other.position;
            objectId = other.objectId;
            size = other.size;
            color = other.color;
            renderMatIdx = other.renderMatIdx;
            physMatIdx = other.physMatIdx;
            physics = other.physics;
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

template<typename GasT>
struct EulerianGasState_ {
    GasT data{};
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    float density = 0.0f;
    float pressure = 0.0f;
    
    int objectId = -1;
    uint16_t renderMatIdx = 0;
    Eigen::Vector3f color{1.0f, 1.0f, 1.0f};
};

template<typename T, typename GasT = float, typename IndexType = uint16_t>
struct OctreeNode_ {
    BoundingBox bounds;
    EulerianGasState_<GasT> gasState;

    std::vector<std::shared_ptr<NodeData_<T, IndexType>>> points;
    std::vector<std::unique_ptr<OctreeNode_<T, GasT, IndexType>>> children;
    PointType center;
    float nodeSize;
    std::atomic<uint8_t> flags;
    
    mutable std::shared_ptr<NodeData_<T, IndexType>> lodData;
    mutable std::shared_mutex nodeMutex;

    OctreeNode_(const PointType& min, const PointType& max, bool fat = false) : bounds(min, max), flags(0), lodData(nullptr) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        setFat(fat);
        children.resize(fat ? 32768 : 8); 
        for (auto& child : children) {
            child = nullptr;
        }
        center = (bounds.first + bounds.second) * 0.5;
        nodeSize = (bounds.second - bounds.first).norm();
    }

    std::unique_ptr<OctreeNode_<T, GasT, IndexType>> clone() const {
        auto newNode = std::make_unique<OctreeNode_<T, GasT, IndexType>>(bounds.first, bounds.second);
        newNode->flags.store(flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
        
        newNode->points = points;
        newNode->center = center;
        newNode->nodeSize = nodeSize;
        newNode->lodData = lodData;
        newNode->gasState = gasState;
        
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
    bool isFat() const {
        return flags.load(std::memory_order_relaxed) & FAT_BIT;
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
    void setFat(bool v) {
        if (v) flags.fetch_or(FAT_BIT, std::memory_order_relaxed);
        else flags.fetch_and(~FAT_BIT, std::memory_order_relaxed);
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

    std::string getRegionPath(std::string StoragePath) const {
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

    template<typename D>
    static void serializeFieldData(std::ofstream& out, const D& data) {
        if constexpr (is_shared_ptr<D>::value) {
            bool hasData = (data != nullptr);
            writeVal(out, hasData);
            if (hasData) data->serialize(out);
        } else if constexpr (std::is_pointer_v<D>) {
            bool hasData = (data != nullptr);
            writeVal(out, hasData);
            if (hasData) data->serialize(out);
        } else if constexpr (std::is_class_v<D>) {
            data.serialize(out);
        } else {
            writeVal(out, data);
        }
    }

    template<typename D>
    static void deserializeFieldData(std::ifstream& in, D& data) {
        if constexpr (is_shared_ptr<D>::value) {
            bool hasData;
            readVal(in, hasData);
            if (hasData) {
                using ElemType = typename D::element_type;
                data = ElemType::deserialize(in);
            } else {
                data = nullptr;
            }
        } else if constexpr (std::is_pointer_v<D>) {
            bool hasData;
            readVal(in, hasData);
            if (hasData) {
                using ElemType = std::remove_pointer_t<D>;
                data = ElemType::deserialize(in);
            } else {
                data = nullptr;
            }
        } else if constexpr (std::is_class_v<D>) {
            data = D::deserialize(in);
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

        serializeFieldData(out, gasState.data);
        writeVec3(out, gasState.velocity);
        writeVal(out, gasState.density);
        writeVal(out, gasState.pressure);
        writeVal(out, gasState.objectId);
        writeVal(out, gasState.renderMatIdx);
        writeVec3(out, gasState.color);

        writeVal(out, points.size());
        for (const auto& pt : points) {
            serializeFieldData(out, pt->data);
            writeVec3(out, pt->position);
            writeVal(out, pt->objectId);
            writeVal(out, pt->flags.load(std::memory_order_relaxed));
            writeVal(out, pt->size);
            writeVec3(out, pt->color);
            writeVal(out, pt->renderMatIdx);
            writeVal(out, pt->physMatIdx);
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

        deserializeFieldData(in, gasState.data);
        readVec3(in, gasState.velocity);
        readVal(in, gasState.density);
        readVal(in, gasState.pressure);
        readVal(in, gasState.objectId);
        readVal(in, gasState.renderMatIdx);
        readVec3(in, gasState.color);

        size_t pointCount;
        readVal(in, pointCount);
        points.reserve(pointCount);
        for (size_t i = 0; i < pointCount; ++i) {
            auto pt = std::make_shared<NodeData_<T, IndexType>>();
            deserializeFieldData(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            uint8_t f;
            readVal(in, f);
            pt->flags.store(f, std::memory_order_relaxed);
            readVal(in, pt->size);
            readVec3(in, pt->color);
            readVal(in, pt->renderMatIdx);
            readVal(in, pt->physMatIdx);
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
                    children[i] = std::make_unique<OctreeNode_<T, GasT, IndexType>>(childMin, childMax);
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

        serializeFieldData(out, gasState.data);
        writeVec3(out, gasState.velocity);
        writeVal(out, gasState.density);
        writeVal(out, gasState.pressure);
        writeVal(out, gasState.objectId);
        writeVal(out, gasState.renderMatIdx);
        writeVec3(out, gasState.color);

        writeVal(out, points.size());
        for (const auto& pt : points) {
            serializeFieldData(out, pt->data);
            writeVec3(out, pt->position);
            writeVal(out, pt->objectId);
            writeVal(out, pt->flags.load(std::memory_order_relaxed));
            writeVal(out, pt->size);
            writeVec3(out, pt->color);
            writeVal(out, pt->renderMatIdx);
            writeVal(out, pt->physMatIdx);
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

        deserializeFieldData(in, gasState.data);
        readVec3(in, gasState.velocity);
        readVal(in, gasState.density);
        readVal(in, gasState.pressure);
        readVal(in, gasState.objectId);
        readVal(in, gasState.renderMatIdx);
        readVec3(in, gasState.color);

        size_t pointCount;
        readVal(in, pointCount);
        points.reserve(pointCount);
        for (size_t i = 0; i < pointCount; ++i) {
            auto pt = std::make_shared<NodeData_<T, IndexType>>();
            deserializeFieldData(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            uint8_t f;
            readVal(in, f);
            pt->flags.store(f, std::memory_order_relaxed);
            readVal(in, pt->size);
            readVec3(in, pt->color);
            readVal(in, pt->renderMatIdx);
            readVal(in, pt->physMatIdx);
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
                    children[i] = std::make_unique<OctreeNode_<T, GasT, IndexType>>(childMin, childMax);
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

template<typename T, typename IndexType = uint16_t>
struct RayHit_ {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    float distance;
    PointType normal;
    PointType hitPoint;
};

}