#pragma once
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <shared_mutex>
#include <memory>
#include <algorithm>
#include <mutex>
#include <type_traits>
#include <filesystem>

namespace Grid{

constexpr int Dim = 3;

//point flag bits
static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
static constexpr uint8_t STATIC_BIT = 1 << 7;

//node flag bits
static constexpr uint8_t LEAF_BIT = 1 << 0;
static constexpr uint8_t LOADED_BIT = 1 << 1;
static constexpr uint8_t DIRTY_BIT = 1 << 2;
static constexpr uint8_t LOADQUEUED = 1 << 3;
static constexpr uint8_t SAVEDQUEUED = 1 << 4;
static constexpr uint8_t KEEPLOADED_BIT = 1 << 5;
static constexpr uint8_t FAT_BIT = 1 << 6;

//object flag bits
static constexpr uint8_t OBJ_ALLOW_PARTIAL_UNLOAD_BIT = 1 << 0;

//grid flag bits
static constexpr uint8_t WORKER_ON = 1 << 0;
static constexpr uint8_t AUTO_OPTIMIZE = 1 << 1;
static constexpr uint8_t QUEUE_STREAMING = 1 << 2;
static constexpr uint8_t PHYSICS_COLLIDER_DIRTY = 1 << 3;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
using PointType = Eigen::Matrix<float, Dim, 1>;
using BoundingBox = std::pair<PointType, PointType>;

template<typename T, typename IndexType = uint16_t>
struct OctreeNode_;

inline uint16_t mortonEncodeFatNode(uint8_t x, uint8_t y, uint8_t z);
inline void mortonDecodeFatNode(uint16_t morton, uint8_t& x, uint8_t& y, uint8_t& z);

enum class BodyType : uint8_t {
    STATIC = 0,
    RIGID = 1,
    SOFT = 2,
    FLUID = 3
};

enum class meshMode : uint8_t {
    NAIVE = 0, // each voxel = 12 tris
    GREEDY = 1, // adjacent voxels get merged and then meshed
    MARCHINGCUBES = 2, //marching cubes using the lookup in basicdefines
    NAIVEMARCHING = 3, //marching cubes without the LUT. because why not.
    SURFACENET = 4, // use this for terrain. create tris using exposed voxel faces
    DUALCONTOUR = 5, // uses normals to generate meshes
    MANIFOLDCONTOUR = 6, // something something normals and better weird shapes (need to actually read the paper)
    OCCUPANCY = 7, // this uses a forward pass model network trained ahead of time to guess the mesh.
    CUBICMARCHING = 8, // marching cubes, but with bisections from catmull-rom tricubic sdf, making continuous normals more accurately
    TRANSVOXEL = 9, // for lod only. I dont know what this is, just that its mentioned as lod only all over.
    DUALMARCHING = 10, //dual marching cubes, combine dual contour and marching cubes. can this be used to expand cubic? dual cubic marching manifold cubes?
};

inline uint8_t getOctant(const PointType& point, const PointType& center) {
    return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
}

template<typename T, typename IndexType = uint16_t>
uint16_t getFatCellIndex(const PointType& point, const OctreeNode_<T, IndexType>* node) {
    BoundingBox bounds = node->bounds();
    const PointType& rootMin = bounds.first;
    PointType step = (bounds.second - rootMin) / 32.0f;
    uint8_t x = static_cast<uint8_t>(std::clamp((point[0] - rootMin[0]) / step[0], 0.0f, 31.0f));
    uint8_t y = static_cast<uint8_t>(std::clamp((point[1] - rootMin[1]) / step[1], 0.0f, 31.0f));
    uint8_t z = static_cast<uint8_t>(std::clamp((point[2] - rootMin[2]) / step[2], 0.0f, 31.0f));
    return mortonEncodeFatNode(x, y, z);
}

template<typename V>
inline void writeVal(std::ofstream& out, const V& val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(V));
}

template<typename V>
inline void readVal(std::ifstream& in, V& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(V));
}

inline void writeVec3(std::ofstream& out, const Eigen::Vector3f& vec) {
    writeVal(out, vec.x());
    writeVal(out, vec.y());
    writeVal(out, vec.z());
}

inline void readVec3(std::ifstream& in, Eigen::Vector3f& vec) {
    float x, y, z;
    readVal(in, x);
    readVal(in, y);
    readVal(in, z);
    vec = Eigen::Vector3f(x, y, z);
}

inline uint16_t mortonEncodeFatNode(uint8_t x, uint8_t y, uint8_t z) {
#ifdef SSE
    return _pdep_u32(x, 0x49249) | (_pdep_u32(y, 0x49249) << 1) | (_pdep_u32(z, 0x49249) << 2);
#else
    uint32_t xx = x & 0x1F;
    uint32_t yy = y & 0x1F;
    uint32_t zz = z & 0x1F;
    xx = (xx | (xx << 8)) & 0x100F;
    yy = (yy | (yy << 8)) & 0x100F;
    zz = (zz | (zz << 8)) & 0x100F;
    xx = (xx | (xx << 4)) & 0x10C3;
    yy = (yy | (yy << 4)) & 0x10C3;
    zz = (zz | (zz << 4)) & 0x10C3;
    xx = (xx | (xx << 2)) & 0x1249;
    yy = (yy | (yy << 2)) & 0x1249;
    zz = (zz | (zz << 2)) & 0x1249;
    return static_cast<uint16_t>(xx | (yy << 1) | (zz << 2));
#endif
}

inline void mortonDecodeFatNode(uint16_t morton, uint8_t& x, uint8_t& y, uint8_t& z) {
#ifdef SSE
    x = static_cast<uint8_t>(_pext_u32(morton, 0x49249));
    y = static_cast<uint8_t>(_pext_u32(morton >> 1, 0x49249));
    z = static_cast<uint8_t>(_pext_u32(morton >> 2, 0x49249));
#else
    auto compact = [](uint32_t v) -> uint8_t {
        v &= 0x1249;
        v = (v ^ (v >> 2)) & 0x10C3;
        v = (v ^ (v >> 4)) & 0x100F;
        v = (v ^ (v >> 8)) & 0x001F;
        return static_cast<uint8_t>(v);
    };
    x = compact(morton);
    y = compact(morton >> 1);
    z = compact(morton >> 2);
#endif
}

struct vec3fh {
    size_t operator()(const Eigen::Vector3f& v) const {
        size_t h1 = std::hash<float>()(v.x());
        size_t h2 = std::hash<float>()(v.y());
        size_t h3 = std::hash<float>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct vec3ih {
    size_t operator()(const Eigen::Vector3i& v) const {
        size_t h1 = std::hash<int>()(v.x());
        size_t h2 = std::hash<int>()(v.y());
        size_t h3 = std::hash<int>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Material_ {
    uint32_t chromaticity;
    float roughness;
    float reflective;
    float ior;
    uint32_t absorption;

    Material_(uint32_t e, float r, float m, float i, uint32_t a) : chromaticity(e), roughness(r), reflective(m), ior(i), absorption(a) {}

    bool operator==(const Material_& o) const {
        return chromaticity == o.chromaticity && roughness == o.roughness && reflective == o.reflective &&
               ior == o.ior && absorption == o.absorption;
    }

    bool operator<(const Material_& o) const {
        if (chromaticity != o.chromaticity) return chromaticity < o.chromaticity;
        if (roughness != o.roughness) return roughness < o.roughness;
        if (reflective != o.reflective) return reflective < o.reflective;
        if (absorption != o.absorption) return absorption < o.absorption;
        return ior < o.ior;
    }
    float dist(const Material_& o) const {
        float dr = roughness - o.roughness;
        float dm = reflective - o.reflective;
        float di = ior - o.ior;
        float chromPenalty = (chromaticity != o.chromaticity) ? 1.0f : 0.0f;
        float absPenalty = (absorption != o.absorption) ? 0.5f : 0.0f;
        return dr*dr + dm*dm + di*di + chromPenalty + absPenalty;
    }
};

struct materialHash {
    std::size_t operator()(const Material_& m) const {
        std::size_t h = std::hash<uint32_t>()(m.chromaticity);
        h ^= std::hash<float>()(m.roughness) << 1;
        h ^= std::hash<float>()(m.reflective) << 2;
        h ^= std::hash<float>()(m.ior) << 3;
        h ^= std::hash<uint32_t>()(m.absorption) << 4;
        return h;
    }
};

struct ExtendedMaterial_ {
    float dispersionB = 0.0f;
    Eigen::Vector3f emissionPeak = Eigen::Vector3f::Zero();
    Eigen::Vector3f emissionBandwidth = Eigen::Vector3f::Zero();
};

struct PhysicsMaterial_ {
    BodyType type = BodyType::STATIC;
    float mass = 1.0f;
    float restitution = 1.0f;
    float density = 1.0f;
    
    bool operator==(const PhysicsMaterial_& o) const {
        return type == o.type && mass == o.mass && restitution == o.restitution && density == o.density;
    }

    float dist(const PhysicsMaterial_& o) const {
        float dm = mass - o.mass;
        float dr = restitution - o.restitution;
        float dd = density - o.density;
        float typePenalty = (type != o.type) ? 10.0f : 0.0f;
        return dm*dm + dr*dr + dd*dd + typePenalty;
    }
};

struct physicsMatHash {
    std::size_t operator()(const PhysicsMaterial_& m) const {
        std::size_t h = std::hash<uint8_t>()(static_cast<uint8_t>(m.type));
        h ^= std::hash<float>()(m.mass) << 1;
        h ^= std::hash<float>()(m.restitution) << 2;
        h ^= std::hash<float>()(m.density) << 3;
        return h;
    }
};

template<typename IndexType = uint16_t>
struct vertex {
    IndexType cIdx;
    PointType pos;
};

struct tri {
    size_t a, b, c;
};

template<typename IndexType = uint16_t>
struct mesh {
    std::vector<vertex<IndexType>> vertices;
    std::vector<tri> tris;
};

template<typename T, typename IndexType = uint16_t>
struct PhysicsState_ {
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f force{0.0f, 0.0f, 0.0f};
    float pressure = 0.0f;
};

template<typename T, typename IndexType = uint16_t>
struct GridObject_ {
    int id;
    std::atomic<uint8_t> flags;
    PointType centerPosition = PointType::Zero();
    std::vector<Eigen::Vector3f> colorLookup;
    std::unordered_map<Eigen::Vector3f, IndexType, vec3fh> colorMap;
    std::vector<float> transmissionTable;
    std::unordered_map<float, uint8_t> transmissionMap;
    std::vector<Material_> renderMaterials;
    std::vector<ExtendedMaterial_> extendedRenderMaterials;
    std::unordered_map<Material_, IndexType, materialHash> renderMatMap;
    std::vector<PhysicsMaterial_> physicsMaterials;
    std::unordered_map<PhysicsMaterial_, IndexType, physicsMatHash> physicsMatMap;
    std::vector<PointType> relativeVoxels;
    mutable std::shared_mutex objMutex;

    mesh<IndexType> objMesh;

    GridObject_(int objId = -1) : id(objId), flags(OBJ_ALLOW_PARTIAL_UNLOAD_BIT) {
        transmissionTable.push_back(0.0f);
        transmissionMap[0.0f] = 0;
        transmissionTable.push_back(1.0f);
        transmissionMap[1.0f] = 1;
    }

    bool isPartialUnloadAllowed() const {
        return flags.load(std::memory_order_relaxed) & OBJ_ALLOW_PARTIAL_UNLOAD_BIT;
    }

    void setPartialUnloadAllowed(bool v) {
        if (v) flags.fetch_or(OBJ_ALLOW_PARTIAL_UNLOAD_BIT, std::memory_order_relaxed);
        else flags.fetch_and(~OBJ_ALLOW_PARTIAL_UNLOAD_BIT, std::memory_order_relaxed);
    }

    IndexType getOrAddColorIndex(const Eigen::Vector3f& color) {
        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto a = colorMap.find(color);
            if (a != colorMap.end()) {
                return a->second;
            }
        }

        if (colorLookup.size() < std::numeric_limits<IndexType>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = colorMap.find(color);
            if (a != colorMap.end()) {
                return a->second;
            }
            IndexType newIndex = static_cast<IndexType>(colorLookup.size());
            colorLookup.push_back(color);
            colorMap[color] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            IndexType bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (IndexType i = 0; i < static_cast<IndexType>(colorLookup.size()); ++i) {
                float distSq = (colorLookup[i] - color).squaredNorm();
                if (distSq < dist) {
                    dist = distSq;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    IndexType getOrAddRenderMaterial(const Material_& renderMat, const ExtendedMaterial_& extMat = {}) {
        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto a = renderMatMap.find(renderMat);
            if (a != renderMatMap.end()) {
                return a->second;
            }
        }

        if (renderMaterials.size() < std::numeric_limits<IndexType>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = renderMatMap.find(renderMat);
            if (a != renderMatMap.end()) {
                return a->second;
            }
            IndexType newIndex = static_cast<IndexType>(renderMaterials.size());
            renderMaterials.push_back(renderMat);
            extendedRenderMaterials.push_back(extMat);
            renderMatMap[renderMat] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            IndexType bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (IndexType i = 0; i < static_cast<IndexType>(renderMaterials.size()); ++i) {
                float dist2 = renderMaterials[i].dist(renderMat);
                if (dist2 < dist) {
                    dist = dist2;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    IndexType getOrAddPhysicsMaterial(const PhysicsMaterial_& physMat) {
        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto a = physicsMatMap.find(physMat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
        }

        if (physicsMaterials.size() < std::numeric_limits<IndexType>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = physicsMatMap.find(physMat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
            IndexType newIndex = static_cast<IndexType>(physicsMaterials.size());
            physicsMaterials.push_back(physMat);
            physicsMatMap[physMat] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            IndexType bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (IndexType i = 0; i < static_cast<IndexType>(physicsMaterials.size()); ++i) {
                float dist2 = physicsMaterials[i].dist(physMat);
                if (dist2 < dist) {
                    dist = dist2;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    uint8_t getOrAddTransmission(float t) {
        t = std::clamp(t, 0.0f, 1.0f);

        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto it = transmissionMap.find(t);
            if (it != transmissionMap.end()) return it->second;
        }

        {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto it = transmissionMap.find(t);
            if (it != transmissionMap.end()) return it->second;

            if (transmissionTable.size() < (256 - 2)) {
                uint8_t idx = static_cast<uint8_t>(2 + transmissionTable.size());
                transmissionTable.push_back(t);
                transmissionMap[t] = idx;
                return idx;
            }
        }
        std::shared_lock<std::shared_mutex> readLock(objMutex);
        uint8_t best = 2;
        float bestDist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < transmissionTable.size(); ++i) {
            float d = std::abs(transmissionTable[i] - t);
            if (d < bestDist) {
                bestDist = d;
                best = static_cast<uint8_t>(2 + i);
            }
        }
        return best;
    }

    Eigen::Vector3f getColor(IndexType idx) const {
        return colorLookup[idx];
    }

    Material_ getRenderMaterial(IndexType idx) const {
        return renderMaterials[idx];
    }
    
    ExtendedMaterial_ getExtendedMaterial(IndexType idx) const {
        return extendedRenderMaterials[idx];
    }

    PhysicsMaterial_ getPhysicsMaterial(IndexType idx) const {
        return physicsMaterials[idx];
    }

    float getTransmission(uint8_t idx) const {
        return transmissionTable[idx];
    }

};

template<typename T, typename IndexType = uint16_t>
struct NodeData_ {
    T data;
    PointType position;
    uint16_t objectId;
    Eigen::half size;
    IndexType colorIdx;
    IndexType renderMatIdx;
    IndexType physMatIdx;
    uint8_t transmissionIdx;
    std::atomic<uint8_t> flags;
    PhysicsState_<T, IndexType> physics;
    
    NodeData_(const T& data, const PointType& pos, bool visible, const IndexType colorIdx, float size = 0.01f, bool active = true, IndexType objectId = -1, 
              IndexType rIdx = 0, IndexType pIdx = 0, uint8_t tIdx = 0, bool staticBit = false) : 
               data(data), position(pos), objectId(objectId), size(Eigen::half(size)), colorIdx(colorIdx), renderMatIdx(rIdx),
               physMatIdx(pIdx), transmissionIdx(tIdx) {
        setActive(active);
        setVisible(visible);
        setStatic(staticBit);
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

template<typename T, typename IndexType>
struct OctreeNode_ {
    std::vector<std::shared_ptr<NodeData_<T, IndexType>>> points;
    std::vector<std::unique_ptr<OctreeNode_<T, IndexType>>> children;
    PointType center;
    Eigen::half nodeSize;
    std::atomic<uint8_t> flags;

    mutable std::shared_ptr<NodeData_<T, IndexType>> lodData;
    mutable std::shared_mutex nodeMutex;
    
    OctreeNode_(const PointType& min, const PointType& max, bool fat = false) : flags(0), lodData(nullptr) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        setFat(fat);
        children.resize(fat ? 65536 : 8); 
        for (auto& child : children) {
            child = nullptr;
        }
        center = (min + max) * 0.5;
        nodeSize = Eigen::half((max - min).norm());
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
        float halfsize = static_cast<float>(nodeSize) * 0.5f;
        return (point[0] >= center[0] - halfsize && point[0] <= center[0] + halfsize &&
                point[1] >= center[1] - halfsize && point[1] <= center[1] + halfsize &&
                point[2] >= center[2] - halfsize && point[2] <= center[2] + halfsize);
    }

    BoundingBox bounds() const {
        float halfsize = static_cast<float>(nodeSize) * 0.5f;
        PointType hs = PointType::Constant(halfsize);
        return BoundingBox(center - hs, center + hs);
    }

    bool isEmpty() const {
        if (!points.empty()) return false;
        if (!isLeaf()) {
            for (auto& child : children) {
                if (!child) continue;
                if (!child->isEmpty()) return false;
            }
        }
        return true;
    }

    std::ostringstream getRegionName(const PointType& parentCenter) const {
        std::ostringstream oss;
        if (!isFat()) {
            oss << static_cast<int>(Grid::getOctant(center, parentCenter));
        } else {
            float step = static_cast<float>(nodeSize) / 16.0f;
            uint8_t x = static_cast<uint8_t>(std::clamp((center[0] - parentCenter[0]) / step, 0.0f, 31.0f));
            uint8_t y = static_cast<uint8_t>(std::clamp((center[1] - parentCenter[1]) / step, 0.0f, 31.0f));
            uint8_t z = static_cast<uint8_t>(std::clamp((center[2] - parentCenter[2]) / step, 0.0f, 31.0f));
            oss << mortonEncodeFatNode(x, y, z);
        }
        return oss;
    }

    std::string getRegionPath(const std::string& storagePath, const PointType* parentCenter = nullptr) const {
        std::filesystem::path p(storagePath);
        if (parentCenter != nullptr) {
            p /= getRegionName(*parentCenter);
        }
        std::error_code ec;
        std::filesystem::create_directories(p, ec);
        p /= "data.region";
        return p.string();
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
            for (auto& child : children) {
                if (!child) continue;
                count += child->getSubtreePointCount();
            }
        }
        return count;
    }
    
    bool isSubtreeFullyLoaded() const {
        if (!isLoaded()) return false;
        if (!isLeaf()) {
            for (auto& child : children) {
                if (!child) continue;
                if (!child->isSubtreeFullyLoaded()) return false;
            }
        }
        return true;
    }

    void serializeSubtree(std::ofstream& out) const {
        writeVal(out, flags.load(std::memory_order_relaxed));

        writeVal(out, points.size());
        for (const auto& pt : points) {
            serializeFieldData(out, pt->data);
            writeVec3(out, pt->position);
            writeVal(out, pt->objectId);
            writeVal(out, pt->flags.load(std::memory_order_relaxed));
            writeVal(out, pt->size);
            writeVal(out, pt->colorIdx);
            writeVal(out, pt->renderMatIdx);
            writeVal(out, pt->physMatIdx);
            writeVal(out, pt->transmissionIdx);
        }

        if (!isLeaf()) {
            if (isFat()) {
                uint16_t activeChildren = 0;
                for (int i = 0; i < 65536; ++i) if (children[i]) activeChildren++;
                writeVal(out, activeChildren);
                for (uint16_t i = 0; i < 65536; ++i) {
                    if (children[i]) {
                        writeVal(out, i);
                        children[i]->serializeSubtree(out);
                    }
                }
            } else {
                uint8_t childMask = 0;
                for (int i = 0; i < 8; ++i) if (children[i]) childMask |= (1 << i);
                writeVal(out, childMask);
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->serializeSubtree(out);
                }
            }
        }
    }

    void serialize(std::ofstream& out, size_t regionTargetPoints, const std::string& storagePath, const PointType* parentCenter = nullptr) {
        bool offloaded = !isLoaded();
        size_t subPoints = offloaded ? 0 : getSubtreePointCount();

        bool isRegion = offloaded || (subPoints > 0 && (subPoints <= regionTargetPoints || isLeaf()) && isSubtreeFullyLoaded());

        writeVal(out, isRegion);

        if (isRegion) {
            if (!offloaded && isDirty()) saveRegion(regionTargetPoints, storagePath, parentCenter);
            return;
        }
        serializeSubtree(out);
    }

    void deserializeSubtree(std::ifstream& in) {
        uint8_t f1;
        readVal(in, f1);
        flags.store(f1, std::memory_order_relaxed);

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
            readVal(in, pt->colorIdx);
            readVal(in, pt->renderMatIdx);
            readVal(in, pt->physMatIdx);
            readVal(in, pt->transmissionIdx);
            points.push_back(pt);
        }

        if (!isLeaf()) {
            if (isFat()) {
                uint16_t activeChildren;
                readVal(in, activeChildren);
                for (uint16_t k = 0; k < activeChildren; ++k) {
                    uint16_t i;
                    readVal(in, i);
                    PointType childMin, childMax;
                    const PointType& rootMin = bounds().first;
                    PointType step = (bounds().second - rootMin) / 32.0f;
                    uint8_t x, y, z;
                    mortonDecodeFatNode(i, x, y, z);
                    childMin = rootMin + PointType(x * step[0], y * step[1], z * step[2]);
                    childMax = childMin + step;
                    children[i] = std::make_unique<OctreeNode_<T, IndexType>>(childMin, childMax);
                    std::lock_guard<std::shared_mutex> lock(children[i]->nodeMutex);
                    children[i]->deserializeSubtree(in);
                }
            } else {
                uint8_t childMask;
                readVal(in, childMask);
                for (int i = 0; i < 8; ++i) {
                    if ((childMask >> i) & 1) {
                        PointType childMin, childMax;
                        for (int d = 0; d < Dim; ++d) {
                            bool high = (i >> d) & 1;
                            childMin[d] = high ? center[d] : bounds().first[d];
                            childMax[d] = high ? bounds().second[d] : center[d];
                        }
                        children[i] = std::make_unique<OctreeNode_<T, IndexType>>(childMin, childMax);
                        std::lock_guard<std::shared_mutex> lock(children[i]->nodeMutex);
                        children[i]->deserializeSubtree(in);
                    } else {
                        children[i] = nullptr;
                    }
                }
            }
        }
        setLoaded(true);
        setDirty(false);
    }

    void deserialize(std::ifstream& in) {
        bool isRegion;
        readVal(in, isRegion);

        if (isRegion) {
            setLoaded(false);
            setDirty(false);
            setLeaf(false);
            return;
        }

        deserializeSubtree(in);

    }

    void clearDirtySubtree() {
        setDirty(false);
        if (!isLeaf()) {
            for (auto& child : children) if (child) child->clearDirtySubtree();
        }
    }

    bool saveRegion(size_t regionTargetPoints, const std::string& storagePath, const PointType* parentCenter = nullptr) {
        std::string path = getRegionPath(storagePath, parentCenter);
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        serialize(out, regionTargetPoints, storagePath, parentCenter);
        clearDirtySubtree();
        return true;
    }

    bool loadRegion(const std::string& storagePath, const PointType* parentCenter = nullptr) {
        std::string path = getRegionPath(storagePath, parentCenter);
        std::ifstream in(path, std::ios::binary);
        if (in) {
            deserialize(in);
            setLoaded(true);
            return true;
        }
        return false;
    }

    void offload() {
        std::lock_guard<std::shared_mutex> lock(nodeMutex);
        if (isDirty()) return;
        setLoaded(false);
        for (int i = 0; i < (isFat() ? 65536 : 8); ++i) {
            children[i].reset();
        }
        points.clear();
        points.shrink_to_fit();
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