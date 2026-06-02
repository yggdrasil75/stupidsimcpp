#pragma once

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
namespace fs = std::filesystem;

enum class BodyType : uint8_t {
    STATIC = 0,
    RIGID = 1,
    SOFT = 2,
    FLUID = 3
};

struct vec3fh {
    std::size_t operator()(const Eigen::Vector3f& v) const {
        std::size_t h1 = std::hash<float>()(v.x());
        std::size_t h2 = std::hash<float>()(v.y());
        std::size_t h3 = std::hash<float>()(v.z());
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

    std::vector<VoxelRel> relativeVoxels;

    mutable std::shared_mutex objMutex;

    GridObject_(int objId = -1) : id(objId), flags(OBJ_ALLOW_PARTIAL_UNLOAD_BIT) {
        transmissionMap.push_back(0.0f);
        transmissionMap.push_back(1.0f);
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
                float dist2 = (renderMaterials[i] - renderMat).dist();
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
            auto a = PhysicsMatMap.find(physMat);
            if (a != PhysicsMatMap.end()) {
                return a->second;
            }
        }

        if (physicsMaterials.size() < std::numeric_limits<IndexType>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = PhysicsMatMap.find(physMat);
            if (a != PhysicsMatMap.end()) {
                return a->second;
            }
            IndexType newIndex = static_cast<IndexType>(physicsMaterials.size());
            physicsMaterials.push_back(physMat);
            PhysicsMatMap[physMat] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            IndexType bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (IndexType i = 0; i < static_cast<IndexType>(physicsMaterials.size()); ++i) {
                float dist2 = physicsMaterials[i].dist(mat);
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

    PhysicsMaterial_ getTransmission(uint8_t idx) const {
        return transmissionTable[idx];
    }

};

}