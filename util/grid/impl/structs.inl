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

template<size_t N>
struct GridStoragePath {
    char value[N];
    constexpr GridStoragePath(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            value[i] = str[i];
        }
    }
};

enum class BodyType : uint8_t {
    STATIC = 0,
    KINEMATIC = 1,
    RIGID = 2,
    SOFT = 3,
    FLUID = 4,
    // GAS = 5
};

static inline uint32_t packRGB9E5(const Eigen::Vector3f& c) {
    float rc = std::max(0.0f, c.x());
    float gc = std::max(0.0f, c.y());
    float bc = std::max(0.0f, c.z());
    float max_c = std::max({rc, gc, bc});
    if (max_c <= 0.0f) return 0;

    int exp_val;
    std::frexp(max_c, &exp_val);
    exp_val = std::max(-15, std::min(16, exp_val));
    float scale = std::pow(2.0f, -(exp_val - 9));
    
    uint32_t r = static_cast<uint32_t>(std::clamp(rc * scale, 0.0f, 511.0f));
    uint32_t g = static_cast<uint32_t>(std::clamp(gc * scale, 0.0f, 511.0f));
    uint32_t b = static_cast<uint32_t>(std::clamp(bc * scale, 0.0f, 511.0f));
    uint32_t e = static_cast<uint32_t>(exp_val + 15);
    
    return r | (g << 9) | (b << 18) | (e << 27);
}

static inline Eigen::Vector3f unpackRGB9E5(uint32_t c) {
    if (c == 0) return Eigen::Vector3f::Zero();
    int e = static_cast<int>(c >> 27) - 15;
    float scale = std::pow(2.0f, static_cast<float>(e - 9));
    float r = static_cast<float>(c & 0x1FF) * scale;
    float g = static_cast<float>((c >> 9) & 0x1FF) * scale;
    float b = static_cast<float>((c >> 18) & 0x1FF) * scale;
    return Eigen::Vector3f(r, g, b);
}

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct PhysicsState_ {
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f force{0.0f, 0.0f, 0.0f};
    float density = 1.0f;
    float pressure = 0.0f;
};

struct SPHKernels {
    float h, h2, h3, h4, h6, h9;
    float poly6_k, spiky_k, visc_k, visc_l_k, gauss_k, wendland_k, spline_k;

    SPHKernels(float smoothingRadius = 0.2f) {
        update(smoothingRadius);
    }

    void update(float smoothingRadius) {
        h = std::max(smoothingRadius, 0.0001f);
        h2 = h * h;
        h3 = h2 * h;
        h4 = h2 * h2;
        h6 = h3 * h3;
        h9 = h6 * h3;

        constexpr float pi = 3.14159265358979323846f;
        poly6_k = 315.0f / (64.0f * pi * h9);
        spiky_k = 15.0f / (pi * h6);
        visc_k = 15.0f / (2.0f * pi * h3);
        visc_l_k = 45.0f / (pi * h6);
        gauss_k = 1.0f / std::pow(pi * h2, 1.5f);
        wendland_k = 21.0f / (16.0f * pi * h3);
        spline_k = 8.0f / (pi * h3);
    }

    inline float Poly6(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return poly6_k * hr2 * hr2 * hr2;
    }

    inline float Poly6Grad(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return -6.0f * poly6_k * r * hr2 * hr2;
    }

    inline float Poly6Laplacian(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return -6.0f * poly6_k * hr2 * (3.0f * h2 - 7.0f * r*r);
    }

    inline float Spiky(float r) const {
        if (r >= h) return 0.0f;
        float hr = h - r;
        return spiky_k * hr * hr * hr;
    }

    inline float SpikyGrad(float r) const {
        if (r >= h) return 0.0f;
        float hr = h - r;
        return -3.0f * spiky_k * hr * hr;
    }

    inline float SpikyLaplacian(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        float hr = h - r;
        return -6.0f * spiky_k * hr * (h - 2.0f * r) / r;
    }

    inline float Visc(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        return visc_k * (-(r*r*r)/(2.0f*h3) + (r*r)/h2 + h/(2.0f*r) - 1.0f);
    }

    inline float ViscGrad(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        return visc_k * (-1.5f*(r*r)/h3 + 2.0f*r/h2 - h/(2.0f*r*r));
    }

    inline float ViscLaplacian(float r) const {
        if (r >= h) return 0.0f;
        return visc_l_k * (h - r);
    }

    inline float Gauss(float r) const {
        if (r >= h) return 0.0f;
        return gauss_k * std::exp(-(r*r)/h2);
    }

    inline float GaussGrad(float r) const {
        if (r >= h) return 0.0f;
        return Gauss(r) * (-2.0f * r / h2);
    }
    
    inline float GaussLaplacian(float r) const {
        if (r >= h) return 0.0f;
        return Gauss(r) * (4.0f*r*r - 6.0f*h2) / h4;
    }

    inline float Wendland(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return wendland_k * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
    }

    inline float WendlandGrad(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return -20.0f * wendland_k / h * q * (oq*oq*oq);
    }

    inline float WendlandLaplacian(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return -60.0f * wendland_k / h2 * (oq*oq) * (1.0f - 2.0f*q);
    }

    inline float CubicSpline(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        if (q < 0.5f) return spline_k * (1.0f - 6.0f*q*q + 6.0f*q*q*q);
        float oq = 1.0f - q;
        return spline_k * 2.0f * (oq*oq*oq);
    }

    inline float CubicSplineGrad(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        if (q < 0.5f) return spline_k * (6.0f/h) * q * (3.0f*q - 2.0f);
        float oq = 1.0f - q;
        return -6.0f * spline_k / h * (oq*oq);
    }
    
    inline float CubicSplineLaplacian(float r) const {
        if (r >= h) return 0.0f;
        float q = std::max(r / h, 0.0001f);
        if (q < 0.5f) return spline_k * (36.0f/h2) * (2.0f*q - 1.0f);
        return -12.0f * spline_k / h2 * (1.0f - q) * (1.0f - 2.0f*q) / q;
    }
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

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct Material_ {
    float emittance; //replace with eigen::vector3f<eigen::half> for chromaticity
    //or use uint32_t r8g8b8i8.
    // or a reasonable option: rgb9e5 GL_RGB9_E5
    float roughness;
    float metallic; //rename to reflective or something
    float transmission;
    float ior;
    Eigen::Vector3f absorption;
    //bandwidth?
    //dispersion?

    Material_(float e = 0.0f, float r = 1.0f, float m = 0.0f, float t = 0.0f, float i = 1.45f, Eigen::Vector3f a = Eigen::Vector3f::Zero())
        : emittance(e), roughness(r), metallic(m), transmission(t), ior(i), absorption(a) {}

    bool operator==(const Material_& o) const {
        return emittance == o.emittance && roughness == o.roughness &&
               metallic == o.metallic && transmission == o.transmission &&
               ior == o.ior && absorption == o.absorption;
    }
    
    bool operator<(const Material_& o) const {
        if (emittance != o.emittance) return emittance < o.emittance;
        if (roughness != o.roughness) return roughness < o.roughness;
        if (metallic != o.metallic) return metallic < o.metallic;
        if (transmission != o.transmission) return transmission < o.transmission;
        return ior < o.ior;
    }
};

struct PhysicsMaterial_ {
    BodyType type = BodyType::STATIC;
    float mass = 1.0f;
    
    bool operator==(const PhysicsMaterial_& o) const {
        return type == o.type && mass == o.mass;
    }
};

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct GridObject_ {
    int id;
    uint8_t objectFlags;
    PointType centerPosition = PointType::Zero();
    float maxGasVoxelSize = 0.0f;

    std::vector<Material_<T, IndexType, StoragePath>> renderMaterials;
    std::vector<PhysicsMaterial_> physicsMaterials;

    struct VoxelRel {
        PointType relPos;
        uint16_t renderMatIdx;
        uint16_t physMatIdx;
        float size;
    };
    std::vector<VoxelRel> relativeVoxels;

    mutable std::shared_mutex objMutex;

    GridObject_(int objId = -1) : id(objId), objectFlags(OBJ_ALLOW_PARTIAL_UNLOAD_BIT) {}

    bool isPartialUnloadAllowed() const {
        return objectFlags & OBJ_ALLOW_PARTIAL_UNLOAD_BIT;
    }
    void setPartialUnloadAllowed(bool v) {
        if (v) objectFlags |= OBJ_ALLOW_PARTIAL_UNLOAD_BIT;
        else objectFlags &= ~OBJ_ALLOW_PARTIAL_UNLOAD_BIT;
    }

    uint16_t getOrAddRenderMaterial(const Material_<T, IndexType, StoragePath>& mat) {
        std::unique_lock<std::shared_mutex> lock(objMutex);
        for (size_t i = 0; i < renderMaterials.size(); ++i) {
            if (renderMaterials[i] == mat) return static_cast<uint16_t>(i);
        }
        renderMaterials.push_back(mat);
        return static_cast<uint16_t>(renderMaterials.size() - 1);
    }

    uint16_t getOrAddPhysicsMaterial(const PhysicsMaterial_& pmat) {
        std::unique_lock<std::shared_mutex> lock(objMutex);
        for (size_t i = 0; i < physicsMaterials.size(); ++i) {
            if (physicsMaterials[i] == pmat) return static_cast<uint16_t>(i);
        }
        physicsMaterials.push_back(pmat);
        return static_cast<uint16_t>(physicsMaterials.size() - 1);
    }
    
    Material_<T, IndexType, StoragePath> getRenderMaterial(uint16_t idx) const {
        std::shared_lock<std::shared_mutex> lock(objMutex);
        if (idx < renderMaterials.size()) return renderMaterials[idx];
        return Material_<T, IndexType, StoragePath>();
    }
    
    PhysicsMaterial_ getPhysicsMaterial(uint16_t idx) const {
        std::shared_lock<std::shared_mutex> lock(objMutex);
        if (idx < physicsMaterials.size()) return physicsMaterials[idx];
        return PhysicsMaterial_();
    }
};

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct NodeData_ {
    T data;
    PointType position;
    uint16_t objectId;
    float size;
    Eigen::Vector3f color;
    uint16_t renderMatIdx;
    uint16_t physMatIdx;
    std::atomic<uint8_t> flags;
    PhysicsState_<T, IndexType, StoragePath> physics;

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

template<typename T, typename GasT = float, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct OctreeNode_ {
    BoundingBox bounds;
    EulerianGasState_<GasT> gasState;

    std::vector<std::shared_ptr<NodeData_<T, IndexType, StoragePath>>> points;
    std::vector<std::unique_ptr<OctreeNode_<T, GasT, IndexType, StoragePath>>> children;
    PointType center;
    float nodeSize;
    std::atomic<uint8_t> flags;
    
    mutable std::shared_ptr<NodeData_<T, IndexType, StoragePath>> lodData;
    mutable std::shared_mutex nodeMutex;

    OctreeNode_(const PointType& min, const PointType& max, bool fat = false) : bounds(min, max), flags(0), lodData(nullptr) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        setFat(fat);
        children.resize(isFatNode ? 32768 : 8); 
        for (auto& child : children) {
            child = nullptr;
        }
        center = (bounds.first + bounds.second) * 0.5;
        nodeSize = (bounds.second - bounds.first).norm();
    }

    std::unique_ptr<OctreeNode_<T, GasT, IndexType, StoragePath>> clone() const {
        auto newNode = std::make_unique<OctreeNode_<T, GasT, IndexType, StoragePath>>(bounds.first, bounds.second);
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
            auto pt = std::make_shared<NodeData_<T, IndexType, StoragePath>>();
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
                    children[i] = std::make_unique<OctreeNode_<T, GasT, IndexType, StoragePath>>(childMin, childMax);
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
            auto pt = std::make_shared<NodeData_<T, IndexType, StoragePath>>();
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
                    children[i] = std::make_unique<OctreeNode_<T, GasT, IndexType, StoragePath>>(childMin, childMax);
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

template<typename T, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
struct RayHit_ {
    std::shared_ptr<NodeData_<T, IndexType, StoragePath>> node;
    float distance;
    PointType normal;
    PointType hitPoint;
};

}