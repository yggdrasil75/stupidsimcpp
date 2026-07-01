#pragma once
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

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

static constexpr uint8_t OBJ_ALLOW_PARTIAL_UNLOAD_BIT = 1 << 0;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
using PointType = Eigen::Matrix<float, Dim, 1>;
using BoundingBox = std::pair<PointType, PointType>;
namespace fs = std::filesystem;

enum class BodyType : uint8_t {
    STATIC = 0,
    KINEMATIC = 1,
    RIGID = 2,
    SOFT = 3,
    FLUID = 4
};

static inline uint32_t packRGB9E5(const Eigen::Vector3f& c) {
    float rc = std::max(0.0f, c.x());
    float gc = std::max(0.0f, c.y());
    float bc = std::max(0.0f, c.z());
    float max_c = c.maxCoeff();
    if (max_c <= 0.0f) return 0;

    int exp_val;
    std::frexp(max_c, &exp_val);
    exp_val = std::clamp(exp_val, -15, 16);
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

template<typename T>
struct NodeData_;

template<typename T>
struct Bond_ {
    std::weak_ptr<NodeData_<T>> other;
    float restLength = 0.0f;
    float strength   = 0.0f;
    bool  toAnchor   = false;
};

template<typename T>
struct PhysicsState_ {
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f force{0.0f, 0.0f, 0.0f};
    float density = 1.0f;
    float pressure = 0.0f;
    std::vector<Bond_<T>> bonds;
    bool bondsBuilt = false;
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

using v3half = Eigen::Matrix<Eigen::half, 3, 1>;
static constexpr float SELL_LAMBDA_R = 0.610f;
static constexpr float SELL_LAMBDA_G = 0.550f;
static constexpr float SELL_LAMBDA_B = 0.465f;

static inline float sellmeierN(const v3half& B, const v3half& C, float lambdaUm) {
    float l2 = lambdaUm * lambdaUm;
    float n2 = 1.0f;
    for (int j = 0; j < 3; ++j) {
        float Bj = static_cast<float>(B[j]);
        float Cj = static_cast<float>(C[j]);
        float denom = l2 - Cj;
        if (std::abs(denom) > 1e-8f) n2 += Bj * l2 / denom;
    }
    return std::sqrt(std::max(1.0f, n2));
}

static inline void sellmeierFromConstant(float n, v3half& B, v3half& C) {
    float b0 = std::max(0.0f, n * n - 1.0f);
    B = v3half(Eigen::half(b0), Eigen::half(0.0f), Eigen::half(0.0f));
    C = v3half(Eigen::half(0.0f), Eigen::half(0.0f), Eigen::half(0.0f));
}

struct RenderMaterial {
    uint32_t chromaticity;
    float roughness;
    float metallic;
    v3half sellB;
    v3half sellC;
    Eigen::Vector3f absorption;
    //bandwidth?
    //dispersion?

    RenderMaterial(uint32_t e, float r, float m, const v3half& B, const v3half& C,
              Eigen::Vector3f a = Eigen::Vector3f::Zero())
        : chromaticity(e), roughness(r), metallic(m), sellB(B), sellC(C), absorption(a) {}

    RenderMaterial(float e = 0.0f, float r = 1.0f, float m = 0.0f, float i = 1.45f, Eigen::Vector3f a = Eigen::Vector3f::Zero())
        : chromaticity(packRGB9E5(Eigen::Vector3f(e, e, e))), roughness(r), metallic(m), absorption(a) {
        sellmeierFromConstant(i, sellB, sellC);
    }
    float iorGreen() const { return sellmeierN(sellB, sellC, SELL_LAMBDA_G); }
    Eigen::Vector3f emittanceRGB() const { return unpackRGB9E5(chromaticity); }

    bool operator==(const RenderMaterial& o) const {
        return chromaticity == o.chromaticity && roughness == o.roughness &&
               metallic == o.metallic &&
               sellB == o.sellB && sellC == o.sellC
               && absorption == o.absorption;
    }
    
    bool operator<(const RenderMaterial& o) const {
        if (chromaticity != o.chromaticity) return chromaticity < o.chromaticity;
        if (roughness != o.roughness) return roughness < o.roughness;
        if (metallic != o.metallic) return metallic < o.metallic;
        return iorGreen() < o.iorGreen();
    }

    float dist(const RenderMaterial& o) const {
        float dr = roughness - o.roughness;
        float dm = metallic - o.metallic;
        float di = iorGreen() - o.iorGreen();
        Eigen::Vector3f de = emittanceRGB() - o.emittanceRGB();
        float empenalty = de.norm();
        float absPenalty = (absorption != o.absorption) ? 0.5f : 0.0f;
        return dr*dr + dm*dm + di*di + empenalty + absPenalty;
    }
};

struct PhysicsMaterial_ {
    BodyType type = BodyType::STATIC;
    float mass = 1.0f;
    float stiffness  = 4000.0f;
    float breakForce = 60.0f;
    float damping    = 0.4f;
    ///TODO: restitution, density
    
    bool operator==(const PhysicsMaterial_& o) const {
        return type == o.type && mass == o.mass && stiffness == o.stiffness &&
               breakForce == o.breakForce && damping == o.damping;
    }

    float dist(const PhysicsMaterial_& o) const {
        float dm = mass - o.mass;
        float ds = (stiffness - o.stiffness) * 0.001f;
        float db = (breakForce - o.breakForce) * 0.01f;
        float typePenalty = (type != o.type) ? 10.0f : 0.0f;
        return dm*dm + ds*ds + db*db + typePenalty;
    }
};

struct materialHash {
    size_t operator()(const RenderMaterial& m) const {
        std::hash<float> hf;
        size_t h = std::hash<uint32_t>()(m.chromaticity);
        h ^= hf(m.roughness) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.metallic) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int j = 0; j < 3; ++j)
            h ^= hf(static_cast<float>(m.sellB[j])) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int j = 0; j < 3; ++j)
            h ^= hf(static_cast<float>(m.sellC[j])) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.absorption.x()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.absorption.y()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.absorption.z()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct physicsMatHash {
    size_t operator()(const PhysicsMaterial_& m) const {
        std::hash<float> hf;
        size_t h = std::hash<uint8_t>()(static_cast<uint8_t>(m.type));
        h ^= hf(m.mass) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.stiffness) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.breakForce) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct VoxelRel {
    PointType relPos;
};

template<typename T>
struct GridObject_ {
    int id;
    uint8_t objectFlags;
    PointType centerPosition = PointType::Zero();

    std::vector<RenderMaterial> renderMaterials;
    std::unordered_map<RenderMaterial, uint16_t, materialHash> renderMatMap;
    std::vector<PhysicsMaterial_> physicsMaterials;
    std::unordered_map<PhysicsMaterial_, uint16_t, physicsMatHash> physicsMatMap;

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

    uint16_t getOrAddRenderMaterial(const RenderMaterial& renderMat) {
        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto a = renderMatMap.find(renderMat);
            if (a != renderMatMap.end()) {
                return a->second;
            }
        }

        if (renderMaterials.size() < std::numeric_limits<uint16_t>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = renderMatMap.find(renderMat);
            if (a != renderMatMap.end()) {
                return a->second;
            }
            uint16_t newIndex = static_cast<uint16_t>(renderMaterials.size());
            renderMaterials.push_back(renderMat);
            renderMatMap[renderMat] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            uint16_t bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (uint16_t i = 0; i < static_cast<uint16_t>(renderMaterials.size()); ++i) {
                float dist2 = renderMaterials[i].dist(renderMat);
                if (dist2 < dist) {
                    dist = dist2;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    uint16_t getOrAddPhysicsMaterial(const PhysicsMaterial_& pmat) {
        {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            auto a = physicsMatMap.find(pmat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
        }

        if (physicsMaterials.size() < std::numeric_limits<uint16_t>::max()) {
            std::unique_lock<std::shared_mutex> writeLock(objMutex);
            auto a = physicsMatMap.find(pmat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
            uint16_t newIndex = static_cast<uint16_t>(physicsMaterials.size());
            physicsMaterials.push_back(pmat);
            physicsMatMap[pmat] = newIndex;
            return newIndex;
        } else {
            std::shared_lock<std::shared_mutex> readLock(objMutex);
            uint16_t bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (uint16_t i = 0; i < static_cast<uint16_t>(physicsMaterials.size()); ++i) {
                float dist2 = physicsMaterials[i].dist(pmat);
                if (dist2 < dist) {
                    dist = dist2;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }
    
    RenderMaterial getRenderMaterial(uint16_t idx) const {
        std::shared_lock<std::shared_mutex> lock(objMutex);
        if (idx < renderMaterials.size()) return renderMaterials[idx];
        return RenderMaterial();
    }
    
    PhysicsMaterial_ getPhysicsMaterial(uint16_t idx) const {
        std::shared_lock<std::shared_mutex> lock(objMutex);
        if (idx < physicsMaterials.size()) return physicsMaterials[idx];
        return PhysicsMaterial_();
    }
};

template<typename T>
struct NodeData_ {
    T data;
    PointType position;
    int objectId;
    float size;
    Eigen::Vector4f color;
    uint16_t renderMatIdx;
    uint16_t physMatIdx;
    std::atomic<uint8_t> flags;
    PhysicsState_<T> physics;

    NodeData_(const T& data, const PointType& pos, bool visible, const Eigen::Vector4f& color, float size = 0.01f,
                bool active = true, int objectId = -1, uint16_t rIdx = 0, uint16_t pIdx = 0, bool staticbit = 0) 
            : data(data), position(pos), objectId(objectId), size(size), 
                color(color), renderMatIdx(rIdx), physMatIdx(pIdx), flags(0) {
        setActive(active);
        setVisible(visible);
        setStatic(staticbit);
    }
    
    NodeData_() : objectId(-1), size(0.0f), color(Eigen::Vector4f::Zero()), renderMatIdx(0), physMatIdx(0), flags(0) {}

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

template<typename T>
struct OctreeNode_ {
    std::vector<std::shared_ptr<NodeData_<T>>> points;
    std::array<std::unique_ptr<OctreeNode_<T>>, 8> children;
    PointType center;
    float nodeSize;
    std::atomic<uint8_t> flags;
    
    mutable std::shared_ptr<NodeData_<T>> lodData;
    mutable std::shared_mutex nodeMutex;

    OctreeNode_(const PointType& min, const PointType& max) : flags(0), lodData(nullptr) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        for (std::unique_ptr<OctreeNode_<T>>& child : children) {
            child = nullptr;
        }
        center = (min + max) * 0.5;
        nodeSize = (max - min).norm();
    }

    OctreeNode_(const PointType& center, const float& size) : center(center), nodeSize(size), flags(0), lodData(nullptr) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        for (std::unique_ptr<OctreeNode_<T>>& child : children) {
            child = nullptr;
        }
    }

    std::unique_ptr<OctreeNode_<T>> clone() const {
        auto newNode = std::make_unique<OctreeNode_<T>>(center, nodeSize);
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
        BoundingBox b = bounds();
        return (point[0] >= b.first[0] && point[0] <= b.second[0] &&
                point[1] >= b.first[1] && point[1] <= b.second[1] &&
                point[2] >= b.first[2] && point[2] <= b.second[2]);
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

    std::string getRegionPath(const std::string storagepath) const {
        int64_t cx = static_cast<int64_t>(std::floor(center.x()));
        int64_t cy = static_cast<int64_t>(std::floor(center.y()));
        int64_t cz = static_cast<int64_t>(std::floor(center.z()));
        int64_t s = static_cast<int64_t>(std::floor(nodeSize));
        
        fs::path p(storagepath);
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

    static void writeVec4(std::ofstream& out, const Eigen::Vector4f& vec) {
        writeVal(out, vec.x());
        writeVal(out, vec.y());
        writeVal(out, vec.z());
        writeVal(out, vec.w());
    }

    static void readVec4(std::ifstream& in, Eigen::Vector4f& vec) {
        float x, y, z, w;
        readVal(in, x);
        readVal(in, y);
        readVal(in, z);
        readVal(in, w);
        vec = Eigen::Vector4f(x, y, z, w);
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
            writeVec4(out, pt->color);
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

        size_t pointCount;
        readVal(in, pointCount);
        points.reserve(pointCount);
        for (size_t i = 0; i < pointCount; ++i) {
            auto pt = std::make_shared<NodeData_<T>>();
            deserializeData(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            uint8_t f;
            readVal(in, f);
            pt->flags.store(f, std::memory_order_relaxed);
            readVal(in, pt->size);
            readVec4(in, pt->color);
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
                        childMin[d] = high ? center[d] : bounds().first[d];
                        childMax[d] = high ? bounds().second[d] : center[d];
                    }
                    children[i] = std::make_unique<OctreeNode_<T>>(childMin, childMax);
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

    bool saveRegion(const std::string storagepath) {
        std::string path = getRegionPath(storagepath);
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        serializeSubtree(out);
        clearDirtySubtree();
        return true;
    }

    bool loadRegion(const std::string storagepath) {
        std::string path = getRegionPath(storagepath);
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

    void serialize(std::ofstream& out, size_t regionTargetPoints, std::string storagepath) {
        bool offloaded = !isLoaded();
        size_t subPoints = offloaded ? 0 : getSubtreePointCount();

        bool isRegion = offloaded || (subPoints > 0 && (subPoints <= regionTargetPoints || isLeaf()) && isSubtreeFullyLoaded());

        writeVal(out, isRegion);

        if (isRegion) {
            if (!offloaded && isDirty()) saveRegion(storagepath);
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
            writeVec4(out, pt->color);
            writeVal(out, pt->renderMatIdx);
            writeVal(out, pt->physMatIdx);
        }

        if (!isLeaf()) {
            uint8_t childMask = 0;
            for (int i = 0; i < 8; ++i) if (children[i]) childMask |= (1 << i);
            writeVal(out, childMask);
            for (int i = 0; i < 8; ++i) {
                if (children[i]) children[i]->serialize(out, regionTargetPoints, storagepath);
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
            auto pt = std::make_shared<NodeData_<T>>();
            deserializeData(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            uint8_t f;
            readVal(in, f);
            pt->flags.store(f, std::memory_order_relaxed);
            readVal(in, pt->size);
            readVec4(in, pt->color);
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
                        childMin[d] = high ? center[d] : bounds().first[d];
                        childMax[d] = high ? bounds().second[d] : center[d];
                    }
                    children[i] = std::make_unique<OctreeNode_<T>>(childMin, childMax);
                    children[i]->deserialize(in, regionTargetPoints);
                } else {
                    children[i] = nullptr;
                }
            }
        }
        setLoaded(true);
        setDirty(false);
    }

    BoundingBox bounds() const {
        float halfsize = static_cast<float>(nodeSize) * 0.5f;
        PointType hs = PointType::Constant(halfsize);
        return BoundingBox({center - hs, center + hs});
    }
};

template<typename T>
struct RayHit_ {
    std::shared_ptr<NodeData_<T>> node;
    float distance;
    PointType normal;
    PointType hitPoint;
};
    
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

}