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
#include <deque>

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

static constexpr uint8_t REUSE_SETTLE_FRAMES = 3;
static constexpr float REUSE_MAX_TRANSMISSION = 0.05f;
static constexpr float REUSE_MIN_ROUGHNESS = 0.25f;

static constexpr uint32_t WC_INVALID_KEY = 0u;
static constexpr int WC_MAX_AGE = 32;
static constexpr uint32_t WC_CAPACITY = 1u << 20;

static constexpr int DDGI_IRR_RES = 8;
static constexpr int DDGI_DEPTH_RES = 16;
static constexpr int DDGI_RAYS_PER_PROBE = 1024;
static constexpr int DDGI_PROBES_X = 16;
static constexpr int DDGI_PROBES_Y = 8;
static constexpr int DDGI_PROBES_Z = 16;
static constexpr float DDGI_HYSTERESIS = 0.80f;
static constexpr float DDGI_DEPTH_SHARPNESS = 50.0f;
static constexpr float DDGI_NORMAL_BIAS = 0.25f;

static constexpr int SELL_LUT_WAVELENGTHS = 32;
static constexpr int SELL_LUT_SECONDARY   = 8;
static constexpr float SELL_LMIN = 0.380f; // um
static constexpr float SELL_LMAX = 0.720f; // um

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
using Vec3 = Eigen::Vector3f;
using BoundingBox = std::pair<Vec3, Vec3>;
using u_lock = std::unique_lock<std::shared_mutex>;
using s_lock = std::shared_lock<std::shared_mutex>;
namespace fs = std::filesystem;
constexpr uint32_t u32M = std::numeric_limits<uint32_t>::max();
constexpr uint32_t INVALID_IDX = std::numeric_limits<uint32_t>::max();

enum class BodyType : uint8_t {
    STATIC = 0,
    KINEMATIC = 1,
    RIGID = 2,
    SOFT = 3,
    FLUID = 4
};

///@brief What happens to fragments when a rigid body's bond graph splits apart
enum class SplitPolicy : uint8_t {
    NEW_OID = 0,
    KEEP_OID = 1,
    SHED_STATIC = 2,
    DISSOLVE = 3
};

struct Vec3i64Hash {
    std::size_t operator()(const std::array<int64_t, 3>& v) const {
        return (std::size_t)((v[0] * 73856093) ^ (v[1] * 19349663) ^ (v[2] * 83492791));
    }
};

struct Vec3fHash {
    std::size_t operator()(const std::array<int64_t, 3>& v) const {
        return (std::size_t)((v[0] * 73856093) ^ (v[1] * 19349663) ^ (v[2] * 83492791));
    }
};

static inline uint32_t packRGB9E5(const Vec3& c) {
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

static inline Vec3 unpackRGB9E5(uint32_t c) {
    if (c == 0) return Vec3::Zero();
    int e = static_cast<int>(c >> 27) - 15;
    float scale = std::pow(2.0f, static_cast<float>(e - 9));
    float r = static_cast<float>(c & 0x1FF) * scale;
    float g = static_cast<float>((c >> 9) & 0x1FF) * scale;
    float b = static_cast<float>((c >> 18) & 0x1FF) * scale;
    return Vec3(r, g, b);
}

static inline uint32_t packRGB8(const Vec3& c) {
    uint32_t r = static_cast<uint32_t>(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f);
    uint32_t g = static_cast<uint32_t>(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f);
    uint32_t b = static_cast<uint32_t>(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f);
    return r | (g << 8) | (b << 16);
}

static inline uint32_t packRGBA8(const Eigen::Vector4f& c) {
    uint32_t r = static_cast<uint32_t>(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f);
    uint32_t g = static_cast<uint32_t>(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f);
    uint32_t b = static_cast<uint32_t>(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f);
    uint32_t a = static_cast<uint32_t>(std::clamp(c.w(), 0.0f, 1.0f) * 255.0f);
    return r | (g << 8) | (b << 16) | (a << 24);
}

static constexpr uint32_t EXTENT_UNIT = 0u; // 1,1,1 packed; fields store count-1
static constexpr float LATTICE_EPS = 1e-3f; // cell fractions; merge only near-exact grid points
static constexpr uint32_t EXTENT_MAX  = 1024u;
static constexpr uint32_t EXTENT_STATIC_BIT = 1u << 30;
static constexpr uint32_t EXTENT_REUSE_BIT  = 1u << 31;

static inline bool extentIsStatic(uint32_t e) {
    return (e & EXTENT_STATIC_BIT) != 0u;
}
static inline uint32_t extentSetStatic(uint32_t e, bool v) {
    return v ? (e | EXTENT_STATIC_BIT) : (e & ~EXTENT_STATIC_BIT);
}
static inline bool extentIsReusable(uint32_t e) {
    return (e & EXTENT_REUSE_BIT) != 0u;
}
static inline uint32_t extentSetReusable(uint32_t e, bool v) {
    return v ? (e | EXTENT_REUSE_BIT) : (e & ~EXTENT_REUSE_BIT);
}

///@brief Packs a per-axis cell count into three 10-bit fields
///@param ex Cell span along x, clamped to [1, EXTENT_MAX]
///@param ey Cell span along y
///@param ez Cell span along z
///@return Packed extent, x in bits 0-9, y in 10-19, z in 20-29
static inline uint32_t packExtent(uint32_t ex, uint32_t ey, uint32_t ez) {
    uint32_t x = std::clamp(ex, 1u, EXTENT_MAX) - 1u;
    uint32_t y = std::clamp(ey, 1u, EXTENT_MAX) - 1u;
    uint32_t z = std::clamp(ez, 1u, EXTENT_MAX) - 1u;
    return x | (y << 10) | (z << 20);
}

static inline uint32_t quantizeNormal(const Vec3& n) {
    uint32_t major = 0;
    Vec3 a = n.cwiseAbs();
    if (a.y() > a.x() && a.y() >= a.z()) major = 1;
    else if (a.z() > a.x() && a.z() > a.y()) major = 2;
    const uint32_t sign = (n[major] < 0.0f) ? 1u : 0u;
    return major * 2u + sign;
}

static inline uint32_t worldCacheKey(int64_t cx, int64_t cy, int64_t cz, uint32_t nb) {
    uint32_t h = static_cast<uint32_t>(cx) * 73856093u;
    h ^= static_cast<uint32_t>(cy) * 19349663u;
    h ^= static_cast<uint32_t>(cz) * 83492791u;
    h ^= nb * 0x9e3779b9u;
    h ^= h >> 16;
    return h == WC_INVALID_KEY ? 1u : h;
}

///@brief Expands a packed extent into per-axis cell counts
///@param e Packed extent as produced by packExtent
///@return Cell spans as floats, each at least 1.0f
static inline Vec3 unpackExtent(uint32_t e) {
    return Vec3(static_cast<float>((e         & 0x3FFu) + 1u),
                static_cast<float>(((e >> 10) & 0x3FFu) + 1u),
                static_cast<float>(((e >> 20) & 0x3FFu) + 1u));
}

static inline uint32_t packMaterialProps(float roughness, float metallic, uint32_t sellmeierRow) {
    uint32_t r8 = static_cast<uint32_t>(std::clamp(roughness, 0.0f, 1.0f) * 255.0f);
    uint32_t m8 = static_cast<uint32_t>(std::clamp(metallic, 0.0f, 1.0f) * 255.0f);
    uint32_t row16 = sellmeierRow & 0xFFFFu;
    return r8 | (m8 << 8) | (row16 << 16);
}

template<typename V>
static void writeVal(std::ofstream& out, const V& val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(V));
}

template<typename V>
static void readVal(std::ifstream& in, V& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(V));
}

static inline void writeVec3(std::ofstream& out, const Vec3& vec) {
    writeVal(out, vec.x());
    writeVal(out, vec.y());
    writeVal(out, vec.z());
}

static inline void readVec3(std::ifstream& in, Vec3& vec) {
    float x, y, z;
    readVal(in, x);
    readVal(in, y);
    readVal(in, z);
    vec = Vec3(x, y, z);
}

static inline void writeVec4(std::ofstream& out, const Eigen::Vector4f& vec) {
    writeVal(out, vec.x());
    writeVal(out, vec.y());
    writeVal(out, vec.z());
    writeVal(out, vec.w());
}

static inline void readVec4(std::ifstream& in, Eigen::Vector4f& vec) {
    float x, y, z, w;
    readVal(in, x);
    readVal(in, y);
    readVal(in, z);
    readVal(in, w);
    vec = Eigen::Vector4f(x, y, z, w);
}

template<typename T>
struct NodeData_;

template<typename T>
struct Bond_ {
    std::weak_ptr<NodeData_<T>> other;
    float restLength = 0.0f;
    float strength   = 0.0f;
    float damage     = 0.0f;
    bool  toAnchor   = false;
    bool  broken     = false;
};

template<typename T>
struct PhysicsState_ {
    Vec3 velocity{0.0f, 0.0f, 0.0f};
    Vec3 force{0.0f, 0.0f, 0.0f};
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
    float n2 = n * n;

    if (n >= 1.0f) {
        float n2_minus_1 = n2 - 1.0f;
        float c1 = 0.0106f; 
        float c2 = 100.0f;  
        float b1 = n2_minus_1 / 1.030788f;
        float b2 = b1 * 0.2f; 

        B = v3half(Eigen::half(b1), Eigen::half(b2), Eigen::half(0.0f));
        C = v3half(Eigen::half(c1), Eigen::half(c2), Eigen::half(0.0f));

    } else {
        n2 = std::max(0.00001f, n2); 
        float c1 = 0.0f;
        float c2 = -0.1f;
        float b1 = 0.611792f * n2 - 1.0f;
        float b2 = 0.5f * n2;

        B = v3half(Eigen::half(b1), Eigen::half(b2), Eigen::half(0.0f));
        C = v3half(Eigen::half(c1), Eigen::half(c2), Eigen::half(0.0f));
    }
}

struct alignas(16) GPUMaterial {
    uint32_t chromaticity; //RBG9E5
    uint32_t materialProps; //8 bits for roughness. 8 for metallicity. 8 for ior. 8 for ???
    uint32_t absorption; //RBG9E5
    uint32_t albedo; //rgb9e5
};

struct RenderMaterial {
    uint32_t chromaticity;
    float roughness;
    float metallic;
    v3half sellB;
    v3half sellC;
    Vec3 absorption;
    //vec3 scattering? // sigma_s (The "diffuse color" of the leaf/wood)
    //float phase_g? // Forward/backward scattering bias (-1.0 to 1.0)
    //float sheen? // Peach fuzz / moss hair rim lighting
    //float anistropy? // Wood grain highlight stretching
    //float translucency? // Thin geometry sss and transparency
    //float porosity? // Rain simulation without rain voxels?

    RenderMaterial(uint32_t e, float r, float m, const v3half& B, const v3half& C,
              Vec3 a = Eigen::Vector3f::Zero())
        : chromaticity(e), roughness(r), metallic(m), sellB(B), sellC(C), absorption(a) {}

    RenderMaterial(float e = 0.0f, float r = 1.0f, float m = 0.0f, float i = 1.45f, Eigen::Vector3f a = Eigen::Vector3f::Zero())
        : chromaticity(packRGB9E5(Eigen::Vector3f(e, e, e))), roughness(r), metallic(m), absorption(a) {
        sellmeierFromConstant(i, sellB, sellC);
    }
    float iorGreen() const { return sellmeierN(sellB, sellC, SELL_LAMBDA_G); }
    Vec3 emittanceRGB() const { return unpackRGB9E5(chromaticity); }

    bool operator==(const RenderMaterial& o) const {
        return chromaticity == o.chromaticity && roughness == o.roughness &&
               metallic == o.metallic && sellB == o.sellB && sellC == o.sellC
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
        Vec3 de = emittanceRGB() - o.emittanceRGB();
        float empenalty = de.norm();
        float absPenalty = (absorption != o.absorption) ? 0.5f : 0.0f;
        return dr*dr + dm*dm + di*di + empenalty + absPenalty;
    }
};

static inline std::vector<float> buildSellmeierLUT(const std::vector<Grid::RenderMaterial>& mats) {
    int rows = std::max<size_t>(1, mats.size()) * SELL_LUT_SECONDARY;
    std::vector<float> lut(static_cast<size_t>(rows) * SELL_LUT_WAVELENGTHS, 1.0f);
    for (size_t mi = 0; mi < mats.size(); ++mi) {
        const auto& m = mats[mi];
        for (int s = 0; s < SELL_LUT_SECONDARY; ++s) {
            int row = static_cast<int>(mi) * SELL_LUT_SECONDARY + s;
            for (int w = 0; w < SELL_LUT_WAVELENGTHS; ++w) {
                float f = (SELL_LUT_WAVELENGTHS == 1) ? 0.0f : float(w) / float(SELL_LUT_WAVELENGTHS - 1);
                float lambda = SELL_LMIN + f * (SELL_LMAX - SELL_LMIN);
                lut[static_cast<size_t>(row) * SELL_LUT_WAVELENGTHS + w] = Grid::sellmeierN(m.sellB, m.sellC, lambda);
            }
        }
    }
    return lut;
}

static inline void buildGPUMaterialCache(const std::vector<Grid::RenderMaterial>& mats, std::vector<GPUMaterial>& outGpu,
                                         std::vector<float>& outSellLUT, size_t& outSellRows) {
    outGpu.clear();
    outGpu.reserve(mats.size() + 1);
    for (size_t mi = 0; mi < mats.size(); ++mi) {
        const auto& m = mats[mi];
        uint32_t sellRow = static_cast<uint32_t>(mi) * SELL_LUT_SECONDARY;
        outGpu.push_back({
            m.chromaticity,
            packMaterialProps(m.roughness, m.metallic, sellRow),
            packRGB9E5(m.absorption),
            0u
        });
    }
    outGpu.push_back(GPUMaterial{});

    outSellLUT = buildSellmeierLUT(mats);
    outSellRows = std::max<size_t>(1, mats.size()) * SELL_LUT_SECONDARY;
}

struct RMatHash {
    size_t operator()(const RenderMaterial& m) const {
        std::hash<float> hf;
        size_t h = std::hash<uint32_t>()(m.chromaticity);
        h ^= hf(m.roughness) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.metallic) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int j = 0; j < 3; ++j)
            h ^= hf(static_cast<float>(m.sellB[j])) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int j = 0; j < 3; ++j)
            h ^= hf(static_cast<float>(m.sellC[j])) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int j = 0; j < 3; ++j)
            h ^= hf(m.absorption[j]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct RenderMaterialStore {
    std::vector<RenderMaterial> materials;
    std::unordered_map<RenderMaterial, uint32_t, RMatHash> matMap;
    mutable std::shared_mutex mutex;
    uint64_t version = 1;
    struct GPUCache {
        uint64_t builtVersion = 0; // 0 == never built
        std::vector<GPUMaterial> gpuMaterials;
        std::vector<float> sellmeierLUT;
        size_t sellmeierRows = 0;
    };
    GPUCache gpuCache;
    std::mutex gpuCacheMutex;

    uint32_t getOrAdd(const RenderMaterial& renderMat) {
        {
            s_lock readLock(mutex);
            auto a = matMap.find(renderMat);
            if (a != matMap.end()) return a->second;
        }

        if (materials.size() < u32M) {
            u_lock writeLock(mutex);
            auto a = matMap.find(renderMat);
            if (a != matMap.end()) return a->second;
            uint32_t newIndex = static_cast<uint32_t>(materials.size());
            materials.push_back(renderMat);
            matMap[renderMat] = newIndex;
            ++version;
            return newIndex;
        } else {
            s_lock readLock(mutex);
            uint32_t bestIndex = 0;
            float dist = std::numeric_limits<float>::max();
            for (uint32_t i = 0; i < static_cast<uint32_t>(materials.size()); ++i) {
                float dist2 = materials[i].dist(renderMat);
                if (dist2 < dist) {
                    dist = dist2;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    RenderMaterial get(uint32_t idx) const {
        s_lock lock(mutex);
        if (idx < materials.size()) return materials[idx];
        return RenderMaterial();
    }

    size_t size() const {
        s_lock lock(mutex);
        return materials.size();
    }

    template<typename Builder>
    void retrieveGPUMaterials(Builder&& builder, const std::vector<GPUMaterial>*& outMaterials,
                              const std::vector<float>*& outSellLUT, size_t& outSellRows) {
        std::vector<RenderMaterial> snapshot;
        uint64_t storeVersion;
        {
            s_lock readLock(mutex);
            storeVersion = version;
            snapshot = materials;
        }

        std::lock_guard<std::mutex> cacheLock(gpuCacheMutex);
        if (gpuCache.builtVersion != storeVersion || gpuCache.gpuMaterials.empty()) {
            builder(snapshot, gpuCache.gpuMaterials, gpuCache.sellmeierLUT, gpuCache.sellmeierRows);
            gpuCache.builtVersion = storeVersion;
        }
        outMaterials = &gpuCache.gpuMaterials;
        outSellLUT = &gpuCache.sellmeierLUT;
        outSellRows = gpuCache.sellmeierRows;
    }
};

struct PhysicsMaterial_ {
    BodyType type = BodyType::STATIC;
    float mass = 1.0f;
    float stiffness  = 4000.0f;
    float breakForce = 60.0f;
    float damping    = 0.4f;
    ///TODO: restitution, density

    float breakCompressionScale = 4.0f;
    float breakTorque = 0.0f;
    float fatigue = 0.0f;
    uint32_t minFragmentVoxels = 1;

    bool operator==(const PhysicsMaterial_& o) const {
        return type == o.type && mass == o.mass && stiffness == o.stiffness &&
               breakForce == o.breakForce && damping == o.damping &&
               breakCompressionScale == o.breakCompressionScale &&
               breakTorque == o.breakTorque && fatigue == o.fatigue &&
               minFragmentVoxels == o.minFragmentVoxels;
    }

    float dist(const PhysicsMaterial_& o) const {
        float dm = mass - o.mass;
        float ds = (stiffness - o.stiffness) * 0.001f;
        float db = (breakForce - o.breakForce) * 0.01f;
        float dc = breakCompressionScale - o.breakCompressionScale;
        float dt = (breakTorque - o.breakTorque) * 0.01f;
        float df = fatigue - o.fatigue;
        float typePenalty = (type != o.type) ? 10.0f : 0.0f;
        return dm*dm + ds*ds + db*db + dc*dc + dt*dt + df*df + typePenalty;
    }
};

struct PMatHash {
    size_t operator()(const PhysicsMaterial_& m) const {
        std::hash<float> hf;
        size_t h = std::hash<uint8_t>()(static_cast<uint8_t>(m.type));
        h ^= hf(m.mass) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.stiffness) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.breakForce) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.breakCompressionScale) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.breakTorque) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= hf(m.fatigue) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(m.minFragmentVoxels) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct VoxelRel {
    Vec3 relPos;
};

///@brief Full per-voxel material spec used by detailed primitive insertion.
struct VoxelMat {
    Vec3 albedo{0.7f, 0.7f, 0.7f};
    float emittance = 0.0f;
    float roughness = 1.0f;
    float metallic = 0.0f;
    float transmission = 0.0f;
    float ior = 1.45f;
    Vec3 absorption = Vec3::Zero();
    v3half sellB = v3half::Zero();
    v3half sellC = v3half::Zero();
    bool useSellmeier = false;
    BodyType bType = BodyType::STATIC;
    float mass = 1.0f;
};

template<typename T>
struct GridObject_ {
    int id;
    uint8_t objectFlags;
    SplitPolicy splitPolicy = SplitPolicy::NEW_OID;
    Vec3 centerPosition = Vec3::Zero();

    std::vector<PhysicsMaterial_> physicsMaterials;
    std::unordered_map<PhysicsMaterial_, uint16_t, PMatHash> physicsMatMap;

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

    uint16_t getOrAddPhysicsMaterial(const PhysicsMaterial_& pmat) {
        {
            s_lock readLock(objMutex);
            auto a = physicsMatMap.find(pmat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
        }

        if (physicsMaterials.size() < std::numeric_limits<uint16_t>::max()) {
            u_lock writeLock(objMutex);
            auto a = physicsMatMap.find(pmat);
            if (a != physicsMatMap.end()) {
                return a->second;
            }
            uint16_t newIndex = static_cast<uint16_t>(physicsMaterials.size());
            physicsMaterials.push_back(pmat);
            physicsMatMap[pmat] = newIndex;
            return newIndex;
        } else {
            s_lock readLock(objMutex);
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
    
    PhysicsMaterial_ getPhysicsMaterial(uint16_t idx) const {
        s_lock lock(objMutex);
        if (idx < physicsMaterials.size()) return physicsMaterials[idx];
        return PhysicsMaterial_();
    }
};

template<typename T>
struct NodeData_ {
    T data;
    Vec3 position;
    int objectId;
    float size;
    Eigen::Vector4f color;
    uint32_t renderMatIdx;
    uint16_t physMatIdx;
    std::atomic<uint8_t> flags;
    std::atomic<uint8_t> settledFrames;
    PhysicsState_<T> physics;

    NodeData_(const T& data, const Vec3& pos, bool visible, const Eigen::Vector4f& color, float size = 0.01f,
                bool active = true, int objectId = -1, uint32_t rIdx = 0, uint16_t pIdx = 0, bool staticbit = 0) 
            : data(data), position(pos), objectId(objectId), size(size), 
                color(color), renderMatIdx(rIdx), physMatIdx(pIdx), flags(0), settledFrames(0) {
        setActive(active);
        setVisible(visible);
        setStatic(staticbit);
        setSettled(staticbit);
    }
    
    NodeData_() : objectId(-1), size(0.0f), color(Eigen::Vector4f::Zero()), renderMatIdx(0), physMatIdx(0), flags(0), settledFrames(0) {}

    NodeData_(const NodeData_& other) : data(other.data), position(other.position), objectId(other.objectId), size(other.size),
            color(other.color), renderMatIdx(other.renderMatIdx), physMatIdx(other.physMatIdx), flags(other.flags.load(std::memory_order_relaxed)),
            settledFrames(other.settledFrames.load(std::memory_order_relaxed)), physics(other.physics) {}

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
            settledFrames.store(other.settledFrames.load(std::memory_order_relaxed), std::memory_order_relaxed);
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
    bool isSettled() const {
        if (physics.velocity.squaredNorm() <= 0.0f
            && (flags.load(std::memory_order_relaxed) & STATIC_BIT)) return true;
        return settledFrames.load(std::memory_order_relaxed) >= REUSE_SETTLE_FRAMES;
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
    void setSettled(bool asleep) {
        if (!asleep) {
            settledFrames.store(0, std::memory_order_relaxed);
            return;
        }
        uint8_t s = settledFrames.load(std::memory_order_relaxed);
        if (s < REUSE_SETTLE_FRAMES) settledFrames.store(s + 1, std::memory_order_relaxed);
    }
    
    Vec3 getHalfSize() const {
        float sizeh = size * 0.5f;
        return Vec3(sizeh, sizeh, sizeh);
    }
    
    BoundingBox getCubeBounds() const {
        Vec3 halfSize = getHalfSize();
        return {position - halfSize, position + halfSize};
    }
};

template<typename T>
struct OctreeNode_ {
    Vec3 center;
    float nodeSize;
    uint32_t firstChild = INVALID_IDX;
    uint32_t pointBlock = INVALID_IDX;
    uint32_t lodIdx = INVALID_IDX;
    uint8_t childMask = 0;
    std::atomic<uint8_t> flags;

    OctreeNode_() : flags(0) {}

    OctreeNode_(const Vec3& min, const Vec3& max) : flags(0) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
        center = (min + max) * 0.5;
        nodeSize = (max - min).maxCoeff();
    }

    OctreeNode_(const Vec3& center, const float& size) : center(center), nodeSize(size), flags(0) {
        setLeaf(true);
        setLoaded(true);
        setDirty(true);
        setLoadQueued(false);
        setSaveQueued(false);
        setKeepLoaded(false);
    }

    OctreeNode_(const OctreeNode_& other) : center(other.center), nodeSize(other.nodeSize),
            firstChild(other.firstChild), pointBlock(other.pointBlock), lodIdx(other.lodIdx),
            childMask(other.childMask), flags(other.flags.load(std::memory_order_relaxed)) {}

    std::unique_ptr<OctreeNode_<T>> clone() const {
        auto newNode = std::make_unique<OctreeNode_<T>>(center, nodeSize);
        newNode->flags.store(flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
        
        newNode->pointBlock = pointBlock;
        newNode->center = center;
        newNode->nodeSize = nodeSize;
        
        return newNode;
    }

    OctreeNode_& operator=(const OctreeNode_& other) {
        if (this != &other) {
            center = other.center;
            nodeSize = other.nodeSize;
            firstChild = other.firstChild;
            pointBlock = other.pointBlock;
            lodIdx = other.lodIdx;
            childMask = other.childMask;
            flags.store(other.flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
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

    bool hasChild(int i) const { return (childMask >> i) & 1; }

    uint32_t child(int i) const {
        return hasChild(i) ? firstChild + static_cast<uint32_t>(i) : INVALID_IDX;
    }

    bool contains(const Vec3& point) const {
        BoundingBox b = bounds();
        return ((point.array() >= b.first.array()) && (point.array() <= b.second.array())).all();
    }

    bool isEmpty() const {
        return pointBlock == INVALID_IDX && childMask == 0;
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

    BoundingBox bounds() const {
        float halfsize = static_cast<float>(nodeSize) * 0.5f;
        Vec3 hs = Vec3::Constant(halfsize);
        return BoundingBox({center - hs, center + hs});
    }
};

template<typename T>
struct PointStore {

    struct Block {
        uint32_t offset = 0;
        uint32_t count = 0;
        uint32_t capacity = 0;
    };

    std::vector<std::shared_ptr<NodeData_<T>>> pool;
    std::vector<Block> blocks;
    std::vector<uint32_t> freeBlocks;
    mutable std::shared_mutex mutex;

    uint32_t alloc(const std::vector<std::shared_ptr<NodeData_<T>>>& pts) {
        u_lock lock(mutex);
        return allocLocked(pts);
    }

    uint32_t allocLocked(const std::vector<std::shared_ptr<NodeData_<T>>>& pts) {
        uint32_t offset = static_cast<uint32_t>(pool.size());
        for (const auto& p : pts) pool.push_back(p);

        Block b{offset, static_cast<uint32_t>(pts.size()), static_cast<uint32_t>(pts.size())};
        if (!freeBlocks.empty()) {
            uint32_t idx = freeBlocks.back();
            freeBlocks.pop_back();
            blocks[idx] = b;
            return idx;
        }
        blocks.push_back(b);
        return static_cast<uint32_t>(blocks.size() - 1);
    }

    uint32_t count(uint32_t blockIdx) const {
        s_lock lock(mutex);
        if (blockIdx == INVALID_IDX) return 0;
        if (blockIdx >= blocks.size()) return 0;
        return blocks[blockIdx].count;
    }

    struct View {
        const std::shared_ptr<NodeData_<T>>* data = nullptr;
        uint32_t n = 0;
        const std::shared_ptr<NodeData_<T>>* begin() const { return data; }
        const std::shared_ptr<NodeData_<T>>* end()   const { return data + n; }
        uint32_t size() const { return n; }
        bool empty() const { return n == 0; }
        const std::shared_ptr<NodeData_<T>>& operator[](uint32_t i) const { return data[i]; }
    };

    View view(uint32_t blockIdx) const {
        s_lock lock(mutex);
        if (blockIdx == INVALID_IDX) return {};
        if (blockIdx >= blocks.size()) return {};
        const Block& b = blocks[blockIdx];
        if (b.count == 0) return {};
        return View{pool.data() + b.offset, b.count};
    }

    std::vector<std::shared_ptr<NodeData_<T>>> get(uint32_t blockIdx) const {
        s_lock lock(mutex);
        std::vector<std::shared_ptr<NodeData_<T>>> out;
        if (blockIdx == INVALID_IDX) return out;
        if (blockIdx >= blocks.size()) return out;
        const Block& b = blocks[blockIdx];
        out.reserve(b.count);
        for (uint32_t i = 0; i < b.count; ++i) out.push_back(pool[b.offset + i]);
        return out;
    }

    std::shared_ptr<NodeData_<T>> at(uint32_t blockIdx, uint32_t i) const {
        s_lock lock(mutex);
        if (blockIdx == INVALID_IDX) return nullptr;
        if (blockIdx >= blocks.size()) return nullptr;
        const Block& b = blocks[blockIdx];
        if (i >= b.count) return nullptr;
        return pool[b.offset + i];
    }

    uint32_t push(uint32_t blockIdx, const std::shared_ptr<NodeData_<T>>& pt) {
        u_lock lock(mutex);
        if (blockIdx == INVALID_IDX || blockIdx >= blocks.size()) {
            uint32_t offset = static_cast<uint32_t>(pool.size());
            pool.push_back(pt);
            blocks.push_back(Block{offset, 1, 1});
            return static_cast<uint32_t>(blocks.size() - 1);
        }
        Block& b = blocks[blockIdx];
        if (b.offset + b.count == pool.size()) {
            pool.push_back(pt);
            b.count++;
            b.capacity = b.count;
            return blockIdx;
        }
        
        uint32_t newOffset = static_cast<uint32_t>(pool.size());
        for (uint32_t i = 0; i < b.count; ++i) pool.push_back(pool[b.offset + i]);
        pool.push_back(pt);
        b.offset = newOffset;
        b.count++;
        b.capacity = b.count;
        return blockIdx;
    }

    uint32_t assign(uint32_t blockIdx, const std::vector<std::shared_ptr<NodeData_<T>>>& pts) {
        u_lock lock(mutex);
        if (blockIdx != INVALID_IDX && blockIdx < blocks.size()) {
            Block& b = blocks[blockIdx];
            if (pts.size() <= b.capacity) {
                for (size_t i = 0; i < pts.size(); ++i) pool[b.offset + i] = pts[i];
                for (size_t i = pts.size(); i < b.count; ++i) pool[b.offset + i] = nullptr;
                b.count = static_cast<uint32_t>(pts.size());
                return blockIdx;
            }
            freeBlocks.push_back(blockIdx);
        }
        return allocLocked(pts);
    }

    void release(uint32_t blockIdx) {
        if (blockIdx == INVALID_IDX) return;
        u_lock lock(mutex);
        if (blockIdx >= blocks.size()) return;
        Block& b = blocks[blockIdx];
        for (uint32_t i = 0; i < b.count; ++i) pool[b.offset + i] = nullptr;
        b.count = 0;
        freeBlocks.push_back(blockIdx);
    }

    uint32_t addSingle(const std::shared_ptr<NodeData_<T>>& pt) {
        u_lock lock(mutex);
        uint32_t offset = static_cast<uint32_t>(pool.size());
        pool.push_back(pt);
        blocks.push_back(Block{offset, 1, 1});
        return static_cast<uint32_t>(blocks.size() - 1);
    }

    void clear() {
        u_lock lock(mutex);
        pool.clear();
        blocks.clear();
        freeBlocks.clear();
    }

    size_t totalPoints() const {
        s_lock lock(mutex);
        size_t n = 0;
        for (const auto& b : blocks) n += b.count;
        return n;
    }
};

template<typename T>
struct OctreeNodeStore {
    static constexpr size_t StripeCount = 1024;

    std::deque<OctreeNode_<T>> NodeList;
    std::vector<uint32_t> freeBlocks;
    mutable std::shared_mutex mutex;
    mutable std::unique_ptr<std::shared_mutex[]> stripes;

    PointStore<T> points;

    OctreeNodeStore() : stripes(new std::shared_mutex[StripeCount]) {}

    std::shared_mutex& stripe(uint32_t idx) const {
        return stripes[idx % StripeCount];
    }

    size_t size() const {
        s_lock lock(mutex);
        return NodeList.size();
    }

    uint32_t add(const OctreeNode_<T>& node) {
        u_lock lock(mutex);
        if (NodeList.size() >= u32M) return INVALID_IDX;
        NodeList.push_back(node);
        return static_cast<uint32_t>(NodeList.size() - 1);
    }

    uint32_t allocChildren() {
        u_lock lock(mutex);
        if (!freeBlocks.empty()) {
            uint32_t idx = freeBlocks.back();
            freeBlocks.pop_back();
            for (int i = 0; i < 8; ++i) NodeList[idx + i] = OctreeNode_<T>();
            return idx;
        }
        if (NodeList.size() + 8 >= u32M) return INVALID_IDX;
        uint32_t first = static_cast<uint32_t>(NodeList.size());
        for (int i = 0; i < 8; ++i) NodeList.emplace_back();
        return first;
    }

    void freeChildren(uint32_t firstChild) {
        if (firstChild == INVALID_IDX) return;
        u_lock lock(mutex);
        if (firstChild + 7 >= NodeList.size()) return;
        for (int i = 0; i < 8; ++i) NodeList[firstChild + i] = OctreeNode_<T>();
        freeBlocks.push_back(firstChild);
    }

    OctreeNode_<T>* ptr(uint32_t idx) {
        if (idx == INVALID_IDX || idx >= NodeList.size()) return nullptr;
        return &NodeList[idx];
    }

    const OctreeNode_<T>* ptr(uint32_t idx) const {
        if (idx == INVALID_IDX || idx >= NodeList.size()) return nullptr;
        return &NodeList[idx];
    }

    OctreeNode_<T>& operator[](uint32_t idx) { return NodeList[idx]; }
    const OctreeNode_<T>& operator[](uint32_t idx) const { return NodeList[idx]; }

    OctreeNode_<T> get(uint32_t idx) const {
        s_lock lock(mutex);
        if (idx < NodeList.size()) return NodeList[idx];
        return OctreeNode_<T>();
    }

    bool valid(uint32_t idx) const {
        s_lock lock(mutex);
        return idx != INVALID_IDX && idx < NodeList.size();
    }

    void clear() {
        u_lock lock(mutex);
        NodeList.clear();
        freeBlocks.clear();
        points.clear();
    }

    size_t memoryUsage() const {
        s_lock lock(mutex);
        return NodeList.size() * sizeof(OctreeNode_<T>);
    }
};

template<typename T>
struct RayHit_ {
    std::shared_ptr<NodeData_<T>> node;
    float distance = 0;
    Vec3 normal;
    Vec3 hitPoint;
};
    
struct Ray {
    Vec3 origin;
    Vec3 dir;
    Vec3 invDir;
    uint8_t sign[3];
    uint8_t signMask;
    Ray(const Vec3& orig, const Vec3& dir) : origin(orig), dir(dir) {
        invDir = dir.cwiseInverse();
        sign[0] = (invDir[0] < 0);
        sign[1] = (invDir[1] < 0);
        sign[2] = (invDir[2] < 0);
        signMask = (sign[0] | sign[1] << 1 | sign[2] << 2);
    }
};

struct ProgressiveAccum {
    bool valid = false;
    int  samples = 0;
    int  width = 0;
    int  height = 0;
    Vec3 camOrigin{0, 0, 0};
    Vec3 camDir{0, 0, 0};
    Vec3 camUp{0, 0, 0};
    float tanfovx = 0.0f;
    float tanfovy = 0.0f;
    uint64_t sceneEpoch = ~0ull;
};

}