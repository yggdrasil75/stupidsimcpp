#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Grid {

using Vec3 = Eigen::Vector3f;
using BoundingBox = std::pair<Vec3, Vec3>;

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

///@brief Packed extent of a single voxel (fields store count - 1)
static constexpr uint32_t EXTENT_UNIT = 0u;
///@brief Largest off-lattice error, in cells, still treated as on the lattice
static constexpr float LATTICE_EPS = 1e-3f;
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

///@brief Expands a packed extent into per-axis cell counts
///@param e Packed extent as produced by packExtent
///@return Cell spans as floats, each at least 1.0f
static inline Vec3 unpackExtent(uint32_t e) {
    return Vec3(static_cast<float>((e         & 0x3FFu) + 1u),
                static_cast<float>(((e >> 10) & 0x3FFu) + 1u),
                static_cast<float>(((e >> 20) & 0x3FFu) + 1u));
}


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

///@brief Geometry a grid point can take beyond a single cube
enum class ShapeType : uint8_t {
    BOX = 0,
    OBB = 1,
    CAPSULE = 2
};

///@brief Integer lattice coordinate
using Cell = std::array<int64_t, 3>;

///@brief Rounds a lattice-relative coordinate to its cell index
static inline int64_t cellRound(float v) {
    return static_cast<int64_t>(std::llround(v));
}

///@brief Packs a unit quaternion into 32 bits (smallest-three, 2 bit index + 3 x 10 bit)
///@param qin Quaternion to pack, normalized internally
///@return Packed representation matching unpackQuat in shapes.glsl
static inline uint32_t packQuat(const Eigen::Quaternionf& qin) {
    Eigen::Quaternionf q = qin.normalized();
    float c[4] = {q.x(), q.y(), q.z(), q.w()};
    int big = 0;
    for (int i = 1; i < 4; ++i) {
        if (std::abs(c[i]) > std::abs(c[big])) big = i;
    }
    float sgn = c[big] < 0.0f ? -1.0f : 1.0f;
    uint32_t out = static_cast<uint32_t>(big);
    int shift = 2;
    const float inv = 1.0f / 0.70710678f;
    for (int i = 0; i < 4; ++i) {
        if (i == big) continue;
        float v = std::clamp(c[i] * sgn * inv, -1.0f, 1.0f);
        uint32_t u = static_cast<uint32_t>(std::lround((v * 0.5f + 0.5f) * 1023.0f));
        out |= (u & 0x3FFu) << shift;
        shift += 10;
    }
    return out;
}

///@brief Inverse of packQuat
///@param p Packed quaternion
///@return Unit quaternion
static inline Eigen::Quaternionf unpackQuat(uint32_t p) {
    int big = static_cast<int>(p & 3u);
    float c[4];
    float sum = 0.0f;
    int shift = 2;
    for (int i = 0; i < 4; ++i) {
        if (i == big) continue;
        float v = (static_cast<float>((p >> shift) & 0x3FFu) / 1023.0f * 2.0f - 1.0f) * 0.70710678f;
        c[i] = v;
        sum += v * v;
        shift += 10;
    }
    c[big] = std::sqrt(std::max(0.0f, 1.0f - sum));
    return Eigen::Quaternionf(c[3], c[0], c[1], c[2]).normalized();
}

///@brief Primitive attached to a point. The owner supplies the centre and the voxel size.
///       BOX ignores every field. OBB: half = half extents in the local frame.
///       CAPSULE: half.x = radius, half.y = half segment length, axis = rot * +Y.
struct Shape {
    ///@brief Local to world rotation
    Eigen::Quaternionf rot{1.0f, 0.0f, 0.0f, 0.0f};
    ///@brief Half extents (OBB) or radius / half length (CAPSULE)
    Vec3 half{0.0f, 0.0f, 0.0f};
    ShapeType type = ShapeType::BOX;
    ///@brief Keeps the byte image deterministic so the render cache hash is stable
    uint8_t pad_[3] = {0, 0, 0};

    ///@brief Builds an oriented box
    ///@param q Local to world rotation
    ///@param halfExtents Half extents in the local frame
    static Shape obb(const Eigen::Quaternionf& q, const Vec3& halfExtents) {
        Shape s;
        s.type = ShapeType::OBB;
        s.rot = q.normalized();
        s.half = halfExtents;
        return s;
    }

    ///@brief Builds a capsule around the local Y axis
    ///@param q Local to world rotation
    ///@param radius Capsule radius
    ///@param halfLength Half length of the inner segment
    static Shape capsule(const Eigen::Quaternionf& q, float radius, float halfLength) {
        Shape s;
        s.type = ShapeType::CAPSULE;
        s.rot = q.normalized();
        s.half = Vec3(radius, halfLength, 0.0f);
        return s;
    }

    ///@brief Builds a capsule between two world space segment endpoints
    ///@param a First endpoint
    ///@param b Second endpoint
    ///@param radius Capsule radius
    ///@param outCenter Receives the midpoint to use as the point position
    static Shape capsuleBetween(const Vec3& a, const Vec3& b, float radius, Vec3& outCenter) {
        outCenter = (a + b) * 0.5f;
        Vec3 d = b - a;
        float len = d.norm();
        Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
        if (len > 1e-8f) q = Eigen::Quaternionf::FromTwoVectors(Vec3::UnitY(), d / len);
        return capsule(q, radius, 0.5f * len);
    }

    bool isBox() const {
        return type == ShapeType::BOX;
    }

    ///@brief True when the rotation is (up to sign) the identity
    bool identityRot() const {
        return std::abs(std::abs(rot.w()) - 1.0f) < 1e-6f;
    }

    ///@brief World space capsule axis
    Vec3 axis() const {
        return rot * Vec3::UnitY();
    }

    float radius() const {
        return half.x();
    }

    float halfLength() const {
        return half.y();
    }

    ///@brief World point to the shape's local frame
    ///@param p World point
    ///@param c Shape centre
    Vec3 toLocal(const Vec3& p, const Vec3& c) const {
        return rot.conjugate() * (p - c);
    }

    ///@brief Local point to world space
    ///@param p Local point
    ///@param c Shape centre
    Vec3 toWorld(const Vec3& p, const Vec3& c) const {
        return rot * p + c;
    }

    ///@brief Signed distance from a world point
    ///@param p World point
    ///@param c Shape centre
    ///@param size Voxel size, used by BOX
    float sdf(const Vec3& p, const Vec3& c, float size) const {
        switch (type) {
            case ShapeType::OBB: {
                Vec3 q = toLocal(p, c).cwiseAbs() - half;
                return q.cwiseMax(0.0f).norm() + std::min(q.maxCoeff(), 0.0f);
            }
            case ShapeType::CAPSULE: {
                Vec3 l = toLocal(p, c);
                l.y() -= std::clamp(l.y(), -half.y(), half.y());
                return l.norm() - half.x();
            }
            default: {
                Vec3 q = (p - c).cwiseAbs() - Vec3::Constant(0.5f * size);
                return q.cwiseMax(0.0f).norm() + std::min(q.maxCoeff(), 0.0f);
            }
        }
    }

    ///@brief Whether a world point is inside the shape
    ///@param p World point
    ///@param c Shape centre
    ///@param size Voxel size, used by BOX
    ///@param tol Extra distance still counted as inside
    bool contains(const Vec3& p, const Vec3& c, float size, float tol = 0.0f) const {
        return sdf(p, c, size) <= tol;
    }

    ///@brief World axis-aligned bounds
    ///@param c Shape centre
    ///@param size Voxel size, used by BOX
    BoundingBox aabb(const Vec3& c, float size) const {
        switch (type) {
            case ShapeType::OBB: {
                Vec3 r = rot.toRotationMatrix().cwiseAbs() * half;
                return {c - r, c + r};
            }
            case ShapeType::CAPSULE: {
                Vec3 e = axis() * half.y();
                Vec3 r = e.cwiseAbs() + Vec3::Constant(half.x());
                return {c - r, c + r};
            }
            default: {
                Vec3 h = Vec3::Constant(0.5f * size);
                return {c - h, c + h};
            }
        }
    }

    ///@brief Enclosed volume in world units
    ///@param size Voxel size, used by BOX
    float volume(float size) const {
        switch (type) {
            case ShapeType::OBB:
                return 8.0f * half.x() * half.y() * half.z();
            case ShapeType::CAPSULE: {
                const float r = half.x();
                return static_cast<float>(M_PI) * r * r * (2.0f * half.y() + 4.0f / 3.0f * r);
            }
            default:
                return size * size * size;
        }
    }

    ///@brief Ray hit against the shape
    ///@param ray Query ray
    ///@param c Shape centre
    ///@param size Voxel size, used by BOX
    ///@param t Receives the entry distance, or the exit distance when the origin is inside
    ///@param normal Receives the surface normal at t, flipped to face the ray when inside
    ///@param tExit Optionally receives the exit distance
    ///@return True on hit
    bool intersect(const Ray& ray, const Vec3& c, float size, float& t, Vec3& normal, float* tExit = nullptr) const {
        float tIn = 0.0f;
        float tOut = 0.0f;
        bool ok = false;
        Vec3 n = Vec3::Zero();
        if (type == ShapeType::CAPSULE) {
            ok = rayCapsule(ray, c, tIn, tOut);
            if (ok) {
                bool inside = tIn < 0.0f;
                t = inside ? tOut : tIn;
                n = capsuleNormal(ray.origin + ray.dir * t, c);
                if (inside) n = -n;
            }
        } else {
            Vec3 h = isBox() ? Vec3::Constant(0.5f * size) : half;
            Vec3 o = isBox() ? Vec3(ray.origin - c) : toLocal(ray.origin, c);
            Vec3 d = isBox() ? ray.dir : Vec3(rot.conjugate() * ray.dir);
            ok = raySlab(o, d, h, tIn, tOut, n);
            if (ok) {
                bool inside = tIn < 0.0f;
                t = inside ? tOut : tIn;
                if (!isBox()) n = rot * n;
            }
        }
        if (tExit) *tExit = ok ? tOut : 0.0f;
        if (!ok) return false;
        normal = n;
        return true;
    }

    ///@brief Visits the centre of every lattice cell inside the shape. An axis-aligned OBB is
    ///       tiled from its minimum corner; every other shape uses a lattice through the centre.
    ///@param c Shape centre
    ///@param cell Lattice spacing
    ///@param fn Callback taking the cell centre as Vec3
    template<typename Fn>
    void forEachCell(const Vec3& c, float cell, Fn&& fn) const {
        if (cell <= 0.0f) return;
        if (isBox()) {
            fn(c);
            return;
        }
        if (type == ShapeType::OBB && identityRot()) {
            Vec3 origin = c - half + Vec3::Constant(0.5f * cell);
            Vec3 cnt = half * 2.0f / cell;
            int64_t nx = std::max<int64_t>(1, cellRound(cnt.x()));
            int64_t ny = std::max<int64_t>(1, cellRound(cnt.y()));
            int64_t nz = std::max<int64_t>(1, cellRound(cnt.z()));
            for (int64_t z = 0; z < nz; ++z) {
                for (int64_t y = 0; y < ny; ++y) {
                    for (int64_t x = 0; x < nx; ++x) {
                        fn(origin + Vec3(float(x), float(y), float(z)) * cell);
                    }
                }
            }
            return;
        }
        BoundingBox b = aabb(c, cell);
        Vec3 lo = ((b.first - c) / cell).array().floor();
        Vec3 hi = ((b.second - c) / cell).array().ceil();
        for (int64_t z = int64_t(lo.z()); z <= int64_t(hi.z()); ++z) {
            for (int64_t y = int64_t(lo.y()); y <= int64_t(hi.y()); ++y) {
                for (int64_t x = int64_t(lo.x()); x <= int64_t(hi.x()); ++x) {
                    Vec3 p = c + Vec3(float(x), float(y), float(z)) * cell;
                    if (sdf(p, c, cell) <= 0.0f) fn(p);
                }
            }
        }
    }

    ///@brief Number of lattice cells the shape voxelises into
    ///@param c Shape centre
    ///@param cell Lattice spacing
    size_t cellCount(const Vec3& c, float cell) const {
        size_t n = 0;
        forEachCell(c, cell, [&](const Vec3&) { ++n; });
        return n;
    }

private:
    ///@brief Slab test against a box centred on the origin
    ///@param n Receives the face normal at the entry (or exit when inside)
    static bool raySlab(const Vec3& o, const Vec3& d, const Vec3& h, float& tIn, float& tOut, Vec3& n) {
        Vec3 inv = d.cwiseInverse();
        Vec3 t0 = (-h - o).cwiseProduct(inv);
        Vec3 t1 = (h - o).cwiseProduct(inv);
        Vec3 tmin3 = t0.cwiseMin(t1);
        Vec3 tmax3 = t0.cwiseMax(t1);
        tIn = tmin3.maxCoeff();
        tOut = tmax3.minCoeff();
        if (tOut < std::max(0.0f, tIn)) return false;
        bool inside = tIn < 0.0f;
        const Vec3& slab = inside ? tmax3 : tmin3;
        float key = inside ? tOut : tIn;
        int ax = 2;
        if (key == slab.x()) ax = 0;
        else if (key == slab.y()) ax = 1;
        n = Vec3::Zero();
        float s = d[ax] < 0.0f ? 1.0f : -1.0f;
        n[ax] = inside ? -s : s;
        return true;
    }

    ///@brief Ray interval through the capsule. The capsule is the union of a clipped cylinder and
    ///       two spheres; all three are convex and overlap, so the interval is [min entry, max exit].
    bool rayCapsule(const Ray& ray, const Vec3& c, float& tIn, float& tOut) const {
        const Vec3 o = toLocal(ray.origin, c);
        const Vec3 d = rot.conjugate() * ray.dir;
        const float r = half.x();
        const float h = half.y();
        bool any = false;
        tIn = std::numeric_limits<float>::max();
        tOut = -std::numeric_limits<float>::max();
        auto add = [&](float a, float b) {
            any = true;
            tIn = std::min(tIn, a);
            tOut = std::max(tOut, b);
        };

        float a = d.x() * d.x() + d.z() * d.z();
        float bq = 2.0f * (o.x() * d.x() + o.z() * d.z());
        float cq = o.x() * o.x() + o.z() * o.z() - r * r;
        if (a > 1e-12f) {
            float disc = bq * bq - 4.0f * a * cq;
            if (disc >= 0.0f) {
                float sq = std::sqrt(disc);
                float t0 = (-bq - sq) / (2.0f * a);
                float t1 = (-bq + sq) / (2.0f * a);
                if (std::abs(d.y()) > 1e-12f) {
                    float ta = (-h - o.y()) / d.y();
                    float tb = (h - o.y()) / d.y();
                    float lo = std::max(t0, std::min(ta, tb));
                    float hi = std::min(t1, std::max(ta, tb));
                    if (hi >= lo) add(lo, hi);
                } else if (std::abs(o.y()) <= h) {
                    add(t0, t1);
                }
            }
        } else if (cq <= 0.0f && std::abs(d.y()) > 1e-12f) {
            float ta = (-h - o.y()) / d.y();
            float tb = (h - o.y()) / d.y();
            add(std::min(ta, tb), std::max(ta, tb));
        }
        for (float sy : {-h, h}) {
            Vec3 oc = o - Vec3(0.0f, sy, 0.0f);
            float b2 = oc.dot(d);
            float c2 = oc.squaredNorm() - r * r;
            float disc = b2 * b2 - c2;
            if (disc < 0.0f) continue;
            float sq = std::sqrt(disc);
            add(-b2 - sq, -b2 + sq);
        }
        if (!any) return false;
        return tOut >= std::max(0.0f, tIn);
    }

    ///@brief Outward normal at a world point on the capsule surface
    Vec3 capsuleNormal(const Vec3& p, const Vec3& c) const {
        Vec3 l = toLocal(p, c);
        l.y() -= std::clamp(l.y(), -half.y(), half.y());
        float n = l.norm();
        if (n <= 1e-8f) return rot * Vec3::UnitX();
        return rot * Vec3(l / n);
    }
};

///@brief Trailing 16 bytes of GPURenderData. Layout matches shapes.glsl.
struct GPUShapeWords {
    ///@brief packQuat() of the rotation, unused for BOX
    uint32_t rot = 0u;
    ///@brief Capsule radius
    float a = 0.0f;
    ///@brief Capsule half length
    float b = 0.0f;
    ///@brief ShapeType as integer
    uint32_t type = 0u;
};

///@brief Converts a Shape into its GPU words. OBB half extents travel as cell counts in extent.
///@param s Shape to pack
///@param size Voxel size the OBB counts are relative to
///@param extent Extent word to rewrite for OBB, flag bits preserved
static inline GPUShapeWords packShapeWords(const Shape& s, float size, uint32_t& extent) {
    GPUShapeWords w;
    w.type = static_cast<uint32_t>(s.type);
    if (s.isBox()) return w;
    w.rot = packQuat(s.rot);
    if (s.type == ShapeType::OBB) {
        Vec3 cnt = s.half * 2.0f / std::max(size, 1e-12f);
        uint32_t flags = extent & (EXTENT_STATIC_BIT | EXTENT_REUSE_BIT);
        uint32_t nx = uint32_t(std::max<int64_t>(1, cellRound(cnt.x())));
        uint32_t ny = uint32_t(std::max<int64_t>(1, cellRound(cnt.y())));
        uint32_t nz = uint32_t(std::max<int64_t>(1, cellRound(cnt.z())));
        extent = packExtent(nx, ny, nz) | flags;
    } else {
        w.a = s.half.x();
        w.b = s.half.y();
    }
    return w;
}

///@brief Fits a capsule that voxelises back to exactly the given cells. Axis from PCA, then a
///       small search over radius and half length in quarter-cell steps.
///@param pts World cell centres on a lattice of spacing cell
///@param cell Lattice spacing
///@param out Receives the capsule on success
///@param outCenter Receives the lattice-snapped centre on success
///@return True when an exact cover was found
static inline bool fitCapsuleExact(const std::vector<Vec3>& pts, float cell, Shape& out, Vec3& outCenter) {
    const size_t n = pts.size();
    if (n < 3) return false;
    Vec3 mean = Vec3::Zero();
    for (const Vec3& p : pts) mean += p;
    mean /= float(n);
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const Vec3& p : pts) {
        Vec3 d = p - mean;
        cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    if (es.info() != Eigen::Success) return false;
    Vec3 ax = es.eigenvectors().col(2).normalized();
    Vec3 aa = ax.cwiseAbs();
    if (aa.maxCoeff() > 0.999f) {
        int major = 2;
        if (aa.maxCoeff() == aa.x()) major = 0;
        else if (aa.maxCoeff() == aa.y()) major = 1;
        ax = Vec3::Zero();
        ax[major] = 1.0f;
    }
    const Vec3 anchor = pts[0];
    Vec3 center = anchor + ((mean - anchor) / cell).array().round().matrix() * cell;
    float pmax = 0.0f;
    float perpMax = 0.0f;
    for (const Vec3& p : pts) {
        Vec3 d = p - center;
        float a = d.dot(ax);
        pmax = std::max(pmax, std::abs(a));
        perpMax = std::max(perpMax, (d - ax * a).norm());
    }
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Vec3::UnitY(), ax);
    std::unordered_map<Cell, char, Vec3i64Hash> want;
    want.reserve(n * 2);
    for (const Vec3& p : pts) {
        Vec3 r = (p - center) / cell;
        want[{cellRound(r.x()), cellRound(r.y()), cellRound(r.z())}] = 1;
    }
    auto matches = [&](const Shape& s) {
        size_t seen = 0;
        bool ok = true;
        s.forEachCell(center, cell, [&](const Vec3& p) {
            if (!ok) return;
            Vec3 r = (p - center) / cell;
            if (!want.count({cellRound(r.x()), cellRound(r.y()), cellRound(r.z())})) {
                ok = false;
            } else {
                ++seen;
            }
        });
        return ok && seen == n;
    };
    const float step = 0.25f * cell;
    for (int k = 0; k <= 8; ++k) {
        float r = perpMax + step * float(k) + 1e-4f * cell;
        for (int j = 0; j <= 4; ++j) {
            float hl = std::max(0.0f, pmax - r + step * float(j));
            Shape s = Shape::capsule(q, r, hl);
            if (!matches(s)) continue;
            out = s;
            outCenter = center;
            return true;
        }
    }
    return false;
}

///@brief Fits an axis-aligned box that covers exactly the given cells
///@param pts World cell centres on a lattice of spacing cell
///@param cell Lattice spacing
///@param out Receives an identity-rotation OBB on success
///@param outCenter Receives the box centre on success
///@return True when the cells fill their bounding box with no gaps or duplicates
static inline bool fitBoxExact(const std::vector<Vec3>& pts, float cell, Shape& out, Vec3& outCenter) {
    if (pts.empty()) return false;
    Vec3 lo = pts[0];
    Vec3 hi = pts[0];
    for (const Vec3& p : pts) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    Vec3 cnt = ((hi - lo) / cell).array().round() + 1.0f;
    if (double(cnt.x()) * cnt.y() * cnt.z() != double(pts.size())) return false;
    std::unordered_map<Cell, char, Vec3i64Hash> seen;
    for (const Vec3& p : pts) {
        Vec3 r = (p - lo) / cell;
        Cell c{cellRound(r.x()), cellRound(r.y()), cellRound(r.z())};
        Vec3 snapped(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
        if ((r - snapped).cwiseAbs().maxCoeff() > LATTICE_EPS) return false;
        if (!seen.emplace(c, 1).second) return false;
    }
    outCenter = (lo + hi) * 0.5f;
    out = Shape::obb(Eigen::Quaternionf::Identity(), cnt * (0.5f * cell));
    return true;
}

///@brief Greedily carves a set of lattice cells into axis-aligned boxes (x run, then y slab,
///       then z slab, same growth order as the render merge)
///@param cells Integer cell coordinates
///@param fn Callback taking the box's minimum cell and its per-axis cell counts
template<typename Fn>
static inline void greedyBoxes(std::vector<Cell> cells, Fn&& fn) {
    std::sort(cells.begin(), cells.end());
    std::unordered_map<Cell, char, Vec3i64Hash> free;
    free.reserve(cells.size() * 2);
    for (const Cell& c : cells) free[c] = 1;
    auto slabFree = [&](const Cell& base, int64_t sx, int64_t sy) {
        for (int64_t dy = 0; dy < sy; ++dy) {
            for (int64_t dx = 0; dx < sx; ++dx) {
                auto it = free.find({base[0] + dx, base[1] + dy, base[2]});
                if (it == free.end() || !it->second) return false;
            }
        }
        return true;
    };
    auto claim = [&](const Cell& base, int64_t sx, int64_t sy) {
        for (int64_t dy = 0; dy < sy; ++dy) {
            for (int64_t dx = 0; dx < sx; ++dx) {
                free[{base[0] + dx, base[1] + dy, base[2]}] = 0;
            }
        }
    };
    const int64_t lim = static_cast<int64_t>(EXTENT_MAX);
    for (const Cell& c : cells) {
        if (!free[c]) continue;
        int64_t ex = 1;
        int64_t ey = 1;
        int64_t ez = 1;
        claim(c, 1, 1);
        while (ex < lim && slabFree({c[0] + ex, c[1], c[2]}, 1, 1)) {
            claim({c[0] + ex, c[1], c[2]}, 1, 1);
            ++ex;
        }
        while (ey < lim && slabFree({c[0], c[1] + ey, c[2]}, ex, 1)) {
            claim({c[0], c[1] + ey, c[2]}, ex, 1);
            ++ey;
        }
        while (ez < lim && slabFree({c[0], c[1], c[2] + ez}, ex, ey)) {
            claim({c[0], c[1], c[2] + ez}, ex, ey);
            ++ez;
        }
        fn(c, Cell{ex, ey, ez});
    }
}

namespace detail {
    template<typename T, typename = void>
    struct has_member_merge : std::false_type {};
    template<typename T>
    struct has_member_merge<T, std::void_t<decltype(T::merge(std::declval<const T&>(), std::declval<const T&>()))>>
        : std::true_type {};
    template<typename T, typename = void>
    struct has_adl_merge : std::false_type {};
    template<typename T>
    struct has_adl_merge<T, std::void_t<decltype(merge(std::declval<const T&>(), std::declval<const T&>()))>>
        : std::true_type {};
    template<typename T, typename = void>
    struct has_eq : std::false_type {};
    template<typename T>
    struct has_eq<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>>
        : std::true_type {};
}

///@brief True when T provides T::merge(a, b) or an ADL merge(a, b)
template<typename T>
inline constexpr bool hasMergeFunction = detail::has_member_merge<T>::value || detail::has_adl_merge<T>::value;

///@brief Whether two payloads may share one primitive: always for mergeable types, only when
///       equal for comparable types, never for opaque types
template<typename T>
inline bool payloadMergeable(const T& a, const T& b) {
    if constexpr (hasMergeFunction<T>) {
        (void)a;
        (void)b;
        return true;
    } else if constexpr (detail::has_eq<T>::value) {
        return a == b;
    } else {
        (void)a;
        (void)b;
        return false;
    }
}

///@brief Combines two payloads, falling back to the first when T has no merge
template<typename T>
inline T mergePayload(const T& a, const T& b) {
    if constexpr (detail::has_member_merge<T>::value) {
        return T::merge(a, b);
    } else if constexpr (detail::has_adl_merge<T>::value) {
        return merge(a, b);
    } else {
        (void)b;
        return a;
    }
}

}
