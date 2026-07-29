#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>

#include "../eigen/Eigen/Dense"
#include "grid/grid3eigen.hpp"

namespace Character {

using Vec3 = Eigen::Vector3f;
using Grid::BodyType;

static constexpr int OID_SKELETON = 700;
static constexpr int OID_MUSCLE   = 701;
static constexpr int OID_FLESH    = 702;

namespace Palette {
    static const Vec3 BONE  (0.92f, 0.90f, 0.82f);
    static const Vec3 MUSCLE(0.62f, 0.14f, 0.14f);
    static const Vec3 FLESH (0.85f, 0.66f, 0.55f);
}

inline float smin(float a, float b, float k) {
    float h = std::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return b * (1.0f - h) + a * h - k * h * (1.0f - h);
}

inline Vec3 closestOnSegment(const Vec3& p, const Vec3& a, const Vec3& b, float& tOut) {
    Vec3 ab = b - a;
    float denom = ab.squaredNorm();
    float t = denom > 1e-12f ? (p - a).dot(ab) / denom : 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);
    tOut = t;
    return a + ab * t;
}

inline float distToSegment(const Vec3& p, const Vec3& a, const Vec3& b, float& tOut) {
    return (p - closestOnSegment(p, a, b, tOut)).norm();
}

inline float sdCapsule(const Vec3& p, const Vec3& a, const Vec3& b, float r) {
    float t;
    return distToSegment(p, a, b, t) - r;
}

inline float sdEllipsoid(const Vec3& p, const Vec3& c, const Vec3& rad) {
    Vec3 q = (p - c).cwiseQuotient(rad);
    float k0 = q.norm();
    if (k0 < 1e-6f) return -rad.minCoeff();
    Vec3 q2 = q.cwiseQuotient(rad);
    float k1 = q2.norm();
    return k0 * (k0 - 1.0f) / std::max(k1, 1e-6f);
}

enum Joint {
    J_PELVIS, J_SPINE, J_CHEST, J_NECK, J_HEAD, J_HEAD_TOP,
    J_CLAV_L, J_SHOULDER_L, J_ELBOW_L, J_WRIST_L, J_HAND_L,
    J_CLAV_R, J_SHOULDER_R, J_ELBOW_R, J_WRIST_R, J_HAND_R,
    J_HIP_L, J_KNEE_L, J_ANKLE_L, J_HEEL_L, J_TOE_L,
    J_HIP_R, J_KNEE_R, J_ANKLE_R, J_HEEL_R, J_TOE_R,
    J_COUNT
};

inline std::array<Vec3, J_COUNT> normalisedJoints() {
    std::array<Vec3, J_COUNT> j{};
    j[J_PELVIS]   = Vec3( 0.00f, 0.00f, 0.500f);
    j[J_SPINE]    = Vec3( 0.00f,-0.01f, 0.600f);
    j[J_CHEST]    = Vec3( 0.00f,-0.01f, 0.720f);
    j[J_NECK]     = Vec3( 0.00f, 0.00f, 0.810f);
    j[J_HEAD]     = Vec3( 0.00f, 0.01f, 0.880f);
    j[J_HEAD_TOP] = Vec3( 0.00f, 0.01f, 0.985f);

    j[J_CLAV_L]     = Vec3( 0.045f, 0.00f, 0.795f);
    j[J_SHOULDER_L] = Vec3( 0.115f, 0.00f, 0.780f);
    j[J_ELBOW_L]    = Vec3( 0.150f,-0.01f, 0.620f);
    j[J_WRIST_L]    = Vec3( 0.165f, 0.00f, 0.470f);
    j[J_HAND_L]     = Vec3( 0.170f, 0.01f, 0.415f);

    j[J_CLAV_R]     = Vec3(-0.045f, 0.00f, 0.795f);
    j[J_SHOULDER_R] = Vec3(-0.115f, 0.00f, 0.780f);
    j[J_ELBOW_R]    = Vec3(-0.150f,-0.01f, 0.620f);
    j[J_WRIST_R]    = Vec3(-0.165f, 0.00f, 0.470f);
    j[J_HAND_R]     = Vec3(-0.170f, 0.01f, 0.415f);

    j[J_HIP_L]   = Vec3( 0.052f, 0.00f, 0.480f);
    j[J_KNEE_L]  = Vec3( 0.058f, 0.01f, 0.260f);
    j[J_ANKLE_L] = Vec3( 0.058f, 0.00f, 0.035f);
    j[J_HEEL_L]  = Vec3( 0.058f,-0.03f, 0.010f);
    j[J_TOE_L]   = Vec3( 0.058f, 0.08f, 0.010f);

    j[J_HIP_R]   = Vec3(-0.052f, 0.00f, 0.480f);
    j[J_KNEE_R]  = Vec3(-0.058f, 0.01f, 0.260f);
    j[J_ANKLE_R] = Vec3(-0.058f, 0.00f, 0.035f);
    j[J_HEEL_R]  = Vec3(-0.058f,-0.03f, 0.010f);
    j[J_TOE_R]   = Vec3(-0.058f, 0.08f, 0.010f);
    return j;
}

struct Bone {
    Joint a, b;
    float boneRadius;
};

inline std::vector<Bone> boneList() {
    return {
        { J_PELVIS, J_SPINE,  0.022f },
        { J_SPINE,  J_CHEST,  0.024f },
        { J_CHEST,  J_NECK,   0.016f },
        { J_NECK,   J_HEAD,   0.014f },
        { J_CHEST,  J_CLAV_L, 0.012f },
        { J_CLAV_L, J_SHOULDER_L, 0.012f },
        { J_CHEST,  J_CLAV_R, 0.012f },
        { J_CLAV_R, J_SHOULDER_R, 0.012f },
        { J_SHOULDER_L, J_ELBOW_L, 0.013f },
        { J_ELBOW_L,    J_WRIST_L, 0.011f },
        { J_WRIST_L,    J_HAND_L,  0.010f },
        { J_SHOULDER_R, J_ELBOW_R, 0.013f },
        { J_ELBOW_R,    J_WRIST_R, 0.011f },
        { J_WRIST_R,    J_HAND_R,  0.010f },
        { J_PELVIS, J_HIP_L, 0.020f },
        { J_PELVIS, J_HIP_R, 0.020f },
        { J_HIP_L,   J_KNEE_L,  0.018f },
        { J_KNEE_L,  J_ANKLE_L, 0.014f },
        { J_ANKLE_L, J_HEEL_L,  0.011f },
        { J_ANKLE_L, J_TOE_L,   0.010f },
        { J_HIP_R,   J_KNEE_R,  0.018f },
        { J_KNEE_R,  J_ANKLE_R, 0.014f },
        { J_ANKLE_R, J_HEEL_R,  0.011f },
        { J_ANKLE_R, J_TOE_R,   0.010f },
    };
}

struct Muscle {
    Joint a, b;
    float baseRadius;
    float bellyBulge;
    float bellyAt;
};

inline std::vector<Muscle> muscleList() {
    return {
        { J_PELVIS, J_SPINE,  0.075f, 0.030f, 0.55f },
        { J_SPINE,  J_CHEST,  0.090f, 0.045f, 0.60f },
        { J_CHEST,  J_NECK,   0.050f, 0.020f, 0.50f },
        { J_NECK,   J_HEAD,   0.032f, 0.010f, 0.50f },
        { J_CLAV_L, J_SHOULDER_L, 0.045f, 0.030f, 0.85f },
        { J_CLAV_R, J_SHOULDER_R, 0.045f, 0.030f, 0.85f },
        { J_SHOULDER_L, J_ELBOW_L, 0.038f, 0.024f, 0.45f },
        { J_SHOULDER_R, J_ELBOW_R, 0.038f, 0.024f, 0.45f },
        { J_ELBOW_L, J_WRIST_L, 0.030f, 0.018f, 0.30f },
        { J_ELBOW_R, J_WRIST_R, 0.030f, 0.018f, 0.30f },
        { J_WRIST_L, J_HAND_L, 0.022f, 0.006f, 0.50f },
        { J_WRIST_R, J_HAND_R, 0.022f, 0.006f, 0.50f },
        { J_PELVIS, J_HIP_L, 0.058f, 0.028f, 0.70f },
        { J_PELVIS, J_HIP_R, 0.058f, 0.028f, 0.70f },
        { J_HIP_L, J_KNEE_L, 0.055f, 0.035f, 0.45f },
        { J_HIP_R, J_KNEE_R, 0.055f, 0.035f, 0.45f },
        { J_KNEE_L, J_ANKLE_L, 0.042f, 0.028f, 0.30f },
        { J_KNEE_R, J_ANKLE_R, 0.042f, 0.028f, 0.30f },
        { J_ANKLE_L, J_TOE_L, 0.026f, 0.008f, 0.40f },
        { J_ANKLE_R, J_TOE_R, 0.026f, 0.008f, 0.40f },
        { J_ANKLE_L, J_HEEL_L, 0.024f, 0.006f, 0.50f },
        { J_ANKLE_R, J_HEEL_R, 0.024f, 0.006f, 0.50f },
    };
}

inline float muscleRadiusAt(const Muscle& m, float t) {
    float d = (t - m.bellyAt);
    float bulge = std::exp(-6.0f * d * d);
    float endFade = std::min(t, 1.0f - t) * 4.0f;
    endFade = std::clamp(endFade, 0.0f, 1.0f);
    return m.baseRadius + m.bellyBulge * bulge * endFade;
}

struct SkeletonRig {
    std::array<Vec3, J_COUNT> J;
    std::vector<Bone> bones;
    std::vector<Muscle> muscles;
    Vec3 headCentre;
    float headRadius;
    float scale;
    float muscleShellMin;
    float muscleShellMax;
    float skinShellMin;
};

inline SkeletonRig buildRig(float height, const Vec3& base) {
    SkeletonRig r;
    auto jn = normalisedJoints();
    for (int i = 0; i < J_COUNT; ++i) r.J[i] = base + jn[i] * height;
    r.bones   = boneList();
    r.muscles = muscleList();
    for (auto& b : r.bones)   b.boneRadius *= height;
    for (auto& m : r.muscles) {
        m.baseRadius *= height;
        m.bellyBulge *= height;
    }
    r.headCentre = r.J[J_HEAD];
    r.headRadius = 0.075f * height;
    r.scale = height;
    r.muscleShellMin = 0.010f * height;
    r.muscleShellMax = 0.075f * height;
    r.skinShellMin   = 0.008f * height;
    return r;
}

inline float skeletonSD(const SkeletonRig& r, const Vec3& p) {
    float d = 1e30f;
    for (const auto& b : r.bones)
        d = std::min(d, sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius));
    float skull = (p - r.headCentre).norm() - r.headRadius;
    d = std::min(d, skull);
    return d;
}

inline float muscleSD(const SkeletonRig& r, const Vec3& p) {
    float d = 1e30f;
    for (const auto& m : r.muscles) {
        float t;
        (void)distToSegment(p, r.J[m.a], r.J[m.b], t);
        float rad = muscleRadiusAt(m, t);
        float md = sdCapsule(p, r.J[m.a], r.J[m.b], rad);
        d = smin(d, md, 0.03f * r.scale);
    }
    for (const auto& b : r.bones) {
        float shell = std::clamp(b.boneRadius + r.muscleShellMin,
                                 b.boneRadius + r.muscleShellMin, r.muscleShellMax);
        float sd = sdCapsule(p, r.J[b.a], r.J[b.b], shell);
        d = smin(d, sd, 0.02f * r.scale);
    }
    float headShell = (p - r.headCentre).norm() - (r.headRadius + r.muscleShellMin);
    d = smin(d, headShell, 0.02f * r.scale);
    return d;
}

inline float fleshSD(const SkeletonRig& r, const Vec3& p) {
    const auto& J = r.J;
    const float s = r.scale;
    float k = 0.05f * s;
    float d = 1e30f;

    d = std::min(d, sdEllipsoid(p, r.headCentre + Vec3(0,0.0f,0.005f*s),
                                Vec3(0.083f, 0.090f, 0.100f) * s));
    d = smin(d, sdCapsule(p, J[J_NECK], J[J_HEAD], 0.040f * s), k);

    Vec3 chestC = (J[J_CHEST] + J[J_NECK]) * 0.5f;
    d = smin(d, sdEllipsoid(p, chestC, Vec3(0.150f, 0.105f, 0.150f) * s), k);
    Vec3 bellyC = (J[J_SPINE] + J[J_CHEST]) * 0.5f;
    d = smin(d, sdEllipsoid(p, bellyC, Vec3(0.120f, 0.095f, 0.120f) * s), k);
    Vec3 pelvisC = (J[J_PELVIS] + J[J_SPINE]) * 0.5f;
    d = smin(d, sdEllipsoid(p, pelvisC, Vec3(0.130f, 0.100f, 0.110f) * s), k);
    d = smin(d, sdEllipsoid(p, (J[J_HIP_L]+J[J_PELVIS])*0.5f, Vec3(0.075f,0.080f,0.075f)*s), k);
    d = smin(d, sdEllipsoid(p, (J[J_HIP_R]+J[J_PELVIS])*0.5f, Vec3(0.075f,0.080f,0.075f)*s), k);

    d = smin(d, sdEllipsoid(p, J[J_SHOULDER_L], Vec3(0.060f,0.060f,0.060f)*s), k);
    d = smin(d, sdEllipsoid(p, J[J_SHOULDER_R], Vec3(0.060f,0.060f,0.060f)*s), k);

    d = smin(d, sdCapsule(p, J[J_SHOULDER_L], J[J_ELBOW_L], 0.048f*s), k);
    d = smin(d, sdCapsule(p, J[J_ELBOW_L],    J[J_WRIST_L], 0.036f*s), k);
    d = smin(d, sdCapsule(p, J[J_SHOULDER_R], J[J_ELBOW_R], 0.048f*s), k);
    d = smin(d, sdCapsule(p, J[J_ELBOW_R],    J[J_WRIST_R], 0.036f*s), k);
    d = smin(d, sdEllipsoid(p, J[J_HAND_L], Vec3(0.030f,0.045f,0.020f)*s), 0.02f*s);
    d = smin(d, sdEllipsoid(p, J[J_HAND_R], Vec3(0.030f,0.045f,0.020f)*s), 0.02f*s);

    d = smin(d, sdCapsule(p, J[J_HIP_L], J[J_KNEE_L],  0.070f*s), k);
    d = smin(d, sdCapsule(p, J[J_KNEE_L], J[J_ANKLE_L],0.048f*s), k);
    d = smin(d, sdCapsule(p, J[J_HIP_R], J[J_KNEE_R],  0.070f*s), k);
    d = smin(d, sdCapsule(p, J[J_KNEE_R], J[J_ANKLE_R],0.048f*s), k);
    d = smin(d, sdCapsule(p, J[J_ANKLE_L], J[J_TOE_L], 0.028f*s), 0.02f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_R], J[J_TOE_R], 0.028f*s), 0.02f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_L], J[J_HEEL_L], 0.026f*s), 0.02f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_R], J[J_HEEL_R], 0.026f*s), 0.02f*s);

    float mSD = muscleSD(r, p);
    float minSkin = mSD - r.skinShellMin;
    d = std::min(d, minSkin);
    return d;
}

inline int classify(const SkeletonRig& r, const Vec3& p) {
    if (fleshSD(r, p) > 0.0f) return 0;
    if (skeletonSD(r, p) <= 0.0f) return 1;
    if (muscleSD(r, p)   <= 0.0f) return 2;
    return 3;
}

inline void rigBounds(const SkeletonRig& r, float step, Vec3& lo, Vec3& hi) {
    lo = Vec3(1e30f,1e30f,1e30f); hi = Vec3(-1e30f,-1e30f,-1e30f);
    float pad = 0.16f * r.scale + step;
    for (int i = 0; i < J_COUNT; ++i) {
        lo = lo.cwiseMin((r.J[i].array() - pad).matrix());
        hi = hi.cwiseMax((r.J[i].array() + pad).matrix());
    }
    lo.array() -= step;
    hi.array() += step;
}

template <typename T>
inline size_t emitLayer(Grid::Octree<T>& octree, const SkeletonRig& r,
                        float step, int wantLayer) {
    static std::mt19937 rng(20260729u);
    std::uniform_real_distribution<float> jitter(-0.0015f, 0.0015f);

    Vec3 lo, hi;
    rigBounds(r, step, lo, hi);
    size_t inserted = 0;

    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                int layer = classify(r, p);
                if (layer != wantLayer) continue;
                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                if (layer == 1) {
                    octree.insert(T{}, jp, true, Palette::BONE, step, true, OID_SKELETON,
                                0.0f, 0.55f, 0.0f, 0.0f, 1.55f, Vec3::Zero(),
                                BodyType::RIGID, 1.6f, 9000.0f, 220.0f, 0.5f);
                } else if (layer == 2) {
                    octree.insert(T{}, jp, true, Palette::MUSCLE, step, true, OID_MUSCLE,
                                0.0f, 0.7f, 0.0f, 0.0f, 1.4f, Vec3(0.05f, 0.02f, 0.02f),
                                BodyType::SOFT, 1.05f, 2600.0f, 45.0f, 0.6f);
                } else {
                    octree.insert(T{}, jp, true, Palette::FLESH, step, true, OID_FLESH,
                                0.0f, 0.85f, 0.0f, 0.0f, 1.4f, Vec3(0.02f, 0.03f, 0.04f),
                                BodyType::SOFT, 0.95f, 1200.0f, 22.0f, 0.75f);
                }
                ++inserted;
            }
    return inserted;
}

template <typename T>
inline size_t buildSkeleton(Grid::Octree<T>& octree, float height, float detail,
                            const Vec3& base = Vec3::Zero()) {
    return emitLayer(octree, buildRig(height, base), detail, 1);
}
template <typename T>
inline size_t buildMuscle(Grid::Octree<T>& octree, float height, float detail,
                          const Vec3& base = Vec3::Zero()) {
    return emitLayer(octree, buildRig(height, base), detail, 2);
}
template <typename T>
inline size_t buildFlesh(Grid::Octree<T>& octree, float height, float detail,
                         const Vec3& base = Vec3::Zero()) {
    return emitLayer(octree, buildRig(height, base), detail, 3);
}

template <typename T>
inline size_t buildCharacter(Grid::Octree<T>& octree, float height, float detail,
                             const Vec3& base = Vec3::Zero()) {
    SkeletonRig r = buildRig(height, base);
    const float step = detail;
    static std::mt19937 rng(20260729u);
    std::uniform_real_distribution<float> jitter(-0.0015f, 0.0015f);

    Vec3 lo, hi;
    rigBounds(r, step, lo, hi);
    size_t inserted = 0;

    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                int layer = classify(r, p);
                if (layer == 0) continue;
                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                if (layer == 1) {
                    octree.insert(T{}, jp, true, Palette::BONE, step, true, OID_SKELETON,
                                0.0f, 0.55f, 0.0f, 0.0f, 1.55f, Vec3::Zero(),
                                BodyType::RIGID, 1.6f, 9000.0f, 220.0f, 0.5f);
                } else if (layer == 2) {
                    octree.insert(T{}, jp, true, Palette::MUSCLE, step, true, OID_MUSCLE,
                                0.0f, 0.7f, 0.0f, 0.0f, 1.4f, Vec3(0.05f, 0.02f, 0.02f),
                                BodyType::SOFT, 1.05f, 2600.0f, 45.0f, 0.6f);
                } else {
                    octree.insert(T{}, jp, true, Palette::FLESH, step, true, OID_FLESH,
                                0.0f, 0.85f, 0.0f, 0.0f, 1.4f, Vec3(0.02f, 0.03f, 0.04f),
                                BodyType::SOFT, 0.95f, 1200.0f, 22.0f, 0.75f);
                }
                ++inserted;
            }
    return inserted;
}

}