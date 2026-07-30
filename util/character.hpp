#pragma once

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>

#include "../eigen/Eigen/Dense"
#include "grid/grid3eigen.hpp"

namespace Character {

using Vec3 = Eigen::Vector3f;
using Grid::BodyType;

static constexpr int OID_SKELETON_BASE = 700;
static constexpr int OID_MUSCLE        = 690;
static constexpr int OID_FLESH         = 691;
static constexpr int OID_SKELETON      = OID_SKELETON_BASE;

namespace Palette {
    static const Vec3 BONE  (0.92f, 0.90f, 0.82f);
    static const Vec3 MUSCLE(0.62f, 0.14f, 0.14f);
    static const Vec3 FLESH (0.85f, 0.66f, 0.55f);
}

inline float smin(float a, float b, float k) {
    if (k <= 0.0f) return std::min(a, b);
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

inline float sdTaperedCapsule(const Vec3& p, const Vec3& a, const Vec3& b,
                              float ra, float rb) {
    float t;
    float d = distToSegment(p, a, b, t);
    return d - (ra + (rb - ra) * t);
}

inline float sdEllipsoid(const Vec3& p, const Vec3& c, const Vec3& rad) {
    Vec3 q = (p - c).cwiseQuotient(rad);
    float k0 = q.norm();
    if (k0 < 1e-6f) return -rad.minCoeff();
    Vec3 q2 = q.cwiseQuotient(rad);
    float k1 = q2.norm();
    return k0 * (k0 - 1.0f) / std::max(k1, 1e-6f);
}

inline float sdMuscleBelly(const Vec3& p, const Vec3& a, const Vec3& b,
                           float rEnd, float rSide, float rFront,
                           float bellyAt) {
    Vec3 axis = b - a;
    float L = axis.norm();
    if (L < 1e-6f) return (p - a).norm() - rEnd;
    Vec3 u = axis / L;

    Vec3 rel = p - a;
    float t = std::clamp(rel.dot(u) / L, 0.0f, 1.0f);
    Vec3 onAxis = a + u * (t * L);

    Vec3 perp = p - onAxis;
    Vec3 ref = std::abs(u.z()) < 0.9f ? Vec3(0,0,1) : Vec3(1,0,0);
    Vec3 side  = u.cross(ref).normalized();
    Vec3 front = u.cross(side).normalized();
    float ps = perp.dot(side);
    float pf = perp.dot(front);

    float d = (t - bellyAt);
    float span = std::max(bellyAt, 1.0f - bellyAt);
    float prof = 1.0f - (d * d) / (span * span);
    prof = std::clamp(prof, 0.0f, 1.0f);
    prof = std::sqrt(prof);

    float rs = rEnd + (rSide  - rEnd) * prof;
    float rf = rEnd + (rFront - rEnd) * prof;
    rs = std::max(rs, 1e-4f);
    rf = std::max(rf, 1e-4f);

    float q = std::sqrt((ps * ps) / (rs * rs) + (pf * pf) / (rf * rf));
    float rmin = std::min(rs, rf);
    return (q - 1.0f) * rmin;
}

enum Joint {
    J_PELVIS, J_SPINE, J_CHEST, J_NECK, J_HEAD, J_HEAD_TOP,
    J_CLAV_L, J_SHOULDER_L, J_ELBOW_L, J_WRIST_L, J_HAND_L,
    J_CLAV_R, J_SHOULDER_R, J_ELBOW_R, J_WRIST_R, J_HAND_R,
    J_HIP_L, J_KNEE_L, J_ANKLE_L, J_HEEL_L, J_TOE_L,
    J_HIP_R, J_KNEE_R, J_ANKLE_R, J_HEEL_R, J_TOE_R,
    J_COUNT
};

inline const char* jointName(Joint j) {
    switch (j) {
        case J_PELVIS: return "pelvis";
        case J_SPINE: return "spine";
        case J_CHEST: return "chest";
        case J_NECK: return "neck";
        case J_HEAD: return "head";
        case J_HEAD_TOP: return "head_top";
        case J_CLAV_L: return "clav.L";
        case J_SHOULDER_L: return "shoulder.L";
        case J_ELBOW_L: return "elbow.L";
        case J_WRIST_L: return "wrist.L";
        case J_HAND_L: return "hand.L";
        case J_CLAV_R: return "clav.R";
        case J_SHOULDER_R: return "shoulder.R";
        case J_ELBOW_R: return "elbow.R";
        case J_WRIST_R: return "wrist.R";
        case J_HAND_R: return "hand.R";
        case J_HIP_L: return "hip.L";
        case J_KNEE_L: return "knee.L";
        case J_ANKLE_L: return "ankle.L";
        case J_HEEL_L: return "heel.L";
        case J_TOE_L: return "toe.L";
        case J_HIP_R: return "hip.R";
        case J_KNEE_R: return "knee.R";
        case J_ANKLE_R: return "ankle.R";
        case J_HEEL_R: return "heel.R";
        case J_TOE_R: return "toe.R";
        default: return "?";
    }
}

inline std::array<Vec3, J_COUNT> normalisedJoints() {
    std::array<Vec3, J_COUNT> j{};
    j[J_PELVIS]   = Vec3( 0.00f, 0.00f, 0.500f);
    j[J_SPINE]    = Vec3( 0.00f,-0.02f, 0.605f);
    j[J_CHEST]    = Vec3( 0.00f,-0.015f,0.720f);
    j[J_NECK]     = Vec3( 0.00f, 0.00f, 0.820f);
    j[J_HEAD]     = Vec3( 0.00f, 0.015f,0.895f);
    j[J_HEAD_TOP] = Vec3( 0.00f, 0.015f,0.985f);

    j[J_CLAV_L]     = Vec3( 0.040f, 0.00f, 0.800f);
    j[J_SHOULDER_L] = Vec3( 0.110f, 0.00f, 0.795f);
    j[J_ELBOW_L]    = Vec3( 0.135f,-0.015f,0.630f);
    j[J_WRIST_L]    = Vec3( 0.150f, 0.00f, 0.470f);
    j[J_HAND_L]     = Vec3( 0.155f, 0.015f,0.410f);

    j[J_CLAV_R]     = Vec3(-0.040f, 0.00f, 0.800f);
    j[J_SHOULDER_R] = Vec3(-0.110f, 0.00f, 0.795f);
    j[J_ELBOW_R]    = Vec3(-0.135f,-0.015f,0.630f);
    j[J_WRIST_R]    = Vec3(-0.150f, 0.00f, 0.470f);
    j[J_HAND_R]     = Vec3(-0.155f, 0.015f,0.410f);

    j[J_HIP_L]   = Vec3( 0.048f, 0.00f, 0.480f);
    j[J_KNEE_L]  = Vec3( 0.050f, 0.015f,0.255f);
    j[J_ANKLE_L] = Vec3( 0.050f, 0.00f, 0.035f);
    j[J_HEEL_L]  = Vec3( 0.050f,-0.035f,0.010f);
    j[J_TOE_L]   = Vec3( 0.050f, 0.085f,0.008f);

    j[J_HIP_R]   = Vec3(-0.048f, 0.00f, 0.480f);
    j[J_KNEE_R]  = Vec3(-0.050f, 0.015f,0.255f);
    j[J_ANKLE_R] = Vec3(-0.050f, 0.00f, 0.035f);
    j[J_HEEL_R]  = Vec3(-0.050f,-0.035f,0.010f);
    j[J_TOE_R]   = Vec3(-0.050f, 0.085f,0.008f);
    return j;
}

struct Bone {
    Joint a, b;
    float boneRadius;
    const char* name;
};

inline std::vector<Bone> boneList() {
    return {
        { J_PELVIS, J_SPINE,     0.024f, "lumbar" },
        { J_SPINE,  J_CHEST,     0.026f, "thoracic" },
        { J_CHEST,  J_NECK,      0.016f, "cervical" },
        { J_NECK,   J_HEAD,      0.015f, "skull" },
        { J_CHEST,  J_CLAV_L,    0.011f, "clavicle.L" },
        { J_CLAV_L, J_SHOULDER_L,0.011f, "scapula.L" },
        { J_CHEST,  J_CLAV_R,    0.011f, "clavicle.R" },
        { J_CLAV_R, J_SHOULDER_R,0.011f, "scapula.R" },
        { J_SHOULDER_L, J_ELBOW_L, 0.013f, "humerus.L" },
        { J_ELBOW_L,    J_WRIST_L, 0.011f, "forearm.L" },
        { J_WRIST_L,    J_HAND_L,  0.009f, "hand.L" },
        { J_SHOULDER_R, J_ELBOW_R, 0.013f, "humerus.R" },
        { J_ELBOW_R,    J_WRIST_R, 0.011f, "forearm.R" },
        { J_WRIST_R,    J_HAND_R,  0.009f, "hand.R" },
        { J_PELVIS, J_HIP_L,     0.020f, "pelvis.L" },
        { J_PELVIS, J_HIP_R,     0.020f, "pelvis.R" },
        { J_HIP_L,   J_KNEE_L,   0.019f, "femur.L" },
        { J_KNEE_L,  J_ANKLE_L,  0.015f, "tibia.L" },
        { J_ANKLE_L, J_HEEL_L,   0.011f, "heel.L" },
        { J_ANKLE_L, J_TOE_L,    0.010f, "foot.L" },
        { J_HIP_R,   J_KNEE_R,   0.019f, "femur.R" },
        { J_KNEE_R,  J_ANKLE_R,  0.015f, "tibia.R" },
        { J_ANKLE_R, J_HEEL_R,   0.011f, "heel.R" },
        { J_ANKLE_R, J_TOE_R,    0.010f, "foot.R" },
    };
}

struct Muscle {
    Joint a, b;
    float rEnd;
    float rSide;
    float rFront;
    float bellyAt;
    Joint actuates;
    const char* name;
};

inline std::vector<Muscle> muscleList() {
    return {
        { J_PELVIS, J_CHEST,  0.055f, 0.115f, 0.090f, 0.55f, J_SPINE,  "abdomen" },
        { J_SPINE,  J_CHEST,  0.045f, 0.135f, 0.115f, 0.70f, J_CHEST,  "pectoral+lat" },
        { J_CHEST,  J_NECK,   0.030f, 0.050f, 0.045f, 0.45f, J_NECK,   "trapezius" },
        { J_NECK,   J_HEAD,   0.026f, 0.036f, 0.036f, 0.50f, J_HEAD,   "sternocleido" },
        { J_CLAV_L, J_SHOULDER_L, 0.030f, 0.052f, 0.050f, 0.85f, J_SHOULDER_L, "deltoid.L" },
        { J_CLAV_R, J_SHOULDER_R, 0.030f, 0.052f, 0.050f, 0.85f, J_SHOULDER_R, "deltoid.R" },
        { J_SHOULDER_L, J_ELBOW_L, 0.026f, 0.040f, 0.044f, 0.45f, J_ELBOW_L, "biceps.L" },
        { J_SHOULDER_R, J_ELBOW_R, 0.026f, 0.040f, 0.044f, 0.45f, J_ELBOW_R, "biceps.R" },
        { J_ELBOW_L, J_WRIST_L, 0.018f, 0.034f, 0.032f, 0.28f, J_WRIST_L, "forearm.L" },
        { J_ELBOW_R, J_WRIST_R, 0.018f, 0.034f, 0.032f, 0.28f, J_WRIST_R, "forearm.R" },
        { J_WRIST_L, J_HAND_L, 0.016f, 0.022f, 0.014f, 0.50f, J_HAND_L, "hand.L" },
        { J_WRIST_R, J_HAND_R, 0.016f, 0.022f, 0.014f, 0.50f, J_HAND_R, "hand.R" },
        { J_PELVIS, J_HIP_L, 0.040f, 0.070f, 0.075f, 0.60f, J_HIP_L, "glute.L" },
        { J_PELVIS, J_HIP_R, 0.040f, 0.070f, 0.075f, 0.60f, J_HIP_R, "glute.R" },
        { J_HIP_L, J_KNEE_L, 0.035f, 0.062f, 0.066f, 0.45f, J_KNEE_L, "quad.L" },
        { J_HIP_R, J_KNEE_R, 0.035f, 0.062f, 0.066f, 0.45f, J_KNEE_R, "quad.R" },
        { J_KNEE_L, J_ANKLE_L, 0.022f, 0.046f, 0.048f, 0.30f, J_ANKLE_L, "calf.L" },
        { J_KNEE_R, J_ANKLE_R, 0.022f, 0.046f, 0.048f, 0.30f, J_ANKLE_R, "calf.R" },
        { J_ANKLE_L, J_TOE_L, 0.018f, 0.026f, 0.020f, 0.45f, J_TOE_L, "foot.L" },
        { J_ANKLE_R, J_TOE_R, 0.018f, 0.026f, 0.020f, 0.45f, J_TOE_R, "foot.R" },
    };
}

struct SkeletonRig {
    std::array<Vec3, J_COUNT> J;
    std::vector<Bone> bones;
    std::vector<Muscle> muscles;
    Vec3 headCentre;
    float headRadius;
    float scale;
    float skinThickness;
};

inline SkeletonRig buildRig(float height, const Vec3& base) {
    SkeletonRig r;
    auto jn = normalisedJoints();
    for (int i = 0; i < J_COUNT; ++i) r.J[i] = base + jn[i] * height;
    r.bones   = boneList();
    r.muscles = muscleList();
    for (auto& b : r.bones) b.boneRadius *= height;
    for (auto& m : r.muscles) {
        m.rEnd   *= height;
        m.rSide  *= height;
        m.rFront *= height;
    }
    r.headCentre = r.J[J_HEAD];
    r.headRadius = 0.072f * height;
    r.scale = height;
    r.skinThickness = 0.012f * height;
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

inline float boneSD(const SkeletonRig& r, const Bone& b, const Vec3& p) {
    float d = sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius);
    if (b.b == J_HEAD)
        d = std::min(d, (p - r.headCentre).norm() - r.headRadius);
    return d;
}

inline int nearestBone(const SkeletonRig& r, const Vec3& p) {
    int best = -1;
    float bd = 1e30f;
    for (int i = 0; i < (int)r.bones.size(); ++i) {
        float d = boneSD(r, r.bones[i], p);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

inline float muscleSD(const SkeletonRig& r, const Vec3& p) {
    float d = 1e30f;
    float k = 0.02f * r.scale;
    for (const auto& m : r.muscles) {
        float md = sdMuscleBelly(p, r.J[m.a], r.J[m.b],
                                 m.rEnd, m.rSide, m.rFront, m.bellyAt);
        d = smin(d, md, k);
    }
    
    for (const auto& b : r.bones) {
        float sd = sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius + 0.006f * r.scale);
        d = smin(d, sd, k);
    }
    float headShell = (p - r.headCentre).norm() - (r.headRadius + 0.006f * r.scale);
    d = smin(d, headShell, k);
    return d;
}

inline float bodyShapeSD(const SkeletonRig& r, const Vec3& p) {
    const auto& J = r.J;
    const float s = r.scale;
    float k = 0.03f * s;
    float d = 1e30f;

    d = std::min(d, sdEllipsoid(p, r.headCentre + Vec3(0,0.004f*s,0.004f*s),
                                Vec3(0.078f, 0.085f, 0.095f) * s));
    Vec3 chestC = J[J_CHEST] + Vec3(0,0.005f*s,-0.01f*s);
    d = smin(d, sdEllipsoid(p, chestC, Vec3(0.115f, 0.085f, 0.100f) * s), k);
    Vec3 pelvisC = (J[J_PELVIS] + J[J_SPINE]) * 0.5f;
    d = smin(d, sdEllipsoid(p, pelvisC, Vec3(0.095f, 0.080f, 0.075f) * s), k);
    // Hands & feet as small pads.
    d = smin(d, sdEllipsoid(p, J[J_HAND_L], Vec3(0.026f,0.042f,0.016f)*s), 0.015f*s);
    d = smin(d, sdEllipsoid(p, J[J_HAND_R], Vec3(0.026f,0.042f,0.016f)*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_L], J[J_TOE_L], 0.026f*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_R], J[J_TOE_R], 0.026f*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_L], J[J_HEEL_L], 0.024f*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_R], J[J_HEEL_R], 0.024f*s), 0.015f*s);
    return d;
}

inline float fleshSD(const SkeletonRig& r, const Vec3& p) {
    float core = smin(muscleSD(r, p), bodyShapeSD(r, p), 0.03f * r.scale);
    return core - r.skinThickness;
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

}
#include "character_rig.hpp"