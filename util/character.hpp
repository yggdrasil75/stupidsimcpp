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

enum Joint {
    J_PELVIS, J_SPINE, J_CHEST, J_NECK, J_HEAD_TOP,
    J_SHOULDER_L, J_ELBOW_L, J_WRIST_L,
    J_SHOULDER_R, J_ELBOW_R, J_WRIST_R,
    J_HIP_L, J_KNEE_L, J_ANKLE_L, J_FOOT_L,
    J_HIP_R, J_KNEE_R, J_ANKLE_R, J_FOOT_R,
    J_COUNT
};

inline std::array<Vec3, J_COUNT> normalisedJoints() {
    std::array<Vec3, J_COUNT> j{};
    j[J_PELVIS]     = Vec3( 0.00f, 0.0f, 0.50f);
    j[J_SPINE]      = Vec3( 0.00f, 0.0f, 0.60f);
    j[J_CHEST]      = Vec3( 0.00f, 0.0f, 0.72f);
    j[J_NECK]       = Vec3( 0.00f, 0.0f, 0.82f);
    j[J_HEAD_TOP]   = Vec3( 0.00f, 0.0f, 0.97f);

    j[J_SHOULDER_L] = Vec3( 0.10f, 0.0f, 0.80f);
    j[J_ELBOW_L]    = Vec3( 0.16f, 0.0f, 0.62f);
    j[J_WRIST_L]    = Vec3( 0.19f, 0.0f, 0.46f);

    j[J_SHOULDER_R] = Vec3(-0.10f, 0.0f, 0.80f);
    j[J_ELBOW_R]    = Vec3(-0.16f, 0.0f, 0.62f);
    j[J_WRIST_R]    = Vec3(-0.19f, 0.0f, 0.46f);

    j[J_HIP_L]      = Vec3( 0.06f, 0.0f, 0.48f);
    j[J_KNEE_L]     = Vec3( 0.07f, 0.0f, 0.26f);
    j[J_ANKLE_L]    = Vec3( 0.07f, 0.0f, 0.03f);
    j[J_FOOT_L]     = Vec3( 0.07f, 0.08f, 0.01f);

    j[J_HIP_R]      = Vec3(-0.06f, 0.0f, 0.48f);
    j[J_KNEE_R]     = Vec3(-0.07f, 0.0f, 0.26f);
    j[J_ANKLE_R]    = Vec3(-0.07f, 0.0f, 0.03f);
    j[J_FOOT_R]     = Vec3(-0.07f, 0.08f, 0.01f);
    return j;
}

struct Bone {
    Joint a, b;
    float boneRadius;
    float muscleRadius;
    float fleshRadius;
};

inline std::vector<Bone> boneList() {
    return {
        { J_PELVIS,     J_SPINE,     0.020f, 0.055f, 0.115f },
        { J_SPINE,      J_CHEST,     0.022f, 0.070f, 0.135f },
        { J_CHEST,      J_NECK,      0.016f, 0.045f, 0.075f },
        { J_NECK,       J_HEAD_TOP,  0.030f, 0.050f, 0.090f },
        { J_CHEST,      J_SHOULDER_L,0.014f, 0.045f, 0.075f },
        { J_SHOULDER_L, J_ELBOW_L,   0.012f, 0.038f, 0.058f },
        { J_ELBOW_L,    J_WRIST_L,   0.010f, 0.030f, 0.045f },
        { J_CHEST,      J_SHOULDER_R,0.014f, 0.045f, 0.075f },
        { J_SHOULDER_R, J_ELBOW_R,   0.012f, 0.038f, 0.058f },
        { J_ELBOW_R,    J_WRIST_R,   0.010f, 0.030f, 0.045f },
        { J_PELVIS,     J_HIP_L,     0.018f, 0.050f, 0.085f },
        { J_HIP_L,      J_KNEE_L,    0.016f, 0.050f, 0.075f },
        { J_KNEE_L,     J_ANKLE_L,   0.013f, 0.038f, 0.058f },
        { J_ANKLE_L,    J_FOOT_L,    0.012f, 0.025f, 0.045f },
        { J_PELVIS,     J_HIP_R,     0.018f, 0.050f, 0.085f },
        { J_HIP_R,      J_KNEE_R,    0.016f, 0.050f, 0.075f },
        { J_KNEE_R,     J_ANKLE_R,   0.013f, 0.038f, 0.058f },
        { J_ANKLE_R,    J_FOOT_R,    0.012f, 0.025f, 0.045f },
    };
}

template <typename T>
inline size_t buildCharacter(Grid::Octree<T>& octree, float height, float detail,
                             const Vec3& base = Vec3::Zero()) {
    const float step = detail;
    auto jn = normalisedJoints();

    std::array<Vec3, J_COUNT> J{};
    for (int i = 0; i < J_COUNT; ++i)
        J[i] = base + jn[i] * height;

    auto bones = boneList();
    for (auto& b : bones) {
        b.boneRadius   *= height;
        b.muscleRadius *= height;
        b.fleshRadius  *= height;
    }

    static std::mt19937 rng(20260729u);
    std::uniform_real_distribution<float> jitter(-0.0015f, 0.0015f);

    size_t inserted = 0;

    Vec3 lo( 1e30f,  1e30f,  1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (const auto& b : bones) {
        float pad = std::max(b.fleshRadius, b.muscleRadius) + step;
        for (Joint jj : {b.a, b.b}) {
            lo = lo.cwiseMin((J[jj].array() - pad).matrix());
            hi = hi.cwiseMax((J[jj].array() + pad).matrix());
        }
    }
    lo.array() -= step;
    hi.array() += step;

    auto evalFields = [&](const Vec3& p, float& bestBoneDist, float& boneRadHit,
                          float& bestMuscleGap, float& fleshField) {
        bestBoneDist = 1e30f;
        boneRadHit = 0.0f;
        bestMuscleGap = -1e30f;
        fleshField = 0.0f;
        for (const auto& b : bones) {
            float t;
            float d = distToSegment(p, J[b.a], J[b.b], t);

            if (d - b.boneRadius < bestBoneDist - boneRadHit) {
                bestBoneDist = d;
                boneRadHit = b.boneRadius;
            }

            float bulge = 1.0f - 4.0f * (t - 0.5f) * (t - 0.5f);
            float mRad = b.muscleRadius * (0.45f + 0.55f * std::max(0.0f, bulge));
            float gap = mRad - d;
            if (gap > bestMuscleGap) bestMuscleGap = gap;

            float R = b.fleshRadius;
            if (d < R) {
                float x = 1.0f - d / R;
                fleshField += x * x * (3.0f - 2.0f * x);
            }
        }
    };

    const float FLESH_ISO = 0.75f;

    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                float boneDist, boneRad, muscleGap, flesh;
                evalFields(p, boneDist, boneRad, muscleGap, flesh);

                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));

                if (boneDist <= boneRad) {
                    octree.insert(T{}, jp, true, Palette::BONE, step, true, OID_SKELETON,
                                0.0f, 0.55f, 0.0f, 0.0f, 1.55f, Vec3::Zero(),
                                BodyType::RIGID, 1.6f, 9000.0f, 220.0f, 0.5f);
                    ++inserted;
                } else if (muscleGap >= 0.0f) {
                    octree.insert(T{}, jp, true, Palette::MUSCLE, step, true, OID_MUSCLE,
                                0.0f, 0.7f, 0.0f, 0.0f, 1.4f, Vec3(0.05f, 0.02f, 0.02f),
                                BodyType::SOFT, 1.05f,
                                2600.0, 45.0f, 0.6f);
                    ++inserted;
                } else if (flesh >= FLESH_ISO) {
                    octree.insert(T{}, jp, true, Palette::FLESH, step, true, OID_FLESH,
                                0.0f, 0.85f, 0.0f, 0.0f, 1.4f, Vec3(0.02f, 0.03f, 0.04f),
                                BodyType::SOFT, 0.95f, 1200.0f, 22.0f, 0.75f);
                    ++inserted;
                }
            }

    return inserted;
}

}