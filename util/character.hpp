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

static constexpr int OID_MUSCLE        = 690;
static constexpr int OID_FLESH         = 691;
static constexpr int OID_SKELETON_BASE = 1000;
static constexpr int OID_SKELETON      = OID_SKELETON_BASE;

namespace Palette {
    static const Vec3 BONE(0.92f, 0.90f, 0.82f);
    static const Vec3 MUSCLE(0.62f, 0.14f, 0.14f);
    static const Vec3 FLESH(0.85f, 0.66f, 0.55f);
}

struct BuildConfig {
    float skeletonDetail = 0.001f;
    float muscleDetail = 0.001f;
    float skinDetail = 0.0001f;

    float skinThickness = 0.010f;
    float muscleInflate = 0.006f;
    float muscleBlend = 0.020f;

    bool  buildRibcage = true;
    bool  buildHands = true;
    bool  buildFeet = true;
    bool  buildSkullJaw = true;

    float boneRadiusScale = 1.0f;

    bool  useStrandMuscles = true;
    float strandDetail     = 0.004f;
    int   fascicleScale    = 1;
    float wrapOffsetScale  = 1.0f;
    float strandBlend      = 0.012f;

    static BuildConfig fromLegacyDetail(float detail) {
        BuildConfig c;
        c.skeletonDetail = detail * 0.1f;
        c.muscleDetail = detail * 0.1f;
        c.skinDetail = detail * 0.01f;
        return c;
    }
};

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

inline float sdPolyline(const Vec3& p, const std::vector<Vec3>& pts, float r) {
    float d = 1e30f;
    for (size_t i = 1; i < pts.size(); ++i)
        d = std::min(d, sdCapsule(p, pts[i - 1], pts[i], r));
    return d;
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

#define FINGER_JOINTS(SIDE) \
    J_THUMB_MCP_##SIDE, J_THUMB_IP_##SIDE, J_THUMB_TIP_##SIDE, \
    J_INDEX_MCP_##SIDE, J_INDEX_PIP_##SIDE, J_INDEX_DIP_##SIDE, J_INDEX_TIP_##SIDE, \
    J_MIDDLE_MCP_##SIDE, J_MIDDLE_PIP_##SIDE, J_MIDDLE_DIP_##SIDE, J_MIDDLE_TIP_##SIDE, \
    J_RING_MCP_##SIDE, J_RING_PIP_##SIDE, J_RING_DIP_##SIDE, J_RING_TIP_##SIDE, \
    J_PINKY_MCP_##SIDE, J_PINKY_PIP_##SIDE, J_PINKY_DIP_##SIDE, J_PINKY_TIP_##SIDE

#define TOE_JOINTS(SIDE) \
    J_BIGTOE_##SIDE, J_TOE2_##SIDE, J_TOE3_##SIDE, J_TOE4_##SIDE, J_TOE5_##SIDE

enum Joint {
    J_PELVIS, J_SPINE1, J_SPINE2, J_SPINE3, J_CHEST, J_NECK, J_HEAD, J_HEAD_TOP,
    J_JAW,

    J_CLAV_L, J_SHOULDER_L, J_ELBOW_L, J_WRIST_L,
    J_CLAV_R, J_SHOULDER_R, J_ELBOW_R, J_WRIST_R,

    FINGER_JOINTS(L),
    FINGER_JOINTS(R),

    J_HIP_L, J_KNEE_L, J_ANKLE_L, J_HEEL_L, J_BALL_L,
    J_HIP_R, J_KNEE_R, J_ANKLE_R, J_HEEL_R, J_BALL_R,

    TOE_JOINTS(L),
    TOE_JOINTS(R),

    J_COUNT
};

inline const char* jointName(Joint j) {
    switch (j) {
        case J_PELVIS: return "pelvis";
        case J_SPINE1: return "spine1";
        case J_SPINE2: return "spine2";
        case J_SPINE3: return "spine3";
        case J_CHEST: return "chest";
        case J_NECK: return "neck";
        case J_HEAD: return "head";
        case J_HEAD_TOP: return "head_top";
        case J_JAW: return "jaw";

        case J_CLAV_L: return "clav.L";
        case J_SHOULDER_L: return "shoulder.L";
        case J_ELBOW_L: return "elbow.L";
        case J_WRIST_L: return "wrist.L";
        case J_CLAV_R: return "clav.R";
        case J_SHOULDER_R: return "shoulder.R";
        case J_ELBOW_R: return "elbow.R";
        case J_WRIST_R: return "wrist.R";

        case J_THUMB_MCP_L: return "thumb.mcp.L";
        case J_THUMB_IP_L:  return "thumb.ip.L";
        case J_THUMB_TIP_L: return "thumb.tip.L";
        case J_INDEX_MCP_L: return "index.mcp.L";
        case J_INDEX_PIP_L: return "index.pip.L";
        case J_INDEX_DIP_L: return "index.dip.L";
        case J_INDEX_TIP_L: return "index.tip.L";
        case J_MIDDLE_MCP_L: return "middle.mcp.L";
        case J_MIDDLE_PIP_L: return "middle.pip.L";
        case J_MIDDLE_DIP_L: return "middle.dip.L";
        case J_MIDDLE_TIP_L: return "middle.tip.L";
        case J_RING_MCP_L: return "ring.mcp.L";
        case J_RING_PIP_L: return "ring.pip.L";
        case J_RING_DIP_L: return "ring.dip.L";
        case J_RING_TIP_L: return "ring.tip.L";
        case J_PINKY_MCP_L: return "pinky.mcp.L";
        case J_PINKY_PIP_L: return "pinky.pip.L";
        case J_PINKY_DIP_L: return "pinky.dip.L";
        case J_PINKY_TIP_L: return "pinky.tip.L";

        case J_THUMB_MCP_R: return "thumb.mcp.R";
        case J_THUMB_IP_R:  return "thumb.ip.R";
        case J_THUMB_TIP_R: return "thumb.tip.R";
        case J_INDEX_MCP_R: return "index.mcp.R";
        case J_INDEX_PIP_R: return "index.pip.R";
        case J_INDEX_DIP_R: return "index.dip.R";
        case J_INDEX_TIP_R: return "index.tip.R";
        case J_MIDDLE_MCP_R: return "middle.mcp.R";
        case J_MIDDLE_PIP_R: return "middle.pip.R";
        case J_MIDDLE_DIP_R: return "middle.dip.R";
        case J_MIDDLE_TIP_R: return "middle.tip.R";
        case J_RING_MCP_R: return "ring.mcp.R";
        case J_RING_PIP_R: return "ring.pip.R";
        case J_RING_DIP_R: return "ring.dip.R";
        case J_RING_TIP_R: return "ring.tip.R";
        case J_PINKY_MCP_R: return "pinky.mcp.R";
        case J_PINKY_PIP_R: return "pinky.pip.R";
        case J_PINKY_DIP_R: return "pinky.dip.R";
        case J_PINKY_TIP_R: return "pinky.tip.R";

        case J_HIP_L: return "hip.L";
        case J_KNEE_L: return "knee.L";
        case J_ANKLE_L: return "ankle.L";
        case J_HEEL_L: return "heel.L";
        case J_BALL_L: return "ball.L";
        case J_HIP_R: return "hip.R";
        case J_KNEE_R: return "knee.R";
        case J_ANKLE_R: return "ankle.R";
        case J_HEEL_R: return "heel.R";
        case J_BALL_R: return "ball.R";

        case J_BIGTOE_L: return "bigtoe.L";
        case J_TOE2_L: return "toe2.L";
        case J_TOE3_L: return "toe3.L";
        case J_TOE4_L: return "toe4.L";
        case J_TOE5_L: return "toe5.L";
        case J_BIGTOE_R: return "bigtoe.R";
        case J_TOE2_R: return "toe2.R";
        case J_TOE3_R: return "toe3.R";
        case J_TOE4_R: return "toe4.R";
        case J_TOE5_R: return "toe5.R";
        default: return "?";
    }
}

inline Joint mirrorJoint(Joint j) {
    std::string n = jointName(j);
    if (n.size() < 2) return j;
    std::string swapped = n;
    if (n.compare(n.size() - 2, 2, ".L") == 0) swapped.replace(n.size() - 2, 2, ".R");
    else if (n.compare(n.size() - 2, 2, ".R") == 0) swapped.replace(n.size() - 2, 2, ".L");
    else return j;
    for (int i = 0; i < J_COUNT; ++i)
        if (swapped == jointName((Joint)i)) return (Joint)i;
    return j;
}

inline std::array<Vec3, J_COUNT> normalisedJoints() {
    std::array<Vec3, J_COUNT> j{};

    j[J_PELVIS]   = Vec3( 0.00f, 0.000f, 0.500f);
    j[J_SPINE1]   = Vec3( 0.00f,-0.015f, 0.560f);
    j[J_SPINE2]   = Vec3( 0.00f,-0.022f, 0.625f);
    j[J_SPINE3]   = Vec3( 0.00f,-0.018f, 0.685f);
    j[J_CHEST]    = Vec3( 0.00f,-0.010f, 0.735f);
    j[J_NECK]     = Vec3( 0.00f, 0.000f, 0.820f);
    j[J_HEAD]     = Vec3( 0.00f, 0.015f, 0.895f);
    j[J_HEAD_TOP] = Vec3( 0.00f, 0.015f, 0.985f);
    j[J_JAW]      = Vec3( 0.00f, 0.055f, 0.858f);

    j[J_CLAV_L]     = Vec3( 0.040f, 0.010f, 0.800f);
    j[J_SHOULDER_L] = Vec3( 0.110f, 0.000f, 0.795f);
    j[J_ELBOW_L]    = Vec3( 0.135f,-0.015f, 0.630f);
    j[J_WRIST_L]    = Vec3( 0.150f, 0.000f, 0.470f);

    j[J_CLAV_R]     = Vec3(-0.040f, 0.010f, 0.800f);
    j[J_SHOULDER_R] = Vec3(-0.110f, 0.000f, 0.795f);
    j[J_ELBOW_R]    = Vec3(-0.135f,-0.015f, 0.630f);
    j[J_WRIST_R]    = Vec3(-0.150f, 0.000f, 0.470f);

    auto layHand = [&](int sign,
                       Joint tMcp, Joint tIp, Joint tTip,
                       Joint iMcp, Joint iPip, Joint iDip, Joint iTip,
                       Joint mMcp, Joint mPip, Joint mDip, Joint mTip,
                       Joint rMcp, Joint rPip, Joint rDip, Joint rTip,
                       Joint pMcp, Joint pPip, Joint pDip, Joint pTip,
                       const Vec3& wrist) {
        float sx = (float)sign;
        auto finger = [&](Joint a, Joint b, Joint c, Joint d,
                          float xoff, float mcpZ,
                          float lp1, float lp2, float lp3) {
            Vec3 mcp = wrist + Vec3(sx * xoff, 0.030f, mcpZ);
            j[a] = mcp;
            j[b] = mcp + Vec3(0, lp1, 0);
            j[c] = mcp + Vec3(0, lp1 + lp2, 0);
            j[d] = mcp + Vec3(0, lp1 + lp2 + lp3, 0);
        };
        finger(iMcp, iPip, iDip, iTip,  0.028f, -0.008f, 0.024f, 0.016f, 0.012f);
        finger(mMcp, mPip, mDip, mTip,  0.009f, -0.006f, 0.026f, 0.017f, 0.013f);
        finger(rMcp, rPip, rDip, rTip, -0.010f, -0.007f, 0.024f, 0.016f, 0.012f);
        finger(pMcp, pPip, pDip, pTip, -0.028f, -0.010f, 0.019f, 0.013f, 0.010f);
        Vec3 tmcp = wrist + Vec3(sx * 0.036f, 0.006f, 0.006f);
        j[tMcp] = tmcp;
        j[tIp]  = tmcp + Vec3(sx * 0.020f, 0.020f, -0.004f);
        j[tTip] = tmcp + Vec3(sx * 0.030f, 0.038f, -0.008f);
    };

    layHand(+1, J_THUMB_MCP_L, J_THUMB_IP_L, J_THUMB_TIP_L,
                J_INDEX_MCP_L, J_INDEX_PIP_L, J_INDEX_DIP_L, J_INDEX_TIP_L,
                J_MIDDLE_MCP_L, J_MIDDLE_PIP_L, J_MIDDLE_DIP_L, J_MIDDLE_TIP_L,
                J_RING_MCP_L, J_RING_PIP_L, J_RING_DIP_L, J_RING_TIP_L,
                J_PINKY_MCP_L, J_PINKY_PIP_L, J_PINKY_DIP_L, J_PINKY_TIP_L,
                j[J_WRIST_L]);
    layHand(-1, J_THUMB_MCP_R, J_THUMB_IP_R, J_THUMB_TIP_R,
                J_INDEX_MCP_R, J_INDEX_PIP_R, J_INDEX_DIP_R, J_INDEX_TIP_R,
                J_MIDDLE_MCP_R, J_MIDDLE_PIP_R, J_MIDDLE_DIP_R, J_MIDDLE_TIP_R,
                J_RING_MCP_R, J_RING_PIP_R, J_RING_DIP_R, J_RING_TIP_R,
                J_PINKY_MCP_R, J_PINKY_PIP_R, J_PINKY_DIP_R, J_PINKY_TIP_R,
                j[J_WRIST_R]);

    j[J_HIP_L]   = Vec3( 0.048f, 0.000f, 0.480f);
    j[J_KNEE_L]  = Vec3( 0.050f, 0.015f, 0.255f);
    j[J_ANKLE_L] = Vec3( 0.050f, 0.000f, 0.040f);
    j[J_HEEL_L]  = Vec3( 0.050f,-0.035f, 0.012f);
    j[J_BALL_L]  = Vec3( 0.050f, 0.070f, 0.012f);

    j[J_HIP_R]   = Vec3(-0.048f, 0.000f, 0.480f);
    j[J_KNEE_R]  = Vec3(-0.050f, 0.015f, 0.255f);
    j[J_ANKLE_R] = Vec3(-0.050f, 0.000f, 0.040f);
    j[J_HEEL_R]  = Vec3(-0.050f,-0.035f, 0.012f);
    j[J_BALL_R]  = Vec3(-0.050f, 0.070f, 0.012f);

    auto layToes = [&](int sign, Joint b, Joint t2, Joint t3, Joint t4, Joint t5,
                       const Vec3& ball) {
        float sx = (float)sign;
        j[b]  = ball + Vec3(sx * 0.014f, 0.028f, -0.002f);
        j[t2] = ball + Vec3(sx * 0.006f, 0.026f, -0.002f);
        j[t3] = ball + Vec3(sx *-0.002f, 0.024f, -0.002f);
        j[t4] = ball + Vec3(sx *-0.010f, 0.021f, -0.002f);
        j[t5] = ball + Vec3(sx *-0.018f, 0.017f, -0.002f);
    };
    layToes(+1, J_BIGTOE_L, J_TOE2_L, J_TOE3_L, J_TOE4_L, J_TOE5_L, j[J_BALL_L]);
    layToes(-1, J_BIGTOE_R, J_TOE2_R, J_TOE3_R, J_TOE4_R, J_TOE5_R, j[J_BALL_R]);

    return j;
}

struct Bone {
    Joint a, b;
    float boneRadius;
    std::string name;
};

struct RibArc {
    std::vector<Vec3> pts;
    float radius;
    std::string name;
};

inline std::vector<RibArc> ribArcsNormalised() {
    std::vector<RibArc> ribs;
    ribs.push_back({ {
        Vec3(0.0f, 0.075f, 0.760f),
        Vec3(0.0f, 0.078f, 0.720f),
        Vec3(0.0f, 0.075f, 0.680f),
        Vec3(0.0f, 0.068f, 0.645f),
    }, 0.010f, "sternum" });

    struct RibDef {
        float z;
        float halfW;
        float depth;
        float frontZ;
    };
    const RibDef defs[] = {
        { 0.760f, 0.088f, 0.085f, 0.758f },
        { 0.735f, 0.100f, 0.092f, 0.730f },
        { 0.710f, 0.105f, 0.095f, 0.702f },
        { 0.685f, 0.104f, 0.093f, 0.676f },
        { 0.660f, 0.098f, 0.088f, 0.650f },
        { 0.635f, 0.088f, 0.080f, 0.628f },
        { 0.610f, 0.074f, 0.068f, 0.612f },
    };
    int pair = 1;
    for (const auto& d : defs) {
        for (int side = 0; side < 2; ++side) {
            float sx = side == 0 ? 1.0f : -1.0f;
            std::vector<Vec3> pts = {
                Vec3(0.0f, -0.030f, d.z),
                Vec3(sx * d.halfW*0.55f, -0.010f, d.z + 0.004f),
                Vec3(sx * d.halfW, 0.020f, d.z),
                Vec3(sx * d.halfW*0.75f, d.depth*0.7f, d.frontZ),
                Vec3(sx * 0.018f, d.depth, d.frontZ),
            };
            std::string nm = std::string("rib") + std::to_string(pair) +
                             (side == 0 ? ".L" : ".R");
            ribs.push_back({ pts, 0.006f, nm });
        }
        ++pair;
    }
    return ribs;
}

inline std::vector<Bone> boneList(const BuildConfig& cfg) {
    std::vector<Bone> b;

    b.push_back({ J_CHEST,  J_CLAV_L,     0.010f, "clavicle.L" });
    b.push_back({ J_CLAV_L, J_SHOULDER_L, 0.011f, "scapula.L" });
    b.push_back({ J_CHEST,  J_CLAV_R,     0.010f, "clavicle.R" });
    b.push_back({ J_CLAV_R, J_SHOULDER_R, 0.011f, "scapula.R" });

    b.push_back({ J_SHOULDER_L, J_ELBOW_L, 0.013f, "humerus.L" });
    b.push_back({ J_ELBOW_L,    J_WRIST_L, 0.011f, "forearm.L" });
    b.push_back({ J_SHOULDER_R, J_ELBOW_R, 0.013f, "humerus.R" });
    b.push_back({ J_ELBOW_R,    J_WRIST_R, 0.011f, "forearm.R" });

    b.push_back({ J_PELVIS, J_HIP_L, 0.020f, "pelvis.L" });
    b.push_back({ J_PELVIS, J_HIP_R, 0.020f, "pelvis.R" });
    b.push_back({ J_HIP_L,  J_KNEE_L,  0.019f, "femur.L" });
    b.push_back({ J_KNEE_L, J_ANKLE_L, 0.015f, "tibia.L" });
    b.push_back({ J_HIP_R,  J_KNEE_R,  0.019f, "femur.R" });
    b.push_back({ J_KNEE_R, J_ANKLE_R, 0.015f, "tibia.R" });

    if (cfg.buildFeet) {
        b.push_back({ J_ANKLE_L, J_HEEL_L, 0.012f, "calcaneus.L" });
        b.push_back({ J_ANKLE_L, J_BALL_L, 0.011f, "midfoot.L" });
        b.push_back({ J_ANKLE_R, J_HEEL_R, 0.012f, "calcaneus.R" });
        b.push_back({ J_ANKLE_R, J_BALL_R, 0.011f, "midfoot.R" });
        b.push_back({ J_BALL_L, J_BIGTOE_L, 0.006f, "toe1.L" });
        b.push_back({ J_BALL_L, J_TOE2_L,   0.006f, "toe2.L" });
        b.push_back({ J_BALL_L, J_TOE3_L,   0.006f, "toe3.L" });
        b.push_back({ J_BALL_L, J_TOE4_L,   0.006f, "toe4.L" });
        b.push_back({ J_BALL_L, J_TOE5_L,   0.006f, "toe5.L" });
        b.push_back({ J_BALL_R, J_BIGTOE_R, 0.006f, "toe1.R" });
        b.push_back({ J_BALL_R, J_TOE2_R,   0.006f, "toe2.R" });
        b.push_back({ J_BALL_R, J_TOE3_R,   0.006f, "toe3.R" });
        b.push_back({ J_BALL_R, J_TOE4_R,   0.006f, "toe4.R" });
        b.push_back({ J_BALL_R, J_TOE5_R,   0.006f, "toe5.R" });
    } else {
        b.push_back({ J_ANKLE_L, J_HEEL_L, 0.012f, "heel.L" });
        b.push_back({ J_ANKLE_L, J_BALL_L, 0.011f, "foot.L" });
        b.push_back({ J_ANKLE_R, J_HEEL_R, 0.012f, "heel.R" });
        b.push_back({ J_ANKLE_R, J_BALL_R, 0.011f, "foot.R" });
    }

    if (cfg.buildHands) {
        b.push_back({ J_WRIST_L, J_INDEX_MCP_L,  0.006f, "mc.index.L" });
        b.push_back({ J_WRIST_L, J_MIDDLE_MCP_L, 0.006f, "mc.middle.L" });
        b.push_back({ J_WRIST_L, J_RING_MCP_L,   0.006f, "mc.ring.L" });
        b.push_back({ J_WRIST_L, J_PINKY_MCP_L,  0.006f, "mc.pinky.L" });
        b.push_back({ J_WRIST_L, J_THUMB_MCP_L,  0.007f, "mc.thumb.L" });
        b.push_back({ J_WRIST_R, J_INDEX_MCP_R,  0.006f, "mc.index.R" });
        b.push_back({ J_WRIST_R, J_MIDDLE_MCP_R, 0.006f, "mc.middle.R" });
        b.push_back({ J_WRIST_R, J_RING_MCP_R,   0.006f, "mc.ring.R" });
        b.push_back({ J_WRIST_R, J_PINKY_MCP_R,  0.006f, "mc.pinky.R" });
        b.push_back({ J_WRIST_R, J_THUMB_MCP_R,  0.007f, "mc.thumb.R" });

        auto finger = [&](Joint mcp, Joint pip, Joint dip, Joint tip,
                          const std::string& base) {
            b.push_back({ mcp, pip, 0.0055f, base + ".prox" });
            b.push_back({ pip, dip, 0.0048f, base + ".mid" });
            b.push_back({ dip, tip, 0.0040f, base + ".dist" });
        };
        auto thumb = [&](Joint mcp, Joint ip, Joint tip, const std::string& base) {
            b.push_back({ mcp, ip,  0.0065f, base + ".prox" });
            b.push_back({ ip,  tip, 0.0050f, base + ".dist" });
        };
        thumb(J_THUMB_MCP_L, J_THUMB_IP_L, J_THUMB_TIP_L, "thumb.L");
        finger(J_INDEX_MCP_L, J_INDEX_PIP_L, J_INDEX_DIP_L, J_INDEX_TIP_L, "index.L");
        finger(J_MIDDLE_MCP_L, J_MIDDLE_PIP_L, J_MIDDLE_DIP_L, J_MIDDLE_TIP_L, "middle.L");
        finger(J_RING_MCP_L, J_RING_PIP_L, J_RING_DIP_L, J_RING_TIP_L, "ring.L");
        finger(J_PINKY_MCP_L, J_PINKY_PIP_L, J_PINKY_DIP_L, J_PINKY_TIP_L, "pinky.L");
        thumb(J_THUMB_MCP_R, J_THUMB_IP_R, J_THUMB_TIP_R, "thumb.R");
        finger(J_INDEX_MCP_R, J_INDEX_PIP_R, J_INDEX_DIP_R, J_INDEX_TIP_R, "index.R");
        finger(J_MIDDLE_MCP_R, J_MIDDLE_PIP_R, J_MIDDLE_DIP_R, J_MIDDLE_TIP_R, "middle.R");
        finger(J_RING_MCP_R, J_RING_PIP_R, J_RING_DIP_R, J_RING_TIP_R, "ring.R");
        finger(J_PINKY_MCP_R, J_PINKY_PIP_R, J_PINKY_DIP_R, J_PINKY_TIP_R, "pinky.R");
    } else {
        b.push_back({ J_WRIST_L, J_MIDDLE_MCP_L, 0.009f, "hand.L" });
        b.push_back({ J_WRIST_R, J_MIDDLE_MCP_R, 0.009f, "hand.R" });
    }

    if (cfg.boneRadiusScale != 1.0f)
        for (auto& bb : b) bb.boneRadius *= cfg.boneRadiusScale;

    return b;
}

struct Muscle {
    Joint a, b;
    float rEnd;
    float rSide;
    float rFront;
    float bellyAt;
    Joint actuates;
    std::string name;
};

inline std::vector<Muscle> muscleList() {
    return {
        { J_PELVIS, J_CHEST,  0.055f, 0.115f, 0.090f, 0.55f, J_SPINE2, "abdomen" },
        { J_PELVIS, J_SPINE2, 0.030f, 0.055f, 0.045f, 0.55f, J_SPINE1, "erector_lower" },
        { J_SPINE2, J_CHEST,  0.030f, 0.058f, 0.048f, 0.55f, J_SPINE3, "erector_upper" },
        { J_SPINE1, J_CHEST,  0.045f, 0.135f, 0.115f, 0.70f, J_CHEST,  "pectoral+lat" },
        { J_CHEST,  J_NECK,   0.030f, 0.050f, 0.045f, 0.45f, J_NECK,   "trapezius" },
        { J_NECK,   J_HEAD,   0.026f, 0.036f, 0.036f, 0.50f, J_HEAD,   "sternocleido" },
        { J_HEAD,   J_JAW,    0.016f, 0.024f, 0.022f, 0.50f, J_JAW,    "masseter" },
        { J_CLAV_L, J_SHOULDER_L, 0.030f, 0.052f, 0.050f, 0.85f, J_SHOULDER_L, "deltoid.L" },
        { J_CLAV_R, J_SHOULDER_R, 0.030f, 0.052f, 0.050f, 0.85f, J_SHOULDER_R, "deltoid.R" },
        { J_SHOULDER_L, J_ELBOW_L, 0.026f, 0.040f, 0.044f, 0.45f, J_ELBOW_L, "biceps.L" },
        { J_SHOULDER_R, J_ELBOW_R, 0.026f, 0.040f, 0.044f, 0.45f, J_ELBOW_R, "biceps.R" },
        { J_ELBOW_L, J_WRIST_L, 0.018f, 0.034f, 0.032f, 0.28f, J_WRIST_L, "forearm.L" },
        { J_ELBOW_R, J_WRIST_R, 0.018f, 0.034f, 0.032f, 0.28f, J_WRIST_R, "forearm.R" },
        { J_WRIST_L, J_MIDDLE_MCP_L, 0.014f, 0.020f, 0.012f, 0.50f, J_WRIST_L, "thenar.L" },
        { J_WRIST_R, J_MIDDLE_MCP_R, 0.014f, 0.020f, 0.012f, 0.50f, J_WRIST_R, "thenar.R" },
        { J_PELVIS, J_HIP_L, 0.040f, 0.070f, 0.075f, 0.60f, J_HIP_L, "glute.L" },
        { J_PELVIS, J_HIP_R, 0.040f, 0.070f, 0.075f, 0.60f, J_HIP_R, "glute.R" },
        { J_HIP_L, J_KNEE_L, 0.035f, 0.062f, 0.066f, 0.45f, J_KNEE_L, "quad.L" },
        { J_HIP_R, J_KNEE_R, 0.035f, 0.062f, 0.066f, 0.45f, J_KNEE_R, "quad.R" },
        { J_KNEE_L, J_ANKLE_L, 0.022f, 0.046f, 0.048f, 0.30f, J_ANKLE_L, "calf.L" },
        { J_KNEE_R, J_ANKLE_R, 0.022f, 0.046f, 0.048f, 0.30f, J_ANKLE_R, "calf.R" },
        { J_ANKLE_L, J_BALL_L, 0.018f, 0.026f, 0.020f, 0.45f, J_BALL_L, "foot.L" },
        { J_ANKLE_R, J_BALL_R, 0.018f, 0.026f, 0.020f, 0.45f, J_BALL_R, "foot.R" },
    };
}

enum MuscleRole {
    ROLE_FLEXOR,
    ROLE_EXTENSOR,
    ROLE_ABDUCTOR,
    ROLE_ADDUCTOR,
    ROLE_ROTATOR,
    ROLE_AXIAL,
};

inline const char* roleName(MuscleRole r) {
    switch (r) {
        case ROLE_FLEXOR:   return "flexor";
        case ROLE_EXTENSOR: return "extensor";
        case ROLE_ABDUCTOR: return "abductor";
        case ROLE_ADDUCTOR: return "adductor";
        case ROLE_ROTATOR:  return "rotator";
        case ROLE_AXIAL:    return "axial";
        default:            return "?";
    }
}

struct StrandMuscle {
    Joint     origin;
    Joint     insertion;
    Joint     crosses;
    MuscleRole role;
    Vec3      bowHint;
    int       fascicles;
    float     spread;
    float     rBelly;
    float     rTendon;
    float     bellyAt;
    float     bowAmount;
    std::string name;
};

inline std::vector<StrandMuscle> strandMuscleListHalf() {
    std::vector<StrandMuscle> m;
    auto add = [&](Joint o, Joint i, Joint c, MuscleRole role, Vec3 bow,
                   int f, float spread, float rB, float rT, float belly,
                   float bowAmt, const char* nm) {
        m.push_back({ o, i, c, role, bow, f, spread, rB, rT, belly, bowAmt, nm });
    };

    add(J_PELVIS, J_CHEST, J_SPINE2, ROLE_AXIAL, Vec3(0, 1, 0),
        4, 0.045f, 0.016f, 0.006f, 0.55f, 0.055f, "rectus_abdominis.L");
    add(J_PELVIS, J_CHEST, J_SPINE2, ROLE_AXIAL, Vec3(0, -1, 0),
        3, 0.018f, 0.014f, 0.006f, 0.50f, 0.030f, "erector_spinae.L");
    add(J_HIP_L, J_CHEST, J_SPINE2, ROLE_ROTATOR, Vec3(1, 0.4f, 0),
        3, 0.030f, 0.012f, 0.006f, 0.55f, 0.040f, "oblique.L");
    add(J_PELVIS, J_SHOULDER_L, J_SHOULDER_L, ROLE_ADDUCTOR, Vec3(0, -0.6f, 0),
        4, 0.045f, 0.014f, 0.006f, 0.35f, 0.055f, "latissimus.L");
    add(J_CHEST, J_SHOULDER_L, J_SHOULDER_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        4, 0.038f, 0.014f, 0.006f, 0.45f, 0.050f, "pectoralis.L");
    add(J_HEAD, J_SHOULDER_L, J_CLAV_L, ROLE_EXTENSOR, Vec3(0, -1, 0),
        3, 0.035f, 0.012f, 0.005f, 0.50f, 0.030f, "trapezius.L");

    add(J_CHEST, J_HEAD, J_NECK, ROLE_FLEXOR, Vec3(0, 1, 0),
        2, 0.016f, 0.010f, 0.005f, 0.50f, 0.022f, "sternocleido.L");
    add(J_CHEST, J_HEAD, J_NECK, ROLE_EXTENSOR, Vec3(0, -1, 0),
        2, 0.016f, 0.010f, 0.005f, 0.50f, 0.022f, "splenius.L");
    add(J_HEAD, J_JAW, J_JAW, ROLE_FLEXOR, Vec3(0, 1, 0),
        2, 0.010f, 0.008f, 0.004f, 0.50f, 0.010f, "masseter.L");

    add(J_CLAV_L, J_ELBOW_L, J_SHOULDER_L, ROLE_ABDUCTOR, Vec3(1, 0, 0.2f),
        4, 0.026f, 0.014f, 0.006f, 0.40f, 0.045f, "deltoid.L");
    add(J_SHOULDER_L, J_ELBOW_L, J_SHOULDER_L, ROLE_ROTATOR, Vec3(-0.4f, -0.6f, 0),
        2, 0.014f, 0.009f, 0.005f, 0.25f, 0.020f, "rotatorcuff.L");

    add(J_SHOULDER_L, J_WRIST_L, J_ELBOW_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        3, 0.018f, 0.013f, 0.005f, 0.45f, 0.028f, "biceps.L");
    add(J_SHOULDER_L, J_WRIST_L, J_ELBOW_L, ROLE_EXTENSOR, Vec3(0, -1, 0),
        3, 0.018f, 0.013f, 0.005f, 0.45f, 0.028f, "triceps.L");
    add(J_SHOULDER_L, J_WRIST_L, J_ELBOW_L, ROLE_FLEXOR, Vec3(0, 0.8f, 0),
        2, 0.012f, 0.010f, 0.005f, 0.55f, 0.020f, "brachialis.L");

    add(J_ELBOW_L, J_MIDDLE_MCP_L, J_WRIST_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        3, 0.016f, 0.011f, 0.004f, 0.30f, 0.022f, "wristflexors.L");
    add(J_ELBOW_L, J_MIDDLE_MCP_L, J_WRIST_L, ROLE_EXTENSOR, Vec3(0, -1, 0),
        3, 0.016f, 0.011f, 0.004f, 0.30f, 0.022f, "wristextensors.L");

    add(J_PELVIS, J_KNEE_L, J_HIP_L, ROLE_EXTENSOR, Vec3(0, -1, 0),
        4, 0.034f, 0.018f, 0.007f, 0.30f, 0.055f, "glute_max.L");
    add(J_PELVIS, J_KNEE_L, J_HIP_L, ROLE_ABDUCTOR, Vec3(1, 0, 0),
        3, 0.024f, 0.013f, 0.006f, 0.25f, 0.035f, "glute_med.L");
    add(J_SPINE1, J_KNEE_L, J_HIP_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        3, 0.020f, 0.012f, 0.006f, 0.55f, 0.040f, "iliopsoas.L");
    add(J_PELVIS, J_KNEE_L, J_HIP_L, ROLE_ADDUCTOR, Vec3(-1, 0.2f, 0),
        3, 0.024f, 0.014f, 0.006f, 0.45f, 0.035f, "adductors.L");

    add(J_HIP_L, J_ANKLE_L, J_KNEE_L, ROLE_EXTENSOR, Vec3(0, 1, 0),
        4, 0.030f, 0.020f, 0.006f, 0.45f, 0.050f, "quadriceps.L");
    add(J_PELVIS, J_ANKLE_L, J_KNEE_L, ROLE_FLEXOR, Vec3(0, -1, 0),
        4, 0.028f, 0.017f, 0.006f, 0.45f, 0.050f, "hamstrings.L");

    add(J_KNEE_L, J_HEEL_L, J_ANKLE_L, ROLE_EXTENSOR, Vec3(0, -1, 0),
        4, 0.024f, 0.016f, 0.005f, 0.30f, 0.040f, "calf.L");
    add(J_KNEE_L, J_BALL_L, J_ANKLE_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        3, 0.016f, 0.011f, 0.005f, 0.35f, 0.028f, "tibialis_ant.L");

    add(J_WRIST_L, J_MIDDLE_MCP_L, J_WRIST_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        2, 0.010f, 0.008f, 0.004f, 0.50f, 0.008f, "palm.L");
    add(J_ANKLE_L, J_BALL_L, J_BALL_L, ROLE_FLEXOR, Vec3(0, 1, 0),
        2, 0.012f, 0.009f, 0.004f, 0.50f, 0.010f, "sole.L");

    return m;
}

inline Joint mirrorJoint(Joint j);

inline std::vector<StrandMuscle> strandMuscleList() {
    auto half = strandMuscleListHalf();
    std::vector<StrandMuscle> full;
    full.reserve(half.size() * 2);
    for (const auto& sm : half) {
        full.push_back(sm);
        StrandMuscle r = sm;
        r.origin    = mirrorJoint(sm.origin);
        r.insertion = mirrorJoint(sm.insertion);
        r.crosses   = mirrorJoint(sm.crosses);
        r.bowHint.x() = -sm.bowHint.x();
        std::string n = sm.name;
        auto pos = n.rfind(".L");
        if (pos != std::string::npos) n.replace(pos, 2, ".R");
        else n += ".R";
        r.name = n;
        full.push_back(r);
    }
    return full;
}

struct RoutedMuscle {
    int id;
    std::string name;
    MuscleRole role;
    Joint origin, insertion, crosses;
    Vec3 originPos, insertPos;
    std::vector<std::vector<Vec3>> fascicles;
    float rBelly, rTendon, bellyAt;
};

struct SkeletonRig {
    std::array<Vec3, J_COUNT> J;
    std::vector<Bone> bones;
    std::vector<RibArc> ribs;
    std::vector<Muscle> muscles;
    std::vector<RoutedMuscle> routed;
    Vec3 headCentre;
    Vec3 headRadii;
    Vec3 jawCentre;
    Vec3 jawRadii;
    float scale;
    BuildConfig cfg;
    float skinThickness;
};

inline Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
    float n = v.norm();
    return n > 1e-6f ? (v / n) : fallback;
}

inline Vec3 closestOnSegmentPoint(const Vec3& p, const Vec3& a, const Vec3& b) {
    float t; return closestOnSegment(p, a, b, t);
}

inline Vec3 bowDirection(const std::array<Vec3, J_COUNT>& J, const StrandMuscle& sm) {
    Vec3 o = J[sm.origin];
    Vec3 i = J[sm.insertion];
    Vec3 axis = safeNormalize(i - o, Vec3(0, 0, 1));
    Vec3 hint = sm.bowHint;

    switch (sm.role) {
        case ROLE_FLEXOR:
            return safeNormalize(hint - axis * hint.dot(axis), Vec3(0, 1, 0));
        case ROLE_EXTENSOR:
            return safeNormalize(hint - axis * hint.dot(axis), Vec3(0, -1, 0));
        case ROLE_ABDUCTOR:
            return safeNormalize(hint - axis * hint.dot(axis),
                                 Vec3(J[sm.crosses].x() >= 0 ? 1.f : -1.f, 0, 0));
        case ROLE_ADDUCTOR:
            return safeNormalize(hint - axis * hint.dot(axis),
                                 Vec3(J[sm.crosses].x() >= 0 ? -1.f : 1.f, 0, 0));
        case ROLE_ROTATOR:
            return safeNormalize(axis.cross(safeNormalize(hint, Vec3(1, 0, 0))),
                                 Vec3(1, 0, 0));
        case ROLE_AXIAL:
        default:
            return safeNormalize(hint - axis * hint.dot(axis), Vec3(0, 1, 0));
    }
}

inline RoutedMuscle routeStrandMuscle(const std::array<Vec3, J_COUNT>& J,
                                      const StrandMuscle& sm, int id,
                                      float height, const BuildConfig& cfg) {
    RoutedMuscle rm;
    rm.id = id;
    rm.name = sm.name;
    rm.role = sm.role;
    rm.origin = sm.origin;
    rm.insertion = sm.insertion;
    rm.crosses = sm.crosses;
    rm.originPos = J[sm.origin];
    rm.insertPos = J[sm.insertion];
    rm.rBelly = sm.rBelly * height;
    rm.rTendon = sm.rTendon * height;
    rm.bellyAt = sm.bellyAt;

    Vec3 o = rm.originPos, i = rm.insertPos;
    Vec3 axis = safeNormalize(i - o, Vec3(0, 0, 1));
    Vec3 bow = bowDirection(J, sm);
    Vec3 fan = safeNormalize(axis.cross(bow), Vec3(1, 0, 0));

    Vec3 jointP = J[sm.crosses];
    float bowLen = sm.bowAmount * height * cfg.wrapOffsetScale;
    int nF = std::max(1, sm.fascicles * std::max(1, cfg.fascicleScale));
    float spread = sm.spread * height;

    for (int f = 0; f < nF; ++f) {
        float u = nF > 1 ? (float)f / (float)(nF - 1) - 0.5f : 0.0f;
        Vec3 lateral = fan * (u * 2.0f * spread);
        Vec3 oF = o + lateral;
        Vec3 iF = i + lateral;
        float bowScale = 1.0f - 0.35f * std::abs(u) * 2.0f;
        Vec3 mid = closestOnSegmentPoint(jointP, oF, iF);
        mid += bow * (bowLen * bowScale);
        rm.fascicles.push_back({ oF, mid, iF });
    }
    return rm;
}

inline void routeAllMuscles(SkeletonRig& r, float height, const BuildConfig& cfg) {
    r.routed.clear();
    if (!cfg.useStrandMuscles) return;
    auto list = strandMuscleList();
    r.routed.reserve(list.size());
    int id = 0;
    for (const auto& sm : list)
        r.routed.push_back(routeStrandMuscle(r.J, sm, id++, height, cfg));
}

inline SkeletonRig buildRig(float height, const Vec3& base, const BuildConfig& cfg) {
    SkeletonRig r;
    r.cfg = cfg;
    auto jn = normalisedJoints();
    for (int i = 0; i < J_COUNT; ++i) r.J[i] = base + jn[i] * height;
    r.bones = boneList(cfg);
    r.muscles = muscleList();
    if (cfg.buildRibcage) {
        r.ribs = ribArcsNormalised();
        for (auto& rib : r.ribs) {
            for (auto& p : rib.pts) p = base + p * height;
            rib.radius *= height * cfg.boneRadiusScale;
        }
    }
    for (auto& b : r.bones) b.boneRadius *= height;
    for (auto& m : r.muscles) {
        m.rEnd   *= height;
        m.rSide  *= height;
        m.rFront *= height;
    }
    r.headCentre = r.J[J_HEAD];
    r.headRadii  = Vec3(0.072f, 0.078f, 0.088f) * height;
    r.jawCentre  = r.J[J_JAW];
    r.jawRadii   = Vec3(0.055f, 0.045f, 0.030f) * height;
    r.scale = height;
    r.skinThickness = cfg.skinThickness * height;
    routeAllMuscles(r, height, cfg);
    return r;
}

inline SkeletonRig buildRig(float height, const Vec3& base) {
    return buildRig(height, base, BuildConfig{});
}

inline float ribcageSD(const SkeletonRig& r, const Vec3& p) {
    if (r.ribs.empty()) return 1e30f;
    float d = 1e30f;
    for (const auto& rib : r.ribs)
        d = std::min(d, sdPolyline(p, rib.pts, rib.radius));
    return d;
}

inline float fascicleTaperedSD(const Vec3& p, const std::vector<Vec3>& poly,
                               float rBelly, float rTendon, float bellyAt) {
    float bestD = 1e30f;
    float bestParam = 0.0f;
    int nseg = (int)poly.size() - 1;
    for (int s = 0; s < nseg; ++s) {
        float t;
        float d = distToSegment(p, poly[s], poly[s + 1], t);
        if (d < bestD) {
            bestD = d;
            bestParam = ((float)s + t) / (float)nseg;
        }
    }
    float d0 = std::abs(bestParam - bellyAt);
    float span = std::max(bellyAt, 1.0f - bellyAt);
    float prof = std::clamp(1.0f - (d0 * d0) / (span * span), 0.0f, 1.0f);
    prof = std::sqrt(prof);
    float rad = rTendon + (rBelly - rTendon) * prof;
    return bestD - std::max(rad, 1e-4f);
}

inline float strandMuscleSD(const SkeletonRig& r, const Vec3& p) {
    if (r.routed.empty()) return 1e30f;
    float k = r.cfg.strandBlend * r.scale;
    float d = 1e30f;
    for (const auto& rm : r.routed)
        for (const auto& fas : rm.fascicles)
            d = smin(d, fascicleTaperedSD(p, fas, rm.rBelly, rm.rTendon, rm.bellyAt), k);
    return d;
}

inline bool strandFiberDir(const SkeletonRig& r, const Vec3& p,
                           Vec3& dirOut, int& muscleIdOut) {
    if (r.routed.empty()) return false;
    float bestD = 1e30f;
    Vec3 bestDir(0, 0, 1);
    int bestId = -1;
    for (const auto& rm : r.routed) {
        for (const auto& fas : rm.fascicles) {
            for (size_t s = 1; s < fas.size(); ++s) {
                float t;
                float d = distToSegment(p, fas[s - 1], fas[s], t);
                if (d < bestD) {
                    bestD = d;
                    bestDir = safeNormalize(fas[s] - fas[s - 1], Vec3(0, 0, 1));
                    bestId = rm.id;
                }
            }
        }
    }
    if (bestId < 0) return false;
    dirOut = bestDir;
    muscleIdOut = bestId;
    return true;
}

inline int nearestStrandMuscle(const SkeletonRig& r, const Vec3& p) {
    int best = -1; float bd = 1e30f;
    for (const auto& rm : r.routed) {
        for (const auto& fas : rm.fascicles) {
            for (size_t s = 1; s < fas.size(); ++s) {
                float t;
                float d = distToSegment(p, fas[s - 1], fas[s], t);
                if (d < bd) { bd = d; best = rm.id; }
            }
        }
    }
    return best;
}

inline float skeletonSD(const SkeletonRig& r, const Vec3& p) {
    float d = 1e30f;
    for (const auto& b : r.bones)
        d = std::min(d, sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius));
    d = std::min(d, ribcageSD(r, p));
    if (r.cfg.buildSkullJaw) {
        d = std::min(d, sdEllipsoid(p, r.headCentre, r.headRadii));
        d = std::min(d, sdEllipsoid(p, r.jawCentre, r.jawRadii));
    }
    return d;
}

inline float boneSD(const SkeletonRig& r, const Bone& b, const Vec3& p) {
    return sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius);
}

inline int nearestBone(const SkeletonRig& r, const Vec3& p) {
    int best = -1;
    float bd = 1e30f;
    for (int i = 0; i < (int)r.bones.size(); ++i) {
        float d = boneSD(r, r.bones[i], p);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

inline float muscleSD(const SkeletonRig& r, const Vec3& p) {
    float d = 1e30f;
    float k = r.cfg.muscleBlend * r.scale;
    float inflate = r.cfg.muscleInflate * r.scale;
    if (r.cfg.useStrandMuscles && !r.routed.empty()) {
        d = smin(d, strandMuscleSD(r, p), k);
    } else {
        for (const auto& m : r.muscles) {
            float md = sdMuscleBelly(p, r.J[m.a], r.J[m.b],
                                     m.rEnd, m.rSide, m.rFront, m.bellyAt);
            d = smin(d, md, k);
        }
    }
    for (const auto& b : r.bones) {
        float sd = sdCapsule(p, r.J[b.a], r.J[b.b], b.boneRadius + inflate);
        d = smin(d, sd, k);
    }
    float rib = ribcageSD(r, p) - inflate;
    d = smin(d, rib, k);
    if (r.cfg.buildSkullJaw) {
        Vec3 hr = r.headRadii + Vec3::Constant(inflate);
        d = smin(d, sdEllipsoid(p, r.headCentre, hr), k);
    }
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
    Vec3 pelvisC = (J[J_PELVIS] + J[J_SPINE1]) * 0.5f;
    d = smin(d, sdEllipsoid(p, pelvisC, Vec3(0.095f, 0.080f, 0.075f) * s), k);
    d = smin(d, sdEllipsoid(p, J[J_WRIST_L], Vec3(0.026f,0.042f,0.016f)*s), 0.015f*s);
    d = smin(d, sdEllipsoid(p, J[J_WRIST_R], Vec3(0.026f,0.042f,0.016f)*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_L], J[J_BALL_L], 0.026f*s), 0.015f*s);
    d = smin(d, sdCapsule(p, J[J_ANKLE_R], J[J_BALL_R], 0.026f*s), 0.015f*s);
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
    if (muscleSD(r, p) <= 0.0f) return 2;
    return 3;
}

inline void rigBounds(const SkeletonRig& r, float step, Vec3& lo, Vec3& hi) {
    lo = Vec3(1e30f,1e30f,1e30f);
    hi = Vec3(-1e30f,-1e30f,-1e30f);
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