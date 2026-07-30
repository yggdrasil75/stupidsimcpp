#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace Character {

struct JointLimits {
    float swingMin = -3.14159f, swingMax = 3.14159f;
    float twistMin = -1.57080f, twistMax = 1.57080f;
};

struct RigJoint {
    Joint joint;
    int parentBone;
    int childBone;
    int parentObjId;
    int childObjId;
    Vec3 position;
    JointLimits limits;
    std::string name;
};

struct RigBone {
    int index;
    int objectId;
    Joint head, tail;
    Vec3 headPos, tailPos;
    float radius;
    std::string name;
};

struct RigMuscle {
    int index;
    Joint actuates;
    int originObjId;
    int insertObjId;
    Vec3 originPos, insertPos;
    std::string name;
    std::vector<Vec3> voxels;
};

 struct CharacterRig {
    SkeletonRig geom;
    std::vector<RigBone> bones;
    std::vector<RigJoint> joints;
    std::vector<RigMuscle> muscles;
    int fleshObjId = OID_FLESH;
    int muscleObjId = OID_MUSCLE;

    int boneObjectId(int boneIndex) const {
        return OID_SKELETON_BASE + boneIndex;
    }
    const RigBone* findBone(const std::string& n) const {
        for (auto& b : bones) if (b.name == n) return &b;
        return nullptr;
    }
    const RigMuscle* findMuscle(const std::string& n) const {
        for (auto& m : muscles) if (m.name == n) return &m;
        return nullptr;
    }
};

struct RigCallbacks {
    std::function<void(const RigBone&)> onBone;
    std::function<void(const RigJoint&)> onJoint;
    std::function<void(const RigMuscle&)> onMuscle;
    std::function<void(int, int, const Vec3&)> onVoxel;
};

inline void classifyJointBones(const CharacterRig& rig, Joint j,
                               int& parentBone, int& childBone) {
    parentBone = -1;
    childBone = -1;
    for (int i = 0; i < (int)rig.geom.bones.size(); ++i) {
        const Bone& b = rig.geom.bones[i];
        if (b.b == j && parentBone < 0) parentBone = i;
        if (b.a == j && childBone  < 0) childBone  = i;
    }
    if (childBone < 0 && parentBone >= 0) {
        childBone = parentBone;
        parentBone = -1;
    }

    if (parentBone < 0 && childBone >= 0) {
        Joint headJoint = rig.geom.bones[childBone].a;
        for (int i = 0; i < (int)rig.geom.bones.size(); ++i) {
            if (i == childBone) continue;
            const Bone& b = rig.geom.bones[i];
            if (b.b == headJoint) {
                parentBone = i;
                break;
            }
        }
        if (parentBone < 0) {
            for (int i = 0; i < (int)rig.geom.bones.size(); ++i) {
                if (i == childBone) continue;
                const Bone& b = rig.geom.bones[i];
                if (b.a == headJoint) {
                    parentBone = i;
                    break;
                }
            }
        }
    }
}

inline CharacterRig assembleRig(float height, const Vec3& base) {
    CharacterRig rig;
    rig.geom = buildRig(height, base);

    for (int i = 0; i < (int)rig.geom.bones.size(); ++i) {
        const Bone& b = rig.geom.bones[i];
        RigBone rb;
        rb.index = i;
        rb.objectId = rig.boneObjectId(i);
        rb.head = b.a;
        rb.tail = b.b;
        rb.headPos = rig.geom.J[b.a];
        rb.tailPos = rig.geom.J[b.b];
        rb.radius = b.boneRadius;
        rb.name = b.name;
        rig.bones.push_back(rb);
    }

    std::unordered_map<int, bool> seen;
    for (int i = 0; i < (int)rig.geom.bones.size(); ++i) {
        Joint jt = rig.geom.bones[i].b;
        if (seen[(int)jt]) continue;
        seen[(int)jt] = true;

        int pB, cB;
        classifyJointBones(rig, jt, pB, cB);
        if (cB < 0) continue;

        RigJoint rj;
        rj.joint       = jt;
        rj.parentBone  = pB;
        rj.childBone   = cB;
        rj.parentObjId = pB >= 0 ? rig.boneObjectId(pB) : -1;
        rj.childObjId  = rig.boneObjectId(cB);
        rj.position    = rig.geom.J[jt];
        rj.name        = jointName(jt);
        switch (jt) {
            case J_ELBOW_L:
            case J_ELBOW_R:
            case J_KNEE_L:
            case J_KNEE_R:
                rj.limits = { 0.0f, 2.4f, -0.1f, 0.1f };
                break;
            case J_NECK:
            case J_HEAD:
                rj.limits = { -0.9f, 0.9f, -1.2f, 1.2f };
                break;
            default:
                rj.limits = {};
                break;
        }
        rig.joints.push_back(rj);
    }

    for (int i = 0; i < (int)rig.geom.muscles.size(); ++i) {
        const Muscle& m = rig.geom.muscles[i];
        RigMuscle rm;
        rm.index     = i;
        rm.actuates  = m.actuates;
        rm.originPos = rig.geom.J[m.a];
        rm.insertPos = rig.geom.J[m.b];
        rm.name      = m.name;
        rm.originObjId = -1;
        rm.insertObjId = -1;
        auto boneEndingAt = [&](Joint j, bool preferTail) -> int {
            int match = -1;
            for (int k = 0; k < (int)rig.geom.bones.size(); ++k) {
                const Bone& bk = rig.geom.bones[k];
                if (preferTail && bk.b == j) return k;
                if (!preferTail && bk.a == j) return k;
                if (bk.a == j || bk.b == j) match = k;
            }
            return match;
        };
        int bo = boneEndingAt(m.a, true);
        int bi = boneEndingAt(m.b, false);
        if (bo < 0) bo = nearestBone(rig.geom, rm.originPos);
        if (bi < 0) bi = nearestBone(rig.geom, rm.insertPos);
        if (bo >= 0) rm.originObjId = rig.boneObjectId(bo);
        if (bi >= 0) rm.insertObjId = rig.boneObjectId(bi);
        rig.muscles.push_back(rm);
    }

    return rig;
}

namespace detail {

template <typename T>
inline void insertBone(Grid::Octree<T>& octree, int objId, const Vec3& p, float step) {
    octree.insert(T{}, p, true, Palette::BONE, step, true, objId,
                  0.0f, 0.55f, 0.0f, 0.0f, 1.55f, Vec3::Zero(),
                  BodyType::RIGID, 1.6f, 9000.0f, 260.0f, 0.5f);
}
template <typename T>
inline void insertMuscle(Grid::Octree<T>& octree, int objId, const Vec3& p, float step) {
    octree.insert(T{}, p, true, Palette::MUSCLE, step, true, objId,
                  0.0f, 0.7f, 0.0f, 0.0f, 1.4f, Vec3(0.05f, 0.02f, 0.02f),
                  BodyType::SOFT, 1.05f, 2600.0f, 45.0f, 0.6f);
}
template <typename T>
inline void insertFlesh(Grid::Octree<T>& octree, int objId, const Vec3& p, float step) {
    octree.insert(T{}, p, true, Palette::FLESH, step, true, objId,
                  0.0f, 0.85f, 0.0f, 0.0f, 1.4f, Vec3(0.02f, 0.03f, 0.04f),
                  BodyType::SOFT, 0.95f, 1200.0f, 22.0f, 0.75f);
}

}

template <typename T>
inline CharacterRig buildRiggedCharacter(Grid::Octree<T>& octree, float height, float detail,
                                         const Vec3& base = Vec3::Zero(), const RigCallbacks& cb = {}) {
    CharacterRig rig = assembleRig(height, base);
    const SkeletonRig& r = rig.geom;
    const float step = detail;

    static std::mt19937 rng(20260729u);
    std::uniform_real_distribution<float> jitter(-0.0015f, 0.0015f);

    if (cb.onBone) for (auto& b : rig.bones) cb.onBone(b);
    if (cb.onJoint) for (auto& j : rig.joints) cb.onJoint(j);

    Vec3 lo, hi;
    rigBounds(r, step, lo, hi);

    std::vector<std::vector<Vec3>> muscleVoxels(rig.muscles.size());

    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                int layer = classify(r, p);
                if (layer == 0) continue;
                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));

                if (layer == 1) {
                    int bi = nearestBone(r, p);
                    if (bi < 0) continue;
                    int objId = rig.boneObjectId(bi);
                    detail::insertBone(octree, objId, jp, step);
                    if (cb.onVoxel) cb.onVoxel(objId, 1, jp);
                } else if (layer == 2) {
                    detail::insertMuscle(octree, OID_MUSCLE, jp, step);
                    if (cb.onVoxel) cb.onVoxel(OID_MUSCLE, 2, jp);
                    int best = -1;
                    float bd = 1e30f;
                    for (int mi = 0; mi < (int)r.muscles.size(); ++mi) {
                        float t;
                        float d = distToSegment(p, r.J[r.muscles[mi].a], r.J[r.muscles[mi].b], t);
                        if (d < bd) {
                            bd = d;
                            best = mi;
                        }
                    }
                    if (best >= 0) muscleVoxels[best].push_back(jp);
                } else {
                    detail::insertFlesh(octree, OID_FLESH, jp, step);
                    if (cb.onVoxel) cb.onVoxel(OID_FLESH, 3, jp);
                }
            }

    for (int mi = 0; mi < (int)rig.muscles.size(); ++mi) {
        rig.muscles[mi].voxels = std::move(muscleVoxels[mi]);
        if (cb.onMuscle) cb.onMuscle(rig.muscles[mi]);
    }

    return rig;
}

template <typename T>
inline size_t bindJoints(Grid::Octree<T>& octree, const CharacterRig& rig,
                         float detail, float strength = 220.0f, float jointStiffness = 600.0f) {
    size_t bondsMade = 0;
    float cuff = detail * 3.0f;

    for (const auto& j : rig.joints) {
        if (j.parentObjId < 0) continue;
        auto near = octree.findInRadius(j.position, cuff + rig.geom.scale * 0.03f, -1);
        std::vector<std::shared_ptr<Grid::NodeData_<T>>> parentSide, childSide;
        for (auto& n : near) {
            if (n->objectId == j.parentObjId) parentSide.push_back(n);
            else if (n->objectId == j.childObjId) childSide.push_back(n);
        }
        for (auto& a : parentSide)
        for (auto& b : childSide) {
            float rest = (a->position - b->position).norm();
            if (rest < 1e-5f || rest > cuff * 1.5f) continue;
            octree.addJointBond(a, b, rest, strength, jointStiffness);
            ++bondsMade;
        }
    }
    return bondsMade;
}

template <typename T>
inline size_t emitLayer(Grid::Octree<T>& octree, const SkeletonRig& r,
                        float step, int wantLayer) {
    static std::mt19937 rng(20260729u);
    std::uniform_real_distribution<float> jitter(-0.0015f, 0.0015f);
    Vec3 lo, hi; rigBounds(r, step, lo, hi);
    size_t inserted = 0;
    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                if (classify(r, p) != wantLayer) continue;
                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                if (wantLayer == 1) {
                    int bi = nearestBone(r, p);
                    detail::insertBone(octree, OID_SKELETON_BASE + (bi < 0 ? 0 : bi), jp, step);
                } else if (wantLayer == 2) {
                    detail::insertMuscle(octree, OID_MUSCLE, jp, step);
                } else {
                    detail::insertFlesh(octree, OID_FLESH, jp, step);
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
    size_t n = 0;
    RigCallbacks cb;
    cb.onVoxel = [&n](int, int, const Vec3&) { ++n; };
    buildRiggedCharacter(octree, height, detail, base, cb);
    return n;
}

}