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

struct RigStrandMuscle {
    int id;
    std::string name;
    MuscleRole role;
    Joint origin, insertion, crosses;
    int originObjId, insertObjId;
    Vec3 originPos, insertPos;
    std::vector<std::vector<Vec3>> fascicles;
    std::vector<Vec3> voxels;
    std::vector<Vec3> fiberDirs;
};

struct CharacterRig {
    SkeletonRig geom;
    std::vector<RigBone> bones;
    std::vector<RigJoint> joints;
    std::vector<RigMuscle> muscles;
    std::vector<RigStrandMuscle> strandMuscles;
    int fleshObjId = OID_FLESH;
    int muscleObjId = OID_MUSCLE;

    static constexpr int SUBOBJ_MUSCLE_BASE = 1000;

    int boneSubObjectId(int boneIndex) const { return boneIndex + 1; }
    int muscleSubObjectId(int muscleId) const { return SUBOBJ_MUSCLE_BASE + muscleId; }

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
    const RigStrandMuscle* findStrandMuscle(const std::string& n) const {
        for (auto& m : strandMuscles) if (m.name == n) return &m;
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
                if (b.a == headJoint) { parentBone = i; break; }
            }
        }
    }
}

inline JointLimits limitsFor(Joint jt) {
    switch (jt) {
        case J_ELBOW_L: case J_ELBOW_R:
        case J_KNEE_L: case J_KNEE_R:
            return { 0.0f, 2.4f, -0.1f, 0.1f };
        case J_INDEX_PIP_L: case J_INDEX_DIP_L:
        case J_MIDDLE_PIP_L: case J_MIDDLE_DIP_L:
        case J_RING_PIP_L: case J_RING_DIP_L:
        case J_PINKY_PIP_L: case J_PINKY_DIP_L:
        case J_INDEX_PIP_R: case J_INDEX_DIP_R:
        case J_MIDDLE_PIP_R: case J_MIDDLE_DIP_R:
        case J_RING_PIP_R: case J_RING_DIP_R:
        case J_PINKY_PIP_R: case J_PINKY_DIP_R:
        case J_THUMB_IP_L: case J_THUMB_IP_R:
            return { 0.0f, 1.75f, -0.05f, 0.05f };
        case J_INDEX_MCP_L: case J_MIDDLE_MCP_L:
        case J_RING_MCP_L:  case J_PINKY_MCP_L:
        case J_INDEX_MCP_R: case J_MIDDLE_MCP_R:
        case J_RING_MCP_R:  case J_PINKY_MCP_R:
        case J_THUMB_MCP_L: case J_THUMB_MCP_R:
            return { -0.35f, 1.6f, -0.35f, 0.35f };
        case J_BIGTOE_L: case J_TOE2_L: case J_TOE3_L: case J_TOE4_L: case J_TOE5_L:
        case J_BIGTOE_R: case J_TOE2_R: case J_TOE3_R: case J_TOE4_R: case J_TOE5_R:
            return { -0.6f, 1.0f, -0.1f, 0.1f };
        case J_NECK: case J_HEAD:
            return { -0.9f, 0.9f, -1.2f, 1.2f };
        case J_JAW:
            return { 0.0f, 0.5f, -0.1f, 0.1f };
        default:
            return {};
    }
}

inline CharacterRig assembleRig(float height, const Vec3& base, const BuildConfig& cfg) {
    CharacterRig rig;
    rig.geom = buildRig(height, base, cfg);

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
        rj.limits      = limitsFor(jt);
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

    auto boneAtJoint = [&](Joint j, bool preferTail) -> int {
        int match = -1;
        for (int k = 0; k < (int)rig.geom.bones.size(); ++k) {
            const Bone& bk = rig.geom.bones[k];
            if (preferTail && bk.b == j) return k;
            if (!preferTail && bk.a == j) return k;
            if (bk.a == j || bk.b == j) match = k;
        }
        return match;
    };
    for (const auto& routed : rig.geom.routed) {
        RigStrandMuscle sm;
        sm.id        = routed.id;
        sm.name      = routed.name;
        sm.role      = routed.role;
        sm.origin    = routed.origin;
        sm.insertion = routed.insertion;
        sm.crosses   = routed.crosses;
        sm.originPos = routed.originPos;
        sm.insertPos = routed.insertPos;
        sm.fascicles = routed.fascicles;
        int bo = boneAtJoint(routed.origin, true);
        int bi = boneAtJoint(routed.insertion, false);
        if (bo < 0) bo = nearestBone(rig.geom, routed.originPos);
        if (bi < 0) bi = nearestBone(rig.geom, routed.insertPos);
        sm.originObjId = bo >= 0 ? rig.boneObjectId(bo) : -1;
        sm.insertObjId = bi >= 0 ? rig.boneObjectId(bi) : -1;
        rig.strandMuscles.push_back(sm);
    }

    return rig;
}

inline CharacterRig assembleRig(float height, const Vec3& base) {
    return assembleRig(height, base, BuildConfig{});
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

template <typename T, typename Emit>
inline void voxeliseLayer(Grid::Octree<T>& octree, const SkeletonRig& r,
                          int wantLayer, float step, std::mt19937& rng, Emit&& emit) {
    std::uniform_real_distribution<float> jitter(-step * 0.15f, step * 0.15f);
    Vec3 lo, hi; rigBounds(r, step, lo, hi);
    for (float x = lo.x(); x <= hi.x(); x += step)
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float z = lo.z(); z <= hi.z(); z += step) {
                Vec3 p(x, y, z);
                if (classify(r, p) != wantLayer) continue;
                Vec3 jp(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                emit(p, jp, step);
            }
}

}

template <typename T>
inline CharacterRig buildRiggedCharacter(Grid::Octree<T>& octree, float height,
                                         const BuildConfig& cfg,
                                         const Vec3& base = Vec3::Zero(),
                                         const RigCallbacks& cb = {}) {
    CharacterRig rig = assembleRig(height, base, cfg);
    const SkeletonRig& r = rig.geom;

    static std::mt19937 rng(20260729u);

    if (cb.onBone)  for (auto& b : rig.bones)  cb.onBone(b);
    if (cb.onJoint) for (auto& j : rig.joints) cb.onJoint(j);

    const bool useStrands = cfg.useStrandMuscles && !r.routed.empty();
    std::vector<std::vector<Vec3>> legacyVoxels(rig.muscles.size());

    detail::voxeliseLayer(octree, r, 1, cfg.skeletonDetail, rng,
        [&](const Vec3& p, const Vec3& jp, float step) {
            int bi = nearestBone(r, p);
            if (bi < 0) return;
            int objId = rig.boneObjectId(bi);
            detail::insertBone(octree, objId, jp, step);
            if (cb.onVoxel) cb.onVoxel(objId, 1, jp);
        });

    detail::voxeliseLayer(octree, r, 2, cfg.muscleDetail, rng,
        [&](const Vec3& p, const Vec3& jp, float step) {
            detail::insertMuscle(octree, OID_MUSCLE, jp, step);
            if (cb.onVoxel) cb.onVoxel(OID_MUSCLE, 2, jp);
            if (useStrands) {
                Vec3 dir; int mid = -1;
                if (strandFiberDir(r, p, dir, mid) && mid >= 0 &&
                    mid < (int)rig.strandMuscles.size()) {
                    rig.strandMuscles[mid].voxels.push_back(jp);
                    rig.strandMuscles[mid].fiberDirs.push_back(dir);
                }
            } else {
                int best = -1; float bd = 1e30f;
                for (int mi = 0; mi < (int)r.muscles.size(); ++mi) {
                    float t;
                    float d = distToSegment(p, r.J[r.muscles[mi].a],
                                            r.J[r.muscles[mi].b], t);
                    if (d < bd) { bd = d; best = mi; }
                }
                if (best >= 0) legacyVoxels[best].push_back(jp);
            }
        });

    // --- flesh / skin ---
    detail::voxeliseLayer(octree, r, 3, cfg.skinDetail, rng,
        [&](const Vec3&, const Vec3& jp, float step) {
            detail::insertFlesh(octree, OID_FLESH, jp, step);
            if (cb.onVoxel) cb.onVoxel(OID_FLESH, 3, jp);
        });

    if (!useStrands) {
        for (int mi = 0; mi < (int)rig.muscles.size(); ++mi) {
            rig.muscles[mi].voxels = std::move(legacyVoxels[mi]);
            if (cb.onMuscle) cb.onMuscle(rig.muscles[mi]);
        }
    }

    return rig;
}

template <typename T>
inline CharacterRig buildRiggedCharacter(Grid::Octree<T>& octree, float height, float detail,
                                         const Vec3& base = Vec3::Zero(),
                                         const RigCallbacks& cb = {}) {
    return buildRiggedCharacter(octree, height, BuildConfig::fromLegacyDetail(detail), base, cb);
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
    size_t inserted = 0;
    detail::voxeliseLayer(octree, r, wantLayer, step, rng,
        [&](const Vec3& p, const Vec3& jp, float s) {
            if (wantLayer == 1) {
                int bi = nearestBone(r, p);
                detail::insertBone(octree, OID_SKELETON_BASE + (bi < 0 ? 0 : bi), jp, s);
            } else if (wantLayer == 2) {
                detail::insertMuscle(octree, OID_MUSCLE, jp, s);
            } else {
                detail::insertFlesh(octree, OID_FLESH, jp, s);
            }
            ++inserted;
        });
    return inserted;
}

template <typename T>
inline size_t buildSkeleton(Grid::Octree<T>& octree, float height, float detail,
                            const Vec3& base = Vec3::Zero()) {
    BuildConfig cfg = BuildConfig::fromLegacyDetail(detail);
    return emitLayer(octree, buildRig(height, base, cfg), cfg.skeletonDetail, 1);
}
template <typename T>
inline size_t buildMuscle(Grid::Octree<T>& octree, float height, float detail,
                          const Vec3& base = Vec3::Zero()) {
    BuildConfig cfg = BuildConfig::fromLegacyDetail(detail);
    return emitLayer(octree, buildRig(height, base, cfg), cfg.muscleDetail, 2);
}
template <typename T>
inline size_t buildFlesh(Grid::Octree<T>& octree, float height, float detail,
                         const Vec3& base = Vec3::Zero()) {
    BuildConfig cfg = BuildConfig::fromLegacyDetail(detail);
    return emitLayer(octree, buildRig(height, base, cfg), cfg.skinDetail, 3);
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

template <typename T>
inline size_t wireMusculature(Grid::Octree<T>& octree, const CharacterRig& rig,
                              int fibersPerMuscle = 6,
                              float fiberStrength = 120.0f,
                              float maxContraction = 0.35f) {
    using NodeT = Grid::NodeData_<T>;

    for (const auto& b : rig.bones) {
        octree.registerSubObject(b.objectId, rig.boneSubObjectId(b.index),
                                 b.name, Grid::SubObjectKind::BONE);
    }
    auto muscleObj = octree.getOrCreateObject(rig.muscleObjId);
    for (const auto& sm : rig.strandMuscles) {
        int sub = rig.muscleSubObjectId(sm.id);
        octree.registerSubObject(rig.muscleObjId, sub, sm.name,
                                 Grid::SubObjectKind::MUSCLE);
        auto& so = muscleObj->getOrCreateSubObject(sub, sm.name,
                                                   Grid::SubObjectKind::MUSCLE);
        so.originSubObj = -1;
        so.insertSubObj = -1;
    }

    {
        std::vector<std::shared_ptr<NodeT>> mv;
        octree.collectNodesByObjectId(rig.muscleObjId, mv);
        for (auto& nd : mv) {
            if (!nd) continue;
            int mid = nearestStrandMuscle(rig.geom, nd->position);
            if (mid >= 0) nd->subObjectId = rig.muscleSubObjectId(mid);
        }
    }

    size_t fibersMade = 0;
    for (const auto& sm : rig.strandMuscles) {
        int sub = rig.muscleSubObjectId(sm.id);

        int originBoneObj = sm.originObjId;
        int insertBoneObj = sm.insertObjId;
        if (originBoneObj < 0 || insertBoneObj < 0) continue;

        std::vector<std::shared_ptr<NodeT>> originVox, insertVox;
        octree.collectNodesByObjectId(originBoneObj, originVox);
        octree.collectNodesByObjectId(insertBoneObj, insertVox);
        if (originVox.empty() || insertVox.empty()) continue;

        int nF = std::min<int>(fibersPerMuscle, (int)sm.fascicles.size());
        if (nF <= 0) nF = std::min<int>(fibersPerMuscle, 1);

        for (int f = 0; f < nF; ++f) {
            Vec3 oP = sm.originPos, iP = sm.insertPos;
            if (f < (int)sm.fascicles.size() && sm.fascicles[f].size() >= 2) {
                oP = sm.fascicles[f].front();
                iP = sm.fascicles[f].back();
            }
            auto nearest = [](const std::vector<std::shared_ptr<NodeT>>& v,
                              const Vec3& p) -> std::shared_ptr<NodeT> {
                std::shared_ptr<NodeT> best; float bd = 1e30f;
                for (auto& n : v) {
                    if (!n) continue;
                    float d = (n->position - p).squaredNorm();
                    if (d < bd) { bd = d; best = n; }
                }
                return best;
            };
            auto a = nearest(originVox, oP);
            auto b = nearest(insertVox, iP);
            if (!a || !b || a.get() == b.get()) continue;
            if (octree.addMuscleFiber(muscleObj, sub, a, b,
                                      fiberStrength, 0.0f, maxContraction))
                ++fibersMade;
        }
    }
    return fibersMade;
}

}