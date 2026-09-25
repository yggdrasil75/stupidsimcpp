// Physics engine, Vulkan compute, position based dynamics.
// Fluids are solved with PBF density constraints, rigid and soft bodies with
// per-object shape matching. Static voxels are rasterized into a bit
// occupancy grid and every particle is swept through it and pushed out of
// it each iteration, which is what stops liquids seeping through walls.
// Intra-object bonds are strain checked on the GPU for fracture, bonds that
// cross objects (anchors, joints, muscle fibres) are XPBD distance
// constraints. The CPU side packs flat arrays, submits once per frame, then
// writes the results back into the octree.
#include "grid3eigen.hpp"

namespace Grid {

template<typename T>
struct Fragment_ {
    std::vector<std::shared_ptr<NodeData_<T>>> nodes;
    int sourceObjectId = -1;
};

template<typename T>
struct PhysRelocation_ {
    std::shared_ptr<NodeData_<T>> node;
    Vec3 target;
    uint32_t start;
    int depth;
};

template<typename T>
void Octree<T>::stepPhysics(float dt) {
    multiStepPhysics(dt, 1);
}

template<typename T>
const PhysicsMaterial_* Octree<T>::physMatOf(const std::shared_ptr<NodeData>& n,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) const {
    int oi = n->objectId + 1;
    if (oi < 0 || oi >= (int)fastMatsSize) return nullptr;
    if (n->physMatIdx >= fastMats[oi].size()) return nullptr;
    return &fastMats[oi][n->physMatIdx];
}

#ifdef VULKAN_SUPPORT

template<typename T>
void Octree<T>::collectStaticVoxels(const std::vector<std::vector<PhysicsMaterial_>>& fastMats,
                                    std::vector<float>& out, float& minSize) {
    out.clear();
    minSize = std::numeric_limits<float>::max();
    std::vector<uint32_t> stack{root_};
    while (!stack.empty()) {
        uint32_t cur = stack.back();
        stack.pop_back();
        const OctreeNode* node = nodeAt(cur);
        if (!node || !node->isLoaded()) continue;
        for (const auto& pt : pointsView(cur)) {
            if (!pt || !pt->isActive()) continue;
            const PhysicsMaterial_* m = physMatOf(pt, fastMats, fastMats.size());
            bool solid = !m || m->type == BodyType::STATIC || m->type == BodyType::KINEMATIC || pt->isStatic();
            if (!solid) continue;
            out.push_back(pt->position.x());
            out.push_back(pt->position.y());
            out.push_back(pt->position.z());
            out.push_back(pt->size);
            minSize = std::min(minSize, pt->size);
        }
        if (node->isLeaf()) continue;
        for (int i = 0; i < 8; ++i) {
            if (node->hasChild(i)) stack.push_back(node->firstChild + i);
        }
    }
    if (out.empty()) minSize = 0.1f;
}

template<typename T>
void Octree<T>::buildPhysicsMaterialTable(std::vector<std::vector<PhysicsMaterial_>>& fastMats) {
    s_lock lock(objectsMutex_);
    int maxObjId = -1;
    for (const auto& p : objects_) {
        maxObjId = std::max(maxObjId, p.first);
    }
    fastMats.assign(maxObjId + 2, {});
    for (const auto& p : objects_) {
        s_lock ol(p.second->objMutex);
        fastMats[p.first + 1] = p.second->physicsMaterials;
    }
}

template<typename T>
void Octree<T>::gatherDynamicNodes(const std::vector<std::vector<PhysicsMaterial_>>& fastMats,
                                   std::vector<std::shared_ptr<NodeData>>& dyn) {
    std::lock_guard<std::mutex> lock(physicsMutex_);
    std::unordered_set<NodeData*> seen;
    size_t w = 0;
    for (size_t i = 0; i < activePhysicsNodes_.size(); ++i) {
        auto sp = activePhysicsNodes_[i].lock();
        if (!sp || !seen.insert(sp.get()).second) continue;
        activePhysicsNodes_[w++] = activePhysicsNodes_[i];
        if (!sp->isActive() || sp->isStatic()) continue;
        const PhysicsMaterial_* m = physMatOf(sp, fastMats, fastMats.size());
        if (!m) continue;
        if (m->type == BodyType::FLUID || m->type == BodyType::RIGID || m->type == BodyType::SOFT) dyn.push_back(sp);
    }
    activePhysicsNodes_.resize(w);

    // rigid and soft bodies grouped by object first, fluids after
    auto isFluid = [&](const std::shared_ptr<NodeData>& n) {
        return physMatOf(n, fastMats, fastMats.size())->type == BodyType::FLUID;
    };
    std::stable_sort(dyn.begin(), dyn.end(), [&](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
        bool fa = isFluid(a);
        bool fb = isFluid(b);
        if (fa != fb) return !fa;
        return a->objectId < b->objectId;
    });
}

template<typename T>
void Octree<T>::packPhysicsParticles(const std::vector<std::vector<PhysicsMaterial_>>& fastMats,
                                     const std::vector<std::shared_ptr<NodeData>>& dyn,
                                     PhysFrameInput& in, float& maxFluidSize, float& maxSize) {
    const uint32_t N = (uint32_t)dyn.size();
    in.pos.resize(N * 4);
    in.vel.resize(N * 4);
    in.info.assign(N * 4, 0u);
    in.rest.assign(N * 4, 0.0f);
    maxFluidSize = 0.0f;
    maxSize = 0.0f;

    uint32_t objSlot = PhysGpu_NO_OBJ;
    int curObj = std::numeric_limits<int>::min();
    for (uint32_t i = 0; i < N; ++i) {
        const auto& n = dyn[i];
        const PhysicsMaterial_* m = physMatOf(n, fastMats, fastMats.size());
        n->physics.lastTreePos = n->position;
        in.pos[i * 4 + 0] = n->position.x();
        in.pos[i * 4 + 1] = n->position.y();
        in.pos[i * 4 + 2] = n->position.z();
        in.pos[i * 4 + 3] = n->size;
        Vec3 v = n->physics.velocity;
        if (!v.allFinite()) v.setZero();
        in.vel[i * 4 + 0] = v.x();
        in.vel[i * 4 + 1] = v.y();
        in.vel[i * 4 + 2] = v.z();
        in.vel[i * 4 + 3] = std::max(m->mass, 1e-5f);
        std::memcpy(&in.info[i * 4 + 3], &m->restitution, sizeof(float));
        in.rest[i * 4 + 3] = m->friction;
        maxSize = std::max(maxSize, n->size);
        if (m->type == BodyType::FLUID) {
            in.info[i * 4 + 0] = PhysGpu_NO_OBJ;
            in.info[i * 4 + 1] = PhysGpu_TYPE_FLUID;
            maxFluidSize = std::max(maxFluidSize, n->size);
            continue;
        }
        if (n->objectId != curObj) {
            curObj = n->objectId;
            objSlot = (uint32_t)in.objs.size();
            PhysObjGpu o;
            o.start = i;
            o.alpha = (m->type == BodyType::SOFT) ? phys_softStiffness : 1.0f;
            in.objs.push_back(o);
        }
        in.objs.back().count++;
        in.info[i * 4 + 0] = objSlot;
        in.info[i * 4 + 1] = (m->type == BodyType::SOFT) ? PhysGpu_TYPE_SOFT : PhysGpu_TYPE_RIGID;
    }

    // rest positions relative to each object's mass centre. Accumulated in
    // double: a float running sum over thousands of voxels is off by ~1e-4,
    // which shape matching would turn into a steady drift.
    for (auto& o : in.objs) {
        Eigen::Vector3d cmd = Eigen::Vector3d::Zero();
        double mt = 0.0;
        for (uint32_t i = o.start; i < o.start + o.count; ++i) {
            double m = in.vel[i * 4 + 3];
            cmd += m * dyn[i]->position.template cast<double>();
            mt += m;
        }
        cmd /= std::max(mt, 1e-9);
        Vec3 cm = cmd.template cast<float>();
        o.cm[0] = cm.x();
        o.cm[1] = cm.y();
        o.cm[2] = cm.z();
        for (uint32_t i = o.start; i < o.start + o.count; ++i) {
            Vec3 q = dyn[i]->position - cm;
            in.rest[i * 4 + 0] = q.x();
            in.rest[i * 4 + 1] = q.y();
            in.rest[i * 4 + 2] = q.z();
        }
    }
}

template<typename T>
void Octree<T>::packPhysicsBonds(const std::vector<std::vector<PhysicsMaterial_>>& fastMats,
                                 const std::vector<std::shared_ptr<NodeData>>& dyn,
                                 PhysFrameInput& in, std::vector<uint32_t>& ibondIds) {
    const uint32_t N = (uint32_t)dyn.size();
    std::unordered_map<NodeData*, uint32_t> index;
    index.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
        index[dyn[i].get()] = i;
    }

    std::vector<std::vector<PhysXBondGpu>> xPer(N);
    for (uint32_t i = 0; i < N; ++i) {
        const auto& n = dyn[i];
        const PhysicsMaterial_* m = physMatOf(n, fastMats, fastMats.size());
        if (m->type == BodyType::FLUID) continue;
        uint32_t self = n->id;
        for (uint32_t bid = n->physics.bondHead; bid != INVALID_IDX; ) {
            Bond_<T>& bond = store_.bonds.arena[bid];
            uint32_t next = bond.nextFor(self);
            if (!bond.live || bond.broken) {
                bid = next;
                continue;
            }
            uint32_t otherId = bond.other(self);
            auto other = store_.points.byId(otherId);
            if (!other || !other->isActive()) {
                bid = next;
                continue;
            }
            auto it = index.find(other.get());
            float k = (bond.stiffnessOverride > 0.0f) ? bond.stiffnessOverride : m->stiffness;
            bool intra = it != index.end() && other->objectId == n->objectId && !bond.toAnchor && !bond.isFiber;
            if (intra) {
                if (self < otherId) {
                    PhysIBondGpu ib;
                    ib.a = i;
                    ib.b = it->second;
                    ib.rest = bond.restLength;
                    ib.tension = bond.strength;
                    ib.compression = bond.strength * m->breakCompressionScale;
                    ib.stiffness = k;
                    ib.broken = 0u;
                    ib.pad = 0u;
                    in.ibonds.push_back(ib);
                    ibondIds.push_back(bid);
                }
                bid = next;
                continue;
            }
            PhysXBondGpu xb{};
            xb.rest = bond.restLength;
            xb.stiffness = k;
            xb.kind = bond.isFiber ? PhysGpu_XBOND_FIBER : PhysGpu_XBOND_JOINT;
            uint32_t slot = in.info[i * 4 + 0];
            if (slot != PhysGpu_NO_OBJ) {
                if (bond.isFiber) in.objs[slot].fiberBonds++;
                else in.objs[slot].jointBonds++;
            }
            if (it != index.end()) {
                xb.other = it->second;
                xb.isFixed = 0u;
            } else {
                xb.isFixed = 1u;
                xb.fixedPos[0] = other->position.x();
                xb.fixedPos[1] = other->position.y();
                xb.fixedPos[2] = other->position.z();
            }
            xPer[i].push_back(xb);
            bid = next;
        }
    }

    in.xoff.resize(N + 1);
    in.xoff[0] = 0;
    for (uint32_t i = 0; i < N; ++i) {
        in.xoff[i + 1] = in.xoff[i] + (uint32_t)xPer[i].size();
        in.xbonds.insert(in.xbonds.end(), xPer[i].begin(), xPer[i].end());
    }
}

template<typename T>
void Octree<T>::rebuildPhysicsOccupancy(const std::vector<std::vector<PhysicsMaterial_>>& fastMats,
                                        const BoundingBox& dom, PhysFrameInput& in) {
    float minStatic;
    collectStaticVoxels(fastMats, in.statics, minStatic);
    Vec3 ext = dom.second - dom.first;
    float vol = std::max(ext.x() * ext.y() * ext.z(), 1e-6f);
    // capped at 64M cells (8 MB of bits)
    float cs = std::max(minStatic * 0.5f, std::cbrt(vol / 64.0e6f));
    cs = std::max(cs, ext.maxCoeff() / 4096.0f);
    physOccCell_ = cs;
    physOccLo_ = dom.first;
    for (int a = 0; a < 3; ++a) {
        physOccDim_[a] = std::max(1u, (uint32_t)std::ceil(ext[a] / cs));
    }
    physOccBuilt_ = true;
    in.rebuildOcc = true;
}

template<typename T>
void Octree<T>::fillPhysicsPushConstants(const BoundingBox& dom, uint32_t numParticles,
                                         float subDt, float maxFluidSize, float maxSize, PhysPush& pc) {
    for (int a = 0; a < 3; ++a) {
        pc.gridLo[a] = physOccLo_[a];
        pc.gridDim[a] = physOccDim_[a];
        pc.domLo[a] = dom.first[a];
        pc.domHi[a] = dom.second[a];
        pc.gravity[a] = phys_useGravityPoint ? phys_gravityCenter[a] : phys_gravity[a];
    }
    // the kernel radius is widened so coarse fluids still see neighbours
    const float h = std::max(phys_smoothingRadius, 2.2f * maxFluidSize);
    pc.occCell = physOccCell_;
    pc.numParticles = numParticles;
    pc.gravityStrength = phys_gravityStrength;
    pc.dt = subDt;
    pc.h = h;
    pc.hashCellSize = std::max(h, maxSize);
    pc.numHashCells = 0;
    pc.damping = phys_velocityDamping;
    pc.xsph = phys_xsphEpsilon;
    pc.surfaceTension = phys_surfaceTension;
    pc.count = 0;
    pc.iteration = 0;
    pc.flags = 0;
    if (phys_useGravityPoint) pc.flags |= PhysGpu_FLAG_GRAVITY_POINT;
    if (phys_solidBoundary) pc.flags |= PhysGpu_FLAG_SOLID_BOUNDARY;
}

template<typename T>
void Octree<T>::writeBackPhysics(const std::vector<std::shared_ptr<NodeData>>& dyn,
                                 const std::vector<float>& outPos, const std::vector<float>& outVel) {
    const uint32_t N = (uint32_t)dyn.size();
    const float sleep2 = 1e-6f;
    std::vector<PhysRelocation_<T>> relocs;
    relocs.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
        const auto& n = dyn[i];
        Vec3 v(outVel[i * 4], outVel[i * 4 + 1], outVel[i * 4 + 2]);
        Vec3 p(outPos[i * 4], outPos[i * 4 + 1], outPos[i * 4 + 2]);
        if (!v.allFinite()) v.setZero();
        n->physics.velocity = v;
        n->setSettled(v.squaredNorm() < sleep2);
        if (!p.allFinite() || p == n->physics.lastTreePos) continue;
        std::vector<Vec3> span = {n->physics.lastTreePos, p};
        int depth = 0;
        uint32_t start = getHighestCommonNode(span, root_, depth);
        if (start == INVALID_IDX) {
            start = root_;
            depth = 0;
        }
        relocs.push_back({n, p, start, depth});
    }
    std::sort(relocs.begin(), relocs.end(), [](const PhysRelocation_<T>& a, const PhysRelocation_<T>& b) {
        return a.start < b.start;
    });

    size_t g = 0;
    while (g < relocs.size()) {
        size_t gEnd = g;
        while (gEnd < relocs.size() && relocs[gEnd].start == relocs[g].start) ++gEnd;
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = g; i < gEnd; ++i) {
            auto& rc = relocs[i];
            auto pd = rc.node;
            if (!removeRecursive(rc.start, pd->getCubeBounds(), pd)) removeRecursive(root_, pd->getCubeBounds(), pd);
            pd->position = rc.target;
            bool inserted = insertRecursive(rc.start, pd, rc.depth);
            if (!inserted) inserted = insertRecursive(root_, pd, 0);
            if (!inserted) {
                #pragma omp atomic
                size--;
            }
            pd->physics.lastTreePos = rc.target;
        }
        g = gEnd;
    }
}

// Keeps GridObject::relativeVoxels in sync with moved voxels so lookups by
// object id (makeObjectFluid, fracture, getWeakNodesByObjectId) still resolve.
// centerPosition is kept at zero so centerPosition + relPos == position exactly.
template<typename T>
void Octree<T>::refreshPhysicsObjectLayouts(const std::vector<std::shared_ptr<NodeData>>& dyn,
                                            const std::vector<float>& oldPos) {
    const uint32_t N = (uint32_t)dyn.size();
    std::unordered_map<int, std::vector<uint32_t>> byObj;
    for (uint32_t i = 0; i < N; ++i) {
        byObj[dyn[i]->objectId].push_back(i);
    }
    auto key = [](const Vec3& p) {
        return std::array<int64_t, 3>{(int64_t)std::llround(p.x() * 1e4),
                                      (int64_t)std::llround(p.y() * 1e4),
                                      (int64_t)std::llround(p.z() * 1e4)};
    };
    for (auto& [oid, idx] : byObj) {
        auto obj = getObject(oid);
        if (!obj) continue;
        std::vector<Vec3> keep;
        {
            s_lock lock(obj->objMutex);
            if (obj->relativeVoxels.size() > idx.size()) {
                std::unordered_set<std::array<int64_t, 3>, Vec3fHash> moved;
                for (uint32_t i : idx) {
                    moved.insert(key(Vec3(oldPos[i * 4], oldPos[i * 4 + 1], oldPos[i * 4 + 2])));
                }
                for (const auto& rv : obj->relativeVoxels) {
                    Vec3 abs = obj->centerPosition + rv.relPos;
                    if (!moved.count(key(abs))) keep.push_back(abs);
                }
            }
        }
        u_lock lock(obj->objMutex);
        obj->centerPosition = Vec3::Zero();
        obj->relativeVoxels.clear();
        obj->relativeVoxels.reserve(idx.size() + keep.size());
        for (uint32_t i : idx) {
            obj->relativeVoxels.push_back({dyn[i]->position});
        }
        for (const auto& p : keep) {
            obj->relativeVoxels.push_back({p});
        }
    }
}

template<typename T>
void Octree<T>::multiStepPhysics(float dt, int steps) {
    TIME_FUNCTION;
    if (root_ == INVALID_IDX || dt <= 0.0f || steps < 1) return;

    std::vector<std::vector<PhysicsMaterial_>> fastMats;
    buildPhysicsMaterialTable(fastMats);

    std::vector<std::shared_ptr<NodeData>> dyn;
    {
        ScopedFunctionTimer _t("multiStepPhysics.gather");
        gatherDynamicNodes(fastMats, dyn);
    }

    const bool needOcc = physicsCollidersDirty_.exchange(false) || !physOccBuilt_;
    if (dyn.empty() && !needOcc) return;

    PhysFrameInput in;
    float maxFluidSize;
    float maxSize;
    {
        ScopedFunctionTimer _t("multiStepPhysics.pack");
        packPhysicsParticles(fastMats, dyn, in, maxFluidSize, maxSize);
    }
    std::vector<uint32_t> ibondIds;
    {
        ScopedFunctionTimer _t("multiStepPhysics.bonds");
        packPhysicsBonds(fastMats, dyn, in, ibondIds);
    }

    const BoundingBox dom = nodeAt(root_)->bounds();
    if (needOcc) {
        ScopedFunctionTimer _t("multiStepPhysics.occupancy");
        rebuildPhysicsOccupancy(fastMats, dom, in);
    }
    fillPhysicsPushConstants(dom, (uint32_t)dyn.size(), dt / steps, maxFluidSize, maxSize, in.pc);
    in.steps = steps;
    in.iterations = phys_iterations;

    std::vector<float> outPos;
    std::vector<float> outVel;
    std::vector<uint32_t> outBroken;
    {
        ScopedFunctionTimer _t("multiStepPhysics.gpu");
        if (!physGpu_.run(in, outPos, outVel, outBroken)) {
            static bool warned = false;
            if (!warned) {
                std::cerr << "[physics] Vulkan compute unavailable; physics disabled.\n";
                warned = true;
            }
            return;
        }
    }
    if (dyn.empty()) return;

    {
        ScopedFunctionTimer _t("multiStepPhysics.writeback");
        writeBackPhysics(dyn, outPos, outVel);
        refreshPhysicsObjectLayouts(dyn, in.pos);
    }

    std::unordered_set<int> fractured;
    std::vector<uint32_t> tensionSeeds;
    for (size_t b = 0; b < outBroken.size(); ++b) {
        if (!outBroken[b]) continue;
        fractured.insert(dyn[in.ibonds[b].a]->objectId);
        if (outBroken[b] == 1u) tensionSeeds.push_back(ibondIds[b]);
        breakBond(ibondIds[b]);
    }
    propagateCracks(tensionSeeds, fractured);
    for (int objId : fractured) {
        resolveFracture(objId, fastMats, fastMats.size());
    }

    if (pointPoolFragmentation() > 3.0f) store_.points.compact();
}

#else

template<typename T>
void Octree<T>::multiStepPhysics(float, int) {
    static bool warned = false;
    if (!warned) {
        std::cerr << "[physics] built without VULKAN_SUPPORT; physics disabled.\n";
        warned = true;
    }
}

#endif

static constexpr size_t MAX_CRACKS_PER_OBJECT = 8;
static constexpr size_t MAX_CRACK_FRONT = 128;

template<typename T>
void Octree<T>::propagateCracks(const std::vector<uint32_t>& seeds, std::unordered_set<int>& fractured) {
    TIME_FUNCTION;
    for (uint32_t bid : seeds) {
        const Bond_<T>& bond = store_.bonds.arena[bid];
        auto a = store_.points.byId(bond.idA);
        auto b = store_.points.byId(bond.idB);
        if (!a || !b) continue;
        Vec3 dir = b->position - a->position;
        float len = dir.norm();
        if (len < 1e-6f) continue;
        dir /= len;
        Vec3 mid = 0.5f * (a->position + b->position);
        float tol = 0.6f * a->size;
        std::vector<PhysCrack_>& cracks = physCracks_[a->objectId];
        bool merged = false;
        for (auto& c : cracks) {
            if (std::abs(c.normal.dot(dir)) < 0.9f) continue;
            if (std::abs(c.normal.dot(mid - c.origin)) > tol) continue;
            c.front.push_back(mid);
            c.idleFrames = 0;
            merged = true;
            break;
        }
        if (merged) continue;
        if (cracks.size() >= MAX_CRACKS_PER_OBJECT) cracks.erase(cracks.begin());
        PhysCrack_ c;
        c.origin = mid;
        c.normal = dir;
        c.front.push_back(mid);
        cracks.push_back(c);
    }

    for (auto it = physCracks_.begin(); it != physCracks_.end(); ) {
        int objectId = it->first;
        std::vector<PhysCrack_>& cracks = it->second;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        for (auto& c : cracks) {
            std::vector<Vec3> next;
            for (int round = 0; round < 3 && !c.front.empty(); ++round) {
                if (c.front.size() > MAX_CRACK_FRONT) c.front.resize(MAX_CRACK_FRONT);
                Vec3 lo = c.front.front();
                Vec3 hi = lo;
                for (const Vec3& f : c.front) {
                    lo = lo.cwiseMin(f);
                    hi = hi.cwiseMax(f);
                }
                next.clear();
                for (const auto& n : nodes) {
                    float reach = 1.7f * n->size;
                    if ((n->position.array() < lo.array() - reach).any()) continue;
                    if ((n->position.array() > hi.array() + reach).any()) continue;
                    bool near = false;
                    for (const Vec3& f : c.front) {
                        if ((n->position - f).squaredNorm() <= reach * reach) {
                            near = true;
                            break;
                        }
                    }
                    if (!near) continue;
                    float sideA = c.normal.dot(n->position - c.origin);
                    uint32_t selfId = n->id;
                    for (uint32_t bid = n->physics.bondHead; bid != INVALID_IDX; ) {
                        Bond_<T>& bond = store_.bonds.arena[bid];
                        uint32_t nextBid = bond.nextFor(selfId);
                        if (!bond.live || bond.broken || bond.toAnchor || bond.isFiber) {
                            bid = nextBid;
                            continue;
                        }
                        auto other = store_.points.byId(bond.other(selfId));
                        if (!other || other->objectId != objectId) {
                            bid = nextBid;
                            continue;
                        }
                        float sideB = c.normal.dot(other->position - c.origin);
                        bool crosses = (sideA <= 0.0f) != (sideB <= 0.0f);
                        bool close = std::abs(sideA) < 1.2f * n->size && std::abs(sideB) < 1.2f * n->size;
                        if (crosses && close) {
                            next.push_back(0.5f * (n->position + other->position));
                            breakBond(bid);
                            fractured.insert(objectId);
                        }
                        bid = nextBid;
                    }
                }
                c.front = next;
            }
            if (c.front.empty()) c.idleFrames++;
        }
        cracks.erase(std::remove_if(cracks.begin(), cracks.end(),
                                    [](const PhysCrack_& c) { return c.idleFrames > 2; }),
                     cracks.end());
        if (cracks.empty()) it = physCracks_.erase(it);
        else ++it;
    }
}

template<typename T>
void Octree<T>::resolveFracture(int objectId,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) {
    TIME_FUNCTION;

    std::vector<std::shared_ptr<NodeData>> nodes;
    collectNodesByObjectId(objectId, nodes);
    if (nodes.size() < 2) return;

    std::unordered_map<uint32_t, uint32_t> component;
    component.reserve(nodes.size());
    for (const auto& n : nodes) component[n->id] = INVALID_IDX;

    std::vector<Fragment_<T>> fragments;
    std::vector<std::shared_ptr<NodeData>> stack;

    for (const auto& seed : nodes) {
        if (component[seed->id] != INVALID_IDX) continue;
        uint32_t cid = static_cast<uint32_t>(fragments.size());
        fragments.push_back(Fragment_<T>{});
        fragments[cid].sourceObjectId = objectId;

        stack.clear();
        stack.push_back(seed);
        component[seed->id] = cid;

        while (!stack.empty()) {
            std::shared_ptr<NodeData> cur = stack.back();
            stack.pop_back();
            fragments[cid].nodes.push_back(cur);

            uint32_t selfId = cur->id;
            for (uint32_t bid = cur->physics.bondHead; bid != INVALID_IDX; ) {
                Bond_<T>& bond = store_.bonds.arena[bid];
                uint32_t nextBid = bond.nextFor(selfId);
                if (!bond.live || bond.toAnchor) {
                    bid = nextBid;
                    continue;
                }
                uint32_t otherId = bond.other(selfId);
                auto slot = component.find(otherId);
                if (slot == component.end() || slot->second != INVALID_IDX) {
                    bid = nextBid;
                    continue;
                }
                auto other = store_.points.byId(otherId);
                if (!other) {
                    bid = nextBid;
                    continue;
                }
                slot->second = cid;
                stack.push_back(other);
                bid = nextBid;
            }
        }
    }

    if (fragments.size() < 2) return;

    size_t largest = 0;
    for (size_t i = 1; i < fragments.size(); ++i) {
        if (fragments[i].nodes.size() > fragments[largest].nodes.size()) largest = i;
    }

    SplitPolicy policy = SplitPolicy::NEW_OID;
    uint32_t minFragment = 1;
    if (auto obj = getObject(objectId)) {
        policy = obj->splitPolicy;
        int oi = objectId + 1;
        if (oi >= 0 && oi < (int)fastMatsSize && !fastMats[oi].empty())
            minFragment = fastMats[oi][0].minFragmentVoxels;
    }

    if (policy == SplitPolicy::KEEP_OID) return;

    bool frozeAny = false;
    for (size_t i = 0; i < fragments.size(); ++i) {
        if (i == largest) continue;
        Fragment_<T>& frag = fragments[i];

        if (policy == SplitPolicy::DISSOLVE) {
            std::unordered_set<std::shared_ptr<NodeData>> doomed(frag.nodes.begin(), frag.nodes.end());
            size -= removeSpecificNodesBatchRecursive(root_, doomed);
            continue;
        }
        // too small to be its own body: stays with the source object and
        // rides along through shape matching
        if (frag.nodes.size() < minFragment) continue;

        if (policy == SplitPolicy::SHED_STATIC) {
            freezeFragment(frag.nodes);
            frozeAny = true;
            continue;
        }

        reassignFragment(frag.nodes, objectId);
    }

    if (frozeAny) physicsCollidersDirty_.store(true);
}

template<typename T>
void Octree<T>::freezeFragment(const std::vector<std::shared_ptr<NodeData>>& frag) {
    auto obj = getOrCreateObject(frag.front()->objectId);
    PhysicsMaterial_ pmat;
    pmat.type = BodyType::STATIC;
    uint16_t staticIdx = obj->getOrAddPhysicsMaterial(pmat);

    for (const auto& n : frag) {
        n->physics.velocity.setZero();
        n->physics.force.setZero();
        clearBondsOf(n);
        n->physMatIdx = staticIdx;
        n->setStatic(true);
        n->setSettled(false);
    }
}

template<typename T>
void Octree<T>::reassignFragment(const std::vector<std::shared_ptr<NodeData>>& frag, int sourceObjectId) {
    auto src = getObject(sourceObjectId);
    auto dst = getOrCreateObject(-1);
    if (!dst) return;

    if (src) {
        s_lock srcLock(src->objMutex);
        u_lock dstLock(dst->objMutex);
        dst->splitPolicy = src->splitPolicy;
        dst->objectFlags = src->objectFlags;
        dst->physicsMaterials = src->physicsMaterials;
        dst->physicsMatMap = src->physicsMatMap;
    }

    Vec3 center = Vec3::Zero();
    for (const auto& n : frag) center += n->position;
    center /= static_cast<float>(frag.size());

    {
        u_lock dstLock(dst->objMutex);
        dst->centerPosition = center;
        dst->relativeVoxels.clear();
        dst->relativeVoxels.reserve(frag.size());
        for (const auto& n : frag) dst->relativeVoxels.push_back({n->position - center});
    }

    for (const auto& n : frag) n->objectId = dst->id;
}

}
