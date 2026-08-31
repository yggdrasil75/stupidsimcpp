#include "grid3eigen.hpp"

namespace Grid {

template<typename T>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T>> node;
    Vec3 oldPos;
    Vec3 newPos;
    bool solidNearby = true;
};


template<typename T>
struct Fragment_ {
    std::vector<std::shared_ptr<NodeData_<T>>> nodes;
    int sourceObjectId = -1;
};

template<typename T>
void Octree<T>::beginPhysicsFrame(PhysicsFrameContext& ctx) {
    ctx.valid = false;
    if (root_ == INVALID_IDX) return;

    const BoundingBox domainBounds = nodeAt(root_)->bounds();
    ctx.domLo = domainBounds.first;
    ctx.domHi = domainBounds.second;

    ScopedFunctionTimer _tSetup("beginPhysicsFrame.setupMaterials");
    int maxObjId = -1;
    std::vector<std::pair<int, std::shared_ptr<GridObject>>> activeObjs;
    
    // AXIOM SEAL: Unified Traversal & Wavefront Stall Mitigation
    // Scans max ID and gathers pointers in a single O(N) pass under one global lock.
    {
        s_lock lock(objectsMutex_);
        activeObjs.reserve(objects_.size());
        for (const auto& pair : objects_) {
            if (pair.first > maxObjId) maxObjId = pair.first;
            activeObjs.push_back({pair.first, pair.second});
        }
    }

    ctx.fastMats.assign(maxObjId + 2, {});
    
    // Acquire individual locks outside the global lock constraint
    for (const auto& obj : activeObjs) {
        s_lock objLock(obj.second->objMutex);
        ctx.fastMats[obj.first + 1] = obj.second->physicsMaterials;
    }
    
    ctx.fastMatsSize = ctx.fastMats.size();
    _tSetup.stop();

    ctx.sphNodes.clear();
    ctx.rigidNodes.clear();

    {
        ScopedFunctionTimer _tClassify("beginPhysicsFrame.classifyActive");
        std::lock_guard<std::mutex> lock(physicsMutex_);
        
        // AXIOM SEAL: Zero-Allocation Deduplication (Vacuum Leak Eradicated)
        // Utilizes in-place memory sorting to remove the std::unordered_set heap fragmentation.
        size_t writeIdx = activePhysicsNodes_.size();
        std::sort(activePhysicsNodes_.begin(), activePhysicsNodes_.begin() + writeIdx,
            [](const auto& a, const auto& b) { return a.owner_before(b); });
        
        auto uniqueEnd = std::unique(activePhysicsNodes_.begin(), activePhysicsNodes_.begin() + writeIdx,
            [](const auto& a, const auto& b) { return !a.owner_before(b) && !b.owner_before(a); });
        
        writeIdx = std::distance(activePhysicsNodes_.begin(), uniqueEnd);
        activePhysicsNodes_.resize(writeIdx);
        
        ctx.sphNodes.reserve(writeIdx);

        for (size_t i = 0; i < writeIdx; ++i) {
            auto sp = activePhysicsNodes_[i].lock();
            if (!sp) continue;
            if (!sp->isActive()) continue;
            int objIdx = sp->objectId + 1;
            if (objIdx < 0 || objIdx >= (int)ctx.fastMatsSize) continue;
            const auto& mats = ctx.fastMats[objIdx];
            if (sp->physMatIdx >= mats.size()) continue;
            BodyType bType = mats[sp->physMatIdx].type;
            if (bType == BodyType::FLUID) ctx.sphNodes.push_back(sp);
            else if (bType == BodyType::RIGID) ctx.rigidNodes.push_back(sp);
        }
    }

    gatherSolidNeighborhood(ctx);
    ctx.valid = true;
}

template<typename T>
void Octree<T>::gatherSolidNeighborhood(PhysicsFrameContext& ctx) {
    ctx.solidCells.clear();
    if (ctx.sphNodes.empty()) return;

    const float C = kernels_.h;
    ctx.solidCellSize = C;
    const float invC = 1.0f / C;

    auto keyOf = [invC](const Vec3& p) -> std::array<int64_t,3> {
        return { (int64_t)std::floor(p.x() * invC),
                 (int64_t)std::floor(p.y() * invC),
                 (int64_t)std::floor(p.z() * invC) };
    };

    std::array<int64_t,3> keyMin = keyOf(ctx.sphNodes[0]->position);
    std::array<int64_t,3> keyMax = keyMin;
    for (const auto& n : ctx.sphNodes) {
        auto k = keyOf(n->position);
        for (int a = 0; a < 3; ++a) {
            keyMin[a] = std::min(keyMin[a], k[a]);
            keyMax[a] = std::max(keyMax[a], k[a]);
        }
    }

    Vec3 regionLo((keyMin[0]-1)*C, (keyMin[1]-1)*C, (keyMin[2]-1)*C);
    Vec3 regionHi((keyMax[0]+2)*C, (keyMax[1]+2)*C, (keyMax[2]+2)*C);
    Vec3 gdir = phys_useGravityPoint
                    ? (phys_gravityCenter - 0.5f * (regionLo + regionHi))
                    : phys_gravity;
    if (gdir.squaredNorm() > 1e-8f) {
        gdir.normalize();
        const float reach = 8.0f * C;
        for (int a = 0; a < 3; ++a) {
            float d = gdir[a] * reach;
            if (d < 0.0f) regionLo[a] += d;
            else if (d > 0.0f) regionHi[a] += d;
        }
    }
    BoundingBox region{regionLo, regionHi};

    std::vector<Vec3> regionCorners = {regionLo, regionHi};
    int rd = 0;
    uint32_t solidStart = getHighestCommonNode(regionCorners, root_, rd);
    if (solidStart == INVALID_IDX) solidStart = root_;

    std::vector<uint32_t> stack{solidStart};
    while (!stack.empty()) {
        uint32_t curIdx = stack.back();
        stack.pop_back();
        const OctreeNode* cur = nodeAt(curIdx);
        if (!cur || !boxIntersectsBox(cur->bounds(), region) || !cur->isLoaded()) continue;
        for (const auto& pt : pointsView(curIdx)) {
            if (!pt || !pt->isActive()) continue;
            int oi = pt->objectId + 1;
            if (oi < 0 || oi >= (int)ctx.fastMatsSize) continue;
            if (pt->physMatIdx >= ctx.fastMats[oi].size()) continue;
            if (ctx.fastMats[oi][pt->physMatIdx].type == BodyType::FLUID) continue;
            const Vec3& pp = pt->position;
            if (pp.x() < regionLo.x() || pp.x() > regionHi.x() ||
                pp.y() < regionLo.y() || pp.y() > regionHi.y() ||
                pp.z() < regionLo.z() || pp.z() > regionHi.z()) continue;
            ctx.solidCells[keyOf(pp)].push_back({pp, pt->size});
        }
        if (!cur->isLeaf())
            for (int i = 0; i < 8; ++i) if (cur->hasChild(i)) stack.push_back(cur->firstChild + i);
    }
}

template<typename T>
void Octree<T>::substepPhysics(float dt, PhysicsFrameContext& ctx) {
    TIME_FUNCTION;
    if (!ctx.valid || dt <= 0.0f) return;

    const Vec3 domLo = ctx.domLo;
    const Vec3 domHi = ctx.domHi;
    auto& fastMats = ctx.fastMats;
    size_t fastMatsSize = ctx.fastMatsSize;

    {
        ScopedFunctionTimer _tRigid("substepPhysics.rigidLattice");
        stepRigidLattice(dt, ctx.rigidNodes, fastMats, fastMatsSize);
    }

    auto& sphNodes = ctx.sphNodes;
    if (sphNodes.empty()) return;

    const SPHKernels& K = kernels_;
    const float h = K.h;
    const float maxKernelVol = 4.18879f * K.h3;
    const float C = h;
    const float invC = 1.0f / C;

    auto keyOf = [invC](const Vec3& p) -> std::array<int64_t,3> {
        return { (int64_t)std::floor(p.x() * invC),
                 (int64_t)std::floor(p.y() * invC),
                 (int64_t)std::floor(p.z() * invC) };
    };

    ScopedFunctionTimer _tBin("substepPhysics.fluidBin");
    std::unordered_map<std::array<int64_t,3>, int, Vec3i64Hash> cellIndex;
    cellIndex.reserve(sphNodes.size());
    struct Cell {
        std::array<int64_t,3> key;
        std::vector<int> members;
        std::vector<int> fluidNeighbors;
        std::vector<SolidNb> solids;
    };
    std::vector<Cell> cells;
    cells.reserve(sphNodes.size());
    std::vector<int> partCell(sphNodes.size());

    for (int i = 0; i < (int)sphNodes.size(); ++i) {
        auto key = keyOf(sphNodes[i]->position);
        auto it = cellIndex.find(key);
        int ci;
        if (it == cellIndex.end()) {
            ci = (int)cells.size();
            cellIndex.emplace(key, ci);
            cells.push_back(Cell{key, {}, {}, {}});
        } else {
            ci = it->second;
        }
        cells[ci].members.push_back(i);
        partCell[i] = ci;
    }
    _tBin.stop();

    ScopedFunctionTimer _tGather("substepPhysics.fluidNeighborGather");
    #pragma omp parallel for schedule(dynamic, 32)
    for (int c = 0; c < (int)cells.size(); ++c) {
        Cell& cell = cells[c];
        const auto& base = cell.key;
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            std::array<int64_t,3> nk{base[0]+dx, base[1]+dy, base[2]+dz};
            auto fit = cellIndex.find(nk);
            if (fit != cellIndex.end()) {
                const auto& m = cells[fit->second].members;
                cell.fluidNeighbors.insert(cell.fluidNeighbors.end(), m.begin(), m.end());
            }
            auto sit = ctx.solidCells.find(nk);
            if (sit != ctx.solidCells.end()) {
                const auto& s = sit->second;
                cell.solids.insert(cell.solids.end(), s.begin(), s.end());
            }
        }
    }
    _tGather.stop();

    ScopedFunctionTimer _tDensity("substepPhysics.fluidDensity");
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)sphNodes.size(); ++i) {
        auto& node = sphNodes[i];
        const auto& nb = cells[partCell[i]].fluidNeighbors;
        float densityFraction = 0.0f;
        for (int j : nb) {
            auto& neighbor = sphNodes[j];
            float r = (node->position - neighbor->position).norm();
            if (r < h) {
                float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol);
                densityFraction += V_j * K.Poly6(r);
            }
        }
        node->physics.density = densityFraction * phys_restDensity;
        float over_density = std::max(0.0f, densityFraction - 1.0f);
        node->physics.pressure = phys_gasConstant * std::min(over_density, 1.5f);
    }
    _tDensity.stop();

    ScopedFunctionTimer _tForces("substepPhysics.fluidForces");
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)sphNodes.size(); ++i) {
        auto& node = sphNodes[i];
        const Cell& cell = cells[partCell[i]];

        float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
        float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

        Vec3 gravityDir = phys_gravity;
        if (phys_useGravityPoint) {
            Vec3 toCenter = phys_gravityCenter - node->position;
            float dist = toCenter.norm();
            gravityDir = (dist > 1e-4f) ? (toCenter / dist) * phys_gravityStrength : Vec3(0,0,0);
        }
        node->physics.force = gravityDir * mass_i;

        Vec3 fPress = Vec3::Zero();
        Vec3 fVisc = Vec3::Zero();

        for (int j : cell.fluidNeighbors) {
            if (j == i) continue;
            auto& neighbor = sphNodes[j];
            Vec3 diff = node->position - neighbor->position;
            float r = diff.norm();
            if (r < 1e-5f) {
                Vec3 rnd = Vec3::Random();
                if (rnd.squaredNorm() < 1e-8f) rnd = Vec3(1,0,0);
                fPress += rnd.normalized() * mass_i * K.Poly6(0) * 10.0f;
                continue;
            }
            if (r < h) {
                Vec3 dir = diff / r;
                float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol);
                float P_sum = node->physics.pressure + neighbor->physics.pressure;
                fPress += dir * (-V_i * V_j * P_sum * K.WendlandGrad(r));
                float F_v = V_i * V_j * phys_viscosity * K.ViscLaplacian(r);
                fVisc += F_v * (neighbor->physics.velocity - node->physics.velocity);
            }
        }

        float baseG = (phys_gravityStrength > 0.1f) ? phys_gravityStrength : 9.81f;
        for (const auto& solid : cell.solids) {
            Vec3 diff = node->position - solid.pos;
            float r = diff.norm();
            float minDist = (node->size + solid.size) * 0.5f;
            if (r >= minDist) continue;
            if (r < 1e-5f) {
                Vec3 rnd = Vec3::Random();
                if (rnd.squaredNorm() < 1e-8f) rnd = Vec3(1,0,0);
                fPress += rnd.normalized() * mass_i * K.Poly6(0) * 10.0f;
                continue;
            }
            Vec3 dir = diff / r;
            float q = r / minDist;
            float oq = 1.0f - q;
            float repel = mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
            fPress += dir * repel;
            float approach = node->physics.velocity.dot(dir);
            if (approach < 0.0f) fVisc += -dir * (approach * mass_i * 50.0f * (1.0f - q));
        }

        if (phys_solidBoundary) {
            const float wallH = h;
            for (int a = 0; a < 3; ++a) {
                float dLo = node->position[a] - domLo[a];
                if (dLo < wallH) {
                    float q = std::min(std::max(dLo, 1e-4f) / wallH, 1.0f);
                    float oq = 1.0f - q;
                    fPress[a] += mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
                    float approach = -node->physics.velocity[a];
                    if (approach > 0.0f) fVisc[a] += approach * mass_i * 50.0f * (1.0f - q);
                }
                float dHi = domHi[a] - node->position[a];
                if (dHi < wallH) {
                    float q = std::min(std::max(dHi, 1e-4f) / wallH, 1.0f);
                    float oq = 1.0f - q;
                    fPress[a] -= mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
                    float approach = node->physics.velocity[a];
                    if (approach > 0.0f) fVisc[a] -= approach * mass_i * 50.0f * (1.0f - q);
                }
            }
        }

        node->physics.force += fPress + fVisc;
    }
    _tForces.stop();
    
    ScopedFunctionTimer _tIntegrate("substepPhysics.fluidIntegrate");
    const float sleepVel2 = (0.02f * h) * (0.02f * h);
    const float cfl = 0.4f * h;
    const float maxVel = std::max(cfl / dt, 8.0f);
    const float xsph = phys_xsphEpsilon;

    std::vector<Vec3> smoothedVel(sphNodes.size());
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)sphNodes.size(); ++i) {
        auto& node = sphNodes[i];
        Vec3 accum = Vec3::Zero();
        if (xsph > 0.0f) {
            const auto& nb = cells[partCell[i]].fluidNeighbors;
            for (int j : nb) {
                if (j == i) continue;
                auto& neighbor = sphNodes[j];
                float r = (node->position - neighbor->position).norm();
                if (r < h) {
                    float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol);
                    accum += V_j * K.Poly6(r) * (neighbor->physics.velocity - node->physics.velocity);
                }
            }
        }
        smoothedVel[i] = node->physics.velocity + xsph * accum;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)sphNodes.size(); ++i) {
        auto& node = sphNodes[i];

        float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
        float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

        Vec3 accel = node->physics.force / mass_i;
        if (!accel.allFinite()) accel = Vec3::Zero();
        if (accel.squaredNorm() > 1000.0f * 1000.0f) accel = accel.normalized() * 1000.0f;

        Vec3 vel = smoothedVel[i] + accel * dt;
        if (!vel.allFinite()) vel = Vec3::Zero();

        vel *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);
        if (vel.squaredNorm() > maxVel * maxVel) vel = vel.normalized() * maxVel;
        node->physics.velocity = vel;

        if (vel.squaredNorm() < sleepVel2) {
            node->physics.velocity.setZero();
            node->setStatic(true);
            node->setSettled(true);
            continue;
        }
        node->setStatic(false);
        node->setSettled(false);

        Vec3 np = node->position + vel * dt;

        {
            Vec3 diff = np - node->position;
            float dist = diff.norm();
            if (dist > 1e-5f) {
                Vec3 dir = diff / dist;
                RayHit_<T> hit;
                if (this->raycast(node->position, dir, dist + node->size * 0.5f, hit, node, true, true, &fastMats)) {
                    np = hit.hitPoint + hit.normal * (node->size * 0.51f);
                    float vn = node->physics.velocity.dot(hit.normal);
                    if (vn < 0.0f) {
                        node->physics.velocity -= vn * hit.normal;
                        node->physics.velocity *= 0.5f;
                    }
                }
            }
        }
        
        if (phys_solidBoundary) {
            float half = node->size * 0.5f;
            for (int a = 0; a < 3; ++a) {
                float lo = domLo[a] + half;
                float hi = domHi[a] - half;
                if (lo > hi) {
                    np[a] = (domLo[a] + domHi[a]) * 0.5f;
                    continue;
                }
                if (np[a] < lo) {
                    np[a] = lo;
                    if (node->physics.velocity[a] < 0.0f) node->physics.velocity[a] *= -0.3f;
                } else if (np[a] > hi) {
                    np[a] = hi;
                    if (node->physics.velocity[a] > 0.0f) node->physics.velocity[a] *= -0.3f;
                }
            }
        }

        if (np.allFinite()) node->position = np;
    }
    _tIntegrate.stop();
}

template<typename T>
void Octree<T>::endPhysicsFrame(PhysicsFrameContext& ctx) {
    TIME_FUNCTION;
    if (!ctx.valid) return;
    if (ctx.sphNodes.empty()) {
        if (pointPoolFragmentation() > 3.0f) store_.points.compact();
        return;
    }

    ScopedFunctionTimer _tRelocate("endPhysicsFrame.fluidRelocate");

    struct Relocation {
        std::shared_ptr<NodeData> node;
        Vec3 oldKeyPos;
        uint32_t start;
        int depth;
    };
    std::vector<Relocation> relocs;
    relocs.reserve(ctx.sphNodes.size());

    for (auto& node : ctx.sphNodes) {
        if (!node) continue;
        if (node->position == node->physics.lastTreePos) continue;
        std::vector<Vec3> span = { node->physics.lastTreePos, node->position };
        int depth = 0;
        uint32_t start = getHighestCommonNode(span, root_, depth);
        if (start == INVALID_IDX) start = root_;
        relocs.push_back({node, node->physics.lastTreePos, start, depth});
    }

    std::sort(relocs.begin(), relocs.end(),
              [](const Relocation& a, const Relocation& b) { return a.start < b.start; });

    size_t g = 0;
    while (g < relocs.size()) {
        size_t gEnd = g;
        uint32_t node = relocs[g].start;
        while (gEnd < relocs.size() && relocs[gEnd].start == node) ++gEnd;

        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = g; i < gEnd; ++i) {
            auto& rc = relocs[i];
            auto pd = rc.node;
            Vec3 target = pd->position;
            pd->position = rc.oldKeyPos;
            bool removed = removeRecursive(rc.start, pd->getCubeBounds(), pd);
            if (!removed) removeRecursive(root_, pd->getCubeBounds(), pd);
            pd->position = target;
            bool inserted = insertRecursive(rc.start, pd, rc.depth);
            if (!inserted) inserted = insertRecursive(root_, pd, 0);
            if (!inserted) {
                #pragma omp atomic
                size--;
            }
            pd->physics.lastTreePos = target;
        }
        g = gEnd;
    }
    _tRelocate.stop();

    if (pointPoolFragmentation() > 3.0f) store_.points.compact();
}

template<typename T>
void Octree<T>::multiStepPhysics(float dt, int steps) {
    TIME_FUNCTION;
    if (root_ == INVALID_IDX || dt <= 0.0f || steps < 1) return;

    PhysicsFrameContext ctx;
    beginPhysicsFrame(ctx);
    if (!ctx.valid) return;

    for (auto& node : ctx.sphNodes)
        if (node) node->physics.lastTreePos = node->position;

    float subDt = dt / steps;
    for (int s = 0; s < steps; ++s) substepPhysics(subDt, ctx);

    endPhysicsFrame(ctx);
}

template<typename T>
void Octree<T>::stepPhysics(float dt) {
    multiStepPhysics(dt, 1);
}

template<typename T>
void Octree<T>::stepRigidLattice(
        float dt, std::vector<std::shared_ptr<NodeData>>& rigidNodes,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) {
    if (rigidNodes.empty() || dt <= 0.0f) return;
    TIME_FUNCTION;

    ScopedFunctionTimer _tRForces("stepRigidLattice.forces");
    std::atomic<bool> anyBroke{false};

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        auto& node = rigidNodes[i];
        const PhysicsMaterial_* m = physMatOf(node, fastMats, fastMatsSize);
        if (!m) {
            node->physics.force.setZero();
            continue;
        }

        float mass = std::max(m->mass, 1e-4f);

        Vec3 g = phys_gravity;
        if (phys_useGravityPoint) {
            Vec3 toC = phys_gravityCenter - node->position;
            float d = toC.norm();
            g = (d > 1e-4f) ? (toC / d) * phys_gravityStrength : Vec3(0,0,0);
        }
        Vec3 force = g * mass;

        uint32_t selfId = node->id;
        for (uint32_t bid = node->physics.bondHead; bid != INVALID_IDX; ) {
            Bond_<T>& bond = store_.bonds.arena[bid];
            uint32_t nextBid = bond.nextFor(selfId);
            if (!bond.live || bond.broken) {
                bid = nextBid;
                continue;
            }

            auto other = store_.points.byIdLocked(bond.other(selfId));
            bool mayMutate = selfId < bond.other(selfId);
            if (!other || !other->isActive()) {
                if (mayMutate) bond.broken = true;
                bid = nextBid;
                continue;
            }

            Vec3 d = other->position - node->position;
            float len = d.norm();
            if (len < 1e-6f) {
                bid = nextBid;
                continue;
            }
            Vec3 dir = d / len;

            float k = (bond.stiffnessOverride > 0.0f) ? bond.stiffnessOverride : m->stiffness;

            float ext = len - bond.restLength;
            float springF = k * ext;

            Vec3 relVel = other->physics.velocity - node->physics.velocity;
            float dampF = m->damping * relVel.dot(dir) * k * 0.02f;

            float total = springF + dampF;
            force += dir * total;

            if (mayMutate) {
                float limit = (ext >= 0.0f) ? bond.strength : bond.strength * m->breakCompressionScale;
                float load = std::abs(springF);

                if (m->breakTorque > 0.0f) {
                    Vec3 lateral = relVel - dir * relVel.dot(dir);
                    float shear = k * lateral.norm() * 0.02f;
                    if (shear > m->breakTorque) {
                        bond.broken = true;
                        anyBroke.store(true, std::memory_order_relaxed);
                        bid = nextBid;
                        continue;
                    }
                }

                if (load > limit) {
                    bond.broken = true;
                    anyBroke.store(true, std::memory_order_relaxed);
                } else if (m->fatigue > 0.0f && load > limit * 0.5f) {
                    bond.damage = std::min(1.0f, bond.damage + m->fatigue * (load / limit - 0.5f) * dt);
                    bond.strength = m->breakForce * (1.0f - bond.damage);
                    if (bond.damage >= 1.0f) {
                        bond.broken = true;
                        anyBroke.store(true, std::memory_order_relaxed);
                    }
                }
            }
            bid = nextBid;
        }

        node->physics.force = force;
    }
    _tRForces.stop();

    ScopedFunctionTimer _tRIntegrate("stepRigidLattice.integrate");
    std::vector<FluidMoveAction<T>> moves(rigidNodes.size());
    std::vector<char> valid(rigidNodes.size(), 0);
    const float sleepV2 = 1e-5f;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        auto& node = rigidNodes[i];
        const PhysicsMaterial_* m = physMatOf(node, fastMats, fastMatsSize);
        if (!m) continue;
        float mass = std::max(m->mass, 1e-4f);

        Vec3 accel = node->physics.force / mass;
        if (!accel.allFinite()) accel = Vec3(0,0,0);
        float maxA = 2000.0f;
        if (accel.squaredNorm() > maxA*maxA) accel = accel.normalized() * maxA;

        node->physics.velocity += accel * dt;
        node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);
        if (!node->physics.velocity.allFinite()) node->physics.velocity.setZero();

        float maxV = std::max(node->size / dt, 20.0f);
        if (node->physics.velocity.squaredNorm() > maxV*maxV)
            node->physics.velocity = node->physics.velocity.normalized() * maxV;

        if (node->physics.velocity.squaredNorm() < sleepV2) {
            node->physics.velocity.setZero();
            node->setStatic(true);
            node->setSettled(true);
            continue;
        }
        node->setStatic(false);
        node->setSettled(false);
        moves[i].node = node;
        moves[i].oldPos = node->position;
        Vec3 np = node->position + node->physics.velocity * dt;
        if (!np.allFinite() || !node->position.allFinite()) {
            node->physics.velocity.setZero();
            if (!node->position.allFinite()) {
                valid[i] = 0;
                continue;
            }
            np = node->position;
        }
        moves[i].newPos = np;
        valid[i] = 1;
    }

    _tRIntegrate.stop();

    ScopedFunctionTimer _tRRelocate("stepRigidLattice.relocate");

    std::unordered_set<int> fracturedObjects;
    std::vector<uint32_t> brokenBonds;
    std::unordered_set<uint32_t> seenBonds;
    for (auto& node : rigidNodes) {
        uint32_t selfId = node->id;
        for (uint32_t bid = node->physics.bondHead; bid != INVALID_IDX; ) {
            Bond_<T>& bond = store_.bonds.arena[bid];
            uint32_t nextBid = bond.nextFor(selfId);
            if (bond.broken && bond.live && seenBonds.insert(bid).second) {
                fracturedObjects.insert(node->objectId);
                brokenBonds.push_back(bid);
            }
            bid = nextBid;
        }
    }
    for (uint32_t bid : brokenBonds) breakBond(bid);

    struct RRelocation {
        int idx;
        uint32_t start;
        int depth;
    };
    std::vector<RRelocation> rrelocs;
    rrelocs.reserve(rigidNodes.size());
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        if (!valid[i]) continue;
        auto& mv = moves[i];
        std::vector<Vec3> span = { mv.oldPos, mv.newPos };
        int depth = 0;
        uint32_t start = getHighestCommonNode(span, root_, depth);
        if (start == INVALID_IDX) start = root_;
        rrelocs.push_back({i, start, depth});
    }

    std::sort(rrelocs.begin(), rrelocs.end(),
              [](const RRelocation& a, const RRelocation& b) { return a.start < b.start; });

    size_t g = 0;
    while (g < rrelocs.size()) {
        size_t gEnd = g;
        uint32_t node = rrelocs[g].start;
        while (gEnd < rrelocs.size() && rrelocs[gEnd].start == node) ++gEnd;

        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t r = g; r < gEnd; ++r) {
            auto& rc = rrelocs[r];
            auto& mv = moves[rc.idx];
            auto pd = mv.node;
            if (!removeRecursive(rc.start, pd->getCubeBounds(), pd))
                removeRecursive(root_, pd->getCubeBounds(), pd);
            pd->position = mv.newPos;
            if (!insertRecursive(rc.start, pd, rc.depth))
                if (!insertRecursive(root_, pd, 0)) {
                    #pragma omp atomic
                    size--;
                }
        }
        g = gEnd;
    }
    _tRRelocate.stop();

    if (!fracturedObjects.empty()) {
        ScopedFunctionTimer _tFrac("stepRigidLattice.resolveFractures");
        for (int objId : fracturedObjects) resolveFracture(objId, fastMats, fastMatsSize);
    }
}

template<typename T>
const PhysicsMaterial_* Octree<T>::physMatOf(const std::shared_ptr<NodeData>& n,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) const {
    int oi = n->objectId + 1;
    if (oi < 0 || oi >= (int)fastMatsSize) return nullptr;
    if (n->physMatIdx >= fastMats[oi].size()) return nullptr;
    return &fastMats[oi][n->physMatIdx];
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
                if (slot == component.end() || slot->second != INVALID_IDX) { bid = nextBid; continue; }
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

    for (size_t i = 0; i < fragments.size(); ++i) {
        if (i == largest) continue;
        Fragment_<T>& frag = fragments[i];

        if (frag.nodes.size() < minFragment || policy == SplitPolicy::DISSOLVE) {
            std::unordered_set<std::shared_ptr<NodeData>> doomed(frag.nodes.begin(), frag.nodes.end());
            size -= removeSpecificNodesBatchRecursive(root_, doomed);
            continue;
        }

        if (policy == SplitPolicy::SHED_STATIC) {
            freezeFragment(frag.nodes);
            continue;
        }

        reassignFragment(frag.nodes, objectId);
    }

    physicsCollidersDirty_.store(true);
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
