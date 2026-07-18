#include "grid3eigen.hpp"

namespace Grid {

template<typename T>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T>> node;
    Vec3 oldPos;
    Vec3 newPos;
    bool solidNearby = true;
};

struct SolidNb { Vec3 pos; float size; };


template<typename T>
void Octree<T>::stepPhysics(float dt) {
    TIME_FUNCTION;
    if (root_ == INVALID_IDX || dt <= 0.0f) return;
    
    const BoundingBox domainBounds = nodeAt(root_)->bounds();
    const Vec3 domLo = domainBounds.first;
    const Vec3 domHi = domainBounds.second;

    ScopedFunctionTimer _tSetup("stepPhysics.setupMaterials");
    int maxObjId = -1;
    {
        s_lock lock(objectsMutex_);
        for (const auto& pair : objects_) {
            if (pair.first > maxObjId) maxObjId = pair.first;
        }
    }

    std::vector<std::vector<PhysicsMaterial_>> fastMats(maxObjId + 2);
    {
        s_lock lock(objectsMutex_);
        for (const auto& pair : objects_) {
            s_lock objLock(pair.second->objMutex);
            fastMats[pair.first + 1] = pair.second->physicsMaterials;
        }
    }
    size_t fastMatsSize = fastMats.size();
    _tSetup.stop();

    std::vector<std::shared_ptr<NodeData>> sphNodes;
    std::vector<std::shared_ptr<NodeData>> rigidNodes;

    {
        ScopedFunctionTimer _tClassify("stepPhysics.classifyActive");
        std::lock_guard<std::mutex> lock(physicsMutex_);
        
        size_t writeIdx = 0;
        for (size_t i = 0; i < activePhysicsNodes_.size(); ++i) {
            if (!activePhysicsNodes_[i].expired()) {
                activePhysicsNodes_[writeIdx++] = activePhysicsNodes_[i];
            }
        }
        activePhysicsNodes_.resize(writeIdx);
        
        sphNodes.reserve(writeIdx);

        for (size_t i = 0; i < writeIdx; ++i) {
            if (auto sp = activePhysicsNodes_[i].lock()) {
                if (sp->isActive()) {
                    int objIdx = sp->objectId + 1;
                    if (objIdx >= 0 && objIdx < fastMatsSize) {
                        const auto& mats = fastMats[objIdx];
                        if (sp->physMatIdx < mats.size()) {
                            BodyType bType = mats[sp->physMatIdx].type;
                            if (bType == BodyType::FLUID) {
                                sphNodes.push_back(sp);
                            } else if (bType == BodyType::RIGID) {
                                rigidNodes.push_back(sp);
                            }
                        }
                    }
                }
            }
        }
    }

    {
        ScopedFunctionTimer _tRigid("stepPhysics.rigidLattice");
        stepRigidLattice(dt, rigidNodes, fastMats, fastMatsSize);
    }

    std::vector<FluidMoveAction<T>> pendingFluidMoves;

    if (!sphNodes.empty()) {
        const SPHKernels& K = kernels_;
        const float h = K.h;
        const float maxKernelVol = 4.18879f * K.h3;
        const float C = h;
        const float invC = 1.0f / C;

        // Cell key for a position.
        auto keyOf = [invC](const Vec3& p) -> std::array<int64_t,3> {
            return { (int64_t)std::floor(p.x() * invC),
                     (int64_t)std::floor(p.y() * invC),
                     (int64_t)std::floor(p.z() * invC) };
        };

        // (1) Bin fluids into a sparse cell map, then flatten to a compact cell
        //     list so we can iterate cells in parallel. Particles in the same
        //     cell share the same neighbor cells, so neighbor data is gathered
        //     once per cell rather than once per particle.
        ScopedFunctionTimer _tBin("stepPhysics.fluidBin");
        std::unordered_map<std::array<int64_t,3>, int, Vec3i64Hash> cellIndex;
        cellIndex.reserve(sphNodes.size());
        struct Cell {
            std::array<int64_t,3> key;
            std::vector<int> members;        // fluid indices in this cell
            std::vector<int> fluidNeighbors; // fluid indices in the 27-cell block
            std::vector<SolidNb> solids;     // nearby static voxels
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

        // (2) Per-cell neighbor gathering (parallel). Fluid neighbors come from
        //     the 27 surrounding cells. Solid neighbors are read straight from
        //     the SVO: we find the lowest octree node containing the cell's
        //     search box and walk down from THERE (not the root), so the tree
        //     itself is the broad-phase — no duplicate collider structure, no
        //     extra memory. The walk only takes shared locks (read-only), so
        //     cells process concurrently.
        ScopedFunctionTimer _tGather("stepPhysics.fluidNeighborGather");

        std::array<int64_t,3> keyMin = cells[0].key, keyMax = cells[0].key;
        for (const auto& cl : cells)
            for (int a = 0; a < 3; ++a) {
                keyMin[a] = std::min(keyMin[a], cl.key[a]);
                keyMax[a] = std::max(keyMax[a], cl.key[a]);
            }

        // Fluid region expanded by one cell (== one smoothing radius, C == h).
        Vec3 regionLo((keyMin[0]-1)*C, (keyMin[1]-1)*C, (keyMin[2]-1)*C);
        Vec3 regionHi((keyMax[0]+2)*C, (keyMax[1]+2)*C, (keyMax[2]+2)*C);
        BoundingBox region{regionLo, regionHi};

        std::unordered_map<std::array<int64_t,3>, std::vector<SolidNb>, Vec3i64Hash> solidCells;

        std::vector<Vec3> regionCorners = {regionLo, regionHi};
        int rd = 0;
        uint32_t solidStart = getHighestCommonNode(regionCorners, root_, rd);
        if (solidStart == INVALID_IDX) solidStart = root_;

        std::vector<uint32_t> stack{solidStart};
        while (!stack.empty()) {
            uint32_t curIdx = stack.back(); stack.pop_back();
            const OctreeNode* cur = nodeAt(curIdx);
            if (!cur || !boxIntersectsBox(cur->bounds(), region) || !cur->isLoaded()) continue;
            for (const auto& pt : pointsView(curIdx)) {
                if (!pt || !pt->isActive()) continue;
                int oi = pt->objectId + 1;
                if (pt->physMatIdx >= fastMats[oi].size()) continue;
                if (oi < 0 || oi >= (int)fastMatsSize) continue;
                if (fastMats[oi][pt->physMatIdx].type == BodyType::FLUID) continue;
                const Vec3& pp = pt->position;
                if (pp.x() < regionLo.x() || pp.x() > regionHi.x() ||
                    pp.y() < regionLo.y() || pp.y() > regionHi.y() ||
                    pp.z() < regionLo.z() || pp.z() > regionHi.z()) continue;
                solidCells[keyOf(pp)].push_back({pp, pt->size});
            }
            if (!cur->isLeaf())
                for (int i=0;i<8;++i) if (cur->hasChild(i)) stack.push_back(cur->firstChild + i);
        }

        #pragma omp parallel for schedule(dynamic, 32)
        for (int c = 0; c < (int)cells.size(); ++c) {
            Cell& cell = cells[c];
            const auto& base = cell.key;

            for (int dx=-1; dx<=1; ++dx)
            for (int dy=-1; dy<=1; ++dy)
            for (int dz=-1; dz<=1; ++dz) {
                std::array<int64_t,3> nk{base[0]+dx, base[1]+dy, base[2]+dz};
                auto fit = cellIndex.find(nk);
                if (fit != cellIndex.end()) {
                    const auto& m = cells[fit->second].members;
                    cell.fluidNeighbors.insert(cell.fluidNeighbors.end(), m.begin(), m.end());
                }
                auto sit = solidCells.find(nk);
                if (sit != solidCells.end()) {
                    const auto& s = sit->second;
                    cell.solids.insert(cell.solids.end(), s.begin(), s.end());
                }
            }
        }

        _tGather.stop();

        // (3) DENSITY + PRESSURE pass — parallel over particles.
        ScopedFunctionTimer _tDensity("stepPhysics.fluidDensity");
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

        // (4) FORCES (parallel over particles).
        ScopedFunctionTimer _tForces("stepPhysics.fluidForces");
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
                gravityDir = (dist > 1e-4f) ? (toCenter / dist) * phys_gravityStrength
                                            : Vec3(0,0,0);
            }
            node->physics.force = gravityDir * mass_i;

            Vec3 fPress = Vec3::Zero();
            Vec3 fVisc = Vec3::Zero();

            // Fluid-fluid.
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

            // Fluid-solid (static voxels from the SVO walk; velocity == 0).
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
                // Boundary repulsion scaled to the local cell spacing.
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
                    float dLo = node->position[a] - domLo[a]; // distance to low face
                    if (dLo < wallH) {
                        float q = std::min(std::max(dLo, 1e-4f) / wallH, 1.0f);
                        float oq = 1.0f - q;
                        fPress[a] += mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
                        float approach = -node->physics.velocity[a]; // >0 means moving into wall
                        if (approach > 0.0f) fVisc[a] += approach * mass_i * 50.0f * (1.0f - q);
                    }
                    float dHi = domHi[a] - node->position[a]; // distance to high face
                    if (dHi < wallH) {
                        float q = std::min(std::max(dHi, 1e-4f) / wallH, 1.0f);
                        float oq = 1.0f - q;
                        fPress[a] -= mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
                        float approach = node->physics.velocity[a]; // >0 means moving into wall
                        if (approach > 0.0f) fVisc[a] -= approach * mass_i * 50.0f * (1.0f - q);
                    }
                }
            }

            node->physics.force += fPress + fVisc;
        }

        _tForces.stop();

        // (5) INTEGRATE (parallel). Sleeping: settled particles skip relocation.
        ScopedFunctionTimer _tIntegrate("stepPhysics.fluidIntegrate");
        std::vector<FluidMoveAction<T>> perMoves(sphNodes.size());
        std::vector<char> moveValid(sphNodes.size(), 0);
        const float sleepVel2 = (0.02f * h) * (0.02f * h);
        const float maxVel = std::max(h / dt, 25.0f);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)sphNodes.size(); ++i) {
            auto& node = sphNodes[i];
            
            float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
            float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

            Vec3 accel = node->physics.force / mass_i;
            if (!accel.allFinite()) accel = Vec3::Zero();
            if (accel.squaredNorm() > 1000.0f * 1000.0f) accel = accel.normalized() * 1000.0f;

            node->physics.velocity += accel * dt;
            if (!node->physics.velocity.allFinite()) node->physics.velocity = Vec3::Zero();
            
            node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);
            if (node->physics.velocity.squaredNorm() > maxVel * maxVel)
                node->physics.velocity = node->physics.velocity.normalized() * maxVel;

            if (node->physics.velocity.squaredNorm() < sleepVel2) {
                node->physics.velocity.setZero();
                continue;
            }

            perMoves[i].node = node;
            perMoves[i].oldPos = node->position;
            perMoves[i].newPos = node->position + node->physics.velocity * dt;
            perMoves[i].solidNearby = !cells[partCell[i]].solids.empty();
            moveValid[i] = 1;
        }

        for (int i = 0; i < (int)sphNodes.size(); ++i)
            if (moveValid[i]) pendingFluidMoves.push_back(std::move(perMoves[i]));
        _tIntegrate.stop();
    }

    ScopedFunctionTimer _tRelocate("stepPhysics.fluidRelocate_SERIAL");

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < pendingFluidMoves.size(); ++i) {
        auto& mv = pendingFluidMoves[i];
        Vec3 diff = mv.newPos - mv.oldPos;
        float dist = diff.norm();
        if (dist > 1e-5f && mv.solidNearby) {
            Vec3 dir = diff / dist;
            RayHit_<T> hit;               // thread-local
            // Pass fastMats so the solid-only filter is lock-free (no objectsMutex_
            // / material-copy per point) — this is the dominant relocate cost.
            if (this->raycast(mv.oldPos, dir, dist + mv.node->size * 0.5f, hit, mv.node, true, true, &fastMats)) {
                mv.newPos = hit.hitPoint + hit.normal * (mv.node->size * 0.51f);
                float vn = mv.node->physics.velocity.dot(hit.normal);
                if (vn < 0.0f) {
                    mv.node->physics.velocity -= vn * hit.normal;
                    mv.node->physics.velocity *= 0.5f;
                }
            }
        }
        
        if (phys_solidBoundary) {
            float half = mv.node->size * 0.5f;
            for (int a = 0; a < 3; ++a) {
                float lo = domLo[a] + half, hi = domHi[a] - half;
                if (lo > hi) { mv.newPos[a] = (domLo[a] + domHi[a]) * 0.5f; continue; }
                if (mv.newPos[a] < lo) {
                    mv.newPos[a] = lo;
                    if (mv.node->physics.velocity[a] < 0.0f)
                        mv.node->physics.velocity[a] *= -0.3f;
                } else if (mv.newPos[a] > hi) {
                    mv.newPos[a] = hi;
                    if (mv.node->physics.velocity[a] > 0.0f)
                        mv.node->physics.velocity[a] *= -0.3f;
                }
            }
        }
    }

    // Phase B: serial tree edits.
    std::vector<Vec3> span(2);   // reused to avoid a heap alloc per particle
    for (size_t i = 0; i < pendingFluidMoves.size(); ++i) {
        auto& mv = pendingFluidMoves[i];
        auto pd = mv.node;
        span[0] = mv.oldPos;
        span[1] = mv.newPos;
        int depth = 0;
        uint32_t start = getHighestCommonNode(span, root_, depth);
        if (start == INVALID_IDX) start = root_;

        if (!removeRecursive(start, pd->getCubeBounds(), pd))
            removeRecursive(root_, pd->getCubeBounds(), pd);
        pd->position = mv.newPos;
        if (!insertRecursive(start, pd, depth))
            if (!insertRecursive(root_, pd, 0)) size--;
    }
}

template<typename T>
void Octree<T>::stepRigidLattice(
        float dt, std::vector<std::shared_ptr<NodeData>>& rigidNodes,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) {
    if (rigidNodes.empty() || dt <= 0.0f) return;
    TIME_FUNCTION;

    auto matOf = [&](const std::shared_ptr<NodeData>& n) -> const PhysicsMaterial_* {
        int oi = n->objectId + 1;
        if (oi < 0 || oi >= (int)fastMatsSize) return nullptr;
        if (n->physMatIdx >= fastMats[oi].size()) return nullptr;
        return &fastMats[oi][n->physMatIdx];
    };

    ScopedFunctionTimer _tRForces("stepRigidLattice.forces");
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        auto& node = rigidNodes[i];
        const PhysicsMaterial_* m = matOf(node);
        if (!m) { node->physics.force.setZero(); continue; }

        float mass = std::max(m->mass, 1e-4f);

        Vec3 g = phys_gravity;
        if (phys_useGravityPoint) {
            Vec3 toC = phys_gravityCenter - node->position;
            float d = toC.norm();
            g = (d > 1e-4f) ? (toC / d) * phys_gravityStrength : Vec3(0,0,0);
        }
        Vec3 force = g * mass;

        for (auto& bond : node->physics.bonds) {
            auto other = bond.other.lock();
            if (!other || !other->isActive()) { bond.strength = -1.0f; continue; }

            Vec3 d = other->position - node->position;
            float len = d.norm();
            if (len < 1e-6f) continue;
            Vec3 dir = d / len;

            float ext = len - bond.restLength;
            float springF = m->stiffness * ext;

            Vec3 relVel = other->physics.velocity - node->physics.velocity;
            float dampF = m->damping * relVel.dot(dir) * m->stiffness * 0.02f;

            float total = springF + dampF;
            force += dir * total;

            if (std::abs(springF) > bond.strength) bond.strength = -1.0f;
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
        const PhysicsMaterial_* m = matOf(node);
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
            continue;
        }
        moves[i].node = node;
        moves[i].oldPos = node->position;
        moves[i].newPos = node->position + node->physics.velocity * dt;
        valid[i] = 1;
    }

    _tRIntegrate.stop();

    ScopedFunctionTimer _tRRelocate("stepRigidLattice.relocate_SERIAL");
    for (auto& node : rigidNodes) {
        auto& b = node->physics.bonds;
        b.erase(std::remove_if(b.begin(), b.end(),
                   [](const Bond_<T>& x){ return x.strength < 0.0f; }),
                b.end());
    }

    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        if (!valid[i]) continue;
        auto& mv = moves[i];
        auto pd = mv.node;
        std::vector<Vec3> span = { mv.oldPos, mv.newPos };
        int depth = 0;
        uint32_t start = getHighestCommonNode(span, root_, depth);
        if (start == INVALID_IDX) start = root_;
        if (!removeRecursive(start, pd->getCubeBounds(), pd))
            removeRecursive(root_, pd->getCubeBounds(), pd);
        pd->position = mv.newPos;
        if (!insertRecursive(start, pd, depth))
            if (!insertRecursive(root_, pd, 0)) size--;
    }
}

}