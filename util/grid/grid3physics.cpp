#include "grid3eigen.hpp"

namespace Grid {

template<typename T>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T>> node;
    PointType oldPos;
    PointType newPos;
};

struct SolidNb { PointType pos; float size; };

struct Vec3i64Hash {
    std::size_t operator()(const std::array<int64_t, 3>& v) const {
        return (std::size_t)((v[0] * 73856093) ^ (v[1] * 19349663) ^ (v[2] * 83492791));
    }
};

template<typename T>
void Octree<T>::stepPhysics(float dt) {
    if (!root_ || dt <= 0.0f) return;

    int maxObjId = -1;
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        for (const auto& pair : objects_) {
            if (pair.first > maxObjId) maxObjId = pair.first;
        }
    }

    std::vector<std::vector<PhysicsMaterial_>> fastMats(maxObjId + 2);
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        for (const auto& pair : objects_) {
            std::shared_lock<std::shared_mutex> objLock(pair.second->objMutex);
            fastMats[pair.first + 1] = pair.second->physicsMaterials;
        }
    }
    size_t fastMatsSize = fastMats.size();

    std::vector<std::shared_ptr<NodeData>> sphNodes;
    std::vector<std::shared_ptr<NodeData>> rigidNodes;

    {
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

    stepRigidLattice(dt, rigidNodes, fastMats, fastMatsSize);

    std::vector<FluidMoveAction<T>> pendingFluidMoves;

    if (!sphNodes.empty()) {
        const SPHKernels& K = kernels_;
        const float h = K.h;
        const float maxKernelVol = 4.18879f * K.h3;
        const float C = h;
        const float invC = 1.0f / C;

        // Cell key for a position.
        auto keyOf = [invC](const PointType& p) -> std::array<int64_t,3> {
            return { (int64_t)std::floor(p.x() * invC),
                     (int64_t)std::floor(p.y() * invC),
                     (int64_t)std::floor(p.z() * invC) };
        };

        // (1) Bin fluids into a sparse cell map, then flatten to a compact cell
        //     list so we can iterate cells in parallel. Particles in the same
        //     cell share the same neighbor cells, so neighbor data is gathered
        //     once per cell rather than once per particle.
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

        // (2) Per-cell neighbor gathering (parallel). Fluid neighbors come from
        //     the 27 surrounding cells. Solid neighbors are read straight from
        //     the SVO: we find the lowest octree node containing the cell's
        //     search box and walk down from THERE (not the root), so the tree
        //     itself is the broad-phase — no duplicate collider structure, no
        //     extra memory. The walk only takes shared locks (read-only), so
        //     cells process concurrently.
        #pragma omp parallel for schedule(dynamic, 32)
        for (int c = 0; c < (int)cells.size(); ++c) {
            Cell& cell = cells[c];
            const auto& base = cell.key;

            for (int dx=-1; dx<=1; ++dx)
            for (int dy=-1; dy<=1; ++dy)
            for (int dz=-1; dz<=1; ++dz) {
                auto it = cellIndex.find({base[0]+dx, base[1]+dy, base[2]+dz});
                if (it != cellIndex.end()) {
                    const auto& m = cells[it->second].members;
                    cell.fluidNeighbors.insert(cell.fluidNeighbors.end(), m.begin(), m.end());
                }
            }

            // Search box: this cell expanded by one smoothing radius.
            PointType lo((base[0]    ) * C - h, (base[1]    ) * C - h, (base[2]    ) * C - h);
            PointType hi((base[0]+1.0f) * C + h, (base[1]+1.0f) * C + h, (base[2]+1.0f) * C + h);
            BoundingBox box{lo, hi};

            std::vector<PointType> corners = {lo, hi};
            int d = 0;
            OctreeNode* start = getHighestCommonNode(corners, root_.get(), d);
            if (!start) start = root_.get();

            std::vector<OctreeNode*> stack{start};
            while (!stack.empty()) {
                OctreeNode* cur = stack.back(); stack.pop_back();
                if (!cur || !boxIntersectsBox(cur->bounds(), box) || !cur->isLoaded()) continue;
                {
                    std::shared_lock<std::shared_mutex> lock(cur->nodeMutex);
                    for (const auto& pt : cur->points) {
                        if (!pt->isActive()) continue;
                        int oi = pt->objectId + 1;
                        if (oi < 0 || oi >= (int)fastMatsSize) continue;
                        if (fastMats[oi][pt->physMatIdx].type == BodyType::FLUID) continue;
                        const PointType& pp = pt->position;
                        if (pp.x() >= lo.x() && pp.x() <= hi.x() &&
                            pp.y() >= lo.y() && pp.y() <= hi.y() &&
                            pp.z() >= lo.z() && pp.z() <= hi.z()) {
                            cell.solids.push_back({pp, pt->size});
                        }
                    }
                    if (!cur->isLeaf())
                        for (int i=0;i<8;++i) if (cur->children[i]) stack.push_back(cur->children[i].get());
                }
            }
        }

        // (3) DENSITY + PRESSURE pass — parallel over particles.
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

        // (4) FORCES (parallel over particles).
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)sphNodes.size(); ++i) {
            auto& node = sphNodes[i];
            const Cell& cell = cells[partCell[i]];

            float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
            float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

            Eigen::Vector3f gravityDir = phys_gravity;
            if (phys_useGravityPoint) {
                Eigen::Vector3f toCenter = phys_gravityCenter - node->position;
                float dist = toCenter.norm();
                gravityDir = (dist > 1e-4f) ? (toCenter / dist) * phys_gravityStrength
                                            : Eigen::Vector3f(0,0,0);
            }
            node->physics.force = gravityDir * mass_i;

            Eigen::Vector3f fPress = Eigen::Vector3f::Zero();
            Eigen::Vector3f fVisc = Eigen::Vector3f::Zero();

            // Fluid-fluid.
            for (int j : cell.fluidNeighbors) {
                if (j == i) continue;
                auto& neighbor = sphNodes[j];
                PointType diff = node->position - neighbor->position;
                float r = diff.norm();
                if (r < 1e-5f) {
                    Eigen::Vector3f rnd = Eigen::Vector3f::Random();
                    if (rnd.squaredNorm() < 1e-8f) rnd = Eigen::Vector3f(1,0,0);
                    fPress += rnd.normalized() * mass_i * K.Poly6(0) * 10.0f;
                    continue;
                }
                if (r < h) {
                    PointType dir = diff / r;
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
                PointType diff = node->position - solid.pos;
                float r = diff.norm();
                float minDist = (node->size + solid.size) * 0.5f;
                if (r >= minDist) continue;
                if (r < 1e-5f) {
                    Eigen::Vector3f rnd = Eigen::Vector3f::Random();
                    if (rnd.squaredNorm() < 1e-8f) rnd = Eigen::Vector3f(1,0,0);
                    fPress += rnd.normalized() * mass_i * K.Poly6(0) * 10.0f;
                    continue;
                }
                PointType dir = diff / r;
                // Boundary repulsion scaled to the local cell spacing.
                float q = r / minDist;
                float oq = 1.0f - q;
                float repel = mass_i * baseG * 250.0f * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
                fPress += dir * repel;
                float approach = node->physics.velocity.dot(dir);
                if (approach < 0.0f) fVisc += -dir * (approach * mass_i * 50.0f * (1.0f - q));
            }

            node->physics.force += fPress + fVisc;
        }

        // (5) INTEGRATE (parallel). Sleeping: settled particles skip relocation.
        std::vector<FluidMoveAction<T>> perMoves(sphNodes.size());
        std::vector<char> moveValid(sphNodes.size(), 0);
        const float sleepVel2 = (0.02f * h) * (0.02f * h);
        const float maxVel = std::max(h / dt, 25.0f);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)sphNodes.size(); ++i) {
            auto& node = sphNodes[i];
            
            float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
            float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

            Eigen::Vector3f accel = node->physics.force / mass_i;
            if (!accel.allFinite()) accel = Eigen::Vector3f::Zero();
            if (accel.squaredNorm() > 1000.0f * 1000.0f) accel = accel.normalized() * 1000.0f;

            node->physics.velocity += accel * dt;
            if (!node->physics.velocity.allFinite()) node->physics.velocity = Eigen::Vector3f::Zero();
            
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
            moveValid[i] = 1;
        }

        for (int i = 0; i < (int)sphNodes.size(); ++i)
            if (moveValid[i]) pendingFluidMoves.push_back(std::move(perMoves[i]));
    }

    // (6) Collision resolve + relocation (serial: mutates the tree). Relocation
    //     reuses removeRecursive/insertRecursive but starts from the lowest
    //     common ancestor of (oldPos,newPos) rather than the root, keeping each
    //     edit shallow.
    for (size_t i = 0; i < pendingFluidMoves.size(); ++i) {
        auto& mv = pendingFluidMoves[i];
        PointType diff = mv.newPos - mv.oldPos;
        float dist = diff.norm();
        if (dist > 1e-5f) {
            RayHit_<T> hit;
            PointType dir = diff / dist;
            if (this->raycast(mv.oldPos, dir, dist + mv.node->size * 0.5f, hit, mv.node, true, true)) {
                mv.newPos = hit.hitPoint + hit.normal * (mv.node->size * 0.51f);
                float vn = mv.node->physics.velocity.dot(hit.normal);
                if (vn < 0.0f) {
                    mv.node->physics.velocity -= vn * hit.normal;
                    mv.node->physics.velocity *= 0.5f;
                }
            }
        }

        auto pd = mv.node;
        std::vector<PointType> span = { mv.oldPos, mv.newPos };
        int depth = 0;
        OctreeNode* start = getHighestCommonNode(span, root_.get(), depth);
        if (!start) start = root_.get();

        if (!removeRecursive(start, pd->getCubeBounds(), pd))
            removeRecursive(root_.get(), pd->getCubeBounds(), pd);
        pd->position = mv.newPos;
        ensureBounds(pd->getCubeBounds());
        if (!insertRecursive(start, pd, depth))
            if (!insertRecursive(root_.get(), pd, 0)) size--;
    }
}

template<typename T>
void Octree<T>::stepRigidLattice(
        float dt, std::vector<std::shared_ptr<NodeData>>& rigidNodes,
        const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize) {
    if (rigidNodes.empty() || dt <= 0.0f) return;

    auto matOf = [&](const std::shared_ptr<NodeData>& n) -> const PhysicsMaterial_* {
        int oi = n->objectId + 1;
        if (oi < 0 || oi >= (int)fastMatsSize) return nullptr;
        if (n->physMatIdx >= fastMats[oi].size()) return nullptr;
        return &fastMats[oi][n->physMatIdx];
    };

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        auto& node = rigidNodes[i];
        const PhysicsMaterial_* m = matOf(node);
        if (!m) { node->physics.force.setZero(); continue; }

        float mass = std::max(m->mass, 1e-4f);

        Eigen::Vector3f g = phys_gravity;
        if (phys_useGravityPoint) {
            Eigen::Vector3f toC = phys_gravityCenter - node->position;
            float d = toC.norm();
            g = (d > 1e-4f) ? (toC / d) * phys_gravityStrength : Eigen::Vector3f(0,0,0);
        }
        Eigen::Vector3f force = g * mass;

        for (auto& bond : node->physics.bonds) {
            auto other = bond.other.lock();
            if (!other || !other->isActive()) { bond.strength = -1.0f; continue; }

            Eigen::Vector3f d = other->position - node->position;
            float len = d.norm();
            if (len < 1e-6f) continue;
            Eigen::Vector3f dir = d / len;

            float ext = len - bond.restLength;
            float springF = m->stiffness * ext;

            Eigen::Vector3f relVel = other->physics.velocity - node->physics.velocity;
            float dampF = m->damping * relVel.dot(dir) * m->stiffness * 0.02f;

            float total = springF + dampF;
            force += dir * total;

            if (std::abs(springF) > bond.strength) bond.strength = -1.0f;
        }

        node->physics.force = force;
    }

    std::vector<FluidMoveAction<T>> moves(rigidNodes.size());
    std::vector<char> valid(rigidNodes.size(), 0);
    const float sleepV2 = 1e-5f;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)rigidNodes.size(); ++i) {
        auto& node = rigidNodes[i];
        const PhysicsMaterial_* m = matOf(node);
        if (!m) continue;
        float mass = std::max(m->mass, 1e-4f);

        Eigen::Vector3f accel = node->physics.force / mass;
        if (!accel.allFinite()) accel = Eigen::Vector3f(0,0,0);
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
        std::vector<PointType> span = { mv.oldPos, mv.newPos };
        int depth = 0;
        OctreeNode* start = getHighestCommonNode(span, root_.get(), depth);
        if (!start) start = root_.get();
        if (!removeRecursive(start, pd->getCubeBounds(), pd))
            removeRecursive(root_.get(), pd->getCubeBounds(), pd);
        pd->position = mv.newPos;
        ensureBounds(pd->getCubeBounds());
        if (!insertRecursive(start, pd, depth))
            if (!insertRecursive(root_.get(), pd, 0)) size--;
    }
}

}