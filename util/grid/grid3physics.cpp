#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    PointType oldPos;
    PointType newPos;
};

struct SolidNb { PointType pos; float size; };

struct Vec3i64Hash {
    std::size_t operator()(const std::array<int64_t, 3>& v) const {
        return (std::size_t)((v[0] * 73856093) ^ (v[1] * 19349663) ^ (v[2] * 83492791));
    }
};

template<typename T, typename IndexType>
void Octree<T, IndexType>::stepPhysics(float dt) {
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
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<FluidMoveAction<T, IndexType>> pendingFluidMoves;

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
                if (!cur || !boxIntersectsBox(cur->bounds, box) || !cur->isLoaded()) continue;
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
        std::vector<FluidMoveAction<T, IndexType>> perMoves(sphNodes.size());
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
            RayHit_<T, IndexType> hit;
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

template<typename T, typename IndexType>
void Octree<T, IndexType>::stepGasFields(float dt) {
    if (!root_ || dt <= 0.0f) return;

    std::vector<OctreeNode*> nodes;
    {
        std::lock_guard<std::mutex> lock(gasMutex_);
        size_t w = 0;
        for (size_t i = 0; i < activeGasNodes_.size(); ++i) {
            OctreeNode* n = activeGasNodes_[i];
            if (!n) continue;
            std::shared_lock<std::shared_mutex> nl(n->nodeMutex);
            if (n->gasField && !n->gasField->isEmpty()) {
                activeGasNodes_[w++] = n;
            }
        }
        activeGasNodes_.resize(w);
        nodes = activeGasNodes_;
    }
    if (nodes.empty()) return;

    Eigen::Vector3f gravityDir = phys_gravity;
    if (phys_useGravityPoint) gravityDir = Eigen::Vector3f(0.0f, -1.0f, 0.0f) * phys_gravityStrength;
    Eigen::Vector3f up = gravityDir.squaredNorm() > 1e-8f ? (-gravityDir.normalized()) : Eigen::Vector3f(0.0f, 1.0f, 0.0f);

    float diffuse = phys_gasDiffusion;
    float dissip = std::clamp(phys_gasDissipation * dt, 0.0f, 1.0f);

    struct Outflow { PointType pos; uint16_t globalSpecies; float amount; T data; };
    std::vector<Outflow> outflows;
    std::mutex outflowMutex;

    for (OctreeNode* node : nodes) {
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        auto* field = node->gasField.get();
        if (!field) continue;

        const int R = field->res;
        if (R <= 0) continue;
        const size_t N = field->cells.size();
        if (phys_gasPressure > 0.0f) {
            std::vector<float> pressure(N, 0.0f);
            for (size_t i = 0; i < N; ++i) {
                float over = field->cells[i].totalDensity() - phys_gasRestDensity;
                pressure[i] = phys_gasPressure * std::max(0.0f, over);
            }

            const float invDx = (field->cellSize.x() > 1e-8f) ? 1.0f / field->cellSize.x() : 0.0f;
            const float invDy = (field->cellSize.y() > 1e-8f) ? 1.0f / field->cellSize.y() : 0.0f;
            const float invDz = (field->cellSize.z() > 1e-8f) ? 1.0f / field->cellSize.z() : 0.0f;

            for (int z = 0; z < R; ++z)
            for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                size_t idx = field->index(x, y, z);
                float dens = field->cells[idx].totalDensity();
                if (dens <= 1e-6f) continue;

                int xm = x > 0     ? x - 1 : x;
                int xp = x < R - 1 ? x + 1 : x;
                int ym = y > 0     ? y - 1 : y;
                int yp = y < R - 1 ? y + 1 : y;
                int zm = z > 0     ? z - 1 : z;
                int zp = z < R - 1 ? z + 1 : z;

                float gx = (xp != xm) ? (pressure[field->index(xp, y, z)] - pressure[field->index(xm, y, z)])
                                        / float(xp - xm) * invDx : 0.0f;
                float gy = (yp != ym) ? (pressure[field->index(x, yp, z)] - pressure[field->index(x, ym, z)])
                                        / float(yp - ym) * invDy : 0.0f;
                float gz = (zp != zm) ? (pressure[field->index(x, y, zp)] - pressure[field->index(x, y, zm)])
                                        / float(zp - zm) * invDz : 0.0f;

                Eigen::Vector3f gradP(gx, gy, gz);
                field->cells[idx].velocity += -gradP * (dt / dens);
            }
        }

        for (auto& c : field->cells) {
            float dens = c.totalDensity();
            if (dens <= 1e-6f) continue;
            c.velocity += up * (phys_gasBuoyancy * dens * dt);
            c.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);
        }

        std::vector<GasCell_<T>> next(N);
        for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
        for (int x = 0; x < R; ++x) {
            size_t idx = field->index(x, y, z);
            GasCell_<T>& dst = next[idx];
            const GasCell_<T>& cur = field->cells[idx];

            PointType centre = field->cellCenter(x, y, z);
            PointType src = centre - cur.velocity * dt;

            int sx, sy, sz;
            bool inside = field->worldToCell(src, sx, sy, sz);
            if (!inside) {
                sx = std::clamp(sx, 0, R - 1);
                sy = std::clamp(sy, 0, R - 1);
                sz = std::clamp(sz, 0, R - 1);
            }
            {
                const GasCell_<T>& s = field->cells[field->index(sx, sy, sz)];
                dst.amount = s.amount;
                dst.velocity = s.velocity;
                dst.data = s.data;
            }

            if (cur.totalDensity() > 1e-5f) {
                PointType outPos = centre + cur.velocity * dt;
                int ox, oy, oz;
                if (!field->worldToCell(outPos, ox, oy, oz)) {
                    for (uint8_t sp = 0; sp < field->slotCount; ++sp) {
                        float a = cur.amount[sp];
                        if (a <= 1e-6f) continue;
                        uint16_t g = field->slotToGlobal[sp];
                        if (g == GasField_<T, IndexType>::INVALID_SLOT) continue;
                        std::lock_guard<std::mutex> ol(outflowMutex);
                        outflows.push_back({outPos, g, a * 0.5f, cur.data});
                    }
                    for (uint8_t sp = 0; sp < MAX_GAS_SPECIES; ++sp) dst.amount[sp] *= 0.5f;
                }
            }
        }

        if (diffuse > 0.0f) {
            const float keep = 1.0f - diffuse;
            const float share = diffuse / 6.0f;
            const int off[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

            std::vector<GasCell_<T>> diff(N);
            for (int z = 0; z < R; ++z)
            for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                size_t idx = field->index(x, y, z);
                diff[idx].velocity = next[idx].velocity;
                diff[idx].data = next[idx].data;
                for (uint8_t sp = 0; sp < MAX_GAS_SPECIES; ++sp)
                    diff[idx].amount[sp] = next[idx].amount[sp] * keep;
            }
            for (int z = 0; z < R; ++z)
            for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                size_t idx = field->index(x, y, z);
                const auto& c = next[idx];
                if (c.totalDensity() <= 1e-6f) continue;
                for (auto& o : off) {
                    int nx = x+o[0], ny = y+o[1], nz = z+o[2];
                    if (!field->inRange(nx, ny, nz)) {
                        for (uint8_t sp = 0; sp < MAX_GAS_SPECIES; ++sp)
                            diff[idx].amount[sp] += c.amount[sp] * share;
                        continue;
                    }
                    auto& nd = diff[field->index(nx, ny, nz)];
                    for (uint8_t sp = 0; sp < MAX_GAS_SPECIES; ++sp)
                        nd.amount[sp] += c.amount[sp] * share;
                }
            }
            next.swap(diff);
        }

        if (dissip > 0.0f) {
            float scale = 1.0f - dissip;
            for (auto& c : next) {
                for (uint8_t sp = 0; sp < MAX_GAS_SPECIES; ++sp) c.amount[sp] *= scale;
            }
        }

        field->cells.swap(next);
        node->setDirty(true);
    }

    for (auto& o : outflows) {
        addGas(o.pos, o.globalSpecies, o.amount, o.data);
    }
}

}