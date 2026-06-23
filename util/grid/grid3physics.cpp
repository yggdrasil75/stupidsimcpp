#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    PointType oldPos;
    PointType newPos;
};

struct Vec3iHash {
    std::size_t operator()(const std::array<int64_t, 3>& v) const {
        return (v[0] * 73856093) ^ (v[1] * 19349663) ^ (v[2] * 83492791);
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
        float h = phys_smoothingRadius;
        float h2 = h * h;
        float maxKernelVol = 4.18879f * h2 * h;
        
        float C = phys_smoothingRadius;
        std::unordered_map<std::array<int64_t, 3>, std::vector<std::shared_ptr<NodeData>>, Vec3iHash> grid;
        
        for (auto& node : sphNodes) {
            std::array<int64_t, 3> key = {
                static_cast<int64_t>(std::floor(node->position.x() / C)),
                static_cast<int64_t>(std::floor(node->position.y() / C)),
                static_cast<int64_t>(std::floor(node->position.z() / C))
            };
            grid[key].push_back(node);
        }

        for (auto& cell : grid) {
            std::array<int64_t, 3> cellKey = cell.first;
            auto& cellNodes = cell.second;

            std::vector<std::shared_ptr<NodeData>> dynamicNeighbors;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        std::array<int64_t, 3> nKey = {cellKey[0]+dx, cellKey[1]+dy, cellKey[2]+dz};
                        auto it = grid.find(nKey);
                        if (it != grid.end()) {
                            dynamicNeighbors.insert(dynamicNeighbors.end(), it->second.begin(), it->second.end());
                        }
                    }
                }
            }

            for (auto& node : cellNodes) {
                int objIdx = node->objectId + 1;
                PhysicsMaterial_ myPmat = fastMats[objIdx][node->physMatIdx];
                
                float densityFraction = 0.0f;

                for (auto& neighbor : dynamicNeighbors) {
                    float r = (node->position - neighbor->position).norm();
                    float H_ij = std::max(h, (node->size + neighbor->size) * 0.75f);
                    
                    if (r < H_ij) {
                        float S = h / H_ij;
                        float r_scaled = r * S;
                        float W = kernels_.Poly6(r_scaled) * (S * S * S);
                        
                        float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol); 
                        densityFraction += V_j * W;
                    }
                }

                node->physics.density = densityFraction * phys_restDensity; 
                
                if (myPmat.type == BodyType::FLUID) {
                    float over_density = std::max(0.0f, densityFraction - 1.0f);
                    node->physics.pressure = phys_gasConstant * std::min(over_density, 1.5f);
                }
            }
        }

        for (auto& cell : grid) {
            std::array<int64_t, 3> cellKey = cell.first;
            auto& cellNodes = cell.second;

            std::vector<std::shared_ptr<NodeData>> dynamicNeighbors;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        std::array<int64_t, 3> nKey = {cellKey[0]+dx, cellKey[1]+dy, cellKey[2]+dz};
                        auto it = grid.find(nKey);
                        if (it != grid.end()) {
                            dynamicNeighbors.insert(dynamicNeighbors.end(), it->second.begin(), it->second.end());
                        }
                    }
                }
            }

            BoundingBox queryBounds;
            queryBounds.first = PointType((cellKey[0]-1)*C, (cellKey[1]-1)*C, (cellKey[2]-1)*C);
            queryBounds.second = PointType((cellKey[0]+2)*C, (cellKey[1]+2)*C, (cellKey[2]+2)*C);
            
            std::vector<std::shared_ptr<NodeData>> solidNeighbors;
            
            std::vector<OctreeNode*> stack;
            stack.push_back(root_.get());
            while(!stack.empty()) {
                OctreeNode* curr = stack.back();
                stack.pop_back();
                
                if (!curr || !this->boxIntersectsBox(curr->bounds, queryBounds)) continue;
                if (!curr->isLoaded()) continue; 
                
                std::shared_lock<std::shared_mutex> lock(curr->nodeMutex);
                for (const auto& pt : curr->points) {
                    if (!pt->isActive()) continue;

                    int objIdx = pt->objectId + 1;
                    if (objIdx >= 0 && objIdx < fastMatsSize) {
                        BodyType bType = fastMats[objIdx][pt->physMatIdx].type;
                        if (bType == BodyType::FLUID) continue;
                    } else {
                        continue;
                    }

                    if (pt->position.x() >= queryBounds.first.x() && pt->position.x() <= queryBounds.second.x() &&
                        pt->position.y() >= queryBounds.first.y() && pt->position.y() <= queryBounds.second.y() &&
                        pt->position.z() >= queryBounds.first.z() && pt->position.z() <= queryBounds.second.z()) {
                        solidNeighbors.push_back(pt);
                    }
                }
                if (!curr->isLeaf()) {
                    for (int i=0; i<8; ++i) {
                        if (curr->children[i]) stack.push_back(curr->children[i].get());
                    }
                }
            }

            for (auto& node : cellNodes) {
                int objIdx = node->objectId + 1;
                PhysicsMaterial_ myPmat = fastMats[objIdx][node->physMatIdx];
                
                float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
                float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

                Eigen::Vector3f gravityDir = phys_gravity;
                if (phys_useGravityPoint) {
                    Eigen::Vector3f toCenter = phys_gravityCenter - node->position;
                    float dist = toCenter.norm();
                    if (dist > 1e-4f) gravityDir = (toCenter / dist) * phys_gravityStrength;
                    else gravityDir = Eigen::Vector3f::Zero();
                }

                node->physics.force = gravityDir * mass_i;

                Eigen::Vector3f fPress = Eigen::Vector3f::Zero();
                Eigen::Vector3f fVisc = Eigen::Vector3f::Zero();

                for (auto& neighbor : dynamicNeighbors) {
                    if (neighbor == node) continue;

                    PointType diff = node->position - neighbor->position;
                    float r = diff.norm();
                    float H_ij = std::max(h, (node->size + neighbor->size) * 0.75f);
                    
                    if (r < 1e-5f) {
                        Eigen::Vector3f rnd = Eigen::Vector3f::Random();
                        if (rnd.squaredNorm() < 1e-8f) rnd = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                        fPress += rnd.normalized() * mass_i * kernels_.Poly6(0) * 10.0f;
                        continue;
                    }
                    
                    if (r < H_ij) {
                        float S = h / H_ij;
                        float r_scaled = r * S;
                        PointType dir = diff / r; 
                        
                        float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol);
                        float P_sum = node->physics.pressure + neighbor->physics.pressure;
                        
                        float wendlandGrad = kernels_.WendlandGrad(r_scaled) * (S * S * S * S);
                        float F_p_mag = -V_i * V_j * P_sum * wendlandGrad; 
                        
                        fPress += dir * F_p_mag;

                        float viscLap = kernels_.ViscLaplacian(r_scaled) * (S * S * S * S * S);
                        float F_v_mag = V_i * V_j * phys_viscosity * viscLap;
                        
                        fVisc += F_v_mag * (neighbor->physics.velocity - node->physics.velocity);
                    }
                }

                for (auto& neighbor : solidNeighbors) {
                    PointType diff = node->position - neighbor->position;
                    float r = diff.norm();
                    float minDist = (node->size + neighbor->size) * 0.5f;
                    
                    if (r < minDist) {
                        if (r < 1e-5f) {
                            Eigen::Vector3f rnd = Eigen::Vector3f::Random();
                            if (rnd.squaredNorm() < 1e-8f) rnd = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                            fPress += rnd.normalized() * mass_i * kernels_.Poly6(0) * 10.0f;
                            continue;
                        }

                        PointType dir = diff / r; 
                        float S = h / minDist;
                        float r_scaled = r * S;
                        
                        float baseG = (phys_gravityStrength > 0.1f) ? phys_gravityStrength : 9.81f;
                        float stiffness = mass_i * baseG * 250.0f; 
                        float F_repel = stiffness * kernels_.Wendland(r_scaled);
                        fPress += dir * F_repel;
                        
                        Eigen::Vector3f relVel = node->physics.velocity - neighbor->physics.velocity;
                        float approachSpeed = relVel.dot(dir);
                        
                        if (approachSpeed < 0.0f) {
                            float v_lap = kernels_.ViscLaplacian(r_scaled);
                            float damping = mass_i * 50.0f * v_lap;
                            fVisc += -dir * (approachSpeed * damping);
                        }
                    }
                }
                
                node->physics.force += fPress + fVisc;
            }
        }

        for (size_t i = 0; i < sphNodes.size(); ++i) {
            auto& node = sphNodes[i];
            
            float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
            float mass_i = std::max(V_i * phys_restDensity, 1e-6f);

            Eigen::Vector3f accel = node->physics.force / mass_i;
            if (!accel.allFinite()) accel = Eigen::Vector3f::Zero();
            
            float maxAccel = 1000.0f; 
            if (accel.squaredNorm() > maxAccel * maxAccel) {
                accel = accel.normalized() * maxAccel;
            }
            
            node->physics.velocity += accel * dt;
            if (!node->physics.velocity.allFinite()) node->physics.velocity = Eigen::Vector3f::Zero();
            
            node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);

            float maxVel = std::max(h / dt, 25.0f);
            if (node->physics.velocity.squaredNorm() > maxVel * maxVel) {
                node->physics.velocity = node->physics.velocity.normalized() * maxVel;
            }

            FluidMoveAction<T, IndexType> fm;
            fm.node = node;
            fm.oldPos = node->position;
            fm.newPos = node->position + node->physics.velocity * dt;
            pendingFluidMoves.push_back(fm);
        }
    }


    for (size_t i = 0; i < pendingFluidMoves.size(); ++i) {
        auto& move_act = pendingFluidMoves[i];
        
        RayHit_<T, IndexType> hit;
        PointType diff = move_act.newPos - move_act.oldPos;
        float dist = diff.norm();
        if (dist > 1e-5f) {
            PointType dir = diff / dist;
            if (this->raycast(move_act.oldPos, dir, dist + move_act.node->size * 0.5f, hit, move_act.node, true, true)) {
                move_act.newPos = hit.hitPoint + hit.normal * (move_act.node->size * 0.51f);
                float vn = move_act.node->physics.velocity.dot(hit.normal);
                if (vn < 0.0f) {
                    move_act.node->physics.velocity -= vn * hit.normal;
                    move_act.node->physics.velocity *= 0.5f;
                }
            }
        }
        
        this->move(move_act.oldPos, move_act.newPos);
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
            if (field->worldToCell(src, sx, sy, sz)) {
                const GasCell_<T>& s = field->cells[field->index(sx, sy, sz)];
                dst.amount = s.amount;
                dst.velocity = s.velocity;
                dst.data = s.data;
            } else {
                dst.amount = cur.amount;
                dst.velocity = cur.velocity;
                dst.data = cur.data;
                if (cur.totalDensity() > 1e-5f) {
                    PointType outPos = centre + cur.velocity * dt;
                    if (!field->worldToCell(outPos, sx, sy, sz)) {
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
                    if (!field->inRange(nx, ny, nz)) continue;
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