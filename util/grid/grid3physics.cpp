#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    PointType oldPos;
    PointType newPos;
};

template<typename T, typename IndexType>
struct GasMoveAction {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    PointType oldPos;
    PointType newPos;
    float newSize;
};

template<typename T, typename IndexType>
struct GasSplitChildDef {
    PointType pos;
    float size;
    Eigen::Vector3f vel;
};

template<typename T, typename IndexType>
struct GasSplitAction {
    PointType oldPos;
    std::shared_ptr<NodeData_<T, IndexType>> parent;
    std::vector<GasSplitChildDef<T, IndexType>> children;
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
                            if (bType == BodyType::FLUID || bType == BodyType::GAS) {
                                sphNodes.push_back(sp);
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<FluidMoveAction<T, IndexType>> pendingFluidMoves;
    std::vector<GasMoveAction<T, IndexType>> pendingGasMoves;
    std::vector<GasSplitAction<T, IndexType>> pendingGasSplits;

    if (!sphNodes.empty()) {
        float h = phys_smoothingRadius;
        float h2 = h * h;
        float maxKernelVol = 4.18879f * h2 * h;
        
        float C = std::max(phys_smoothingRadius, phys_maxGasSize);
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
                } else if (myPmat.type == BodyType::GAS) {
                    float df = densityFraction;
                    float p_ideal = phys_gasConstant * 0.5f * df;
                    float p_cohesion = phys_gasConstant * 0.8f * df * df;
                    float p_core = phys_gasConstant * 0.3f * df * df * df;
                    
                    node->physics.pressure = p_ideal - p_cohesion + p_core;
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
                        if (bType == BodyType::FLUID || bType == BodyType::GAS) continue;
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
                float raw_mass = (myPmat.type == BodyType::FLUID) ? (V_i * phys_restDensity) : myPmat.mass;
                float mass_i = std::max(raw_mass, 1e-6f);

                Eigen::Vector3f gravityDir = phys_gravity;
                if (phys_useGravityPoint) {
                    Eigen::Vector3f toCenter = phys_gravityCenter - node->position;
                    float dist = toCenter.norm();
                    if (dist > 1e-4f) gravityDir = (toCenter / dist) * phys_gravityStrength;
                    else gravityDir = Eigen::Vector3f::Zero();
                }
                
                if (myPmat.type == BodyType::FLUID) {
                    node->physics.force = gravityDir * mass_i;
                } else {
                    float actualDensity = mass_i / std::max(V_i, 0.0001f);
                    float effectiveDensity = std::max(actualDensity, phys_airDensity * 0.1f);
                    node->physics.force = gravityDir * (effectiveDensity - phys_airDensity) * V_i;
                }

                Eigen::Vector3f fPress = Eigen::Vector3f::Zero();
                Eigen::Vector3f fVisc = Eigen::Vector3f::Zero();

                for (auto& neighbor : dynamicNeighbors) {
                    if (neighbor == node) continue;
                    
                    int nObjIdx = neighbor->objectId + 1;
                    PhysicsMaterial_ nPmat = fastMats[nObjIdx][neighbor->physMatIdx];

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
                        float viscMulti = (myPmat.type == BodyType::GAS || nPmat.type == BodyType::GAS) ? 0.1f : 1.0f;
                        float F_v_mag = V_i * V_j * phys_viscosity * viscMulti * viscLap;
                        
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
            int objIdx = node->objectId + 1;
            PhysicsMaterial_ myPmat = fastMats[objIdx][node->physMatIdx];
            
            float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
            float raw_mass = (myPmat.type == BodyType::FLUID) ? (V_i * phys_restDensity) : myPmat.mass;
            float mass_i = std::max(raw_mass, 1e-6f);

            Eigen::Vector3f accel = node->physics.force / mass_i;
            if (!accel.allFinite()) accel = Eigen::Vector3f::Zero();
            
            float maxAccel = (myPmat.type == BodyType::GAS) ? 500.0f : 1000.0f; 
            if (accel.squaredNorm() > maxAccel * maxAccel) {
                accel = accel.normalized() * maxAccel;
            }
            
            node->physics.velocity += accel * dt;
            if (!node->physics.velocity.allFinite()) node->physics.velocity = Eigen::Vector3f::Zero();
            
            if (myPmat.type == BodyType::GAS) {
                float drag = phys_velocityDamping + (node->size * 0.5f);
                node->physics.velocity *= std::max(0.0f, 1.0f - drag * dt);
            } else {
                node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);
            }

            float maxVel = (myPmat.type == BodyType::GAS) ? 50.0f : std::max(h / dt, 25.0f);
            if (node->physics.velocity.squaredNorm() > maxVel * maxVel) {
                node->physics.velocity = node->physics.velocity.normalized() * maxVel;
            }

            if (myPmat.type == BodyType::FLUID) {
                FluidMoveAction<T, IndexType> fm;
                fm.node = node;
                fm.oldPos = node->position;
                fm.newPos = node->position + node->physics.velocity * dt;
                pendingFluidMoves.push_back(fm);
            } else {
                auto obj = getObject(node->objectId);
                float maxGasSize = (obj && obj->maxGasVoxelSize > 0.0f) ? obj->maxGasVoxelSize : phys_maxGasSize;
                
                float currentSize = node->size + phys_gasExpansionRate * dt;
                
                if (obj && currentSize > node->size) {
                    float oldSize = node->size;
                    float V_new = currentSize * currentSize * currentSize;
                    float oldVol = oldSize * oldSize * oldSize;
                    float ratio = oldVol / V_new; 
                    
                    Material_<T, IndexType> rMat = obj->getRenderMaterial(node->renderMatIdx);
                    
                    rMat.absorption *= ratio;
                    rMat.emittance *= ratio;

                    float lengthRatio = oldSize / currentSize;
                    float sqLengthRatio = lengthRatio * lengthRatio;
                    float tr = std::max(rMat.transmission, 0.001f);
                    rMat.transmission = std::pow(tr, sqLengthRatio);

                    rMat.transmission = std::round(rMat.transmission * 100.0f) / 100.0f;
                    rMat.absorption.x() = std::round(rMat.absorption.x() * 100.0f) / 100.0f;
                    rMat.absorption.y() = std::round(rMat.absorption.y() * 100.0f) / 100.0f;
                    rMat.absorption.z() = std::round(rMat.absorption.z() * 100.0f) / 100.0f;
                    
                    node->renderMatIdx = obj->getOrAddRenderMaterial(rMat);
                    
                    if (rMat.transmission >= 0.99f && rMat.absorption.norm() < 0.05f) {
                        node->setActive(false);
                        node->setVisible(false);
                    }
                }

                if (currentSize >= maxGasSize && node->isActive()) {
                    GasSplitAction<T, IndexType> split;
                    split.oldPos = node->position;
                    split.parent = node;
                    
                    float newSize = currentSize * 0.5f;
                    for (int dx : {-1, 1}) {
                        for (int dy : {-1, 1}) {
                            for (int dz : {-1, 1}) {
                                PointType offset(dx * newSize * 0.5f, dy * newSize * 0.5f, dz * newSize * 0.5f);
                                GasSplitChildDef<T, IndexType> c;
                                c.pos = node->position + offset;
                                c.size = newSize;
                                c.vel = node->physics.velocity + offset.normalized() * (phys_gasExpansionRate * 0.5f);
                                split.children.push_back(c);
                            }
                        }
                    }
                    pendingGasSplits.push_back(split);
                } else {
                    GasMoveAction<T, IndexType> gm;
                    gm.node = node;
                    gm.oldPos = node->position;
                    gm.newPos = node->position + node->physics.velocity * dt;
                    gm.newSize = currentSize;
                    pendingGasMoves.push_back(gm);
                }
            }
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

    for (auto& gm : pendingGasMoves) {
        RayHit_<T, IndexType> hit;
        PointType diff = gm.newPos - gm.oldPos;
        float dist = diff.norm();
        if (dist > 1e-5f) {
            PointType dir = diff / dist;
            if (this->raycast(gm.oldPos, dir, dist + gm.node->size * 0.5f, hit, gm.node, true, true)) {
                gm.newPos = hit.hitPoint + hit.normal * (gm.node->size * 0.51f);
                float vn = gm.node->physics.velocity.dot(hit.normal);
                if (vn < 0.0f) {
                    gm.node->physics.velocity -= vn * hit.normal;
                    gm.node->physics.velocity *= 0.5f; 
                }
            }
        }

        this->update(gm.oldPos, gm.newPos, gm.node->data, gm.node->isVisible(), gm.node->color, 
                     gm.newSize, gm.node->isActive(), gm.node->objectId, 
                     -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, EPSILON);
    }

    for (auto& split : pendingGasSplits) {
        if (this->remove(split.oldPos, EPSILON)) {
            auto obj = this->getObject(split.parent->objectId);
            Material_<T, IndexType> rMat = obj ? obj->getRenderMaterial(split.parent->renderMatIdx) : Material_<T, IndexType>();
            float newMass = fastMats[split.parent->objectId + 1][split.parent->physMatIdx].mass * 0.125f;

            for (auto& c : split.children) {
                this->set(split.parent->data, c.pos, split.parent->isVisible(), split.parent->color, c.size, split.parent->isActive(), split.parent->objectId,
                          rMat.emittance, rMat.roughness, rMat.metallic, rMat.transmission, rMat.ior, rMat.absorption,
                          BodyType::GAS, newMass);
                
                if (auto n = this->find(c.pos, EPSILON)) {
                    n->physics.velocity = c.vel;
                }
            }
        }
    }

}
}