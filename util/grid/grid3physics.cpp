#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType, GridStoragePath StoragePath>
void Octree<T, IndexType, StoragePath>::stepPhysics(float dt) {

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

    std::vector<std::shared_ptr<NodeData>> fluidNodes;
    {
        std::lock_guard<std::mutex> lock(physicsMutex_);
        
        size_t writeIdx = 0;
        for (size_t i = 0; i < activePhysicsNodes_.size(); ++i) {
            if (!activePhysicsNodes_[i].expired()) {
                activePhysicsNodes_[writeIdx++] = activePhysicsNodes_[i];
            }
        }
        activePhysicsNodes_.resize(writeIdx);
        
        fluidNodes.reserve(writeIdx);
        for (size_t i = 0; i < writeIdx; ++i) {
            if (auto sp = activePhysicsNodes_[i].lock()) {
                if (sp->isActive()) {
                    int objIdx = sp->objectId + 1;
                    if (objIdx >= 0 && objIdx < fastMatsSize) {
                        const auto& mats = fastMats[objIdx];
                        if (sp->physMatIdx < mats.size() && mats[sp->physMatIdx].type == BodyType::FLUID) {
                            fluidNodes.push_back(sp);
                        }
                    }
                }
            }
        }
    }

    if (fluidNodes.empty()) return;

    float h = phys_smoothingRadius;
    float h2 = h * h;
    float maxKernelVol = 4.18879f * h2 * h;
    std::vector<std::shared_ptr<NodeData>> allNeighbors;
    allNeighbors.reserve(fluidNodes.size() * 64);
    std::vector<size_t> neighborOffsets(fluidNodes.size() + 1, 0);

    // --- Pass 1: Gather Neighbors, Compute Density & Pressure ---
    for (size_t i = 0; i < fluidNodes.size(); ++i) {
        auto& node = fluidNodes[i];
        PointType center = node->position;
        
        size_t startOffset = allNeighbors.size();

        OctreeNode* stack[256];
        int stackPtr = 0;
        stack[stackPtr++] = root_.get();
        
        while (stackPtr > 0) {
            OctreeNode* currNode = stack[--stackPtr];
            
            PointType closest;
            for (int k = 0; k < Dim; ++k) {
                closest[k] = std::max(currNode->bounds.first[k], std::min(center[k], currNode->bounds.second[k]));
            }
            if ((closest - center).squaredNorm() > h2) continue;
            
            if (!currNode->isLoaded()) continue;

            std::shared_lock<std::shared_mutex> lock(currNode->nodeMutex);
            for (size_t k = 0; k < currNode->points.size(); ++k) {
                const auto& pt = currNode->points[k];
                if (pt->isActive() && (pt->position - center).squaredNorm() <= h2) {
                    allNeighbors.push_back(pt);
                }
            }
            
            if (!currNode->isLeaf()) {
                for (int k = 0; k < 8; ++k) {
                    if (currNode->children[k]) stack[stackPtr++] = currNode->children[k].get();
                }
            }
        }
        
        size_t endOffset = allNeighbors.size();
        neighborOffsets[i+1] = endOffset;

        // Dimensionless Density (Volume Fraction)
        float densityFraction = 0.0f;
        
        for (size_t j = startOffset; j < endOffset; ++j) {
            auto& neighbor = allNeighbors[j];
            int nObjIdx = neighbor->objectId + 1;
            
            PhysicsMaterial_ nPmat;
            if (nObjIdx >= 0 && nObjIdx < fastMatsSize && neighbor->physMatIdx < fastMats[nObjIdx].size()) {
                nPmat = fastMats[nObjIdx][neighbor->physMatIdx];
            }

            if (nPmat.type == BodyType::FLUID) {
                float r = (center - neighbor->position).norm();
                float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol); 
                densityFraction += V_j * kernels_.Poly6(r);
            }
        }

        node->physics.density = densityFraction * phys_restDensity; 
        
        // Compute pressure using Volume formulation (1.0 = perfect rest density).
        // Strongly clamp the max over-density to prevent explosive single-frame shocks.
        float over_density = std::max(0.0f, densityFraction - 1.0f);
        over_density = std::min(over_density, 1.5f); // Cap maximum pressure force multiplier
        node->physics.pressure = phys_gasConstant * over_density; 
    }

    // --- Pass 2: Compute Forces ---
    for (size_t i = 0; i < fluidNodes.size(); ++i) {
        auto& node = fluidNodes[i];
        
        float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
        float mass_i = V_i * phys_restDensity;

        // Calculate Gravity
        Eigen::Vector3f gravityDir = phys_useGravityPoint ? 
            (phys_gravityCenter - node->position).normalized() * phys_gravityStrength : phys_gravity;
        
        node->physics.force = gravityDir * mass_i;

        Eigen::Vector3f fPress = Eigen::Vector3f::Zero();
        Eigen::Vector3f fVisc = Eigen::Vector3f::Zero();

        size_t startOffset = neighborOffsets[i];
        size_t endOffset = neighborOffsets[i+1];

        for (size_t j = startOffset; j < endOffset; ++j) {
            auto& neighbor = allNeighbors[j];
            if (neighbor == node) continue;
            
            int nObjIdx = neighbor->objectId + 1;
            PhysicsMaterial_ nPmat;
            if (nObjIdx >= 0 && nObjIdx < fastMatsSize && neighbor->physMatIdx < fastMats[nObjIdx].size()) {
                nPmat = fastMats[nObjIdx][neighbor->physMatIdx];
            }

            PointType diff = node->position - neighbor->position;
            float r = diff.norm();
            
            // Stacked singularity fallback (prevents perfectly overlapping random spawns from generating NaNs)
            if (r < 1e-5f) {
                fPress += Eigen::Vector3f::Random().normalized() * mass_i * 25.0f;
                continue;
            }
            
            if (r < h) {
                PointType dir = diff / r; // Direction pointing AWAY from neighbor
                
                if (nPmat.type == BodyType::FLUID) {
                    float V_j = std::min(neighbor->size * neighbor->size * neighbor->size, maxKernelVol);

                    // Fluid Pressure (Volume SPH Formulation)
                    // F_press = - V_i * V_j * (P_i + P_j) * \nabla W
                    float P_sum = node->physics.pressure + neighbor->physics.pressure;
                    float F_p_mag = -V_i * V_j * P_sum * kernels_.SpikyGrad(r); // SpikyGrad is negative, resulting in a positive magnitude
                    fPress += dir * F_p_mag;

                    // Fluid Viscosity
                    float F_v_mag = V_i * V_j * phys_viscosity * kernels_.ViscLaplacian(r);
                    fVisc += F_v_mag * (neighbor->physics.velocity - node->physics.velocity);

                } else if (nPmat.type == BodyType::STATIC || nPmat.type == BodyType::KINEMATIC) {
                    // Fluid <-> Solid Interaction (Robust Penalty Force - decoupled from SPH density)
                    float minDist = (node->size + neighbor->size) * 0.5f;
                    if (r < minDist) {
                        float overlap = minDist - r;
                        
                        // Dynamic stiffnes calculated to comfortably counteract gravity at minimal penetration
                        float baseG = (phys_gravityStrength > 0.1f) ? phys_gravityStrength : 9.81f;
                        float stiffness = mass_i * baseG * 200.0f; 
                        fPress += dir * overlap * stiffness;
                        
                        // Impact damping against the wall to absorb kinetic energy
                        Eigen::Vector3f relVel = node->physics.velocity - neighbor->physics.velocity;
                        float approachSpeed = relVel.dot(dir);
                        if (approachSpeed < 0.0f) {
                            fPress += -dir * approachSpeed * mass_i * 20.0f;
                        }
                    }
                }
            }
        }
        node->physics.force += fPress + fVisc;
    }

    // --- Pass 3: Integration and Movement ---
    struct MoveAction {
        std::shared_ptr<NodeData> node;
        PointType newPos;
    };
    std::vector<MoveAction> pendingMoves;
    pendingMoves.reserve(fluidNodes.size());

    for (size_t i = 0; i < fluidNodes.size(); ++i) {
        auto& node = fluidNodes[i];
        
        float V_i = std::min(node->size * node->size * node->size, maxKernelVol);
        float mass_i = V_i * phys_restDensity;

        Eigen::Vector3f accel = node->physics.force / mass_i;
        
        // Strict game-friendly acceleration clamp (Max 100 Gs)
        float maxAccel = 1000.0f; 
        if (accel.squaredNorm() > maxAccel * maxAccel) {
            accel = accel.normalized() * maxAccel;
        }
        
        // Semi-implicit Euler integration
        node->physics.velocity += accel * dt;
        
        // Apply global velocity damping
        node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);

        // Limit Velocity (Prevent particles from tunneling through the grid in one frame)
        float maxVel = std::max(h / dt, 25.0f);
        if (node->physics.velocity.squaredNorm() > maxVel * maxVel) {
            node->physics.velocity = node->physics.velocity.normalized() * maxVel;
        }

        PointType newPos = node->position + node->physics.velocity * dt;
        pendingMoves.push_back({node, newPos});
    }

    // --- Execute Spatial Movements ---
    for (size_t i = 0; i < pendingMoves.size(); ++i) {
        auto& move = pendingMoves[i];
        auto& node = move.node;
        
        // Extract, update position, and re-insert
        if (removeRecursive(root_.get(), node->getCubeBounds(), node)) {
            node->position = move.newPos;
            ensureBounds(node->getCubeBounds());
            if (!insertRecursive(root_.get(), node, 0)) {
                size--; // Revert counter if re-insertion fails (e.g. invalid bounds)
            }
        }
    }

}
}