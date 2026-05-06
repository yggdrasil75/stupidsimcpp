#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType, GridStoragePath StoragePath>
void Octree<T, IndexType, StoragePath>::stepPhysics(float dt) {
    TIME_FUNCTION;
    if (!root_) return;

    std::vector<std::shared_ptr<NodeData>> dynamicNodes;
    
    {
        std::lock_guard<std::mutex> lock(physicsMutex_);
        activePhysicsNodes_.erase(
            std::remove_if(activePhysicsNodes_.begin(), activePhysicsNodes_.end(),
                [](const std::weak_ptr<NodeData>& wp) { return wp.expired(); }),
            activePhysicsNodes_.end()
        );
        
        dynamicNodes.reserve(activePhysicsNodes_.size());
        for (const auto& wp : activePhysicsNodes_) {
            if (auto sp = wp.lock()) {
                if (sp->isActive()) dynamicNodes.push_back(sp);
            }
        }
    }

    if (dynamicNodes.empty()) return;

#ifdef VULKAN_SUPPORT
    std::vector<GPUParticle> gpuParticles(dynamicNodes.size());
    for (size_t i = 0; i < dynamicNodes.size(); ++i) {
        auto& node = dynamicNodes[i];
        gpuParticles[i].pos_mass = Eigen::Vector4f(node->position.x(), node->position.y(), node->position.z(), node->physics.mass);
        gpuParticles[i].vel_density = Eigen::Vector4f(node->physics.velocity.x(), node->physics.velocity.y(), node->physics.velocity.z(), node->physics.density);
        gpuParticles[i].force_press = Eigen::Vector4f(0.0f, 0.0f, 0.0f, node->physics.pressure);
        
        int sizeInt;
        std::memcpy(&sizeInt, &node->size, sizeof(float));
        gpuParticles[i].type_pad = Eigen::Vector4i(static_cast<int>(node->physics.type), sizeInt, 0, 0);
    }

    SPHDensityPC dpc{};
    dpc.h = phys_smoothingRadius;
    dpc.h2 = phys_smoothingRadius * phys_smoothingRadius;
    dpc.poly6_k = kernels_.poly6_k;
    dpc.restDensity = phys_restDensity;
    dpc.gasConstant = phys_gasConstant;
    dpc.numParticles = dynamicNodes.size();

    SPHForcePC fpc{};
    fpc.h = phys_smoothingRadius;
    fpc.spiky_k = kernels_.spiky_k;
    fpc.visc_l_k = kernels_.visc_l_k;
    fpc.viscosity = phys_viscosity;
    fpc.gravX = phys_gravity.x();
    fpc.gravY = phys_gravity.y();
    fpc.gravZ = phys_gravity.z();
    fpc.gravStrength = phys_gravityStrength;
    fpc.gravCX = phys_gravityCenter.x();
    fpc.gravCY = phys_gravityCenter.y();
    fpc.gravCZ = phys_gravityCenter.z();
    fpc.useGravityPoint = phys_useGravityPoint ? 1 : 0;
    fpc.numParticles = dynamicNodes.size();

    vkCtx.dispatchPhysics(gpuParticles, dpc, fpc);

    for (size_t i = 0; i < dynamicNodes.size(); ++i) {
        dynamicNodes[i]->physics.density = gpuParticles[i].vel_density.w();
        dynamicNodes[i]->physics.pressure = gpuParticles[i].force_press.w();
        dynamicNodes[i]->physics.force = Eigen::Vector3f(gpuParticles[i].force_press.x(), gpuParticles[i].force_press.y(), gpuParticles[i].force_press.z());
    }
#else
    float phys_h2 = phys_smoothingRadius * phys_smoothingRadius;
    std::vector<std::shared_ptr<NodeData>> allNeighbors;
    allNeighbors.reserve(dynamicNodes.size() * 40);
    std::vector<size_t> neighborOffsets(dynamicNodes.size() + 1, 0);
    for (size_t i = 0; i < dynamicNodes.size(); ++i) {
        auto& node = dynamicNodes[i];
        
        if (node->physics.type == BodyType::FLUID || node->physics.type == BodyType::GAS) {
            node->physics.density = 0.0f;
            size_t startOffset = allNeighbors.size();
            searchNode(root_.get(), node->position, phys_h2, -1, allNeighbors);
            size_t endOffset = allNeighbors.size();
            neighborOffsets[i+1] = endOffset;

            for (size_t j = startOffset; j < endOffset; ++j) {
                auto& neighbor = allNeighbors[j];
                if (neighbor->physics.type != BodyType::FLUID && neighbor->physics.type != BodyType::GAS) continue;

                float r = (node->position - neighbor->position).norm();
                node->physics.density += neighbor->physics.mass * kernels_.Poly6(r);
            }

            if (node->physics.density < 0.001f) node->physics.density = 0.001f;
            
            float maxDensity = phys_restDensity * 3.0f;
            float clampedDensity = std::min(node->physics.density, maxDensity);

            if (node->physics.type == BodyType::GAS) {
                node->physics.pressure = phys_gasConstant * clampedDensity;
            } else {
                node->physics.pressure = phys_gasConstant * (clampedDensity - phys_restDensity);
            }
        } else {
            neighborOffsets[i+1] = allNeighbors.size();
        }
    }

    for (size_t i = 0; i < dynamicNodes.size(); ++i) {
        auto& node = dynamicNodes[i];

        if (phys_useGravityPoint) {
            Eigen::Vector3f dir = phys_gravityCenter - node->position;
            float distSq = dir.squaredNorm();
            if (distSq > 0.0001f) {
                node->physics.force = dir.normalized() * (phys_gravityStrength * node->physics.mass);
            } else {
                node->physics.force.setZero();
            }
        } else {
            node->physics.force = phys_gravity * node->physics.mass;
        }

        if (node->physics.type == BodyType::FLUID || node->physics.type == BodyType::GAS) {
            Eigen::Vector3f fPress = Eigen::Vector3f::Zero();
            Eigen::Vector3f fVisc = Eigen::Vector3f::Zero();

            size_t startOffset = neighborOffsets[i];
            size_t endOffset = neighborOffsets[i+1];

            for (size_t j = startOffset; j < endOffset; ++j) {
                auto& neighbor = allNeighbors[j];
                if (neighbor == node) continue;
                if (neighbor->physics.type != BodyType::FLUID && neighbor->physics.type != BodyType::GAS) continue;

                PointType diff = node->position - neighbor->position;
                float r = diff.norm();
                
                if (r > 0.0001f && r < phys_smoothingRadius) {
                    PointType dir = diff / r;
                    float pressureTerm = (node->physics.pressure / (node->physics.density * node->physics.density)) + 
                                         (neighbor->physics.pressure / (neighbor->physics.density * neighbor->physics.density));
                    
                    float sizeFactor = node->size * neighbor->size;
                    float massFactor = node->physics.mass * neighbor->physics.mass;

                    fPress += -dir * massFactor * pressureTerm * kernels_.SpikyGrad(r) * sizeFactor;

                    fVisc += phys_viscosity * massFactor * 
                             (neighbor->physics.velocity - node->physics.velocity) / neighbor->physics.density * 
                             kernels_.ViscLaplacian(r) * sizeFactor;
                }
            }
            node->physics.force += fPress + fVisc;
        }
    }

#endif
    std::vector<std::pair<std::shared_ptr<NodeData>, PointType>> movesBuffer;
    movesBuffer.reserve(dynamicNodes.size());

    #pragma omp parallel for
    for (auto& node : dynamicNodes) {
        if (node->physics.type == BodyType::KINEMATIC) continue;
        if (!node->physics.force.allFinite()) {
            node->physics.force.setZero();
        }

        Eigen::Vector3f acceleration = node->physics.force / node->physics.mass;
        float maxAccel = 50000.0f;
        if (acceleration.squaredNorm() > maxAccel * maxAccel) {
            acceleration = acceleration.normalized() * maxAccel;
        }

        node->physics.velocity += acceleration * dt;
        node->physics.velocity *= std::max(0.0f, 1.0f - phys_velocityDamping * dt);

        float maxVel = 50.0f;
        if (node->physics.velocity.squaredNorm() > maxVel * maxVel) {
            node->physics.velocity = node->physics.velocity.normalized() * maxVel;
        }
        PointType predictedPos = node->position + node->physics.velocity * dt;

        if (!predictedPos.allFinite()) {
            predictedPos = node->position;
            node->physics.velocity.setZero();
        }
        
        PointType castDir = node->physics.velocity;
        if (castDir.squaredNorm() < 1e-6f) {
            castDir = PointType(0.0f, -1.0f, 0.0f);
        }

        float stepDist = node->physics.velocity.norm() * dt;
        RayHit hit;
        if (raycast(node->position, castDir, stepDist + node->size * 0.505f, hit, node, true, true)) {
            if (hit.node) {
                predictedPos = hit.hitPoint + hit.normal * (node->size * 0.51f);
                
                float restitution = 0.3f;
                Eigen::Vector3f vNorm = node->physics.velocity.dot(hit.normal) * hit.normal;
                Eigen::Vector3f vTan = node->physics.velocity - vNorm;
                
                node->physics.velocity = (vTan * 0.8f) - (vNorm * restitution);
            }
        }

        float halfSize = node->size * 0.5f;
        for (int i = 0; i < Dim; ++i) {
            if (predictedPos[i] < root_->bounds.first[i] + halfSize) {
                predictedPos[i] = root_->bounds.first[i] + halfSize;
                node->physics.velocity[i] *= -0.3f;
            } else if (predictedPos[i] > root_->bounds.second[i] - halfSize) {
                predictedPos[i] = root_->bounds.second[i] - halfSize;
                node->physics.velocity[i] *= -0.3f;
            }
        }

        movesBuffer.push_back({node, predictedPos});
    }

    for (auto& movePair : movesBuffer) {
        auto& node = movePair.first;
        PointType& newPos = movePair.second;
        
        removeRecursive(root_.get(), node->getCubeBounds(), node);
        node->position = newPos;
        ensureBounds(node->getCubeBounds());
        insertRecursive(root_.get(), node, 0);
    }
}
}