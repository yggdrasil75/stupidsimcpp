#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <random>
#include <chrono>
#include <cstdint>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <type_traits>
#include <future>
#include "impl/structs.inl"
#include "impl/skybox.inl"
#include "impl/rendering.inl"

#ifdef SSE
#include <immintrin.h>
#endif

namespace Grid {
    // note for myself: use weak pointers to track objects due to them changing from grid physics.
    // I could also use callbacks
    // for physics I could make a object class and store in that an index for subobject physic types (ie: skin vs bone vs cartilege all stored in an animal object)
    // I need to make sure that objects are either fully loaded or fully unloaded. I cant have a partially loaded object. so that has to be included in the lazy unload at some point.
    // further notes on objects: could bring back indexed render materials, store a vector of materials in the object, index in the item. have it be 1 byte instead of 4.
    // same with physical materials, ie: have a "bone" material, "skin" "muscle" "blood" etc when simulating the physical body of a character. (well, maybe not that detailed. but something similar)
    // also need to store relative positions to the object center, both "resting" and "current". 
    // some rendering options: sgvf (temporal accumulation)
    // also some more simplistic temporal model might be easier.
    
    // stratification: halton or sobol, cranley-patterson rotations (look into these)
    // stochastic transparency (might be for fast version?)
    // /*At half res (scale=0.5), you're searching 3 low-res pixels = 6 full-res pixels
    //     The object ID matching fallback radius is 9 = 18 full-res pixels
    //     This is likely causing excessive blurring at geometric edges. You need to adapt the radius to your scale:
    // */
    // change emittance to a GL_RGB9_E5 to support color light/hdr

// importance sampling: first hit finds objects that are more complex to render, less complex objects have less samples.
// every 1/10th the samples per pixel, find regions with little variance, drop those in favor of focusing on the ones with large variance.
// screen space path reuse for nearby pixels (I really need wavefront path tracing.)
// first hit hitpoint and norm really need reuse.

    // a-trous wavelet denoising intead of bilateral filtering.
    // secondary pass of edge sharpening.
    // have pbr buffer output sum and sum of squares to make better blending easier to manage
    // edge aware median prefilter for luminance variance clamping passes
    // dynamically resize smoothing window based on noise level

    //object crud:
    /*
    insert list of voxels.
    get a vector of weak pointers to all voxels in an object
    import an obj.
    change the current allow partial offload bit to be a flag so I can add more there later.
    changing a render material by its index
    getting a render material index from a voxel
    changing a physics material by its index
    getting a physics material index from a voxel
    object transforms (rotation, moving, etc)
    removing an object
    subdivide object (split each voxel into 8)
    smooth object? (subdivide and then remove corner?)

    */

    /*
    gas ideas:
    eulerian field using nodes as sample points.
    would probably need a minimum size to be able to be reasonable.


    fluid ideas:
    bodies of water can be simulated with eulerian as well.
    rivers and contained fluids will use the current sph voxels methodology.
    as voxels in rivers reach oceans or lakes, they will "dissolve" into the surrounding euler field.
    as they rain down from a cloud, they will "clump" into voxels.

    static: 
    static. no physics move it, it can block fluids, gases, and rigid. it doesnt move. its static.

    rigid:
    can move and be moved by fluids and gases (needs lots of pressure for gases to move it) and other rigids.
    physics forces like gravity, wind, etc.
    the ai models we train can move these directly if they are part of their object. alternatively, they can call the kinematics.
    need to tract connections (voxel 1 is connected to voxel 2 at this precise 3d angle)

    kinematic:
    we will create a "test chamber" where rigid bodies are controlled by the models trained for them.
    this will generate kinematics for fast calls by the ai, and then have an "untested" solver where the physics will operate a rigid body.
    that will be added to the test chamber if it happens often, creating a new predefined kinematic for the ai to call.

    soft body:
    can move and be moved by fluids and gases (needs lots of pressure for gases to move it) and other rigids.
    physics forces like gravity, wind, etc.
    can be deformed (move voxels) will require voxel positions to be stored twice: current and "resting".
    might need permanent deforms: major deform will break it (cut/tear) and long term deforms will move the static.
    could also just have a "natural connection" instead where the physics just push back instead of remembering the resting position.
    dont know how to do that.

    */

    /*
    eventual creature stuff:
    Creature = {
        Skeleton: hierarchy of Bone nodes,
        Joint motors: angles/torques for AI to control,
        SoftTissueSolver: (PBD/FEM) attached to bone frames,
        Material palettes: per‑creature, shared by all voxels.
    }
    */

template<typename T, typename GasT = float, typename IndexType = uint16_t, GridStoragePath StoragePath = ".">
class Octree {
public:
    using NodeData = NodeData_<T, IndexType, StoragePath>;
    using OctreeNode = OctreeNode_<T, GasT, IndexType, StoragePath>;
    using Material = Material_<T, IndexType, StoragePath>;
    using RayHit = RayHit_<T, IndexType, StoragePath>;
    using RenderNode = RenderNode_<T, IndexType, StoragePath>;
    using RenderData = RenderData_<T, IndexType, StoragePath>;
    using RenderBuffer = RenderBuffer_<T, IndexType, StoragePath>;
    using GridObject = GridObject_<T, IndexType, StoragePath>;
    using EulerianGasState = EulerianGasState_<GasT>;

private:
    int countBits(uint8_t mask) const {
        int count = 0;
        while (mask) {
            mask &= (mask - 1);
            count++;
        }
        return count;
    }

    std::unique_ptr<OctreeNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    std::unordered_map<int, std::shared_ptr<GridObject>> objects_;
    mutable std::shared_mutex objectsMutex_;
    
    Skybox skybox_;
    Eigen::Vector3f skylight_ = {0.1f, 0.1f, 0.1f};
    Eigen::Vector3f backgroundColor_ = {0.53f, 0.81f, 0.92f};

    mutable std::queue<std::function<void()>> taskQueue_;
    mutable std::mutex taskMutex_;
    mutable std::condition_variable taskCV_;
    std::thread workerThread_;
    std::atomic<bool> stopWorker_{false};
    std::atomic<bool> autoOptimize_{true};
    std::atomic<bool> streamingQueued_{false};
    std::atomic<uint32_t> frameCounter_{0};

    float minLodVolume_ = 0.0f;
    float minLodSize_ = 0.0f;
    size_t regionTargetPoints_ = 4096;

    std::vector<std::weak_ptr<NodeData>> activePhysicsNodes_;
    std::mutex physicsMutex_;

    float phys_smoothingRadius = 0.2f;
    float phys_restDensity = 1000.0f;
    float phys_gasConstant = 2000.0f;
    float phys_viscosity = 200.0f;
    float phys_velocityDamping = 0.5f;
    float phys_airDensity = 1.225f;
    float phys_gasExpansionRate = 0.5f;
    float phys_maxGasSize = 1.0f;
    Eigen::Vector3f phys_gravity{0.0f, -9.81f, 0.0f};

    SPHKernels kernels_{phys_smoothingRadius};
    float phys_h2 = phys_smoothingRadius * phys_smoothingRadius;
    float phys_poly6 = 315.0f / (64.0f * M_PI * std::pow(phys_smoothingRadius, 9));
    float phys_spikyGrad = -45.0f / (M_PI * std::pow(phys_smoothingRadius, 6));
    float phys_viscLap = 45.0f / (M_PI * std::pow(phys_smoothingRadius, 6));
    
    bool phys_useGravityPoint = true;
    PointType phys_gravityCenter{0.0f, 0.0f, 0.0f};
    float phys_gravityStrength = 9.81f;
    std::atomic<bool> physicsCollidersDirty_{true};

    void collectCollidersRecursive(OctreeNode* node, std::vector<std::pair<PointType, float>>& colliders) {
        if (!node) return;
        ensureLoaded(node, false);
        for (const auto& pt : node->points) {
            auto obj = getObject(pt->objectId);
            BodyType bType = obj ? obj->getPhysicsMaterial(pt->physMatIdx).type : BodyType::STATIC;
            
            if (pt->isActive() && (bType == BodyType::STATIC || bType == BodyType::KINEMATIC)) {
                colliders.emplace_back(pt->position, pt->size);
            }
        }
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) collectCollidersRecursive(child.get(), colliders);
            }
        }
    }

    void lazilyOffload(OctreeNode* node) {
        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            if (!node->isLoaded() || node->isSaveQueued()) return;

            node->setSaveQueued(true);
            node->setLoadQueued(false);
        }

        enqueueTask([this, node]() {
            {
                std::shared_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (node->isLoaded() && node->isSaveQueued()) {
                    if (node->isDirty()) {
                        node->saveRegion();
                    }
                }
            }
            node->offload();
            {
                std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                node->setSaveQueued(false);
            }
        });
    }

    void ensureLoaded(OctreeNode* node, bool asyncLoad = false) {
        {
            // std::unique_lock<std::shared_mutex> lock(node->nodeMutex); // using atomics so dont need this?
            if (node->isLoaded() || node->isQueued()) return; 
            else {
                node->setLoadQueued(true);
                node->setSaveQueued(false);
            }
        }

        if (asyncLoad) {
            enqueueTask([this, node]() {
                bool justLoaded = false;
                {
                    std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                    if (!node->isLoaded()) {
                        node->loadRegion();
                        justLoaded = node->isLoaded();
                    }
                    node->setLoadQueued(false);
                }
                if (justLoaded) {
                    ensureLOD(node);
                }
                
            });
        } else {
            {
                std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (!node->isLoaded()) node->loadRegion();
                node->setLoadQueued(false);
            }
            if (node->isLoaded()) {
                ensureLOD(node);
            }
        }
    }

    void startWorkerThread() {
        stopWorker_.store(false);
        workerThread_ = std::thread([this]() {
            auto lastOptimize = std::chrono::steady_clock::now();
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(taskMutex_);
                    auto nextOptimize = lastOptimize + std::chrono::seconds(60);
                    
                    bool timedOut = !taskCV_.wait_until(lock, nextOptimize, [this] { 
                        return stopWorker_ || !taskQueue_.empty(); 
                    });
                    
                    if (stopWorker_ && taskQueue_.empty()) return;
                    
                    if (!taskQueue_.empty()) {
                        task = std::move(taskQueue_.front());
                        taskQueue_.pop();
                    } else if (timedOut && autoOptimize_) {
                        task = [this]() { this->optimize(); };
                        lastOptimize = std::chrono::steady_clock::now();
                    }
                }
                if (task) {
                    task();
                    lastOptimize = std::chrono::steady_clock::now();
                }
            }
        });
    }

    void stopWorkerThread() {
        stopWorker_.store(true);
        taskCV_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    BoundingBox getNodesBounds(const std::vector<std::shared_ptr<NodeData>>& nodes) const {
        if (nodes.empty()) return {PointType::Zero(), PointType::Zero()};
        BoundingBox bounds = nodes[0]->getCubeBounds();
        for (size_t i = 1; i < nodes.size(); ++i) {
            BoundingBox cb = nodes[i]->getCubeBounds();
            bounds.first = bounds.first.cwiseMin(cb.first);
            bounds.second = bounds.second.cwiseMax(cb.second);
        }
        return bounds;
    }

    OctreeNode* getHighestCommonNode(OctreeNode* current, const BoundingBox& bounds, int currentDepth, int& outDepth) const {
        if (!current || current->isLeaf()) {
            outDepth = currentDepth;
            return current;
        }
        for (int i = 0; i < 8; ++i) {
            if (current->children[i]) {
                BoundingBox cb = createChildBounds(current, i);
                if (boxContainsBox(cb, bounds)) {
                    return getHighestCommonNode(current->children[i].get(), bounds, currentDepth + 1, outDepth);
                }
            }
        }
        outDepth = currentDepth;
        return current;
    }

    size_t removeObjectBatchRecursive(OctreeNode* node, int objectId) {
        if (!node) return 0;
        ensureLoaded(node, false);
        size_t removed = 0;
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            auto it = std::partition(node->points.begin(), node->points.end(),
                [objectId](const auto& pt) { return pt->objectId != objectId; });
            if (it != node->points.end()) {
                removed += std::distance(it, node->points.end());
                node->points.erase(it, node->points.end());
            }
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    removed += removeObjectBatchRecursive(node->children[i].get(), objectId);
                }
            }
        }
        if (removed > 0) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        return removed;
    }

    size_t removeSpecificNodesBatchRecursive(OctreeNode* node, const std::unordered_set<std::shared_ptr<NodeData>>& nodesToRemove) {
        if (!node || nodesToRemove.empty()) return 0;
        ensureLoaded(node, false);
        size_t removed = 0;
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            auto it = std::partition(node->points.begin(), node->points.end(),
                [&](const auto& pt) { return nodesToRemove.find(pt) == nodesToRemove.end(); });
            if (it != node->points.end()) {
                removed += std::distance(it, node->points.end());
                node->points.erase(it, node->points.end());
            }
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    removed += removeSpecificNodesBatchRecursive(node->children[i].get(), nodesToRemove);
                }
            }
        }
        if (removed > 0) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        return removed;
    }

public:
    std::shared_ptr<GridObject> getOrCreateObject(int id) {
        std::unique_lock<std::shared_mutex> lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        auto obj = std::make_shared<GridObject>(id);
        objects_[id] = obj;
        return obj;
    }

    std::shared_ptr<GridObject> getObject(int id) const {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        return nullptr;
    }

    void addSkyBody(int id, const PointType& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance = 255) {
        skybox_.addBody(id, dir, angularRadius, r, g, b, emittance);
    }

    void moveSkyBody(int id, const PointType& newDir) {
        skybox_.moveBody(id, newDir);
    }

    void removeSkyBody(int id) {
        skybox_.removeBody(id);
    }

    void bakeSkyBody(int id) {
        skybox_.bakeBody(id);
    }

    void waitForIdle() {
        if (std::this_thread::get_id() == workerThread_.get_id()) {
            return;
        }
        std::promise<void> p;
        auto f = p.get_future();
        enqueueTask([&p]{ p.set_value(); });
        f.wait();
    }

    void setPhysicsSmoothingRadius(float radius) {
        phys_smoothingRadius = radius;
        kernels_.update(radius);
    }

    void setPhysicsVelocityDamping(float damping) {
        phys_velocityDamping = damping;
    }
    void setPhysicsGasConstant(float c) { phys_gasConstant = c; }
    void setPhysicsViscosity(float v) { phys_viscosity = v; }
    void setPhysicsRestDensity(float d) { phys_restDensity = d; }
    void setPhysicsAirDensity(float d) { phys_airDensity = d; }
    void setPhysicsGasExpansionRate(float rate) { phys_gasExpansionRate = rate; }
    void setPhysicsMaxGasSize(float maxGas) { phys_maxGasSize = maxGas; }
    void setphys_gravityCenter(PointType n) {
        phys_gravityCenter = n;
    }
private:

    float lodFalloffRate_ = 0.1f; // Lower = better, higher = worse. 0-1
    float invLodf = 1 / lodFalloffRate_;
    float lodMinDistance_ = 100.0f;
    float maxDistance_ = lodMinDistance_ * lodMinDistance_;
    float keepDistance_ = maxDistance_ * 1.2;
    
    struct Ray {
        PointType origin;
        PointType dir;
        PointType invDir;
        uint8_t sign[3];
        uint8_t signMask;
        Ray(const PointType& orig, const PointType& dir) : origin(orig), dir(dir) {
            invDir = dir.cwiseInverse();
            sign[0] = (invDir[0] < 0);
            sign[1] = (invDir[1] < 0);
            sign[2] = (invDir[2] < 0);
            signMask = (sign[0] | sign[1] << 1 | sign[2] << 2);
        }
    };

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
        // uint8_t octant = 0;
        // if (point[0] >= center[0]) octant |= 1;
        // if (point[1] >= center[1]) octant |= 2;
        // if (point[2] >= center[2]) octant |= 4;
        // return octant;
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint8_t octant) const {
        PointType childMin, childMax;
        const PointType& center = node->center;
        
        childMin[0] = (octant & 1) ? center[0] : node->bounds.first[0];
        childMax[0] = (octant & 1) ? node->bounds.second[0] : center[0];
        
        childMin[1] = (octant & 2) ? center[1] : node->bounds.first[1];
        childMax[1] = (octant & 2) ? node->bounds.second[1] : center[1];
        
        childMin[2] = (octant & 4) ? center[2] : node->bounds.first[2];
        childMax[2] = (octant & 4) ? node->bounds.second[2] : center[2];

        return {childMin, childMax};
    }

    bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const {
        return (a.first[0] <= b.second[0] && a.second[0] >= b.first[0] &&
                a.first[1] <= b.second[1] && a.second[1] >= b.first[1] &&
                a.first[2] <= b.second[2] && a.second[2] >= b.first[2]);
    }

    bool boxContainsBox(const BoundingBox& outer, const BoundingBox& inner) const {
        return (inner.first[0] >= outer.first[0] && inner.second[0] <= outer.second[0] &&
                inner.first[1] >= outer.first[1] && inner.second[1] <= outer.second[1] &&
                inner.first[2] >= outer.first[2] && inner.second[2] <= outer.second[2]);
    }

    void splitNode(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
            node->children[i]->gasState = node->gasState;
        }

        std::vector<std::shared_ptr<NodeData>> keep;
        for (const auto& pointData : node->points) {
            BoundingBox cubeBounds = pointData->getCubeBounds();
            bool placedInChild = false;
            for (int i = 0; i < 8; ++i) {
                if (boxContainsBox(node->children[i]->bounds, cubeBounds)) {
                    node->children[i]->points.emplace_back(pointData);
                    placedInChild = true;
                    break;
                }
            }
            if (!placedInChild) {
                keep.emplace_back(pointData);
            }
        }

        node->points = std::move(keep);
        node->setLeaf(false);

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        ensureLoaded(node);

        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxIntersectsBox(node->bounds, cubeBounds)) return false;

        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        if (node->isLeaf()) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            node->setDirty(true);
            return true;
        } else {
            bool insertedInChild = false;
            OctreeNode* targetChild = nullptr;
            
            {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                for (int i = 0; i < 8; ++i) {
                    BoundingBox childBounds = createChildBounds(node, i);
                    if (boxContainsBox(childBounds, cubeBounds)) {
                        if (!node->children[i]) {
                            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
                            node->children[i]->gasState = node->gasState;
                        }
                        targetChild = node->children[i].get();
                        break;
                    }
                }
            }
            
            if (targetChild) {
                insertedInChild = insertRecursive(targetChild, pointData, depth + 1);
            }
            
            if (!insertedInChild) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                node->points.emplace_back(pointData);
                node->setDirty(true);
            }
            return true;
        }
    }

    OctreeNode* ensureNodeAtDepthRecursive(OctreeNode* node, const PointType& pos, size_t depth, size_t targetDepth) {
        ensureLoaded(node);
        if (depth >= targetDepth || depth >= maxDepth) return node;
        
        if (node->isLeaf()) {
            splitNode(node, depth);
        }
        
        int octant = getOctant(pos, node->center);
        if (!node->children[octant]) {
            BoundingBox childBounds = createChildBounds(node, octant);
            node->children[octant] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
            node->children[octant]->gasState = node->gasState;
        }
        return ensureNodeAtDepthRecursive(node->children[octant].get(), pos, depth + 1, targetDepth);
    }

    OctreeNode* ensureNodeAtDepth(const PointType& pos, size_t targetDepth) {
        if (!root_) return nullptr;
        if (!root_->contains(pos)) return nullptr;
        return ensureNodeAtDepthRecursive(root_.get(), pos, 0, targetDepth);
    }

    bool invalidateNodeLODRecursive(OctreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        ensureLoaded(node);
        
        std::array<OctreeNode*, 8> safeChildren = {nullptr};
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
            if (!node->isLeaf()) {
                for(int i = 0; i < 8; ++i) safeChildren[i] = node->children[i].get();
            }
        }
        
        for (int i = 0; i < 8; ++i) {
            if (safeChildren[i]) {
                invalidateNodeLODRecursive(safeChildren[i], bounds);
            }
        }
        return true;
    }

    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        if (root_ && pointData) {
            invalidateNodeLODRecursive(root_.get(), pointData->getCubeBounds());
        }
    }
    
    void ensureBounds(const BoundingBox& targetBounds) {
        if (!targetBounds.first.allFinite() || !targetBounds.second.allFinite()) return;

        if (!root_) {
            PointType center = (targetBounds.first + targetBounds.second) * 0.5f;
            PointType size = targetBounds.second - targetBounds.first;
            float maxDim = size.maxCoeff();
            if (maxDim <= 0.0f) maxDim = 1.0f;
            PointType halfSize = PointType::Constant(maxDim * 0.5f);
            root_ = std::make_unique<OctreeNode>(center - halfSize, center + halfSize);
            return;
        }

        int maxExpansions = 100;
        int expansionCount = 0;

        while (true) {
            if (expansionCount++ > maxExpansions) {
                std::cerr << "[Octree] WARNING: Max bounds expansion reached. Particle escaped or NaN." << std::endl;
                break;
            }

            bool xInside = root_->bounds.first.x() <= targetBounds.first.x() && root_->bounds.second.x() >= targetBounds.second.x();
            bool yInside = root_->bounds.first.y() <= targetBounds.first.y() && root_->bounds.second.y() >= targetBounds.second.y();
            bool zInside = root_->bounds.first.z() <= targetBounds.first.z() && root_->bounds.second.z() >= targetBounds.second.z();

            if (xInside && yInside && zInside) {
                break;
            }

            PointType min = root_->bounds.first;
            PointType max = root_->bounds.second;
            PointType size = max - min;
            
            if (size.x() <= 0.0f) size.x() = 1.0f;
            if (size.y() <= 0.0f) size.y() = 1.0f;
            if (size.z() <= 0.0f) size.z() = 1.0f;
            
            int expandX = (targetBounds.first.x() < min.x()) ? -1 : 1;
            int expandY = (targetBounds.first.y() < min.y()) ? -1 : 1;
            int expandZ = (targetBounds.first.z() < min.z()) ? -1 : 1;
            
            PointType newMin = min;
            PointType newMax = max;
            
            if (expandX < 0) newMin.x() -= size.x();
            else newMax.x() += size.x();
            if (expandY < 0) newMin.y() -= size.y();
            else newMax.y() += size.y();
            if (expandZ < 0) newMin.z() -= size.z();
            else newMax.z() += size.z();
            
            auto newRoot = std::make_unique<OctreeNode>(newMin, newMax);
            newRoot->setLeaf(false);
            
            uint8_t oldOctant = 0;
            if (expandX < 0) oldOctant |= 1;
            if (expandY < 0) oldOctant |= 2;
            if (expandZ < 0) oldOctant |= 4;
            
            newRoot->children[oldOctant] = std::move(root_);
            root_ = std::move(newRoot);
            
            maxDepth++; 
        }
    }

    void ensureLOD(OctreeNode* node) {
        ensureLoaded(node);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        
        bool requiresVoxelLOD = (node->lodData == nullptr);

        Eigen::Vector3f avgPos = Eigen::Vector3f::Zero();
        Eigen::Vector3f avgColor = Eigen::Vector3f::Zero();
        float avgEmittance = 0.0;
        float avgRoughness = 0.0;
        float avgMetallic = 0.0;
        float avgTransmission = 0.0;
        float avgIor = 0.0;
        float totalVolume = 0.0;
        int count = 0;

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!requiresVoxelLOD) return;
            if (!item || !item->isActive() || !item->isVisible()) return;
            float v = item->size * item->size * item->size;
            if (v <= 0.0) return;

            totalVolume += v;
            avgPos += item->position * v;
            avgColor += item->color * v;
            
            auto obj = getObject(item->objectId);
            Material mat = obj ? obj->getRenderMaterial(item->renderMatIdx) : Material();
            
            avgEmittance += mat.emittance * v;
            avgRoughness += mat.roughness * v;
            avgMetallic += mat.metallic * v;
            avgTransmission += mat.transmission * v;
            avgIor += mat.ior * v;
            count++;
        };

        if (requiresVoxelLOD) {
            if (node->isLeaf()) {
                if (node->points.empty()) {
                    auto lod = std::make_shared<NodeData>();
                    lod->position = node->center;
                    node->lodData = lod;
                } else if (node->points.size() == 1) {
                    const auto& pt = node->points[0];
                    if (pt->isActive() && pt->isVisible()) {
                        double v = static_cast<double>(pt->size) * pt->size * pt->size;
                        if (v > static_cast<double>(minLodVolume_)) {
                            node->lodData = pt;
                        } else {
                            auto lod = std::make_shared<NodeData>();
                            lod->position = node->center;
                            node->lodData = lod;
                        }
                    } else {
                        auto lod = std::make_shared<NodeData>();
                        lod->position = node->center;
                        node->lodData = lod;
                    }
                }
            }
            
            if (node->isLeaf() && node->lodData != nullptr) {
                // Done with leaf voxel LOD
            } else {
                for(const auto& pt : node->points) accumulate(pt);
            }
        }

        for (const auto& child : node->children) {
            if (child) {
                ensureLOD(child.get());
                if (requiresVoxelLOD && child->lodData) {
                    accumulate(child->lodData);
                }
            }
        }

        if (!node->isLeaf()) {
            float totalGasVol = 0.0f;
            float weightedDensity = 0.0f;
            Eigen::Vector3f weightedVel = Eigen::Vector3f::Zero();
            float weightedPressure = 0.0f;
            
            float maxDensity = -1.0f;
            GasT bestGasData{};

            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    auto& cGas = node->children[i]->gasState;
                    float v = node->children[i]->nodeSize;
                    float vol = v * v * v;
                    totalGasVol += vol;
                    
                    weightedDensity += cGas.density * vol;
                    weightedVel += cGas.velocity * (cGas.density * vol);
                    weightedPressure += cGas.pressure * vol;
                    
                    if (cGas.density > maxDensity) {
                        maxDensity = cGas.density;
                        bestGasData = cGas.data;
                    }
                }
            }
            
            if (totalGasVol > 0.0f) {
                node->gasState.density = weightedDensity / totalGasVol;
                if (weightedDensity > 0.0f) {
                    node->gasState.velocity = weightedVel / weightedDensity;
                } else {
                    node->gasState.velocity = Eigen::Vector3f::Zero();
                }
                node->gasState.pressure = weightedPressure / totalGasVol;
                node->gasState.data = bestGasData;
            }
        }

        if (requiresVoxelLOD && !node->isLeaf() && (node->lodData == nullptr)) {
            if (count > 0 && totalVolume > minLodVolume_) {
                double invVol = 1.0 / totalVolume;
                
                auto lod = std::make_shared<NodeData>();
                lod->position = (avgPos * invVol);
                lod->size = std::cbrt(totalVolume);

                lod->color = (avgColor * invVol);
                Material avgMat(avgEmittance * invVol, avgRoughness * invVol, avgMetallic * invVol,
                                avgTransmission * invVol, avgIor * invVol);
                
                auto obj = getOrCreateObject(-1);
                lod->renderMatIdx = obj->getOrAddRenderMaterial(avgMat);
                
                lod->setActive(true);
                lod->setVisible(true);
                lod->objectId = -1; 
                node->lodData = lod;
            } else {
                auto lod = std::make_shared<NodeData>();
                lod->position = node->center;
                node->lodData = lod;
            }
        }
    }

    void loadSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLoaded(node, true);
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                loadSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void loadAndLodSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLOD(node);
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                loadAndLodSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void updateStreamingRecursive(OctreeNode* node, const PointType& camPos, const PointType& camDir) {
        if (!node) return;
        
        float minDistSq = 0.0f;
        float maxDistSq = 0.0f;

        // if (!node->contains(camPos)) {
        for(int i = 0; i < Dim; ++i) {
            float v = camPos[i];
            float minBound = node->bounds.first[i];
            float maxBound = node->bounds.second[i];
            
            float d1 = v - minBound;
            float d2 = v - maxBound;
            
            if(v < minBound) {
                minDistSq += d1 * d1;
            } else if(v > maxBound) {
                minDistSq += d2 * d2;
            }
            
            float maxD = std::max(std::abs(d1), std::abs(d2));
            maxDistSq += maxD * maxD;
        }

        bool isBehind = false;
        PointType maxPoint;
        maxPoint.x() = (camDir.x() >= 0) ? node->bounds.second.x() : node->bounds.first.x();
        maxPoint.y() = (camDir.y() >= 0) ? node->bounds.second.y() : node->bounds.first.y();
        maxPoint.z() = (camDir.z() >= 0) ? node->bounds.second.z() : node->bounds.first.z();
        
        if ((maxPoint - camPos).dot(camDir) < -0.05f) {
            isBehind = true;
        }
        
        float lodDistSq = lodMinDistance_ * lodMinDistance_;
        float maxDistSq_max = maxDistance_ * maxDistance_;
        float keepDistSq = keepDistance_ * keepDistance_;
        
        if (maxDistSq <= lodDistSq) {
            loadSubtreeRecursive(node);
            return;
        }

        if (maxDistSq <= maxDistSq_max && minDistSq > lodDistSq) {
            loadAndLodSubtreeRecursive(node);
            return;
        }
        
        if (minDistSq > keepDistSq) {
            if (!node->isLoaded()) return;
            size_t subPoints = node->getSubtreePointCount();
            bool fullyLoaded = node->isSubtreeFullyLoaded();

            if ((subPoints > regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
                if (subPoints > 0) lazilyOffload(node);
                return;
            }
            if (!node->isLeaf()){
                for (int i = 0; i < 8; ++i) {
                    updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                }
            }
            return;
        }

        if (minDistSq > lodDistSq) {
            ensureLOD(node);
        } else {
            ensureLoaded(node, true);
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                }
            }
        }
    }

    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, const PointType& pos, float tolerance) {
        if (!node->contains(pos)) return nullptr;
        ensureLoaded(node, false);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        
        for (const auto& pointData : node->points) {
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        if (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                return findRecursive(node->children[octant].get(), pos, tolerance);
            }
        }
        return nullptr;
    }

    bool removeRecursive(OctreeNode* node, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        ensureLoaded(node, false);
        bool foundAny = false;
        
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            
            auto it = std::remove_if(node->points.begin(), node->points.end(),
                [&](const std::shared_ptr<NodeData>& pointData) {
                    return pointData == targetPt;
                });
            
            if (it != node->points.end()) {
                node->points.erase(it, node->points.end());
                foundAny = true;
            }
            if (foundAny) {
                node->lodData = nullptr; 
                node->setDirty(true);
            }
        }
        if (!node->isLeaf()) {
            std::array<OctreeNode*, 8> safeChildren = {nullptr};
            {
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                for (int i = 0; i < 8; ++i) safeChildren[i] = node->children[i].get();
            }

            for (int i = 0; i < 8; ++i) {
                if (safeChildren[i]) {
                    foundAny |= removeRecursive(safeChildren[i], bounds, targetPt);
                }
            }
            if (foundAny) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                node->lodData = nullptr;
                node->setDirty(true);
            }
        }
        return foundAny;
    }

    void searchNodeRecursive(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) {
        PointType closestPoint;
        for (int i = 0; i < Dim; ++i) {
            closestPoint[i] = std::max(node->bounds.first[i], std::min(center[i], node->bounds.second[i]));
        }
        
        float distSq = (closestPoint - center).squaredNorm();
        if (distSq > radiusSq) return;
        
        ensureLoaded(node, false);
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid == -1)) {
                if (seen.insert(pointData).second) results.emplace_back(pointData);
            }
        }
        
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) searchNodeRecursive(child.get(), center, radiusSq, objectid, results, seen);
            }
        }
    }

    void searchNode(OctreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results) {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        searchNodeRecursive(node, center, radiusSq, objectid, results, seen);
    }
    
    void clearNode(OctreeNode* node) {
        if (!node) return;
        
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        node->gasState = EulerianGasState();
        
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                clearNode(node->children[i].get());
                node->children[i].reset(nullptr);
            }
        }
        
        node->setLeaf(true);
    }

    void printStatsRecursive(const OctreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes, size_t& unloaded) const {
        if (!node) return;
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (!node->isLoaded()) {
            unloaded++;
            return;
        }

        if (node->lodData) lodGeneratedNodes++;
        
        size_t pts = node->points.size();
        actualPoints += pts;

        if (node->isLeaf()) {
            leafNodes++;
            maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
            minPointsInLeaf = std::min(minPointsInLeaf, pts);
        } else {
            for (const auto& child : node->children) {
                printStatsRecursive(child.get(), depth + 1, totalNodes, leafNodes, actualPoints, 
                                    maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes, unloaded);
            }
        }
    }

    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;
        if (!node->isLoaded()) return; 

        if (node->isLeaf()) {
            return;
        }

        std::array<OctreeNode*, 8> safeChildren = {nullptr};
        {
            std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
            for (int i = 0; i < 8; ++i) safeChildren[i] = node->children[i].get();
        }

        for (int i = 0; i < 8; ++i) {
            if (safeChildren[i]) {
                optimizeRecursive(safeChildren[i]);
            }
        }

        bool childrenAreLeaves = true;
        {
            std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
            for (int i = 0; i < 8; ++i) {
                if (node->children[i] && !node->children[i]->isLeaf()) {
                    childrenAreLeaves = false;
                    break;
                }
            }
        }

        if (childrenAreLeaves) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            bool stillLeaves = true;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i] && !node->children[i]->isLeaf()) {
                    stillLeaves = false;
                    break;
                }
            }
            
            if (stillLeaves) {
                std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
                for (int i = 0; i < 8; ++i) {
                    if (node->children[i]) {
                        std::shared_lock<std::shared_mutex> childLock(node->children[i]->nodeMutex);
                        allPoints.insert(allPoints.end(), node->children[i]->points.begin(), node->children[i]->points.end());
                    }
                }

                std::sort(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                    return a.get() < b.get();
                });
                allPoints.erase(std::unique(allPoints.begin(), allPoints.end(), [](const std::shared_ptr<NodeData>& a, const std::shared_ptr<NodeData>& b) {
                    return a.get() == b.get();
                }), allPoints.end());

                if (allPoints.size() <= maxPointsPerNode) {
                    node->points = std::move(allPoints);
                    for (int i = 0; i < 8; ++i) {
                        node->children[i].reset(nullptr);
                    }
                    node->setLeaf(true);
                    node->setDirty(true);
                    
                    node->lodData = nullptr;
                }
            }
        }
    }

    void offloadRecursive(OctreeNode* node) {
        if (!node->isLoaded()) return;
        
        size_t subPoints = node->getSubtreePointCount();
        bool fullyLoaded = node->isSubtreeFullyLoaded();
        
        if (subPoints > 0 && (subPoints <= regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
            if (node->isDirty()) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                node->saveRegion();
            }
            node->offload();
            return;
        }

        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) offloadRecursive(node->children[i].get());
            }
        }
    }

    bool rayBoxIntersect(const Ray& ray, const BoundingBox& box, float& tMin, float& tMax) const {
        float tx1 = (box.first[0] - ray.origin[0]) * ray.invDir[0];
        float tx2 = (box.second[0] - ray.origin[0]) * ray.invDir[0];

        tMin = std::min(tx1, tx2);
        tMax = std::max(tx1, tx2);

        float ty1 = (box.first[1] - ray.origin[1]) * ray.invDir[1];
        float ty2 = (box.second[1] - ray.origin[1]) * ray.invDir[1];

        tMin = std::max(tMin, std::min(ty1, ty2));
        tMax = std::min(tMax, std::max(ty1, ty2));

        float tz1 = (box.first[2] - ray.origin[2]) * ray.invDir[2];
        float tz2 = (box.second[2] - ray.origin[2]) * ray.invDir[2];

        tMin = std::max(tMin, std::min(tz1, tz2));
        tMax = std::min(tMax, std::max(tz1, tz2));

        return tMax >= std::max(0.0f, tMin);
    }

    bool rayCubeIntersect(const Ray& ray, const RenderData* cube, float& t, PointType& normal, PointType& hitPoint, float* tExit = nullptr) const {
        float t0x = (cube->boundsMin[0] - ray.origin[0]) * ray.invDir[0];
        float t1x = (cube->boundsMax[0] - ray.origin[0]) * ray.invDir[0];
        if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

        float t0y = (cube->boundsMin[1] - ray.origin[1]) * ray.invDir[1];
        float t1y = (cube->boundsMax[1] - ray.origin[1]) * ray.invDir[1];
        if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

        float t0z = (cube->boundsMin[2] - ray.origin[2]) * ray.invDir[2];
        float t1z = (cube->boundsMax[2] - ray.origin[2]) * ray.invDir[2];
        if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

        float tMin = std::max({t0x, t0y, t0z});
        float tMax = std::min({t1x, t1y, t1z});

        if (tExit) *tExit = tMax;

        if (tMax < std::max(0.0f, tMin) || tMax < 0.0f) {
            return false;
        }

        t = tMin < 0.0f ? tMax : tMin;
        
        hitPoint = ray.origin + ray.dir * t;
        
        PointType dMin = (hitPoint - cube->boundsMin).cwiseAbs();
        PointType dMax = (hitPoint - cube->boundsMax).cwiseAbs();
        
        float minDist = std::numeric_limits<float>::max();
        int minAxis = 0;
        float sign = 1.0f;

        for (int i = 0; i < Dim; ++i) {
            if (dMin[i] < minDist) {
                minDist = dMin[i];
                minAxis = i;
                sign = -1.0f;
            }
            if (dMax[i] < minDist) {
                minDist = dMax[i];
                minAxis = i;
                sign = 1.0f;
            }
        }
        
        normal = PointType::Zero();
        normal[minAxis] = sign;
        return true;
    }

    void collectNodesByObjectIdRecursive(OctreeNode* node, int id,
          std::vector<std::shared_ptr<NodeData>>& results,
          std::unordered_set<std::shared_ptr<NodeData>>& seen) {
        if (!node) return;
        ensureLoaded(node);
        
        for (const auto& pt : node->points) {
            if (pt->isActive() && (id == -1 || pt->objectId == id)) {
                if (seen.insert(pt).second) {
                    results.push_back(pt);
                }
            }
        }
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) {
                    collectNodesByObjectIdRecursive(child.get(), id, results, seen);
                }
            }
        }
    }

    void collectNodesByObjectId(OctreeNode* node, int id, std::vector<std::shared_ptr<NodeData>>& results) {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        collectNodesByObjectIdRecursive(node, id, results, seen);
    }

    bool raycastRecursive(OctreeNode* node, const Ray& ray, float tMin, float tMax, float& maxDist, RayHit& hit, const std::shared_ptr<NodeData>& ignoreNode, bool hitOnlySolid, bool resolvePenetration, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects) {
        if (!node->isLoaded()) {
            ensureLoaded(node, true);
            return false;
        }

        auto getObj = [&](int id) -> std::shared_ptr<GridObject> {
            auto it = localObjects.find(id);
            return it != localObjects.end() ? it->second : nullptr;
        };

        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        bool hitSomething = false;

        for (const auto& pt : node->points) {
            if (!pt->isActive() || pt == ignoreNode) continue;
            if (hitOnlySolid) {
                auto obj = getObj(pt->objectId);
                BodyType bType = obj ? obj->getPhysicsMaterial(pt->physMatIdx).type : BodyType::STATIC;
                if (bType == BodyType::FLUID) continue;
            }
            
            BoundingBox bounds = pt->getCubeBounds();
            
            float t0x = (bounds.first[0] - ray.origin[0]) * ray.invDir[0];
            float t1x = (bounds.second[0] - ray.origin[0]) * ray.invDir[0];
            if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

            float t0y = (bounds.first[1] - ray.origin[1]) * ray.invDir[1];
            float t1y = (bounds.second[1] - ray.origin[1]) * ray.invDir[1];
            if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

            float tMinPt = std::max(t0x, t0y);
            float tMaxPt = std::min(t1x, t1y);

            float t0z = (bounds.first[2] - ray.origin[2]) * ray.invDir[2];
            float t1z = (bounds.second[2] - ray.origin[2]) * ray.invDir[2];
            if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

            tMinPt = std::max(tMinPt, t0z);
            tMaxPt = std::min(tMaxPt, t1z);

            if (tMaxPt >= std::max(0.0f, tMinPt) && tMaxPt >= 0.0f) {
                float t = tMinPt < 0.0f ? (resolvePenetration ? 0.0f : tMaxPt) : tMinPt;
                if (t >= 0 && t <= maxDist && t <= tMax + 0.001f) {
                    maxDist = t;
                    hit.node = pt;
                    hit.distance = t;
                    hitSomething = true;
                    
                    hit.hitPoint = ray.origin + ray.dir * t;
                    PointType dMin = (hit.hitPoint - bounds.first).cwiseAbs();
                    PointType dMax = (hit.hitPoint - bounds.second).cwiseAbs();
                    float minDist = std::numeric_limits<float>::max();
                    int minAxis = 0;
                    float sign = 1.0f;
                    
                    for (int i = 0; i < Dim; ++i) {
                        if (dMin[i] < minDist) {
                            minDist = dMin[i];
                            minAxis = i;
                            sign = -1.0f;
                        }
                        if (dMax[i] < minDist) {
                            minDist = dMax[i];
                            minAxis = i;
                            sign = 1.0f;
                        }
                    }
                    hit.normal = PointType::Zero();
                    hit.normal[minAxis] = sign;
                }
            }
        }

        if (node->isLeaf()) return hitSomething;

        Eigen::Vector3f ttt = (node->center - ray.origin).cwiseProduct(ray.invDir);
        int currIdx = ((tMin >= ttt.x()) ? 1 : 0) | ((tMin >= ttt.y()) ? 2 : 0) | ((tMin >= ttt.z()) ? 4 : 0);

        while(tMin < tMax && tMin <= maxDist) {
            Eigen::Vector3f next_t;
            next_t[0] = (currIdx & 1) ? tMax : ttt[0];
            next_t[1] = (currIdx & 2) ? tMax : ttt[1];
            next_t[2] = (currIdx & 4) ? tMax : ttt[2];

            float tNext = next_t.minCoeff();
            int physIdx = currIdx ^ ray.signMask;

            if (node->children[physIdx]) {
                if (raycastRecursive(node->children[physIdx].get(), ray, tMin, tNext, maxDist, hit, ignoreNode, hitOnlySolid, resolvePenetration, localObjects)) {
                    hitSomething = true;
                }
            }

            tMin = tNext;
            currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
        }
        
        return hitSomething;
    }

    void buildRender(RenderBuffer_<T, IndexType, StoragePath>& buffer);
    void buildRenderNodeAt(OctreeNode* node, RenderBuffer_<T, IndexType, StoragePath>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects);
    const RenderData* fastVoxelTraverse(const RenderBuffer_<T, IndexType, StoragePath>& buffer, const Ray& ray, float maxDist);
public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16) :
            root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
            maxDepth(maxDepth), size(0), skybox_(1024, 1024),
            streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), skybox_(1024, 1024), streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    ~Octree() {
        stopWorkerThread();
    }
    
    Octree(const Octree& other) : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(other.skylight_), backgroundColor_(other.backgroundColor_), autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(other.skybox_), regionTargetPoints_(other.regionTargetPoints_),
            minLodSize_(other.minLodSize_), minLodVolume_(other.minLodVolume_) {
        if (other.root_) root_ = other.root_->clone();
        
        {
            std::shared_lock<std::shared_mutex> lockOther(other.objectsMutex_);
            std::unique_lock<std::shared_mutex> lockThis(objectsMutex_);
            for (const auto& pair : other.objects_) {
                objects_[pair.first] = std::make_shared<GridObject>(*pair.second);
            }
        }
        startWorkerThread();
    }

    Octree(Octree&& other) noexcept : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(std::move(other.skylight_)), backgroundColor_(std::move(other.backgroundColor_)),
            autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(std::move(other.skybox_)), regionTargetPoints_(other.regionTargetPoints_),
            minLodSize_(other.minLodSize_), minLodVolume_(other.minLodVolume_) {
        other.stopWorkerThread();
        root_ = std::move(other.root_);
        
        {
            std::unique_lock<std::shared_mutex> lockOther(other.objectsMutex_);
            std::unique_lock<std::shared_mutex> lockThis(objectsMutex_);
            objects_ = std::move(other.objects_);
        }
        
        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
    }

    Octree& operator=(const Octree& other) {
        if (this == &other) return *this;
        
        stopWorkerThread();
        clear();
        
        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        skylight_ = other.skylight_;
        backgroundColor_ = other.backgroundColor_;
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        skybox_ = other.skybox_;
        regionTargetPoints_ = other.regionTargetPoints_;
        minLodSize_ = other.minLodSize_;
        minLodVolume_ = other.minLodVolume_;

        if (other.root_) root_ = other.root_->clone();
        
        {
            std::shared_lock<std::shared_mutex> lockOther(other.objectsMutex_);
            std::unique_lock<std::shared_mutex> lockThis(objectsMutex_);
            objects_.clear();
            for (const auto& pair : other.objects_) {
                objects_[pair.first] = std::make_shared<GridObject>(*pair.second);
            }
        }

        startWorkerThread();
        return *this;
    }

    Octree& operator=(Octree&& other) noexcept {
        if (this == &other) return *this;

        stopWorkerThread();
        other.stopWorkerThread();

        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        skylight_ = std::move(other.skylight_);
        backgroundColor_ = std::move(other.backgroundColor_);
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        skybox_ = std::move(other.skybox_);
        regionTargetPoints_ = std::move(other.regionTargetPoints_);
        minLodSize_ = other.minLodSize_;
        minLodVolume_ = other.minLodVolume_;
        
        root_ = std::move(other.root_);
        
        {
            std::unique_lock<std::shared_mutex> lockOther(other.objectsMutex_);
            std::unique_lock<std::shared_mutex> lockThis(objectsMutex_);
            objects_ = std::move(other.objects_);
        }

        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
        return *this;
    }

    void enqueueTask(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(taskMutex_);
            taskQueue_.push(std::move(task));
        }
        taskCV_.notify_one();
    }

    void offloadRegions() {
        if (root_) offloadRecursive(root_.get());
    }

    void setAutoOptimize(bool v) { 
        autoOptimize_.store(v); 
    }

    void setSkylight(const Eigen::Vector3f& skylight) { 
        skylight_ = skylight; 
    }

    Eigen::Vector3f getSkylight() const { 
        return skylight_; 
    }

    void setBackgroundColor(const Eigen::Vector3f& color) { 
        backgroundColor_ = color; 
        skybox_.setBackground(color.x(), color.y(), color.z(), 1.0f);
    }

    Eigen::Vector3f getBackgroundColor() const { 
        return backgroundColor_; 
    }

    void setLODFalloff(float rate) {
        lodFalloffRate_ = rate;
        invLodf = 1 / rate;
    }
    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }
    void setMaxDistance(float dist) {
        maxDistance_ = dist;
        keepDistance_ = dist * 1.2;
    }
    void setMinLODSize(float size) {
        minLodSize_ = size;
        minLodVolume_ = size * size * size;
    }
    float getMinLODSize() const { return minLodSize_; }
    void setRegionTargetPoints(size_t points) { regionTargetPoints_ = points; }
    size_t getRegionTargetPoints() const { return regionTargetPoints_; }

    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, 
             float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f, Eigen::Vector3f absorp = Eigen::Vector3f::Zero(),
             BodyType bType = BodyType::STATIC, float mass = 1.0f) {
        
        auto obj = getOrCreateObject(objectId);
        Material mat(emittance, roughness, metallic, transmission, ior, absorp);
        uint16_t rIdx = obj->getOrAddRenderMaterial(mat);
        
        PhysicsMaterial_ pmat{bType, mass};
        uint16_t pIdx = obj->getOrAddPhysicsMaterial(pmat);

        auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId, rIdx, pIdx, bType == BodyType::STATIC);
        
        PointType relPos = pos - obj->centerPosition;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            obj->relativeVoxels.push_back({relPos, rIdx, pIdx, size});
        }
        
        ensureBounds(pointData->getCubeBounds());
        
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            if (bType != BodyType::STATIC) {
                std::lock_guard<std::mutex> lock(physicsMutex_);
                activePhysicsNodes_.push_back(pointData);
            }
            return true;
        }
        return false;
    }

    bool setGas(const PointType& pos, const GasT& gasData, float density, const Eigen::Vector3f& velocity = Eigen::Vector3f::Zero(), float pressure = 0.0f) {
        ensureBounds({pos, pos});
        OctreeNode* node = ensureNodeAtDepth(pos, maxDepth);
        if (node) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->gasState.data = gasData;
            node->gasState.density = density;
            node->gasState.velocity = velocity;
            node->gasState.pressure = pressure;
            node->setDirty(true);
            return true;
        }
        return false;
    }

    EulerianGasState* getGas(const PointType& pos) {
        if (!root_ || !root_->contains(pos)) return nullptr;
        OctreeNode* node = root_.get();
        while (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                node = node->children[octant].get();
            } else {
                break;
            }
        }
        return &node->gasState;
    }

    void vaporizeObject(int objectId, float initialDensity = 1.0f) {
        if (!root_) return;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if (nodes.empty()) return;

        for (auto& pt : nodes) {
            OctreeNode* leaf = ensureNodeAtDepth(pt->position, maxDepth);
            if (leaf) {
                std::unique_lock<std::shared_mutex> lock(leaf->nodeMutex);
                if (leaf->gasState.density <= 0.001f) {
                    leaf->gasState.color = pt->color;
                    leaf->gasState.renderMatIdx = pt->renderMatIdx;
                    leaf->gasState.objectId = pt->objectId;
                    leaf->gasState.velocity = pt->physics.velocity;
                } else {
                    float w1 = leaf->gasState.density;
                    float w2 = initialDensity;
                    leaf->gasState.color = (leaf->gasState.color * w1 + pt->color * w2) / (w1 + w2);
                    leaf->gasState.velocity = (leaf->gasState.velocity * w1 + pt->physics.velocity * w2) / (w1 + w2);
                }
                leaf->gasState.density += initialDensity;
                leaf->setDirty(true);
            }
        }

        BoundingBox oldBounds = getNodesBounds(nodes);
        int startDepth = 0;
        OctreeNode* startNode = getHighestCommonNode(root_.get(), oldBounds, 0, startDepth);
        size_t removed = removeObjectBatchRecursive(startNode, objectId);
        size -= removed;

        generateLODs();
    }

    bool insertVoxels(int objectId, const std::vector<PointType>& positions, const T& defaultData, 
                      float voxelSize = 0.1f, const Eigen::Vector3f& color = {1.0f, 1.0f, 1.0f},
                      BodyType bType = BodyType::STATIC, float mass = 1.0f,
                      float emittance = 0.0f, float roughness = 1.0f, 
                      float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f,
                      Eigen::Vector3f absorp = Eigen::Vector3f::Zero()) {
        if (!root_) return false;
        auto obj = getOrCreateObject(objectId);
        Material mat(emittance, roughness, metallic, transmission, ior, absorp);
        uint16_t rIdx = obj->getOrAddRenderMaterial(mat);
        PhysicsMaterial_ pmat{bType, mass};
        uint16_t pIdx = obj->getOrAddPhysicsMaterial(pmat);

        std::vector<std::shared_ptr<NodeData>> newPhysNodes;
        bool allInserted = true;
        
        for (const auto& pos : positions) {
            auto pointData = std::make_shared<NodeData>(defaultData, pos, true, color, voxelSize, true, objectId, rIdx, pIdx, bType == BodyType::STATIC);
            
            PointType relPos = pos - obj->centerPosition;
            {
                std::unique_lock<std::shared_mutex> lock(obj->objMutex);
                obj->relativeVoxels.push_back({relPos, rIdx, pIdx, voxelSize});
            }
            
            ensureBounds(pointData->getCubeBounds());
            if (insertRecursive(root_.get(), pointData, 0)) {
                this->size++;
                if (bType != BodyType::STATIC) {
                    newPhysNodes.push_back(pointData);
                }
            } else {
                allInserted = false;
            }
        }
        
        if (!newPhysNodes.empty()) {
            std::lock_guard<std::mutex> lock(physicsMutex_);
            for (auto& n : newPhysNodes) {
                activePhysicsNodes_.push_back(n);
            }
        }
        
        return allInserted;
    }

    bool importOBJ(int objectId, const std::string& filepath, const T& defaultData, 
                   float voxelSize = 0.1f, const Eigen::Vector3f& color = {1.0f, 1.0f, 1.0f}) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        std::vector<PointType> positions;
        std::string line;
        while (std::getline(file, line)) {
            if (line.length() >= 2 && line[0] == 'v' && line[1] == ' ') {
                std::istringstream s(line.substr(2));
                PointType p;
                s >> p.x() >> p.y() >> p.z();
                
                p.x() = std::round(p.x() / voxelSize) * voxelSize;
                p.y() = std::round(p.y() / voxelSize) * voxelSize;
                p.z() = std::round(p.z() / voxelSize) * voxelSize;

                positions.push_back(p);
            }
        }
        
        auto ptLess = [](const PointType& a, const PointType& b) {
            if (a.x() != b.x()) return a.x() < b.x();
            if (a.y() != b.y()) return a.y() < b.y();
            return a.z() < b.z();
        };
        std::sort(positions.begin(), positions.end(), ptLess);
        positions.erase(std::unique(positions.begin(), positions.end(), 
            [](const PointType& a, const PointType& b) {
                return (a - b).squaredNorm() < 0.0001f;
            }), positions.end());

        return insertVoxels(objectId, positions, defaultData, voxelSize, color);
    }

    int getRenderMaterialIndex(const PointType& pos, float tolerance = 0.0001f) {
        auto pt = find(pos, tolerance);
        if (!pt) return -1;
        return pt->renderMatIdx;
    }

    int getPhysicsMaterialIndex(const PointType& pos, float tolerance = 0.0001f) {
        auto pt = find(pos, tolerance);
        if (!pt) return -1;
        return pt->physMatIdx;
    }

    bool updateRenderMaterial(int objectId, uint16_t index, const Material& mat) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            if (index >= obj->renderMaterials.size()) return false;
            obj->renderMaterials[index] = mat;
        }
        std::vector<std::shared_ptr<NodeData>> nodes;
        if (root_) collectNodesByObjectId(root_.get(), objectId, nodes);
        for (auto& n : nodes) invalidateLODForPoint(n);
        return true;
    }

    bool updatePhysicsMaterial(int objectId, uint16_t index, const PhysicsMaterial_& pmat) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            if (index >= obj->physicsMaterials.size()) return false;
            obj->physicsMaterials[index] = pmat;
        }
        markPhysicsCollidersDirty();
        return true;
    }

    bool removeObject(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        if (root_) collectNodesByObjectId(root_.get(), objectId, nodes);
        if (nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int startDepth = 0;
        OctreeNode* startNode = getHighestCommonNode(root_.get(), oldBounds, 0, startDepth);

        size_t removed = removeObjectBatchRecursive(startNode, objectId);
        size -= removed;

        {
            std::unique_lock<std::shared_mutex> lock(objectsMutex_);
            objects_.erase(objectId);
        }
        return true;
    }

    bool rotateObject(int objectId, const Eigen::Matrix3f& rotation, const PointType& pivot) {
        if (!root_) return false;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if (nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int oldDepth = 0;
        OctreeNode* oldStart = getHighestCommonNode(root_.get(), oldBounds, 0, oldDepth);

        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        for (auto& n : nodes) {
            PointType offset = n->position - pivot;
            n->position = pivot + (rotation * offset);
        }

        BoundingBox newBounds = getNodesBounds(nodes);
        ensureBounds(newBounds);

        int newDepth = 0;
        OctreeNode* newStart = getHighestCommonNode(root_.get(), newBounds, 0, newDepth);

        size_t added = 0;
        for (auto& n : nodes) {
            if (insertRecursive(newStart, n, newDepth)) {
                added++;
            }
        }
        size += added;
        
        auto obj = getObject(objectId);
        if (obj) {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            obj->centerPosition = pivot + (rotation * (obj->centerPosition - pivot));
            for (auto& rv : obj->relativeVoxels) {
                rv.relPos = rotation * rv.relPos;
            }
        }
        
        return true;
    }

    bool rotateObjectCenter(int objectId, const Eigen::Matrix3f& rotation) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        return rotateObject(objectId, rotation, obj->centerPosition);
    }

    bool subdivideObject(int objectId) {
        if (!root_) return false;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if (nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int oldDepth = 0;
        OctreeNode* oldStart = getHighestCommonNode(root_.get(), oldBounds, 0, oldDepth);

        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        size_t added = 0;
        for (auto& n : nodes) {
            float newSize = n->size * 0.5f;
            float offset = newSize * 0.5f;
            
            for (int i = 0; i < 8; ++i) {
                PointType newPos = n->position;
                newPos.x() += (i & 1) ? offset : -offset;
                newPos.y() += (i & 2) ? offset : -offset;
                newPos.z() += (i & 4) ? offset : -offset;

                auto childNode = std::make_shared<NodeData>(*n);
                childNode->position = newPos;
                childNode->size = newSize;

                if (insertRecursive(oldStart, childNode, oldDepth)) {
                    added++;
                }
            }
        }
        size += added;
        return true;
    }

    bool smoothObject(int objectId) {
        if (!subdivideObject(objectId)) return false;

        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);

        std::unordered_set<std::shared_ptr<NodeData>> toRemove;
        std::vector<std::shared_ptr<NodeData>> toRemoveVec;
        
        for (auto& n : nodes) {
            int exposedFaces = 0;
            float step = n->size;
            const std::array<PointType, 6> dirs = {
                PointType(1, 0, 0), PointType(-1, 0, 0),
                PointType(0, 1, 0), PointType(0, -1, 0),
                PointType(0, 0, 1), PointType(0, 0, -1)
            };
            
            for (const auto& dir : dirs) {
                auto neighbor = find(n->position + (dir * step), step * 0.25f);
                if (!neighbor || !neighbor->isActive() || neighbor->objectId != objectId) {
                    exposedFaces++;
                }
            }
            
            if (exposedFaces >= 3) {
                toRemove.insert(n);
                toRemoveVec.push_back(n);
            }
        }

        if (!toRemoveVec.empty()) {
            BoundingBox remBounds = getNodesBounds(toRemoveVec);
            int remDepth = 0;
            OctreeNode* start = getHighestCommonNode(root_.get(), remBounds, 0, remDepth);
            size_t removed = removeSpecificNodesBatchRecursive(start, toRemove);
            size -= removed;
        }
        
        return true;
    }

    void queuedset(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, float metallic = 0.0f, float transmission = 0.0f,
             float ior = 1.45f, Eigen::Vector3f absorp = Eigen::Vector3f::Zero(),
             BodyType bType = BodyType::STATIC, float mass = 1.0f) {
        enqueueTask([this, data, pos, visible, color, size, active, objectId, emittance, roughness, metallic, transmission, ior, absorp, bType, mass]() {
            auto obj = getOrCreateObject(objectId);
            Material mat(emittance, roughness, metallic, transmission, ior, absorp);
            uint16_t rIdx = obj->getOrAddRenderMaterial(mat);
            
            PhysicsMaterial_ pmat{bType, mass};
            uint16_t pIdx = obj->getOrAddPhysicsMaterial(pmat);

            auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId, rIdx, pIdx, bType == BodyType::STATIC);
            
            ensureBounds(pointData->getCubeBounds());
            
            if (insertRecursive(root_.get(), pointData, 0)) {
                this->size++;
                if (bType != BodyType::STATIC) {
                    std::lock_guard<std::mutex> lock(physicsMutex_);
                    activePhysicsNodes_.push_back(pointData);
                }
            }
        });
    }

    void updateStreaming(const Camera& cam) {
        if (streamingQueued_.exchange(true, std::memory_order_acquire)) return;
        PointType camPos = cam.origin;
        PointType camDir = cam.direction.normalized();
        enqueueTask([this, camPos, camDir]() {
            if (root_) {
                updateStreamingRecursive(root_.get(), camPos, camDir);
            }
            streamingQueued_.store(false, std::memory_order_release);
        });
    }

    bool save(const std::string& filename) {
        if (!root_) return false;

        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x79676733;
        OctreeNode::writeVal(out, magic);
        OctreeNode::writeVal(out, maxDepth);
        OctreeNode::writeVal(out, maxPointsPerNode);
        OctreeNode::writeVal(out, size);
        OctreeNode::writeVal(out, regionTargetPoints_);
        
        OctreeNode::writeVec3(out, skylight_);
        OctreeNode::writeVec3(out, backgroundColor_);
        
        OctreeNode::writeVec3(out, root_->bounds.first);
        OctreeNode::writeVec3(out, root_->bounds.second);

        {
            std::shared_lock<std::shared_mutex> lock(objectsMutex_);
            uint32_t numObjects = objects_.size();
            OctreeNode::writeVal(out, numObjects);
            for (const auto& pair : objects_) {
                OctreeNode::writeVal(out, pair.first);
                auto obj = pair.second;
                
                std::shared_lock<std::shared_mutex> objLock(obj->objMutex);
                OctreeNode::writeVal(out, obj->objectFlags);
                OctreeNode::writeVec3(out, obj->centerPosition);
                OctreeNode::writeVal(out, obj->maxGasVoxelSize);
                
                uint32_t numRMat = obj->renderMaterials.size();
                OctreeNode::writeVal(out, numRMat);
                for (const auto& mat : obj->renderMaterials) {
                    OctreeNode::writeVal(out, mat);
                }
                
                uint32_t numPMat = obj->physicsMaterials.size();
                OctreeNode::writeVal(out, numPMat);
                for (const auto& pmat : obj->physicsMaterials) {
                    OctreeNode::writeVal(out, pmat);
                }
            }
        }

        root_->serialize(out, regionTargetPoints_);
        
        out.close();
        std::cout << "successfully saved grid to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        uint32_t magic;
        OctreeNode::readVal(in, magic);
        if (magic != 0x79676733) {
            std::cerr << "Invalid Octree file format" << std::endl;
            return false;
        }

        OctreeNode::readVal(in, maxDepth);
        OctreeNode::readVal(in, maxPointsPerNode);
        OctreeNode::readVal(in, size);
        OctreeNode::readVal(in, regionTargetPoints_);
        
        OctreeNode::readVec3(in, skylight_);
        OctreeNode::readVec3(in, backgroundColor_);

        PointType minBound, maxBound;
        OctreeNode::readVec3(in, minBound);
        OctreeNode::readVec3(in, maxBound);

        {
            std::unique_lock<std::shared_mutex> lock(objectsMutex_);
            objects_.clear();
            uint32_t numObjects = 0;
            OctreeNode::readVal(in, numObjects);
            for (uint32_t i = 0; i < numObjects; ++i) {
                int id;
                OctreeNode::readVal(in, id);
                auto obj = std::make_shared<GridObject>(id);
                OctreeNode::readVal(in, obj->objectFlags);
                OctreeNode::readVec3(in, obj->centerPosition);
                OctreeNode::readVal(in, obj->maxGasVoxelSize);
                
                uint32_t numRMat;
                OctreeNode::readVal(in, numRMat);
                obj->renderMaterials.resize(numRMat);
                for (uint32_t j = 0; j < numRMat; ++j) {
                    OctreeNode::readVal(in, obj->renderMaterials[j]);
                }
                
                uint32_t numPMat;
                OctreeNode::readVal(in, numPMat);
                obj->physicsMaterials.resize(numPMat);
                for (uint32_t j = 0; j < numPMat; ++j) {
                    OctreeNode::readVal(in, obj->physicsMaterials[j]);
                }
                objects_[id] = obj;
            }
        }

        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        root_->deserialize(in, regionTargetPoints_);

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = EPSILON) {
        return findRecursive(root_.get(), pos, tolerance);
    }

    std::shared_ptr<NodeData> findwNode(const PointType& pos, OctreeNode* node, float tolerance = EPSILON) {
        // node = root_.get();
        return findRecursive(node, pos, tolerance);
    }

    bool inGrid(PointType pos) {
        return root_->contains(pos);
    }

    bool remove(const PointType& pos, float tolerance = EPSILON) {
        auto pt = find(pos, tolerance);
        if (!pt) return false;
        if (removeRecursive(root_.get(), pt->getCubeBounds(), pt)) {
            size--;
            return true;
        }
        return false;
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius, int objectid = -1) {
        std::vector<std::shared_ptr<NodeData>> results;
        if (!root_) return results;
        
        float radiusSq = radius * radius;
        searchNode(root_.get(), center, radiusSq, objectid, results);
        
        return results;
    }
    
    void makeObjectFluid(int objectId, float newMass, BodyType newType = BodyType::FLUID) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        
        auto obj = getOrCreateObject(objectId);
        PhysicsMaterial_ pmat{newType, newMass};
        uint16_t newIdx = obj->getOrAddPhysicsMaterial(pmat);

        std::lock_guard<std::mutex> lock(physicsMutex_);
        for (auto& n : nodes) {
            PhysicsMaterial_ oldPmat = obj->getPhysicsMaterial(n->physMatIdx);
            if (oldPmat.type == BodyType::STATIC && newType != BodyType::STATIC) {
                activePhysicsNodes_.push_back(n);
            }
            n->physMatIdx = newIdx;
        }
        physicsCollidersDirty_.store(true);
    }

    void markPhysicsCollidersDirty() {
        physicsCollidersDirty_.store(true);
    }

    std::vector<std::weak_ptr<NodeData>> getWeakNodesByObjectId(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        if (root_) collectNodesByObjectId(root_.get(), objectId, nodes);
        std::vector<std::weak_ptr<NodeData>> weakNodes;
        weakNodes.reserve(nodes.size());
        for (auto& n : nodes) weakNodes.push_back(n);
        return weakNodes;
    }

    bool update(const PointType& pos, const T& newData) {
        auto pointData = find(pos);
        if (!pointData) return false;
        else pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedupdate(const PointType pos, const T newData) {
        enqueueTask([this, pos, newData]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node);
            if (!pointData) return;
            else {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->data = newData;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool update(const PointType& oldPos, const PointType& newPos, const T& newData, bool newVisible = true, 
                Eigen::Vector3f newColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, float newEmittance = -1.0f, float newRoughness = -1.0f, 
                float newMetallic = -1.0f, float newTransmission = -1.0f, float newIor = -1.0f, float tolerance = EPSILON) {

        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;

        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        
        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->setVisible(newVisible);
        
        if (newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f)) pointData->color = newColor;
        if (newSize > 0) pointData->size = newSize;
        pointData->setActive(newActive);
        pointData->objectId = targetObjId;
        
        auto obj = getOrCreateObject(targetObjId);
        Material mat = obj->getRenderMaterial(pointData->renderMatIdx);
        
        if (newEmittance >= 0) mat.emittance = newEmittance;
        if (newRoughness >= 0) mat.roughness = newRoughness;
        if (newMetallic >= 0) mat.metallic = newMetallic;
        if (newTransmission >= 0) mat.transmission = newTransmission;
        if (newIor >= 0) mat.ior = newIor;
        
        pointData->renderMatIdx = obj->getOrAddRenderMaterial(mat);
        
        ensureBounds(pointData->getCubeBounds());
        bool res = insertRecursive(root_.get(), pointData, 0);
        
        if(!res) {
            size--;
        }

        return res;
    }

    bool move(const PointType& pos, const PointType& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        pointData->position = newPos;
        ensureBounds(pointData->getCubeBounds());

        if (insertRecursive(root_.get(), pointData, 0)) {
            return true;
        }
        size--;
        return false;
    }

    void queuedmove(const PointType pos, const PointType newPos) {
        enqueueTask([this, pos, newPos]() {
            auto pointData = find(pos);
            if (!pointData) return;

            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            pointData->position = newPos;
            ensureBounds(pointData->getCubeBounds());

            if (insertRecursive(root_.get(), pointData, 0)) {
                return;
            }
            size--;
            return;
        });
    }

    void queuedupdate(const PointType pos, const PointType newPos, const T newData) {
        enqueueTask([this, pos, newPos, newData]() {
            auto pointData = find(pos);
            if (!pointData) return;
            
            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            
            auto newPointData = std::make_shared<NodeData>(*pointData);
            newPointData->position = newPos;
            newPointData->data = newData;
            
            ensureBounds(newPointData->getCubeBounds());
            if (!insertRecursive(root_.get(), newPointData, 0)) {
                size--;
            }
        });
    }

    bool setObjectId(const PointType& pos, int objectId, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->objectId = objectId;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool updateData(const PointType& pos, const T& newData, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setActive(const PointType& pos, bool active, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->setActive(active);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setVisible(const PointType& pos, bool visible, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->setVisible(visible);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setColor(const PointType& pos, Eigen::Vector3f color, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->color = color;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedsetColor(const PointType& pos, Eigen::Vector3f color, float tolerance = EPSILON) {
        enqueueTask([this, pos, color, tolerance]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node, tolerance);
            if (!pointData) return;
            {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->color = color;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool setEmittance(const PointType& pos, float emittance, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        auto obj = getOrCreateObject(pointData->objectId);
        Material mat = obj->getRenderMaterial(pointData->renderMatIdx);
        mat.emittance = emittance;
        pointData->renderMatIdx = obj->getOrAddRenderMaterial(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setRoughness(const PointType& pos, float roughness, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        auto obj = getOrCreateObject(pointData->objectId);
        Material mat = obj->getRenderMaterial(pointData->renderMatIdx);
        mat.roughness = roughness;
        pointData->renderMatIdx = obj->getOrAddRenderMaterial(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setMetallic(const PointType& pos, float metallic, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        auto obj = getOrCreateObject(pointData->objectId);
        Material mat = obj->getRenderMaterial(pointData->renderMatIdx);
        mat.metallic = metallic;
        pointData->renderMatIdx = obj->getOrAddRenderMaterial(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setTransmission(const PointType& pos, float transmission, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        auto obj = getOrCreateObject(pointData->objectId);
        Material mat = obj->getRenderMaterial(pointData->renderMatIdx);
        mat.transmission = transmission;
        pointData->renderMatIdx = obj->getOrAddRenderMaterial(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    void setMaterialByObjectId(int objectId, float emittance, float roughness, float metallic) {
        auto obj = getOrCreateObject(objectId);
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            for (auto& mat : obj->renderMaterials) {
                mat.emittance = emittance;
                mat.roughness = roughness;
                mat.metallic = metallic;
            }
        }
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        for (auto& n : nodes) {
            invalidateLODForPoint(n);
        }
    }

    bool raycast(const PointType& origin, const PointType& direction, float maxDist, RayHit& hit,
                 const std::shared_ptr<NodeData>& ignoreNode = nullptr, bool hitOnlySolid = false, bool resolvePenetration = false) {
        if (!root_) return false;
        
        std::unordered_map<int, std::shared_ptr<GridObject>> localObjects;
        {
            std::shared_lock<std::shared_mutex> lock(objectsMutex_);
            localObjects = objects_;
        }
        auto getObj = [&](int id) -> std::shared_ptr<GridObject> {
            auto it = localObjects.find(id);
            return it != localObjects.end() ? it->second : nullptr;
        };
        
        Ray ray(origin, direction.normalized());
        
        float tMin, tMax;
        if (!rayBoxIntersect(ray, root_->bounds, tMin, tMax)) return false;
        tMax = std::min(tMax, maxDist);
        float currentMaxDist = maxDist;
        
        bool hitSomething = false;

        struct StackItem {
            OctreeNode* node;
            float tMin;
            float tMax;
        };
        StackItem stack[256];
        int stackPtr = 0;
        stack[stackPtr++] = {root_.get(), std::max(0.0f, tMin), tMax};

        while(stackPtr > 0) {
            StackItem current = stack[--stackPtr];
            if (current.tMin > currentMaxDist) continue;

            OctreeNode* node = current.node;

            if (!node->isLoaded()) {
                ensureLoaded(node, true);
                continue;
            }

            std::shared_lock<std::shared_mutex> lock(node->nodeMutex);

            for (const auto& pt : node->points) {
                if (!pt->isActive() || pt == ignoreNode) continue;
                if (hitOnlySolid) {
                    auto obj = getObj(pt->objectId);
                    BodyType bType = obj ? obj->getPhysicsMaterial(pt->physMatIdx).type : BodyType::STATIC;
                    if (bType == BodyType::FLUID) continue;
                }
                
                BoundingBox bounds = pt->getCubeBounds();
                
                float t0x = (bounds.first[0] - ray.origin[0]) * ray.invDir[0];
                float t1x = (bounds.second[0] - ray.origin[0]) * ray.invDir[0];
                if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

                float t0y = (bounds.first[1] - ray.origin[1]) * ray.invDir[1];
                float t1y = (bounds.second[1] - ray.origin[1]) * ray.invDir[1];
                if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

                float tMinPt = std::max(t0x, t0y);
                float tMaxPt = std::min(t1x, t1y);

                float t0z = (bounds.first[2] - ray.origin[2]) * ray.invDir[2];
                float t1z = (bounds.second[2] - ray.origin[2]) * ray.invDir[2];
                if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

                tMinPt = std::max(tMinPt, t0z);
                tMaxPt = std::min(tMaxPt, t1z);

                if (tMaxPt >= std::max(0.0f, tMinPt) && tMaxPt >= 0.0f) {
                    float t = tMinPt < 0.0f ? (resolvePenetration ? 0.0f : tMaxPt) : tMinPt;
                    if (t >= 0 && t <= currentMaxDist && t <= current.tMax + 0.001f) {
                        currentMaxDist = t;
                        hit.node = pt;
                        hit.distance = t;
                        hitSomething = true;
                        
                        hit.hitPoint = ray.origin + ray.dir * t;
                        PointType dMin = (hit.hitPoint - bounds.first).cwiseAbs();
                        PointType dMax = (hit.hitPoint - bounds.second).cwiseAbs();
                        float minDist = std::numeric_limits<float>::max();
                        int minAxis = 0;
                        float sign = 1.0f;
                        
                        for (int i = 0; i < Dim; ++i) {
                            if (dMin[i] < minDist) {
                                minDist = dMin[i];
                                minAxis = i;
                                sign = -1.0f;
                            }
                            if (dMax[i] < minDist) {
                                minDist = dMax[i];
                                minAxis = i;
                                sign = 1.0f;
                            }
                        }
                        hit.normal = PointType::Zero();
                        hit.normal[minAxis] = sign;
                    }
                }
            }

            if (node->isLeaf()) continue;

            float t0 = current.tMin;
            float t1 = current.tMax;

            Eigen::Vector3f ttt = (node->center - ray.origin).cwiseProduct(ray.invDir);
            int currIdx = ((t0 >= ttt.x()) ? 1 : 0) | ((t0 >= ttt.y()) ? 2 : 0) | ((t0 >= ttt.z()) ? 4 : 0);

            struct ChildInterval {
                OctreeNode* node;
                float tMin;
                float tMax;
            };
            ChildInterval children[4];
            int childCount = 0;

            while(t0 < t1 && t0 <= currentMaxDist) {
                Eigen::Vector3f next_t;
                next_t[0] = (currIdx & 1) ? t1 : ttt[0];
                next_t[1] = (currIdx & 2) ? t1 : ttt[1];
                next_t[2] = (currIdx & 4) ? t1 : ttt[2];

                float tNext = next_t.minCoeff();
                int physIdx = currIdx ^ ray.signMask;

                if (node->children[physIdx]) {
                    children[childCount++] = {node->children[physIdx].get(), t0, tNext};
                }

                t0 = tNext;
                currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
            }

            if (stackPtr + childCount > 256) continue;

            for (int i = childCount - 1; i >= 0; --i) {
                stack[stackPtr++] = {children[i].node, children[i].tMin, children[i].tMax};
            }
        }
        
        return hitSomething;
    }

    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB);
    frame blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale = 0.5f,
                frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 1,
                int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    frame fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB);
    frame renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB,
        int samplesPerPixel = 2, int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    void stepPhysics(float dt);

    std::vector<std::shared_ptr<NodeData>> getExternalNodes(int targetObjectId) {
        std::vector<std::shared_ptr<NodeData>> candidates;
        std::vector<std::shared_ptr<NodeData>> surfaceNodes;
        
        collectNodesByObjectId(root_.get(), targetObjectId, candidates);

        if (candidates.empty()) return surfaceNodes;

        surfaceNodes.reserve(candidates.size());

        const std::array<PointType, 6> directions = {
            PointType(1, 0, 0), PointType(-1, 0, 0), PointType(0, 1, 0), 
            PointType(0, -1, 0), PointType(0, 0, 1), PointType(0, 0, -1)
        };

        for (const auto& node : candidates) {
            bool isExposed = false;
            float step = node->size; 
            for (const auto& dir : directions) {
                PointType probePos = node->position + (dir * step);
                auto neighbor = find(probePos, step * 0.25f);

                if (neighbor == nullptr || !neighbor->isActive() || neighbor->objectId != node->objectId) {
                    isExposed = true;
                    break;
                }
            }

            if (isExposed) {
                surfaceNodes.push_back(node);
            }
        }
        surfaceNodes.shrink_to_fit();

        return surfaceNodes;
    }

    void isolateInternalNodes(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes); 

        if(nodes.empty()) return;
        float checkRad = nodes[0]->size * 1.5f;

        for(auto& node : nodes) {
            int hiddenSides = 0;
            PointType dirs[6] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
            for(int i=0; i<6; ++i) {
                auto neighbor = find(node->position + dirs[i] * node->size, checkRad);
                if(neighbor && neighbor->objectId == objectId && neighbor->isActive()) {
                    auto nObj = getObject(neighbor->objectId);
                    Material nMat = nObj ? nObj->getRenderMaterial(neighbor->renderMatIdx) : Material();
                    if (nMat.transmission < 0.01f) {
                        hiddenSides++;
                    }
                }
            }
        }
    }

    bool moveObject(int objectId, const PointType& offset) {
        if (!root_) return false;
        
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if(nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int oldDepth = 0;
        OctreeNode* oldStart = getHighestCommonNode(root_.get(), oldBounds, 0, oldDepth);
        
        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        BoundingBox newBounds = oldBounds;
        newBounds.first += offset;
        newBounds.second += offset;
        ensureBounds(newBounds);

        int newDepth = 0;
        OctreeNode* newStart = getHighestCommonNode(root_.get(), newBounds, 0, newDepth);

        size_t added = 0;
        for(auto& n : nodes) {
            n->position += offset;
            if (insertRecursive(newStart, n, newDepth)) {
                added++;
            }
        }
        size += added;
        
        auto obj = getObject(objectId);
        if (obj) {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            obj->centerPosition += offset;
        }
        
        return true;
    }

    void optimize() {
        if (root_) {
            optimizeRecursive(root_.get());
            generateLODs();
        }
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) {
            os << "[Octree Stats] Tree is null/empty." << std::endl;
            return;
        }

        size_t totalNodes = 0;
        size_t leafNodes = 0;
        size_t actualPoints = 0;
        size_t maxTreeDepth = 0;
        size_t maxPointsInLeaf = 0;
        size_t minPointsInLeaf = std::numeric_limits<size_t>::max();
        size_t lodGeneratedNodes = 0;
        size_t unloaded = 0;

        printStatsRecursive(root_.get(), 0, totalNodes, leafNodes, actualPoints, 
                            maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes, unloaded);

        if (leafNodes == 0) minPointsInLeaf = 0;
        double avgPointsPerLeaf = totalNodes > 0 ? (double)actualPoints / totalNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16); 

        os << "========================================\n";
        os << "             OCTREE STATS               \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
        os << "  LOD Falloff Rate  : " << lodFalloffRate_ << "\n";
        os << "  LOD Min Distance  : " << lodMinDistance_ << "\n";
        os << "Structure:\n";
        os << "  Total Nodes       : " << totalNodes << "\n";
        os << "  Leaf Nodes        : " << leafNodes << "\n";
        os << "  Non-Leaf Nodes    : " << (totalNodes - leafNodes) << "\n";
        os << "  LODs Generated    : " << lodGeneratedNodes << "\n";
        os << "  Tree Height       : " << maxTreeDepth << "\n";
        os << "  Unloaded Nodes    : " << unloaded << "\n";
        os << "Data:\n";
        os << "  Total Points      : " << size << " (Tracked) / " << actualPoints << " (Counted)\n";
        os << "  Points/Leaf (Avg) : " << std::fixed << std::setprecision(2) << avgPointsPerLeaf << "\n";
        os << "  Points/Leaf (Min) : " << minPointsInLeaf << "\n";
        os << "  Points/Leaf (Max) : " << maxPointsInLeaf << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }

    void clear() {
        if (root_) {
            clearNode(root_.get());
            root_.reset();
        }
        
        size = 0;
    }
    
    void getLoadedStatsSafe(const OctreeNode* node, size_t& loadedNodes, size_t& loadedPoints) const {
        if (!node) return;
        loadedNodes++;
        
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLoaded()) return;
        
        loadedPoints += node->points.size();
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    getLoadedStatsSafe(node->children[i].get(), loadedNodes, loadedPoints);
                }
            }
        }
    }

    size_t getEstimatedMemoryUsageMB() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_.get(), loadedNodes, loadedPoints);
        
        size_t nodeMem = loadedNodes * sizeof(OctreeNode);
        size_t pointMem = loadedPoints * (sizeof(NodeData) + sizeof(std::shared_ptr<NodeData>));
        
        return (nodeMem + pointMem) / (1024 * 1024);
    }

    size_t getLoadedPointCount() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_.get(), loadedNodes, loadedPoints);
        return loadedPoints;
    }
};
}
#endif
