#ifndef g3eigen
#define g3eigen
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
#include "impl/structs.inl"
#include "impl/sphkernel.inl"
#include "impl/skybox.inl"
#include "impl/rendering.inl"
#include "impl/meshing.inl"

#ifdef SSE
#include <immintrin.h>
#endif

namespace Grid {

template<typename T, typename GasT = float, typename IndexType = uint16_t>
class Octree {
//declarations
public:
    using NodeData = NodeData_<T, IndexType>;
    using OctreeNode = OctreeNode_<T, IndexType>;
    using Material = Material_;
    using ExtendedMaterial = ExtendedMaterial_;
    using PhysicsMaterial = PhysicsMaterial_;
    using RayHit = RayHit_<T, IndexType>;
    using RenderNode = RenderNode_<T, IndexType>;
    using RenderData = RenderData_<T, IndexType>;
    using RenderBuffer = RenderBuffer_<T, IndexType>;
    using GridObject = GridObject_<T, IndexType>;
    // using EulerianGasState = EulerianGasState_<GasT>;
    using MeshVoxels = std::vector<std::shared_ptr<NodeData>>;
    

//variables
private: 
    
    //grid basics
    std::unique_ptr<OctreeNode> root_;
    long long size;
    uint8_t maxPointsPerNode; //this should likely get tuned based on system, larger might be better or worse for some systems. default 8 for now.
    uint16_t maxDepth; //if I need more than 65536 then I can start using fat nodes more dynamically.
    /*
    00000001=worker on
    00000010=auto optimize enabled
    00000100=queue streaming
    00001000=physics collider dirty (do I actually need this?)
    */
    std::atomic<uint8_t> flags{0};
    std::string StoragePath = ".";
    int nextObjId = 3;

    //skybox
    Skybox skybox_;
    Eigen::Vector3f skylight_ = {0.1f, 0.1f, 0.1f};
    Eigen::Vector3f backgroundColor_ = {0.53f, 0.81f, 0.92f};

    //grid physics objects
    std::unordered_map<int, std::shared_ptr<GridObject>> objects_;
    mutable std::shared_mutex objectsMutex_;

    //background task management
    mutable std::queue<std::function<void()>> taskQueue_;
    mutable std::mutex taskMutex_;
    mutable std::condition_variable taskCV_;
    std::thread workerThread_;

    //rendering
    std::atomic<uint32_t> frameCounter_{0};
    float minLodVolume_ = 0.0f;
    float minLodSize_ = 0.0f;
    size_t regionTargetPoints_ = 4096;
    float lodFalloffRate_ = 0.1f;
    float invLodf = 1 / lodFalloffRate_;
    float lodMinDistance_ = 100.0f;
    float maxDistance_ = lodMinDistance_ * lodMinDistance_;
    float keepDistance_ = maxDistance_ * 1.2;

    //physics
    std::mutex physicsMutex_;
    PointType phys_gravity{0.0f, -9.81f, 0.0f};
    float phys_gravityStrength = 9.81f;
    //sph
    float phys_smoothingRadius = 0.2f;
    float phys_restDensity = 1000.0f;
    float phys_viscosity = 200.0f;
    float phys_velocityDamping = 0.5f;
    SPHKernels kernels_{phys_smoothingRadius};

    //hard body

    //soft body

    //gases



//variable get/set
public: 
    //constructors:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16, std::string savePath = ".") :
            root_(std::make_unique<OctreeNode>(minBound, maxBound, true)), maxPointsPerNode(maxPointsPerNode),
            maxDepth(maxDepth), size(0), skybox_(1024, 1024), StoragePath(savePath) {
        setQueueStreaming(false);
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        auto terrain = getOrCreateObject(1);
        // terrain->setPreferredMesh(meshMode::SURFACENET);
        auto terrainWater = getOrCreateObject(2);
        // terrainWater->setPreferredMesh(meshMode::MANIFOLDCONTOUR);
        
        startWorkerThread();
    }

    //default constructor to prevent errors
    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), skybox_(1024, 1024) {
        setQueueStreaming(false);
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    //destructor
    ~Octree() {
        stopWorkerThread();
        setQueueStreaming(false);
        clear();
    }
    
    //copy constructors
    Octree(const Octree& other) {
        deserialize(other.serialize());
        startWorkerThread();
    }

    Octree(Octree&& other) noexcept {
        other.stopWorkerThread();
        deserialize(other.serialize());
        startWorkerThread();
    }

    //flags:
    bool isWorkerRunning() const {
        return flags.load(std::memory_order_relaxed) & WORKER_ON;
    }

    void setWorkerRunning(bool v) {
        if (v) flags.fetch_or(WORKER_ON, std::memory_order_relaxed);
        else flags.fetch_and(~WORKER_ON, std::memory_order_relaxed);
    }

    bool isAutoOptimizing() const {
        return flags.load(std::memory_order_relaxed) & AUTO_OPTIMIZE;
    }

    void setAutoOptimizing(bool v) {
        if (v) flags.fetch_or(AUTO_OPTIMIZE, std::memory_order_relaxed);
        else flags.fetch_and(~AUTO_OPTIMIZE, std::memory_order_relaxed);
    }

    bool isQueueStreaming() const {
        return flags.load(std::memory_order_relaxed) & QUEUE_STREAMING;
    }

    void setQueueStreaming(bool v) {
        if (v) flags.fetch_or(QUEUE_STREAMING, std::memory_order_relaxed);
        else flags.fetch_and(~QUEUE_STREAMING, std::memory_order_relaxed);
    }

    bool isPhysicsColliderDirty() const {
        return flags.load(std::memory_order_relaxed) & PHYSICS_COLLIDER_DIRTY;
    }

    void setPhysicsColliderDirty(bool v) {
        if (v) flags.fetch_or(PHYSICS_COLLIDER_DIRTY, std::memory_order_relaxed);
        else flags.fetch_and(~PHYSICS_COLLIDER_DIRTY, std::memory_order_relaxed);
    }

//recursives
    OctreeNode* getHighestCommonNodeRecursive(const PointType& Min, const PointType& Max, OctreeNode* current, int& depth) const {
        depth++;
        std::shared_lock<std::shared_mutex> lock(current->nodeMutex);
        if (current->isFat()) {
            uint16_t mcell = getFatCellIndex(Min, current);
            if (mcell == getFatCellIndex(Max, current) && current->children[mcell]) {
                return getHighestCommonNodeRecursive(Min, Max, current->children[mcell].get(), depth);
            }
        } else {
            uint8_t mcell = getOctant(Min, current->center);
            if (mcell == getOctant(Max, current->center) && current->children[mcell]) {
                return getHighestCommonNodeRecursive(Min, Max, current->children[mcell].get(), depth);
            }
        }
        depth--;
        return current;
    }

    void splitNodeRecursive(OctreeNode* node, int depth) {
        // std::cout << "splitting node at depth: " << depth << std::endl;
        if (depth >= maxDepth) return;
        std::vector<std::shared_ptr<NodeData>> keep;
        keep.reserve(node->points.size());
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);

        if (node->isFat()) {
            PointType rootMin = node->bounds().first;
            PointType step = (node->bounds().second - node->bounds().first) / 32.0f;
            // #pragma omp parallel for collapse(3)
            for (uint8_t z = 0; z < 32; ++z) {
                for (uint8_t y = 0; y < 32; ++y) {
                    for (uint8_t x = 0; x < 32; ++x) {
                        uint16_t index = mortonEncodeFatNode(x, y, z);
                        PointType childMin = rootMin + PointType(x * step[0], y * step[1], z * step[2]);
                        PointType childMax = childMin + step;
                        auto child = std::make_unique<OctreeNode>(childMin, childMax);
                        node->children[index] = std::move(child);
                    }
                }
            }

            for (auto& pointData : node->points) {
                BoundingBox cubeBounds = pointData->getCubeBounds();
                uint16_t targetIndex = getFatCellIndex(pointData->position, node);
                if (boxContainsBox(node->children[targetIndex]->bounds(), cubeBounds)) {
                    node->children[targetIndex]->points.emplace_back(std::move(pointData));
                } else {
                    keep.emplace_back(std::move(pointData));
                }
            }

            node->points = std::move(keep);
            node->setLeaf(false);

            for (int i = 0; i < 65536; ++i) {
                if (node->children[i] && node->children[i]->points.size() > maxPointsPerNode) {
                    splitNodeRecursive(node->children[i].get(), depth + 1);
                }
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
            }

            for (auto& pointData : node->points) {
                BoundingBox cubeBounds = pointData->getCubeBounds();
                PointType boundsCenter = (cubeBounds.first + cubeBounds.second) * 0.5f;
                uint8_t targetIndex = getOctant(boundsCenter, node->center);
                if (boxContainsBox(node->children[targetIndex]->bounds(), cubeBounds)) {
                    node->children[targetIndex]->points.emplace_back(std::move(pointData));
                } else {
                    keep.emplace_back(std::move(pointData));
                }
            }

            node->points = std::move(keep);
            node->setLeaf(false);

            for (int i = 0; i < 8; ++i) {
                if (node->children[i] && node->children[i]->points.size() > maxPointsPerNode) {
                    splitNodeRecursive(node->children[i].get(), depth + 1);
                }
            }

        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        // std::cout << "inserting recursively at depth: " << depth << std::endl;
        ensureLoaded(node);
        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxContainsBox(node->bounds(), cubeBounds)) return false;

        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        if (node->isLeaf()) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode) {
                lock.unlock();
                splitNodeRecursive(node, depth);
                lock.lock();
            }
            node->setDirty(true);
            return true;
        } else {
            lock.unlock();
            bool insertedInChild = false;
            OctreeNode* targetChild = nullptr;
            
            if (node->isFat()) {
                uint16_t targetIndex = getFatCellIndex(pointData->position, node);
                targetChild = node->children[targetIndex].get();
            } else {
                uint8_t targetIndex = getOctant(pointData->position, node->center);
                targetChild = node->children[targetIndex].get();
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
            // std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            splitNodeRecursive(node, depth);
        }
        
        uint16_t idx;
        if (node->isFat()) {
            idx = getFatCellIndex(pos, node);
        } else {
            idx = getOctant(pos, node->center);
        }
        
        return ensureNodeAtDepthRecursive(node->children[idx].get(), pos, depth + 1, targetDepth);
    }

    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, const PointType& pos, int objectId, float tolerance) {
        if (!node->contains(pos)) return nullptr;
        ensureLoaded(node);
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        for (const auto& pointData : node->points) {
            if (pointData->objectId != objectId && objectId != -2) continue;
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }
        if (!node->isLeaf()) {
            uint16_t octant;
            if (node->isFat()) {
                octant = getFatCellIndex(pos, node);
                if (node->children[octant]) {
                    return findRecursive(node->children[octant].get(), pos, objectId, tolerance);
                }
            } else {
                octant = getOctant(pos, node->center);
                if (node->children[octant]) {
                    return findRecursive(node->children[octant].get(), pos, objectId, tolerance);
                }
            }
        }
        return nullptr;
    }

    void searchNodeRecursive(OctreeNode* node, const PointType& center, float radiusSq, int objectId, std::vector<std::shared_ptr<NodeData>>& results) {
        ensureLoaded(node, false);
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectId || objectId < 0)) {
                results.emplace_back(pointData);
            }
        }
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) searchNodeRecursive(child.get(), center, radiusSq, objectId, results);
            }
        }
    }

    void updateStreamingRecursive(OctreeNode* node, const PointType& camPos, const PointType& camDir) {
        // std::cout << "updating streaming" << std::endl;
        if (!node) return;
        
        float minDistSq = 0.0f;
        float maxDistSq = 0.0f;
        BoundingBox nodeBounds = node->bounds();

        for(int i = 0; i < Dim; ++i) {
            float v = camPos[i];
            float minBound = nodeBounds.first[i];
            float maxBound = nodeBounds.second[i];
            
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
        maxPoint.x() = (camDir.x() >= 0) ? nodeBounds.second.x() : nodeBounds.first.x();
        maxPoint.y() = (camDir.y() >= 0) ? nodeBounds.second.y() : nodeBounds.first.y();
        maxPoint.z() = (camDir.z() >= 0) ? nodeBounds.second.z() : nodeBounds.first.z();
        
        if ((maxPoint - camPos).dot(camDir) < -0.05f) {
            isBehind = true;
        }
        
        float lodDistSq = lodMinDistance_ * lodMinDistance_;
        float maxDistSq_Max = maxDistance_ * maxDistance_;
        float keepDistSq = keepDistance_ * keepDistance_;
        
        if (maxDistSq <= lodDistSq) {
            loadSubtreeRecursive(node);
            return;
        }

        if (maxDistSq <= maxDistSq_Max && minDistSq > lodDistSq) {
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
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                if (node->isFat()) {
                    for (int i = 0; i < 65536; ++i) {
                        updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                    }
                } else {
                    for (int i = 0; i < 8; ++i) {
                        updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                    }
                }
            }
            return;
        }

        if (minDistSq > lodDistSq) {
            ensureLOD(node);
        } else {
            ensureLoaded(node, true);
        }
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLeaf()) {
            const int childCount = node->isFat() ? 65536 : 8;
            for (int i = 0; i < childCount; ++i) {
                if (node->children[i]) {
                    updateStreamingRecursive(node->children[i].get(), camPos, camDir);
                }
            }
        }
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

        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
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

    void loadSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLoaded(node, true);
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLeaf()) {
            int count = node->isFat() ? 65536 : 8;
            for (int i = 0; i < count; ++i) {
                loadSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void loadAndLodSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLOD(node);
        std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
        if (!node->isLeaf()) {
            int count = node->isFat() ? 65536 : 8;
            for (int i = 0; i < count; ++i) {
                loadAndLodSubtreeRecursive(node->children[i].get());
            }
        }
    }

    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;
        if (!node->isLoaded() || node->isLeaf() || !node->isKeepLoaded()) return;

        if (node->isFat()) {
            std::array<OctreeNode*, 65536> safeChildren = {nullptr};
            {
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                for (int i = 0; i < 65536; ++i) safeChildren[i] = node->children[i].get();
            }

            for (auto sc : safeChildren) {
                optimizeRecursive(sc);
            }

            bool childrenAreLeaves = true;
            {
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                for (auto sc : safeChildren) {
                    if (sc && !sc->isLeaf()) {
                        childrenAreLeaves = false;
                        break;
                    }
                }
            }

            if (childrenAreLeaves) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
                for (auto sc : safeChildren) {
                    if (sc) {
                        std::shared_lock<std::shared_mutex> childLock(sc->nodeMutex);
                        allPoints.insert(allPoints.end(), sc->points.begin(), sc->points.end());
                    }
                }

                if (allPoints.size() <= maxPointsPerNode) {
                    node->points = std::move(allPoints);
                    for (int i = 0; i < 65536; ++i) {
                        node->children[i].reset();
                    }
                    node->setLeaf(true);
                    node->setDirty(true);
                    
                    node->lodData = nullptr;
                }
            }
        } else {
            std::array<OctreeNode*, 8> safeChildren = {nullptr};
            {
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                for (int i = 0; i < 8; ++i) safeChildren[i] = node->children[i].get();
            }

            for (auto sc : safeChildren) {
                optimizeRecursive(sc);
            }

            bool childrenAreLeaves = true;
            {
                std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
                for (auto sc : safeChildren) {
                    if (sc && !sc->isLeaf()) {
                        childrenAreLeaves = false;
                        break;
                    }
                }
            }

            if (childrenAreLeaves) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
                for (auto sc : safeChildren) {
                    if (sc) {
                        std::shared_lock<std::shared_mutex> childLock(sc->nodeMutex);
                        allPoints.insert(allPoints.end(), sc->points.begin(), sc->points.end());
                    }
                }

                if (allPoints.size() <= maxPointsPerNode) {
                    node->points = std::move(allPoints);
                    for (int i = 0; i < 8; ++i) {
                        node->children[i].reset();
                    }
                    node->setLeaf(true);
                    node->setDirty(true);
                    
                    node->lodData = nullptr;
                }
            }
        }
    }

//tasks
    void enqueueTask(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(taskMutex_);
            taskQueue_.push(std::move(task));
        }
        taskCV_.notify_one();
    }

    void lazilyOffload(OctreeNode* node) {
        {
            if (!node->isLoaded() || node->isSaveQueued()) return;

            node->setSaveQueued(true);
            node->setLoadQueued(false);
        }

        enqueueTask([this, node]() {
            {
                std::shared_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (node->isLoaded() && node->isSaveQueued() && node->isDirty()) {
                    node->saveRegion(regionTargetPoints_, StoragePath);
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
            if (node->isLoaded() || node->isQueued()) return; 
            else {
                node->setLoadQueued(true);
                node->setSaveQueued(false);
            }
        }

        if (asyncLoad) { //lazy load
            enqueueTask([this, node]() {
                bool justLoaded = false;
                {
                    std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                    if (!node->isLoaded()) {
                        node->loadRegion(StoragePath);
                        justLoaded = node->isLoaded();
                    }
                    node->setLoadQueued(false);
                }
                if (justLoaded) {
                    ensureLOD(node);
                }
                
            });
        } else { //immediate load
            {
                std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (!node->isLoaded()) node->loadRegion(StoragePath);
                node->setLoadQueued(false);
            }
            if (node->isLoaded()) {
                ensureLOD(node);
            }
        }
    }

    void ensureLOD(OctreeNode* node) {
        ensureLoaded(node);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        // if (node-isEmpty()) return;
        // if (node->points.size() == 1 && node->isLeaf()) {
        //     node->lodData = node->points[0];
        // }
        // std::shared_ptr<GridObject> lodObj = getOrCreateObject(3);
        // if (!node->points.empty()) {
        //     auto lod = std::make_shared<NodeData>();
        //     Eigen::Vector3f averagePosition = node->center;
        //     Eigen::Vector3f avgEmittance = Eigen::Vector3f::Zero();
        //     Eigen::Vector3f avgAbsorption = Eigen::Vector3f::Zero();
        //     float averageRoughness = 0;
        //     float averageReflective = 0;
        //     float averageIOR = 0;
            
        //     double totalSize = 0;
        //     for (const auto& pt : points) {
        //         if (!pt->isActiveAndVisible()) continue;
        //         std::shared_ptr<GridObject> tempObj = getObject(pt->objectId);
        //         averagePosition += pt->position * pt->size;
        //         totalSize += pt->size;
        //         Material rmat = tempObj->getRenderMaterial(pt->rIdx);
        //         avgEmittance += unpackRGB9E5(rmat.chromaticity);
        //         avgAbsorption += unpackRGB9E5(rmat.absorption);
        //         averageRoughness += rmat.roughness;
        //         averageReflective += rmat.reflective;
        //         averageIOR += rmat.ior;
        //     }
        //     averagePosition = averagePosition / totalSize;
        // }
        ///TODO: put the lod generation in the object instead?
    }

    void startWorkerThread() {
        setWorkerRunning(true);
        workerThread_ = std::thread([this]() {
            auto lastOptimize = std::chrono::steady_clock::now();
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(taskMutex_);
                    auto nextOptimize = lastOptimize + std::chrono::seconds(60);
                    
                    bool timedOut = !taskCV_.wait_until(lock, nextOptimize, [this] {
                        return !isWorkerRunning() || !taskQueue_.empty();
                    });
                    
                    if (!isWorkerRunning() && taskQueue_.empty()) return;
                    
                    if (!taskQueue_.empty()) {
                        task = std::move(taskQueue_.front());
                        taskQueue_.pop();
                    } else if (timedOut && isAutoOptimizing()) {
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
        setWorkerRunning(false);
        taskCV_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

//other private functions
    BoundingBox getNodesBounds(const std::vector<std::shared_ptr<NodeData>>& nodes) const {
        BoundingBox bounds = nodes[0]->getCubeBounds();
        for (size_t i = 1; i < nodes.size(); ++i) {
            BoundingBox cb = nodes[i]->getCubeBounds();
            bounds.first = bounds.first.cwiseMin(cb.first);
            bounds.second = bounds.second.cwiseMax(cb.second);
        }
        return bounds;
    }

    OctreeNode* getHighestCommonNode(const std::vector<PointType>& positions, OctreeNode* current = nullptr, int& depth = 0) const {
        if (!current) current = root_.get();
        PointType min = positions[0];
        PointType max = positions[0];
        for (const auto& pos : positions) {
            min = min.cwiseMin(pos);
            max = max.cwiseMax(pos);
        }

        return getHighestCommonNodeRecursive(min, max, current, depth);
    }

    OctreeNode* getHighestCommonNode(const std::vector<std::shared_ptr<NodeData>>& nodes, OctreeNode* current = nullptr, int& depth = 0) const {
        if (!current) current = root_.get();
        PointType min = nodes[0]->position;
        PointType max = nodes[0]->position;
        for (const auto& node : nodes) {
            min = min.cwiseMin(node->position);
            max = max.cwiseMax(node->position);
        }
        return getHighestCommonNodeRecursive(min, max, current, depth);
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint16_t octant) const {
        // std::cout << "Creating child bounds" << std::endl;
        BoundingBox bounds = node->bounds();
        if (node->isFat()) {
            const PointType& rootMin = bounds.first;
            PointType step = (bounds.second - bounds.first) / 32.0f;
            uint8_t x, y, z;
            mortonDecodeFatNode(octant, x, y, z);
            PointType childMin, childMax;
            childMin[0] = rootMin[0] + x * step[0];
            childMin[1] = rootMin[1] + y * step[1];
            childMin[2] = rootMin[2] + z * step[2];
            
            childMax = childMin + step;
            
            return {childMin, childMax};
        } else {
            PointType childMin, childMax;
            const PointType& center = node->center;
            
            childMin[0] = (octant & 1) ? center[0] : bounds.first[0];
            childMax[0] = (octant & 1) ? bounds.second[0] : center[0];
            
            childMin[1] = (octant & 2) ? center[1] : bounds.first[1];
            childMax[1] = (octant & 2) ? bounds.second[1] : center[1];
            
            childMin[2] = (octant & 4) ? center[2] : bounds.first[2];
            childMax[2] = (octant & 4) ? bounds.second[2] : center[2];

            return {childMin, childMax};
        }
    }

    OctreeNode* ensureNodeAtDepth(const PointType& pos, size_t targetDepth) {
        if (!root_->contains(pos)) return nullptr;
        return ensureNodeAtDepthRecursive(root_.get(), pos, 0, targetDepth);
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

//public functions
public:

//setters
    std::shared_ptr<GridObject> getOrCreateObject(int id) {
        // 0 is "common junk", -1 is next, 1 is immobile terrain, and 2 is mobile terrain (water, topsoil), 3 is lod
        if (id < 0) {
            id = nextObjId++;
        }
        std::unique_lock<std::shared_mutex> lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        auto obj = std::make_shared<GridObject>(id);
        objects_[id] = obj;
        return obj;
    }

    void setPhysicsSmoothingRadius(float radius) {
        phys_smoothingRadius = radius;
        kernels_.update(radius);
    }

    void setPhysicsVelocityDamping(float damping) { phys_velocityDamping = damping; }
    void setPhysicsViscosity(float v) { phys_viscosity = v; }
    void setPhysicsRestDensity(float d) { phys_restDensity = d; }
    void setphys_gravityCenter(PointType n) { phys_gravity = n; }
    void setphys_gravityStrength(float d) { phys_gravityStrength = d; }

    void setLODFalloff(float rate) {
        lodFalloffRate_ = rate;
        invLodf = 1.0f / rate;
    }

    void setMaxDistance(float dist) {
        maxDistance_ = dist;
        keepDistance_ = dist * 1.2f;
    }

    void setMinLODSize(float size) {
        minLodSize_ = size;
        minLodVolume_ = size * size * size;
    }

    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }

    bool insert(const std::shared_ptr<NodeData>& point) {
        return insertRecursive(root_.get(), point, 0);
    }

    //emittance, absorption: rgb9e5
    bool insert(const T& data, const PointType& pos, Eigen::Vector3f color, bool visible = true, float size = 0.01f, bool active = true, int objectId = 0,
                uint32_t emittance = 0, float roughness = 1.0f, float reflective = 0.0f, float transmission = 0.0f, float ior = 1.45f, 
                uint32_t absorption = 0, BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f, bool staticb = false) {
        std::shared_ptr<GridObject> obj = getOrCreateObject(objectId);
        Material rmat(emittance, roughness, reflective, ior, absorption);
        IndexType rIdx = obj->getOrAddRenderMaterial(rmat);

        PhysicsMaterial pmat{bType, mass, restitution, density};
        IndexType pIdx = obj->getOrAddPhysicsMaterial(pmat);
        Eigen::Vector4f albedo = Eigen::Vector4f(color.x(), color.y(), color.z(), transmission);
        IndexType colorIdx = obj->getOrAddColorIndex(albedo);
        auto pointData = std::make_shared<NodeData>(data, pos, visible, colorIdx, size, active, objectId, rIdx, pIdx, staticb);

        PointType relPos = pos - obj->centerPosition;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            obj->relativeVoxels.push_back(relPos);
        }
        obj->setMeshClean(false);

        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        } else return false;
    }

    //fix these defaults later.
    std::function<bool(const T&, const PointType&, Eigen::Vector3f, bool, float)> setTerrain =
        [this](const T& d, const PointType& p, Eigen::Vector3f c, bool vis, float sz) {
            return insert(d, p, c, vis, sz, true, 1, Eigen::Vector3f::Zero(), 1.0f, 0.05f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), BodyType::STATIC, 1.0f, 0.1f, 1.0f, true);
        };

    std::function<bool(const T&, const PointType&, Eigen::Vector3f)> setWater =
        [this](const T& d, const PointType& p, Eigen::Vector3f c) {
            return insert(d, p, c, true, 0.001f, true, 2, Eigen::Vector3f::Zero(), 0.2f, 0.02f, 0.95f, 1.33f, Eigen::Vector3f(0.15f, 0.05f, 0.01f), BodyType::FLUID, 1.0f, 0.05f, 1000.0f);
        };

    bool insert(const T& data, const PointType& pos, Eigen::Vector3f color, bool visible = true, float size = 0.01f, bool active = true, int objectId = -1,
                Eigen::Vector3f emittance = Eigen::Vector3f::Zero(), float roughness = 1.0f, float reflective = 0.0f, float transmission = 0.0f, float ior = 1.45f, 
                Eigen::Vector3f absorption = Eigen::Vector3f::Zero(), BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f, bool staticb = false) {
        return insert(data, pos, color, visible, size, active, objectId, packRGB9E5(emittance), roughness, reflective, transmission, ior, packRGB9E5(absorption), bType, mass, restitution, density, staticb);
    }
    
    bool bulkInsert(const T& data, std::vector<PointType> positions, Eigen::Vector3f color, bool visible = true, float size = 0.01f, bool active = true, int objectId = -1,
                Eigen::Vector3f emittance = 0, float roughness = 1.0f, float reflective = 0.0f, float transmission = 0.0f, float ior = 1.45f, 
                Eigen::Vector3f absorption = 0, BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f, bool staticb = false) {
        std::shared_ptr<GridObject> obj = getOrCreateObject(objectId);
        Material rmat(packRGB9E5(emittance), roughness, reflective, ior, packRGB9E5(absorption));
        IndexType rIdx = obj->getOrAddRenderMaterial(rmat);

        PhysicsMaterial pmat{bType, mass, restitution, density};
        IndexType pIdx = obj->getOrAddPhysicsMaterial(pmat);

        Eigen::Vector4f albedo = Eigen::Vector4f(color.x(), color.y(), color.z(), transmission);
        IndexType colorIndex = obj->getOrAddColorIndex(albedo);

        int depth = 0;
        OctreeNode* commonNode = getHighestCommonNode(positions, root_.get(), depth);
        commonNode->setKeepLoaded(true);
        bool anyFailed = false;
        
        for (const auto& pos : positions) {
            auto pointData = std::make_shared<NodeData>(data, pos, visible, colorIndex, size, active, objectId, rIdx, pIdx, staticb);
            
            PointType relPos = pos - obj->centerPosition;
            {
                std::unique_lock<std::shared_mutex> lock(obj->objMutex);
                obj->relativeVoxels.push_back(relPos);
            }

            if (insertRecursive(commonNode, pointData, depth)) {
                this->size++;
            } else anyFailed = true;
        }
        
        commonNode->setKeepLoaded(false);
        obj->setMeshClean(false);
        return !anyFailed;
    }

    //lets deal with gases again later.
    // bool vaporizeObject(const PointType& pos, int objectId, const GasT& gasData, float pressure = 0.0f) {
    //     OctreeNode* node = ensureNodeAtDepth(pos, 4);
    //     auto point = find(pos, objectId);
    //     if (node) {
    //         std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
    //         node->gasState.data = gasData;
    //         node->gasState.density = point.density;
    //         node->gasState.pressure = pressure;
    //         node->setDirty(true);
    //         return true;
    //     }
    //     return false;
    // }

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

    void setRegionTargetPoints(size_t points) {
        regionTargetPoints_ = points;
    }

    size_t getRegionTargetPoints() const {
        return regionTargetPoints_;
    }

    void setMaterialByObjectId(int objectId, Eigen::Vector3f chromaticity, float roughness, float reflective, float ior, Eigen::Vector3f absorption) {
        auto obj = getOrCreateObject(objectId);
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            Material newmat = Material({packRGB9E5(chromaticity), roughness, reflective, ior, packRGB9E5(absorption)});
            IndexType id = obj->getOrAddRenderMaterial(newmat);
            std::vector<std::shared_ptr<NodeData>> points;
            collectPointsByObjectId(objectId, points);
            for (auto& pt : points) {
                pt->renderMatIdx = id;
            }
        }
    }

//getters
    std::shared_ptr<GridObject> getObject(int id) const {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        return nullptr;
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

    float getMinLODSize() const { return minLodSize_; }
    
    int getRenderMaterialIndex(const PointType& pos, float tolerance = EPSILON) {
        auto pt = find(pos, -2, tolerance);
        if (!pt) return -1;
        return pt->renderMatIdx;
    }

    int getPhysicsMaterialIndex(const PointType& pos, float tolerance = EPSILON) {
        auto pt = find(pos, -2, tolerance);
        if (!pt) return -1;
        return pt->physMatIdx;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, int objectId = -2, float tolerance = EPSILON, OctreeNode* node = nullptr) {
        if (!node) node = root_.get();
        return findRecursive(node, pos, objectId, tolerance);
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius, int objectid = -2) {
        std::vector<std::shared_ptr<NodeData>> results;
        
        float radiusSq = radius * radius;
        int depth = 0;
        OctreeNode* startingPoint = getHighestCommonNodeRecursive(center - PointType::Constant(radius), center + PointType::Constant(radius), root_.get(), depth);
        searchNodeRecursive(startingPoint, center, radiusSq, objectid, results);
        
        return results;
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) {
            os << "[Octree Stats] Tree is null/empty." << std::endl;
            return;
        }
        std::cout << "printing stats " << std::endl;

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
        os << "  Min               : [" << root_->bounds().first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds().second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }
    
    void collectPointsByObjectId(int id, std::vector<std::shared_ptr<NodeData>>& results) {
        auto obj = getObject(id);
        if (!obj) return;
        
        std::vector<PointType> absolutePositions;
        {
            std::shared_lock<std::shared_mutex> lock(obj->objMutex);
            if (obj->relativeVoxels.empty()) return;
            
            absolutePositions.reserve(obj->relativeVoxels.size());
            for (const auto& relPos : obj->relativeVoxels) {
                absolutePositions.push_back(obj->centerPosition + relPos);
            }
        }
        int depth = 0;
        OctreeNode* commonNode = getHighestCommonNode(absolutePositions, root_.get(), depth);
        results.reserve(absolutePositions.size());
        
        for (const auto& absPos : absolutePositions) {
            auto pt = find(absPos, id, EPSILON, commonNode);
            
            if (pt && pt->isActive()) {
                results.push_back(pt);
            }
        }
    }

//updates
    bool updateRenderMaterial(int objectId, IndexType index, const Material& mat) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            if (index >= obj->renderMaterials.size()) return false;
            obj->renderMaterials[index] = mat;
        }
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectPointsByObjectId(objectId, nodes);
        for (auto& n : nodes) invalidateLODForPoint(n);
        return true;
    }

    bool updatePhysicsMaterial(int objectId, IndexType index, const PhysicsMaterial& pmat) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            if (index >= obj->physicsMaterials.size()) return false;
            obj->physicsMaterials[index] = pmat;
        }
        return true;
    }

    bool setMesh(int objectId, meshMode mode = meshMode::NAIVE) {
        TIME_FUNCTION;
        auto obj = getObject(objectId);
        if (!obj) return false;
        if (obj->isMeshClean()) return true;

        if (mode == meshMode::OCCUPANCY || mode == meshMode::TRANSVOXEL) {
            throw std::runtime_error("NotImplementedException");
        }

        std::vector<std::shared_ptr<NodeData>> voxels;
        collectPointsByObjectId(objectId, voxels);

        voxels.erase(std::remove_if(voxels.begin(), voxels.end(), [](const std::shared_ptr<NodeData>& p) { return !p->isVisible(); }),
            voxels.end());
        
        std::unique_lock<std::shared_mutex> lock(obj->objMutex);
        obj->objMesh.vertices.clear();
        obj->objMesh.tris.clear();

        switch (mode) {
            case meshMode::NAIVE:          mesh_naive(*obj, voxels); break;
            case meshMode::GREEDY:         mesh_greedy(*obj, voxels); break;
            case meshMode::MARCHINGCUBES:  mesh_marchingCubes(*obj, voxels, false); break;
            case meshMode::NAIVEMARCHING:  mesh_marchingCubes(*obj, voxels, true); break;
            case meshMode::SURFACENET:     mesh_surfaceNet(*obj, voxels); break;
            case meshMode::DUALCONTOUR:    mesh_dualContour(*obj, voxels, false); break;
            case meshMode::MANIFOLDCONTOUR:mesh_dualContour(*obj, voxels, true); break;
            case meshMode::CUBICMARCHING:  mesh_cubicMarching(*obj, voxels); break;
            case meshMode::DUALMARCHING:   mesh_dualMarching(*obj, voxels); break;
            case meshMode::OCCUPANCY:
            case meshMode::TRANSVOXEL:
            default:
                throw std::runtime_error("NotImplementedException");
        }

        // std::cout << "done meshing" << std::endl;
        obj->setMeshClean(true);
        return true;
    }

    bool setMesh(int objectId) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        return setMesh(objectId, obj->getPreferredMesh());
    }

    void setObjectMeshMode(int objectId, meshMode mode) {
        auto obj = getOrCreateObject(objectId);
        obj->setPreferredMesh(mode);
    }

    void updateStreaming(const Camera& cam) {
        if (isQueueStreaming()) return;
        PointType camPos = cam.origin;
        PointType camDir = cam.direction;
        setQueueStreaming(true);
        enqueueTask([this, camPos, camDir]() {
            updateStreamingRecursive(root_.get(), camPos, camDir);
            setQueueStreaming(false);
        });
    }
    
    void makeObjectFluid(int objectId, float newMass = -1, BodyType newType = BodyType::FLUID) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectPointsByObjectId(objectId, nodes);
        
        auto obj = getOrCreateObject(objectId);
        PhysicsMaterial_ pmat{newType, newMass, 1.0, 1.0};
        
        uint16_t newIdx = obj->getOrAddPhysicsMaterial(pmat);

        std::lock_guard<std::mutex> lock(physicsMutex_);
        for (auto& n : nodes) {
            PhysicsMaterial_ oldPmat = obj->getPhysicsMaterial(n->physMatIdx);
            if (newMass < 0) {
                pmat = {newType, oldPmat.mass, oldPmat.restitution, oldPmat.density};
                newIdx = obj->getOrAddPhysicsMaterial(pmat);
            }
            n->physMatIdx = newIdx;
        }
    }

    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }
    
    void generateMeshes() {
        for (auto [id, obj] : objects_) {
            setMesh(id, obj->getPreferredMesh());
        }
    }

    void optimize() {
        optimizeRecursive(root_.get());
        generateLODs();
        generateMeshes();
    }
//removals
    bool removeObject(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectPointsByObjectId(objectId, nodes);
        if (nodes.empty()) return false;

        int depth = 0;
        OctreeNode* startNode = getHighestCommonNode(nodes, root_.get(), depth);

        size_t removed = removeObjectRecursive(startNode, objectId);
        size -= removed;

        {
            std::unique_lock<std::shared_mutex> lock(objectsMutex_);
            objects_.erase(objectId);
        }
        return true;
    }

    void clear() {
        if (root_) {
            clearNode(root_.get());
            // root_.reset();
        }

        size = 0;
    }
    
    void clearNode(OctreeNode* node) {
        if (!node) return;
        
        std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        if (node->isFat()) {
            for (int i = 0; i < 65536; ++i) {
                if (node->children[i]) {
                    clearNode(node->children[i].get());
                    node->children[i].reset(nullptr);
                }
            }
        }
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                clearNode(node->children[i].get());
                node->children[i].reset(nullptr);
            }
        }
        
        node->setLeaf(true);
    }
//static helpers

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

    bool rayCubeIntersect(const Ray& ray, const BoundingBox& box, float& t, PointType& normal) {
        float t0x = (box.first[0] - ray.origin[0]) * ray.invDir[0];
        float t1x = (box.second[0] - ray.origin[0]) * ray.invDir[0];
        if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

        float t0y = (box.first[1] - ray.origin[1]) * ray.invDir[1];
        float t1y = (box.second[1] - ray.origin[1]) * ray.invDir[1];
        if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

        float t0z = (box.first[2] - ray.origin[2]) * ray.invDir[2];
        float t1z = (box.second[2] - ray.origin[2]) * ray.invDir[2];
        if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

        float tMin = std::max({t0x, t0y, t0z});
        float tMax = std::min({t1x, t1y, t1z});

        if (tMax < std::max(0.0f, tMin) || tMax < 0.0f) {
            return false;
        }

        t = tMin < 0.0f ? tMax : tMin;
        
        PointType hitPoint = ray.origin + ray.dir * t;
        
        PointType dMin = (hitPoint - box.first).cwiseAbs();
        PointType dMax = (hitPoint - box.second).cwiseAbs();
        
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

    inline float edgeFn(float ax, float ay, float bx, float by, float px, float py) {
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    }

//declarations
    void rasterize(const Camera& cam, int height, int width, frame* colorOut, frame* depthOut,
                       frame* normalOut, frame* objectOut, std::vector<float>* linearDepth = nullptr);
    frame renderDepthMap(const Camera& cam, int height, int width);
    frame renderNormalMap(const Camera& cam, int height, int width);
    frame renderObjectMap(const Camera& cam, int height, int width);
    frame renderColorMap(const Camera& cam, int height, int width);
    Eigen::Vector3f traceVoxelRay(const PointType& origin, const PointType& dir, float minT, float maxT, const Eigen::Vector3f& bgColor);
    bool skipPaths() const;

    void invalidateLODForPoint(const std::shared_ptr<NodeData>& n);
    size_t removeObjectRecursive(OctreeNode* node, int objectId);
    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB, bool rasterOnly = false);
    frame blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale = 0.5f,
                frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 1,
                int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    frame fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB);
    frame renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB,
        int samplesPerPixel = 2, int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    void stepPhysics(float dt);

    
    struct MeshGrid { PointType origin; float cellSize; };
    std::shared_ptr<NodeData> meshVoxelAt(const MeshGrid& g, const Eigen::Vector3i& k,
                                          int objectId, OctreeNode* ancestor);
    OctreeNode* meshAncestor(const MeshVoxels& voxels);
    static float pickCellSize(const MeshVoxels& voxels);
    static void latticeBounds(const MeshVoxels& voxels, float cellSize,
                              PointType& origin, Eigen::Vector3i& outMin, Eigen::Vector3i& outMax);
    void mesh_naive(GridObject& obj, const MeshVoxels& voxels);
    void mesh_greedy(GridObject& obj, const MeshVoxels& voxels);
    void mesh_marchingCubes(GridObject& obj, const MeshVoxels& voxels, bool naiveNoLUT);
    void mesh_surfaceNet(GridObject& obj, const MeshVoxels& voxels);
    void mesh_dualContour(GridObject& obj, const MeshVoxels& voxels, bool manifold);
    void mesh_cubicMarching(GridObject& obj, const MeshVoxels& voxels);
    void mesh_dualMarching(GridObject& obj, const MeshVoxels& voxels);
};

}

#endif