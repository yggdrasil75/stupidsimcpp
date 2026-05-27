#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
#include "impl/structs.inl"
#include "impl/skybox.inl"
#include "impl/rendering.inl"

#ifdef SSE
#include <immintrin.h>
#endif

namespace Grid {

template<typename T, typename GasT = float, typename IndexType = uint16_t>
class Octree {
//declarations
public:
    using NodeData = NodeData_<T, IndexType>;
    using OctreeNode = OctreeNode_<T, GasT, IndexType>;
    using Material = Material_<T, IndexType>;
    using RayHit = RayHit_<T, IndexType>;
    using RenderNode = RenderNode_<T, IndexType>;
    using RenderData = RenderData_<T, IndexType>;
    using RenderBuffer = RenderBuffer_<T, IndexType>;
    using GridObject = GridObject_<T, IndexType>;
    using EulerianGasState = EulerianGasState_<GasT>;

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
    std::atomic<uint8_t> flags;
    std::string StoragePath = ".";

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
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16, std::string savePath) :
            root_(std::make_unique<OctreeNode>(minBound, maxBound, true)), maxPointsPerNode(maxPointsPerNode),
            maxDepth(maxDepth), size(0), skybox_(1024, 1024), StoragePath(savePath) {
        setQeuueStreaming(false);
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    //default constructor to prevent errors
    Octree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), skybox_(1024, 1024) {
        setQeuueStreaming(false);
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    //destructor
    ~Octree() {
        stopWorkerThread();
        setQeuueStreaming(false);
        clear();
    }
    
    //copy constructors
    Octree(const Octree& other) {
        other.stopWorkerThread();
        other.save("./temp");
        load("./temp");
        startWorkerThread();
    }

    Octree(Octree&& other) noexcept {
        other.stopWorkerThread();
        other.save("./temp");
        load("./temp");
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

    bool isQeuueStreaming() const {
        return flags.load(std::memory_order_relaxed) & QUEUE_STREAMING;
    }

    void setQeuueStreaming(bool v) {
        if (v) flags.fetch_or(QUEUE_STREAMING, std::memory_order_relaxed);
        else flags.fetch_and(~QUEUE_STREAMING, std::memory_order_relaxed);
    }

    bool isPhysicColliderDirty() const {
        return flags.load(std::memory_order_relaxed) & PHYSICS_COLLIDER_DIRTY;
    }

    void setPhysicColliderDirty(bool v) {
        if (v) flags.fetch_or(PHYSICS_COLLIDER_DIRTY, std::memory_order_relaxed);
        else flags.fetch_and(~PHYSICS_COLLIDER_DIRTY, std::memory_order_relaxed);
    }


//recursives
    OctreeNode* getHighestCommonNodeRecursive(const PointType Min, const PointType Max, OctreeNode* current) const {
        if (current->isFat()) {
            uint16_t coct = getFatCellIndex(Min, current);
            if (coct = getFatCellIndex(Max, current)) {
                getHighestCommonNodeRecursive(Min, Max, current->children[coct].get());
            }
        } else {
            uint8_t coct = getOctant(Min, current->center);
            if (coct == getOctant(Max, current->center)) {
                getHighestCommonNodeRecursive(Min, Max, current->children[coct])
            }
        }
        return current;
    }

    void splitNodeRecusive(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        std::vector<std::shared_ptr<NodeData>> keep;
        keep.reserve(node->points.size());

        if (node->isFat()) {
            PointType rootMin = node->bounds.first;
            PointType step = (node->bounds.second - node->bounds.first) / 32.0f;
            #pragma omp parallel for collapse(3)
            for (uint8_t z = 0; z < 32; ++z) {
                for (uint8_t y = 0; y < 32; ++y) {
                    for (uint8_t x = 0; x < 32; ++x) {
                        uint16_t index = mortonEncodeFatNode(x, y, z);
                        PointType childMin = rootMin + PointType(x * step[0], y * step[1], z * step[2]);
                        PointType childMax = childMin + step;
                        auto child = std::make_unique<OctreeNode>(childMin, childMax);
                        child->bounds = {childMin, childMax};
                        child->center = (childMin + childMax) * 0.5f;
                        node->children[index] = std::move(child);
                    }
                }
            }

            for (const auto& pointData : node->points) {
                BoundingBox cubeBounds = pointData->getCubeBounds();
                uint16_t targetIndex = getFatCellIndex(pointData->position, node);
                if (boxContainsBox(node->children[targetIndex]->bounds, cubeBounds)) {
                    node->children[targetIndex]->points.emplace_back(std::move(pointData));
                } else {
                    keep.emplace_back(std::move(pointData));
                }
            }

            node->points = std::move(keep);
            node->setLeaf(false);

            for (int i = 0; i < 32768; ++i) {
                if (node->children[i]->points.size() > maxPointsPerNode) {
                    splitNodeRecusive(node->children[i].get(), depth + 1);
                }
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
            }

            for (const auto& pointData : node->points) {
                BoundingBox cubeBounds = pointData->getCubeBounds();
                PointType boundsCenter = (cubeBounds.first + cubeBounds.second) * 0.5f;
                uint8_t targetIndex = getOctant(boundsCenter, node->center);
                if (boxContainsBox(node->children[targetIndex]->bounds, cubeBounds)) {
                    node->children[targetIndex]->points.emplace_back(std::move(pointData));
                } else {
                    keep.emplace_back(std::move(pointData));
                }
            }

            node->points = std::move(keep);
            node->setLeaf(false);

            for (int i = 0; i < 8; ++i) {
                if (node->children[i]->points.size() > maxPointsPerNode) {
                    splitNodeRecusive(node->children[i].get(), depth + 1);
                }
            }

        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        ensureLoaded(node);
        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxContainsBox(node->bounds, cubeBounds)) return false;

        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }
        

        if (node->isLeaf()) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode) {
                splitNode(node, depth);
            }
            node->setDirty(true);
            return true;
        } else {
            bool insertedInChild = false;
            OctreeNode* targetChild = nullptr;
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            
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
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            splitNodeRecusive(node, depth);
        }
        
        uint16_t idx;
        if (node->isFat()) {
            idx = getFatCellIndex(point, node);
        } else {
            idx = getOctant(pos, node->center);
        }
        
        return ensureNodeAtDepthRecursive(node->children[octant].get(), pos, depth + 1, targetDepth);
    }

//tasks
    void lazilyOffload(OctreeNode* node) {
        {
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
                        node->loadRegion();
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
                if (!node->isLoaded()) node->loadRegion();
                node->setLoadQueued(false);
            }
            if (node->isLoaded()) {
                ensureLOD(node);
            }
        }
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
                        return isWorkerRunning() || !taskQueue_.empty(); 
                    });
                    
                    if (isWorkerRunning() && taskQueue_.empty()) return;
                    
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

    OctreeNode* getHighestCommonNode(const std::vector<PointType>& positions, OctreeNode* current = nullptr) const {
        if (!current) current = root_.get();
        PointType min = positions[0];
        PointType max = positions[0];
        for (const auto pos : positions) {
            min = min.cwiseMin(pos);
            max = max.cwiseMax(pos);
        }

        getHighestCommonNodeRecursive(min, max, current);
        return current;
    }

    OctreeNode* getHighestCommonNode(const vector<NodeData_>& nodes, OctreeNode* current = nullptr) const {
        if (!current) current = root_.get();
        PointType min = nodes[0]->position;
        PointType max = nodes[0]->position;
        for (auto node : nodes) {
            min = min.cwiseMin(node->position);
            max = max.cwiseMax(node->position);
        }
        getHighestCommonNodeRecursive(min, max, current);
        return current;
    }

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
    }

    uint16_t getFatCellIndex(const PointType& point, OctreeNode* node = root_.get()) const {
        PointType rootMin = node.bounds.first;
        PointType step = (node->bounds.second - node->bounds.first) / 32.0f;
        uint8_t x = std::clamp(static_cast<uint8_t>((point[0] - rootMin[0]) / step), 0, 31);
        uint8_t y = std::clamp(static_cast<uint8_t>((point[1] - rootMin[1]) / step), 0, 31);
        uint8_t z = std::clamp(static_cast<uint8_t>((point[2] - rootMin[2]) / step), 0, 31);
        return mortonEncodeFatNode(x, y, z);
    }

    uint16_t mortonEncodeFatNode(uint8_t x, uint8_t y, uint8_t z) {
        // uint32_t xx = x & 0x1F;
        // uint32_t yy = y & 0x1F;
        // uint32_t zz = z & 0x1F;
        // xx = (xx | (xx << 8)) & 0x100F;
        // yy = (yy | (yy << 8)) & 0x100F;
        // zz = (zz | (zz << 8)) & 0x100F;
        // xx = (xx | (xx << 4)) & 0x10C3;
        // yy = (yy | (yy << 4)) & 0x10C3;
        // zz = (zz | (zz << 4)) & 0x10C3;
        // xx = (xx | (xx << 2)) & 0x1249;
        // yy = (yy | (yy << 2)) & 0x1249;
        // zz = (zz | (zz << 2)) & 0x1249;
        // return xx | (yy << 1) | (zz << 2);
        
        return _pdep_u32(x, 0x49249) | _pdep_u32(y, 0x49249) << 1 | _pdep_u32(z, 0x49249) << 2;
    }

    void mortonDecodeFatNode(uint16_t morton, uint8_t *x, uint8_t *y, uint8_t *z) {
        uint32_t x_pext = _pext_u32(morton, 0x49249);
        uint32_t y_pext = _pext_u32(morton >> 1, 0x49249);
        uint32_t z_pext = _pext_u32(morton >> 2, 0x49249);
        
        *x = (uint8_t)x_pext;
        *y = (uint8_t)y_pext;
        *z = (uint8_t)z_pext;
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint16_t octant) const {
        if (node.isFat()) {
            PointType rootMin = node->bounds.first;
            PointType step = (node->bounds.second - node->bounds.first) / 32.0f;
            uint8_t x, y, z;
            mortonDecodeFatNode(octant, x, y, z);
            PointType childMin, childMax;
            childMin[0] = rootMin[0] + x * step;
            childMin[1] = rootMin[1] + y * step;
            childMin[2] = rootMin[2] + z * step;
            
            childMax[0] = childMin[0] + step;
            childMax[1] = childMin[1] + step;
            childMax[2] = childMin[2] + step;
            
            return {childMin, childMax};
        } else {
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
        invLodf = 1 / rate;
    }

    void setMaxDistance(float dist) {
        maxDistance_ = dist;
        keepDistance_ = dist * 1.2;
    }

    void setMinLODSize(float size) {
        minLodSize_ = size;
        minLodVolume_ = size * size * size;
    }

    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }

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

    

//updates

}

}

#endif