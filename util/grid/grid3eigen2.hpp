#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"

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
    /*
    00000001=worker on
    00000010=auto optimize enabled
    00000100=queue streaming
    00001000=physics collider dirty (do I actually need this?)
    */
    std::atomic<uint8_t> flags;
    PointType fatStep;

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

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
    }

    uint16_t getFatCellIndex(const PointType& point) const {
        PointType rootMin = root_.get().bounds.first;
        uint8_t x = static_cast<uint8_t>((point[0] - rootMin[0]) / xStep);
        uint8_t y = static_cast<uint8_t>((point[1] - rootMin[1]) / yStep);
        uint8_t z = static_cast<uint8_t>((point[2] - rootMin[2]) / zStep);
        x &= 0x0F;
        y &= 0x0F;
        z &= 0x0F;
        return mortonEncode16B(x, y, z);
    }

    uint16_t mortonEncode16B(uint8_t x, uint8_t y, uint8_t z) {
        uint16_t xx = (x | (x << 8)) & 0x0F0F;
        uint16_t yy = (y | (y << 8)) & 0x0F0F;
        uint16_t zz = (z | (z << 8)) & 0x0F0F;
        
        xx = (xx | (xx << 4)) & 0x0C3;
        yy = (yy | (yy << 4)) & 0x0C3;
        zz = (zz | (zz << 4)) & 0x0C3;
        
        return xx | (yy << 1) | (zz << 2);
    }


//recursives
    OctreeNode* getHighestCommonNodeRecursive(const PointType Min, const PointType Max, OctreeNode* current) const {
        if (current->isFat()) {
            uint16_t coct = getFatCellIndex(Min);
            if (coct = getFatCellIndex(Max)) {
                getHighestCommonNodeRecursive(Min, Max, current);
            }
        } else {
            uint8_t coct = getOctant(Min, current->center);
            if (coct == getOctant(Max, current->center)) {
                getHighestCommonNodeRecursive(Min, Max, current->children[coct])
            }
        }
        return current;
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

    OctreeNode* getHighestCommonNode(const std::vector<PointType>& positions, OctreeNode* current) const {
        PointType min, max;
        for (auto pos : positions) {
            if (min.x > pos.x) min.x = pos.x;
            if (min.y > pos.y) min.y = pos.y;
            if (min.z > pos.z) min.z = pos.z;
            if (max.x > pos.x) max.x = pos.x;
            if (max.y > pos.y) max.y = pos.y;
            if (max.z > pos.z) max.z = pos.z;
        }

        OctreeNode current = root.get();
        getHighestCommonNodeRecursive(min, max, current);
        return current;
    }

    OctreeNode* getHighestCommonNode(const vector<NodeData_>& nodes, OctreeNode* current) const {
        PointType min, max;
        for (auto node : nodes) {
            if (min.x > node.position.x) min.x = node.position.x;
            if (min.y > node.position.y) min.y = node.position.y;
            if (min.z > node.position.z) min.z = node.position.z;
            if (max.x > node.position.x) max.x = node.position.x;
            if (max.y > node.position.y) max.y = node.position.y;
            if (max.z > node.position.z) max.z = node.position.z;
        }

        OctreeNode current = root.get();
        getHighestCommonNodeRecursive(min, max, current);
        return current;
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

#endif