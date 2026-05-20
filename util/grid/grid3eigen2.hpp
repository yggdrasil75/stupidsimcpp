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
    00001000=physics collider (do I actually need this?)
    */
    std::atomic<uint8_t> flags;

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


//recursives

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
        setWorkerRunning(false);
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

    OctreeNode* getHighestCommonNodeRecursive(const std::vector<std::shared_ptr<NodeData>>& nodes) const {
        
    }

//public functions
public:

}

#endif