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
    using Material = Material_;
    using ExtendedMaterial = ExtendedMaterial_;
    using PhysicsMaterial = PhysicsMaterial_;
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
        if (current->isFat()) {
            uint16_t mcell = getFatCellIndex(Min, current);
            if (mcell == getFatCellIndex(Max, current) && current->children[minCell]) {
                return getHighestCommonNodeRecursive(Min, Max, current->children[mcell].get(), depth + 1);
            }
        } else {
            uint8_t mcell = getOctant(Min, current->center);
            if (mcell == getOctant(Max, current->center) && current->children[minCell]) {
                return getHighestCommonNodeRecursive(Min, Max, current->children[mcell].get(), depth + 1)
            }
        }
        return current;
    }

    void splitNodeRecursive(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        std::vector<std::shared_ptr<NodeData>> keep;
        keep.reserve(node->points.size());

        if (node->isFat()) {
            PointType rootMin = node->bounds().first;
            PointType step = (node->bounds().second - node->bounds().first) / 32.0f;
            #pragma omp parallel for collapse(3)
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
        ensureLoaded(node);
        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxContainsBox(node->bounds(), cubeBounds)) return false;

        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        if (node->isLeaf()) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode) {
                splitNodeRecursive(node, depth);
            }
            node->setDirty(true);
            return true;
        } else {
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
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
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
                if (node->isLoaded() && node->isSaveQueued() && node->isDirty()) {
                    node->saveRegion();
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

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
    }

    uint16_t getFatCellIndex(const PointType& point, const OctreeNode* node) const {
        BoundingBox bounds = node->bounds();
        const PointType& rootMin = bounds.first;
        PointType step = (bounds.second - rootMin) / 32.0f;
        uint8_t x = static_cast<uint8_t>(std::clamp((point[0] - rootMin[0]) / step[0], 0.0f, 31.0f));
        uint8_t y = static_cast<uint8_t>(std::clamp((point[1] - rootMin[1]) / step[1], 0.0f, 31.0f));
        uint8_t z = static_cast<uint8_t>(std::clamp((point[2] - rootMin[2]) / step[2], 0.0f, 31.0f));
        return mortonEncodeFatNode(x, y, z);
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint16_t octant) const {
        BoundingBox bounds = node->bounds();
        if (node->isFat()) {
            const PointType& rootMin = bounds.first;
            PointType step = (bounds.second - bounds.first) / 32.0f;
            uint8_t x, y, z;
            mortonDecodeFatNode(octant, &x, &y, &z);
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
        // 0 is "common junk", -1 is next, 1 is immobile terrain, and 2 is mobile terrain (water, topsoil)
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
                uint32_t absorption = 0, BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f) {
        std::shared_ptr<GridObject> obj = getOrCreateObject(objectId);
        Material rmat(emittance, roughness, reflective, ior, absorption);
        IndexType rIdx = obj->getOrAddRenderMaterial(rmat);

        PhysicsMaterial pmat{bType, mass, restitution, density};
        IndexType pIdx = obj->getOrAddPhysicsMaterial(pmat);
        uint8_t tIdx = obj->getOrAddTransmission(transmission);
        IndexType colorIdx = obj->getOrAddColorIndex(color);
        auto pointData = std::make_shared<NodeData>(data, pos, visible, colorIdx, size, active, objectId, rIdx, pIdx, tIdx);

        PointType relPos = pos - obj->centerPosition;
        {
            std::unique_lock<std::shared_mutex> lock(obj->objMutex);
            obj->relativeVoxels.push_back({relPos, rIdx, pIdx, tIdx, size});
        }

        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        } else return false;
    }

    //fix these defaults later.
    auto setTerrain = std::bind(&Octree::insert, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true, std::placeholders::_4,
                                true, 1, Eigen::Vector3f::Zero(), 1.0f, 0.05f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), BodyType::STATIC, 1.0f, 0.1f, 1.0f);
    auto setWater = std::bind(&Octree::insert, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true, 0.001f,
                                true, 2, Eigen::Vector3f::Zero(), 0.2f, 0.02f, 0.95f, 1.33f, Eigen::Vector3f(0.15f, 0.05f, 0.01f), BodyType::FLUID, 1.0f, 0.05f, 1000.0f);

    bool insert(const T& data, const PointType& pos, Eigen::Vector3f color, bool visible = true, float size = 0.01f, bool active = true, int objectId = -1,
                Eigen::Vector3f emittance = Eigen::Vector3f::Zero(), float roughness = 1.0f, float reflective = 0.0f, float transmission = 0.0f, float ior = 1.45f, 
                Eigen::Vector3f absorption = Eigen::Vector3f::Zero(), BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f) {
        return insert(data, pos, color, visible, size, active, objectId, packRGB9E5(emittance), roughness, reflective, transmission, ior, packRGB9E5(absorption), bType, mass, restitution, density);
    }
    
    bool bulkInsert(const T& data, std::vector<PointType> positions, Eigen::Vector3f color, bool visible = true, float size = 0.01f, bool active = true, int objectId = -1,
                uint32_t emittance = 0, float roughness = 1.0f, float reflective = 0.0f, float transmission = 0.0f, float ior = 1.45f, 
                uint32_t absorption = 0, BodyType bType = BodyType::STATIC, float mass = 1.0f, float restitution = 1.0f, float density = 1.0f) {
        std::shared_ptr<GridObject> obj = getOrCreateObject(objectId);
        Material rmat(emittance, roughness, reflective, ior, absorption);
        IndexType rIdx = obj->getOrAddRenderMaterial(rmat);

        PhysicsMaterial pmat{bType, mass, restitution, density};
        IndexType pIdx = obj->getOrAddPhysicsMaterial(pmat);
        uint8_t tIdx = obj->getOrAddTransmission(transmission);

        IndexType colorIndex = obj->getOrAddColorIndex(color);

        int depth = 0;
        OctreeNode* commonNode = getHighestCommonNode(positions, root_.get(), depth);
        bool anyFailed = false;
        
        for (const auto& pos : positions) {
            auto pointData = std::make_shared<NodeData>(data, pos, visible, colorIndex, size, active, objectId, rIdx, pIdx, tIdx);
            
            PointType relPos = pos - obj->centerPosition;
            {
                std::unique_lock<std::shared_mutex> lock(obj->objMutex);
                obj->relativeVoxels.push_back({relPos, rIdx, pIdx, tIdx, size});
            }

            if (insertRecursive(commonNode, pointData, depth)) {
                this->size++;
            } else anyFailed = true;
        }
        //returns true for success, so inverts anyfailed.
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

    std::shared_ptr<NodeData> find(const PointType& pos, int objectId = -2, float tolerance = EPSILON, OctreeNode* node = root_.get()) {
        return findRecursive(node, pos, objectId, tolerance);
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
        if (root_) collectNodesByObjectId(root_.get(), objectId, nodes);
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

//removals
    bool removeObject(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
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

//static helpers
    static uint32_t packRGB9E5(const Eigen::Vector3f& color) {
        float maxv = color.maxCoeff();
        int exponent;
        std::frexp(maxv, &exponent);
        exponent = std::clamp(exponent, -16, 15);
        float scale = std::exp2f(-(exponent - 9));
        uint32_t r = std::round(std::clamp(color.x() * scale, 0.0f, 511.0f));
        uint32_t g = std::round(std::clamp(color.y() * scale, 0.0f, 511.0f));
        uint32_t b = std::round(std::clamp(color.z() * scale, 0.0f, 511.0f));
        return (static_cast<uint32_t>(exponent + 15) << 27) | ((b & 0x1FF) << 18) | ((g & 0x1FF) << 9) | (r & 0x1FF);
    }

    static Eigen::Vector3f unpackRGB9E5(uint32_t c) {
        int e = static_cast<int>(c >> 27) - 15;
        float scale = std::exp2f(e - 9);
        float r = static_cast<float>((c & 0x1FF)) * scale;
        float g = static_cast<float>((c >> 9) & 0x1FF) * scale;
        float b = static_cast<float>((c >> 18) & 0x1FF) * scale;
        return Eigen::Vector3f(r, g, b);
    }

//declarations
    void ensureLOD(OctreeNode* node);
    void invalidateLODForPoint(const std::shared_ptr<NodeData>& n);
    void collectNodesByObjectId(OctreeNode* node, int objectId, std::vector<std::shared_ptr<NodeData>>& nodes) const;
    size_t removeObjectRecursive(OctreeNode* node, int objectId);
    void optimize();


}

}

#endif