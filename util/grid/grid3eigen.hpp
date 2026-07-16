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
    // also need to store relative positions to the object center, both "resting" and "current". 
    // some rendering options: sgvf (temporal accumulation)


///@brief A highly optimized sparse voxel Octree tailored for physics, rendering, LOD, and streaming.
///@tparam T The custom data type attached to each node/voxel.
template<typename T>
class Octree {
public:
    ///@brief Alias for node data structures used in the octree
    using NodeData = NodeData_<T>;
    ///@brief Alias for octree nodes
    using OctreeNode = OctreeNode_<T>;
    ///@brief Alias for raycasting hit information
    using RayHit = RayHit_<T>;
    ///@brief Alias for render nodes
    using RenderNode = RenderNode_<T>;
    ///@brief Alias for the rendering buffer
    using RenderBuffer = RenderBuffer_<T>;
    ///@brief Alias for a group of linked octree components
    using GridObject = GridObject_<T>;

private:
    ///@brief Counts the number of active bits in an 8-bit mask
    ///@param mask The 8-bit mask to analyze
    ///@return The number of bits set to 1
    int countBits(uint8_t mask) const {
        int count = 0;
        while (mask) {
            mask &= (mask - 1);
            count++;
        }
        return count;
    }

    ///@brief The root node of the octree structure
    std::unique_ptr<OctreeNode> root_;
    
    ///@brief Total number of elements/points in the octree
    size_t size;
    
    ///@brief Maximum allowed points inside a single leaf node before splitting
    size_t maxPointsPerNode;
    
    ///@brief Counter used to assign unique IDs to new GridObjects
    int nextObjId = 0;
    
    ///@brief Hash map storing managed GridObjects by their ID
    std::unordered_map<int, std::shared_ptr<GridObject>> objects_;
    
    ///@brief Mutex securing read/write operations to the objects map
    mutable std::shared_mutex objectsMutex_;
    RenderMaterialStore renderMaterials_;

    Skybox skybox_;
    Vec3 skylight_ = {0.1f, 0.1f, 0.1f};
    Vec3 backgroundColor_ = {0.53f, 0.81f, 0.92f};

    ///@brief World-space participating-media boxes (dust/haze/mist), consumed
    ///       by the GPU wavefront path tracer. See addFogVolume().
    struct FogVolume {
        Vec3 minB, maxB;      // axis-aligned bounds
        float density;        // extinction scale sigma_t (per world unit)
        Vec3 scatterColor;    // scattering albedo tint (what the dust reflects)
        Vec3 absorption;      // absorption tint (what the dust eats)
    };
    std::vector<FogVolume> fogVolumes_;

    mutable std::vector<Eigen::Vector4f> skyDataCache_;
    mutable size_t skyDataCacheW_ = 0;
    mutable size_t skyDataCacheH_ = 0;
    mutable uint64_t skyDataCacheVersion_ = ~0ull;
    std::atomic<uint64_t> skyboxVersion_{0};

    ///@brief Queue storing background tasks
    mutable std::queue<std::function<void()>> taskQueue_;
    
    ///@brief Mutex protecting the background task queue
    mutable std::mutex taskMutex_;
    
    ///@brief Condition variable to wake the background worker thread
    mutable std::condition_variable taskCV_;
    
    ///@brief Background thread handling async tasks (streaming, physics, offloading)
    std::thread workerThread_;
    
    ///@brief Atomic flag signaling the worker thread to stop
    std::atomic<bool> stopWorker_{false};
    
    ///@brief Toggles automatic background optimization of the octree
    std::atomic<bool> autoOptimize_{true};
    
    ///@brief Prevents multiple overlapping streaming requests
    std::atomic<bool> streamingQueued_{false};
    
    ///@brief Incremental counter for keeping track of rendering frames
    std::atomic<uint32_t> frameCounter_{0};

    ///@brief Minimum volume threshold for generating LOD blocks
    float minLodVolume_ = 0.0f;
    
    ///@brief Minimum size threshold for generating LOD blocks
    float minLodSize_ = 0.0f;
    
    ///@brief Threshold point count for a region to trigger offloading or splitting
    size_t regionTargetPoints_ = 4096;

    ///@brief Tracks loaded nodes that require active physics simulation
    std::vector<std::weak_ptr<NodeData>> activePhysicsNodes_;
    
    ///@brief Mutex safeguarding the active physics nodes list
    std::mutex physicsMutex_;
    
    ///@brief Directory path where the octree live data offload regions are stored
    std::string storagepath = ".";

    ///@brief Distance scalar for particle interactions
    float phys_smoothingRadius = 0.2f;
    
    ///@brief Base fluid density constant
    float phys_restDensity = 1000.0f;
    
    ///@brief Gas pressure constant for physics calculations
    float phys_gasConstant = 2000.0f;
    
    ///@brief Coefficient of viscosity
    float phys_viscosity = 200.0f;
    
    ///@brief Drag multiplier to progressively slow particles
    float phys_velocityDamping = 0.5f;
    
    ///@brief Ambient air density value for drag calculations
    float phys_airDensity = 1.225f;
    
    ///@brief Base global directional gravity vector
    Vec3 phys_gravity{0.0f, -9.81f, 0.0f};

    ///@brief SPH math kernel state based on smoothing radius
    SPHKernels kernels_{phys_smoothingRadius};
    
    ///@brief Toggle for radial point gravity vs directional gravity
    bool phys_useGravityPoint = true;
    
    ///@brief The origin point for point-based gravity
    Vec3 phys_gravityCenter{0.0f, 0.0f, 0.0f};
    
    ///@brief Intensity of the gravity vector
    float phys_gravityStrength = 9.81f;
    bool phys_solidBoundary = true;
    
    ///@brief Flag indicating the physics colliders need spatial reorganization
    std::atomic<bool> physicsCollidersDirty_{true};

    ///@brief Submits a node to be background saved and released from RAM
    ///@param node The node to offload
    void lazilyOffload(OctreeNode* node) {
        {
            u_lock lock(node->nodeMutex);
            if (!node->isLoaded() || node->isSaveQueued()) return;

            node->setSaveQueued(true);
            node->setLoadQueued(false);
        }

        enqueueTask([this, node]() {
            {
                s_lock nlock(node->nodeMutex);
                if (node->isLoaded() && node->isSaveQueued() && node->isDirty()) {
                    node->saveRegion(storagepath);
                }
            }
            node->offload();
            {
                u_lock nlock(node->nodeMutex);
                node->setSaveQueued(false);
            }
        });
    }

    ///@brief Validates that a node is loaded, pulling it from disk if needed
    ///@param node The node to evaluate
    ///@param asyncLoad If true, loads the node in the background thread
    void ensureLoaded(OctreeNode* node, bool asyncLoad = false) {
        {
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
                    u_lock nlock(node->nodeMutex);
                    if (!node->isLoaded()) {
                        node->loadRegion(storagepath);
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
                u_lock nlock(node->nodeMutex);
                if (!node->isLoaded()) node->loadRegion(storagepath);
                node->setLoadQueued(false);
            }
            if (node->isLoaded()) {
                ensureLOD(node);
            }
        }
    }

    ///@brief Initializes and launches the asynchronous worker thread
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

    ///@brief Halts the asynchronous worker thread and waits for it to join
    void stopWorkerThread() {
        stopWorker_.store(true);
        taskCV_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    ///@brief Computes the combined bounding box of multiple points
    ///@param nodes List of node data pointers
    ///@return The overall bounding box enclosing all points
    inline BoundingBox getNodesBounds(const std::vector<std::shared_ptr<NodeData>>& nodes) const {
        if (nodes.empty()) return {Vec3::Zero(), Vec3::Zero()};
        BoundingBox bounds = nodes[0]->getCubeBounds();
        for (size_t i = 1; i < nodes.size(); ++i) {
            BoundingBox cb = nodes[i]->getCubeBounds();
            bounds.first = bounds.first.cwiseMin(cb.first);
            bounds.second = bounds.second.cwiseMax(cb.second);
        }
        return bounds;
    }
    
    ///@brief Recursively searches for the deepest shared parent octree node containing the defined Min and Max positions
    ///@param Min the lower-bound corner of the search region
    ///@param Max the upper-bound corner of the search region
    ///@param current The current evaluation node
    ///@param depth Tracks the recursive depth offset output
    ///@return A pointer to the highest shared node
    inline OctreeNode* getHighestCommonNodeRecursive(const Vec3& Min, const Vec3& Max, OctreeNode* current, int& depth) const {
        depth++;
        s_lock lock(current->nodeMutex);
        uint8_t mcell = getOctant(Min, current->center);
        if (mcell == getOctant(Max, current->center) && current->children[mcell]) {
            return getHighestCommonNodeRecursive(Min, Max, current->children[mcell].get(), depth);
        }
        depth--;
        return current;
    }

    ///@brief Searches for the deepest shared parent octree node encapsulating the given bounding box
    ///@param bounds The bounding box to encompass
    ///@param current The current evaluation node
    ///@param currentDepth Current depth layer
    ///@param outDepth Tracks the final derived recursive depth
    ///@return A pointer to the highest shared node
    OctreeNode* getHighestCommonNode(const BoundingBox& bounds, OctreeNode* current, int currentDepth, int& outDepth) const {
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

    ///@brief Derives the bounds from a list of positions and finds the deepest common node
    ///@param positions Set of 3D spatial positions
    ///@param current The current search node
    ///@param depth Reference to write final relative tree depth
    ///@return A pointer to the deepest shared node
    OctreeNode* getHighestCommonNode(const std::vector<Vec3>& positions, OctreeNode* current = nullptr, int& depth = 0) const {
        if (!current) current = root_.get();
        Vec3 min = positions[0];
        Vec3 max = positions[0];
        for (const auto& pos : positions) {
            min = min.cwiseMin(pos);
            max = max.cwiseMax(pos);
        }

        return getHighestCommonNodeRecursive(min, max, current, depth);
    }

    ///@brief Derives the bounds from a list of nodes and finds the deepest common node
    ///@param nodes Set of NodeData instances
    ///@param current The current search node
    ///@param depth Reference to write final relative tree depth
    ///@return A pointer to the deepest shared node
    OctreeNode* getHighestCommonNode(const std::vector<std::shared_ptr<NodeData>>& nodes, OctreeNode* current = nullptr, int& depth = 0) const {
        if (!current) current = root_.get();
        Vec3 min = nodes[0]->position;
        Vec3 max = nodes[0]->position;
        for (const auto& node : nodes) {
            min = min.cwiseMin(node->position);
            max = max.cwiseMax(node->position);
        }
        return getHighestCommonNodeRecursive(min, max, current, depth);
    }

    ///@brief Recursively drops every point matching the given objectId starting from a node
    ///@param node The starting search node
    ///@param objectId The internal ID to erase
    ///@return The number of erased points
    inline size_t removeObjectBatchRecursive(OctreeNode* node, int objectId) {
        if (!node) return 0;
        ensureLoaded(node, false);
        size_t removed = 0;
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            int oldSize = node->points.size();
            std::erase_if(node->points, [objectId](const auto& pt) {
                return pt->objectId == objectId;
            });
            removed += oldSize - node->points.size();
        }
        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                if (child) {
                    removed += removeObjectBatchRecursive(child.get(), objectId);
                }
            }
        }
        
        if (removed > 0) {
            u_lock lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        return removed;
    }
    
    ///@brief Fully purges an object by ID from the tree and registry
    ///@param objectId The target object ID
    ///@return True if successfully removed at least some related components
    bool removeObject(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        OctreeNode* startNode = collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) return false;

        size_t removed = removeObjectBatchRecursive(startNode, objectId);
        size -= removed;

        {
            u_lock lock(objectsMutex_);
            objects_.erase(objectId);
        }
        return true;
    }

    ///@brief Erases specific nodes dynamically without needing bounds search for each
    ///@param node The common ancestor node to parse from
    ///@param nodesToRemove The mapped instances to delete
    ///@return The number of elements deleted
    size_t removeSpecificNodesBatchRecursive(OctreeNode* node, const std::unordered_set<std::shared_ptr<NodeData>>& nodesToRemove) {
        if (!node || nodesToRemove.empty()) return 0;
        ensureLoaded(node, false);
        size_t removed = 0;
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            int oldSize = node->points.size();
            std::erase_if(node->points, [nodesToRemove](const auto& pt) {
                nodesToRemove.find(pt) == nodesToRemove.end();
            });
            removed += oldSize - node->points.size();
        }
        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                if (child) {
                    removed += removeSpecificNodesBatchRecursive(child.get(), nodesToRemove);
                }
            }
        }

        if (removed > 0) {
            u_lock lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        return removed;
    }

public:
    ///@brief Look up an existing GridObject or allocate a new one if missing
    ///@param id The target object ID, -1 generates an auto incremented new ID
    ///@return Shared pointer to the designated GridObject
    std::shared_ptr<GridObject> getOrCreateObject(int id) {
        if (id < 0) id = nextObjId++;
        u_lock lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        auto obj = std::make_shared<GridObject>(id);
        objects_[id] = obj;
        return obj;
    }

    ///@brief Extracts a loaded object record if available
    ///@param id The object internal ID
    ///@return Shared pointer to GridObject or nullptr if missing
    std::shared_ptr<GridObject> getObject(int id) const {
        s_lock lock(objectsMutex_);
        auto it = objects_.find(id);
        if (it != objects_.end()) return it->second;
        return nullptr;
    }

    ///@brief Looks up a render material by its grid-global index
    ///@param idx Index into the grid-wide render material store
    ///@return The material, or a default RenderMaterial if idx is out of range
    RenderMaterial getRenderMaterial(uint32_t idx) const {
        return renderMaterials_.get(idx);
    }

    ///@brief Interns a render material in the grid-global store, deduplicating
    ///@param mat The material to insert or look up
    ///@return The grid-global index for the material
    uint32_t getOrAddRenderMaterial(const RenderMaterial& mat) {
        return renderMaterials_.getOrAdd(mat);
    }
    
    ///@brief Computes and returns the environmental cached skybox flattened array
    ///@param outW Updates variable to the width of the rendered skybox
    ///@param outH Updates variable to the height of the rendered skybox
    ///@return Flattened RGBA list representing the mapped sky texture
    const std::vector<Eigen::Vector4f>& getCachedSkyData(size_t& outW, size_t& outH) {
        size_t skyW = skybox_.skybox.getWidth();
        size_t skyH = skybox_.skybox.getHeight();
        if (skyW == 0 || skyH == 0) { skyW = 1; skyH = 1; }

        uint64_t ver = skyboxVersion_.load(std::memory_order_relaxed);
        if (ver != skyDataCacheVersion_ || skyW != skyDataCacheW_ || skyH != skyDataCacheH_) {
            skyDataCache_.assign(skyW * skyH, Eigen::Vector4f(0, 0, 0, 1));
            if (skybox_.skybox.getWidth() > 0) {
                for (size_t y = 0; y < skyH; ++y) {
                    float v = (static_cast<float>(y) + 0.5f) / skyH;
                    for (size_t x = 0; x < skyW; ++x) {
                        float u = (static_cast<float>(x) + 0.5f) / skyW;
                        Vec3 skyDir = skybox_.uvToDir(u, v);
                        Vec3 color = skybox_.sampleVector(skyDir);
                        skyDataCache_[y * skyW + x] = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
                    }
                }
            }
            skyDataCacheW_ = skyW;
            skyDataCacheH_ = skyH;
            skyDataCacheVersion_ = ver;
        }
        outW = skyDataCacheW_;
        outH = skyDataCacheH_;
        return skyDataCache_;
    }
    
    ///@brief Registers a celestial body (like a sun) into the skybox map
    ///@param id A unique internal tracker ID
    ///@param dir A normalized vector pointing toward the body
    ///@param angularRadius Apparent size of the object in sky rendering
    ///@param r Red emission factor
    ///@param g Green emission factor
    ///@param b Blue emission factor
    ///@param emittance Multiplier scalar for body radiance
    void addSkyBody(int id, const Vec3& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance = 255) {
        skybox_.addBody(id, dir, angularRadius, r, g, b, emittance);
        skyboxVersion_++;
    }

    ///@brief Re-aims an existing sky body direction vector
    ///@param id The internal tracker ID
    ///@param newDir The updated unit vector direction
    void moveSkyBody(int id, const Vec3& newDir) {
        skybox_.moveBody(id, newDir);
        skyboxVersion_++;
    }

    ///@brief Erases a celestial object from the skybox
    ///@param id The specific tracker ID
    void removeSkyBody(int id) {
        skybox_.removeBody(id);
        skyboxVersion_++;
    }

    ///@brief Precalculates the luminance influence onto the static background map
    ///@param id The target celestial element
    void bakeSkyBody(int id) {
        skybox_.bakeBody(id);
        skyboxVersion_++;
    }

    ///@brief Blocks thread until all background task queues run empty
    void waitForIdle() {
        if (std::this_thread::get_id() == workerThread_.get_id()) return;
        std::promise<void> p;
        auto f = p.get_future();
        enqueueTask([&p]{ p.set_value(); });
        f.wait();
    }

    ///@brief Tunes the interaction threshold radius for voxel SPH fluid simulations
    ///@param radius Distance threshold
    void setPhysicsSmoothingRadius(float radius) {
        phys_smoothingRadius = radius;
        kernels_.update(radius);
    }
    
    ///@brief Modifies the static background gravity vector
    ///@param g The global downward drift vector
    void setPhysicsGravity(const Vec3& g) {
        phys_gravity = g;
        phys_gravityStrength = g.norm();
    }

    ///@brief Updates fluid physics drag reduction parameter
    ///@param damping The physics damping value
    void setPhysicsVelocityDamping(float damping) { phys_velocityDamping = damping; }
    ///@brief Sets physics constant that modifies SPH gas dynamics
    ///@param c New physics gas constant
    void setPhysicsGasConstant(float c) { phys_gasConstant = c; }
    ///@brief Sets physics resistance to fluid shearing flows
    ///@param v The viscosity scale
    void setPhysicsViscosity(float v) { phys_viscosity = v; }
    ///@brief Sets ideal fluid volumetric mass constant
    ///@param d Target particle rest density
    void setPhysicsRestDensity(float d) { phys_restDensity = d; }
    ///@brief Changes global atmosphere weight affecting aerodynamic bounds
    ///@param d Ambient air density value
    void setPhysicsAirDensity(float d) { phys_airDensity = d; }
    ///@brief Adjusts the coordinate point acting as a radial gravity focus
    ///@param n Target XYZ world location
    void setphys_gravityCenter(Vec3 n) { phys_gravityCenter = n; }
    ///@brief Flips physics between directional global gravity and point-source mass gravity
    ///@param use True to activate radial mass gravity
    void setPhysicsUseGravityPoint(bool use) { phys_useGravityPoint = use; }
    ///@brief Defines overall power coefficient of the gravity calculations
    ///@param s Value scaler of physical gravity pull
    void setPhysicsGravityStrength(float s) { phys_gravityStrength = s; }
    void setPhysicsSolidBoundary(bool v) { phys_solidBoundary = v; }
    bool getPhysicsSolidBoundary() const { return phys_solidBoundary; }
private:

    ///@brief Multiplier scaling how aggressively Levels of Detail decay. Increasing drops quicker
    float lodFalloffRate_ = 0.1f;
    ///@brief Reciprocal of lodFalloffRate_ for optimized computation
    float invLodf = 1 / lodFalloffRate_;
    ///@brief Absolute distance before generating coarser details
    float lodMinDistance_ = 100.0f;
    ///@brief Precalculated square of lodMinDistance_
    float lodMinDistanceSq = 100 * 100;
    ///@brief Strict clipping boundary for rendering calculations
    float maxDistance_ = lodMinDistance_ * lodMinDistance_;
    ///@brief Precalculated square of maxDistance_
    float maxDistSq_max = maxDistance_ * maxDistance_;
    ///@brief Distance beyond which data regions can be paged to disk
    float keepDistance_ = maxDistance_ * 1.2;
    ///@brief Precalculated square of keepDistance_
    float keepDistSq = keepDistance_ * keepDistance_;

    ///@brief Finds which octant index a coordinate lands into based on a core point
    ///@param point Coordinate being tested
    ///@param center The dividing pivot coordinate
    ///@return Binary bitmasked octant slot [0..7]
    inline uint8_t getOctant(const Vec3& point, const Vec3& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1) | ((point[2] >= center[2]) << 2);
    }

    ///@brief Constructs the specific cubic bounding dimensions for an explicit child index
    ///@param node The active parent node
    ///@param octant Index ID mapping the 3D grid [0..7]
    ///@return Bounding min/max array for the child zone
    inline BoundingBox createChildBounds(const OctreeNode* node, uint8_t octant) const {
        Vec3 childMin, childMax;
        const Vec3& center = node->center;
        const BoundingBox bounds = node->bounds();
        
        childMin[0] = (octant & 1) ? center[0] : bounds.first[0];
        childMax[0] = (octant & 1) ? bounds.second[0] : center[0];
        
        childMin[1] = (octant & 2) ? center[1] : bounds.first[1];
        childMax[1] = (octant & 2) ? bounds.second[1] : center[1];
        
        childMin[2] = (octant & 4) ? center[2] : bounds.first[2];
        childMax[2] = (octant & 4) ? bounds.second[2] : center[2];

        return {childMin, childMax};
    }

    ///@brief Determines if bounding box A overlaps bounding box B
    ///@param a Reference box A
    ///@param b Reference box B
    ///@return True if their volumes intersect
    inline bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const {
        return ((a.first.array() <= b.second.array()) && (a.second.array() >= b.first.array())).all();
    }

    ///@brief Verifies if bounding box Inner is totally encapsulated by box Outer
    ///@param outer Evaluation perimeter
    ///@param inner Geometry being enclosed
    ///@return True if the entire inner box resides inside the outer box limits
    inline bool boxContainsBox(const BoundingBox& outer, const BoundingBox& inner) const {
        return ((inner.first.array() >= outer.first.array()) && (inner.second.array() <= outer.second.array())).all();
    }

    ///@brief Subdivides an active leaf node into 8 smaller octant child nodes
    ///@param node The full node pending division
    ///@param depth Internal recursive tracker
    void splitNodeRecursive(OctreeNode* node, int depth) {
        std::vector<std::shared_ptr<NodeData>> keep;
        keep.reserve(node->points.size());
        u_lock lock(node->nodeMutex);
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
        }

        for (auto& pointData : node->points) {
            Vec3 c = pointData->position;
            float size = pointData->size;
            BoundingBox cubeBounds = pointData->getCubeBounds();
            uint8_t targetIndex = getOctant(c, node->center);
            if (boxContainsBox(node->children[targetIndex]->bounds(), cubeBounds)) {
                node->children[targetIndex]->points.emplace_back(std::move(pointData));
            } else {
                keep.emplace_back(std::move(pointData));
            }
        }
        node->points = std::move(keep);
        node->setLeaf(false);

        for (auto& child : node->children) {
            if (child && child->points.size() > maxPointsPerNode) {
                splitNodeRecursive(child.get(), depth + 1);
            }
        }
    }
    
    ///@brief Steps down tree paths attempting to graft new node point data accurately
    ///@param node Base node validating geometric rules
    ///@param pointData Populated element to be stored
    ///@param depth Hierarchy loop tracker
    ///@return True if insertion successfully found an appropriate branch
    inline bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        if (!node) return false;
        ensureLoaded(node);
        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxContainsBox(node->bounds(), cubeBounds)) return false;

        {
            u_lock lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        if (node->isLeaf() && node->points.size() == maxPointsPerNode) {
            splitNodeRecursive(node, depth);
        }
        u_lock lock(node->nodeMutex);

        if (node->isLeaf()) {
            node->points.emplace_back(pointData);
            node->setDirty(true);
            return true;
        } else {
            lock.unlock();
            bool insertedInChild = false;
            uint8_t targetIndex = getOctant(pointData->position, node->center);
            OctreeNode* targetChild = node->children[targetIndex].get();
            if (targetChild) {
                insertedInChild = insertRecursive(targetChild, pointData, depth);
            }
            
            if (!insertedInChild) {
                u_lock lock(node->nodeMutex);
                node->points.emplace_back(pointData);
                node->setDirty(true);
            }
            return true;
        }
    }

    ///@brief Wipes out generated cache details downward for a mutated physical space
    ///@param node Scope root node
    ///@param bounds Targeted refresh footprint
    ///@return True if any bounding intersection verified cleanup
    bool invalidateNodeLODRecursive(OctreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds(), bounds)) return false;
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

    ///@brief Proxy trigger for invalidating all LOD scales tracking a modified element
    ///@param pointData The modified memory node mapping out invalidation zone
    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        if (root_ && pointData) {
            invalidateNodeLODRecursive(root_.get(), pointData->getCubeBounds());
        }
    }

    ///@brief Repopulates LOD proxies utilizing volumetric rendering logic
    ///@param node Operation target
    void ensureLOD(OctreeNode* node) {
        ensureLoaded(node);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        if (node->lodData != nullptr) return;

        if (node->isLeaf()) {
            if (node->points.empty()) {
                auto lod = std::make_shared<NodeData>();
                lod->position = node->center;
                node->lodData = lod;
                return;
            } else if (node->points.size() == 1) {
                const auto& pt = node->points[0];
                if (pt->isActive() && pt->isVisible()) {
                    double v = static_cast<double>(pt->size) * pt->size * pt->size;
                    if (v > static_cast<double>(minLodVolume_)) {
                        node->lodData = pt;
                        return;
                    }
                }
                auto lod = std::make_shared<NodeData>();
                lod->position = node->center;
                node->lodData = lod;
                return;
            }
        }

        Vec3 avgPos = Vec3::Zero();
        Eigen::Vector4f avgColor = Eigen::Vector4f::Zero();
        Vec3 avgChromaticity = Vec3::Zero();
        float avgRoughness = 0.0;
        float avgMetallic = 0.0;
        float avgTransmission = 0.0;
        Vec3 avgSellB = Vec3::Zero();
        Vec3 avgSellC = Vec3::Zero();
        float totalVolume = 0.0;
        int count = 0;

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!item || !item->isActive() || !item->isVisible()) return;
            float v = item->size * item->size * item->size;
            if (v <= 0.0) return;

            totalVolume += v;
            avgPos += item->position * v;
            avgColor += item->color * v;
            
            RenderMaterial mat = renderMaterials_.get(item->renderMatIdx);
            
            avgChromaticity += mat.emittanceRGB() * v;
            avgRoughness += mat.roughness * v;
            avgMetallic += mat.metallic * v;
            for (int j = 0; j < 3; ++j) {
                avgSellB[j] += static_cast<float>(mat.sellB[j]) * v;
                avgSellC[j] += static_cast<float>(mat.sellC[j]) * v;
            }
            count++;
        };

        for(const auto& pt : node->points) accumulate(pt);

        for (const auto& child : node->children) {
            if (child) {
                ensureLOD(child.get());
                if (child->lodData) {
                    accumulate(child->lodData);
                }
            }
        }

        if (count > 0 && totalVolume > minLodVolume_) {
            double invVol = 1.0 / totalVolume;
            
            auto lod = std::make_shared<NodeData>();
            lod->position = (avgPos * invVol);
            lod->size = std::cbrt(totalVolume);

            lod->color = (avgColor * invVol);
            Vec3 e = avgChromaticity * float(invVol);
            Grid::v3half B = (avgSellB * invVol).cast<Eigen::half>();
            Grid::v3half C = (avgSellC * invVol).cast<Eigen::half>();
            RenderMaterial avgMat(packRGB9E5(e), float(avgRoughness * invVol),
                            float(avgMetallic * invVol), B, C);
            
            lod->renderMatIdx = renderMaterials_.getOrAdd(avgMat);
            
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

    ///@brief Drills through every sub-branch verifying block data resides in memory
    ///@param node Evaluated origin block
    void loadSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLoaded(node, true);
        s_lock lock(node->nodeMutex);
        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                loadSubtreeRecursive(child.get());
            }
        }
    }

    ///@brief Similar to loadSubtreeRecursive but explicitly invokes LOD generation concurrently
    ///@param node Evaluated origin block
    void loadAndLodSubtreeRecursive(OctreeNode* node) {
        if (!node) return;
        ensureLOD(node);
        s_lock lock(node->nodeMutex);
        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                loadAndLodSubtreeRecursive(child.get());
            }
        }
    }

    ///@brief Primary spatial distance manager pushing chunks of trees off to disk outside rendering
    ///@param node Scope analysis node
    ///@param camPos Rendering camera anchor
    ///@param camDir Rendering camera facing vector for frustum logic
    void updateStreamingRecursive(OctreeNode* node, const Vec3& camPos, const Vec3& camDir) {
        if (!node) return;
        
        float minDistSq = 0.0f;
        float maxDistSq = 0.0f;
        BoundingBox nb = node->bounds();

        for(int i = 0; i < Dim; ++i) {
            float v = camPos[i];
            float minBound = nb.first[i];
            float maxBound = nb.second[i];
            
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
        Vec3 maxPoint;
        maxPoint.x() = (camDir.x() >= 0) ? nb.second.x() : nb.first.x();
        maxPoint.y() = (camDir.y() >= 0) ? nb.second.y() : nb.first.y();
        maxPoint.z() = (camDir.z() >= 0) ? nb.second.z() : nb.first.z();
        
        if ((maxPoint - camPos).dot(camDir) < -0.05f) {
            isBehind = true;
        }
        
        if (maxDistSq <= lodMinDistanceSq) {
            loadSubtreeRecursive(node);
            return;
        }

        if (maxDistSq <= maxDistSq_max && minDistSq > lodMinDistanceSq) {
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

        if (minDistSq > lodMinDistanceSq) {
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

    ///@brief Pathfinding helper isolating specific point entities across distances
    ///@param node Starting domain
    ///@param pos The 3D position to match
    ///@param objectId Target identity filter
    ///@param tolerance Accepted drift scale
    ///@return Located physical voxel data structure or nullptr
    std::shared_ptr<NodeData> findRecursive(OctreeNode* node, const Vec3& pos, int objectId, float tolerance) {
        if (!node->contains(pos)) return nullptr;
        ensureLoaded(node, false);
        s_lock lock(node->nodeMutex);
        
        for (const auto& pointData : node->points) {
            if (pointData->objectId != objectId && objectId >= 0) continue;
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        if (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->children[octant]) {
                return findRecursive(node->children[octant].get(), pos, objectId, tolerance);
            }
        }
        return nullptr;
    }

    ///@brief Hunts and deletes explicit pointer references recursively
    ///@param node Operation bounding parent
    ///@param bounds Targeted boundary region containing element
    ///@param targetPt Specific node item pointer targeting cleanup
    ///@return True if an actual element erasure triggered
    bool removeRecursive(OctreeNode* node, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
        if (!boxIntersectsBox(node->bounds(), bounds)) return false;
        ensureLoaded(node, false);
        bool foundAny = false;
        
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            int oldSize = node->points.size();

            std::erase_if(node->points, [targetPt](const std::shared_ptr<NodeData>& pointData){
                return pointData == targetPt;
            });
            if (oldSize > node->points.size()) foundAny = true;
            if (foundAny) {
                node->lodData = nullptr; 
                node->setDirty(true);
            }
        }
        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                if (child) {
                    foundAny |= removeRecursive(child.get(), bounds, targetPt);
                }
            }
            if (foundAny) {
                u_lock lock(node->nodeMutex);
                node->lodData = nullptr;
                node->setDirty(true);
            }
        }
        return foundAny;
    }

    ///@brief Sweeps an exact geometric distance aggregating elements satisfying boundaries
    ///@param node Active boundary search step
    ///@param center The spatial pivot center for capture sphere
    ///@param radiusSq Radial mathematical max extent
    ///@param objectid Filter for collecting specific object segments
    ///@param results Target return collection passed functionally
    void searchNodeRecursive(OctreeNode* node, const Vec3& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results) {
        ensureLoaded(node, false);
        s_lock lock(node->nodeMutex);
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActive()) continue;
            
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid < 0)) {
                results.emplace_back(pointData);
            }
        }
        
        if (!node->isLeaf()) {
            for (const auto& child : node->children) {
                if (child) searchNodeRecursive(child.get(), center, radiusSq, objectid, results);
            }
        }
    }
    
    ///@brief Brute force destructive wipe of all components cascading down from a block
    ///@param node Element marked for termination
    void clearNode(OctreeNode* node) {
        if (!node) return;
        
        u_lock lock(node->nodeMutex);
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        
        for (auto& child : node->children) {
            if (child) {
                clearNode(child.get());
                child.reset(nullptr);
            }
        }
        
        node->setLeaf(true);
    }

    ///@brief Analytical data compiler walking branches mapping density metrics
    ///@param node Recursive level operator
    ///@param depth Internal loop hierarchy index
    ///@param totalNodes Metric output for node instances
    ///@param leafNodes Metric output for terminating blocks
    ///@param actualPoints Metric output of instantiated point elements
    ///@param maxTreeDepth Metric output logging the worst-case recursive drill down
    ///@param maxPointsInLeaf Metric output tracking most congested end leaf
    ///@param minPointsInLeaf Metric output tracking sparsest valid block
    ///@param lodGeneratedNodes Metric mapping proxy generator usages
    ///@param unloaded Metric capturing disk-banked regions currently dropped
    void printStatsRecursive(const OctreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes, size_t& unloaded) const {
        if (!node) return;
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (!node->isLoaded()) {
            unloaded++;
            return;
        }

        s_lock lock(node->nodeMutex);
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

    ///@brief Merges sparse nodes into shared parents restoring tree performance density
    ///@param node Evaluation starting anchor
    void optimizeRecursive(OctreeNode* node) {
        if (!node) return;
        if (!node->isLoaded() || node->isLeaf()) return; 


        for (auto& child : node->children) {
            if (child) {
                optimizeRecursive(child.get());
            }
        }

        bool childrenAreLeaves = true;
        {
            s_lock lock(node->nodeMutex);
            for (auto& child : node->children) {
                if (child && !child->isLeaf()) {
                    childrenAreLeaves = false;
                    break;
                }
            }
        }

        if (childrenAreLeaves) {
            u_lock lock(node->nodeMutex);
            bool stillLeaves = true;
            for (auto& child : node->children) {
                if (child && !child->isLeaf()) {
                    stillLeaves = false;
                    break;
                }
            }
            
            if (stillLeaves) {
                std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
                for (auto& child : node->children) {
                    if (child) {
                        s_lock childLock(child->nodeMutex);
                        allPoints.insert(allPoints.end(), child->points.begin(), child->points.end());
                    }
                }

                if (allPoints.size() <= maxPointsPerNode) {
                    node->points = std::move(allPoints);
                    for (auto& child : node->children) {
                        child.reset(nullptr);
                    }
                    node->setLeaf(true);
                    node->setDirty(true);
                    
                    node->lodData = nullptr;
                }
            }
        }
    }

    ///@brief Immediately evaluates saving/offloading rules down an entire subtree
    ///@param node Topmost region targeted for cleanup tests
    void offloadRecursive(OctreeNode* node) {
        if (!node->isLoaded()) return;
        
        size_t subPoints = node->getSubtreePointCount();
        bool fullyLoaded = node->isSubtreeFullyLoaded();
        
        if (subPoints > 0 && (subPoints <= regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
            if (node->isDirty()) {
                u_lock lock(node->nodeMutex);
                node->saveRegion(storagepath);
            }
            node->offload();
            return;
        }

        if (!node->isLeaf()) {
            for (auto& child : node->children) {
                if (child) offloadRecursive(child.get());
            }
        }
    }

    ///@brief Mathematical test identifying ray penetration path spanning a boundary box
    ///@param ray Directional unit query line
    ///@param box Defined 3D span tested
    ///@param tMin Extracted entrance magnitude
    ///@param tMax Extracted exit magnitude
    ///@return True if a legitimate pass-through happened
    inline bool rayBoxIntersect(const Ray& ray, const BoundingBox& box, float& tMin, float& tMax) const {
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

    ///@brief Granular hit registration returning precise structural collision values
    ///@param ray Penetration directional probe
    ///@param cube Optimized simplified node properties wrapper
    ///@param t Reference absorbing final entry travel distance
    ///@param normal Output yielding surface rejection angle
    ///@param hitPoint Output capturing physical absolute coordinates
    ///@param tExit Optionally extracts outgoing intersection bounds length
    ///@return True on functional penetration validation
    inline bool rayCubeIntersect(const Ray& ray, const RenderData* cube, float& t, Vec3& normal, Vec3& hitPoint, float* tExit = nullptr) const {
        float t0x = (cube->boundsMin()[0] - ray.origin[0]) * ray.invDir[0];
        float t1x = (cube->boundsMax()[0] - ray.origin[0]) * ray.invDir[0];
        if (ray.invDir[0] < 0.0f) std::swap(t0x, t1x);

        float t0y = (cube->boundsMin()[1] - ray.origin[1]) * ray.invDir[1];
        float t1y = (cube->boundsMax()[1] - ray.origin[1]) * ray.invDir[1];
        if (ray.invDir[1] < 0.0f) std::swap(t0y, t1y);

        float t0z = (cube->boundsMin()[2] - ray.origin[2]) * ray.invDir[2];
        float t1z = (cube->boundsMax()[2] - ray.origin[2]) * ray.invDir[2];
        if (ray.invDir[2] < 0.0f) std::swap(t0z, t1z);

        float tMin = std::max({t0x, t0y, t0z});
        float tMax = std::min({t1x, t1y, t1z});

        if (tExit) *tExit = tMax;

        if (tMax < std::max(0.0f, tMin) || tMax < 0.0f) {
            return false;
        }

        t = tMin < 0.0f ? tMax : tMin;
        
        hitPoint = ray.origin + ray.dir * t;
        
        Vec3 dMin = (hitPoint - cube->boundsMin()).cwiseAbs();
        Vec3 dMax = (hitPoint - cube->boundsMax()).cwiseAbs();
        
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
        
        normal = Vec3::Zero();
        normal[minAxis] = sign;
        return true;
    }

    ///@brief Traverses the octree to construct a simplified array layout for fast rendering routines
    ///@param buffer The render buffer structure to populate
    void buildRender(RenderBuffer_<T>& buffer);
    
    ///@brief Deep compiler function mapping node structures into linear contiguous render arrays
    ///@param node Operating point scope
    ///@param buffer Linearized data destination buffer
    ///@param nodeIdx Mapped relative insertion point
    ///@param localObjects Readonly copy of instantiated object metadata
    void buildRenderNodeAt(OctreeNode* node, RenderBuffer_<T>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects);
    
    ///@brief Rapid DDA-style hierarchical step iterator for extremely fast line intersection logic
    ///@param buffer Renderable linear data footprint mapping tree hierarchy
    ///@param ray Tracing origin direction
    ///@param maxDist Safety cap to prevent unconstrained distance execution
    ///@return Linear list of render properties intersected along the path sequence
    std::vector<RenderData*> fastVoxelTraverse(const RenderBuffer_<T>& buffer, const Ray& ray, float maxDist);
public:
    ///@brief Primary instantiation configuring bounding parameters mapping to disk
    ///@param minBound Coordinate floor extent corner
    ///@param maxBound Coordinate ceiling extent corner
    ///@param storagepath Relative or direct string referencing save directory
    ///@param maxPointsPerNode Scaling threshold managing recursive depth optimization
    Octree(const Vec3& minBound, const Vec3& maxBound, std::string storagepath, size_t maxPointsPerNode=8) :
            root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
            size(0), skybox_(1024, 1024), storagepath(storagepath),
            streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    ///@brief Defualt parameter-less initializer building 1.0x1.0 core unit block footprint
    Octree() : root_(std::make_unique<OctreeNode>(Vec3::Constant(-0.5f), Vec3::Constant(0.5f))), maxPointsPerNode(8), size(0), skybox_(1024, 1024), streamingQueued_(false) {
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    ///@brief Standard clean down destructor guaranteeing asynchronous elements fully suspend
    ~Octree() {
        stopWorkerThread();
    }

    ///@brief Sets disk directory managing offloaded node caching and persistent layout saves
    ///@param path Format string conforming OS directory style
    void setGridStoragePath(const std::string& path) {
        storagepath = path;
    }

    ///@brief Extracts current storage operation directory assigned to the structure
    ///@return Configured persistent file string
    const std::string& getGridStoragePath() const {
        return storagepath;
    }
    
    ///@brief Copy construction duplicating entire data structure accurately mapping references
    ///@param other Instance being cloned
    Octree(const Octree& other) : size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(other.skylight_), backgroundColor_(other.backgroundColor_), autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(other.skybox_), regionTargetPoints_(other.regionTargetPoints_),
            minLodSize_(other.minLodSize_), minLodVolume_(other.minLodVolume_) {
        if (other.root_) root_ = other.root_->clone();
        
        {
            s_lock lockOther(other.objectsMutex_);
            u_lock lockThis(objectsMutex_);
            for (const auto& pair : other.objects_) {
                objects_[pair.first] = std::make_shared<GridObject>(*pair.second);
            }
        }
        {
            s_lock lockOther(other.renderMaterials_.mutex);
            renderMaterials_.materials = other.renderMaterials_.materials;
            renderMaterials_.matMap = other.renderMaterials_.matMap;
            renderMaterials_.version = other.renderMaterials_.version;
        }
        startWorkerThread();
    }

    ///@brief Movement constructor handing off dynamic instances natively blocking old access
    ///@param other Target framework yielding data ownership gracefully
    Octree(Octree&& other) noexcept : size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            skylight_(std::move(other.skylight_)), backgroundColor_(std::move(other.backgroundColor_)),
            autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false), skybox_(std::move(other.skybox_)), regionTargetPoints_(other.regionTargetPoints_),
            minLodSize_(other.minLodSize_), minLodVolume_(other.minLodVolume_) {
        other.stopWorkerThread();
        root_ = std::move(other.root_);
        
        {
            u_lock lockOther(other.objectsMutex_);
            u_lock lockThis(objectsMutex_);
            objects_ = std::move(other.objects_);
        }
        {
            u_lock lockOther(other.renderMaterials_.mutex);
            renderMaterials_.materials = std::move(other.renderMaterials_.materials);
            renderMaterials_.matMap = std::move(other.renderMaterials_.matMap);
            renderMaterials_.version = other.renderMaterials_.version;
        }
        
        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
    }

    ///@brief Deep overriding assignment wiping internal state with new copy layout
    ///@param other The host structure defining copy overrides
    ///@return Current instance pointer modifying variables
    Octree& operator=(const Octree& other) {
        if (this == &other) return *this;
        
        stopWorkerThread();
        clear();
        
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
            s_lock lockOther(other.objectsMutex_);
            u_lock lockThis(objectsMutex_);
            objects_.clear();
            for (const auto& pair : other.objects_) {
                objects_[pair.first] = std::make_shared<GridObject>(*pair.second);
            }
        }
        {
            s_lock lockOther(other.renderMaterials_.mutex);
            u_lock lockThis(renderMaterials_.mutex);
            renderMaterials_.materials = other.renderMaterials_.materials;
            renderMaterials_.matMap = other.renderMaterials_.matMap;
            renderMaterials_.version = other.renderMaterials_.version;
        }

        startWorkerThread();
        return *this;
    }

    ///@brief Direct fast memory overriding assignment bypassing standard cloning execution paths
    ///@param other The transferring object yielding footprint control
    ///@return Updated referencing to internal pointer structure
    Octree& operator=(Octree&& other) noexcept {
        if (this == &other) return *this;

        stopWorkerThread();
        other.stopWorkerThread();

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
            u_lock lockOther(other.objectsMutex_);
            u_lock lockThis(objectsMutex_);
            objects_ = std::move(other.objects_);
        }
        {
            u_lock lockOther(other.renderMaterials_.mutex);
            u_lock lockThis(renderMaterials_.mutex);
            renderMaterials_.materials = std::move(other.renderMaterials_.materials);
            renderMaterials_.matMap = std::move(other.renderMaterials_.matMap);
            renderMaterials_.version = other.renderMaterials_.version;
        }

        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
        return *this;
    }

    ///@brief Exposes raw internal async command queuing to permit custom external routine injection
    ///@param task Closure routine prepared for background worker launch
    void enqueueTask(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(taskMutex_);
            taskQueue_.push(std::move(task));
        }
        taskCV_.notify_one();
    }

    ///@brief Forces massive disk serialization offloading currently cached block data matching rules
    void offloadRegions() {
        if (root_) offloadRecursive(root_.get());
    }

    ///@brief Adjusts atomic background periodic tree sorting optimization behavior
    ///@param v Boolean control enabling or preventing idle processing
    void setAutoOptimize(bool v) { 
        autoOptimize_.store(v); 
    }


    ///@brief Adds a world-space fog volume (an AABB of participating medium) for atmospheric dust/haze/mist. Evaluated ON THE GPU inside the
    ///       wavefront path tracer's existing medium machinery: camera and bounce rays can scatter inside the box (real light shafts, since
    ///       scattered rays are shadow-tested like everything else) and shadow rays are attenuated passing through it. Costs zero voxels.
    ///       Density is constant inside the box; layer several boxes of decreasing density to fake a height gradient, and place small
    ///       dense boxes exactly where you want visible haze.
    ///@param minB          Minimum corner of the box
    ///@param maxB          Maximum corner of the box
    ///@param density       Extinction per world unit (try 0.02-0.15; mean free
    ///                     path is 1/density world units)
    ///@param scatterColor  Scattering albedo tint, 0..1 (dust ~ (0.9,0.85,0.75))
    ///@param absorption    Absorption tint, 0..1 (usually small or zero)
    ///@return Index of the volume (for removeFogVolume)
    size_t addFogVolume(const Vec3& minB, const Vec3& maxB, float density,
                        const Vec3& scatterColor = Vec3(0.9f, 0.86f, 0.78f),
                        const Vec3& absorption = Vec3::Zero()) {
        fogVolumes_.push_back({minB.cwiseMin(maxB), minB.cwiseMax(maxB),
                               std::max(0.0f, density), scatterColor, absorption});
        return fogVolumes_.size() - 1;
    }

    ///@brief Removes a fog volume by index (indices above shift down)
    void removeFogVolume(size_t index) {
        if (index < fogVolumes_.size()) fogVolumes_.erase(fogVolumes_.begin() + index);
    }

    ///@brief Removes all fog volumes
    void clearFogVolumes() { fogVolumes_.clear(); }

    ///@brief Number of active fog volumes
    size_t fogVolumeCount() const { return fogVolumes_.size(); }

    ///@brief Applies raw directional lighting data for simplistic fallback illumination mappings
    ///@param skylight Light intensity and hue mapping block
    void setSkylight(const Vec3& skylight) { 
        skylight_ = skylight; 
    }

    ///@brief Extracts simple directional lighting settings configured internally
    ///@return Configured light block value vector
    Vec3 getSkylight() const { 
        return skylight_; 
    }

    ///@brief Rewrites absolute background space rendering color map logic
    ///@param color 3 channel rgb layout mapped between 0.0-1.0 limits
    void setBackgroundColor(const Vec3& color) { 
        backgroundColor_ = color; 
        skybox_.setBackground(color.x(), color.y(), color.z(), 1.0f);
    }

    ///@brief Gets current base skybox base coloring output fallback values
    ///@return 3 channel background ambient map mapping vector
    Vec3 getBackgroundColor() const { 
        return backgroundColor_; 
    }

    ///@brief Sets threshold scaling curve how quickly proxy objects transition resolutions
    ///@param rate Scaling curve modifier mapping value usually range bounded 0-1
    void setLODFalloff(float rate) {
        lodFalloffRate_ = rate;
        invLodf = 1 / rate;
    }

    ///@brief Minimum base offset distance before lower detail elements are created
    ///@param dist Mathematical boundary trigger boundary limit
    void setLODMinDistance(float dist) {
        lodMinDistance_ = dist;
        lodMinDistanceSq = dist * dist;
    }

    ///@brief Hard distance threshold where nodes will not render
    ///@param dist Absolute radius offset limiter value
    void setMaxDistance(float dist) {
        maxDistance_ = dist;
        keepDistance_ = dist * 1.2;
        maxDistSq_max = dist * dist;
        keepDistSq = keepDistance_ * keepDistance_;
    }
    
    ///@brief Limits detail level by defining smallest potential physical space LOD chunk bounds
    ///@param size Radius block span limiter for volumetric groupings
    void setMinLODSize(float size) {
        minLodSize_ = size;
        minLodVolume_ = size * size * size;
    }

    ///@brief Provides current minimum block geometry limits for dynamic proxy generations
    ///@return Minimum spatial diameter configured threshold
    float getMinLODSize() const { return minLodSize_; }
    ///@brief Determines cluster size thresholds that justify background offloading cycles saving to disk
    ///@param points Threshold count of contained components necessary to compress blocks out
    void setRegionTargetPoints(size_t points) { regionTargetPoints_ = points; }
    ///@brief Readout for current node points chunk limit that dictates memory paging rules
    ///@return Size count limiting memory retention boundaries
    size_t getRegionTargetPoints() const { return regionTargetPoints_; }

    ///@brief Forces explicit re-evaluation of all node hierarchies creating structural block representations
    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }

    ///@brief Generates and inserts a precise physical mapped voxel piece defining simulation interactions
    ///@param data Underlying template generic content map
    ///@param pos Absolute mapped world center relative tracking coordinate
    ///@param visible Render availability flag exposing element in output queries
    ///@param color Direct fallback visual representation channel layout mapping array (RGB)
    ///@param size Radius limit encapsulating specific single block point scale
    ///@param active Toggle setting node computation behaviors toggled off to optimize
    ///@param objectId Relational grouped item tag locking multiple segments functionally together
    ///@param emittance Multiplier representing physical block glowing intensity limits
    ///@param roughness Rendering surface roughness/scattering property
    ///@param metallic Rendering surface conductive property limit scale
    ///@param transmission Rendering light penetration multiplier transparency
    ///@param ior Optical physical density refraction limits configuring lens simulations
    ///@param absorp Multi-channel internal chromatic shifting filtering logic
    ///@param bType Physical dynamics rigidity limits mapping particle simulation properties
    ///@param mass Fundamental gravitational inertial mass multiplier physics scale
    ///@param stiffness Restorative internal force metric mapping elastic bouncing rules
    ///@param breakForce Critical threshold limits tearing structure physics interactions apart
    ///@param damping Frictional movement dampening slowing kinetic physics motions
    ///@return True on successful octree structure insertion
    bool insert(const T& data, const Vec3& pos, bool visible, Vec3 color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, float metallic = 0.0f, float transmission = 0.0f,
             float ior = 1.45f, Vec3 absorp = Vec3::Zero(), BodyType bType = BodyType::STATIC, float mass = 1.0f,
             float stiffness = 4000.0f, float breakForce = 60.0f, float damping = 0.4f) {
        if (!pos.allFinite() || !root_->contains(pos)) {
            return false;
        }
        auto obj = getOrCreateObject(objectId);
        RenderMaterial rmat(emittance, roughness, metallic, ior, absorp);
        uint32_t rIdx = renderMaterials_.getOrAdd(rmat);
        
        PhysicsMaterial_ pmat{bType, mass, stiffness, breakForce, damping};
        uint16_t pIdx = obj->getOrAddPhysicsMaterial(pmat);
        Eigen::Vector4f color4(color.x(), color.y(), color.z(), std::clamp(1.0f - transmission, 0.0f, 1.0f));

        auto pointData = std::make_shared<NodeData>(data, pos, visible, color4, size, active, objectId, rIdx, pIdx, bType == BodyType::STATIC);
        
        Vec3 relPos = pos - obj->centerPosition;
        {
            u_lock lock(obj->objMutex);
            obj->relativeVoxels.push_back({relPos});
        }
        
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            if (bType != BodyType::STATIC) {
                std::lock_guard<std::mutex> lock(physicsMutex_);
                activePhysicsNodes_.push_back(pointData);
            }
            if (bType == BodyType::RIGID) {
                bondRigidVoxel(pointData);
            }
            return true;
        }
        return false;
    }
    
    ///@brief Generates explicit connected rigid internal constraints mapping rigid structures
    ///@param node Element scanning nearby segments looking for identical mapping relationships
    void bondRigidVoxel(const std::shared_ptr<NodeData>& node) {
        float reach = node->size * 1.8f;
        auto neighbors = findInRadius(node->position, reach, -1);

        float strength = 60.0f;
        if (auto obj = getObject(node->objectId))
            strength = obj->getPhysicsMaterial(node->physMatIdx).breakForce;

        for (auto& nb : neighbors) {
            if (nb.get() == node.get() || !nb->isActive()) continue;

            BodyType nbType = BodyType::STATIC;
            if (auto obj = getObject(nb->objectId))
                nbType = obj->getPhysicsMaterial(nb->physMatIdx).type;

            float restLen = (node->position - nb->position).norm();
            if (restLen < 1e-5f) continue;

            if (nbType == BodyType::STATIC) {
                if (nb->objectId != node->objectId) node->physics.bonds.push_back({nb, restLen, strength, true});
            } else if (nbType == BodyType::RIGID && nb->objectId == node->objectId) {
                node->physics.bonds.push_back({nb, restLen, strength, false});
                nb->physics.bonds.push_back({node, restLen, strength, false});
            }
        }
        node->physics.bondsBuilt = true;
    }

    ///@brief Extracts a registered rendering ID profile mapped from coordinates specific location index
    ///@param pos The 3D location to search for component
    ///@param tolerance Permissive distance allowance bounding precise location
    ///@return Valid material mapping index array offset or -1 on invalid block
    int getRenderMaterialIndex(const Vec3& pos, float tolerance = 0.0001f) {
        auto pt = find(pos, tolerance);
        if (!pt) return -1;
        return pt->renderMatIdx;
    }

    int getPhysicsMaterialIndex(const Vec3& pos, float tolerance = 0.0001f) {
        auto pt = find(pos, tolerance);
        if (!pt) return -1;
        return pt->physMatIdx;
    }
    
    void collectNodesByObjectId(int id, std::vector<std::shared_ptr<NodeData>>& results) {
        auto obj = getObject(id);
        if (!obj) return;
        
        std::vector<Vec3> absolutePositions;
        {
            s_lock lock(obj->objMutex);
            if (obj->relativeVoxels.empty()) return;
            
            absolutePositions.reserve(obj->relativeVoxels.size());
            for (const auto& relPos : obj->relativeVoxels) {
                absolutePositions.push_back(obj->centerPosition + relPos.relPos);
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

    bool updateRenderMaterial(int objectId, uint32_t index, const RenderMaterial& mat) {
        {
            u_lock lock(renderMaterials_.mutex);
            if (index >= renderMaterials_.materials.size()) return false;
            renderMaterials_.matMap.erase(renderMaterials_.materials[index]);
            renderMaterials_.materials[index] = mat;
            renderMaterials_.matMap[mat] = index;
            ++renderMaterials_.version;
        }
        std::vector<std::shared_ptr<NodeData>> nodes;
        if (root_) collectNodesByObjectId(objectId, nodes);
        for (auto& n : nodes) invalidateLODForPoint(n);
        return true;
    }
    
    bool updatePhysicsMaterial(int objectId, uint16_t index, const PhysicsMaterial_& pmat) {
        auto obj = getObject(objectId);
        if (!obj) return false;
        {
            u_lock lock(obj->objMutex);
            if (index >= obj->physicsMaterials.size()) return false;
            obj->physicsMaterialIndex.erase(obj->physicsMaterials[index]);
            obj->physicsMaterials[index] = pmat;
            obj->physicsMaterialIndex[pmat] = index;
        }
        markPhysicsCollidersDirty();
        return true;
    }

    bool rotateObject(int objectId, const Eigen::Matrix3f& rotation, const Vec3& pivot) {
        if (!root_) return false;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int oldDepth = 0;
        OctreeNode* oldStart = getHighestCommonNode(oldBounds, root_.get(), 0, oldDepth);

        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        for (auto& n : nodes) {
            Vec3 offset = n->position - pivot;
            n->position = pivot + (rotation * offset);
        }

        BoundingBox newBounds = getNodesBounds(nodes);

        int newDepth = 0;
        OctreeNode* newStart = getHighestCommonNode(newBounds, root_.get(), 0, newDepth);

        size_t added = 0;
        for (auto& n : nodes) {
            if (insertRecursive(newStart, n, newDepth)) {
                added++;
            }
        }
        size += added;
        
        auto obj = getObject(objectId);
        if (obj) {
            u_lock lock(obj->objMutex);
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
        OctreeNode* oldStart = collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) return false;

        int oldDepth = 0;

        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        size_t added = 0;
        for (auto& n : nodes) {
            float newSize = n->size * 0.5f;
            float offset = newSize * 0.5f;
            
            for (int i = 0; i < 8; ++i) {
                Vec3 newPos = n->position;
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
        collectNodesByObjectId(objectId, nodes);

        std::unordered_set<std::shared_ptr<NodeData>> toRemove;
        std::vector<std::shared_ptr<NodeData>> toRemoveVec;
        
        for (auto& n : nodes) {
            int exposedFaces = 0;
            float step = n->size;
            const std::array<Vec3, 6> dirs = {
                Vec3(1, 0, 0), Vec3(-1, 0, 0),
                Vec3(0, 1, 0), Vec3(0, -1, 0),
                Vec3(0, 0, 1), Vec3(0, 0, -1)
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
            OctreeNode* start = getHighestCommonNode(remBounds, root_.get(), 0, remDepth);
            size_t removed = removeSpecificNodesBatchRecursive(start, toRemove);
            size -= removed;
        }
        
        return true;
    }

    void queuedset(const T& data, const Vec3& pos, bool visible, Vec3 color, float size = 0.01f, bool active = true,
             int objectId = -1, float emittance = 0.0f, float roughness = 1.0f, float metallic = 0.0f, float transmission = 0.0f,
             float ior = 1.45f, Vec3 absorp = Vec3::Zero(),
             BodyType bType = BodyType::STATIC, float mass = 1.0f) {
        enqueueTask([this, data, pos, visible, color, size, active, objectId, emittance, roughness, metallic, transmission, ior, absorp, bType, mass]() {
            if (!pos.allFinite() || !root_->contains(pos)) {
                return;
            }
            auto obj = getOrCreateObject(objectId);
            RenderMaterial mat(emittance, roughness, metallic, ior, absorp);
            uint32_t rIdx = renderMaterials_.getOrAdd(mat);
            
            PhysicsMaterial_ pmat{bType, mass};
            uint16_t pIdx = obj->getOrAddPhysicsMaterial(pmat);

            Eigen::Vector4f color4(color.x(), color.y(), color.z(), std::clamp(1.0f - transmission, 0.0f, 1.0f));
            auto pointData = std::make_shared<NodeData>(data, pos, visible, color4, size, active, objectId, rIdx, pIdx, bType == BodyType::STATIC);
            
            Vec3 relPos = pos - obj->centerPosition;
            {
                u_lock lock(obj->objMutex);
                obj->relativeVoxels.push_back({relPos});
            }

            if (insertRecursive(root_.get(), pointData, 0)) {
                this->size++;
                if (bType != BodyType::STATIC) {
                    std::lock_guard<std::mutex> lock(physicsMutex_);
                    activePhysicsNodes_.push_back(pointData);
                }
                if (bType == BodyType::RIGID) {
                    bondRigidVoxel(pointData);
                }
            }
        });
    }

    void updateStreaming(const Camera& cam) {
        if (streamingQueued_.exchange(true, std::memory_order_acquire)) return;
        Vec3 camPos = cam.origin;
        Vec3 camDir = cam.direction.normalized();
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
        writeVal(out, magic);
        writeVal(out, maxPointsPerNode);
        writeVal(out, size);
        writeVal(out, regionTargetPoints_);
        
        writeVec3(out, skylight_);
        writeVec3(out, backgroundColor_);
        
        writeVec3(out, root_->bounds().first);
        writeVec3(out, root_->bounds().second);

        {
            s_lock matLock(renderMaterials_.mutex);
            uint32_t numRMat = renderMaterials_.materials.size();
            writeVal(out, numRMat);
            for (const auto& mat : renderMaterials_.materials) {
                writeVal(out, mat);
            }
        }

        {
            s_lock lock(objectsMutex_);
            uint32_t numObjects = objects_.size();
            writeVal(out, numObjects);
            for (const auto& pair : objects_) {
                writeVal(out, pair.first);
                auto obj = pair.second;
                
                s_lock objLock(obj->objMutex);
                writeVal(out, obj->objectFlags);
                writeVec3(out, obj->centerPosition);
                
                uint32_t numPMat = obj->physicsMaterials.size();
                writeVal(out, numPMat);
                for (const auto& pmat : obj->physicsMaterials) {
                    writeVal(out, pmat);
                }
            }
        }

        u_lock rlock(root_->nodeMutex);
        root_->serialize(out, regionTargetPoints_, storagepath);
        
        out.close();
        std::cout << "successfully saved grid to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        uint32_t magic;
        readVal(in, magic);
        if (magic != 0x79676733) {
            std::cerr << "Invalid Octree file format" << std::endl;
            return false;
        }

        readVal(in, maxPointsPerNode);
        readVal(in, size);
        readVal(in, regionTargetPoints_);
        
        readVec3(in, skylight_);
        readVec3(in, backgroundColor_);

        Vec3 minBound, maxBound;
        readVec3(in, minBound);
        readVec3(in, maxBound);

        {
            u_lock matLock(renderMaterials_.mutex);
            renderMaterials_.materials.clear();
            renderMaterials_.matMap.clear();
            uint32_t numRMat = 0;
            readVal(in, numRMat);
            renderMaterials_.materials.resize(numRMat);
            for (uint32_t j = 0; j < numRMat; ++j) {
                readVal(in, renderMaterials_.materials[j]);
                renderMaterials_.matMap[renderMaterials_.materials[j]] = j;
            }
            ++renderMaterials_.version;
        }

        {
            u_lock lock(objectsMutex_);
            objects_.clear();
            uint32_t numObjects = 0;
            readVal(in, numObjects);
            for (uint32_t i = 0; i < numObjects; ++i) {
                int id;
                readVal(in, id);
                auto obj = std::make_shared<GridObject>(id);
                readVal(in, obj->objectFlags);
                readVec3(in, obj->centerPosition);
                
                uint32_t numPMat;
                readVal(in, numPMat);
                obj->physicsMaterials.resize(numPMat);
                for (uint32_t j = 0; j < numPMat; ++j) {
                    readVal(in, obj->physicsMaterials[j]);
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

    std::shared_ptr<NodeData> find(const Vec3& pos, int objectId = -2, float tolerance = EPSILON, OctreeNode* node = nullptr) {
        if (!node) node = root_.get();
        return findRecursive(node, pos, objectId, tolerance);
    }

    std::shared_ptr<NodeData> findwNode(const Vec3& pos, OctreeNode* node, int objectId = -2, float tolerance = EPSILON) {
        return findRecursive(node, pos, objectId, tolerance);
    }

    bool inGrid(Vec3 pos) {
        return root_->contains(pos);
    }

    bool remove(const Vec3& pos, float tolerance = EPSILON) {
        auto pt = find(pos, tolerance);
        if (!pt) return false;
        if (removeRecursive(root_.get(), pt->getCubeBounds(), pt)) {
            size--;
            return true;
        }
        return false;
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const Vec3& center, float radius, int objectid = -1) {
        std::vector<std::shared_ptr<NodeData>> results;
        
        float radiusSq = radius * radius;
        int depth = 0;
        OctreeNode* startingPoint = getHighestCommonNodeRecursive(center - Vec3::Constant(radius), center + Vec3::Constant(radius), root_.get(), depth);
        searchNodeRecursive(startingPoint, center, radiusSq, objectid, results);
        
        return results;
    }
    
    void makeObjectFluid(int objectId, float newMass, BodyType newType = BodyType::FLUID) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        
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
        if (root_) collectNodesByObjectId(objectId, nodes);
        std::vector<std::weak_ptr<NodeData>> weakNodes;
        weakNodes.reserve(nodes.size());
        for (auto& n : nodes) weakNodes.push_back(n);
        return weakNodes;
    }

    bool update(const Vec3& pos, const T& newData) {
        auto pointData = find(pos);
        if (!pointData) return false;
        else pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedupdate(const Vec3 pos, const T newData) {
        enqueueTask([this, pos, newData]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node, -2);
            if (!pointData) return;
            else {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->data = newData;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool update(const Vec3& oldPos, const Vec3& newPos, const T& newData, bool newVisible = true, 
                Vec3 newColor = Vec3(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, float newEmittance = -1.0f, float newRoughness = -1.0f, 
                float newMetallic = -1.0f, float newTransmission = -1.0f, float newIor = -1.0f, float tolerance = EPSILON) {

        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;

        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        
        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->setVisible(newVisible);
        
        if (newColor != Vec3(1.0f, 1.0f, 1.0f)) {
            pointData->color.template head<3>() = newColor;
        }
        if (newSize > 0) pointData->size = newSize;
        pointData->setActive(newActive);
        pointData->objectId = targetObjId;
        
        getOrCreateObject(targetObjId);
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        
        if (newEmittance >= 0) mat.chromaticity = packRGB9E5(Vec3::Constant(newEmittance));
        if (newRoughness >= 0) mat.roughness = newRoughness;
        if (newMetallic >= 0) mat.metallic = newMetallic;
        if (newTransmission >= 0) pointData->color.w() = std::clamp(1.0f - newTransmission, 0.0f, 1.0f);
        if (newIor >= 0) sellmeierFromConstant(newIor, mat.sellB, mat.sellC);
        
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        
        bool res = insertRecursive(root_.get(), pointData, 0);
        
        if(!res) {
            size--;
        }

        return res;
    }

    bool move(const Vec3& pos, const Vec3& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
        pointData->position = newPos;

        if (insertRecursive(root_.get(), pointData, 0)) {
            return true;
        }
        size--;
        return false;
    }

    void queuedmove(const Vec3 pos, const Vec3 newPos) {
        enqueueTask([this, pos, newPos]() {
            auto pointData = find(pos);
            if (!pointData) return;

            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            pointData->position = newPos;

            if (insertRecursive(root_.get(), pointData, 0)) {
                return;
            }
            size--;
            return;
        });
    }

    void queuedupdate(const Vec3 pos, const Vec3 newPos, const T newData) {
        enqueueTask([this, pos, newPos, newData]() {
            auto pointData = find(pos);
            if (!pointData) return;
            
            removeRecursive(root_.get(), pointData->getCubeBounds(), pointData);
            
            auto newPointData = std::make_shared<NodeData>(*pointData);
            newPointData->position = newPos;
            newPointData->data = newData;
            
            if (!insertRecursive(root_.get(), newPointData, 0)) {
                size--;
            }
        });
    }

    bool setObjectId(const Vec3& pos, int objectId, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->objectId = objectId;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool updateData(const Vec3& pos, const T& newData, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setActive(const Vec3& pos, bool active, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->setActive(active);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setVisible(const Vec3& pos, bool visible, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->setVisible(visible);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setColor(const Vec3& pos, Vec3 color, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->color.template head<3>() = color;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedsetColor(const Vec3& pos, Vec3 color, float tolerance = EPSILON) {
        enqueueTask([this, pos, color, tolerance]() {
            OctreeNode* node = root_.get();
            auto pointData = findwNode(pos, node, -2, tolerance);
            if (!pointData) return;
            {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->color.template head<3>() = color;
            }
            invalidateLODForPoint(pointData);
            return;
        });
    }

    bool setEmittance(const Vec3& pos, float emittance, float tolerance = EPSILON) {
        return setEmittance(pos, Vec3::Constant(emittance), tolerance);
    }

    bool setEmittance(const Vec3& pos, const Vec3& chromaticity, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        mat.chromaticity = packRGB9E5(chromaticity);
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setIor(const Vec3& pos, float ior, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        sellmeierFromConstant(ior, mat.sellB, mat.sellC);
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setSellmeier(const Vec3& pos, const v3half& B, const v3half& C, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        mat.sellB = B;
        mat.sellC = C;
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setRoughness(const Vec3& pos, float roughness, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        mat.roughness = roughness;
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setMetallic(const Vec3& pos, float metallic, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        RenderMaterial mat = renderMaterials_.get(pointData->renderMatIdx);
        mat.metallic = metallic;
        pointData->renderMatIdx = renderMaterials_.getOrAdd(mat);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setTransmission(const Vec3& pos, float transmission, float tolerance = EPSILON) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->color.w() = std::clamp(1.0f - transmission, 0.0f, 1.0f);
        invalidateLODForPoint(pointData);
        return true;
    }

    void setMaterialByObjectId(int objectId, float emittance, float roughness, float metallic) {
        getOrCreateObject(objectId);
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        for (auto& n : nodes) {
            RenderMaterial mat = renderMaterials_.get(n->renderMatIdx);
            mat.chromaticity = packRGB9E5(Vec3::Constant(emittance));
            mat.roughness = roughness;
            mat.metallic = metallic;
            n->renderMatIdx = renderMaterials_.getOrAdd(mat);
            invalidateLODForPoint(n);
        }
    }

    bool raycast(const Vec3& origin, const Vec3& direction, float maxDist, RayHit& hit,
                 const std::shared_ptr<NodeData>& ignoreNode = nullptr, bool hitOnlySolid = false, bool resolvePenetration = false,
                 const std::vector<std::vector<PhysicsMaterial_>>* solidClassMats = nullptr) {
        if (!root_) return false;
        
        Ray ray(origin, direction.normalized());
        
        float tMin, tMax;
        if (!rayBoxIntersect(ray, root_->bounds(), tMin, tMax)) return false;
        tMax = std::min(tMax, maxDist);
        
        float currentMaxDist = maxDist;
        std::shared_ptr<NodeData> bestNode = nullptr;

        struct StackItem {
            OctreeNode* node;
            float tMin;
            float tMax;
        };
        
        StackItem stack[128]; 
        int stackPtr = 0;
        stack[stackPtr++] = {root_.get(), std::max(0.0f, tMin), tMax};

        const float ro[3] = {ray.origin.x(), ray.origin.y(), ray.origin.z()};
        const float rd_inv[3] = {ray.invDir.x(), ray.invDir.y(), ray.invDir.z()};
        const int s[3] = {ray.sign[0], ray.sign[1], ray.sign[2]};

        while(stackPtr > 0) {
            StackItem current = stack[--stackPtr];
            
            if (current.tMin > currentMaxDist) continue;

            OctreeNode* node = current.node;

            if (!node->isLoaded()) {
                ensureLoaded(node, true);
                continue;
            }

            s_lock lock(node->nodeMutex);

            for (const auto& pt : node->points) {
                if (!pt->isActive() || pt == ignoreNode) continue;
                if (hitOnlySolid) {
                    bool isFluid = false;
                    if (solidClassMats) {
                        int oi = pt->objectId + 1;
                        if (oi >= 0 && oi < (int)solidClassMats->size()) {
                            const auto& mats = (*solidClassMats)[oi];
                            if (pt->physMatIdx < mats.size() && mats[pt->physMatIdx].type == BodyType::FLUID)
                                isFluid = true;
                        }
                    } else {
                        s_lock objLock(objectsMutex_);
                        auto it = objects_.find(pt->objectId);
                        if (it != objects_.end() && it->second->getPhysicsMaterial(pt->physMatIdx).type == BodyType::FLUID)
                            isFluid = true;
                    }
                    if (isFluid) continue;
                }
                
                BoundingBox bounds = pt->getCubeBounds();
                const float b[2][3] = {
                    {bounds.first.x(), bounds.first.y(), bounds.first.z()},
                    {bounds.second.x(), bounds.second.y(), bounds.second.z()}
                };

                float tmin_pt = (b[s[0]][0] - ro[0]) * rd_inv[0];
                float tmax_pt = (b[1 - s[0]][0] - ro[0]) * rd_inv[0];

                float tymin = (b[s[1]][1] - ro[1]) * rd_inv[1];
                float tymax = (b[1 - s[1]][1] - ro[1]) * rd_inv[1];

                if ((tmin_pt > tymax) || (tymin > tmax_pt)) continue;
                if (tymin > tmin_pt) tmin_pt = tymin;
                if (tymax < tmax_pt) tmax_pt = tymax;

                float tzmin = (b[s[2]][2] - ro[2]) * rd_inv[2];
                float tzmax = (b[1 - s[2]][2] - ro[2]) * rd_inv[2];

                if ((tmin_pt > tzmax) || (tzmin > tmax_pt)) continue;
                if (tzmin > tmin_pt) tmin_pt = tzmin;
                if (tzmax < tmax_pt) tmax_pt = tzmax;

                if (tmax_pt >= std::max(0.0f, tmin_pt) && tmax_pt >= 0.0f) {
                    float t = (tmin_pt < 0.0f) ? (resolvePenetration ? 0.0f : tmax_pt) : tmin_pt;
                    
                    if (t >= 0 && t <= currentMaxDist && t <= current.tMax + 0.001f) {
                        currentMaxDist = t;
                        bestNode = pt;
                    }
                }
            }

            if (node->isLeaf()) continue;

            // Traverse children
            float t0 = current.tMin;
            float t1 = current.tMax;

            float ttt_x = (node->center.x() - ro[0]) * rd_inv[0];
            float ttt_y = (node->center.y() - ro[1]) * rd_inv[1];
            float ttt_z = (node->center.z() - ro[2]) * rd_inv[2];

            int currIdx = ((t0 >= ttt_x) ? 1 : 0) | ((t0 >= ttt_y) ? 2 : 0) | ((t0 >= ttt_z) ? 4 : 0);

            struct ChildInterval { OctreeNode* node; float tMin; float tMax; };
            ChildInterval children[4];
            int childCount = 0;

            while (t0 < t1 && t0 <= currentMaxDist) {
                float next_tx = (currIdx & 1) ? t1 : ttt_x;
                float next_ty = (currIdx & 2) ? t1 : ttt_y;
                float next_tz = (currIdx & 4) ? t1 : ttt_z;

                float tNext = std::min({next_tx, next_ty, next_tz});
                int physIdx = currIdx ^ ray.signMask;

                if (node->children[physIdx]) {
                    children[childCount++] = {node->children[physIdx].get(), t0, tNext};
                }

                t0 = tNext;
                currIdx |= ((next_tx <= tNext) ? 1 : 0) | ((next_ty <= tNext) ? 2 : 0) | ((next_tz <= tNext) ? 4 : 0);
            }

            if (stackPtr + childCount > 128) continue;

            for (int i = childCount - 1; i >= 0; --i) {
                stack[stackPtr++] = {children[i].node, children[i].tMin, children[i].tMax};
            }
        }
        
        if (bestNode) {
            hit.node = bestNode;
            hit.distance = currentMaxDist;
            hit.hitPoint = ray.origin + ray.dir * currentMaxDist;

            BoundingBox bounds = bestNode->getCubeBounds();
            Vec3 dMin = (hit.hitPoint - bounds.first).cwiseAbs();
            Vec3 dMax = (hit.hitPoint - bounds.second).cwiseAbs();
            
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
            hit.normal = Vec3::Zero();
            hit.normal[minAxis] = sign;
            return true;
        }
        
        return false;
    }

    frame fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB);
    frame blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale = 0.5f,
                frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 1,
                int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    frame superBlendedRenderFrameVulkan(const Camera& cam, int height, int width, float ptScale = 0.25f,
                frame::colormap colorformat = frame::colormap::RGB, int samplesPerPixel = 100,
                int maxBounces = 4, bool globalIllumination = true, bool useLod = false,
                int minSamplesPerPixel = 4);
    frame GameStyleRenderFrame(const Camera& cam, int height, int width,
                frame::colormap colorformat = frame::colormap::RGB);
    frame fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB);
    frame renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB,
        int samplesPerPixel = 2, int maxBounces = 4, bool globalIllumination = false, bool useLod = true);
    void stepPhysics(float dt);
    void stepRigidLattice(float dt, std::vector<std::shared_ptr<NodeData>>& rigidNodes,
                          const std::vector<std::vector<PhysicsMaterial_>>& fastMats, size_t fastMatsSize);

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

    bool empty() const { return size == 0; }

    void clear(Vec3 minBound = Vec3::Constant(-1.0), Vec3 maxBound = Vec3::Constant(1.0)) {
        if (root_) {
            clearNode(root_.get());
            root_.reset();
        }
        root_ = std::make_unique<OctreeNode>(minBound, maxBound);
        size = 0;
    }
    
    void getLoadedStatsSafe(const OctreeNode* node, size_t& loadedNodes, size_t& loadedPoints) const {
        if (!node) return;
        loadedNodes++;
        
        s_lock lock(node->nodeMutex);
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

    ///@brief stepPhysics over multiple steps to prevent issues
    ///@param dt total time (divided among steps)
    ///@param steps number of steps
    void multiStepPhysics(float dt, int steps) {
        dt = dt / steps;
        while (steps > 0) {
            stepPhysics(dt);
            steps--;
        }
    }
};
}
#endif
