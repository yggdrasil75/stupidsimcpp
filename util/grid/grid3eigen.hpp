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

    ///@brief Flat storage backing every node in the tree.
    ///       Nodes are addressed by stable uint32_t index rather than pointer.
    OctreeNodeStore<T> store_;

    ///@brief Index of the root node inside store_.
    uint32_t root_ = INVALID_IDX;

    ///@brief Convenience accessors into the flat store.
    OctreeNode* nodeAt(uint32_t idx) { return store_.ptr(idx); }
    const OctreeNode* nodeAt(uint32_t idx) const { return store_.ptr(idx); }
    bool validNode(uint32_t idx) const { return idx != INVALID_IDX; }

    ///@brief Points of a node, copied out of the point store.
    std::vector<std::shared_ptr<NodeData>> pointsOf(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n) return {};
        return store_.points.get(n->pointBlock);
    }

    ///@brief Zero-copy view of a node's points, for read-only hot paths.
    typename PointStore<T>::View pointsView(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n) return {};
        return store_.points.view(n->pointBlock);
    }

    void setPoints(uint32_t idx, const std::vector<std::shared_ptr<NodeData>>& pts) {
        OctreeNode* n = store_.ptr(idx);
        if (!n) return;
        n->pointBlock = store_.points.assign(n->pointBlock, pts);
    }

    size_t pointCountOf(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n) return 0;
        return store_.points.count(n->pointBlock);
    }

    std::shared_ptr<NodeData> lodOf(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n || n->lodIdx == INVALID_IDX) return nullptr;
        return store_.points.at(n->lodIdx, 0);
    }

    void setLod(uint32_t idx, const std::shared_ptr<NodeData>& lod) {
        OctreeNode* n = store_.ptr(idx);
        if (!n) return;
        if (!lod) { n->lodIdx = INVALID_IDX; return; }
        n->lodIdx = store_.points.addSingle(lod);
    }

    ///@brief Recursive point count over a subtree.
    size_t subtreePointCount(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n || !n->isLoaded()) return 0;
        size_t count = store_.points.count(n->pointBlock);
        if (!n->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (n->hasChild(i)) count += subtreePointCount(n->firstChild + i);
            }
        }
        return count;
    }

    ///@brief Writes one node's points to a stream.
    void writePoints(std::ofstream& out, uint32_t idx) const {
        auto pts = pointsOf(idx);
        writeVal(out, pts.size());
        for (const auto& pt : pts) {
            OctreeNode::serializeData(out, pt->data);
            writeVec3(out, pt->position);
            writeVal(out, pt->objectId);
            writeVal(out, pt->flags.load(std::memory_order_relaxed));
            writeVal(out, pt->size);
            writeVec4(out, pt->color);
            writeVal(out, pt->renderMatIdx);
            writeVal(out, pt->physMatIdx);
        }
    }

    ///@brief Reads a node's points back from a stream.
    void readPoints(std::ifstream& in, uint32_t idx) {
        size_t pointCount = 0;
        readVal(in, pointCount);
        std::vector<std::shared_ptr<NodeData>> pts;
        pts.reserve(pointCount);
        for (size_t i = 0; i < pointCount; ++i) {
            auto pt = std::make_shared<NodeData>();
            OctreeNode::deserializeData(in, pt->data);
            readVec3(in, pt->position);
            readVal(in, pt->objectId);
            uint8_t f;
            readVal(in, f);
            pt->flags.store(f, std::memory_order_relaxed);
            readVal(in, pt->size);
            readVec4(in, pt->color);
            readVal(in, pt->renderMatIdx);
            readVal(in, pt->physMatIdx);
            pts.push_back(pt);
        }
        setPoints(idx, pts);
    }

    ///@brief Recursively serializes a subtree (structure + points).
    void serializeSubtree(std::ofstream& out, uint32_t idx) const {
        const OctreeNode* n = nodeAt(idx);
        writeVal(out, n->isLeaf());
        writePoints(out, idx);
        if (!n->isLeaf()) {
            writeVal(out, n->childMask);
            for (int i = 0; i < 8; ++i) {
                if (n->hasChild(i)) serializeSubtree(out, n->firstChild + i);
            }
        }
    }

    ///@brief Recursively rebuilds a subtree, allocating child blocks as needed.
    void deserializeSubtree(std::ifstream& in, uint32_t idx) {
        bool leaf;
        readVal(in, leaf);
        nodeAt(idx)->setLeaf(leaf);
        readPoints(in, idx);

        if (!leaf) {
            uint8_t childMask;
            readVal(in, childMask);
            uint32_t first = store_.allocChildren();
            OctreeNode* n = nodeAt(idx);
            n->firstChild = first;
            n->childMask = childMask;
            Vec3 c = n->center;
            BoundingBox b = n->bounds();

            for (int i = 0; i < 8; ++i) {
                if ((childMask >> i) & 1) {
                    Vec3 childMin, childMax;
                    for (int d = 0; d < Dim; ++d) {
                        bool high = (i >> d) & 1;
                        childMin[d] = high ? c[d] : b.first[d];
                        childMax[d] = high ? b.second[d] : c[d];
                    }
                    store_[first + i] = OctreeNode(childMin, childMax);
                    deserializeSubtree(in, first + i);
                }
            }
        }
        OctreeNode* n = nodeAt(idx);
        n->setLoaded(true);
        n->setDirty(false);
    }

    void clearDirtySubtree(uint32_t idx) {
        OctreeNode* n = nodeAt(idx);
        if (!n) return;
        n->setDirty(false);
        if (!n->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (n->hasChild(i)) clearDirtySubtree(n->firstChild + i);
            }
        }
    }

    ///@brief Saves a node's subtree to its own region file.
    bool saveRegion(uint32_t idx) {
        const OctreeNode* n = nodeAt(idx);
        if (!n) return false;
        std::ofstream out(n->getRegionPath(storagepath), std::ios::binary);
        if (!out) return false;
        serializeSubtree(out, idx);
        clearDirtySubtree(idx);
        return true;
    }

    ///@brief Loads a node's subtree back from its region file.
    bool loadRegion(uint32_t idx) {
        OctreeNode* n = nodeAt(idx);
        if (!n) return false;
        std::ifstream in(n->getRegionPath(storagepath), std::ios::binary);
        if (!in) return false;
        deserializeSubtree(in, idx);
        nodeAt(idx)->setLoaded(true);
        return true;
    }

    ///@brief Deep-copies a subtree from another octree's store into this one.
    uint32_t cloneSubtree(const Octree& other, uint32_t srcIdx) {
        const OctreeNode* src = other.store_.ptr(srcIdx);
        if (!src) return INVALID_IDX;

        uint32_t dst = store_.add(*src);
        OctreeNode* d = nodeAt(dst);
        d->firstChild = INVALID_IDX;
        d->pointBlock = INVALID_IDX;
        d->lodIdx = INVALID_IDX;

        auto pts = other.store_.points.get(src->pointBlock);
        if (!pts.empty()) setPoints(dst, pts);

        if (!src->isLeaf() && src->firstChild != INVALID_IDX) {
            uint32_t first = store_.allocChildren();
            nodeAt(dst)->firstChild = first;
            nodeAt(dst)->childMask = src->childMask;
            for (int i = 0; i < 8; ++i) {
                if (!((src->childMask >> i) & 1)) continue;
                uint32_t sub = cloneSubtree(other, src->firstChild + i);
                if (sub != INVALID_IDX) {
                    store_[first + i] = store_[sub];
                }
            }
        }
        return dst;
    }

    bool subtreeFullyLoaded(uint32_t idx) const {
        const OctreeNode* n = store_.ptr(idx);
        if (!n || !n->isLoaded()) return false;
        if (!n->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (n->hasChild(i) && !subtreeFullyLoaded(n->firstChild + i)) return false;
            }
        }
        return true;
    }
    
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
    void lazilyOffload(uint32_t idx) {
        (void)idx;
    }

    ///@brief Validates that a node is loaded, pulling it from disk if needed
    ///@param node The node to evaluate
    ///@param asyncLoad If true, loads the node in the background thread
    void ensureLoaded(uint32_t idx, bool asyncLoad = false) {
        OctreeNode* node = nodeAt(idx);
        if (!node) return;
        if (node->isLoaded() || node->isQueued()) return;

        {
            u_lock nlock(store_.stripe(idx));
            node->setLoadQueued(true);
            node->setSaveQueued(false);
        }

        if (asyncLoad) {
            enqueueTask([this, idx]() {
                bool justLoaded = false;
                {
                    u_lock nlock(store_.stripe(idx));
                    OctreeNode* n = nodeAt(idx);
                    if (n && !n->isLoaded()) {
                        loadRegion(idx);
                        justLoaded = n->isLoaded();
                    }
                    if (n) n->setLoadQueued(false);
                }
                if (justLoaded) ensureLOD(idx);
            });
        } else {
            {
                u_lock nlock(store_.stripe(idx));
                OctreeNode* n = nodeAt(idx);
                if (n && !n->isLoaded()) loadRegion(idx);
                if (n) n->setLoadQueued(false);
            }
            if (nodeAt(idx)->isLoaded()) ensureLOD(idx);
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
    inline uint32_t getHighestCommonNodeRecursive(const Vec3& Min, const Vec3& Max, uint32_t current, int& depth) const {
        const OctreeNode* n = nodeAt(current);
        if (!n) return current;
        depth++;
        uint8_t mcell = getOctant(Min, n->center);
        if (mcell == getOctant(Max, n->center) && n->hasChild(mcell)) {
            return getHighestCommonNodeRecursive(Min, Max, n->firstChild + mcell, depth);
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
    uint32_t getHighestCommonNode(const BoundingBox& bounds, uint32_t current, int currentDepth, int& outDepth) const {
        const OctreeNode* n = nodeAt(current);
        if (!n || n->isLeaf()) {
            outDepth = currentDepth;
            return current;
        }
        for (int i = 0; i < 8; ++i) {
            if (n->hasChild(i)) {
                BoundingBox cb = createChildBounds(n, i);
                if (boxContainsBox(cb, bounds)) {
                    return getHighestCommonNode(bounds, n->firstChild + i, currentDepth + 1, outDepth);
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
    uint32_t getHighestCommonNode(const std::vector<Vec3>& positions, uint32_t current, int& depth) const {
        if (current == INVALID_IDX) current = root_;
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
    uint32_t getHighestCommonNode(const std::vector<std::shared_ptr<NodeData>>& nodes, uint32_t current, int& depth) const {
        if (current == INVALID_IDX) current = root_;
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
    inline size_t removeObjectBatchRecursive(uint32_t idx, int objectId) {
        if (idx == INVALID_IDX) return 0;
        ensureLoaded(idx, false);
        OctreeNode* node = nodeAt(idx);
        if (!node) return 0;
        size_t removed = 0;
        {
            u_lock lock(store_.stripe(idx));
            auto pts = pointsOf(idx);
            size_t oldSize = pts.size();
            std::erase_if(pts, [objectId](const auto& pt) {
                return pt && pt->objectId == objectId;
            });
            removed += oldSize - pts.size();
            if (oldSize != pts.size()) setPoints(idx, pts);
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    removed += removeObjectBatchRecursive(node->firstChild + i, objectId);
                }
            }
        }
        
        if (removed > 0) {
            u_lock lock(store_.stripe(idx));
            node->lodIdx = INVALID_IDX;
            node->setDirty(true);
        }
        return removed;
    }
    
    ///@brief Fully purges an object by ID from the tree and registry
    ///@param objectId The target object ID
    ///@return True if successfully removed at least some related components
    bool removeObject(int objectId) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        uint32_t startNode = collectNodesByObjectId(objectId, nodes);
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
    size_t removeSpecificNodesBatchRecursive(uint32_t idx, const std::unordered_set<std::shared_ptr<NodeData>>& nodesToRemove) {
        if (idx == INVALID_IDX || nodesToRemove.empty()) return 0;
        ensureLoaded(idx, false);
        OctreeNode* node = nodeAt(idx);
        if (!node) return 0;
        size_t removed = 0;
        {
            u_lock lock(store_.stripe(idx));
            auto pts = pointsOf(idx);
            size_t oldSize = pts.size();
            // Was a no-op before: the lambda never returned its predicate.
            std::erase_if(pts, [&nodesToRemove](const auto& pt) {
                return nodesToRemove.find(pt) != nodesToRemove.end();
            });
            removed += oldSize - pts.size();
            if (oldSize != pts.size()) setPoints(idx, pts);
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    removed += removeSpecificNodesBatchRecursive(node->firstChild + i, nodesToRemove);
                }
            }
        }

        if (removed > 0) {
            u_lock lock(store_.stripe(idx));
            node->lodIdx = INVALID_IDX;
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
    void splitNodeRecursive(uint32_t idx, int depth) {
        OctreeNode* node = nodeAt(idx);
        if (!node) return;

        uint32_t first = store_.allocChildren();
        if (first == INVALID_IDX) return;

        for (int i = 0; i < 8; ++i) {
            BoundingBox cb = createChildBounds(node, i);
            store_[first + i] = OctreeNode(cb.first, cb.second);
        }

        auto pts = pointsOf(idx);
        std::vector<std::shared_ptr<NodeData>> keep;
        keep.reserve(pts.size());
        std::array<std::vector<std::shared_ptr<NodeData>>, 8> buckets;

        for (auto& pointData : pts) {
            if (!pointData) continue;
            BoundingBox cubeBounds = pointData->getCubeBounds();
            uint8_t targetIndex = getOctant(pointData->position, node->center);
            if (boxContainsBox(store_[first + targetIndex].bounds(), cubeBounds)) {
                buckets[targetIndex].emplace_back(std::move(pointData));
            } else {
                keep.emplace_back(std::move(pointData));
            }
        }

        node = nodeAt(idx);
        node->firstChild = first;
        node->childMask = 0xFF;
        node->setLeaf(false);
        setPoints(idx, keep);

        for (int i = 0; i < 8; ++i) {
            if (!buckets[i].empty()) setPoints(first + i, buckets[i]);
        }

        for (int i = 0; i < 8; ++i) {
            if (pointCountOf(first + i) > maxPointsPerNode) {
                splitNodeRecursive(first + i, depth + 1);
            }
        }
    }
    
    ///@brief Steps down tree paths attempting to graft new node point data accurately
    ///@param node Base node validating geometric rules
    ///@param pointData Populated element to be stored
    ///@param depth Hierarchy loop tracker
    ///@return True if insertion successfully found an appropriate branch
    inline bool insertRecursive(uint32_t idx, const std::shared_ptr<NodeData>& pointData, int depth) {
        if (idx == INVALID_IDX) return false;
        ensureLoaded(idx);
        OctreeNode* node = nodeAt(idx);
        if (!node) return false;

        BoundingBox cubeBounds = pointData->getCubeBounds();
        if (!boxContainsBox(node->bounds(), cubeBounds)) return false;

        node->lodIdx = INVALID_IDX;

        if (node->isLeaf() && pointCountOf(idx) >= maxPointsPerNode) {
            splitNodeRecursive(idx, depth);
            node = nodeAt(idx);
        }

        if (node->isLeaf()) {
            u_lock lock(store_.stripe(idx));
            node->pointBlock = store_.points.push(node->pointBlock, pointData);
            node->setDirty(true);
            return true;
        }

        uint8_t targetIndex = getOctant(pointData->position, node->center);
        bool insertedInChild = false;
        if (node->hasChild(targetIndex)) {
            insertedInChild = insertRecursive(node->firstChild + targetIndex, pointData, depth);
        }

        if (!insertedInChild) {
            node = nodeAt(idx);
            u_lock lock(store_.stripe(idx));
            node->pointBlock = store_.points.push(node->pointBlock, pointData);
            node->setDirty(true);
        }
        return true;
    }

    ///@brief Wipes out generated cache details downward for a mutated physical space
    ///@param node Scope root node
    ///@param bounds Targeted refresh footprint
    ///@return True if any bounding intersection verified cleanup
    bool invalidateNodeLODRecursive(uint32_t idx, const BoundingBox& bounds) {
        OctreeNode* node = nodeAt(idx);
        if (!node) return false;
        if (!boxIntersectsBox(node->bounds(), bounds)) return false;
        ensureLoaded(idx);
        node = nodeAt(idx);

        std::array<uint32_t, 8> safeChildren;
        safeChildren.fill(INVALID_IDX);
        {
            u_lock lock(store_.stripe(idx));
            node->lodIdx = INVALID_IDX;
            node->setDirty(true);
            if (!node->isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (node->hasChild(i)) safeChildren[i] = node->firstChild + i;
                }
            }
        }
        
        for (int i = 0; i < 8; ++i) {
            if (safeChildren[i] != INVALID_IDX) {
                invalidateNodeLODRecursive(safeChildren[i], bounds);
            }
        }
        return true;
    }

    ///@brief Proxy trigger for invalidating all LOD scales tracking a modified element
    ///@param pointData The modified memory node mapping out invalidation zone
    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        if (root_ != INVALID_IDX && pointData) {
            invalidateNodeLODRecursive(root_, pointData->getCubeBounds());
        }
    }

    ///@brief Repopulates LOD proxies utilizing volumetric rendering logic
    ///@param node Operation target
    void ensureLOD(uint32_t idx) {
        ensureLoaded(idx);
        OctreeNode* node = nodeAt(idx);
        if (!node) return;
        if (node->lodIdx != INVALID_IDX) return;

        auto nodePoints = pointsView(idx);

        if (node->isLeaf()) {
            if (nodePoints.empty()) {
                auto lod = std::make_shared<NodeData>();
                lod->position = node->center;
                setLod(idx, lod);
                return;
            } else if (nodePoints.size() == 1) {
                const auto& pt = nodePoints[0];
                if (pt && pt->isActive() && pt->isVisible()) {
                    double v = static_cast<double>(pt->size) * pt->size * pt->size;
                    if (v > static_cast<double>(minLodVolume_)) {
                        setLod(idx, pt);
                        return;
                    }
                }
                auto lod = std::make_shared<NodeData>();
                lod->position = node->center;
                setLod(idx, lod);
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

        for (const auto& pt : nodePoints) accumulate(pt);

        {
            const OctreeNode* n = nodeAt(idx);
            if (n && !n->isLeaf()) {
                uint32_t first = n->firstChild;
                uint8_t mask = n->childMask;
                for (int i = 0; i < 8; ++i) {
                    if ((mask >> i) & 1) {
                        ensureLOD(first + i);
                        auto childLod = lodOf(first + i);
                        if (childLod) accumulate(childLod);
                    }
                }
            }
        }
        node = nodeAt(idx);

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
            setLod(idx, lod);
        } else {
            auto lod = std::make_shared<NodeData>();
            lod->position = node->center;
            setLod(idx, lod);
        }
    }

    ///@brief Drills through every sub-branch verifying block data resides in memory
    ///@param node Evaluated origin block
    void loadSubtreeRecursive(uint32_t idx) {
        if (idx == INVALID_IDX) return;
        ensureLoaded(idx, true);
        const OctreeNode* node = nodeAt(idx);
        if (!node || node->isLeaf()) return;
        for (int i = 0; i < 8; ++i) {
            if (node->hasChild(i)) loadSubtreeRecursive(node->firstChild + i);
        }
    }

    ///@brief Similar to loadSubtreeRecursive but explicitly invokes LOD generation concurrently
    ///@param node Evaluated origin block
    void loadAndLodSubtreeRecursive(uint32_t idx) {
        if (idx == INVALID_IDX) return;
        ensureLOD(idx);
        const OctreeNode* node = nodeAt(idx);
        if (!node || node->isLeaf()) return;
        for (int i = 0; i < 8; ++i) {
            if (node->hasChild(i)) loadAndLodSubtreeRecursive(node->firstChild + i);
        }
    }

    ///@brief Primary spatial distance manager pushing chunks of trees off to disk outside rendering
    ///@param node Scope analysis node
    ///@param camPos Rendering camera anchor
    ///@param camDir Rendering camera facing vector for frustum logic
    void updateStreamingRecursive(uint32_t idx, const Vec3& camPos, const Vec3& camDir) {
        OctreeNode* node = nodeAt(idx);
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
            loadSubtreeRecursive(idx);
            return;
        }

        if (maxDistSq <= maxDistSq_max && minDistSq > lodMinDistanceSq) {
            loadAndLodSubtreeRecursive(idx);
            return;
        }
        
        if (minDistSq > keepDistSq) {
            if (!node->isLoaded()) return;
            size_t subPoints = subtreePointCount(idx);
            bool fullyLoaded = subtreeFullyLoaded(idx);

            if ((subPoints > regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
                if (subPoints > 0) lazilyOffload(idx);
                return;
            }
            if (!node->isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (node->hasChild(i)) {
                        updateStreamingRecursive(node->firstChild + i, camPos, camDir);
                    }
                }
            }
            return;
        }

        if (minDistSq > lodMinDistanceSq) {
            ensureLOD(idx);
        } else {
            ensureLoaded(idx, true);
        }

        node = nodeAt(idx);
        if (node && !node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    updateStreamingRecursive(node->firstChild + i, camPos, camDir);
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
    std::shared_ptr<NodeData> findRecursive(uint32_t idx, const Vec3& pos, int objectId, float tolerance) {
        const OctreeNode* node = nodeAt(idx);
        if (!node || !node->contains(pos)) return nullptr;
        ensureLoaded(idx, false);
        node = nodeAt(idx);

        for (const auto& pointData : pointsView(idx)) {
            if (!pointData) continue;
            if (pointData->objectId != objectId && objectId >= 0) continue;
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) return pointData;
        }

        if (!node->isLeaf()) {
            int octant = getOctant(pos, node->center);
            if (node->hasChild(octant)) {
                return findRecursive(node->firstChild + octant, pos, objectId, tolerance);
            }
        }
        return nullptr;
    }

    ///@brief Hunts and deletes explicit pointer references recursively
    ///@param node Operation bounding parent
    ///@param bounds Targeted boundary region containing element
    ///@param targetPt Specific node item pointer targeting cleanup
    ///@return True if an actual element erasure triggered
    bool removeRecursive(uint32_t idx, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
        OctreeNode* node = nodeAt(idx);
        if (!node || !boxIntersectsBox(node->bounds(), bounds)) return false;
        ensureLoaded(idx, false);
        node = nodeAt(idx);
        bool foundAny = false;
        
        {
            u_lock lock(store_.stripe(idx));
            auto pts = pointsOf(idx);
            size_t oldSize = pts.size();
            std::erase_if(pts, [&targetPt](const std::shared_ptr<NodeData>& pointData) {
                return pointData == targetPt;
            });
            if (oldSize > pts.size()) {
                foundAny = true;
                setPoints(idx, pts);
                node->lodIdx = INVALID_IDX;
                node->setDirty(true);
            }
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    foundAny |= removeRecursive(node->firstChild + i, bounds, targetPt);
                }
            }
            if (foundAny) {
                u_lock lock(store_.stripe(idx));
                node->lodIdx = INVALID_IDX;
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
    void searchNodeRecursive(uint32_t idx, const Vec3& center, float radiusSq, int objectid,
                               std::vector<std::shared_ptr<NodeData>>& results) {
        if (idx == INVALID_IDX) return;
        ensureLoaded(idx, false);
        const OctreeNode* node = nodeAt(idx);
        if (!node) return;

        for (const auto& pointData : pointsView(idx)) {
            if (!pointData || !pointData->isActive()) continue;
            float pointDistSq = (pointData->position - center).squaredNorm();
            if (pointDistSq <= radiusSq && (pointData->objectId == objectid || objectid < 0)) {
                results.emplace_back(pointData);
            }
        }
        
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    searchNodeRecursive(node->firstChild + i, center, radiusSq, objectid, results);
                }
            }
        }
    }
    
    ///@brief Brute force destructive wipe of all components cascading down from a block
    ///@param node Element marked for termination
    void clearNode(uint32_t idx) {
        OctreeNode* node = nodeAt(idx);
        if (!node) return;

        store_.points.release(node->pointBlock);
        node->pointBlock = INVALID_IDX;
        node->lodIdx = INVALID_IDX;

        if (!node->isLeaf() && node->firstChild != INVALID_IDX) {
            uint32_t first = node->firstChild;
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) clearNode(first + i);
            }
            store_.freeChildren(first);
        }

        node = nodeAt(idx);
        node->firstChild = INVALID_IDX;
        node->childMask = 0;
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
    void printStatsRecursive(uint32_t idx, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints,
                            size_t& maxTreeDepth, size_t& maxPointsInLeaf, size_t& minPointsInLeaf, size_t& lodGeneratedNodes, size_t& unloaded) const {
        const OctreeNode* node = nodeAt(idx);
        if (!node) return;
        
        totalNodes++;
        maxTreeDepth = std::max(maxTreeDepth, depth);

        if (!node->isLoaded()) {
            unloaded++;
            return;
        }

        if (node->lodIdx != INVALID_IDX) lodGeneratedNodes++;

        size_t pts = store_.points.count(node->pointBlock);
        actualPoints += pts;

        if (node->isLeaf()) {
            leafNodes++;
            maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
            minPointsInLeaf = std::min(minPointsInLeaf, pts);
        } else {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    printStatsRecursive(node->firstChild + i, depth + 1, totalNodes, leafNodes, actualPoints,
                                        maxTreeDepth, maxPointsInLeaf, minPointsInLeaf, lodGeneratedNodes, unloaded);
                }
            }
        }
    }

    ///@brief Merges sparse nodes into shared parents restoring tree performance density
    ///@param node Evaluation starting anchor
    void optimizeRecursive(uint32_t idx) {
        OctreeNode* node = nodeAt(idx);
        if (!node) return;
        if (!node->isLoaded() || node->isLeaf()) return;

        uint32_t first = node->firstChild;
        uint8_t mask = node->childMask;

        for (int i = 0; i < 8; ++i) {
            if ((mask >> i) & 1) optimizeRecursive(first + i);
        }

        node = nodeAt(idx);
        if (node->isLeaf()) return;

        bool childrenAreLeaves = true;
        for (int i = 0; i < 8; ++i) {
            if (node->hasChild(i) && !store_[node->firstChild + i].isLeaf()) {
                childrenAreLeaves = false;
                break;
            }
        }
        if (!childrenAreLeaves) return;

        u_lock lock(store_.stripe(idx));
        std::vector<std::shared_ptr<NodeData>> allPoints = pointsOf(idx);
        for (int i = 0; i < 8; ++i) {
            if (node->hasChild(i)) {
                auto cp = pointsOf(node->firstChild + i);
                allPoints.insert(allPoints.end(), cp.begin(), cp.end());
            }
        }

        if (allPoints.size() <= maxPointsPerNode) {
            setPoints(idx, allPoints);
            uint32_t block = node->firstChild;
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) {
                    OctreeNode& c = store_[block + i];
                    store_.points.release(c.pointBlock);
                    c.pointBlock = INVALID_IDX;
                    c.lodIdx = INVALID_IDX;
                }
            }
            store_.freeChildren(block);
            node = nodeAt(idx);
            node->firstChild = INVALID_IDX;
            node->childMask = 0;
            node->setLeaf(true);
            node->setDirty(true);
            node->lodIdx = INVALID_IDX;
        }
    }

    ///@brief Immediately evaluates saving/offloading rules down an entire subtree
    ///@param node Topmost region targeted for cleanup tests
    ///@brief Walks the tree saving dirty regions to disk.
    void offloadRecursive(uint32_t idx) {
        OctreeNode* node = nodeAt(idx);
        if (!node || !node->isLoaded()) return;

        size_t subPoints = subtreePointCount(idx);
        bool fullyLoaded = subtreeFullyLoaded(idx);

        if (subPoints > 0 && (subPoints <= regionTargetPoints_ || node->isLeaf()) && fullyLoaded) {
            if (node->isDirty()) {
                u_lock lock(store_.stripe(idx));
                saveRegion(idx);
            }
            return;
        }

        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i)) offloadRecursive(node->firstChild + i);
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
    void buildRenderNodeAt(uint32_t nodeIndex, RenderBuffer_<T>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects);
    
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
            maxPointsPerNode(maxPointsPerNode),
            size(0), skybox_(1024, 1024), storagepath(storagepath),
            streamingQueued_(false) {
        root_ = store_.add(OctreeNode(minBound, maxBound));
        skybox_.setBackground(backgroundColor_.x(), backgroundColor_.y(), backgroundColor_.z(), 1.0f);
        startWorkerThread();
    }

    ///@brief Defualt parameter-less initializer building 1.0x1.0 core unit block footprint
    Octree() : maxPointsPerNode(8), size(0), skybox_(1024, 1024), streamingQueued_(false) {
        root_ = store_.add(OctreeNode(Vec3::Constant(-0.5f), Vec3::Constant(0.5f)));
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
        if (other.root_ != INVALID_IDX) root_ = cloneSubtree(other, other.root_);
        
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
        root_ = other.root_;
        other.root_ = INVALID_IDX;
        
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

        if (other.root_ != INVALID_IDX) root_ = cloneSubtree(other, other.root_);
        
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
        
        root_ = other.root_;
        other.root_ = INVALID_IDX;
        
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
        if (root_ != INVALID_IDX) offloadRecursive(root_);
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
        if (root_ == INVALID_IDX) return;
        ensureLOD(root_);
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
        if (!pos.allFinite() || !nodeAt(root_)->contains(pos)) {
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
        
        if (insertRecursive(root_, pointData, 0)) {
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
    
    uint32_t collectNodesByObjectId(int id, std::vector<std::shared_ptr<NodeData>>& results, int* outDepth = nullptr) {
        auto obj = getObject(id);
        if (!obj) return INVALID_IDX;
        
        std::vector<Vec3> absolutePositions;
        {
            s_lock lock(obj->objMutex);
            if (obj->relativeVoxels.empty()) return INVALID_IDX;
            
            absolutePositions.reserve(obj->relativeVoxels.size());
            for (const auto& relPos : obj->relativeVoxels) {
                absolutePositions.push_back(obj->centerPosition + relPos.relPos);
            }
        }
        int depth = 0;
        uint32_t commonNode = getHighestCommonNode(absolutePositions, root_, depth);
        if (outDepth) {
            *outDepth = depth;
        }
        results.reserve(absolutePositions.size());
        
        for (const auto& absPos : absolutePositions) {
            auto pt = find(absPos, id, EPSILON, commonNode);
            
            if (pt && pt->isActive()) {
                results.push_back(pt);
            }
        }
        return commonNode;
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
        if (root_ != INVALID_IDX) collectNodesByObjectId(objectId, nodes);
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
        if (root_ == INVALID_IDX) return false;
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) return false;

        BoundingBox oldBounds = getNodesBounds(nodes);
        int oldDepth = 0;
        uint32_t oldStart = getHighestCommonNode(oldBounds, root_, 0, oldDepth);

        size_t removed = removeObjectBatchRecursive(oldStart, objectId);
        size -= removed;

        for (auto& n : nodes) {
            Vec3 offset = n->position - pivot;
            n->position = pivot + (rotation * offset);
        }

        BoundingBox newBounds = getNodesBounds(nodes);

        int newDepth = 0;
        uint32_t newStart = getHighestCommonNode(newBounds, root_, 0, newDepth);

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
        if (root_ == INVALID_IDX) return false;
        std::vector<std::shared_ptr<NodeData>> nodes;
        int oldDepth = 0;
        uint32_t oldStart = collectNodesByObjectId(objectId, nodes, &oldDepth);
        if (nodes.empty()) return false;

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
            uint32_t start = getHighestCommonNode(remBounds, root_, 0, remDepth);
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
            if (!pos.allFinite() || !nodeAt(root_)->contains(pos)) {
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

            if (insertRecursive(root_, pointData, 0)) {
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
            if (root_ != INVALID_IDX) {
                updateStreamingRecursive(root_, camPos, camDir);
            }
            streamingQueued_.store(false, std::memory_order_release);
        });
    }

    bool save(const std::string& filename) {
        if (root_ == INVALID_IDX) return false;

        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x79676733;
        writeVal(out, magic);
        writeVal(out, maxPointsPerNode);
        writeVal(out, size);
        writeVal(out, regionTargetPoints_);
        
        writeVec3(out, skylight_);
        writeVec3(out, backgroundColor_);
        
        writeVec3(out, nodeAt(root_)->bounds().first);
        writeVec3(out, nodeAt(root_)->bounds().second);

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

        serializeSubtree(out, root_);
        
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

        store_.clear();
        root_ = store_.add(OctreeNode(minBound, maxBound));
        deserializeSubtree(in, root_);

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> find(const Vec3& pos, int objectId = -2, float tolerance = EPSILON, uint32_t node = INVALID_IDX) {
        if (node == INVALID_IDX) node = root_;
        return findRecursive(node, pos, objectId, tolerance);
    }

    std::shared_ptr<NodeData> findwNode(const Vec3& pos, uint32_t node, int objectId = -2, float tolerance = EPSILON) {
        return findRecursive(node, pos, objectId, tolerance);
    }

    bool inGrid(Vec3 pos) {
        return nodeAt(root_)->contains(pos);
    }

    bool remove(const Vec3& pos, float tolerance = EPSILON) {
        auto pt = find(pos, -2, tolerance);
        if (!pt) return false;
        if (removeRecursive(root_, pt->getCubeBounds(), pt)) {
            size--;
            return true;
        }
        return false;
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const Vec3& center, float radius, int objectid = -1) {
        std::vector<std::shared_ptr<NodeData>> results;
        
        float radiusSq = radius * radius;
        int depth = 0;
        uint32_t startingPoint = getHighestCommonNodeRecursive(center - Vec3::Constant(radius), center + Vec3::Constant(radius), root_, depth);
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
        if (root_ != INVALID_IDX) collectNodesByObjectId(objectId, nodes);
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
            uint32_t node = root_;
            auto pointData = findwNode(pos, node, -2);
            if (!pointData) return;
            else {
                u_lock lock(store_.stripe(node));
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
        
        removeRecursive(root_, pointData->getCubeBounds(), pointData);
        
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
        
        bool res = insertRecursive(root_, pointData, 0);
        
        if(!res) {
            size--;
        }

        return res;
    }

    bool move(const Vec3& pos, const Vec3& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        removeRecursive(root_, pointData->getCubeBounds(), pointData);
        pointData->position = newPos;

        if (insertRecursive(root_, pointData, 0)) {
            return true;
        }
        size--;
        return false;
    }

    void queuedmove(const Vec3 pos, const Vec3 newPos) {
        enqueueTask([this, pos, newPos]() {
            auto pointData = find(pos);
            if (!pointData) return;

            removeRecursive(root_, pointData->getCubeBounds(), pointData);
            pointData->position = newPos;

            if (insertRecursive(root_, pointData, 0)) {
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
            
            removeRecursive(root_, pointData->getCubeBounds(), pointData);
            
            auto newPointData = std::make_shared<NodeData>(*pointData);
            newPointData->position = newPos;
            newPointData->data = newData;
            
            if (!insertRecursive(root_, newPointData, 0)) {
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
            uint32_t node = root_;
            auto pointData = findwNode(pos, node, -2, tolerance);
            if (!pointData) return;
            {
                u_lock lock(store_.stripe(node));
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
        if (root_ == INVALID_IDX) return false;

        Ray ray(origin, direction.normalized());
        
        float tMin, tMax;
        if (!rayBoxIntersect(ray, nodeAt(root_)->bounds(), tMin, tMax)) return false;
        tMax = std::min(tMax, maxDist);
        
        float currentMaxDist = maxDist;
        std::shared_ptr<NodeData> bestNode = nullptr;

        struct StackItem {
            uint32_t node;
            float tMin;
            float tMax;
        };
        
        StackItem stack[128];
        int stackPtr = 0;
        stack[stackPtr++] = {root_, std::max(0.0f, tMin), tMax};

        const float ro[3] = {ray.origin.x(), ray.origin.y(), ray.origin.z()};
        const float rd_inv[3] = {ray.invDir.x(), ray.invDir.y(), ray.invDir.z()};
        const int s[3] = {ray.sign[0], ray.sign[1], ray.sign[2]};

        while(stackPtr > 0) {
            StackItem current = stack[--stackPtr];
            
            if (current.tMin > currentMaxDist) continue;

            uint32_t nodeIdx = current.node;
            OctreeNode* node = nodeAt(nodeIdx);
            if (!node) continue;

            if (!node->isLoaded()) {
                ensureLoaded(nodeIdx, true);
                continue;
            }

            for (const auto& pt : pointsView(nodeIdx)) {
                if (!pt || !pt->isActive() || pt == ignoreNode) continue;
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

            struct ChildInterval { uint32_t node; float tMin; float tMax; };
            ChildInterval children[4];
            int childCount = 0;

            while (t0 < t1 && t0 <= currentMaxDist) {
                float next_tx = (currIdx & 1) ? t1 : ttt_x;
                float next_ty = (currIdx & 2) ? t1 : ttt_y;
                float next_tz = (currIdx & 4) ? t1 : ttt_z;

                float tNext = std::min({next_tx, next_ty, next_tz});
                int physIdx = currIdx ^ ray.signMask;

                if (node->hasChild(physIdx)) {
                    children[childCount++] = {node->firstChild + (uint32_t)physIdx, t0, tNext};
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
        if (root_ != INVALID_IDX) {
            optimizeRecursive(root_);
            generateLODs();
        }
    }

    void printStats(std::ostream& os = std::cout) const {
        if (root_ == INVALID_IDX) {
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

        printStatsRecursive(root_, 0, totalNodes, leafNodes, actualPoints, 
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
        os << "  Min               : [" << nodeAt(root_)->bounds().first.transpose() << "]\n";
        os << "  Max               : [" << nodeAt(root_)->bounds().second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }

    void clear(Vec3 minBound = Vec3::Constant(-1.0), Vec3 maxBound = Vec3::Constant(1.0)) {
        if (root_ != INVALID_IDX) {
            clearNode(root_);
            clearNode(root_);
        }
        store_.clear();
        root_ = store_.add(OctreeNode(minBound, maxBound));
        size = 0;
    }
    
    void getLoadedStatsSafe(uint32_t idx, size_t& loadedNodes, size_t& loadedPoints) const {
        const OctreeNode* node = nodeAt(idx);
        if (!node) return;
        loadedNodes++;

        if (!node->isLoaded()) return;

        loadedPoints += store_.points.count(node->pointBlock);
        if (!node->isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (node->hasChild(i))
                    getLoadedStatsSafe(node->firstChild + i, loadedNodes, loadedPoints);
            }
        }
    }

    size_t getEstimatedMemoryUsageMB() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_, loadedNodes, loadedPoints);
        
        size_t nodeMem = loadedNodes * sizeof(OctreeNode);
        size_t pointMem = loadedPoints * (sizeof(NodeData) + sizeof(std::shared_ptr<NodeData>));
        
        return (nodeMem + pointMem) / (1024 * 1024);
    }

    size_t getLoadedPointCount() const {
        size_t loadedNodes = 0;
        size_t loadedPoints = 0;
        getLoadedStatsSafe(root_, loadedNodes, loadedPoints);
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
