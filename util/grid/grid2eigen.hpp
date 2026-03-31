#ifndef g2eigen
#define g2eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
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

#ifdef SSE
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

#ifndef gridheaders
#define gridheaders
static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
static constexpr uint8_t STATIC_BIT = 1 << 7;

static constexpr uint8_t LEAF_BIT = 1 << 0;
static constexpr uint8_t LOADED_BIT = 1 << 1;
static constexpr uint8_t DIRTY_BIT = 1 << 2;
static constexpr uint8_t LOADQUEUED = 1 << 3;
static constexpr uint8_t SAVEDQUEUED = 1 << 4;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
#endif

template<typename T, typename IndexType = uint16_t>
class Quadtree {
public:
    using PointType = Eigen::Matrix<float, 2, 1>;
    using BoundingBox = std::pair<PointType, PointType>;

    struct Vector3fCompare {
        bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
            return std::tie(a.x(), a.y(), a.z()) < std::tie(b.x(), b.y(), b.z());
        }
    };
    
    struct NodeData {
        T data;
        PointType position;
        int objectId;
        float size;
        IndexType colorIdx;
        IndexType layer;
        uint8_t flags;

        NodeData(const T& data, const PointType& pos, bool visible, IndexType colorIdx, float size = 0.01f,
                 bool active = true, int objectId = -1, IndexType layer = 0, bool staticbit = 0) 
                : data(data), position(pos), objectId(objectId), size(size), 
                  colorIdx(colorIdx), layer(layer), flags(0) {
            setActive(active);
            setVisible(visible);
            setStatic(staticbit);
        }
        
        NodeData() : objectId(-1), size(0.0f), colorIdx(0), layer(0), flags(0) {}

        bool isActive() const {
            return flags & ACTIVE_BIT;
        }
        bool isVisible() const {
            return flags & VISIBLE_BIT;
        }
        bool isStatic() const {
            return flags & STATIC_BIT;
        }
        bool isActiveAndVisible() const {
            return (flags & (ACTIVE_BIT | VISIBLE_BIT)) != (ACTIVE_BIT | VISIBLE_BIT);
        }

        void setActive(bool v) {
            if (v) flags |= ACTIVE_BIT;
            else flags &= ~ACTIVE_BIT;
        }
        void setVisible(bool v) {
            if (v) flags |= VISIBLE_BIT;
            else flags &= ~VISIBLE_BIT;
        }
        void setStatic(bool v) {
            if (v) flags |= STATIC_BIT;
            else flags &= ~STATIC_BIT;
        }
        
        PointType getHalfSize() const {
            return PointType(size * 0.5f, size * 0.5f);
        }
        
        BoundingBox getSquareBounds() const {
            PointType halfSize = getHalfSize();
            return {position - halfSize, position + halfSize};
        }
    };

    struct QuadtreeNode {
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::unique_ptr<QuadtreeNode>, 4> children;
        PointType center;
        float nodeSize;
        uint8_t flags;
        
        mutable std::shared_ptr<NodeData> lodData;
        mutable std::shared_mutex nodeMutex;

        QuadtreeNode(const PointType& min, const PointType& max) : bounds(min,max), flags(0), lodData(nullptr) {
            setLeaf(true);
            setLoaded(true);
            setDirty(true);
            setLoadQueued(false);
            setSaveQueued(false);
            for (std::unique_ptr<QuadtreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5f;
            nodeSize = (bounds.second - bounds.first).norm();
        }

        bool isLeaf() const {
            return flags & LEAF_BIT;
        }
        bool isLoaded() const {
            return flags & LOADED_BIT;
        }
        bool isDirty() const {
            return flags & DIRTY_BIT;
        }
        bool isQueued() const {
            return flags & LOADQUEUED;
        }
        bool isSaveQueued() const {
            return flags & SAVEDQUEUED;
        }

        void setLeaf(bool v) {
            if (v) flags |= LEAF_BIT;
            else flags &= ~LEAF_BIT;
        }
        void setLoaded(bool v) {
            if (v) flags |= LOADED_BIT;
            else flags &= ~LOADED_BIT;
        }
        void setDirty(bool v) {
            if (v) flags |= DIRTY_BIT;
            else flags &= ~DIRTY_BIT;
        }
        void setLoadQueued(bool v) {
            if (v) flags |= LOADQUEUED;
            else flags &= ~LOADQUEUED;
        }
        void setSaveQueued(bool v) {
            if (v) flags |= SAVEDQUEUED;
            else flags &= ~SAVEDQUEUED;
        }

        bool contains(const PointType& point) const {
            return (point[0] >= bounds.first[0] && point[0] <= bounds.second[0] &&
                    point[1] >= bounds.first[1] && point[1] <= bounds.second[1]);
        }

        bool isEmpty() const {
            if (!points.empty()) return false;
            if (!isLeaf()) {
                for (int i = 0; i < 4; ++i) {
                    if (children[i] && !children[i]->isEmpty()) return false;
                }
            }
            return true;
        }

        std::string getRegionPath(const std::string& basePath) const {
            int64_t cx = static_cast<int64_t>(std::floor(center.x()));
            int64_t cy = static_cast<int64_t>(std::floor(center.y()));
            int64_t s = static_cast<int64_t>(std::floor(nodeSize));
            
            fs::path p(basePath);
            p /= std::to_string(s);
            p /= std::to_string(cx);
            p /= std::to_string(cy);
            
            std::error_code ec;
            fs::create_directories(p, ec);
            
            p /= "data.region";
            return p.string();
        }

        template<typename V>
        static void writeVal(std::ofstream& out, const V& val) {
            out.write(reinterpret_cast<const char*>(&val), sizeof(V));
        }

        template<typename V>
        static void readVal(std::ifstream& in, V& val) {
            in.read(reinterpret_cast<char*>(&val), sizeof(V));
        }

        static void writeVec2(std::ofstream& out, const Eigen::Vector2f& vec) {
            writeVal(out, vec.x());
            writeVal(out, vec.y());
        }

        static void readVec2(std::ifstream& in, Eigen::Vector2f& vec) {
            float x, y;
            readVal(in, x);
            readVal(in, y);
            vec = Eigen::Vector2f(x, y);
        }

        static void writeVec3(std::ofstream& out, const Eigen::Vector3f& vec) {
            writeVal(out, vec.x());
            writeVal(out, vec.y());
            writeVal(out, vec.z());
        }

        static void readVec3(std::ifstream& in, Eigen::Vector3f& vec) {
            float x, y, z;
            readVal(in, x);
            readVal(in, y);
            readVal(in, z);
            vec = Eigen::Vector3f(x, y, z);
        }

        static void serializeData(std::ofstream& out, const T& data) {
            if constexpr (is_shared_ptr<T>::value) {
                bool hasData = (data != nullptr);
                writeVal(out, hasData);
                if (hasData) data->serialize(out);
            } else if constexpr (std::is_pointer_v<T>) {
                bool hasData = (data != nullptr);
                writeVal(out, hasData);
                if (hasData) data->serialize(out);
            } else {
                writeVal(out, data);
            }
        }

        static void deserializeData(std::ifstream& in, T& data) {
            if constexpr (is_shared_ptr<T>::value) {
                bool hasData;
                readVal(in, hasData);
                if (hasData) {
                    using ElemType = typename T::element_type;
                    data = ElemType::deserialize(in);
                } else {
                    data = nullptr;
                }
            } else if constexpr (std::is_pointer_v<T>) {
                bool hasData;
                readVal(in, hasData);
                if (hasData) {
                    using ElemType = std::remove_pointer_t<T>;
                    data = ElemType::deserialize(in);
                } else {
                    data = nullptr;
                }
            } else {
                readVal(in, data);
            }
        }

        void serializeSubtree(std::ofstream& out) const {
            writeVal(out, isLeaf());
            writeVal(out, points.size());
            for (const auto& pt : points) {
                serializeData(out, pt->data);
                writeVec2(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->flags);
                writeVal(out, pt->size);
                writeVal(out, pt->colorIdx);
                writeVal(out, pt->layer);
            }

            if (!isLeaf()) {
                uint8_t childMask = 0;
                for (int i = 0; i < 4; ++i)
                    if (children[i]) childMask |= (1 << i);
                writeVal(out, childMask);
                for (int i = 0; i < 4; ++i) {
                    if (children[i]) children[i]->serializeSubtree(out);
                }
            }
        }

        void deserializeSubtree(std::ifstream& in) {
            bool leaf;
            readVal(in, leaf);
            setLeaf(leaf);

            size_t pointCount;
            readVal(in, pointCount);
            points.reserve(pointCount);
            for (size_t i = 0; i < pointCount; ++i) {
                auto pt = std::make_shared<NodeData>();
                deserializeData(in, pt->data);
                readVec2(in, pt->position);
                readVal(in, pt->objectId);
                readVal(in, pt->flags);
                readVal(in, pt->size);
                readVal(in, pt->colorIdx);
                readVal(in, pt->layer);
                points.push_back(pt);
            }

            if (!isLeaf()) {
                uint8_t childMask;
                readVal(in, childMask);
                for (int i = 0; i < 4; ++i) {
                    if ((childMask >> i) & 1) {
                        PointType childMin, childMax;
                        for (int d = 0; d < 2; ++d) {
                            bool high = (i >> d) & 1;
                            childMin[d] = high ? center[d] : bounds.first[d];
                            childMax[d] = high ? bounds.second[d] : center[d];
                        }
                        children[i] = std::make_unique<QuadtreeNode>(childMin, childMax);
                        children[i]->deserializeSubtree(in);
                    } else {
                        children[i] = nullptr;
                    }
                }
            }
            setLoaded(true);
            setDirty(false);
        }

        void clearDirtySubtree() {
            setDirty(false);
            if (!isLeaf()) {
                for (auto& child : children) if (child) child->clearDirtySubtree();
            }
        }

        bool saveRegion(const std::string& basePath) {
            std::string path = getRegionPath(basePath);
            std::ofstream out(path, std::ios::binary);
            if (!out) return false;
            serializeSubtree(out);
            clearDirtySubtree();
            return true;
        }

        bool loadRegion(const std::string& basePath) {
            std::string path = getRegionPath(basePath);
            std::ifstream in(path, std::ios::binary);
            if (in) {
                deserializeSubtree(in);
                setLoaded(true);
                return true;
            }
            return false;
        }

        void offload() {
            std::lock_guard<std::shared_mutex> lock(nodeMutex);
            if (isDirty()) return;
            setLoaded(false);
            for (int i = 0; i < 4; ++i) {
                children[i].reset();
            }
            points.clear();
            points.shrink_to_fit();
        }

        void serialize(std::ofstream& out, float regionTargetSize, const std::string& basePath) {
            float sideLength = bounds.second.x() - bounds.first.x();
            if (sideLength <= regionTargetSize + 1e-4f) {
                if (isDirty() && isLoaded()) saveRegion(basePath);
                return;
            }

            writeVal(out, isLeaf());
            writeVal(out, points.size());
            for (const auto& pt : points) {
                serializeData(out, pt->data);
                writeVec2(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->flags);
                writeVal(out, pt->size);
                writeVal(out, pt->colorIdx);
                writeVal(out, pt->layer);
            }

            if (!isLeaf()) {
                uint8_t childMask = 0;
                for (int i = 0; i < 4; ++i)
                    if (children[i]) childMask |= (1 << i);
                writeVal(out, childMask);
                for (int i = 0; i < 4; ++i) {
                    if (children[i]) children[i]->serialize(out, regionTargetSize, basePath);
                }
            }
        }

        void deserialize(std::ifstream& in, float regionTargetSize, const std::string& basePath) {
            float sideLength = bounds.second.x() - bounds.first.x();
            if (sideLength <= regionTargetSize + 1e-4f) {
                setLoaded(false);
                setDirty(false);
                setLeaf(false);
                return;
            }

            bool leaf;
            readVal(in, leaf);
            setLeaf(leaf);

            size_t pointCount;
            readVal(in, pointCount);
            points.reserve(pointCount);
            for (size_t i = 0; i < pointCount; ++i) {
                auto pt = std::make_shared<NodeData>();
                deserializeData(in, pt->data);
                readVec2(in, pt->position);
                readVal(in, pt->objectId);
                readVal(in, pt->flags);
                readVal(in, pt->size);
                readVal(in, pt->colorIdx);
                readVal(in, pt->layer);
                points.push_back(pt);
            }

            if (!isLeaf()) {
                uint8_t childMask;
                readVal(in, childMask);
                for (int i = 0; i < 4; ++i) {
                    if ((childMask >> i) & 1) {
                        PointType childMin, childMax;
                        for (int d = 0; d < 2; ++d) {
                            bool high = (i >> d) & 1;
                            childMin[d] = high ? center[d] : bounds.first[d];
                            childMax[d] = high ? bounds.second[d] : center[d];
                        }
                        children[i] = std::make_unique<QuadtreeNode>(childMin, childMax);
                        children[i]->deserialize(in, regionTargetSize, basePath);
                    } else {
                        children[i] = nullptr;
                    }
                }
            }
            setLoaded(true);
            setDirty(false);
        }
    };

private:
    std::unique_ptr<QuadtreeNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    Eigen::Vector3f backgroundColor_ = {0.0f, 0.0f, 0.0f};
    
    std::unique_ptr<std::mutex> mapMutex_;
    std::vector<Eigen::Vector3f> colorMap_;
    std::map<Eigen::Vector3f, IndexType, Vector3fCompare> colorToIndex_;

    mutable std::queue<std::function<void()>> taskQueue_;
    mutable std::mutex taskMutex_;
    mutable std::condition_variable taskCV_;
    std::thread workerThread_;
    std::atomic<bool> stopWorker_{false};
    std::atomic<bool> autoOptimize_{true};
    std::atomic<bool> streamingQueued_{false};

    std::string basePath_ = ".";
    float regionTargetSize_ = 256;

    static std::string extractBasePath(const std::string& filename) {
        fs::path p(filename);
        p.replace_extension("");
        return p.string();
    }

    void lazilyOffload(QuadtreeNode* node) {
        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            if (!node->isLoaded() || node->isSaveQueued()) return;

            node->setSaveQueued(true);
            lock.unlock();
        }

        enqueueTask([this, node]() {
            {
                std::shared_lock<std::shared_mutex> nlock(node->nodeMutex);
                if (node->isLoaded() && node->isSaveQueued()) {
                    if (node->isDirty()) {
                        node->saveRegion(basePath_);
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

    void ensureLoaded(QuadtreeNode* node, bool asyncLoad = false) {
        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            if (node->isLoaded() || node->isQueued()) return;
            else {
                node->setLoadQueued(true);
            }
        }

        if (asyncLoad) {
            enqueueTask([this, node]() {
                bool justLoaded = false;
                {
                    std::unique_lock<std::shared_mutex> nlock(node->nodeMutex);
                    if (!node->isLoaded()) {
                        node->loadRegion(basePath_);
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
                node->loadRegion(basePath_);
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
                        task = [this]() {
                            this->optimize();
                        };
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

    uint32_t expandBits16(uint32_t v) const noexcept {
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        return v;
    }

    uint32_t morton2D_16(float x, float y) const noexcept {
        x = std::max(0.0f, std::min(1.0f, x));
        y = std::max(0.0f, std::min(1.0f, y));
        uint32_t xx = static_cast<uint32_t>(x * 65535.0f);
        uint32_t yy = static_cast<uint32_t>(y * 65535.0f);
        return (expandBits16(yy) << 1) | expandBits16(xx);
    }

    void reorderMapsZCurve() {
        if (!root_) return;
        std::lock_guard<std::mutex> lock(*mapMutex_);
        
        size_t numColors = colorMap_.size();
        if (numColors == 0) return;

        std::vector<PointType> colorSum(numColors, PointType::Zero());
        std::vector<size_t> colorCount(numColors, 0);
        std::unordered_set<NodeData*> visited;
        std::vector<QuadtreeNode*> stack;
        stack.push_back(root_.get());
        while (!stack.empty()) {
            QuadtreeNode* curr = stack.back();
            stack.pop_back();
            if (!curr->isLoaded()) continue;
            
            for (auto& pt : curr->points) {
                if (visited.insert(pt.get()).second) {
                    if (pt->colorIdx < numColors) {
                        colorSum[pt->colorIdx] += pt->position;
                        colorCount[pt->colorIdx]++;
                    }
                }
            }
            if (!curr->isLeaf()) {
                for (int i = 0; i < 4; ++i) {
                    if (curr->children[i]) stack.push_back(curr->children[i].get());
                }
            }
        }

        PointType rootMin = root_->bounds.first;
        PointType rootSize = root_->bounds.second - root_->bounds.first;
        PointType invSize(
            rootSize.x() > 0 ? 1.0f / rootSize.x() : 1.0f,
            rootSize.y() > 0 ? 1.0f / rootSize.y() : 1.0f
        );

        struct SIdx {
            IndexType oldIdx;
            uint32_t morton;
            bool operator<(const SIdx& o) const {
                return morton < o.morton;
            }
        };

        std::vector<SIdx> sortedColors;
        sortedColors.reserve(numColors);
        for (size_t i = 0; i < numColors; ++i) {
            if (colorCount[i] > 0) {
                PointType avg = colorSum[i] / static_cast<float>(colorCount[i]);
                PointType norm = (avg - rootMin).cwiseProduct(invSize);
                sortedColors.push_back({static_cast<IndexType>(i), morton2D_16(norm.x(), norm.y())});
            } else {
                sortedColors.push_back({static_cast<IndexType>(i), 0});
            }
        }
        std::sort(sortedColors.begin(), sortedColors.end());

        std::vector<IndexType> colorRemap(numColors);
        std::vector<Eigen::Vector3f> newColorMap(numColors);
        colorToIndex_.clear();
        for (size_t newIdx = 0; newIdx < numColors; ++newIdx) {
            IndexType oldIdx = sortedColors[newIdx].oldIdx;
            colorRemap[oldIdx] = static_cast<IndexType>(newIdx);
            newColorMap[newIdx] = colorMap_[oldIdx];
            colorToIndex_[newColorMap[newIdx]] = static_cast<IndexType>(newIdx);
        }
        colorMap_ = std::move(newColorMap);

        visited.clear();
        stack.push_back(root_.get());
        while (!stack.empty()) {
            QuadtreeNode* curr = stack.back();
            stack.pop_back();

            if (!curr->isLoaded()) continue;
            
            for (auto& pt : curr->points) {
                if (visited.insert(pt.get()).second) {
                    if (pt->colorIdx < numColors) pt->colorIdx = colorRemap[pt->colorIdx];
                }
            }
            
            {
                std::unique_lock<std::shared_mutex> lodLock(curr->nodeMutex);
                curr->lodData = nullptr;
            }
            
            if (!curr->isLeaf()) {
                for (int i = 0; i < 4; ++i) {
                    if (curr->children[i]) stack.push_back(curr->children[i].get());
                }
            }
        }
    }
    
public:
    inline IndexType getColorIndex(const Eigen::Vector3f& color) {
        std::lock_guard<std::mutex> lock(*mapMutex_);
        auto it = colorToIndex_.find(color);
        if (it != colorToIndex_.end()) return it->second;
        
        if (colorMap_.size() >= std::numeric_limits<IndexType>::max()) {
            IndexType bestIdx = 0;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < colorMap_.size(); ++i) {
                float dist = (colorMap_[i] - color).squaredNorm();
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<IndexType>(i);
                }
            }
            return bestIdx;
        }

        IndexType idx = static_cast<IndexType>(colorMap_.size());
        colorMap_.push_back(color);
        colorToIndex_[color] = idx;
        return idx;
    }

    inline const Eigen::Vector3f& getColor(IndexType idx) const {
        if (idx < colorMap_.size()) return colorMap_[idx];
        static const Eigen::Vector3f fallback = Eigen::Vector3f::Zero();
        return fallback;
    }

private:

    float lodMinDistance_ = 100.0f;
    float maxDistance_ = size * size;

    uint8_t getQuadrant(const PointType& point, const PointType& center) const {
        return (point[0] >= center[0]) | ((point[1] >= center[1]) << 1);
    }

    BoundingBox createChildBounds(const QuadtreeNode* node, uint8_t quadrant) const {
        PointType childMin, childMax;
        const PointType& center = node->center;
        
        childMin[0] = (quadrant & 1) ? center[0] : node->bounds.first[0];
        childMax[0] = (quadrant & 1) ? node->bounds.second[0] : center[0];
        
        childMin[1] = (quadrant & 2) ? center[1] : node->bounds.first[1];
        childMax[1] = (quadrant & 2) ? node->bounds.second[1] : center[1];
        
        return {childMin, childMax};
    }

    bool boxIntersectsBox(const BoundingBox& a, const BoundingBox& b) const {
        return (a.first[0] <= b.second[0] && a.second[0] >= b.first[0] &&
                a.first[1] <= b.second[1] && a.second[1] >= b.first[1]);
    }
    
    bool boxContainsPoint(const BoundingBox& b, const PointType& p) const {
        return p[0] >= b.first[0] && p[0] <= b.second[0] &&
               p[1] >= b.first[1] && p[1] <= b.second[1];
    }

    void splitNode(QuadtreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 4; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<QuadtreeNode>(childBounds.first, childBounds.second);
        }

        std::vector<std::shared_ptr<NodeData>> keep;
        for (const auto& pointData : node->points) {
            // Keep massive objects in the parent
            if (pointData->size >= node->nodeSize) {
                keep.emplace_back(pointData);
                continue;
            }

            BoundingBox squareBounds = pointData->getSquareBounds();
            for (int i = 0; i < 4; ++i) {
                if (boxIntersectsBox(node->children[i]->bounds, squareBounds)) {
                    node->children[i]->points.emplace_back(pointData);
                }
            }
        }

        node->points = std::move(keep);
        node->setLeaf(false);

        for (int i = 0; i < 4; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(QuadtreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        ensureLoaded(node);

        BoundingBox squareBounds = pointData->getSquareBounds();
        if (!boxIntersectsBox(node->bounds, squareBounds)) return false;
        {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
        }

        if (!node->isLeaf() && pointData->size >= node->nodeSize) {
            std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
            node->points.emplace_back(pointData);
            node->setDirty(true);
            return true;
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
            bool inserted = false;
            for (int i = 0; i < 4; ++i) {
                BoundingBox childBounds = createChildBounds(node, i);
                if (boxIntersectsBox(childBounds, squareBounds)) {
                    if (!node->children[i]) {
                        node->children[i] = std::make_unique<QuadtreeNode>(childBounds.first, childBounds.second);
                    }
                    inserted |= insertRecursive(node->children[i].get(), pointData, depth + 1);
                }
            }
            if (inserted) {
                std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                node->setDirty(true);
            }
            return inserted;
        }
    }

    bool invalidateNodeLODRecursive(QuadtreeNode* node, const BoundingBox& bounds) {
        if (!boxIntersectsBox(node->bounds, bounds)) return false;
        ensureLoaded(node);
        {
            std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
            node->lodData = nullptr;
            node->setDirty(true);
        }
        if (!node->isLeaf()) {
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    invalidateNodeLODRecursive(node->children[i].get(), bounds);
                }
            }
        }
        return true;
    }

    void invalidateLODForPoint(const std::shared_ptr<NodeData>& pointData) {
        if (root_ && pointData) {
            invalidateNodeLODRecursive(root_.get(), pointData->getSquareBounds());
        }
    }
    
    void ensureBounds(const BoundingBox& targetBounds) {
        if (!root_) {
            PointType center = (targetBounds.first + targetBounds.second) * 0.5f;
            PointType size = targetBounds.second - targetBounds.first;
            float maxDim = size.maxCoeff();
            if (maxDim <= 0.0f) maxDim = 1.0f;
            PointType halfSize = PointType::Constant(maxDim * 0.5f);
            root_ = std::make_unique<QuadtreeNode>(center - halfSize, center + halfSize);
            return;
        }

        while (true) {
            bool xInside = root_->bounds.first.x() <= targetBounds.first.x() && root_->bounds.second.x() >= targetBounds.second.x();
            bool yInside = root_->bounds.first.y() <= targetBounds.first.y() && root_->bounds.second.y() >= targetBounds.second.y();

            if (xInside && yInside) {
                break;
            }

            PointType min = root_->bounds.first;
            PointType max = root_->bounds.second;
            PointType size = max - min;
            
            int expandX = (targetBounds.first.x() < min.x()) ? -1 : 1;
            int expandY = (targetBounds.first.y() < min.y()) ? -1 : 1;
            
            PointType newMin = min;
            PointType newMax = max;
            
            if (expandX < 0) newMin.x() -= size.x();
            else newMax.x() += size.x();
            if (expandY < 0) newMin.y() -= size.y();
            else newMax.y() += size.y();
            
            auto newRoot = std::make_unique<QuadtreeNode>(newMin, newMax);
            newRoot->setLeaf(false);
            
            uint8_t oldQuadrant = 0;
            if (expandX < 0) oldQuadrant |= 1;
            if (expandY < 0) oldQuadrant |= 2;
            
            newRoot->children[oldQuadrant] = std::move(root_);
            root_ = std::move(newRoot);
            
            maxDepth++;
        }
    }

    void ensureLOD(QuadtreeNode* node) {
        ensureLoaded(node);
        std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
        if (node->lodData != nullptr) return;

        Eigen::Vector3f avgColor = Eigen::Vector3f::Zero();
        int count = 0;
        IndexType maxLayer = 0;
        
        if (node->isLeaf() && node->points.size() == 1) {
            node->lodData = node->points[0];
            return;
        } else if (node->isLeaf() && node->points.empty()) {
            return;
        }

        auto accumulate = [&](const std::shared_ptr<NodeData>& item) {
            if (!item || !item->isActive() || !item->isVisible()) return;
            avgColor += getColor(item->colorIdx);
            if (item->layer > maxLayer) maxLayer = item->layer;
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

        if (count > 0) {
            float invCount = 1.0f / count;
            
            auto lod = std::make_shared<NodeData>();
            lod->position = node->center;
            
            PointType nodeDims = node->bounds.second - node->bounds.first;
            lod->size = nodeDims.maxCoeff();

            lod->colorIdx = getColorIndex(avgColor * invCount);
            lod->layer = maxLayer;
            
            lod->setActive(true);
            lod->setVisible(true);
            lod->objectId = -1;

            node->lodData = lod;
        }
    }

    void updateStreamingRecursive(QuadtreeNode* node, const PointType& camPos) {
        if (!node) return;

        if (!node->contains(camPos)) {
            float distSq = 0.0f;
            for(int i = 0; i < 2; ++i) {
                float v = camPos[i];
                if(v < node->bounds.first[i]) {
                    distSq += (node->bounds.first[i] - v) * (node->bounds.first[i] - v);
                } else if(v > node->bounds.second[i]) {
                    distSq += (v - node->bounds.second[i]) * (v - node->bounds.second[i]);
                }
            }
            float dist = std::sqrt(distSq);

            if (dist > maxDistance_) {
                lazilyOffload(node);
                return;
            }

            if (dist > lodMinDistance_) {
                ensureLOD(node);
                if (!node->isLeaf()) {
                    for (int i = 0; i < 4; ++i) {
                        if (node->children[i]) {
                            updateStreamingRecursive(node->children[i].get(), camPos);
                        }
                    }
                }
            } else if (!node->isLoaded()) {
                ensureLoaded(node, true);
                if (!node->isLeaf()){
                    for (int i = 0; i < 4; ++i) {
                        if (node->children[i]) {
                            updateStreamingRecursive(node->children[i].get(), camPos);
                        }
                    }
                }
            }
            return;
        }
        if (!node->isLeaf()){
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    updateStreamingRecursive(node->children[i].get(), camPos);
                }
            }
        }
    }

    void getTopMostRecursive(QuadtreeNode* node, const PointType& pos, std::shared_ptr<NodeData>& best, IndexType& maxLayer) {
        if (!node->contains(pos)) return;
        ensureLoaded(node, false);
        
        for (const auto& pointData : node->points) {
            if (!pointData->isActiveAndVisible()) continue;
            if (boxContainsPoint(pointData->getSquareBounds(), pos)) {
                if (!best || pointData->layer >= maxLayer) {
                    best = pointData;
                    maxLayer = pointData->layer;
                }
            }
        }

        if (!node->isLeaf()) {
            int quadrant = getQuadrant(pos, node->center);
            if (node->children[quadrant]) {
                getTopMostRecursive(node->children[quadrant].get(), pos, best, maxLayer);
            }
        }
    }

    std::shared_ptr<NodeData> findRecursive(QuadtreeNode* node, const PointType& pos, float tolerance) {
        if (!node->contains(pos)) return nullptr;
        ensureLoaded(node, false);
        
        for (const auto& pointData : node->points) {
            float distSq = (pointData->position - pos).squaredNorm();
            if (distSq <= tolerance * tolerance) {
                return pointData;
            }
        }

        if (!node->isLeaf()) {
            int quadrant = getQuadrant(pos, node->center);
            if (node->children[quadrant]) {
                return findRecursive(node->children[quadrant].get(), pos, tolerance);
            }
        }
        return nullptr;
    }

    bool removeRecursive(QuadtreeNode* node, const BoundingBox& bounds, const std::shared_ptr<NodeData>& targetPt) {
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
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    foundAny |= removeRecursive(node->children[i].get(), bounds, targetPt);
                }
            }
            if (foundAny) {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                node->lodData = nullptr;
                node->setDirty(true);
            }
        }
        return foundAny;
    }

    void searchNodeRecursive(QuadtreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results, std::unordered_set<std::shared_ptr<NodeData>>& seen) {
        PointType closestPoint;
        for (int i = 0; i < 2; ++i) {
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

    void searchNode(QuadtreeNode* node, const PointType& center, float radiusSq, int objectid, 
                               std::vector<std::shared_ptr<NodeData>>& results) {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        searchNodeRecursive(node, center, radiusSq, objectid, results, seen);
    }
    
    void clearNode(QuadtreeNode* node) {
        if (!node) return;
        
        node->points.clear();
        node->points.shrink_to_fit();
        node->lodData = nullptr;
        
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                clearNode(node->children[i].get());
                node->children[i].reset(nullptr);
            }
        }
        
        node->setLeaf(true);
    }

    void printStatsRecursive(const QuadtreeNode* node, size_t depth, size_t& totalNodes, size_t& leafNodes, size_t& actualPoints, 
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

    void optimizeRecursive(QuadtreeNode* node) {
        if (!node) return;
        if (!node->isLoaded()) return;

        if (node->isLeaf()) return;

        bool childrenAreLeaves = true;
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                optimizeRecursive(node->children[i].get());
                if (!node->children[i]->isLeaf()) {
                    childrenAreLeaves = false;
                }
            }
        }

        if (childrenAreLeaves) {
            std::vector<std::shared_ptr<NodeData>> allPoints = node->points;
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
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
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                node->points = std::move(allPoints);
                for (int i = 0; i < 4; ++i) {
                    node->children[i].reset(nullptr);
                }
                node->setLeaf(true);
                node->setDirty(true);
                
                node->lodData = nullptr;
            }
        }
    }

    void offloadRecursive(QuadtreeNode* node) {
        float sideLength = node->bounds.second.x() - node->bounds.first.x();
        if (sideLength <= regionTargetSize_ + 1e-4f) {
            if (node->isLoaded()) {
                if (node->isDirty()) {
                    std::unique_lock<std::shared_mutex> lock(node->nodeMutex);
                    node->saveRegion(basePath_);
                }
                node->offload();
            }
            return;
        }
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) offloadRecursive(node->children[i].get());
        }
    }

    void collectNodesByObjectIdRecursive(QuadtreeNode* node, int id,
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

    void collectNodesByObjectId(QuadtreeNode* node, int id, std::vector<std::shared_ptr<NodeData>>& results) const {
        std::unordered_set<std::shared_ptr<NodeData>> seen;
        collectNodesByObjectIdRecursive(node, id, results, seen);
    }
public:
    Quadtree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=8, size_t maxDepth = 16) :
            root_(std::make_unique<QuadtreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
            maxDepth(maxDepth), size(0), mapMutex_(std::make_unique<std::mutex>()),
            streamingQueued_(false) {
        startWorkerThread();
    }

    Quadtree() : root_(nullptr), maxPointsPerNode(8), maxDepth(16), size(0), mapMutex_(std::make_unique<std::mutex>()), streamingQueued_(false) {
        startWorkerThread();
    }

    ~Quadtree() {
        stopWorkerThread();
    }
    
    Quadtree(const Quadtree& other) : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            backgroundColor_(other.backgroundColor_),
            mapMutex_(std::make_unique<std::mutex>()), autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false),
            basePath_(other.basePath_), regionTargetSize_(other.regionTargetSize_) {
        {
            std::lock_guard<std::mutex> lock(*other.mapMutex_);
            colorMap_ = other.colorMap_;
            colorToIndex_ = other.colorToIndex_;
        }
        if (other.root_) root_ = std::make_unique<QuadtreeNode>(*other.root_);
        startWorkerThread();
    }

    Quadtree(Quadtree&& other) noexcept : maxDepth(other.maxDepth), size(other.size), maxPointsPerNode(other.maxPointsPerNode),
            backgroundColor_(std::move(other.backgroundColor_)),
            mapMutex_(std::move(other.mapMutex_)), colorMap_(std::move(other.colorMap_)),
            colorToIndex_(std::move(other.colorToIndex_)), autoOptimize_(other.autoOptimize_.load()),
            streamingQueued_(false),
            basePath_(std::move(other.basePath_)), regionTargetSize_(other.regionTargetSize_) {
        other.stopWorkerThread();
        root_ = std::move(other.root_);
        
        {
            std::lock_guard<std::mutex> lock(other.taskMutex_);
            taskQueue_ = std::move(other.taskQueue_);
        }
        
        other.size = 0;
        startWorkerThread();
    }

    Quadtree& operator=(const Quadtree& other) {
        if (this == &other) return *this;
        
        stopWorkerThread();
        clear();
        
        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        backgroundColor_ = other.backgroundColor_;
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        basePath_ = other.basePath_;
        regionTargetSize_ = other.regionTargetSize_;
        
        {
            std::lock(*mapMutex_, *other.mapMutex_);
            std::lock_guard<std::mutex> l1(*mapMutex_, std::adopt_lock);
            std::lock_guard<std::mutex> l2(*other.mapMutex_, std::adopt_lock);
            
            colorMap_ = other.colorMap_;
            colorToIndex_ = other.colorToIndex_;
        }

        if (other.root_) root_ = std::make_unique<QuadtreeNode>(*other.root_);
        
        startWorkerThread();
        return *this;
    }

    Quadtree& operator=(Quadtree&& other) noexcept {
        if (this == &other) return *this;

        stopWorkerThread();
        other.stopWorkerThread();

        maxDepth = other.maxDepth;
        size = other.size;
        maxPointsPerNode = other.maxPointsPerNode;
        backgroundColor_ = std::move(other.backgroundColor_);
        autoOptimize_.store(other.autoOptimize_.load());
        streamingQueued_.store(false);
        basePath_ = std::move(other.basePath_);
        regionTargetSize_ = std::move(other.regionTargetSize_);
        
        mapMutex_ = std::move(other.mapMutex_);
        colorMap_ = std::move(other.colorMap_);
        colorToIndex_ = std::move(other.colorToIndex_);
        
        root_ = std::move(other.root_);
        
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

    void setBackgroundColor(const Eigen::Vector3f& color) { 
        backgroundColor_ = color;
    }

    Eigen::Vector3f getBackgroundColor() const { 
        return backgroundColor_;
    }

    void setLODMinDistance(float dist) { lodMinDistance_ = dist; }
    void setMaxDistance(float dist) { maxDistance_ = dist; }
    void setRegionTargetSize(float size) { regionTargetSize_ = size; }
    float getRegionTargetSize() const { return regionTargetSize_; }

    void generateLODs() {
        if (!root_) return;
        ensureLOD(root_.get());
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, IndexType layer = 0) {
        
        IndexType cIdx = getColorIndex(color);
        auto pointData = std::make_shared<NodeData>(data, pos, visible, cIdx, size, active, objectId, layer);
        
        ensureBounds(pointData->getSquareBounds());
        
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        }
        return false;
    }

    void queuedset(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f, bool active = true,
             int objectId = -1, IndexType layer = 0) {
        enqueueTask([this, data, pos, visible, color, size, active, objectId, layer]() {
            IndexType cIdx = getColorIndex(color);
            auto pointData = std::make_shared<NodeData>(data, pos, visible, cIdx, size, active, objectId, layer);
            
            ensureBounds(pointData->getSquareBounds());
            
            if (insertRecursive(root_.get(), pointData, 0)) {
                this->size++;
            }
        });
    }

    void updateStreaming(const Camera2D& cam) {
        PointType camPos = cam.origin;
        enqueueTask([this, camPos]() {
            if (root_) {
                updateStreamingRecursive(root_.get(), camPos);
            }
        });
    }

    bool save(const std::string& filename) {
        if (!root_) return false;
        basePath_ = extractBasePath(filename);

        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x79676751;
        QuadtreeNode::writeVal(out, magic);
        QuadtreeNode::writeVal(out, maxDepth);
        QuadtreeNode::writeVal(out, maxPointsPerNode);
        QuadtreeNode::writeVal(out, size);
        QuadtreeNode::writeVal(out, regionTargetSize_);
        
        QuadtreeNode::writeVec3(out, backgroundColor_);

        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            size_t cMapSize = colorMap_.size();
            QuadtreeNode::writeVal(out, cMapSize);
            for (const auto& c : colorMap_) {
                QuadtreeNode::writeVec3(out, c);
            }
        }
        
        QuadtreeNode::writeVec2(out, root_->bounds.first);
        QuadtreeNode::writeVec2(out, root_->bounds.second);

        root_->serialize(out, regionTargetSize_, basePath_);
        
        out.close();
        std::cout << "successfully saved grid to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;
        basePath_ = extractBasePath(filename);

        uint32_t magic;
        QuadtreeNode::readVal(in, magic);
        if (magic != 0x79676751) {
            std::cerr << "Invalid Quadtree file format" << std::endl;
            return false;
        }

        QuadtreeNode::readVal(in, maxDepth);
        QuadtreeNode::readVal(in, maxPointsPerNode);
        QuadtreeNode::readVal(in, size);
        QuadtreeNode::readVal(in, regionTargetSize_);
        
        QuadtreeNode::readVec3(in, backgroundColor_);

        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            colorMap_.clear();
            colorToIndex_.clear();

            size_t cMapSize;
            QuadtreeNode::readVal(in, cMapSize);
            colorMap_.resize(cMapSize);
            for (size_t i = 0; i < cMapSize; ++i) {
                QuadtreeNode::readVec3(in, colorMap_[i]);
                colorToIndex_[colorMap_[i]] = static_cast<IndexType>(i);
            }
        }

        PointType minBound, maxBound;
        QuadtreeNode::readVec2(in, minBound);
        QuadtreeNode::readVec2(in, maxBound);

        root_ = std::make_unique<QuadtreeNode>(minBound, maxBound);
        root_->deserialize(in, regionTargetSize_, basePath_);

        in.close();
        std::cout << "successfully loaded grid from " << filename << std::endl;
        return true;
    }

    std::shared_ptr<NodeData> getTopMost(const PointType& pos) {
        std::shared_ptr<NodeData> best = nullptr;
        IndexType maxLayer = 0;
        if (root_) getTopMostRecursive(root_.get(), pos, best, maxLayer);
        return best;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = 1e-5f) {
        return findRecursive(root_.get(), pos, tolerance);
    }

    std::shared_ptr<NodeData> findwNode(const PointType& pos, QuadtreeNode* node, float tolerance = 1e-5f) {
        return findRecursive(node, pos, tolerance);
    }

    bool inGrid(PointType pos) {
        return root_->contains(pos);
    }

    bool remove(const PointType& pos, float tolerance = 1e-5f) {
        auto pt = find(pos, tolerance);
        if (!pt) return false;
        if (removeRecursive(root_.get(), pt->getSquareBounds(), pt)) {
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

    bool update(const PointType& pos, const T& newData) {
        auto pointData = find(pos);
        if (!pointData) return false;
        else pointData->data = newData;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedupdate(const PointType& pos, const T& newData) {
        enqueueTask([this, pos, newData]() {
            QuadtreeNode* node = root_.get();
            auto pointData = findwNode(pos, node);
            if (!pointData) return;
            else {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->data = newData;
            }
            invalidateLODForPoint(pointData);
        });
    }

    bool update(const PointType& oldPos, const PointType& newPos, const T& newData, bool newVisible = true, 
                Eigen::Vector3f newColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f), float newSize = 0.01f, bool newActive = true,
                int newObjectId = -2, int newLayer = -1, float tolerance = 1e-5f) {

        auto pointData = find(oldPos, tolerance);
        if (!pointData) return false;

        int targetObjId = (newObjectId != -2) ? newObjectId : pointData->objectId;
        IndexType targetLayer = (newLayer != -1) ? static_cast<IndexType>(newLayer) : pointData->layer;
        
        removeRecursive(root_.get(), pointData->getSquareBounds(), pointData);
        
        pointData->data = newData;
        pointData->position = newPos;
        pointData->setVisible(newVisible);
        
        if (newColor != Eigen::Vector3f(1.0f, 1.0f, 1.0f)) pointData->colorIdx = getColorIndex(newColor);
        if (newSize > 0) pointData->size = newSize;
        pointData->setActive(newActive);
        pointData->objectId = targetObjId;
        pointData->layer = targetLayer;
        
        ensureBounds(pointData->getSquareBounds());
        bool res = insertRecursive(root_.get(), pointData, 0);
        
        if(!res) {
            size--;
        }

        return res;
    }

    bool move(const PointType& pos, const PointType& newPos) {
        auto pointData = find(pos);
        if (!pointData) return false;

        removeRecursive(root_.get(), pointData->getSquareBounds(), pointData);
        pointData->position = newPos;
        ensureBounds(pointData->getSquareBounds());

        if (insertRecursive(root_.get(), pointData, 0)) {
            return true;
        }
        size--;
        return false;
    }

    void queuedmove(const PointType& pos, const PointType& newPos) {
        enqueueTask([this, pos, newPos]() {
            auto pointData = find(pos);
            if (!pointData) return;

            removeRecursive(root_.get(), pointData->getSquareBounds(), pointData);
            pointData->position = newPos;
            ensureBounds(pointData->getSquareBounds());

            if (!insertRecursive(root_.get(), pointData, 0)) {
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
        pointData->colorIdx = getColorIndex(color);
        invalidateLODForPoint(pointData);
        return true;
    }

    bool setLayer(const PointType& pos, IndexType layer, float tolerance = 1e-5f) {
        auto pointData = find(pos, tolerance);
        if (!pointData) return false;
        pointData->layer = layer;
        invalidateLODForPoint(pointData);
        return true;
    }

    void queuedsetColor(const PointType& pos, Eigen::Vector3f color, float tolerance = 1e-5f) {
        enqueueTask([this, pos, color, tolerance]() {
            QuadtreeNode* node = root_.get();
            auto pointData = findwNode(pos, node, tolerance);
            if (!pointData) return;
            {
                std::lock_guard<std::shared_mutex> lock(node->nodeMutex);
                pointData->colorIdx = getColorIndex(color);
            }
            invalidateLODForPoint(pointData);
        });
    }

    frame renderFrame(const Camera2D& cam, int width, int height, frame::colormap colorformat = frame::colormap::RGB) {
        updateStreaming(cam);
        
        frame outFrame(width, height, colorformat);
        std::vector<float> colorBuffer;
        int channels = 3;
        colorBuffer.resize(width * height * channels);

        float viewWidth = cam.viewWidth;
        float viewHeight = viewWidth * ((float)height / width);
        float pxSizeX = viewWidth / width;
        float pxSizeY = viewHeight / height;
        
        PointType topLeft(cam.origin.x() - viewWidth / 2.0f, cam.origin.y() - viewHeight / 2.0f);

        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pidx = (y * width + x);
                int idx = pidx * channels;

                PointType worldPos = topLeft + PointType(x * pxSizeX, y * pxSizeY);
                
                auto hit = getTopMost(worldPos);
                
                Eigen::Vector3f color = hit ? getColor(hit->colorIdx) : backgroundColor_;
                
                color = color.cwiseMax(0.0f).cwiseMin(1.0f);

                colorBuffer[idx    ] = color[0];
                colorBuffer[idx + 1] = color[1];
                colorBuffer[idx + 2] = color[2];
            }
        }
        
        outFrame.setData(colorBuffer, frame::colormap::RGB);
        return outFrame;
    }

    bool moveObject(int objectId, const PointType& offset) {
        if (!root_) return false;
        
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(root_.get(), objectId, nodes);
        if(nodes.empty()) return false;

        for(auto& n : nodes) {
            if (removeRecursive(root_.get(), n->getSquareBounds(), n)) {
                size--;
            }
        }
        for(auto& n : nodes) {
            n->position += offset;
            ensureBounds(n->getSquareBounds());
            if (insertRecursive(root_.get(), n, 0)) {
                size++;
            }
        }
        return true;
    }

    void optimize() {
        if (root_) {
            optimizeRecursive(root_.get());
            reorderMapsZCurve();
            generateLODs();
        }
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) {
            os << "[Quadtree Stats] Tree is null/empty." << std::endl;
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
        double avgPointsPerLeaf = leafNodes > 0 ? (double)actualPoints / leafNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(QuadtreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16);
        size_t mapMem = colorMap_.size() * sizeof(Eigen::Vector3f);
        size_t maxSize = ((1 << (sizeof(IndexType)*8 - 2) - 1) * 2) + 1;

        os << "========================================\n";
        os << "             QUADTREE STATS             \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
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
        os << "Maps:\n";
        os << "  Unique Colors     : " << colorMap_.size() << "/" << maxSize << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "  Dictionary Maps   : " << (mapMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }

    void clear() {
        if (root_) {
            clearNode(root_.get());
        }
        PointType minBound = root_ ? root_->bounds.first : PointType::Zero();
        PointType maxBound = root_ ? root_->bounds.second : PointType::Zero();
        root_ = std::make_unique<QuadtreeNode>(minBound, maxBound);
        
        {
            std::lock_guard<std::mutex> lock(*mapMutex_);
            colorMap_.clear();
            colorToIndex_.clear();
        }
        
        size = 0;
    }
};

#endif