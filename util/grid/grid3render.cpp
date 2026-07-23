#include "grid3eigen.hpp"
namespace Grid {

template<typename T>
void Octree<T>::buildRender(RenderBuffer_<T>& buffer) {
    // TIME_FUNCTION;
    buffer.clear();
    if (root_ == INVALID_IDX) return;
    buffer.nodes.emplace_back();
    buffer.points.reserve(size);

    std::unordered_map<int, std::shared_ptr<GridObject_<T>>> localObjects;
    {
        s_lock lock(objectsMutex_);
        localObjects = objects_;
    }

    {
        s_lock matLock(renderMaterials_.mutex);
        buffer.materials = renderMaterials_.materials;
    }
    buffer.defaultMatIdx = buffer.materials.size();
    buffer.materials.push_back(RenderMaterial());

    buildRenderNodeAt(root_, buffer, 0, localObjects);
}

template<typename T>
void Octree<T>::buildRenderNodeAt(uint32_t nodeIndex, RenderBuffer_<T>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects) {
    const OctreeNode_<T>* node = nodeAt(nodeIndex);
    if (!node) return;
    bool isLoaded = node->isLoaded();
    
    RenderNode_<T> rnode;
    rnode.center = node->center;
    rnode.nodeSize = node->nodeSize;
    rnode.isLeaf = node->isLeaf();
    rnode.isLoaded = isLoaded;
    rnode.originalNode = nodeIndex;
    
    rnode.firstPoint = static_cast<uint32_t>(buffer.points.size());
    if (isLoaded) {
        const auto pts = pointsView(nodeIndex);
        for (const auto& pt : pts) {
            if (!pt || !pt->isActive() || !pt->isVisible()) continue; 
            RenderData rd;
            rd.position = pt->position;
            rd.size = pt->size;
            rd.color = pt->color;
            
            rd.materialIdx = (pt->renderMatIdx < buffer.defaultMatIdx) ? pt->renderMatIdx : buffer.defaultMatIdx;
            
            rd.objectId = pt->objectId;
            rd.extent = EXTENT_UNIT;
            buffer.points.push_back(rd);
        }
    }

    const uint32_t rawCount = static_cast<uint32_t>(buffer.points.size()) - rnode.firstPoint;
    auto cacheIt = buffer.mergeCache.find(nodeIndex);
    const bool cacheUsable = !node->isDirty() && cacheIt != buffer.mergeCache.end()
                             && cacheIt->second.sourceCount == rawCount;
    if (cacheUsable) {
        buffer.points.resize(rnode.firstPoint);
        for (const RenderData& b : cacheIt->second.boxes) buffer.points.push_back(b);
        rnode.pointCount = static_cast<uint32_t>(cacheIt->second.boxes.size());
    } else {
        rnode.pointCount = mergeLeafPoints(buffer, rnode.firstPoint);
        MergeCacheEntry entry;
        entry.sourceCount = rawCount;
        entry.boxes.assign(buffer.points.begin() + rnode.firstPoint, buffer.points.end());
        buffer.mergeCache[nodeIndex] = std::move(entry);
        if (isLoaded) const_cast<OctreeNode_<T>*>(node)->setDirty(false);
    }
    
    rnode.lodPoint = -1;
    auto lodData = lodOf(nodeIndex);
    if (lodData) {
        RenderData ld;
        ld.position = lodData->position;
        ld.size = lodData->size;
        ld.color = lodData->color;

        ld.materialIdx = (lodData->renderMatIdx < buffer.defaultMatIdx)
                             ? lodData->renderMatIdx : buffer.defaultMatIdx;

        ld.objectId = lodData->objectId;
        ld.extent = EXTENT_UNIT; 
        rnode.lodPoint = static_cast<int32_t>(buffer.points.size());
        buffer.points.push_back(ld);
    }
    
    rnode.childMask = 0;
    rnode.firstChild = 0;
    
    if (!node->isLeaf() && isLoaded) {
        const uint32_t srcFirstChild = node->firstChild;
        const uint8_t mask = node->childMask;
        node = nullptr;
        int childCount = 0;
        for (int i = 0; i < 8; ++i) {
            if (mask & (1 << i)) childCount++;
        }
        rnode.childMask = mask;
        if (childCount > 0) {
            rnode.firstChild = static_cast<uint32_t>(buffer.nodes.size());
            buffer.nodes.resize(buffer.nodes.size() + childCount);
            int cidx = 0;
            for (int i = 0; i < 8; ++i) {
                if (mask & (1 << i)) {
                    buildRenderNodeAt(srcFirstChild + i, buffer, rnode.firstChild + cidx, localObjects);
                    cidx++;
                }
            }
        }
    }
    
    buffer.nodes[nodeIdx] = rnode;
}

struct MergeCell {
    uint32_t src;
    uint32_t claimedBy;
};

using MergeLattice = std::unordered_map<std::array<int64_t, 3>, MergeCell, Vec3i64Hash>;

///@brief Tests whether a spanX-by-spanY slab at base is free and matches the seed
static bool slabMatches(const MergeLattice& lattice, const std::vector<RenderData>& pts, uint32_t first,
                        const RenderData& seed, const std::array<int64_t, 3>& base,
                        int64_t spanX, int64_t spanY) {
    for (int64_t dy = 0; dy < spanY; ++dy) {
        for (int64_t dx = 0; dx < spanX; ++dx) {
            auto it = lattice.find({base[0] + dx, base[1] + dy, base[2]});
            if (it == lattice.end() || it->second.claimedBy != INVALID_IDX) return false;
            const RenderData& p = pts[first + it->second.src];
            if (p.materialIdx != seed.materialIdx || p.objectId != seed.objectId
                || !p.color.isApprox(seed.color)) return false;
        }
    }
    return true;
}

///@brief Marks a spanX-by-spanY slab at base as owned, after slabMatches approved it
static void claimSlab(MergeLattice& lattice, const std::array<int64_t, 3>& base,
                      int64_t spanX, int64_t spanY, uint32_t owner) {
    for (int64_t dy = 0; dy < spanY; ++dy) {
        for (int64_t dx = 0; dx < spanX; ++dx) {
            lattice[{base[0] + dx, base[1] + dy, base[2]}].claimedBy = owner;
        }
    }
}

template<typename T>
uint32_t Octree<T>::mergeLeafPoints(RenderBuffer_<T>& buffer, uint32_t first) {
    const uint32_t count = static_cast<uint32_t>(buffer.points.size()) - first;
    if (count < 2) return count;

    const float cell = buffer.points[first].size;
    if (cell <= 0.0f) return count;
    Vec3 lo = buffer.points[first].position;
    for (uint32_t i = 1; i < count; ++i) {
        const RenderData& p = buffer.points[first + i];
        if (p.size != cell) return count;
        lo = lo.cwiseMin(p.position);
    }

    const float invCell = 1.0f / cell;
    std::unordered_map<std::array<int64_t, 3>, MergeCell, Vec3i64Hash> lattice;
    lattice.reserve(count * 2);
    std::vector<std::array<int64_t, 3>> coords(count);

    std::vector<uint32_t> passthrough;
    for (uint32_t i = 0; i < count; ++i) {
        const Vec3 rel = (buffer.points[first + i].position - lo) * invCell;
        coords[i] = {static_cast<int64_t>(std::llround(rel.x())),
                     static_cast<int64_t>(std::llround(rel.y())),
                     static_cast<int64_t>(std::llround(rel.z()))};
        const Vec3 snapped(static_cast<float>(coords[i][0]), static_cast<float>(coords[i][1]),
                           static_cast<float>(coords[i][2]));
        if ((rel - snapped).cwiseAbs().maxCoeff() > LATTICE_EPS) {
            passthrough.push_back(i);
            continue;
        }
        if (!lattice.emplace(coords[i], MergeCell{i, INVALID_IDX}).second) {
            passthrough.push_back(i);
        }
    }

    std::vector<RenderData> merged;
    merged.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        auto seedIt = lattice.find(coords[i]);
        if (seedIt == lattice.end() || seedIt->second.src != i) continue;
        MergeCell& self = seedIt->second;
        if (self.claimedBy != INVALID_IDX) continue;

        const RenderData& seed = buffer.points[first + i];
        const std::array<int64_t, 3> c = coords[i];
        const uint32_t owner = static_cast<uint32_t>(merged.size());
        self.claimedBy = owner;

        const int64_t lim = static_cast<int64_t>(EXTENT_MAX);
        int64_t ex = 1, ey = 1, ez = 1;

        while (ex < lim && slabMatches(lattice, buffer.points, first, seed,
                                       {c[0] + ex, c[1], c[2]}, 1, 1)) {
            claimSlab(lattice, {c[0] + ex, c[1], c[2]}, 1, 1, owner);
            ex++;
        }
        while (ey < lim && slabMatches(lattice, buffer.points, first, seed,
                                       {c[0], c[1] + ey, c[2]}, ex, 1)) {
            claimSlab(lattice, {c[0], c[1] + ey, c[2]}, ex, 1, owner);
            ey++;
        }
        while (ez < lim && slabMatches(lattice, buffer.points, first, seed,
                                       {c[0], c[1], c[2] + ez}, ex, ey)) {
            claimSlab(lattice, {c[0], c[1], c[2] + ez}, ex, ey, owner);
            ez++;
        }

        RenderData box = seed;
        box.extent = packExtent(static_cast<uint32_t>(ex), static_cast<uint32_t>(ey),
                                static_cast<uint32_t>(ez));
        merged.push_back(box);
    }

    for (uint32_t i : passthrough) merged.push_back(buffer.points[first + i]);

    buffer.points.resize(first);
    for (const RenderData& b : merged) buffer.points.push_back(b);
    return static_cast<uint32_t>(merged.size());
}

template<typename T>
std::vector<RenderData*> Octree<T>::fastVoxelTraverse(const RenderBuffer_<T>& buffer, const Ray& ray, float maxDist) {
    std::vector<RenderData*> hits;
    if (buffer.nodes.empty()) return hits;
    std::vector<float> tv;
    float tMin, tMax;
    BoundingBox rootBounds(buffer.nodes[0].boundsMin(), buffer.nodes[0].boundsMax());
    if (rayBoxIntersect(ray, rootBounds, tMin, tMax)) {
        tMax = std::min(tMax, maxDist);
        float currentMaxDist = maxDist;
        
        struct StackItem {
            uint32_t nodeIdx;
            float tMin;
            float tMax;
        };
        
        StackItem stack[256];
        int stackPtr = 0;
        stack[stackPtr++] = {0, std::max(0.0f, tMin), tMax};

        while(stackPtr > 0) {
            StackItem current = stack[--stackPtr];
            if (current.tMin > currentMaxDist) continue;
            
            const RenderNode_<T>& node = buffer.nodes[current.nodeIdx];

            if (!node.isLoaded && node.originalNode != INVALID_IDX) {
                ensureLoaded(node.originalNode, true);
            }

            if (node.isLoaded && node.pointCount == 0 && node.lodPoint == -1
                && (node.isLeaf || node.childMask == 0)) continue;

            if (!node.isLeaf && node.lodPoint != -1) {
                float dist = (node.center - ray.origin).norm();
                if (dist > lodMinDistance_ && (dist / node.nodeSize) > invLodf) {
                    float t;
                    Vec3 n, h;
                    if (rayCubeIntersect(ray, &buffer.points[node.lodPoint], t, n, h)) {
                        if (t >= 0 && t <= currentMaxDist) {
                            hits.emplace_back(const_cast<RenderData*>(&buffer.points[node.lodPoint]));
                            tv.emplace_back(t);
                            if (buffer.points[node.lodPoint].color.w() >= 0.9f) {
                                currentMaxDist = std::min(currentMaxDist, t);
                            }
                        }
                    }
                    continue;
                }
            }

            for (uint32_t i = 0; i < node.pointCount; ++i) {
                const RenderData& pt = buffer.points[node.firstPoint + i];
                float t;
                Vec3 n, h;
                if (rayCubeIntersect(ray, &pt, t, n, h)) {
                    if (t >= 0 && t <= currentMaxDist) {
                        hits.emplace_back(const_cast<RenderData*>(&pt));
                        tv.emplace_back(t);
                        if (pt.color.w() >= 0.9f) {
                            currentMaxDist = std::min(currentMaxDist, t);
                        }
                    }
                }
            }

            if (node.isLeaf || !node.isLoaded) continue;

            float t0 = current.tMin;
            float t1 = current.tMax;

            Vec3 ttt = (node.center - ray.origin).cwiseProduct(ray.invDir);
            int currIdx = ((t0 >= ttt.x()) ? 1 : 0) | ((t0 >= ttt.y()) ? 2 : 0) | ((t0 >= ttt.z()) ? 4 : 0);
            
            struct ChildInterval {
                uint32_t nodeIdx;
                float tMin;
                float tMax;
            };
            ChildInterval children[4];
            int childCount = 0;

            while(t0 < t1 && t0 <= currentMaxDist) {
                Vec3 next_t;
                next_t[0] = (currIdx & 1) ? t1 : ttt[0];
                next_t[1] = (currIdx & 2) ? t1 : ttt[1];
                next_t[2] = (currIdx & 4) ? t1 : ttt[2];

                float tNext = next_t.minCoeff();
                
                int physIdx = currIdx ^ ray.signMask;
                if (node.childMask & (1 << physIdx)) {
                    int childOffset = countBits(node.childMask & ((1 << physIdx) - 1));
                    children[childCount++] = {node.firstChild + childOffset, t0, tNext};
                }
                
                t0 = tNext;
                currIdx |= ((next_t[0] <= tNext) ? 1 : 0) | ((next_t[1] <= tNext) ? 2 : 0) | ((next_t[2] <= tNext) ? 4 : 0);
            }

            if (stackPtr + childCount > 256) continue;

            for (int i = childCount - 1; i >= 0; --i) {
                stack[stackPtr++] = {children[i].nodeIdx, children[i].tMin, children[i].tMax};
            }
        }
    }
    std::vector<std::pair<RenderData*, float>> zipped;
    zipped.reserve(hits.size());
    for (std::size_t i = 0; i < hits.size(); ++i)
        zipped.emplace_back(hits[i], tv[i]);
    std::sort(zipped.begin(), zipped.end(), [](const auto& a, const auto& b) {return a.second < b.second;});
    for (std::size_t i = 0; i < zipped.size(); ++i) {
        hits[i] = zipped[i].first;
        tv[i] = zipped[i].second;
    }
    return hits;
}

template<typename T>
frame Octree<T>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
    updateStreaming(cam);
    
    thread_local RenderBuffer_<T> tl_buffer;
    buildRender(tl_buffer);
    const RenderBuffer_<T>& shared_buffer = tl_buffer;

    Vec3 origin = cam.origin;
    Vec3 dir = cam.direction.normalized();
    Vec3 up = cam.up.normalized();
    Vec3 right = cam.right();
    
    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer;
    colorBuffer.resize(width * height * 3);

    const float aspect = static_cast<float>(width) / height;
    const float fovRad = cam.fovRad();
    const float tanHalfFov = tan(fovRad * 0.5f);
    const float tanfovy = tanHalfFov;
    const float tanfovx = tanHalfFov * aspect;
    
    const Vec3 globalLightDir = (-cam.direction * 0.2f).normalized();
    const float minVisibility = 0.1f;
    float maxmin = (maxDistance_ - lodMinDistance_);
    float invMaxMin = 1 / maxmin;
    
    #pragma omp parallel for schedule(dynamic, 128) collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pidx = (y * width + x);
            int idx = pidx * 3;

            float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
            float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
            
            Vec3 rayDir = dir + (right * px) + (up * py);
            rayDir.normalize();

            Vec3 scolor = skybox_.sampleVector(rayDir);
            Vec3 color = Vec3(0.0f, 0.0f, 0.0f);
            float accumAlpha = 0.0f;
            Ray ray(origin, rayDir);
            
            // const RenderData* hit = nullptr;
            std::vector<RenderData*> hits = fastVoxelTraverse(shared_buffer, ray, maxDistance_);

            int prevOid = -1;
            bool hasPrev = false;

            for (auto hit : hits) {
                if (accumAlpha >= 0.99f) break;
                if (hit == nullptr) continue;
                float t = 0.0f;
                float tExit = 0.0f;
                Vec3 normal, hitPoint;

                rayCubeIntersect(ray, hit, t, normal, hitPoint, &tExit);
                Vec3 hitColor = hit->color.template head<3>();
                float alpha = hit->color.w();
                RenderMaterial objMat = shared_buffer.materials[hit->materialIdx];

                float coverage;
                bool isInternalVoxel = (hasPrev && hit->objectId == prevOid);
                if (isInternalVoxel) {
                    float thickness = tExit - t;
                    float trPerUnit = std::clamp(1.0f - alpha, 0.0f, 1.0f);
                    coverage = 1.0f - std::pow(trPerUnit, thickness);
                } else {
                    coverage = alpha;
                }
                
                if (objMat.chromaticity != 0u) {
                    hitColor = hitColor.cwiseProduct(objMat.emittanceRGB());
                } else {
                    float diffuse = 0.0f;
                    float ambient = 0.35f;
                    if (!isInternalVoxel) {
                        diffuse = std::max(0.0f, normal.dot(globalLightDir));
                    }
                    float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                    hitColor = hitColor * intensity;
                }
                
                float fogFactor = std::clamp((maxDistance_ - t) * invMaxMin, minVisibility, 1.0f);
                
                hitColor = hitColor * fogFactor + scolor * (1.0f - fogFactor);
                
                color += hitColor * coverage * (1.0f - accumAlpha);
                accumAlpha += coverage * (1.0f - accumAlpha);
                prevOid = hit->objectId;
                hasPrev = true;
            }
            
            if (accumAlpha < 0.99f) {
                color += scolor * (1.0f - accumAlpha);
            }

            color = color.cwiseMax(0.0f).cwiseMin(1.0f);

            colorBuffer[idx    ] = color[0];
            colorBuffer[idx + 1] = color[1];
            colorBuffer[idx + 2] = color[2];
        }
    }
    
    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

struct PointSort {
    uint64_t morton;
    size_t idx;
    bool operator<(const PointSort& o) const { return morton < o.morton; }
};

static inline uint64_t mortonSplit3(uint64_t v) {
    v &= 0x1fffffULL;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8)  & 0x100f00f00f00f00fULL;
    v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2)  & 0x1249249249249249ULL;
    return v;
}

static inline uint64_t mortonEncode21(uint32_t x, uint32_t y, uint32_t z) {
    return mortonSplit3(x) | (mortonSplit3(y) << 1) | (mortonSplit3(z) << 2);
}

static void mortonRadixSort(std::vector<PointSort>& v, std::vector<PointSort>& scratch) {
    const size_t n = v.size();
    if (n < 2) return;
    scratch.resize(n);
    for (int shift = 0; shift < 64; shift += 8) {
        size_t count[257] = {0};
        for (size_t i = 0; i < n; ++i) ++count[((v[i].morton >> shift) & 0xFF) + 1];
        if (count[((v[0].morton >> shift) & 0xFF) + 1] == n) continue;
        for (int k = 0; k < 256; ++k) count[k + 1] += count[k];
        for (size_t i = 0; i < n; ++i) scratch[count[(v[i].morton >> shift) & 0xFF]++] = v[i];
        v.swap(scratch);
    }
}
template<typename GetPos>
static void mortonSortIndices(const std::vector<size_t>& indices, const Vec3& boundsMin, const Vec3& boundsMax,
                              GetPos&& getPos, std::vector<PointSort>& out) {
    Vec3 extent = boundsMax - boundsMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Vec3 invExtent = extent.cwiseInverse();

    out.clear();
    out.resize(indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
        const size_t idx = indices[k];
        Vec3 normPos = (getPos(idx) - boundsMin).cwiseProduct(invExtent);
        uint32_t x = static_cast<uint32_t>(std::min(std::max(normPos.x() * 2097151.0f, 0.0f), 2097151.0f));
        uint32_t y = static_cast<uint32_t>(std::min(std::max(normPos.y() * 2097151.0f, 0.0f), 2097151.0f));
        uint32_t z = static_cast<uint32_t>(std::min(std::max(normPos.z() * 2097151.0f, 0.0f), 2097151.0f));
        out[k] = {mortonEncode21(x, y, z), idx};
    }

    static thread_local std::vector<PointSort> scratch;
    mortonRadixSort(out, scratch);
}

///@brief Fingerprint of a built point set, used to decide if the cache is stale
static uint64_t hashRenderPoints(const void* data, size_t bytes) {
    // TIME_FUNCTION;
    uint64_t h = 1469598103934665603ull;
    const uint64_t* words = static_cast<const uint64_t*>(data);
    const size_t nWords = bytes / sizeof(uint64_t);
    for (size_t i = 0; i < nWords; ++i) {
        h ^= words[i];
        h *= 1099511628211ull;
    }
    const uint8_t* tail = static_cast<const uint8_t*>(data) + nWords * sizeof(uint64_t);
    for (size_t i = 0; i < bytes % sizeof(uint64_t); ++i) {
        h ^= tail[i];
        h *= 1099511628211ull;
    }
    return h;
}

///@brief Geometry-only products of the render prologue, reused across frames
struct SceneCache {
    std::vector<GPURenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    Vec3 boundsMin = Vec3::Zero();
    Vec3 boundsMax = Vec3::Ones();
    int emissiveCount = 0;
    uint64_t pointHash = 0;
    uint64_t materialHash = 0;
    bool sorted = false;
    bool valid = false;
};

///@brief Rebuilds the cached point set only when the tree contents changed
///@param buffer Freshly built render buffer for this frame
///@param wantSort Morton-order the points (PBR paths) or keep tree order (fast path)
///@param expandByRadius Grow bounds by each point's half size (fast/VCT paths)
///@param cache Cache to fill or reuse
///@return true when the cache was rebuilt, false when the previous one was reused
template<typename BufferT>
static bool refreshSceneCache(const BufferT& buffer, bool wantSort, bool expandByRadius,
                              SceneCache& cache) {
    //TIME_FUNCTION;
    const size_t pointBytes = buffer.points.size() * sizeof(buffer.points[0]);
    const uint64_t pHash = buffer.points.empty() ? 0
                         : hashRenderPoints(buffer.points.data(), pointBytes);
    const size_t matBytes = buffer.materials.size() * sizeof(buffer.materials[0]);
    const uint64_t mHash = buffer.materials.empty() ? 0
                         : hashRenderPoints(buffer.materials.data(), matBytes);

    if (cache.valid && cache.pointHash == pHash && cache.materialHash == mHash
        && cache.sorted == wantSort) {
        return false;
    }

    std::vector<bool> isLodPoint(buffer.points.size(), false);
    for (const auto& n : buffer.nodes) {
        if (n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Vec3 globalMin = Vec3::Constant(std::numeric_limits<float>::max());
    Vec3 globalMax = Vec3::Constant(std::numeric_limits<float>::lowest());

    std::vector<size_t> validIndices;
    validIndices.reserve(buffer.points.size());
    for (size_t i = 0; i < buffer.points.size(); ++i) {
        if (isLodPoint[i]) continue;
        const auto& p = buffer.points[i];
        validIndices.push_back(i);
        if (expandByRadius) {
            const Vec3 h = Vec3::Constant(p.size * 0.5f);
            globalMin = globalMin.cwiseMin(p.position - h);
            globalMax = globalMax.cwiseMax(p.position + h);
        } else {
            globalMin = globalMin.cwiseMin(p.position);
            globalMax = globalMax.cwiseMax(p.position);
        }
    }
    if (globalMin.x() > globalMax.x()) {
        globalMin.setZero();
        globalMax.setOnes();
    }

    cache.gpuPoints.clear();
    cache.gpuLights.clear();
    cache.gpuPoints.reserve(validIndices.size());

    if (wantSort) {
        std::vector<PointSort> sortedPoints;
        mortonSortIndices(validIndices, globalMin, globalMax,
                          [&](size_t idx) { return buffer.points[idx].position; },
                          sortedPoints);
        for (const auto& sp : sortedPoints) {
            const auto& p = buffer.points[sp.idx];
            cache.gpuPoints.push_back({p.position, p.size, packRGBA8(p.color),
                                       p.materialIdx, p.objectId, p.extent});
            if (buffer.materials[p.materialIdx].chromaticity != 0u) {
                cache.gpuLights.push_back(cache.gpuPoints.size() - 1);
            }
        }
    } else {
        for (const size_t idx : validIndices) {
            const auto& p = buffer.points[idx];
            cache.gpuPoints.push_back({p.position, p.size, packRGBA8(p.color),
                                       p.materialIdx, p.objectId, p.extent});
            if (buffer.materials[p.materialIdx].chromaticity != 0u) {
                cache.gpuLights.push_back(cache.gpuPoints.size() - 1);
            }
        }
    }

    cache.emissiveCount = static_cast<int>(cache.gpuLights.size());
    if (cache.gpuPoints.empty()) cache.gpuPoints.push_back(GPURenderData{});
    if (cache.gpuLights.empty()) cache.gpuLights.push_back(0);

    cache.boundsMin = globalMin;
    cache.boundsMax = globalMax;
    cache.pointHash = pHash;
    cache.materialHash = mHash;
    cache.sorted = wantSort;
    cache.valid = true;
    return true;
}

static void runWavefrontTilesMultiGPU(int width, int height, const GPUCameraData& camTemplate,
                                      int samplesPerPixel, int maxBounces, int sampleOffset, size_t pixFloats, size_t bufBytes,
                                      bool buffersPreSeeded = false) {
    // TIME_FUNCTION;
    using Tile = Eigen::Matrix<int, 4, 1>;

    gpuFleet.probeTileTargets(camTemplate);

    int start = 0;
    const size_t nGPU = gpuFleet.count();

    if (nGPU <= 1) {
        const std::vector<Tile> tiles = buildTiles(width, height, vkCtx.tileTarget());
        for (const auto& t : tiles) {
            GPUCameraData cd = camTemplate;
            cd.tileOffsetX = t.x();
            cd.tileOffsetY = t.y();
            cd.currentSampleOffset = sampleOffset;
            cd.dispatchSamples = samplesPerPixel;
            vkCtx.updateCameraData(cd);
            vkCtx.dispatchWavefront(t.z(), t.w(), maxBounces, samplesPerPixel, start);
            cd.tileOffsetX = t.x();
            cd.tileOffsetY = t.y();
            cd.tileW = t.z();
            cd.tileH = t.w();
        }
        return;
    }


    std::vector<float> zeros(pixFloats, 0.0f);
    std::vector<float> seedPix, seedAd;
    if (buffersPreSeeded) {
        seedPix.resize(pixFloats);
        seedAd.resize(pixFloats);
        vkCtx.downloadFromBuffer(vkCtx.outBuffer, seedPix.data(), bufBytes);
        vkCtx.downloadFromBuffer(vkCtx.adaptiveBuffer, seedAd.data(), bufBytes);
    }

    static std::vector<double> speedEMA;
    if (speedEMA.size() != nGPU) speedEMA.assign(nGPU, 0.0);
    bool needCalib = false;
    for (size_t g = 0; g < nGPU; ++g) needCalib |= (speedEMA[g] <= 0.0);
    if (needCalib) {
        int calibTarget = TILE_MAX;
        for (size_t g = 0; g < nGPU; ++g) {
            calibTarget = std::min(calibTarget, gpuFleet.ctx(g).tileTarget());
        }
        const std::vector<Tile> calibTiles = buildTiles(width, height, calibTarget);
        const Tile& ct = calibTiles[calibTiles.size() / 2];
        std::vector<std::thread> calib;
        for (size_t g = 0; g < nGPU; ++g) {
            calib.emplace_back([&, g] {
                auto& ctx = gpuFleet.ctx(g);
                GPUCameraData cd = camTemplate;
                cd.tileOffsetX = ct.x();
                cd.tileOffsetY = ct.y();
                cd.currentSampleOffset = 1;
                cd.dispatchSamples = 1;
                ctx.updateCameraData(cd);
                auto t0 = std::chrono::steady_clock::now();
                ctx.dispatchWavefront(ct.z(), ct.w(), maxBounces, 1, 0);
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                speedEMA[g] = 1.0 / std::max(ms, 1e-3);
            });
        }
        for (auto& t : calib) t.join();
    }

    double speedSum = 0.0;
    for (size_t g = 0; g < nGPU; ++g) speedSum += speedEMA[g];
    std::vector<int> counts(nGPU, 0);
    int assigned = 0;
    size_t fastest = 0;
    for (size_t g = 0; g < nGPU; ++g) {
        counts[g] = (int)std::floor(samplesPerPixel * speedEMA[g] / speedSum);
        assigned += counts[g];
        if (speedEMA[g] > speedEMA[fastest]) fastest = g;
    }
    counts[fastest] += samplesPerPixel - assigned;

    for (size_t g = 0; g < nGPU; ++g) {
        auto& ctx = gpuFleet.ctx(g);
        const float* pix = (g == 0 && buffersPreSeeded) ? seedPix.data() : zeros.data();
        const float* ad  = (g == 0 && buffersPreSeeded) ? seedAd.data()  : zeros.data();
        ctx.uploadToBuffer(ctx.outBuffer, pix, bufBytes);
        ctx.uploadToBuffer(ctx.adaptiveBuffer, ad, bufBytes);
    }

    std::vector<double> msSpent(nGPU, 0.0);
    std::vector<std::thread> workers;
    for (size_t g = 0; g < nGPU; ++g) {
        int myStart = start, myCount = counts[g];
        start += myCount;
        if (myCount <= 0) continue;
        workers.emplace_back([&, g, myStart, myCount] {
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<Tile> myTiles = buildTiles(width, height, gpuFleet.ctx(g).tileTarget());
            for (const auto& t : myTiles) {
                GPUCameraData cd = camTemplate;
                cd.tileOffsetX = t.x();
                cd.tileOffsetY = t.y();
                cd.currentSampleOffset = sampleOffset;
                cd.dispatchSamples = myCount;
                gpuFleet.ctx(g).updateCameraData(cd);
                gpuFleet.ctx(g).dispatchWavefront(t.z(), t.w(), maxBounces, myCount, myStart);
            }
            msSpent[g] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        });
    }
    for (auto& w : workers) w.join();

    for (size_t g = 0; g < nGPU; ++g) {
        if (counts[g] > 0 && msSpent[g] > 0.0) {
            double measured = double(counts[g]) / msSpent[g];
            speedEMA[g] = 0.5 * speedEMA[g] + 0.5 * measured;
        }
    }

    std::vector<float> merged(pixFloats, 0.0f), mergedAd(pixFloats, 0.0f), tmp(pixFloats);
    for (size_t g = 0; g < nGPU; ++g) {
        auto& ctx = gpuFleet.ctx(g);
        if (g != 0 && counts[g] <= 0) continue;
        const float* src = ctx.readbackOut(bufBytes);
        for (size_t i = 0; i < pixFloats; ++i) merged[i] += src[i];

        ctx.downloadFromBuffer(ctx.adaptiveBuffer, tmp.data(), bufBytes);
        for (size_t i = 0; i < pixFloats; ++i) mergedAd[i] += tmp[i];
    }
    vkCtx.uploadToBuffer(vkCtx.outBuffer, merged.data(), bufBytes);
    vkCtx.uploadToBuffer(vkCtx.adaptiveBuffer, mergedAd.data(), bufBytes);
}

template<typename T>
static void uploadFogVolumes(GpuContext& vkCtx, GPUCameraData& camData,
                             const std::vector<T>& fogVolumes) {
    std::vector<GPUFogVolume> gpuFog;
    gpuFog.reserve(fogVolumes.size());
    for (const auto& fv : fogVolumes) {
        GPUFogVolume g{};
        g.minB = fv.minB;
        g.density = fv.density;
        g.maxB = fv.maxB;
        g.scatter = fv.scatterColor;
        g.absorb = fv.absorption;
        gpuFog.push_back(g);
    }
    vkCtx.updateFogBuffer(gpuFog);
    camData.fogVolumeCount = static_cast<int>(gpuFog.size());
}

template<typename T>
InFlightFrame Octree<T>::beginRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    vkCtx.awaitPostPass();
    vkCtx.awaitFastFullFrame();
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    gpuFleet.init();

    const std::vector<GPUMaterial>* gpuMaterials = nullptr;
    const std::vector<float>* sellLUT = nullptr;
    size_t sellRows = 0;
    renderMaterials_.retrieveGPUMaterials(buildGPUMaterialCache, gpuMaterials, sellLUT, sellRows);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        ctx.updateMaterialBuffer(*gpuMaterials);
        ctx.updateSellmeierBuffer(*sellLUT, SELL_LUT_WAVELENGTHS, sellRows);
    }

    thread_local SceneCache tl_scene;
    const bool sceneChanged = refreshSceneCache(tl_buffer, true, false, tl_scene);
    const std::vector<GPURenderData>& gpuPoints = tl_scene.gpuPoints;
    const std::vector<uint32_t>& gpuLights = tl_scene.gpuLights;
    const int emissiveCount = tl_scene.emissiveCount;

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    size_t skyW, skyH;
    const std::vector<Eigen::Vector4f>& skyData = getCachedSkyData(skyW, skyH);

    GPUCameraData camData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0, 
        0, (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, samplesPerPixel
    };
    camData.sellWidth = SELL_LUT_WAVELENGTHS;
    camData.sellSecondary = SELL_LUT_SECONDARY;

    size_t pixFloats = width * height * 5;
    size_t outSize = pixFloats * sizeof(float);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        uploadFogVolumes(ctx, camData, fogVolumes_);
        ctx.updateCommonBuffers(outSize, camData);
        ctx.updateSkyboxBuffer(skyData);
        if (sceneChanged || !ctx.pbrPointsResident) {
            ctx.updateLightBuffer(gpuLights);
            ctx.updatePBRBuffers(gpuPoints);
            ctx.pbrPointsResident = true;
        }
    }

    runWavefrontTilesMultiGPU(width, height, camData, samplesPerPixel, maxBounces, 0, pixFloats, outSize);

    frameCounter_++;

    vkCtx.submitSmooth(width, height, samplesPerPixel);

    InFlightFrame pending;
    pending.width = width;
    pending.height = height;
    pending.outSize = outSize;
    pending.colorformat = colorformat;
    pending.pending = true;
    return pending;
}

template<typename T>
static frame collectFinalOut(InFlightFrame& pending) {
    if (!pending.pending) return frame();
    vkCtx.awaitPostPass();
    pending.pending = false;

    frame outFrame(pending.width, pending.height, pending.colorformat);
    std::vector<float> colorBuffer(size_t(pending.width) * size_t(pending.height) * 3);
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}


template<typename T>
frame Octree<T>::endRenderFrameVulkan(InFlightFrame& pending) {
    TIME_FUNCTION;
    return collectFinalOut<T>(pending);
}

template<typename T>
frame Octree<T>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    InFlightFrame pending = beginRenderFrameVulkan(cam, height, width, colorformat, samplesPerPixel,
                                                   maxBounces, globalIllumination, useLod);
    return endRenderFrameVulkan(pending);
}

template<typename T>
InFlightFrame Octree<T>::beginFastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    TIME_FUNCTION;
    // ScopedFunctionTimer frfv("fast render frame vulkan startup");
    vkCtx.awaitFastFullFrame();
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();
    
    const std::vector<GPUMaterial>* gpuMaterials = nullptr;
    const std::vector<float>* sellLUT = nullptr;
    size_t sellRows = 0;
    renderMaterials_.retrieveGPUMaterials(buildGPUMaterialCache, gpuMaterials, sellLUT, sellRows);
    vkCtx.updateMaterialBuffer(*gpuMaterials);
    vkCtx.updateSellmeierBuffer(*sellLUT, SELL_LUT_WAVELENGTHS, sellRows);

    thread_local SceneCache tl_fastScene;
    const bool sceneChanged = refreshSceneCache(tl_buffer, false, true, tl_fastScene);
    const std::vector<GPURenderData>& gpuPoints = tl_fastScene.gpuPoints;
    const std::vector<uint32_t>& gpuLights = tl_fastScene.gpuLights;
    const int emissiveCount = tl_fastScene.emissiveCount;
    const Vec3 vctMin = tl_fastScene.boundsMin;
    const Vec3 vctMax = tl_fastScene.boundsMax;

    size_t skyW, skyH;
    const std::vector<Eigen::Vector4f>& skyData = getCachedSkyData(skyW, skyH);

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    GPUCameraData camData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, 1, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        0, (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(outSize, camData);
    vkCtx.updateSkyboxBuffer(skyData);
    if (sceneChanged || !vkCtx.fastPointsResident) {
        vkCtx.updateLightBuffer(gpuLights);
        vkCtx.updateFastBuffers(gpuPoints);
        vkCtx.fastPointsResident = true;
    }

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuPoints.size(), vctMin, vctMax, keyLight, true);
    }
    
    // frfv.stop();
    vkCtx.submitFastFullFrame(width, height);

    InFlightFrame pending;
    pending.width = width;
    pending.height = height;
    pending.outSize = outSize;
    pending.colorformat = colorformat;
    pending.pending = true;
    return pending;
}

template<typename T>
frame Octree<T>::endFastRenderFrameVulkan(InFlightFrame& pending) {
    TIME_FUNCTION;
    if (!pending.pending) return frame();

    vkCtx.awaitFastFullFrame();
    pending.pending = false;

    const int width = pending.width;
    const int height = pending.height;
    frame outFrame(width, height, pending.colorformat);
    std::vector<float> colorBuffer(width * height * 3);

    const float* raw = vkCtx.readbackOut(pending.outSize);
    const int pixelCount = width * height;
    for (int i = 0; i < pixelCount; ++i) {
        int outIdx = i * 3;
        int inIdx = i * 5;
        colorBuffer[outIdx] = std::clamp(raw[inIdx], 0.0f, 1.0f);
        colorBuffer[outIdx + 1] = std::clamp(raw[inIdx + 1], 0.0f, 1.0f);
        colorBuffer[outIdx + 2] = std::clamp(raw[inIdx + 2], 0.0f, 1.0f);
    }

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T>
frame Octree<T>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    InFlightFrame pending = beginFastRenderFrameVulkan(cam, height, width, colorformat);
    return endFastRenderFrameVulkan(pending);
}

template<typename T>
InFlightFrame Octree<T>::beginBlendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    vkCtx.awaitPostPass();
    vkCtx.awaitFastFullFrame();
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    gpuFleet.init();

    const std::vector<GPUMaterial>* gpuMaterials = nullptr;
    const std::vector<float>* sellLUT = nullptr;
    size_t sellRows = 0;
    renderMaterials_.retrieveGPUMaterials(buildGPUMaterialCache, gpuMaterials, sellLUT, sellRows);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        ctx.updateMaterialBuffer(*gpuMaterials);
        ctx.updateSellmeierBuffer(*sellLUT, SELL_LUT_WAVELENGTHS, sellRows);
    }

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Vec3 globalMin = Vec3::Constant(std::numeric_limits<float>::max());
    Vec3 globalMax = Vec3::Constant(std::numeric_limits<float>::lowest());
    
    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }
    
    std::vector<PointSort> sortedPoints;
    mortonSortIndices(validIndices, globalMin, globalMax,
                      [&](size_t idx) { return tl_buffer.points[idx].position; },
                      sortedPoints);

    std::vector<GPURenderData> gpuPBRPoints;
    gpuPBRPoints.reserve(sortedPoints.size());
    std::vector<GPURenderData> gpuFastPoints;
    gpuFastPoints.reserve(sortedPoints.size());
    std::vector<uint32_t> gpuLights;

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];
        
        gpuPBRPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.extent
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.extent
        });

        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            gpuLights.push_back(gpuPBRPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPURenderData{});
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPURenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    size_t skyW, skyH;
    const std::vector<Eigen::Vector4f>& skyData = getCachedSkyData(skyW, skyH);
    vkCtx.updateSkyboxBuffer(skyData);

    int lowW = std::max(1, static_cast<int>(width * pbrScale));
    int lowH = std::max(1, static_cast<int>(height * pbrScale));

    GPUCameraData pbrCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        lowW, lowH, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0, 
        0, (uint32_t)gpuPBRPoints.size(), 0, 0, emissiveCount, samplesPerPixel,
        SELL_LUT_WAVELENGTHS, SELL_LUT_SECONDARY
    };

    size_t pixFloats = lowW * lowH * 5;
    size_t pbrOutSize = pixFloats * sizeof(float);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        uploadFogVolumes(ctx, pbrCamData, fogVolumes_);
        ctx.updateCommonBuffers(pbrOutSize, pbrCamData);
        ctx.updateSkyboxBuffer(skyData);
        ctx.updateLightBuffer(gpuLights);
        ctx.updatePBRBuffers(gpuPBRPoints);
    }

    runWavefrontTilesMultiGPU(lowW, lowH, pbrCamData, samplesPerPixel, maxBounces, 0, pixFloats, pbrOutSize);

    vkCtx.dispatchSmoothPasses(lowW, lowH, samplesPerPixel, 2, false);
    vkCtx.ensureLowResBuffer(pbrOutSize);
    vkCtx.copyBuffer(vkCtx.device, vkCtx.outBuffer, vkCtx.lowResOutBuffer, pbrOutSize);

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        0, (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t fastOutSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.updateFastBuffers(gpuFastPoints);

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuFastPoints.size(),
                             globalMin, globalMax, keyLight, true);
    }

    fastCamData.tileOffsetX = 0;
    fastCamData.tileOffsetY = 0;
    vkCtx.updateCameraData(fastCamData);
    vkCtx.dispatchFastFullFrame(width, height);
    vkCtx.dispatchBlend(width, height, lowW, lowH, pbrScale, 1, true);

    InFlightFrame pending;
    pending.width = width;
    pending.height = height;
    pending.outSize = size_t(width) * size_t(height) * 5 * sizeof(float);
    pending.colorformat = colorformat;
    pending.pending = true;
    return pending;
}

template<typename T>
frame Octree<T>::endBlendedRenderFrameVulkan(InFlightFrame& pending) {
    TIME_FUNCTION;
    return collectFinalOut<T>(pending);
}

template<typename T>
frame Octree<T>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    InFlightFrame pending = beginBlendedRenderFrameVulkan(cam, height, width, pbrScale, colorformat,
                                                          samplesPerPixel, maxBounces, globalIllumination, useLod);
    return endBlendedRenderFrameVulkan(pending);
}

template<typename T>
InFlightFrame Octree<T>::beginGameStyleRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    TIME_FUNCTION;
    vkCtx.awaitFastFullFrame();
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);

    gpuFleet.init();

    const std::vector<GPUMaterial>* gpuMaterials = nullptr;
    const std::vector<float>* sellLUT = nullptr;
    size_t sellRows = 0;
    renderMaterials_.retrieveGPUMaterials(buildGPUMaterialCache, gpuMaterials, sellLUT, sellRows);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        ctx.updateMaterialBuffer(*gpuMaterials);
        ctx.updateSellmeierBuffer(*sellLUT, SELL_LUT_WAVELENGTHS, sellRows);
    }

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Vec3 globalMin = Vec3::Constant(std::numeric_limits<float>::max());
    Vec3 globalMax = Vec3::Constant(std::numeric_limits<float>::lowest());

    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }

    std::vector<PointSort> sortedPoints;
    mortonSortIndices(validIndices, globalMin, globalMax,
                      [&](size_t idx) { return tl_buffer.points[idx].position; },
                      sortedPoints);
    std::vector<GPURenderData> gpuFastPoints;
    gpuFastPoints.reserve(sortedPoints.size());
    struct LightRef { float power; uint32_t idx; };
    std::vector<LightRef> lightRefs;

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];

        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.extent
        });

        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            Vec3 emit = unpackRGB9E5(tl_buffer.materials[p.materialIdx].chromaticity);
            float lum = 0.2126f * emit.x() * p.color.x() + 0.7152f * emit.y() * p.color.y() + 0.0722f * emit.z() * p.color.z();
            lightRefs.push_back({lum * p.size * p.size, (uint32_t)(gpuFastPoints.size() - 1)});
        }
    }

    std::sort(lightRefs.begin(), lightRefs.end(),
              [](const LightRef& a, const LightRef& b) { return a.power > b.power; });
    std::vector<uint32_t> gpuLights;
    gpuLights.reserve(lightRefs.size());
    for (const auto& lr : lightRefs) gpuLights.push_back(lr.idx);

    int emissiveCount = (int)gpuLights.size();
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPURenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);
    
    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    size_t skyW, skyH;
    const std::vector<Eigen::Vector4f>& skyData = getCachedSkyData(skyW, skyH);
    vkCtx.updateSkyboxBuffer(skyData);

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, 1, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 2,
        0, (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t fastOutSize = size_t(width) * size_t(height) * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updateFastBuffers(gpuFastPoints);

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuFastPoints.size(),
                             globalMin, globalMax, keyLight, true);
    }


    fastCamData.tileOffsetX = 0;
    fastCamData.tileOffsetY = 0;
    vkCtx.updateCameraData(fastCamData);
    vkCtx.submitFastFullFrame(width, height);

    InFlightFrame pending;
    pending.width = width;
    pending.height = height;
    pending.outSize = fastOutSize;
    pending.colorformat = colorformat;
    pending.pending = true;
    return pending;
}

template<typename T>
frame Octree<T>::endGameStyleRenderFrame(InFlightFrame& pending) {
    TIME_FUNCTION;
    if (!pending.pending) return frame();

    vkCtx.awaitFastFullFrame();
    pending.pending = false;

    const int width = pending.width;
    const int height = pending.height;
    frame outFrame(width, height, pending.colorformat);
    std::vector<float> guide(size_t(width) * size_t(height) * 5);
    {
        const float* raw = vkCtx.readbackOut(pending.outSize);
        memcpy(guide.data(), raw, pending.outSize);
    }

    std::vector<float> colorBuffer(size_t(width) * size_t(height) * 3);
    for (size_t px = 0; px < size_t(width) * size_t(height); ++px) {
        colorBuffer[px * 3 + 0] = guide[px * 5 + 0];
        colorBuffer[px * 3 + 1] = guide[px * 5 + 1];
        colorBuffer[px * 3 + 2] = guide[px * 5 + 2];
    }

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T>
frame Octree<T>::GameStyleRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    InFlightFrame pending = beginGameStyleRenderFrame(cam, height, width, colorformat);
    return endGameStyleRenderFrame(pending);
}

template<typename T>
InFlightFrame Octree<T>::beginSuperBlendedRenderFrameVulkan(const Camera& cam, int height, int width, float ptScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination,
                bool useLod, int minSamplesPerPixel) {
    TIME_FUNCTION;
    vkCtx.awaitPostPass();
    vkCtx.awaitFastFullFrame();
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);

    gpuFleet.init();

    const std::vector<GPUMaterial>* gpuMaterials = nullptr;
    const std::vector<float>* sellLUT = nullptr;
    size_t sellRows = 0;
    renderMaterials_.retrieveGPUMaterials(buildGPUMaterialCache, gpuMaterials, sellLUT, sellRows);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        ctx.updateMaterialBuffer(*gpuMaterials);
        ctx.updateSellmeierBuffer(*sellLUT, SELL_LUT_WAVELENGTHS, sellRows);
    }

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Vec3 globalMin = Vec3::Constant(std::numeric_limits<float>::max());
    Vec3 globalMax = Vec3::Constant(std::numeric_limits<float>::lowest());

    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }

    std::vector<PointSort> sortedPoints;
    mortonSortIndices(validIndices, globalMin, globalMax,
                      [&](size_t idx) { return tl_buffer.points[idx].position; },
                      sortedPoints);

    std::vector<GPURenderData> gpuPBRPoints;
    gpuPBRPoints.reserve(sortedPoints.size());
    std::vector<GPURenderData> gpuFastPoints;
    gpuFastPoints.reserve(sortedPoints.size());


    struct LightRef { float power; uint32_t idx; };
    std::vector<LightRef> lightRefs;

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];

        gpuPBRPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.extent
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.extent
        });

        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            Vec3 emit = unpackRGB9E5(tl_buffer.materials[p.materialIdx].chromaticity);
            float lum = 0.2126f * emit.x() * p.color.x()
                      + 0.7152f * emit.y() * p.color.y()
                      + 0.0722f * emit.z() * p.color.z();
            lightRefs.push_back({lum * p.size * p.size, (uint32_t)(gpuPBRPoints.size() - 1)});
        }
    }

    std::sort(lightRefs.begin(), lightRefs.end(),
              [](const LightRef& a, const LightRef& b) { return a.power > b.power; });
    std::vector<uint32_t> gpuLights;
    gpuLights.reserve(lightRefs.size());
    for (const auto& lr : lightRefs) gpuLights.push_back(lr.idx);

    int emissiveCount = (int)gpuLights.size();
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPURenderData{});
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPURenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    size_t skyW, skyH;
    const std::vector<Eigen::Vector4f>& skyData = getCachedSkyData(skyW, skyH);
    vkCtx.updateSkyboxBuffer(skyData);

    int lowW = std::max(1, static_cast<int>(width * ptScale));
    int lowH = std::max(1, static_cast<int>(height * ptScale));

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_, (int)skyW, (int)skyH, 0, 1, 2,
        0, (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t fastOutSize = size_t(width) * size_t(height) * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updateFastBuffers(gpuFastPoints);

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuFastPoints.size(),
                             globalMin, globalMax, keyLight, true);
    }

    fastCamData.tileOffsetX = 0;
    fastCamData.tileOffsetY = 0;
    vkCtx.updateCameraData(fastCamData);
    vkCtx.dispatchFastFullFrame(width, height);

    std::vector<float> guide(size_t(width) * size_t(height) * 5);
    {
        const float* raw = vkCtx.readbackOut(fastOutSize);
        memcpy(guide.data(), raw, fastOutSize);
    }

    int minS = std::clamp(minSamplesPerPixel, 1, samplesPerPixel);
    std::vector<float> adaptiveSeed(size_t(lowW) * size_t(lowH) * 5, 0.0f);
    std::vector<float> pixelSeed(size_t(lowW) * size_t(lowH) * 5, 0.0f);

    float invScaleX = float(width)  / float(lowW);
    float invScaleY = float(height) / float(lowH);

    for (int ly = 0; ly < lowH; ++ly) {
        int y0 = (int)(ly * invScaleY);
        int y1 = std::min(height, std::max(y0 + 1, (int)((ly + 1) * invScaleY)));
        for (int lx = 0; lx < lowW; ++lx) {
            int x0 = (int)(lx * invScaleX);
            int x1 = std::min(width, std::max(x0 + 1, (int)((lx + 1) * invScaleX)));

            float lumMin = 1e30f, lumMax = -1e30f;
            float depthMin = 1e30f, depthMax = -1e30f;
            float firstObj = guide[(size_t(y0) * width + x0) * 5 + 4];
            bool objEdge = false;

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    size_t gi = (size_t(y) * width + x) * 5;
                    float lum = 0.2126f * guide[gi] + 0.7152f * guide[gi + 1] + 0.0722f * guide[gi + 2];
                    lumMin = std::min(lumMin, lum);
                    lumMax = std::max(lumMax, lum);
                    float d = guide[gi + 3];
                    if (d < 1e29f) {
                        depthMin = std::min(depthMin, d);
                        depthMax = std::max(depthMax, d);
                    }
                    if (guide[gi + 4] != firstObj) objEdge = true;
                }
            }

            float contrast = std::max(0.0f, lumMax - lumMin);
            float depthEdge = 0.0f;
            if (depthMax > depthMin && depthMin < 1e29f) {
                depthEdge = std::min(1.0f, (depthMax - depthMin) / std::max(depthMin, 0.1f));
            }
            float importance = 0.15f + 1.5f * contrast
                             + 0.4f * (objEdge ? 1.0f : 0.0f)
                             + 0.4f * depthEdge;
            importance = std::clamp(importance, 0.0f, 1.0f);
            int req = std::clamp((int)std::lround(samplesPerPixel * importance), minS, samplesPerPixel);

            size_t ai = (size_t(ly) * lowW + lx) * 5;
            adaptiveSeed[ai]     = 0.0f;
            adaptiveSeed[ai + 1] = 0.0f;
            adaptiveSeed[ai + 2] = (float)req;
            adaptiveSeed[ai + 3] = 0.0f;
            adaptiveSeed[ai + 4] = 0.0f;

            int cx = std::min(width - 1,  (int)((lx + 0.5f) * invScaleX));
            int cy = std::min(height - 1, (int)((ly + 0.5f) * invScaleY));
            pixelSeed[ai + 4] = guide[(size_t(cy) * width + cx) * 5 + 4];
        }
    }

    GPUCameraData pbrCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        lowW, lowH, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_++,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0,
        0, (uint32_t)gpuPBRPoints.size(), 0, 0, emissiveCount, samplesPerPixel,
        SELL_LUT_WAVELENGTHS, SELL_LUT_SECONDARY
    };

    size_t pixFloats = lowW * lowH * 5;
    size_t pbrOutSize = pixFloats * sizeof(float);
    for (size_t g = 0; g < gpuFleet.count(); ++g) {
        auto& ctx = gpuFleet.ctx(g);
        uploadFogVolumes(ctx, pbrCamData, fogVolumes_);
        ctx.updateCommonBuffers(pbrOutSize, pbrCamData);
        if (g > 0) {
            ctx.updateSkyboxBuffer(skyData);
            ctx.updateLightBuffer(gpuLights);
        }
        ctx.updatePBRBuffers(gpuPBRPoints);
        ctx.uploadToBuffer(ctx.adaptiveBuffer, adaptiveSeed.data(), adaptiveSeed.size() * sizeof(float));
        ctx.uploadToBuffer(ctx.outBuffer, pixelSeed.data(), pixelSeed.size() * sizeof(float));
    }

    runWavefrontTilesMultiGPU(lowW, lowH, pbrCamData, samplesPerPixel, maxBounces, 1, pixFloats, pbrOutSize, true);
    vkCtx.dispatchSmoothPasses(lowW, lowH, samplesPerPixel, 2, false);
    vkCtx.ensureLowResBuffer(pbrOutSize);
    vkCtx.copyBuffer(vkCtx.device, vkCtx.outBuffer, vkCtx.lowResOutBuffer, pbrOutSize);

    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.uploadToBuffer(vkCtx.outBuffer, guide.data(), fastOutSize);

    vkCtx.dispatchBlend(width, height, lowW, lowH, ptScale, 1, true);

    InFlightFrame pending;
    pending.width = width;
    pending.height = height;
    pending.outSize = fastOutSize;
    pending.colorformat = colorformat;
    pending.pending = true;
    return pending;
}

template<typename T>
frame Octree<T>::endSuperBlendedRenderFrameVulkan(InFlightFrame& pending) {
    TIME_FUNCTION;
    return collectFinalOut<T>(pending);
}

template<typename T>
frame Octree<T>::superBlendedRenderFrameVulkan(const Camera& cam, int height, int width, float ptScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination,
                bool useLod, int minSamplesPerPixel) {
    InFlightFrame pending = beginSuperBlendedRenderFrameVulkan(cam, height, width, ptScale, colorformat,
                                samplesPerPixel, maxBounces, globalIllumination, useLod, minSamplesPerPixel);
    return endSuperBlendedRenderFrameVulkan(pending);
}

}