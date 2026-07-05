#include "grid3eigen.hpp"
namespace Grid {

template<typename T>
void Octree<T>::buildRender(RenderBuffer_<T>& buffer) {
    buffer.clear();
    if (!root_) return;
    buffer.nodes.emplace_back();

    std::unordered_map<int, std::shared_ptr<GridObject_<T>>> localObjects;
    {
        s_lock lock(objectsMutex_);
        localObjects = objects_;
    }

    for (auto& kv : localObjects) {
        buffer.objMaterialOffsets[kv.first] = buffer.materials.size();
        s_lock oLock(kv.second->objMutex);
        for (auto& m : kv.second->renderMaterials) {
            buffer.materials.push_back(m);
        }
    }
    buffer.defaultMatIdx = buffer.materials.size();
    buffer.materials.push_back(RenderMaterial());

    buildRenderNodeAt(root_.get(), buffer, 0, localObjects);
}

template<typename T>
void Octree<T>::buildRenderNodeAt(OctreeNode_<T>* node, RenderBuffer_<T>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects) {
    s_lock lock(node->nodeMutex);
    bool isLoaded = node->isLoaded();
    
    RenderNode_<T> rnode;
    rnode.center = node->center;
    rnode.nodeSize = node->nodeSize;
    rnode.isLeaf = node->isLeaf();
    rnode.isLoaded = isLoaded;
    rnode.originalNode = node; 
    
    rnode.firstPoint = static_cast<uint32_t>(buffer.points.size());
    if (isLoaded) {
        for (const auto& pt : node->points) {
            if (!pt->isActive() || !pt->isVisible()) continue; 
            RenderData rd;
            rd.position = pt->position;
            rd.size = pt->size;
            rd.color = pt->color;
            
            auto objIt = localObjects.find(pt->objectId);
            
            rd.materialIdx = buffer.defaultMatIdx;
            auto it = buffer.objMaterialOffsets.find(pt->objectId);
            if (it != buffer.objMaterialOffsets.end()) {
                rd.materialIdx = it->second + pt->renderMatIdx;
            }
            
            rd.objectId = pt->objectId;
            buffer.points.push_back(rd);
        }
    }

    rnode.pointCount = static_cast<uint32_t>(buffer.points.size() - rnode.firstPoint);
    
    rnode.lodPoint = -1;
    if (node->lodData) {
        RenderData ld;
        ld.position = node->lodData->position;
        ld.size = node->lodData->size;
        ld.color = node->lodData->color;
        
        
        ld.materialIdx = buffer.defaultMatIdx;
        auto it = buffer.objMaterialOffsets.find(node->lodData->objectId);
        if (it != buffer.objMaterialOffsets.end()) {
            ld.materialIdx = it->second + node->lodData->renderMatIdx;
        }
        
        ld.objectId = node->lodData->objectId; 
        rnode.lodPoint = static_cast<int32_t>(buffer.points.size());
        buffer.points.push_back(ld);
    }
    
    rnode.childMask = 0;
    rnode.firstChild = 0;
    
    if (!node->isLeaf() && isLoaded) {
        uint8_t mask = 0;
        int childCount = 0;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                mask |= (1 << i);
                childCount++;
            }
        }
        rnode.childMask = mask;
        if (childCount > 0) {
            rnode.firstChild = static_cast<uint32_t>(buffer.nodes.size());
            buffer.nodes.resize(buffer.nodes.size() + childCount);
            int cidx = 0;
            for (int i = 0; i < 8; ++i) {
                if (mask & (1 << i)) {
                    buildRenderNodeAt(node->children[i].get(), buffer, rnode.firstChild + cidx, localObjects);
                    cidx++;
                }
            }
        }
    }
    
    buffer.nodes[nodeIdx] = rnode;
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

            if (!node.isLoaded && node.originalNode) {
                ensureLoaded(node.originalNode, true);
            }

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

#ifdef VULKAN_SUPPORT

static inline uint32_t packRGB8(const Vec3& c) {
    uint32_t r = static_cast<uint32_t>(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f);
    uint32_t g = static_cast<uint32_t>(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f);
    uint32_t b = static_cast<uint32_t>(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f);
    return r | (g << 8) | (b << 16);
}

static inline uint32_t packRGBA8(const Eigen::Vector4f& c) {
    uint32_t r = static_cast<uint32_t>(std::clamp(c.x(), 0.0f, 1.0f) * 255.0f);
    uint32_t g = static_cast<uint32_t>(std::clamp(c.y(), 0.0f, 1.0f) * 255.0f);
    uint32_t b = static_cast<uint32_t>(std::clamp(c.z(), 0.0f, 1.0f) * 255.0f);
    uint32_t a = static_cast<uint32_t>(std::clamp(c.w(), 0.0f, 1.0f) * 255.0f);
    return r | (g << 8) | (b << 16) | (a << 24);
}

static inline uint32_t packMaterialProps(float roughness, float metallic, uint32_t sellmeierRow) {
    uint32_t r8 = static_cast<uint32_t>(std::clamp(roughness, 0.0f, 1.0f) * 255.0f);
    uint32_t m8 = static_cast<uint32_t>(std::clamp(metallic, 0.0f, 1.0f) * 255.0f);
    uint32_t row16 = sellmeierRow & 0xFFFFu;
    return r8 | (m8 << 8) | (row16 << 16);
}
static constexpr int SELL_LUT_WAVELENGTHS = 32;
static constexpr int SELL_LUT_SECONDARY   = 8;
static constexpr float SELL_LMIN = 0.380f; // um
static constexpr float SELL_LMAX = 0.720f; // um

static inline std::vector<float> buildSellmeierLUT(const std::vector<Grid::RenderMaterial>& mats) {
    int rows = std::max<size_t>(1, mats.size()) * SELL_LUT_SECONDARY;
    std::vector<float> lut(static_cast<size_t>(rows) * SELL_LUT_WAVELENGTHS, 1.0f);
    for (size_t mi = 0; mi < mats.size(); ++mi) {
        const auto& m = mats[mi];
        for (int s = 0; s < SELL_LUT_SECONDARY; ++s) {
            int row = static_cast<int>(mi) * SELL_LUT_SECONDARY + s;
            for (int w = 0; w < SELL_LUT_WAVELENGTHS; ++w) {
                float f = (SELL_LUT_WAVELENGTHS == 1) ? 0.0f
                          : float(w) / float(SELL_LUT_WAVELENGTHS - 1);
                float lambda = SELL_LMIN + f * (SELL_LMAX - SELL_LMIN);
                lut[static_cast<size_t>(row) * SELL_LUT_WAVELENGTHS + w] =
                    Grid::sellmeierN(m.sellB, m.sellC, lambda);
            }
        }
    }
    return lut;
}

struct PointSort {
    uint64_t morton;
    size_t idx;
    bool operator<(const PointSort& o) const { return morton < o.morton; }
};

template<typename T>
void Octree<T>::buildGPUMaterials(const RenderBuffer_<T>& buf, std::vector<GPUMaterial>& out) {
    out.clear();
    out.reserve(buf.materials.size());
    for (size_t mi = 0; mi < buf.materials.size(); ++mi) {
        const auto& m = buf.materials[mi];
        uint32_t sellRow = static_cast<uint32_t>(mi) * SELL_LUT_SECONDARY;
        uint32_t albedoPacked = 0;
        out.push_back({
            m.chromaticity,
            packMaterialProps(m.roughness, m.metallic, sellRow),
            packRGB8(m.absorption),
            albedoPacked
        });
    }
    if (out.empty()) out.push_back(GPUMaterial{});
}


#ifdef VULKAN_SUPPORT
static void runWavefrontTilesMultiGPU(int width, int height, const GPUCameraData& camTemplate,
                                      int samplesPerPixel, int maxBounces, int sampleOffset) {
    TIME_FUNCTION;
    constexpr int tileW = 512, tileH = 512;

    struct Tile { int x, y, w, h; int owner; };
    std::vector<Tile> tiles;
    for (int y = 0; y < height; y += tileH)
        for (int x = 0; x < width; x += tileW)
            tiles.push_back({x, y, std::min(tileW, width - x), std::min(tileH, height - y), -1});

    auto renderOn = [&](VulkanContext& ctx, Tile& t) {
        GPUCameraData cd = camTemplate;
        cd.tileOffsetX = t.x;
        cd.tileOffsetY = t.y;
        cd.currentSampleOffset = sampleOffset;
        cd.dispatchSamples = samplesPerPixel;
        ctx.updateCameraData(cd);
        ctx.dispatchWavefront(t.w, t.h, maxBounces, samplesPerPixel);
    };

    const size_t nGPU = gpuMgr.count();
    if (nGPU <= 1) {
        for (auto& t : tiles) renderOn(vkCtx, t);
        return;
    }

    // Work-stealing dispatch: one host thread per GPU. Vulkan calls target
    // disjoint VkDevices, so no cross-thread synchronization is needed beyond
    // the tile counter.
    std::atomic<size_t> next{0};
    std::vector<int> tilesDone(nGPU, 0);
    std::vector<double> msSpent(nGPU, 0.0);
    std::vector<std::thread> workers;
    workers.reserve(nGPU);
    for (size_t g = 0; g < nGPU; ++g) {
        workers.emplace_back([&, g] {
            auto& ctx = gpuMgr.ctx(g);
            for (;;) {
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= tiles.size()) break;
                tiles[i].owner = (int)g;
                auto t0 = std::chrono::steady_clock::now();
                renderOn(ctx, tiles[i]);
                msSpent[g] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++tilesDone[g];
            }
        });
    }
    for (auto& w : workers) w.join();

    if (std::getenv("SSC_GPU_VERBOSE")) {
        for (size_t g = 0; g < nGPU; ++g)
            std::cout << "  gpu[" << g << "] " << gpuMgr.ctx(g).deviceName << ": "
                      << tilesDone[g] << " tiles, " << (int)msSpent[g] << " ms" << std::endl;
    }

    // Merge: pull each GPU's tiles out of its own accumulation buffer into a
    // host image, then push the stitched image into the primary context.
    const size_t floatsPerPixel = 5;
    std::vector<float> merged(size_t(width) * size_t(height) * floatsPerPixel);
    for (size_t g = 0; g < nGPU; ++g) {
        auto& ctx = gpuMgr.ctx(g);
        void* mapped = nullptr;
        vkMapMemory(ctx.device, ctx.outMem, 0, merged.size() * sizeof(float), 0, &mapped);
        if (!ctx.outMemCoherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = ctx.outMem;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(ctx.device, 1, &range);
        }
        const float* src = static_cast<const float*>(mapped);
        for (const auto& t : tiles) {
            if (t.owner != (int)g) continue;
            for (int row = t.y; row < t.y + t.h; ++row) {
                size_t off = (size_t(row) * width + t.x) * floatsPerPixel;
                memcpy(merged.data() + off, src + off, size_t(t.w) * floatsPerPixel * sizeof(float));
            }
        }
        vkUnmapMemory(ctx.device, ctx.outMem);
    }
    vkCtx.uploadToBuffer(vkCtx.outBuffer, merged.data(), (uint32_t)(merged.size() * sizeof(float)));
}
#endif

// Uploads the octree's fog volumes to the GPU and stamps the count into camData.
template<typename T>
static void uploadFogVolumes(VulkanContext& vkCtx, GPUCameraData& camData,
                             const std::vector<T>& fogVolumes) {
    std::vector<GPUFogVolume> gpuFog;
    gpuFog.reserve(fogVolumes.size());
    for (const auto& fv : fogVolumes) {
        GPUFogVolume g{};
        g.minB = fv.minB;  g.density = fv.density;
        g.maxB = fv.maxB;
        g.scatter = fv.scatterColor;
        g.absorb = fv.absorption;
        gpuFog.push_back(g);
    }
    vkCtx.updateFogBuffer(gpuFog);
    camData.fogVolumeCount = static_cast<int>(gpuFog.size());
}

template<typename T>
frame Octree<T>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    gpuMgr.init();

    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    std::vector<float> sellLUT = buildSellmeierLUT(tl_buffer.materials);
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
        ctx.updateMaterialBuffer(gpuMaterials);
        ctx.updateSellmeierBuffer(sellLUT,
                                  SELL_LUT_WAVELENGTHS,
                                  std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);
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
    
    Vec3 extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Vec3 invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for(size_t idx : validIndices) {
        Vec3 normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
        uint32_t x = std::min(std::max(normPos.x() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t y = std::min(std::max(normPos.y() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t z = std::min(std::max(normPos.z() * 2097151.0f, 0.0f), 2097151.0f);
        
        uint64_t m = 0;
        for (int i = 0; i < 21; ++i) {
            m |= ((uint64_t)((x >> i) & 1) << (3 * i)) |
                 ((uint64_t)((y >> i) & 1) << (3 * i + 1)) |
                 ((uint64_t)((z >> i) & 1) << (3 * i + 2));
        }
        sortedPoints.push_back({m, idx});
    }
    
    std::sort(sortedPoints.begin(), sortedPoints.end());

    std::vector<GPUPBRRenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    gpuPoints.reserve(sortedPoints.size());
    
    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];
        
        gpuPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId
        });

        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuPoints.empty()) gpuPoints.push_back(GPUPBRRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

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

    size_t outSize = width * height * 5 * sizeof(float);
    // Upload the full scene to every GPU; tiles are then distributed across
    // all of them by runWavefrontTilesMultiGPU (work stealing = dynamic load
    // balancing across heterogeneous GPUs).
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
        uploadFogVolumes(ctx, camData, fogVolumes_);
        ctx.updateCommonBuffers(outSize, camData);
        ctx.updateSkyboxBuffer(skyData);
        ctx.updateLightBuffer(gpuLights);
        ctx.updatePBRBuffers(gpuPoints);
    }

    runWavefrontTilesMultiGPU(width, height, camData, samplesPerPixel, maxBounces, /*sampleOffset=*/0);

    frameCounter_++;

    vkCtx.dispatchSmooth(width, height, samplesPerPixel);

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T>
frame Octree<T>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    TIME_FUNCTION;
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();
    
    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    vkCtx.updateMaterialBuffer(gpuMaterials);
    vkCtx.updateSellmeierBuffer(buildSellmeierLUT(tl_buffer.materials), SELL_LUT_WAVELENGTHS,
                                std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    std::vector<GPUFastRenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    gpuPoints.reserve(tl_buffer.points.size());
    Vec3 vctMin = Vec3::Constant(std::numeric_limits<float>::max());
    Vec3 vctMax = Vec3::Constant(std::numeric_limits<float>::lowest());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        const auto& p = tl_buffer.points[i];
        
        gpuPoints.push_back({p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId});
        
        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }

        float h = p.size * 0.5f;
        vctMin = vctMin.cwiseMin(p.position - Vec3::Constant(h));
        vctMax = vctMax.cwiseMax(p.position + Vec3::Constant(h));
    }
    if (vctMin.x() > vctMax.x()) { vctMin.setZero(); vctMax.setOnes(); } // empty scene guard

    int emissiveCount = gpuLights.size();
    if(gpuPoints.empty()) gpuPoints.push_back(GPUFastRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

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
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updateFastBuffers(gpuPoints);

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuPoints.size(),
                             vctMin, vctMax, keyLight, /*enabled=*/true);
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
    vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipeline);
    vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipelineLayout, 0, 1, &vkCtx.fastDescSet, 0, nullptr);
    
    vkCmdDispatch(vkCtx.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
    vkEndCommandBuffer(vkCtx.commandBuffer);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vkCtx.commandBuffer;

    if (vkCtx.renderFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &vkCtx.renderFence);
    } else {
        vkResetFences(vkCtx.device, 1, &vkCtx.renderFence);
    }

    {
        vkQueueSubmit(vkCtx.queue, 1, &submitInfo, vkCtx.renderFence);
        vkWaitForFences(vkCtx.device, 1, &vkCtx.renderFence, VK_TRUE, UINT64_MAX);
    }

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);

    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.outMem, 0, outSize, 0, &mappedData);
    if (!vkCtx.outMemCoherent) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = vkCtx.outMem;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(vkCtx.device, 1, &range);
    }
    const float* raw = static_cast<const float*>(mappedData);
    const int pixelCount = width * height;
    for (int i = 0; i < pixelCount; ++i) {
        int outIdx = i * 3;
        int inIdx = i * 5;
        colorBuffer[outIdx]     = std::clamp(raw[inIdx],     0.0f, 1.0f);
        colorBuffer[outIdx + 1] = std::clamp(raw[inIdx + 1], 0.0f, 1.0f);
        colorBuffer[outIdx + 2] = std::clamp(raw[inIdx + 2], 0.0f, 1.0f);
    }
    vkUnmapMemory(vkCtx.device, vkCtx.outMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T>
frame Octree<T>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    gpuMgr.init();

    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    std::vector<float> sellLUT = buildSellmeierLUT(tl_buffer.materials);
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
        ctx.updateMaterialBuffer(gpuMaterials);
        ctx.updateSellmeierBuffer(sellLUT,
                                  SELL_LUT_WAVELENGTHS,
                                  std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);
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
    
    Vec3 extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Vec3 invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for(size_t idx : validIndices) {
        Vec3 normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
        uint32_t x = std::min(std::max(normPos.x() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t y = std::min(std::max(normPos.y() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t z = std::min(std::max(normPos.z() * 2097151.0f, 0.0f), 2097151.0f);
        
        uint64_t m = 0;
        for (int i = 0; i < 21; ++i) {
            m |= ((uint64_t)((x >> i) & 1) << (3 * i)) |
                 ((uint64_t)((y >> i) & 1) << (3 * i + 1)) |
                 ((uint64_t)((z >> i) & 1) << (3 * i + 2));
        }
        sortedPoints.push_back({m, idx});
    }
    
    std::sort(sortedPoints.begin(), sortedPoints.end());

    std::vector<GPUPBRRenderData> gpuPBRPoints;
    gpuPBRPoints.reserve(sortedPoints.size());
    std::vector<GPUFastRenderData> gpuFastPoints;
    gpuFastPoints.reserve(sortedPoints.size());
    std::vector<uint32_t> gpuLights;

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];
        
        gpuPBRPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId
        });

        if (tl_buffer.materials[p.materialIdx].chromaticity != 0u) {
            gpuLights.push_back(gpuPBRPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPUPBRRenderData{});
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPUFastRenderData{});
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

    size_t pbrOutSize = lowW * lowH * 5 * sizeof(float);
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
        uploadFogVolumes(ctx, pbrCamData, fogVolumes_);
        ctx.updateCommonBuffers(pbrOutSize, pbrCamData);
        ctx.updateSkyboxBuffer(skyData);
        ctx.updateLightBuffer(gpuLights);
        ctx.updatePBRBuffers(gpuPBRPoints);
    }

    runWavefrontTilesMultiGPU(width, height, pbrCamData, samplesPerPixel, maxBounces, 0);

    vkCtx.dispatchSmoothPasses(lowW, lowH, samplesPerPixel, 2, false);
    vkCtx.ensureLowResBuffer(pbrOutSize);
    vkCtx.copyBuffer(vkCtx.outBuffer, vkCtx.lowResOutBuffer, pbrOutSize);

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        0, (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t fastOutSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.updateFastBuffers(gpuFastPoints);

    // VCT: build the radiance volume once for the whole frame (shared across tiles).
    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuFastPoints.size(),
                             globalMin, globalMax, keyLight, /*enabled=*/true);
    }

    int tileW = 512;
    int tileH = 512;
    for (int y = 0; y < height; y += tileH) {
        for (int x = 0; x < width; x += tileW) {
            int drawW = std::min(tileW, width - x);
            int drawH = std::min(tileH, height - y);
            fastCamData.tileOffsetX = x;
            fastCamData.tileOffsetY = y;
            vkCtx.updateCameraData(fastCamData);

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
            vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipeline);
            vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipelineLayout, 0, 1, &vkCtx.fastDescSet, 0, nullptr);
            vkCmdDispatch(vkCtx.commandBuffer, (drawW + 7) / 8, (drawH + 7) / 8, 1);
            vkEndCommandBuffer(vkCtx.commandBuffer);

            VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &vkCtx.commandBuffer;

            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence fence;
            vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &fence);
            vkQueueSubmit(vkCtx.queue, 1, &submitInfo, fence);
            vkWaitForFences(vkCtx.device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(vkCtx.device, fence, nullptr);
        }
    }

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    
    vkCtx.dispatchBlend(width, height, lowW, lowH, pbrScale, 1);
    
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T>
frame Octree<T>::superBlendedRenderFrameVulkan(const Camera& cam, int height, int width, float ptScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination,
                bool useLod, int minSamplesPerPixel) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);

    gpuMgr.init();

    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    std::vector<float> sellLUT = buildSellmeierLUT(tl_buffer.materials);
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
        ctx.updateMaterialBuffer(gpuMaterials);
        ctx.updateSellmeierBuffer(sellLUT,
                                  SELL_LUT_WAVELENGTHS,
                                  std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);
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

    Vec3 extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Vec3 invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for(size_t idx : validIndices) {
        Vec3 normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
        uint32_t x = std::min(std::max(normPos.x() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t y = std::min(std::max(normPos.y() * 2097151.0f, 0.0f), 2097151.0f);
        uint32_t z = std::min(std::max(normPos.z() * 2097151.0f, 0.0f), 2097151.0f);

        uint64_t m = 0;
        for (int i = 0; i < 21; ++i) {
            m |= ((uint64_t)((x >> i) & 1) << (3 * i)) |
                 ((uint64_t)((y >> i) & 1) << (3 * i + 1)) |
                 ((uint64_t)((z >> i) & 1) << (3 * i + 2));
        }
        sortedPoints.push_back({m, idx});
    }
    std::sort(sortedPoints.begin(), sortedPoints.end());

    std::vector<GPUPBRRenderData> gpuPBRPoints;
    gpuPBRPoints.reserve(sortedPoints.size());
    std::vector<GPUFastRenderData> gpuFastPoints;
    gpuFastPoints.reserve(sortedPoints.size());

    // Lights sorted by emissive power (emittance * albedo luminance * area) so
    // the guide pass can just take the first three as the "primary" lights.
    struct LightRef { float power; uint32_t idx; };
    std::vector<LightRef> lightRefs;

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];

        gpuPBRPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId
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
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPUPBRRenderData{});
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPUFastRenderData{});
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

    // ------------- Stage 1: full-resolution deterministic guide pass -------------
    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_, (int)skyW, (int)skyH, 0, 1,
        /*globalIllumination = 2 -> top-3-light guide mode in fast_raytrace_hw*/ 2,
        0, (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t fastOutSize = size_t(width) * size_t(height) * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updateFastBuffers(gpuFastPoints);

    {
        Vec3 keyLight = (-cam.direction.normalized());
        vkCtx.vctBuildVolume(vkCtx.fastPointBuffer, (uint32_t)gpuFastPoints.size(),
                             globalMin, globalMax, keyLight, /*enabled=*/true);
    }

    {
        int tileW = 512, tileH = 512;
        for (int y = 0; y < height; y += tileH) {
            for (int x = 0; x < width; x += tileW) {
                int drawW = std::min(tileW, width - x);
                int drawH = std::min(tileH, height - y);
                fastCamData.tileOffsetX = x;
                fastCamData.tileOffsetY = y;
                vkCtx.updateCameraData(fastCamData);

                VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
                vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipeline);
                vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipelineLayout, 0, 1, &vkCtx.fastDescSet, 0, nullptr);
                vkCmdDispatch(vkCtx.commandBuffer, (drawW + 7) / 8, (drawH + 7) / 8, 1);
                vkEndCommandBuffer(vkCtx.commandBuffer);

                VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &vkCtx.commandBuffer;

                VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                VkFence fence;
                vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &fence);
                vkQueueSubmit(vkCtx.queue, 1, &submitInfo, fence);
                vkWaitForFences(vkCtx.device, 1, &fence, VK_TRUE, UINT64_MAX);
                vkDestroyFence(vkCtx.device, fence, nullptr);
            }
        }
    }

    // Read the guide image back (rgb, depth, objId per pixel).
    std::vector<float> guide(size_t(width) * size_t(height) * 5);
    {
        void* mappedData;
        vkMapMemory(vkCtx.device, vkCtx.outMem, 0, fastOutSize, 0, &mappedData);
        if (!vkCtx.outMemCoherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = vkCtx.outMem;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(vkCtx.device, 1, &range);
        }
        memcpy(guide.data(), mappedData, fastOutSize);
        vkUnmapMemory(vkCtx.device, vkCtx.outMem);
    }

    // ------------- Stage 2: importance map from the guide -------------
    // For each low-res path-trace pixel, inspect its footprint in the guide:
    // luminance contrast and object/depth discontinuities mean the MIS
    // integrator needs more samples there; flat areas get the minimum.
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
            adaptiveSeed[ai]     = 0.0f;           // running luminance sum
            adaptiveSeed[ai + 1] = 0.0f;           // running luminance^2 sum
            adaptiveSeed[ai + 2] = (float)req;     // pre-seeded sample budget
            adaptiveSeed[ai + 3] = 0.0f;
            adaptiveSeed[ai + 4] = 0.0f;

            // rgb + depth accumulate from zero; objId is normally only written
            // on the first-ever sample, so seed it from the guide for the blend.
            int cx = std::min(width - 1,  (int)((lx + 0.5f) * invScaleX));
            int cy = std::min(height - 1, (int)((ly + 0.5f) * invScaleY));
            pixelSeed[ai + 4] = guide[(size_t(cy) * width + cx) * 5 + 4];
        }
    }

    // ------------- Stage 3: low-res MIS path trace with seeded budgets -------------
    GPUCameraData pbrCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        lowW, lowH, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_++,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0,
        0, (uint32_t)gpuPBRPoints.size(), 0, 0, emissiveCount, samplesPerPixel,
        SELL_LUT_WAVELENGTHS, SELL_LUT_SECONDARY
    };

    size_t pbrOutSize = size_t(lowW) * size_t(lowH) * 5 * sizeof(float);
    for (size_t g = 0; g < gpuMgr.count(); ++g) {
        auto& ctx = gpuMgr.ctx(g);
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

    runWavefrontTilesMultiGPU(lowW, lowH, pbrCamData, samplesPerPixel, maxBounces, 1);
    vkCtx.dispatchSmoothPasses(lowW, lowH, samplesPerPixel, 2, false);
    vkCtx.ensureLowResBuffer(pbrOutSize);
    vkCtx.copyBuffer(vkCtx.outBuffer, vkCtx.lowResOutBuffer, pbrOutSize);

    // ------------- Stage 4: blend guide (full res) with path trace (low res) -------------
    // Restore the guide image into the full-res slot instead of re-running the
    // fast pass; it is the same image.
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData);
    vkCtx.uploadToBuffer(vkCtx.outBuffer, guide.data(), fastOutSize);

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(size_t(width) * size_t(height) * 3);

    vkCtx.dispatchBlend(width, height, lowW, lowH, ptScale, 1);

    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

#endif
}