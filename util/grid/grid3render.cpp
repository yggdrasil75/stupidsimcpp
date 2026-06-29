#include "grid3eigen.hpp"
namespace Grid {

template<typename T, typename IndexType>
void Octree<T, IndexType>::buildRender(RenderBuffer_<T, IndexType>& buffer) {
    buffer.clear();
    if (!root_) return;
    buffer.nodes.emplace_back();

    std::unordered_map<int, std::shared_ptr<GridObject_<T, IndexType>>> localObjects;
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        localObjects = objects_;
    }

    for (auto& kv : localObjects) {
        buffer.objMaterialOffsets[kv.first] = buffer.materials.size();
        std::shared_lock<std::shared_mutex> oLock(kv.second->objMutex);
        for (auto& m : kv.second->renderMaterials) {
            buffer.materials.push_back(m);
        }
    }
    buffer.defaultMatIdx = buffer.materials.size();
    buffer.materials.push_back(Material_());
    
    buffer.gasMaterialOffset = static_cast<uint32_t>(buffer.materials.size());
    {
        size_t gasCount = gasRegistry_.size();
        for (size_t i = 0; i < gasCount; ++i) {
            GasSpecies_ s = gasRegistry_.get(static_cast<uint16_t>(i));
            Material_ gm{};
            gm.absorption = s.effectiveAbsorption();
            gm.roughness = 1.0f;
            gm.metallic = 0.0f;
            gm.emittance = s.emittance;
            buffer.materials.push_back(gm);
        }
    }

    buildRenderNodeAt(root_.get(), buffer, 0, localObjects);
}

template<typename T, typename IndexType>
void Octree<T, IndexType>::buildRenderNodeAt(OctreeNode_<T, IndexType>* node, RenderBuffer_<T, IndexType>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects) {
    std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
    bool isLoaded = node->isLoaded();
    
    RenderNode_<T, IndexType> rnode;
    rnode.boundsMin = node->bounds.first;
    rnode.boundsMax = node->bounds.second;
    rnode.center = node->center;
    rnode.nodeSize = node->nodeSize;
    rnode.isLeaf = node->isLeaf();
    rnode.isLoaded = isLoaded;
    rnode.originalNode = node; 
    
    rnode.firstPoint = static_cast<uint32_t>(buffer.points.size());
    if (isLoaded) {
        for (const auto& pt : node->points) {
            if (!pt->isActive() || !pt->isVisible()) continue; 
            RenderData_ rd;
            rd.position = pt->position;
            rd.size = pt->size;
            rd.color = pt->color;
            
            uint32_t isGas = 0;
            auto objIt = localObjects.find(pt->objectId);
            if (objIt != localObjects.end()) {
                if (objIt->second->getPhysicsMaterial(pt->physMatIdx).type == BodyType::GAS) {
                    isGas = 1;
                }
            }
            
            rd.materialIdx = buffer.defaultMatIdx;
            auto it = buffer.objMaterialOffsets.find(pt->objectId);
            if (it != buffer.objMaterialOffsets.end()) {
                rd.materialIdx = it->second + pt->renderMatIdx;
            }
            rd.isGas = isGas;
            
            BoundingBox bb = pt->getCubeBounds();
            rd.boundsMin = bb.first;
            rd.boundsMax = bb.second;
            rd.objectId = pt->objectId;
            buffer.points.push_back(rd);
        }
    }
    
    if (isLoaded && node->gasField && !node->gasField->isEmpty()) {
        auto* field = node->gasField.get();
        const uint32_t R = field->res;
        const size_t cellCount = static_cast<size_t>(R) * R * R;

        GPUGasField gf{};
        gf.boundsMin = Eigen::Vector4f(field->bounds.first.x(), field->bounds.first.y(), field->bounds.first.z(), 0.0f);
        gf.boundsMax = Eigen::Vector4f(field->bounds.second.x(), field->bounds.second.y(), field->bounds.second.z(), 0.0f);
        gf.cellSize = Eigen::Vector4f(field->cellSize.x(), field->cellSize.y(), field->cellSize.z(), 0.0f);
        gf.res = R;
        gf.slotCount = field->slotCount;
        gf.cellOffset = static_cast<uint32_t>(buffer.gasCells.size() / MAX_GAS_SPECIES);
        for (int s = 0; s < MAX_GAS_SPECIES; ++s) {
            uint16_t g = field->slotToGlobal[s];
            // Store the global *material* index (offset folded in) or 0xFFFFFFFF.
            gf.slotToGlobal[s] = (g == GasField_<T, IndexType>::INVALID_SLOT)
                                     ? 0xFFFFFFFFu
                                     : (buffer.gasMaterialOffset + g);
        }

        // Flatten cell densities (MAX_GAS_SPECIES floats per cell, row-major).
        size_t base = buffer.gasCells.size();
        buffer.gasCells.resize(base + cellCount * MAX_GAS_SPECIES);
        for (size_t i = 0; i < cellCount; ++i) {
            const auto& c = field->cells[i];
            float* dst = &buffer.gasCells[base + i * MAX_GAS_SPECIES];
            for (int s = 0; s < MAX_GAS_SPECIES; ++s) dst[s] = c.amount[s];
        }

        buffer.gasFields.push_back(gf);
    }
    rnode.pointCount = static_cast<uint32_t>(buffer.points.size() - rnode.firstPoint);
    
    rnode.lodPoint = -1;
    if (node->lodData) {
        RenderData_ ld;
        ld.position = node->lodData->position;
        ld.size = node->lodData->size;
        ld.color = node->lodData->color;
        
        uint32_t isGas = 0;
        auto objIt = localObjects.find(node->lodData->objectId);
        if (objIt != localObjects.end()) {
            if (objIt->second->getPhysicsMaterial(node->lodData->physMatIdx).type == BodyType::GAS) {
                isGas = 1;
            }
        }
        
        ld.materialIdx = buffer.defaultMatIdx;
        auto it = buffer.objMaterialOffsets.find(node->lodData->objectId);
        if (it != buffer.objMaterialOffsets.end()) {
            ld.materialIdx = it->second + node->lodData->renderMatIdx;
        }
        ld.isGas = isGas;
        
        BoundingBox bb = node->lodData->getCubeBounds();
        ld.boundsMin = bb.first;
        ld.boundsMax = bb.second;
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

template<typename T, typename IndexType>
std::vector<RenderData_*> Octree<T, IndexType>::fastVoxelTraverse(const RenderBuffer_<T, IndexType>& buffer, const Ray& ray, float maxDist) {
    std::vector<RenderData_*> hits;
    if (buffer.nodes.empty()) return hits;
    std::vector<float> tv;
    float tMin, tMax;
    BoundingBox rootBounds(buffer.nodes[0].boundsMin, buffer.nodes[0].boundsMax);
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
            
            const RenderNode_<T, IndexType>& node = buffer.nodes[current.nodeIdx];

            if (!node.isLoaded && node.originalNode) {
                ensureLoaded(node.originalNode, true);
            }

            if (!node.isLeaf && node.lodPoint != -1) {
                float dist = (node.center - ray.origin).norm();
                if (dist > lodMinDistance_ && (dist / node.nodeSize) > invLodf) {
                    float t;
                    PointType n, h;
                    if (rayCubeIntersect(ray, &buffer.points[node.lodPoint], t, n, h)) {
                        if (t >= 0 && t <= currentMaxDist) {
                            hits.emplace_back(const_cast<RenderData_*>(&buffer.points[node.lodPoint]));
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
                const RenderData_& pt = buffer.points[node.firstPoint + i];
                float t;
                PointType n, h;
                if (rayCubeIntersect(ray, &pt, t, n, h)) {
                    if (t >= 0 && t <= currentMaxDist) {
                        hits.emplace_back(const_cast<RenderData_*>(&pt));
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

            Eigen::Vector3f ttt = (node.center - ray.origin).cwiseProduct(ray.invDir);
            int currIdx = ((t0 >= ttt.x()) ? 1 : 0) | ((t0 >= ttt.y()) ? 2 : 0) | ((t0 >= ttt.z()) ? 4 : 0);
            
            struct ChildInterval {
                uint32_t nodeIdx;
                float tMin;
                float tMax;
            };
            ChildInterval children[4];
            int childCount = 0;

            while(t0 < t1 && t0 <= currentMaxDist) {
                Eigen::Vector3f next_t;
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
    std::vector<std::pair<RenderData_*, float>> zipped;
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

template<typename T, typename IndexType>
frame Octree<T, IndexType>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
    updateStreaming(cam);
    
    thread_local RenderBuffer_<T, IndexType> tl_buffer;
    buildRender(tl_buffer);
    const RenderBuffer_<T, IndexType>& shared_buffer = tl_buffer;

    PointType origin = cam.origin;
    PointType dir = cam.direction.normalized();
    PointType up = cam.up.normalized();
    PointType right = cam.right();
    
    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer;
    colorBuffer.resize(width * height * 3);

    const float aspect = static_cast<float>(width) / height;
    const float fovRad = cam.fovRad();
    const float tanHalfFov = tan(fovRad * 0.5f);
    const float tanfovy = tanHalfFov;
    const float tanfovx = tanHalfFov * aspect;
    
    const PointType globalLightDir = (-cam.direction * 0.2f).normalized();
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
            
            PointType rayDir = dir + (right * px) + (up * py);
            rayDir.normalize();

            Eigen::Vector3f scolor = skybox_.sampleVector(rayDir);
            Eigen::Vector3f color = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            float accumAlpha = 0.0f;
            Ray ray(origin, rayDir);
            
            // const RenderData_* hit = nullptr;
            std::vector<RenderData_*> hits = fastVoxelTraverse(shared_buffer, ray, maxDistance_);

            int prevOid = -1;
            bool hasPrev = false;

            for (auto hit : hits) {
                if (accumAlpha >= 0.99f) break;
                if (hit == nullptr) continue;
                float t = 0.0f;
                float tExit = 0.0f;
                PointType normal, hitPoint;

                rayCubeIntersect(ray, hit, t, normal, hitPoint, &tExit);
                Eigen::Vector3f hitColor = hit->color.template head<3>();
                float alpha = hit->color.w();
                Material_ objMat = shared_buffer.materials[hit->materialIdx];

                float coverage;
                bool isInternalVoxel = (hasPrev && hit->objectId == prevOid);
                if (isInternalVoxel) {
                    float thickness = tExit - t;
                    float trPerUnit = std::clamp(1.0f - alpha, 0.0f, 1.0f);
                    coverage = 1.0f - std::pow(trPerUnit, thickness);
                } else {
                    coverage = alpha;
                }
                
                if (objMat.emittance != 0u) {
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

static inline uint32_t packRGB8(const Eigen::Vector3f& c) {
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

static inline std::vector<float> buildSellmeierLUT(const std::vector<Grid::Material_>& mats) {
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

template<typename T, typename IndexType>
void Octree<T, IndexType>::buildGPUMaterials(const RenderBuffer_<T, IndexType>& buf, std::vector<GPUMaterial>& out) {
    out.clear();
    out.reserve(buf.materials.size());
    size_t gasCount = gasRegistry_.size();
    for (size_t mi = 0; mi < buf.materials.size(); ++mi) {
        const auto& m = buf.materials[mi];
        uint32_t sellRow = static_cast<uint32_t>(mi) * SELL_LUT_SECONDARY;
        uint32_t albedoPacked = 0;
        if (gasCount > 0 && mi >= buf.gasMaterialOffset && mi < buf.gasMaterialOffset + gasCount) {
            GasSpecies_ sp = gasRegistry_.get(static_cast<uint16_t>(mi - buf.gasMaterialOffset));
            albedoPacked = packRGB8(sp.albedo);
        }
        out.push_back({
            m.emittance,
            packMaterialProps(m.roughness, m.metallic, sellRow),
            packRGB8(m.absorption),
            albedoPacked
        });
    }
    if (out.empty()) out.push_back(GPUMaterial{});
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();

    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    vkCtx.updateMaterialBuffer(gpuMaterials);
    vkCtx.updateSellmeierBuffer(buildSellmeierLUT(tl_buffer.materials),
                                SELL_LUT_WAVELENGTHS,
                                std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Eigen::Vector3f globalMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f globalMax = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    
    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }
    
    Eigen::Vector3f extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Eigen::Vector3f invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for(size_t idx : validIndices) {
        Eigen::Vector3f normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
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
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.isGas
        });

        if (tl_buffer.materials[p.materialIdx].emittance != 0u) {
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

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(outSize, camData);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPoints);
    vkCtx.updateGasBuffers(tl_buffer.gasFields, tl_buffer.gasCells);
    camData.gasFieldCount = vkCtx.getGasFieldCount();
    {
        int tileW = 512;
        int tileH = 512;

        for (int y = 0; y < height; y += tileH) {
            for (int x = 0; x < width; x += tileW) {
                int drawW = std::min(tileW, width - x);
                int drawH = std::min(tileH, height - y);

                camData.tileOffsetX = x;
                camData.tileOffsetY = y;
                camData.currentSampleOffset = 0;
                camData.dispatchSamples = samplesPerPixel;

                vkCtx.updateCameraData(camData);
                vkCtx.dispatchWavefront(drawW, drawH, maxBounces, samplesPerPixel);
            }
        }
    }

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

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderFrameLightTracedVulkan(const Camera& cam, int height, int width,
        frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool useLod,
        const std::function<void(int, const frame&)>& onSample) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);

    vkCtx.init();

    std::vector<GPUMaterial> gpuMaterials;
    buildGPUMaterials(tl_buffer, gpuMaterials);
    vkCtx.updateMaterialBuffer(gpuMaterials);
    vkCtx.updateSellmeierBuffer(buildSellmeierLUT(tl_buffer.materials),
                                SELL_LUT_WAVELENGTHS,
                                std::max<size_t>(1, tl_buffer.materials.size()) * SELL_LUT_SECONDARY);

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for (const auto& n : tl_buffer.nodes) {
        if (n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    Eigen::Vector3f globalMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f globalMax = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for (size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if (isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }

    Eigen::Vector3f extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Eigen::Vector3f invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for (size_t idx : validIndices) {
        Eigen::Vector3f normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
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
    for (const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];
        gpuPoints.push_back({ p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.isGas });
        if (tl_buffer.materials[p.materialIdx].emittance != 0u)
            gpuLights.push_back(gpuPoints.size() - 1);
    }
    int emissiveCount = gpuLights.size();
    if (gpuPoints.empty()) gpuPoints.push_back(GPUPBRRenderData{});
    if (gpuLights.empty()) gpuLights.push_back(0);

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
        (int)skyW, (int)skyH, 0, 0, 1,
        0, (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, samplesPerPixel
    };
    // Light-tracing parameters.
    camData.lightTracing = 1.0f;
    // Fixed-point splat scale: large enough to preserve dim splats, small enough
    // to avoid uint overflow when many paths stack on one pixel.
    camData.ltSplatScale = 1024.0f;
    camData.ltPad0       = 0.0f;
    camData.ltPad1       = 0.0f;

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(outSize, camData);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPoints);
    vkCtx.updateGasBuffers(tl_buffer.gasFields, tl_buffer.gasCells);
    camData.gasFieldCount = vkCtx.getGasFieldCount();

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    std::vector<float> rawPixels(width * height * 5);

    for (int s = 0; s < samplesPerPixel; ++s) {
        camData.currentSampleOffset = 0;
        camData.dispatchSamples = 1;
        vkCtx.updateCameraData(camData);
        vkCtx.dispatchLightTraceSample(width, height, maxBounces, s);

        // Read back accumulated pixels[] and normalize by sample count so the
        // progressive image is correctly exposed at every step.
        void* mapped;
        vkMapMemory(vkCtx.device, vkCtx.outMem, 0, rawPixels.size() * sizeof(float), 0, &mapped);
        memcpy(rawPixels.data(), mapped, rawPixels.size() * sizeof(float));
        vkUnmapMemory(vkCtx.device, vkCtx.outMem);

        float invN = 1.0f / float(s + 1);
        for (int i = 0; i < width * height; ++i) {
            colorBuffer[i * 3 + 0] = rawPixels[i * 5 + 0] * invN;
            colorBuffer[i * 3 + 1] = rawPixels[i * 5 + 1] * invN;
            colorBuffer[i * 3 + 2] = rawPixels[i * 5 + 2] * invN;
        }
        {
            size_t p1 = (size_t(width) * size_t(height) - 1) * 5;
            size_t p2 = (size_t(width) * size_t(height) - 2) * 5;
            size_t p3 = (size_t(width) * size_t(height) - 3) * 5;
            long seeded      = (long)rawPixels[p1 + 3];
            long shadeEntries= (long)rawPixels[p1 + 4];
            long enteringGate= (long)rawPixels[p2 + 3];
            long connections = (long)rawPixels[p2 + 4];
            long survivors   = (long)rawPixels[p3 + 3];
            std::cout << "[lt] s" << s
                      << " seeded=" << seeded
                      << " shade=" << shadeEntries
                      << " entering=" << enteringGate
                      << " connect=" << connections
                      << " survive=" << survivors << std::endl;
        }
        outFrame.setData(colorBuffer, frame::colormap::RGB);
        if (onSample) onSample(s, outFrame);
    }

    frameCounter_++;
    return outFrame;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
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
    Eigen::Vector3f vctMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f vctMax = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        const auto& p = tl_buffer.points[i];
        
        gpuPoints.push_back({p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.isGas});
        
        if (tl_buffer.materials[p.materialIdx].emittance != 0u) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }

        float h = p.size * 0.5f;
        vctMin = vctMin.cwiseMin(p.position - Eigen::Vector3f::Constant(h));
        vctMax = vctMax.cwiseMax(p.position + Eigen::Vector3f::Constant(h));
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
        Eigen::Vector3f keyLight = (-cam.direction.normalized());
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

template<typename T, typename IndexType>
frame Octree<T, IndexType>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
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

    Eigen::Vector3f globalMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f globalMax = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    
    std::vector<size_t> validIndices;
    validIndices.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        validIndices.push_back(i);
        globalMin = globalMin.cwiseMin(tl_buffer.points[i].position);
        globalMax = globalMax.cwiseMax(tl_buffer.points[i].position);
    }
    
    Eigen::Vector3f extent = globalMax - globalMin;
    if (extent.x() <= 0.0f) extent.x() = 1.0f;
    if (extent.y() <= 0.0f) extent.y() = 1.0f;
    if (extent.z() <= 0.0f) extent.z() = 1.0f;
    Eigen::Vector3f invExtent = extent.cwiseInverse();

    std::vector<PointSort> sortedPoints;
    sortedPoints.reserve(validIndices.size());
    for(size_t idx : validIndices) {
        Eigen::Vector3f normPos = (tl_buffer.points[idx].position - globalMin).cwiseProduct(invExtent);
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
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.isGas
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGBA8(p.color), p.materialIdx, p.objectId, p.isGas
        });

        if (tl_buffer.materials[p.materialIdx].emittance != 0u) {
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
    vkCtx.updateCommonBuffers(pbrOutSize, pbrCamData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPBRPoints);
    vkCtx.updateGasBuffers(tl_buffer.gasFields, tl_buffer.gasCells);
    pbrCamData.gasFieldCount = vkCtx.getGasFieldCount();

    {
        int tileW = 512;
        int tileH = 512;
        for (int y = 0; y < lowH; y += tileH) {
            for (int x = 0; x < lowW; x += tileW) {
                int drawW = std::min(tileW, lowW - x);
                int drawH = std::min(tileH, lowH - y);
                pbrCamData.tileOffsetX = x;
                pbrCamData.tileOffsetY = y;
                pbrCamData.currentSampleOffset = 0;
                pbrCamData.dispatchSamples = samplesPerPixel;
                vkCtx.updateCameraData(pbrCamData);
                vkCtx.dispatchWavefront(drawW, drawH, maxBounces, samplesPerPixel);
            }
        }
    }

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
        Eigen::Vector3f keyLight = (-cam.direction.normalized());
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
    
    vkCtx.dispatchBlend(width, height, lowW, lowH, pbrScale, samplesPerPixel);
    
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

#endif
}