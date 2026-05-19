#include "grid3eigen.hpp"
namespace Grid {

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
void Octree<T, GasT, IndexType, StoragePath>::buildRender(RenderBuffer_<T, IndexType, StoragePath>& buffer) {
    buffer.clear();
    if (!root_) return;
    buffer.nodes.emplace_back();

    std::unordered_map<int, std::shared_ptr<GridObject_<T, IndexType, StoragePath>>> localObjects;
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
    buffer.materials.push_back(Material_<T, IndexType, StoragePath>());

    buildRenderNodeAt(root_.get(), buffer, 0, localObjects);
}

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
void Octree<T, GasT, IndexType, StoragePath>::buildRenderNodeAt(OctreeNode_<T, GasT, IndexType, StoragePath>* node, RenderBuffer_<T, IndexType, StoragePath>& buffer, uint32_t nodeIdx, const std::unordered_map<int, std::shared_ptr<GridObject>>& localObjects) {
    std::shared_lock<std::shared_mutex> lock(node->nodeMutex);
    bool isLoaded = node->isLoaded();
    
    RenderNode_<T, IndexType, StoragePath> rnode;
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
            RenderData_<T, IndexType, StoragePath> rd;
            rd.position = pt->position;
            rd.size = pt->size;
            rd.color = pt->color;
            
            float gasDensity = 0.0f;
            // auto objIt = localObjects.find(pt->objectId);
            // if (objIt != localObjects.end()) {
            //     if (objIt->second->getPhysicsMaterial(pt->physMatIdx).type == BodyType::GAS) {
            //         gasDensity = 1.0f;
            //     }
            // }
            
            rd.materialIdx = buffer.defaultMatIdx;
            auto it = buffer.objMaterialOffsets.find(pt->objectId);
            if (it != buffer.objMaterialOffsets.end()) {
                rd.materialIdx = it->second + pt->renderMatIdx;
            }
            rd.gasDensity = gasDensity;
            
            BoundingBox bb = pt->getCubeBounds();
            rd.boundsMin = bb.first;
            rd.boundsMax = bb.second;
            rd.objectId = pt->objectId;
            buffer.points.push_back(rd);
        }

        if (node->isLeaf() && node->gasState.density > 0.001f) {
            RenderData_<T, IndexType, StoragePath> gd;
            gd.position = node->center;
            gd.size = node->nodeSize;
            gd.color = node->gasState.color;
            
            gd.materialIdx = buffer.defaultMatIdx;
            auto it = buffer.objMaterialOffsets.find(node->gasState.objectId);
            if (it != buffer.objMaterialOffsets.end()) {
                gd.materialIdx = it->second + node->gasState.renderMatIdx;
            }
            gd.gasDensity = node->gasState.density;
            
            PointType halfSize = PointType(node->nodeSize * 0.5f, node->nodeSize * 0.5f, node->nodeSize * 0.5f);
            gd.boundsMin = node->center - halfSize;
            gd.boundsMax = node->center + halfSize;
            gd.objectId = node->gasState.objectId;
            buffer.points.push_back(gd);
        }
    }
    rnode.pointCount = static_cast<uint32_t>(buffer.points.size() - rnode.firstPoint);
    
    rnode.lodPoint = -1;
    if (node->lodData) {
        RenderData_<T, IndexType, StoragePath> ld;
        ld.position = node->lodData->position;
        ld.size = node->lodData->size;
        ld.color = node->lodData->color;
        
        float gasDensity = 0.0f;
        // auto objIt = localObjects.find(node->lodData->objectId);
        // if (objIt != localObjects.end()) {
        //     if (objIt->second->getPhysicsMaterial(node->lodData->physMatIdx).type == BodyType::GAS) {
        //         gasDensity = 1.0f;
        //     }
        // }
        
        ld.materialIdx = buffer.defaultMatIdx;
        auto it = buffer.objMaterialOffsets.find(node->lodData->objectId);
        if (it != buffer.objMaterialOffsets.end()) {
            ld.materialIdx = it->second + node->lodData->renderMatIdx;
        }
        ld.gasDensity = gasDensity;
        
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

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
const RenderData_<T, IndexType, StoragePath>* Octree<T, GasT, IndexType, StoragePath>::fastVoxelTraverse(const RenderBuffer_<T, IndexType, StoragePath>& buffer, const Ray& ray, float maxDist) {
    const RenderData_<T, IndexType, StoragePath>* hit = nullptr;
    if (buffer.nodes.empty()) return hit;
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
            
            const RenderNode_<T, IndexType, StoragePath>& node = buffer.nodes[current.nodeIdx];

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
                            hit = &buffer.points[node.lodPoint];
                            currentMaxDist = t;
                        }
                    }
                    continue;
                }
            }

            for (uint32_t i = 0; i < node.pointCount; ++i) {
                const RenderData_<T, IndexType, StoragePath>& pt = buffer.points[node.firstPoint + i];
                float t;
                PointType n, h;
                if (rayCubeIntersect(ray, &pt, t, n, h)) {
                    if (t >= 0 && t <= currentMaxDist && t <= current.tMax + 0.001f) {
                        currentMaxDist = t;
                        hit = &pt;
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
    return hit;
}

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, GasT, IndexType, StoragePath>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
    updateStreaming(cam);
    
    thread_local RenderBuffer_<T, IndexType, StoragePath> tl_buffer;
    buildRender(tl_buffer);
    const RenderBuffer_<T, IndexType, StoragePath>& shared_buffer = tl_buffer;

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

            Eigen::Vector3f color = skybox_.sampleVector(rayDir);
            Ray ray(origin, rayDir);
            
            const RenderData_<T, IndexType, StoragePath>* hit = nullptr;
            if (x % 10 == 0 && y % 10 == 0) hit = fastVoxelTraverse(shared_buffer, ray, maxDistance_);
            else hit = fastVoxelTraverse(shared_buffer, ray, maxDistance_);
            
            if (hit != nullptr) {
                float t = 0.0f;
                PointType normal, hitPoint;

                rayCubeIntersect(ray, hit, t, normal, hitPoint);
                color = hit->color;
                Material_<T, IndexType, StoragePath> objMat = shared_buffer.materials[hit->materialIdx];
                
                if (objMat.emittance > 0.0f) {
                    color = color * objMat.emittance;
                } else {
                    float diffuse = std::max(0.0f, normal.dot(globalLightDir));
                    float ambient = 0.35f;
                    float intensity = std::min(1.0f, ambient + diffuse * 0.65f);
                    color = color * intensity;
                }
                
                float fogFactor = std::clamp((maxDistance_ - t) * invMaxMin, minVisibility, 1.0f);
                
                color = color * fogFactor + skybox_.sampleVector(rayDir) * (1.0f - fogFactor);
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

static inline uint32_t packMaterialProps(float roughness, float metallic, float transmission, float ior) {
    uint32_t r8 = static_cast<uint32_t>(std::clamp(roughness, 0.0f, 1.0f) * 255.0f);
    uint32_t m8 = static_cast<uint32_t>(std::clamp(metallic, 0.0f, 1.0f) * 255.0f);
    uint32_t t8 = static_cast<uint32_t>(std::clamp(transmission, 0.0f, 1.0f) * 255.0f);
    float mappedIor = (std::clamp(ior, 1.0f, 2.5f) - 1.0f) / 1.5f;
    uint32_t i8 = static_cast<uint32_t>(std::clamp(mappedIor, 0.0f, 1.0f) * 255.0f);
    return r8 | (m8 << 8) | (t8 << 16) | (i8 << 24);
}

struct PointSort {
    uint64_t morton;
    size_t idx;
    bool operator<(const PointSort& o) const { return morton < o.morton; }
};

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, GasT, IndexType, StoragePath>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();

    std::vector<GPUMaterial> gpuMaterials;
    gpuMaterials.reserve(tl_buffer.materials.size());
    for (const auto& m : tl_buffer.materials) {
        gpuMaterials.push_back({
            m.emittance,
            packMaterialProps(m.roughness, m.metallic, m.transmission, m.ior),
            packRGB8(m.absorption),
            0
        });
    }
    if (gpuMaterials.empty()) gpuMaterials.push_back(GPUMaterial{});
    vkCtx.updateMaterialBuffer(gpuMaterials);

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
            p.position, p.size, packRGB8(p.color), p.materialIdx, p.objectId, p.gasDensity
        });

        if (tl_buffer.materials[p.materialIdx].emittance > 0.0f) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuPoints.empty()) gpuPoints.push_back(GPUPBRRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for (const auto& n : tl_buffer.nodes) {
        GPURenderNode gn;
        gn.boundsMin = n.boundsMin;
        gn.padding1 = 0.0f;
        gn.boundsMax = n.boundsMax;
        gn.padding2 = 0.0f;
        gn.center = n.center;
        gn.nodeSize = n.nodeSize;
        gn.isLeaf = n.isLeaf ? 1 : 0;
        gn.isLoaded = n.isLoaded ? 1 : 0;
        gn.childMask = n.childMask;
        gn.firstPoint = n.firstPoint;
        gn.pointCount = n.pointCount;
        gn.lodPoint = n.lodPoint;
        gn.firstChild = n.firstChild;
        gn.padding3 = 0;
        gpuNodes.push_back(gn);
    }
    if (gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    size_t skyW = skybox_.skybox.getWidth();
    size_t skyH = skybox_.skybox.getHeight();
    if (skyW == 0 || skyH == 0) { skyW = 1; skyH = 1; }
    std::vector<Eigen::Vector4f> skyData(skyW * skyH, Eigen::Vector4f(0,0,0,1));
    if (skybox_.skybox.getWidth() > 0) {
        for (size_t y = 0; y < skyH; ++y) {
            float v = (static_cast<float>(y) + 0.5f) / skyH;
            for (size_t x = 0; x < skyW; ++x) {
                float u = (static_cast<float>(x) + 0.5f) / skyW;
                PointType skyDir = skybox_.uvToDir(u, v);
                Eigen::Vector3f color = skybox_.sampleVector(skyDir);
                skyData[y * skyW + x] = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
            }
        }
    }

    GPUCameraData camData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, samplesPerPixel
    };

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(outSize, camData, gpuNodes);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPoints);

    int currentSampleOffset = 0;
    
    const long long maxWorkloadBudget = 4194304; 
    const long long pixelsInFrame = (long long)width * height;

    while (currentSampleOffset < samplesPerPixel) {
        int samplesInBatch = std::max(1, (int)(maxWorkloadBudget / pixelsInFrame));
        samplesInBatch = std::min(samplesInBatch, samplesPerPixel - currentSampleOffset);
        
        camData.currentSampleOffset = currentSampleOffset;
        camData.dispatchSamples = samplesInBatch;

        int tileW = 512;
        int tileH = 512;

        for (int y = 0; y < height; y += tileH) {
            for (int x = 0; x < width; x += tileW) {
                int drawW = std::min(tileW, width - x);
                int drawH = std::min(tileH, height - y);

                camData.tileOffsetX = x;
                camData.tileOffsetY = y;
                
                // Update camera for this specific tile/sample batch
                vkCtx.updateCameraData(camData);

                VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
                vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.pbrPipeline);
                vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.pbrPipelineLayout, 0, 1, &vkCtx.pbrDescSet, 0, nullptr);
                
                // Dispatch only for the tile size
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
        
        currentSampleOffset += samplesInBatch;
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

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, GasT, IndexType, StoragePath>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
    updateStreaming(cam);
    // optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();
    
    std::vector<GPUMaterial> gpuMaterials;
    gpuMaterials.reserve(tl_buffer.materials.size());
    for (const auto& m : tl_buffer.materials) {
        gpuMaterials.push_back({
            m.emittance,
            packMaterialProps(m.roughness, m.metallic, m.transmission, m.ior),
            packRGB8(m.absorption),
            0
        });
    }
    if (gpuMaterials.empty()) gpuMaterials.push_back(GPUMaterial{});
    vkCtx.updateMaterialBuffer(gpuMaterials);

    std::vector<GPUFastRenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    
    gpuPoints.reserve(tl_buffer.points.size());
    for (size_t i = 0; i < tl_buffer.points.size(); ++i) {
        const auto& pt = tl_buffer.points[i];
        GPUFastRenderData data;
        data.position = pt.position;
        data.size = pt.size;
        data.color = packRGB8(pt.color);
        data.materialIdx = pt.materialIdx;
        data.objectId = pt.objectId;
        data.gasDensity = pt.gasDensity;
        gpuPoints.push_back(data);
        
        if (tl_buffer.materials[pt.materialIdx].emittance > 0.0f) {
            gpuLights.push_back(i);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuPoints.empty()) gpuPoints.push_back(GPUFastRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for (const auto& n : tl_buffer.nodes) {
        GPURenderNode gn;
        gn.boundsMin = n.boundsMin;
        gn.padding1 = 0.0f;
        gn.boundsMax = n.boundsMax;
        gn.padding2 = 0.0f;
        gn.center = n.center;
        gn.nodeSize = n.nodeSize;
        gn.isLeaf = n.isLeaf ? 1 : 0;
        gn.isLoaded = n.isLoaded ? 1 : 0;
        gn.childMask = n.childMask;
        gn.firstPoint = n.firstPoint;
        gn.pointCount = n.pointCount;
        gn.lodPoint = n.lodPoint;
        gn.firstChild = n.firstChild;
        gn.padding3 = 0;
        gpuNodes.push_back(gn);
    }
    if (gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});

    size_t skyW = skybox_.skybox.getWidth();
    size_t skyH = skybox_.skybox.getHeight();
    if (skyW == 0 || skyH == 0) { skyW = 1; skyH = 1; }
    std::vector<Eigen::Vector4f> skyData(skyW * skyH, Eigen::Vector4f(0,0,0,1));
    if (skybox_.skybox.getWidth() > 0) {
        for (size_t y = 0; y < skyH; ++y) {
            float v = (static_cast<float>(y) + 0.5f) / skyH;
            for (size_t x = 0; x < skyW; ++x) {
                float u = (static_cast<float>(x) + 0.5f) / skyW;
                PointType skyDir = skybox_.uvToDir(u, v);
                Eigen::Vector3f color = skybox_.sampleVector(skyDir);
                skyData[y * skyW + x] = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
            }
        }
    }

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);

    GPUCameraData camData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, 1, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, 1
    };

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(outSize, camData, gpuNodes);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updateFastBuffers(gpuPoints);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
    vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipeline);
    vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipelineLayout, 0, 1, &vkCtx.fastDescSet, 0, nullptr);
    
    vkCmdDispatch(vkCtx.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
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

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    
    std::vector<float> rawBuffer(width * height * 5);
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.outMem, 0, outSize, 0, &mappedData);
    memcpy(rawBuffer.data(), mappedData, outSize);
    vkUnmapMemory(vkCtx.device, vkCtx.outMem);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int outIdx = (y * width + x) * 3;
            int inIdx = (y * width + x) * 5;
            colorBuffer[outIdx]     = std::clamp(rawBuffer[inIdx], 0.0f, 1.0f);
            colorBuffer[outIdx + 1] = std::clamp(rawBuffer[inIdx + 1], 0.0f, 1.0f);
            colorBuffer[outIdx + 2] = std::clamp(rawBuffer[inIdx + 2], 0.0f, 1.0f);
        }
    }

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T, typename GasT, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, GasT, IndexType, StoragePath>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();
    
    std::vector<GPUMaterial> gpuMaterials;
    gpuMaterials.reserve(tl_buffer.materials.size());
    for (const auto& m : tl_buffer.materials) {
        gpuMaterials.push_back({
            m.emittance,
            packMaterialProps(m.roughness, m.metallic, m.transmission, m.ior),
            packRGB8(m.absorption),
            0
        });
    }
    if (gpuMaterials.empty()) gpuMaterials.push_back(GPUMaterial{});
    vkCtx.updateMaterialBuffer(gpuMaterials);

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
    std::vector<uint32_t> pbrLights;
    gpuPBRPoints.reserve(sortedPoints.size());

    for(const auto& sp : sortedPoints) {
        const auto& p = tl_buffer.points[sp.idx];
        
        gpuPBRPoints.push_back({
            p.position, p.size, packRGB8(p.color), p.materialIdx, p.objectId, p.gasDensity
        });

        if (tl_buffer.materials[p.materialIdx].emittance > 0.0f) {
            pbrLights.push_back(gpuPBRPoints.size() - 1);
        }
    }

    std::vector<GPUFastRenderData> gpuFastPoints;
    gpuFastPoints.reserve(tl_buffer.points.size());
    std::vector<uint32_t> fastLights;

    for (size_t i = 0; i < tl_buffer.points.size(); ++i) {
        const auto& pt = tl_buffer.points[i];
        GPUFastRenderData data;
        data.position = pt.position;
        data.size = pt.size;
        data.color = packRGB8(pt.color);
        data.materialIdx = pt.materialIdx;
        data.objectId = pt.objectId;
        data.gasDensity = pt.gasDensity;
        gpuFastPoints.push_back(data);

        if (tl_buffer.materials[pt.materialIdx].emittance > 0.0f) {
            fastLights.push_back(i);
        }
    }

    int fastEmissiveCount = fastLights.size();
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPUFastRenderData{});
    if(fastLights.empty()) fastLights.push_back(0);

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for (const auto& n : tl_buffer.nodes) {
        GPURenderNode gn;
        gn.boundsMin = n.boundsMin;
        gn.padding1 = 0.0f;
        gn.boundsMax = n.boundsMax;
        gn.padding2 = 0.0f;
        gn.center = n.center;
        gn.nodeSize = n.nodeSize;
        gn.isLeaf = n.isLeaf ? 1 : 0;
        gn.isLoaded = n.isLoaded ? 1 : 0;
        gn.childMask = n.childMask;
        gn.firstPoint = n.firstPoint;
        gn.pointCount = n.pointCount;
        gn.lodPoint = n.lodPoint;
        gn.firstChild = n.firstChild;
        gn.padding3 = 0;
        gpuNodes.push_back(gn);
    }
    if (gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});

    int pbrEmissiveCount = pbrLights.size();
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPUPBRRenderData{});
    if(pbrLights.empty()) pbrLights.push_back(0);

    size_t skyW = skybox_.skybox.getWidth();
    size_t skyH = skybox_.skybox.getHeight();
    if (skyW == 0 || skyH == 0) { skyW = 1; skyH = 1; }
    std::vector<Eigen::Vector4f> skyData(skyW * skyH, Eigen::Vector4f(0,0,0,1));
    if (skybox_.skybox.getWidth() > 0) {
        for (size_t y = 0; y < skyH; ++y) {
            float v = (static_cast<float>(y) + 0.5f) / skyH;
            for (size_t x = 0; x < skyW; ++x) {
                float u = (static_cast<float>(x) + 0.5f) / skyW;
                PointType skyDir = skybox_.uvToDir(u, v);
                Eigen::Vector3f color = skybox_.sampleVector(skyDir);
                skyData[y * skyW + x] = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
            }
        }
    }

    float aspect = static_cast<float>(width) / height;
    float fovRad = cam.fovRad();
    float tanHalfFov = tan(fovRad * 0.5f);
    float invFogRange = 1.0f / std::max(0.001f, maxDistance_ - lodMinDistance_);
    int lowW = std::max(1, static_cast<int>(width * pbrScale));
    int lowH = std::max(1, static_cast<int>(height * pbrScale));

    GPUCameraData pbrCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        lowW, lowH, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPBRPoints.size(), 0, 0, pbrEmissiveCount, samplesPerPixel
    };

    size_t pbrOutSize = lowW * lowH * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(pbrOutSize, pbrCamData, gpuNodes);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(pbrLights);
    vkCtx.updatePBRBuffers(gpuPBRPoints);

    int currentSampleOffset = 0;
    const long long maxWorkloadBudget = 4194304; 
    const long long pixelsInFrame = (long long)lowW * lowH;

    while (currentSampleOffset < samplesPerPixel) {
        int samplesInBatch = std::max(1, (int)(maxWorkloadBudget / pixelsInFrame));
        samplesInBatch = std::min(samplesInBatch, samplesPerPixel - currentSampleOffset);
        pbrCamData.currentSampleOffset = currentSampleOffset;
        pbrCamData.dispatchSamples = samplesInBatch;

        int tileW = 512;
        int tileH = 512;
        for (int y = 0; y < lowH; y += tileH) {
            for (int x = 0; x < lowW; x += tileW) {
                int drawW = std::min(tileW, lowW - x);
                int drawH = std::min(tileH, lowH - y);
                pbrCamData.tileOffsetX = x;
                pbrCamData.tileOffsetY = y;
                vkCtx.updateCameraData(pbrCamData);

                VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
                vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.pbrPipeline);
                vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.pbrPipelineLayout, 0, 1, &vkCtx.pbrDescSet, 0, nullptr);
                
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
        currentSampleOffset += samplesInBatch;
    }

    vkCtx.ensureLowResBuffer(pbrOutSize);
    vkCtx.copyBuffer(vkCtx.outBuffer, vkCtx.lowResOutBuffer, pbrOutSize);

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuFastPoints.size(), 0, 0, fastEmissiveCount, 1
    };

    size_t fastOutSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(fastOutSize, fastCamData, gpuNodes);
    vkCtx.updateLightBuffer(fastLights);
    vkCtx.updateFastBuffers(gpuFastPoints);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(vkCtx.commandBuffer, &beginInfo);
    vkCmdBindPipeline(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipeline);
    vkCmdBindDescriptorSets(vkCtx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkCtx.fastPipelineLayout, 0, 1, &vkCtx.fastDescSet, 0, nullptr);
    vkCmdDispatch(vkCtx.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
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

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    
    vkCtx.dispatchBlend(width, height, lowW, lowH, pbrScale, samplesPerPixel);
    // vkCtx.dispatchSmooth(width, height, samplesPerPixel);
    
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.finalOutMem, 0, colorBuffer.size() * sizeof(float), 0, &mappedData);
    memcpy(colorBuffer.data(), mappedData, colorBuffer.size() * sizeof(float));
    vkUnmapMemory(vkCtx.device, vkCtx.finalOutMem);

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

#endif
}