#include "grid3eigen.hpp"
namespace Grid {

template<typename T, typename IndexType, GridStoragePath StoragePath>
void Octree<T, IndexType, StoragePath>::buildRender(RenderBuffer_<T, IndexType, StoragePath>& buffer) {
    buffer.clear();
    if (!root_) return;
    buffer.nodes.emplace_back();
    buildRenderNodeAt(root_.get(), buffer, 0);
}

template<typename T, typename IndexType, GridStoragePath StoragePath>
void Octree<T, IndexType, StoragePath>::buildRenderNodeAt(OctreeNode_<T, IndexType, StoragePath>* node, RenderBuffer_<T, IndexType, StoragePath>& buffer, uint32_t nodeIdx) {
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
            rd.material = pt->material;
            BoundingBox bb = pt->getCubeBounds();
            rd.boundsMin = bb.first;
            rd.boundsMax = bb.second;
            rd.objectId = pt->objectId;
            buffer.points.push_back(rd);
        }
    }
    rnode.pointCount = static_cast<uint32_t>(buffer.points.size() - rnode.firstPoint);
    
    rnode.lodPoint = -1;
    if (node->lodData) {
        RenderData_<T, IndexType, StoragePath> ld;
        ld.position = node->lodData->position;
        ld.size = node->lodData->size;
        ld.color = node->lodData->color;
        ld.material = node->lodData->material;
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
                    buildRenderNodeAt(node->children[i].get(), buffer, rnode.firstChild + cidx);
                    cidx++;
                }
            }
        }
    }
    
    buffer.nodes[nodeIdx] = rnode;
}

template<typename T, typename IndexType, GridStoragePath StoragePath>
const RenderData_<T, IndexType, StoragePath>* Octree<T, IndexType, StoragePath>::fastVoxelTraverse(const RenderBuffer_<T, IndexType, StoragePath>& buffer, const Ray& ray, float maxDist) {
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

template<typename T, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, IndexType, StoragePath>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
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
                Material_<T, IndexType, StoragePath> objMat = hit->material;
                
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

template<typename T, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, IndexType, StoragePath>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for(const auto& n : tl_buffer.nodes) {
        gpuNodes.push_back({n.boundsMin, 0, n.boundsMax, 0, n.center, n.nodeSize, (uint32_t)n.isLeaf,
            (uint32_t)n.isLoaded, n.childMask, n.firstPoint, n.pointCount, n.lodPoint, n.firstChild, 0});
    }

    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    std::vector<GPUPBRRenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    gpuPoints.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        const auto& p = tl_buffer.points[i];
        
        gpuPoints.push_back({
            p.position, p.size, packRGB8(p.color), p.material.emittance, 
            packMaterialProps(p.material.roughness, p.material.metallic, p.material.transmission, p.material.ior),
            packRGB8(p.material.absorption), p.objectId, 0, 0, 0
        });

        if (p.material.emittance > 0.0f) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});
    if(gpuPoints.empty()) gpuPoints.push_back(GPUPBRRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

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
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, 0
    };


    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(gpuNodes, outSize, camData);
    vkCtx.updateSkyboxBuffer(skyData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPoints);

    int currentSampleOffset = 0;
    
    const long long maxWorkloadBudget = 262144; 
    const long long pixelsInFrame = (long long)width * height;

    while (currentSampleOffset < samplesPerPixel) {
        int samplesInBatch = std::max(1, (int)(maxWorkloadBudget / pixelsInFrame));
        samplesInBatch = std::min(samplesInBatch, samplesPerPixel - currentSampleOffset);
        
        camData.currentSampleOffset = currentSampleOffset;
        camData.dispatchSamples = samplesInBatch;

        int tileW = 256;
        int tileH = 256;

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

    std::vector<float> rawBuffer(width * height * 5);
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.outMem, 0, outSize, 0, &mappedData);
    memcpy(rawBuffer.data(), mappedData, outSize);
    vkUnmapMemory(vkCtx.device, vkCtx.outMem);

    for (size_t i = 0; i < rawBuffer.size(); i += 5) {
        rawBuffer[i] /= samplesPerPixel;
        rawBuffer[i+1] /= samplesPerPixel;
        rawBuffer[i+2] /= samplesPerPixel;
        rawBuffer[i+3] /= samplesPerPixel; 
    }

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    
    const int radius = 2;
    const float spatialSigma = 2.0f;
    const float depthSigma = 1.0f;
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sumWeights = 0.0f;
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            
            int centerIdx = (y * width + x) * 5;
            float centerDepth = rawBuffer[centerIdx + 3];
            int centerObj = static_cast<int>(rawBuffer[centerIdx + 4]);

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    int ny = std::clamp(y + dy, 0, height - 1);
                    int nIdx = (ny * width + nx) * 5;
                    
                    float nDepth = rawBuffer[nIdx + 3];
                    int nObj = static_cast<int>(rawBuffer[nIdx + 4]);
                    
                    float spatialDistSq = static_cast<float>(dx*dx + dy*dy);
                    float spatialWeight = std::exp(-spatialDistSq / (2.0f * spatialSigma * spatialSigma));
                    
                    float depthDiff = nDepth - centerDepth;
                    float depthWeight = std::exp(-(depthDiff * depthDiff) / (2.0f * depthSigma * depthSigma));
                    
                    float objWeight = (centerObj == nObj) ? 1.0f : 0.05f; 

                    float weight = spatialWeight * depthWeight * objWeight;
                    sumWeights += weight;
                    sumR += rawBuffer[nIdx] * weight;
                    sumG += rawBuffer[nIdx + 1] * weight;
                    sumB += rawBuffer[nIdx + 2] * weight;
                }
            }
            
            int outIdx = (y * width + x) * 3;
            colorBuffer[outIdx]     = std::clamp(sumR / sumWeights, 0.0f, 1.0f);
            colorBuffer[outIdx + 1] = std::clamp(sumG / sumWeights, 0.0f, 1.0f);
            colorBuffer[outIdx + 2] = std::clamp(sumB / sumWeights, 0.0f, 1.0f);
        }
    }

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

template<typename T, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, IndexType, StoragePath>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    // TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for(const auto& n : tl_buffer.nodes) {
        gpuNodes.push_back({n.boundsMin, 0, n.boundsMax, 0, n.center, n.nodeSize, (uint32_t)n.isLeaf,
            (uint32_t)n.isLoaded, n.childMask, n.firstPoint, n.pointCount, n.lodPoint, n.firstChild, 0});
    }
    
    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    std::vector<GPUFastRenderData> gpuPoints;
    std::vector<uint32_t> gpuLights;
    gpuPoints.reserve(tl_buffer.points.size());
    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        const auto& p = tl_buffer.points[i];
        
        gpuPoints.push_back({p.position, p.size, packRGB8(p.color), p.material.emittance, p.objectId, 0});
        
        if (p.material.emittance > 0.0f) {
            gpuLights.push_back(gpuPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});
    if(gpuPoints.empty()) gpuPoints.push_back(GPUFastRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

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
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPoints.size(), 0, 0, emissiveCount, 0
    };

    size_t outSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(gpuNodes, outSize, camData);
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

template<typename T, typename IndexType, GridStoragePath StoragePath>
frame Octree<T, IndexType, StoragePath>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    TIME_FUNCTION;
    updateStreaming(cam);
    optimize();
    thread_local RenderBuffer tl_buffer;
    buildRender(tl_buffer);
    
    vkCtx.init();

    std::vector<GPURenderNode> gpuNodes;
    gpuNodes.reserve(tl_buffer.nodes.size());
    for(const auto& n : tl_buffer.nodes) {
        gpuNodes.push_back({n.boundsMin, 0, n.boundsMax, 0, n.center, n.nodeSize, (uint32_t)n.isLeaf,
            (uint32_t)n.isLoaded, n.childMask, n.firstPoint, n.pointCount, n.lodPoint, n.firstChild, 0});
    }
    
    std::vector<bool> isLodPoint(tl_buffer.points.size(), false);
    for(const auto& n : tl_buffer.nodes) {
        if(n.lodPoint != -1) isLodPoint[n.lodPoint] = true;
    }

    std::vector<GPUPBRRenderData> gpuPBRPoints;
    gpuPBRPoints.reserve(tl_buffer.points.size());
    std::vector<GPUFastRenderData> gpuFastPoints;
    gpuFastPoints.reserve(tl_buffer.points.size());
    std::vector<uint32_t> gpuLights;

    for(size_t i = 0; i < tl_buffer.points.size(); ++i) {
        if(isLodPoint[i]) continue;
        const auto& p = tl_buffer.points[i];
        
        gpuPBRPoints.push_back({
            p.position, p.size, packRGB8(p.color), p.material.emittance, 
            packMaterialProps(p.material.roughness, p.material.metallic, p.material.transmission, p.material.ior),
            packRGB8(p.material.absorption), p.objectId, 0, 0, 0
        });
        gpuFastPoints.push_back({
            p.position, p.size, packRGB8(p.color), p.material.emittance, p.objectId, 0
        });

        if (p.material.emittance > 0.0f) {
            gpuLights.push_back(gpuPBRPoints.size() - 1);
        }
    }

    int emissiveCount = gpuLights.size();
    if(gpuNodes.empty()) gpuNodes.push_back(GPURenderNode{});
    if(gpuPBRPoints.empty()) gpuPBRPoints.push_back(GPUPBRRenderData{});
    if(gpuFastPoints.empty()) gpuFastPoints.push_back(GPUFastRenderData{});
    if(gpuLights.empty()) gpuLights.push_back(0);

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
    vkCtx.updateSkyboxBuffer(skyData);

    int lowW = std::max(1, static_cast<int>(width * pbrScale));
    int lowH = std::max(1, static_cast<int>(height * pbrScale));

    GPUCameraData pbrCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        lowW, lowH, maxBounces, useLod ? 1 : 0, invFogRange, frameCounter_,
        (int)skyW, (int)skyH, 0, 0, globalIllumination ? 1 : 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuPBRPoints.size(), 0, 0, emissiveCount, 0
    };

    size_t pbrOutSize = lowW * lowH * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(gpuNodes, pbrOutSize, pbrCamData);
    vkCtx.updateLightBuffer(gpuLights);
    vkCtx.updatePBRBuffers(gpuPBRPoints);

    int currentSampleOffset = 0;
    const long long maxWorkloadBudget = 65536; 
    const long long pixelsInFrame = (long long)lowW * lowH;

    while (currentSampleOffset < samplesPerPixel) {
        int samplesInBatch = std::max(1, (int)(maxWorkloadBudget / pixelsInFrame));
        samplesInBatch = std::min(samplesInBatch, samplesPerPixel - currentSampleOffset);
        pbrCamData.currentSampleOffset = currentSampleOffset;
        pbrCamData.dispatchSamples = samplesInBatch;

        int tileW = 256; int tileH = 256;
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

    std::vector<float> pbrRaw(lowW * lowH * 5);
    void* mappedData;
    vkMapMemory(vkCtx.device, vkCtx.outMem, 0, pbrOutSize, 0, &mappedData);
    memcpy(pbrRaw.data(), mappedData, pbrOutSize);
    vkUnmapMemory(vkCtx.device, vkCtx.outMem);

    for (size_t i = 0; i < pbrRaw.size(); i += 5) {
        pbrRaw[i] /= samplesPerPixel;
        pbrRaw[i+1] /= samplesPerPixel;
        pbrRaw[i+2] /= samplesPerPixel;
        pbrRaw[i+3] /= samplesPerPixel;
    }

    GPUCameraData fastCamData = {
        cam.origin, lodMinDistance_, cam.direction.normalized(), invLodf, cam.up.normalized(), 0.1f, cam.right(), maxDistance_,
        skylight_, tanHalfFov * aspect, backgroundColor_, tanHalfFov,
        width, height, 1, useLod ? 1 : 0, invFogRange, frameCounter_++, (int)skyW, (int)skyH, 0, 1, 0, 
        (uint32_t)gpuNodes.size(), (uint32_t)gpuFastPoints.size(), 0, 0, emissiveCount, 0
    };

    size_t fastOutSize = width * height * 5 * sizeof(float);
    vkCtx.updateCommonBuffers(gpuNodes, fastOutSize, fastCamData);
    vkCtx.updateFastBuffers(gpuFastPoints);

    int tileW = 256; int tileH = 256;
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

    std::vector<float> fastRaw(width * height * 5);
    vkMapMemory(vkCtx.device, vkCtx.outMem, 0, fastOutSize, 0, &mappedData);
    memcpy(fastRaw.data(), mappedData, fastOutSize);
    vkUnmapMemory(vkCtx.device, vkCtx.outMem);

    frame outFrame(width, height, colorformat);
    std::vector<float> colorBuffer(width * height * 3);
    
    const int radius = 2;
    const float spatialSigma = 1.5f;
    const float relativeDepthSigma = 0.05f;
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int fastIdx = (y * width + x) * 5;
            float fastR = fastRaw[fastIdx];
            float fastG = fastRaw[fastIdx + 1];
            float fastB = fastRaw[fastIdx + 2];
            float fastDepth = fastRaw[fastIdx + 3];
            int fastObj = static_cast<int>(fastRaw[fastIdx + 4]);

            int center_lx = static_cast<int>(std::round(x * pbrScale));
            int center_ly = static_cast<int>(std::round(y * pbrScale));

            float sumWeights = 0.0f;
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::clamp(center_lx + dx, 0, lowW - 1);
                    int ny = std::clamp(center_ly + dy, 0, lowH - 1);
                    int pbrIdx = (ny * lowW + nx) * 5;

                    int pbrObj = static_cast<int>(pbrRaw[pbrIdx + 4]);
                    
                    if (fastObj != pbrObj) continue;

                    float pbrDepth = pbrRaw[pbrIdx + 3];

                    float sDistSq = static_cast<float>(dx*dx + dy*dy);
                    float spatialWeight = std::exp(-sDistSq / (2.0f * spatialSigma * spatialSigma));

                    float depthDiff = std::abs(fastDepth - pbrDepth) / std::max(fastDepth, 0.1f);
                    
                    if (fastDepth > 999000.0f && pbrDepth > 999000.0f) depthDiff = 0.0f;
                    
                    float depthWeight = std::exp(-(depthDiff * depthDiff) / (2.0f * relativeDepthSigma * relativeDepthSigma));

                    float weight = spatialWeight * depthWeight;
                    sumWeights += weight;
                    sumR += pbrRaw[pbrIdx] * weight;
                    sumG += pbrRaw[pbrIdx + 1] * weight;
                    sumB += pbrRaw[pbrIdx + 2] * weight;
                }
            }
            
            if (sumWeights == 0.0f && fastObj != -1) {
                int searchRadius = radius * 3;
                float nearestDist = 99999.0f;
                
                for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                    for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                        int nx = std::clamp(center_lx + dx, 0, lowW - 1);
                        int ny = std::clamp(center_ly + dy, 0, lowH - 1);
                        int pbrIdx = (ny * lowW + nx) * 5;

                        if (static_cast<int>(pbrRaw[pbrIdx + 4]) == fastObj) {
                            float dist = static_cast<float>(dx*dx + dy*dy);
                            if (dist < nearestDist) {
                                nearestDist = dist;
                                sumR = pbrRaw[pbrIdx];
                                sumG = pbrRaw[pbrIdx + 1];
                                sumB = pbrRaw[pbrIdx + 2];
                                sumWeights = 1.0f;
                            }
                        }
                    }
                }
            }
            
            int outIdx = (y * width + x) * 3;
            
            if (sumWeights > 0.0f) { 
                colorBuffer[outIdx]     = std::clamp(sumR / sumWeights, 0.0f, 1.0f);
                colorBuffer[outIdx + 1] = std::clamp(sumG / sumWeights, 0.0f, 1.0f);
                colorBuffer[outIdx + 2] = std::clamp(sumB / sumWeights, 0.0f, 1.0f);
            } else { 
                colorBuffer[outIdx]     = std::clamp(fastR, 0.0f, 1.0f);
                colorBuffer[outIdx + 1] = std::clamp(fastG, 0.0f, 1.0f);
                colorBuffer[outIdx + 2] = std::clamp(fastB, 0.0f, 1.0f);
            }
        }
    }

    outFrame.setData(colorBuffer, frame::colormap::RGB);
    return outFrame;
}

#endif
}