#include "grid3eigen.hpp"
#include "../timing_decorator.hpp"
namespace Grid {

template<typename T, typename IndexType>
void Octree<T, IndexType>::rasterize(const Camera& cam, int height, int width, frame* colorOut, frame* depthOut,
            frame* normalOut, frame* objectOut, std::vector<float>* linearDepth) {
    TIME_FUNCTION;
    const size_t pixels = static_cast<size_t>(width) * height;

    std::vector<float> zBuffer(pixels, std::numeric_limits<float>::max());
    std::vector<float> colorBuf;
    std::vector<float> normalBuf;
    std::vector<int>   objectBuf;
    size_t colorCh = 3;
    if (colorOut) {
        colorCh = (colorOut->colorFormat == frame::colormap::RGBA ||
                   colorOut->colorFormat == frame::colormap::BGRA) ? 4 :
                  (colorOut->colorFormat == frame::colormap::B) ? 1 : 3;
        colorBuf.assign(pixels * colorCh, 0.0f);
    }
    if (normalOut) normalBuf.assign(pixels * 3, 0.0f);
    if (objectOut) objectBuf.assign(pixels, -1);

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const Eigen::Matrix4f view = cam.getViewMatrix();
    const float ambient = 0.25f;
    const float nearPlane = 0.1f;
    const Eigen::Matrix4f proj = cam.getProjectionMatrix(aspect, nearPlane, maxDistance_);
    const Eigen::Matrix4f vp = proj * view;
    const Eigen::Vector3f camFwd = cam.forward();
    const Eigen::Vector3f camOrigin = cam.origin;
    const Eigen::Vector3f lightDir = (-cam.direction * 0.2f + Eigen::Vector3f(0.3f, 0.8f, 0.2f)).normalized();

    std::unordered_map<int, std::shared_ptr<GridObject>> localObjects;
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        localObjects = objects_;
    }

    for (const auto& [id, obj] : localObjects) {
        std::shared_lock<std::shared_mutex> objLock(obj->objMutex);
        if (obj->objMesh.tris.empty() || obj->objMesh.vertices.empty()) continue;

        const size_t vcount = obj->objMesh.vertices.size();
        std::vector<Eigen::Vector4f> clip(vcount);
        for (size_t i = 0; i < vcount; ++i) {
            const PointType& p = obj->objMesh.vertices[i].pos;
            clip[i] = vp * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
        }

        for (const auto& tr : obj->objMesh.tris) {
            const size_t idx[3] = {tr.a, tr.b, tr.c};
            if (clip[idx[0]].w() <= nearPlane || clip[idx[1]].w() <= nearPlane || clip[idx[2]].w() <= nearPlane) continue;

            float sx[3];
            float sy[3];
            float invW[3];
            float viewZ[3];
            for (int k = 0; k < 3; ++k) {
                const Eigen::Vector4f& c = clip[idx[k]];
                invW[k] = 1.0f / c.w();
                sx[k] = (c.x() * invW[k] * 0.5f + 0.5f) * width;
                sy[k] = (1.0f - (c.y() * invW[k] * 0.5f + 0.5f)) * height;
                viewZ[k] = (obj->objMesh.vertices[idx[k]].pos - camOrigin).dot(camFwd);
            }

            int minX = std::max(0, (int)std::floor(std::min({sx[0], sx[1], sx[2]})));
            int maxX = std::min(width - 1, (int)std::ceil(std::max({sx[0], sx[1], sx[2]})));
            int minY = std::max(0, (int)std::floor(std::min({sy[0], sy[1], sy[2]})));
            int maxY = std::min(height - 1, (int)std::ceil(std::max({sy[0], sy[1], sy[2]})));
            if (minX > maxX || minY > maxY) continue;

            float area = edgeFn(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
            if (area == 0.0f) continue;
            float invArea = 1.0f / area;

            const PointType& a = obj->objMesh.vertices[idx[0]].pos;
            PointType normal = (obj->objMesh.vertices[idx[1]].pos - a).cross(obj->objMesh.vertices[idx[2]].pos - a).normalized();
            if (normal.dot(camOrigin - a) < 0.0f) normal = -normal;

            Eigen::Vector4f albedo = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.0f);
            if (colorOut) albedo = obj->getColor(obj->objMesh.vertices[idx[0]].cIdx);

            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    float px = x + 0.5f;
                    float py = y + 0.5f;
                    float w0 = edgeFn(sx[1], sy[1], sx[2], sy[2], px, py) * invArea;
                    float w1 = edgeFn(sx[2], sy[2], sx[0], sy[0], px, py) * invArea;
                    float w2 = edgeFn(sx[0], sy[0], sx[1], sy[1], px, py) * invArea;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

                    float wInterp = w0 * invW[0] + w1 * invW[1] + w2 * invW[2];
                    if (wInterp <= 0.0f) continue;
                    float depth = (w0 * invW[0] * viewZ[0] + w1 * invW[1] * viewZ[1] + w2 * invW[2] * viewZ[2]) / wInterp;
                    if (depth <= nearPlane) continue;

                    size_t pix = static_cast<size_t>(y) * width + x;
                    if (depth >= zBuffer[pix]) continue;
                    zBuffer[pix] = depth;

                    if (colorOut) {
                        float lambert = std::max(0.0f, normal.dot(lightDir));
                        float shade = ambient + (1.0f - ambient) * lambert;
                        size_t cb = pix * colorCh;
                        colorBuf[cb] = std::min(albedo.x() * shade, 1.0f);
                        if (colorCh >= 3) {
                            colorBuf[cb + 1] = std::min(albedo.y() * shade, 1.0f);
                            colorBuf[cb + 2] = std::min(albedo.z() * shade, 1.0f);
                        }
                        if (colorCh == 4) colorBuf[cb + 3] = albedo.w();
                    }
                    if (normalOut) {
                        normalBuf[pix * 3    ] = normal.x() * 0.5f + 0.5f;
                        normalBuf[pix * 3 + 1] = normal.y() * 0.5f + 0.5f;
                        normalBuf[pix * 3 + 2] = normal.z() * 0.5f + 0.5f;
                    }
                    if (objectOut) objectBuf[pix] = id;
                }
            }
        }
    }

    if (colorOut)  colorOut->setData(colorBuf, colorOut->colorFormat);
    if (normalOut) normalOut->setData(normalBuf, frame::colormap::RGB);

    if (depthOut) {
        float invRange = 1.0f / std::max(1e-6f, (maxDistance_ - lodMinDistance_));
        std::vector<float> depthBuf(pixels, 0.0f);
        for (size_t i = 0; i < pixels; ++i) {
            float d = zBuffer[i];
            if (d == std::numeric_limits<float>::max()) continue;
            float n = std::clamp((d - lodMinDistance_) * invRange, 0.0f, 1.0f);
            depthBuf[i] = 1.0f - n;
        }
        depthOut->setData(depthBuf, frame::colormap::B);
    }

    if (objectOut) {
        std::vector<uint8_t> objData(pixels * 4, 0);
        for (size_t i = 0; i < pixels; ++i) {
            uint32_t v = static_cast<uint32_t>(objectBuf[i]);
            objData[i * 4    ] = static_cast<uint8_t>( v        & 0xFF);
            objData[i * 4 + 1] = static_cast<uint8_t>((v >> 8)  & 0xFF);
            objData[i * 4 + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
            objData[i * 4 + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
        }
        objectOut->setData(objData, frame::colormap::RGBA);
    }

    if (linearDepth) *linearDepth = std::move(zBuffer);
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderDepthMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::B);
    rasterize(cam, height, width, nullptr, &out, nullptr, nullptr);
    return out;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderColorMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGB);
    rasterize(cam, height, width, &out, nullptr, nullptr, nullptr);
    return out;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderNormalMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGB);
    rasterize(cam, height, width, nullptr, nullptr, &out, nullptr);
    return out;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderObjectMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGBA);
    rasterize(cam, height, width, nullptr, nullptr, nullptr, &out);
    return out;
}

template<typename T, typename IndexType>
Eigen::Vector3f Octree<T, IndexType>::traceVoxelRay(const PointType& origin, const PointType& dir, float minT, float maxT, const Eigen::Vector3f& bgColor) {
    Ray ray(origin, dir);
    Eigen::Vector3f outColor = Eigen::Vector3f::Zero();
    float tranm = 0.0f;
    if (rayBoxIntersect(ray, root->bounds, minT, maxT)) {
        float currentMaxDist = maxT;
        StackItem stack[32768];
        int stackPtr = 0;
        stack[stackPtr++] = {0, std::max(0.0f, minT), maxT};

        while(stackPtr > 0 && tranm < 1) {
            StackItem current = stack[--stackPtr];
            if (current.tMin > currentMaxDist) continue;
            const RenderNode_<T, IndexType>& node = buffer.nodes[current.nodeIdx];
            if (!node.isLoaded && node.originalNode) {
                ensureLoaded(node.originalNode, true);
            }

            for (uint32_t i = 0; i < node.pointCount; ++i) {
                const RenderData_<T, IndexType>& pt = buffer.poitns[node.firstPoint + 1];
                float t;
                PointType n, h;
                if (rayCubeIntersect(ray, &pt, t, n, f)) {
                    if (t >= 0 && t <= currentMaxDist && t <= current.tMax + 0.001f) {
                        currentMaxDist = t;
                        hit = &pt;
                        tranm += hit->color.w();
                        outColor += hit->color.head<3>() * hit->color.w()
                    }
                }
            }

            if (node.isLeaf || !node.isLoaded) continue;
            float t0 = current.tMin;
            float t1 = current.tMax;

            Eigen::Vector3f ttt = (node.center - ray.origin).cwiseProduct(ray.invDir);
            int currIdx = ((t0 >= ttt.x()) ? 1 : 0) | ((t0 >= ttt.y()) ? 2 : 0) | ((t0 >= ttt.z()) ? 4 : 0);

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

            // if (stackPtr + childCount > 256) continue;

            for (int i = childCount - 1; i >= 0; --i) {
                stack[stackPtr++] = {children[i].nodeIdx, children[i].tMin, children[i].tMax};
            }

        }
    }
    return outColor;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    TIME_FUNCTION;
    updateStreaming(cam);
    generateMeshes();
    frame outFrame(width, height, colorformat);

    size_t channels;
    switch (colorformat) {
        case frame::colormap::RGBA:
        case frame::colormap::BGRA:
            channels = 4;
            break;
        case frame::colormap::B:
            channels = 1;
            break;
        case frame::colormap::BGR:
        case frame::colormap::RGB:
        default:
            channels = 3;
            break;
    }

    frame colorFrame(width, height, frame::colormap::RGBA);
    frame objectFrame(width, height, frame::colormap::RGBA);
    std::vector<float> linearDepth;
    rasterize(cam, height, width, &colorFrame, nullptr, nullptr, &objectFrame, &linearDepth);
    std::cout << "done in rasterizer" << std::endl;

    const std::vector<uint8_t>& rasterColor = colorFrame.getData();
    const std::vector<uint8_t>& objData = objectFrame.getData();
    const size_t pixels = static_cast<size_t>(width) * height;

    std::vector<uint8_t> frameData(pixels * channels, 0);

    const Eigen::Vector3f camRight = cam.right();
    const Eigen::Vector3f camUp = cam.up;
    const Eigen::Vector3f camFwd = cam.forward();
    const Eigen::Vector3f camOrigin = cam.origin;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float f = 1.0f / std::tan(cam.fovRad() / 2.0f);
    const float nearPlane = 0.1f;

    float inv_width = 1.0f / width;
    float inv_height = 1.0f / height;
    float two_over_width = 2.0f * inv_width;
    float two_over_height = 2.0f * inv_height;
    float one_minus_inv_height = 1.0f - inv_height;
    float ax = two_over_width;
    float bx = inv_width - 1.0f;
    float ay = -two_over_height;
    float by = one_minus_inv_height;
    Eigen::Vector3f Rx = camRight * (aspect * two_over_width);
    Eigen::Vector3f Ry = camUp * (-two_over_height);
    Eigen::Vector3f C0 = camRight * (aspect * (inv_width - 1.0f)) + camUp * (1.0f - inv_height) + camFwd * f;

    const bool rasterOnly = !skipPaths();
    #pragma omp parallel for schedule(dynamic, 128) collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pix = y * width + x;
            int base = pix * channels;

            float ndc_x = ax * x + bx;
            float ndc_y = ay * y + by;
            // Eigen::Vector3f dir = (camRight * (ndc_x * aspect) + camUp * ndc_y + camFwd * f).normalized();
            Eigen::Vector3f dir = (Rx * x + Ry * y + C0).normalized();

            Eigen::Vector3f sky = skybox_.sampleVector(dir);
            Eigen::Vector3f color = sky;

            float rasterDepth = linearDepth[pix];
            bool haveRaster = (rasterDepth != std::numeric_limits<float>::max());
            float transmission = 1.0f;
            if (haveRaster) {
                size_t cb = pix * 4;
                Eigen::Vector3f surf(rasterColor[cb] / 255.0f, rasterColor[cb + 1] / 255.0f, rasterColor[cb + 2] / 255.0f);
                transmission = rasterColor[cb + 3] / 255.0f;
                color = surf * (1.0f - transmission) + sky * transmission;
            }
            float cosFwd = dir.dot(camFwd);
            float traceMax = maxDistance_;
            if (haveRaster && cosFwd > 1e-4f) {
                float rasterT = rasterDepth / cosFwd;
                traceMax = std::min(traceMax, rasterT - 1e-3f);
            }
            bool skipTrace = haveRaster && transmission <= 0.0f && !rasterOnly;
            if (!skipTrace && traceMax > nearPlane) {
                color = traceVoxelRay(camOrigin, dir, nearPlane, traceMax, color);
            }
            uint8_t r8 = static_cast<uint8_t>(std::clamp(color.x() * 255.0f, 0.0f, 255.0f));
            uint8_t g8 = static_cast<uint8_t>(std::clamp(color.y() * 255.0f, 0.0f, 255.0f));
            uint8_t b8 = static_cast<uint8_t>(std::clamp(color.z() * 255.0f, 0.0f, 255.0f));

            if (channels >= 3) {
                if (colorformat == frame::colormap::RGB || colorformat == frame::colormap::RGBA) {
                    frameData[base] = r8;
                    frameData[base + 1] = g8;
                    frameData[base + 2] = b8;
                } else {
                    frameData[base] = b8;
                    frameData[base + 1] = g8;
                    frameData[base + 2] = r8;
                }
                if (channels == 4) frameData[base + 3] = 255;
            } else {
                frameData[base] = static_cast<uint8_t>(std::clamp(0.2126f * color.x() * 255.0f + 0.7152f * color.y() * 255.0f + 0.0722f * color.z() * 255.0f, 0.0f, 255.0f));
            }
        }
    }

    outFrame.setData(frameData);
    return outFrame;
}

#ifdef VULKAN_SUPPORT

template<typename T, typename IndexType>
frame Octree<T, IndexType>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;
}

template<typename T, typename IndexType>
frame Octree<T, IndexType>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;

}

#endif
}