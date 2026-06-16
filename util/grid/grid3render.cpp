#include "grid3eigen.hpp"
#include "../timing_decorator.hpp"
namespace Grid {

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::rasterize(const Camera& cam, int height, int width, frame* colorOut, frame* depthOut,
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
    // const Eigen::Matrix4f proj = cam.getProjectionMatrix(aspect, nearPlane, maxDistance_);
    const float f = 1.0f / std::tan(cam.fovRad() / 2.0f);
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0, 0) = f / aspect;
    proj(1, 1) = f;
    proj(2, 2) = 1.0f;
    proj(3, 2) = 1.0f;
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

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::renderDepthMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::B);
    rasterize(cam, height, width, nullptr, &out, nullptr, nullptr);
    return out;
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::renderColorMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGB);
    rasterize(cam, height, width, &out, nullptr, nullptr, nullptr);
    return out;
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::renderNormalMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGB);
    rasterize(cam, height, width, nullptr, nullptr, &out, nullptr);
    return out;
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::renderObjectMap(const Camera& cam, int height, int width) {
    frame out(width, height, frame::colormap::RGBA);
    rasterize(cam, height, width, nullptr, nullptr, nullptr, &out);
    return out;
}

template<typename T, typename GasT, typename IndexType>
Eigen::Vector3f Octree<T, GasT, IndexType>::traceVoxelRay(const PointType& origin, const PointType& dir, float minT, float maxT, const Eigen::Vector3f& bgColor) {
    if (!root_) return bgColor;
    Ray ray(origin, dir);
    {
        float rootMin = minT;
        float rootMax = maxT;
        if (!rayBoxIntersect(ray, root_->bounds(), rootMin, rootMax)) return bgColor;
    }
    struct NodeStackItem { OctreeNode* node; float tEnter; };
    std::vector<NodeStackItem> stack;
    stack.reserve(256);
    stack.push_back({root_.get(), std::max(0.0f, minT)});

    float closestT = maxT;
    const NodeData* hit = nullptr;
    PointType hitNormal = PointType::Zero();
    std::vector<NodeStackItem> childOrder;

    while (!stack.empty()) {
        NodeStackItem item = stack.back();
        stack.pop_back();

        OctreeNode* node = item.node;
        if (!node) continue;
        if (item.tEnter > closestT) continue;

        if (!node->isLoaded()) {
            ensureLoaded(node, false);
            if (!node->isLoaded()) continue;
        }

        for (const auto& pt : node->points) {
            if (!pt) continue;
            BoundingBox cube = pt->getCubeBounds();
            float t = 0.0f;
            PointType normal = PointType::Zero();
            if (rayCubeIntersect(ray, cube, t, normal)) {
                if (t >= minT && t < closestT) {
                    closestT = t;
                    hit = pt.get();
                    hitNormal = normal;
                }
            }
        }

        if (node->isLeaf()) continue;

        childOrder.clear();
        for (const auto& child : node->children) {
            if (!child) continue;
            float cMin = minT;
            float cMax = closestT;
            if (rayBoxIntersect(ray, child->bounds(), cMin, cMax)) {
                if (cMax < minT) continue;
                float entry = std::max(minT, cMin);
                if (entry <= closestT) childOrder.push_back({child.get(), entry});
            }
        }
        std::sort(childOrder.begin(), childOrder.end(),
                  [](const NodeStackItem& a, const NodeStackItem& b) { return a.tEnter > b.tEnter; });
        for (const auto& c : childOrder) stack.push_back(c);
    }

    if (!hit) return bgColor;

    Eigen::Vector3f albedo(0.8f, 0.8f, 0.8f);
    if (auto obj = getObject(hit->objectId)) {
        Eigen::Vector4f c = obj->getColor(hit->colorIdx);
        albedo = c.head<3>();
    }

    const PointType lightDir = (-dir * 0.2f + PointType(0.3f, 0.8f, 0.2f)).normalized();
    float diffuse = std::max(0.0f, hitNormal.dot(lightDir));
    Eigen::Vector3f shaded = albedo.cwiseProduct(skylight_ + Eigen::Vector3f::Constant(diffuse));
    shaded = shaded.cwiseMin(1.0f).cwiseMax(0.0f);

    float fogRange = std::max(1e-6f, maxDistance_ - lodMinDistance_);
    float fog = std::clamp((maxDistance_ - closestT) / fogRange, 0.1f, 1.0f);
    return shaded * fog + bgColor * (1.0f - fog);
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat, bool rasterOnly) {
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

    #pragma omp parallel for schedule(dynamic, 128) collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pix = y * width + x;
            int base = pix * channels;

            float ndc_x = ax * x + bx;
            float ndc_y = ay * y + by;
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
                traceMax = std::min(traceMax, rasterT - 0.2f);
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

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::renderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat, int samplesPerPixel,
                int maxBounces, bool globalIllumination, bool useLod) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::fastRenderFrameVulkan(const Camera& cam, int height, int width, frame::colormap colorformat) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::blendedRenderFrameVulkan(const Camera& cam, int height, int width, float pbrScale,
                frame::colormap colorformat, int samplesPerPixel, int maxBounces, bool globalIllumination, bool useLod) {
    updateStreaming(cam);
    frame outFrame(width, height, colorformat);


    return outFrame;

}

#endif
}