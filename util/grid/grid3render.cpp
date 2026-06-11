#include "grid3eigen.hpp"
namespace Grid {

template<typename T, typename GasT, typename IndexType>
typename Octree<T, GasT, IndexType>::RasterBuffers Octree<T, GasT, IndexType>::softwareRasterize(const Camera& cam, int height, int width) {
    RasterBuffers out;
    out.zBuffer.resize(width * height, std::numeric_limits<float>::max());
    out.colorBuffer.resize(width * height, Eigen::Vector3f::Zero());
    out.normalBuffer.resize(width * height, Eigen::Vector3f::Zero());
    out.objectBuffer.resize(width * height, -1);

    Eigen::Matrix4f view = cam.getViewMatrix();
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    Eigen::Matrix4f proj = cam.getProjectionMatrix(aspect, 0.1f, maxDistance_);
    Eigen::Matrix4f vp = proj * view;

    std::unordered_map<int, std::shared_ptr<GridObject>> localObjects;
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        localObjects = objects_;
    }
}

template<typename T, typename GasT, typename IndexType>
frame Octree<T, GasT, IndexType>::fastRenderFrame(const Camera& cam, int height, int width, frame::colormap colorformat) {
    updateStreaming(cam);
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

    std::vector<float> zBuffer(width * height, std::numeric_limits<float>::max());
    std::vector<uint8_t> frameData(width * height * channels, 0);
    Eigen::Vector3f camRight = cam.right();
    Eigen::Vector3f camUp = cam.up;
    Eigen::Vector3f camFwd = cam.forward();
    Eigen::Matrix4f view = cam.getViewMatrix();
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    Eigen::Matrix4f proj = cam.getProjectionMatrix(aspect, 0.1f, 10000.0f);
    Eigen::Matrix4f vp = proj * view;
    float f = 1.0f / std::tan(cam.fovRad() / 2.0f);

    const PointType globalLightDir = (-cam.direction * 0.2f).normalized();
    const float minVisibility = 0.1f;
    float invMaxMin = 1.0f / (maxDistance_ - lodMinDistance_)

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int base = (y * width + x) * channels;
            float ndc_y = 1.0f - (y + 0.5f) / height * 2.0f;
            float ndc_x = (x + 0.5f) / width * 2.0f - 1.0f;
            Eigen::Vector3f dir = (camRight * (ndc_x * aspect) + camUp * ndc_y + camFwd * f).normalized();
            Eigen::Vector3f bg = skybox_.sampleVector(dir);

            if (channels >= 3) {
                if (colorformat == frame::colormap::RGB || colorformat == frame::colormap::RGBA) {
                    frameData[base] = bg.x();
                    frameData[base+1] = bg.y();
                    frameData[base+2] = bg.z();
                } else {
                    frameData[base] = bg.z();
                    frameData[base+1] = bg.y();
                    frameData[base+2] = bg.x();
                }
            } else {
                frameData[base] = static_cast<uint8_t>(std::clamp((0.2126f * bg.x() + 0.7152f * bg.y() + 0.0722f * bg.z()) * 255.0f, 0.0f, 255.0f));
            }
            if (channels == 4) frameData[base+3] = 255;
        }
    }
    std::unordered_map<int, std::shared_ptr<GridObject>> localObjects;
    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        localObjects = objects_;
    }

    std::vector<Eigen::Vector3f> screenspaceTri;
    std::unordered_set<int> hasMeshSet;

    for (const auto& [id, obj] : localObjects) {
        std::shared_lock<std::shared_mutex> objLock(obj->objMutex);
        if (obj->objMesh.tris.empty() || obj->objMesh.vertices.empty()) continue;
        hasMeshSet.insert(id);

        std::vector<Eigen::Vector4f> clip(obj->objMesh.vertices.size());
        for (size_t i = 0; i < obj->objMesh.vertices.size(); ++i) {
            const auto& vert = obj->objMesh.vertices[i];
            clip[i] = vp * Eigen::Vector4f(vert.pos.x(), vert.pos.y(), vert.pos.z(), 1.0f);
        }

        for (const auto& t : obj->objMesh.tris) {
            const size_t idx[3] = {t.a, t.b, t.c};
            if (clip[idx[t.a]].w() <= nearPlane || clip[idx[t.b]].w() <= nearPlane || clip[idx[t.c]].w() <= nearPlane) continue;

            float invW[3];
            float sx[3];
            Eigen::Matrix3f worldW;
            for (int k = 0; k < 3; ++k) {
                const Eigen::Vector4f& c = clip[idx[k]];
                float w = c.w();
                invW[k] = 1.0f / w;
                sx[k] = (c.x() / w * 0.5f + 0.5f) * width;
                sy[k] = (1.0f - (c.y() / w * 0.5f + 0.5f)) * height;
                worldOverW[k] = obj->objMesh.vertices[idx[k]].pos * invW[k];
            }
            minX = std::max(0, (int)std::floor(std::min({sx[0], sx[1], sx[2]})));
            maxX = std::min(width - 1, (int)std::ceil(std::max({sx[0], sx[1], sx[2]})));
            minY = std::max(0, (int)std::floor(std::min({sy[0], sy[1], sy[2]})));
            maxY = std::min(height - 1, (int)std::ceil(std::max({sy[0], sy[1], sy[2]})));
            if (minX > maxX || minY > maxY) continue;

            const PointType& a = obj->objMesh.vertices[idx[0]].pos;
            normal = (obj->objMesh.vertices[idx[1]].pos - a).cross(obj->objMesh.vertices[idx[2]].pos - a).normalized();
            if (st.normal.dot(cam.origin - a) < 0.0f) st.normal = -st.normal;
            color = obj->getColor(obj->objMesh.vertices[idx[0]].cIdx).template head<3>();
        }
    }

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