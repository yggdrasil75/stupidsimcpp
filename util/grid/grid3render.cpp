#include "grid3eigen.hpp"
namespace Grid {

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

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float ndc_y = 1.0f - (y + 0.5f) / height * 2.0f;
            float ndc_x = (x + 0.5f) / width * 2.0f - 1.0f;
            Eigen::Vector3f dir = (camRight * (ndc_x * aspect) + camUp * ndc_y + camFwd * f).normalized();
            Eigen::Vector3f bg = skybox_.sampleVector(dir);

            int base = (y * width + x) * channels;
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
    std::unordered_set<int> hasMeshSet;

    {
        std::shared_lock<std::shared_mutex> lock(objectsMutex_);
        for (const auto& [id, obj] : objects_) {
            std::shared_lock<std::shared_mutex> objLock(obj->objMutex);
            
            if (obj->objMesh.tris.empty() || obj->objMesh.vertices.empty()) continue;
            
            for (size_t i = 0; i < obj->objMesh.vertices(); ++i) {
                const auto& vert = obj->objMesh.vertices[i];
                Eigen::Vector4f worldVert(vert.pos.x(), vert.pos.y(), vert.pos.z(), 1)
            }
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