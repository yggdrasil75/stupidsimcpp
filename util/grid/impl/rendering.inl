#pragma once
#ifdef VULKAN_SUPPORT
#include <vulkan/vulkan.h>
#endif

namespace Grid {

struct Ray {
    PointType origin;
    PointType dir;
    PointType invDir;
    uint8_t sign[3];
    uint8_t signMask;
    Ray(const PointType& orig, const PointType& dir) : origin(orig), dir(dir) {
        invDir = dir.cwiseInverse();
        sign[0] = (invDir[0] < 0);
        sign[1] = (invDir[1] < 0);
        sign[2] = (invDir[2] < 0);
        signMask = (sign[0] | sign[1] << 1 | sign[2] << 2);
    }
};

struct StackItem {
    uint32_t nodeIdx;
    float tMin;
    float tMax;    
};

struct ChildInterval {
    uint32_t nodeIdx;
    float tMin;
    float tMax;
};

template<typename T, typename IndexType>
struct RenderData_ {
    PointType position;
    float size;
    Eigen::Vector4f color;
    uint32_t materialIdx;
    BoundingBox bounds;
    int objectId;
    // float gasDensity;
};

template<typename T, typename IndexType>
struct RenderNode_ {
    BoundingBox bounds;
    PointType center;
    float nodeSize;
    bool isLeaf;
    bool isLoaded;
    uint8_t childMask;
    
    uint32_t firstPoint;
    uint32_t pointCount;
    int32_t lodPoint;
    uint32_t firstChild;
    
    OctreeNode_<T, IndexType>* originalNode; 
};

template<typename T, typename IndexType>
struct RenderBuffer_ {
    std::vector<RenderNode_<T, IndexType>> nodes;
    std::vector<RenderData_<T, IndexType>> points;
    std::vector<Material_> materials;
    std::unordered_map<int, uint32_t> objMaterialOffsets;
    uint32_t defaultMatIdx;

    void clear() {
        nodes.clear();
        points.clear();
        materials.clear();
        objMaterialOffsets.clear();
    }
    
};
    
}