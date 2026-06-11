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

    
}