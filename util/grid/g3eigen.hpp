#ifndef g3eigen
#define g3eigen

#include "../materials/materials.hpp"

#ifdef SSE
#include <immitntrin.h>
#endif

static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
//gap for future options. static is last because it generally will be set once and never changed, but the rest might be changed
static constexpr uint8_t STATIC_BIT = 1 << 7;

template<typename T, typename IndexSize = uint16_t, typename high = double, typename medium = float, typename low = Eigen::half>
class Octree {
public:
    using PointHigh = Eigen::Matrix<high, 3, 1>;
    using PointMedium = Eigen::Matrix<medium, 3, 1>;
    using PointLow = Eigen::Matrix<low, 3, 1>;

    struct Ray {
        PointMedium origin;
        PointMedium dir;
        PointMedium invDir;
        uint8_t sign[3];
        uint8_t signMask;
        Ray(const PointMedium& orig, const PointMedium& dir) : origin(orig), dir(dir) {
            invDir = dir.cwiseInverse();
            sign[0] = (invDir[0] < 0);
            sign[1] = (invDir[1] < 0);
            sign[2] = (invDir[2] < 0);
            signMask = (sign[0] | sign[1] << 1 | sign[2] << 2);
        }
    };

    struct BoundingBox {
        PointMedium bounds[2];

        bool intersect(const Ray& r, float& tMin, float& tMax) {
            float tymin, tymax, tzmin, tzmax;
            
            tMin = (bounds[r.sign[0]].x - r.orig.x) * r.invdir.x;
            tMax = (bounds[1-r.sign[0]].x - r.orig.x) * r.invdir.x;
            tymin = (bounds[r.sign[1]].y - r.orig.y) * r.invdir.y;
            tymax = (bounds[1-r.sign[1]].y - r.orig.y) * r.invdir.y;
            
            if ((tMin > tymax) || (tymin > tMax))
                return false;

            if (tymin > tMin)
                tMin = tymin;
            if (tymax < tMax)
                tMax = tymax;
            
            tzmin = (bounds[r.sign[2]].z - r.orig.z) * r.invdir.z;
            tzmax = (bounds[1-r.sign[2]].z - r.orig.z) * r.invdir.z;
            
            if ((tMin > tzmax) || (tzmin > tMax))
                return false;

            if (tzmin > tMin)
                tMin = tzmin;
            if (tzmax < tMax)
                tMax = tzmax;

            return true;
        }
    };

    struct OBoundingBox : BoundingBox {
        PointMedium center;
        PointMedium extents;
        Eigen::Quaternion<low> orientation;

        bool intersect(const Ray& r, float& tMin, float& tMax) {
            PointMedium localOrigin = orientation.conjugate() * (r.origin - center);
            PointMedium localDir = orientation.conjugate() * r.dir;
            Ray localRay(localOrigin, localDir);
            BoundingBox localBounds;
            localBounds.bounds[0] = -extents;
            localBounds.bounds[1] = extents;
            return localBounds.intersect(localRay, tMin, tMax);
        }
    };

    class NodeData {
        T data;
        PointLow position;
        Eigen::Quaternion<low> orientation;
        int objectId;
        low size;

        IndexSize colorIDX;
        IndexSize materialIdx;
        private:
            uint8_t flags;
            
        NodeData(const T& data, const PointType& pos, bool visible, IndexType colorIdx, float size = 0.01f,
                 bool active = true, int objectId = -1, int subId = 0, IndexType materialIdx = 0, bool staticnode) 
                : data(data), position(pos), objectId(objectId), subId(subId), size(size), 
                  colorIdx(colorIdx), materialIdx(materialIdx), flags(0) {
                    setActive(active);
                    setVisible(visible);
                    setStatic(staticnode);
                  }
        
        NodeData() : objectId(-1), subId(0), size(0.0f), colorIdx(0), materialIdx(0), flags(0) {}

        inline bool isActive() const {
            return flags & ACTIVE_BIT;
        }
        inline bool isVISIBLE() const {
            return flags & VISIBLE_BIT;
        }
        inline bool isStatic() const {
            return flags & STATIC_BIT;
        }
        inline bool isActiveAndVisible() const {
            return ((flags & ACTIVE_BIT) != 0) && ((flags & VISIBLE_BIT) != 0);
        }
        
        inline void setActive(bool val) {
            val ? (flags |= ACTIVE_BIT) : (flags &= ~ACTIVE_BIT);
        }
        inline void setVisible(bool val) {
            val ? (flags |= VISIBLE_BIT) : (flags &= ~VISIBLE_BIT);
        }
        inline void setStatic(bool val) {
            val ? (flags |= STATIC_BIT) : (flags &= ~STATIC_BIT);
        }

        PointLow getHalfSize() const {
            return PointLow(size * 0.5f, size * 0.5f, size * 0.5f);
        }
        
        OBoundingBox getCubeBounds(const NodePoint& nodeCenter) const {
            OBoundingBox obb;
            obb.center = nodeCenter + localPosition.template cast<medium>();
            obb.extents = getHalfSize().template cast<medium>();
            obb.orientation = orientation.template cast<medium>();
            return obb;
        }

        PointMedium center(const NodePoint& nodeCenter) const {
            return nodeCenter + position.template cast<medium>();
        }
    };

    struct OctreeNode {
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::unique_ptr<OctreeNode>, 8> children;
        PointType center;
        float nodeSize;
        bool isLeaf;
        
        mutable std::shared_ptr<NodeData> lodData;
        mutable std::mutex lodMutex; 

        OctreeNode(const PointMedium& min, const PointMedium& max) : isLeaf(true), lodData(nullptr) {
            bounds.bounds[0] = min;
            bounds.bounds[1] = max;
            for (std::unique_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (min + max) * 0.5;
            nodeSize = (max - min).norm();
        }

        bool contains(const PointType& point) const {
            return (point[0] >= bounds.bounds[0][0] && point[0] <= bounds.bounds[1][0] &&
                    point[1] >= bounds.bounds[0][1] && point[1] <= bounds.bounds[1][1] &&
                    point[2] >= bounds.bounds[0][2] && point[2] <= bounds.bounds[1][2]);
        }

        bool isEmpty() const {
            return points.empty();
        }
    };
private:
    std::unique_ptr<OctreeNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    std::map<IndexSize, Eigen::Vector3f> colorMap;
    std::shared_mutex colormutex;
    std::map<IndexSize, Material> materialMap;
    std::shared_mutex materialmutex;
    Material globalMat;
    Eigen::Vector3f globalColor;
    float globalIntensity;

    inline IndexSize getColorIndex(const Eigen::Vector3f& color) {
        IndexSize closestIdx = 0;
        float minDistance = std::numeric_limits<float>::max();
        {
            std::shared_lock<std::shared_mutex> read_lock(*colormutex);
            
            for (const auto& pair : colorMap) {
                float dist = (pair.second - color).norm();
                if (dist < EPSILON) {
                    return pair.first;
                }
                if (dist < minDistance) {
                    minDistance = dist;
                    closestIdx = pair.first;
                }
            }
            
            if (colorMap.size() >= MAX_INDEX) {
                std::cerr << "Warning: Color palette limits reached! Using nearest color\n";
                return closestIdx;
            }
        }
        std::unique_lock<std::shared_mutex> write_lock(*colormutex);
        for (const auto& pair : colorMap) {
            float dist = (pair.second - color).norm();
            if (dist < EPSILON) {
                return pair.first;
            }
        }
        if (colorMap.size() < MAX_INDEX) {
            IndexSize newIdx = static_cast<IndexSize>(colorMap.size());
            colorMap[newIdx] = color;
            return newIdx;
        }

        return closestIdx;
    }

    inline IndexSize getMaterialIndex(const Material& mat) {
        IndexSize closestIdx = 0;
        float minDistance = std::numeric_limits<float>::max();
        ///TODO:
        std::cerr << "Warning: Material map limit reached! Using nearest material." << std::endl;
        return closestIdx;
    }
}

#endif