#ifndef g3eigen
#define g3eigen

#include "../materials/materials.hpp"
#include <mutex>
#include <map>
#include <unordered_map>
#include <shared_mutex>
#include <vector>
#include <unordered_set>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include "../basicdefines.hpp"

#ifdef SSE
#include <immintrin.h>
#endif

static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
//gap for future options. static is last because it generally will be set once and never changed, but the rest might be changed
static constexpr uint8_t STATIC_BIT = 1 << 7;

static constexpr uint8_t NODE_LEAF_BIT = 1 << 0;
static constexpr uint8_t NODE_LOADED_BIT = 1 << 1;
static constexpr uint8_t NODE_LOD_VALID_BIT = 1 << 2;

template<typename T, typename IndexSize = uint16_t, typename high = double, typename medium = float, typename low = Eigen::half>
class Octree {
public:
    using PointMax = Eigen::Matrix<long double, 3, 1>;
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
            signMask = (sign[0] | (sign[1] << 1) | (sign[2] << 2));
        }
    };

    struct BoundingBox {
        PointMedium bounds[2];

        bool intersect(const Ray& r, float& tMin, float& tMax) const {
            float tymin, tymax, tzmin, tzmax;
            
            tMin = (bounds[r.sign[0]][0] - r.origin[0]) * r.invDir[0];
            tMax = (bounds[1 - r.sign[0]][0] - r.origin[0]) * r.invDir[0];
            tymin = (bounds[r.sign[1]][1] - r.origin[1]) * r.invDir[1];
            tymax = (bounds[1 - r.sign[1]][1] - r.origin[1]) * r.invDir[1];
            
            if ((tMin > tymax) || (tymin > tMax)) return false;

            if (tymin > tMin) tMin = tymin;
            if (tymax < tMax) tMax = tymax;
            
            tzmin = (bounds[r.sign[2]][2] - r.origin[2]) * r.invDir[2];
            tzmax = (bounds[1 - r.sign[2]][2] - r.origin[2]) * r.invDir[2];
            
            if ((tMin > tzmax) || (tzmin > tMax)) return false;

            if (tzmin > tMin) tMin = tzmin;
            if (tzmax < tMax) tMax = tzmax;

            return true;
        }
        
        bool intersect(const BoundingBox& otherbox) const {
            return (bounds[0][0] <= otherbox.bounds[1][0] && bounds[1][0] >= otherbox.bounds[0][0] &&
                    bounds[0][1] <= otherbox.bounds[1][1] && bounds[1][1] >= otherbox.bounds[0][1] &&
                    bounds[0][2] <= otherbox.bounds[1][2] && bounds[1][2] >= otherbox.bounds[0][2]);
        }
    };

    struct OBoundingBox : BoundingBox {
        PointMedium center;
        PointMedium extents;
        Eigen::Quaternion<low> orientation;

        bool intersect(const Ray& r, float& tMin, float& tMax) const {
            PointMedium localOrigin = orientation.conjugate().template cast<medium>() * (r.origin - center);
            PointMedium localDir = orientation.conjugate().template cast<medium>() * r.dir;
            Ray localRay(localOrigin, localDir);
            BoundingBox localBounds;
            localBounds.bounds[0] = -extents;
            localBounds.bounds[1] = extents;
            return localBounds.intersect(localRay, tMin, tMax);
        }
    };

    class NodeData {
        T data;
        private:
            PointLow position;
            Eigen::Quaternion<low> orientation;
            int objectId;
            low size;

            IndexSize colorIDX;
            IndexSize materialIDX;
            uint8_t flags;
            
        public:
            NodeData(const T& data, const PointLow& pos, bool visible, IndexSize colorIDX, float size = 0.01f,
                    bool active = true, int objectId = -1, IndexSize materialIdx = 0, bool staticnode = false)
                    : data(data), position(pos), objectId(objectId), size(static_cast<low>(size)), 
                    colorIDX(colorIDX), materialIDX(materialIdx), flags(0) {
                        setActive(active);
                        setVisible(visible);
                        setStatic(staticnode);
                        orientation.setIdentity();
                    }
            
            NodeData() : objectId(-1), size(0.0), colorIDX(0), materialIDX(0), flags(0) {
                orientation.setIdentity();
            }

            inline T getData() const { return data; }

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

            inline IndexSize getColorIDX() const {
                return colorIDX;
            }

            inline IndexSize getMaterialIDX() const {
                return materialIDX;
            }

            inline void setColorIdx(IndexSize idx) {
                colorIDX = idx;
            }

            inline void setMaterialIDX(IndexSize idx) {
                materialIDX = idx;
            }

            inline PointLow getPosition() const { 
                return position;
            }

            inline void setPosition(const PointLow& pos) { 
                position = pos;
            }

            PointLow getHalfSize() const {
                return PointLow(size * 0.5f, size * 0.5f, size * 0.5f);
            }
            
            OBoundingBox getCubeBounds(const PointHigh& nodeCenter) const {
                OBoundingBox obb;
                obb.center = nodeCenter.template cast<medium>() + position.template cast<medium>();
                obb.extents = getHalfSize().template cast<medium>();
                obb.orientation = orientation;
                return obb;
            }

            PointMedium center(const PointHigh& nodeCenter) const {
                return nodeCenter + position.template cast<medium>();
            }

            void serialize(std::ostream& os) const {
                os.write(reinterpret_cast<const char*>(&data), sizeof(T));
                os.write(reinterpret_cast<const char*>(&position), sizeof(position));
                os.write(reinterpret_cast<const char*>(&orientation), sizeof(orientation));
                os.write(reinterpret_cast<const char*>(&objectId), sizeof(objectId));
                os.write(reinterpret_cast<const char*>(&size), sizeof(size));
                os.write(reinterpret_cast<const char*>(&colorIDX), sizeof(colorIDX));
                os.write(reinterpret_cast<const char*>(&materialIDX), sizeof(materialIDX));
                os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
            }

            void deserialize(std::istream& is) {
                is.read(reinterpret_cast<char*>(&data), sizeof(T));
                is.read(reinterpret_cast<char*>(&position), sizeof(position));
                is.read(reinterpret_cast<char*>(&orientation), sizeof(orientation));
                is.read(reinterpret_cast<char*>(&objectId), sizeof(objectId));
                is.read(reinterpret_cast<char*>(&size), sizeof(size));
                is.read(reinterpret_cast<char*>(&colorIDX), sizeof(colorIDX));
                is.read(reinterpret_cast<char*>(&materialIDX), sizeof(materialIDX));
                is.read(reinterpret_cast<char*>(&flags), sizeof(flags));
            }
    };

    struct OctreeNode {
        private:
            BoundingBox bounds;
            PointHigh center;

            inline void setLodValid(bool val) { 
                val ? (flags |= NODE_LOD_VALID_BIT) : (flags &= ~NODE_LOD_VALID_BIT);
            }

            inline void setLeaf(bool val) { 
                val ? (flags |= NODE_LEAF_BIT) : (flags &= ~NODE_LEAF_BIT);
            }

            inline void setLoaded(bool val) { 
                val ? (flags |= NODE_LOADED_BIT) : (flags &= ~NODE_LOADED_BIT);
            }
        public:
            std::vector<std::shared_ptr<NodeData>> points;
            std::array<std::unique_ptr<OctreeNode>, 8> children;
            uint8_t flags;
            
            mutable std::shared_ptr<NodeData> lodData;
            mutable std::mutex lodMutex; 

            OctreeNode(const PointHigh& min, const PointHigh& max) : flags(NODE_LEAF_BIT | NODE_LOADED_BIT), lodData(nullptr) {
                bounds.bounds[0] = min;
                bounds.bounds[1] = max;
                center = (min + max) * 0.5;
            }

            inline bool isLeaf() const { 
                return flags & NODE_LEAF_BIT;
            }
            
            inline bool isLoaded() const { 
                return flags & NODE_LOADED_BIT;
            }
            
            inline bool isLodValid() const { 
                return flags & NODE_LOD_VALID_BIT;
            }
            
            inline void invalidateLod() {
                setLodValid(false);
            }

            bool contains(const PointHigh& point) const {
                return (point[0] >= bounds.bounds[0][0] && point[0] <= bounds.bounds[1][0] &&
                        point[1] >= bounds.bounds[0][1] && point[1] <= bounds.bounds[1][1] &&
                        point[2] >= bounds.bounds[0][2] && point[2] <= bounds.bounds[1][2]);
            }

            std::string getRegionName() const {
                std::ostringstream oss;
                oss << static_cast<int>(std::floor(center.x())) << "." << static_cast<int>(std::floor(center.y())) << "." << static_cast<int>(std::floor(center.z()));
                return oss.str();
            }

            void generateLod() const {
                if (isLodValid()) return;
                std::lock_guard<std::mutex> lock(lodMutex);

                if (isLeaf()) {
                    if (points.empty()) {
                        lodData = nullptr;
                    } else {
                        PointHigh avgPos = PointHigh::Zero();
                        std::unordered_map<IndexSize, int> colors;
                        std::unordered_map<IndexSize, int> mats;
                        for (const auto& p : points) {
                            avgPos += p->getPosition().template cast<high>();
                            colors[p->getColorIDX()]++;
                            mats[p->getMaterialIDX()]++;
                        }

                        size_t count = points.size();
                        avgPos /= static_cast<high>(count);
                        
                        
                    }
                }
            }

            void saveStructure(std::ostream& os) const {
                os.write(reinterpret_cast<const char*>(&bounds), sizeof(bounds));
                os.write(reinterpret_cast<const char*>(&center), sizeof(center));
                os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
                
                uint8_t childMask = 0;
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) childMask |= (1 << i);
                }
                os.write(reinterpret_cast<const char*>(&childMask), sizeof(childMask));
                
                if (!isLeaf()) {
                    for (int i = 0; i < 8; ++i) {
                        if (children[i]) children[i]->saveStructure(os);
                    }
                }
            }

            void loadStructure(std::istream& is) {
                is.read(reinterpret_cast<char*>(&bounds), sizeof(bounds));
                is.read(reinterpret_cast<char*>(&center), sizeof(center));
                is.read(reinterpret_cast<char*>(&flags), sizeof(flags));
                
                uint8_t childMask = 0;
                is.read(reinterpret_cast<char*>(&childMask), sizeof(childMask));
                
                if (!isLeaf()) {
                    for (int i = 0; i < 8; ++i) {
                        if (childMask & (1 << i)) {
                            children[i] = std::make_unique<OctreeNode>(PointHigh::Zero(), PointHigh::Zero());
                            children[i]->loadStructure(is);
                        }
                    }
                }
                setLoaded(false);
            }

            void saveData(const std::filesystem::path& currentDir) {
                if (isLeaf()) {
                    if (!isLoaded()) return;
                    
                    std::filesystem::create_directories(currentDir);
                    std::filesystem::path filePath = currentDir / (getRegionName() + ".leaf");
                    std::ofstream os(filePath, std::ios::binary);
                    
                    size_t numPoints = points.size();
                    os.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
                    for (const auto& pt : points) {
                        pt->serialize(os);
                    }
                } else {
                    std::filesystem::path subDir = currentDir / getRegionName();
                    std::filesystem::create_directories(subDir);
                    for (auto& child : children) {
                        if (child) child->saveData(subDir);
                    }
                }
            }

            void loadData(const std::filesystem::path& currentDir) {
                if (isLeaf()) {
                    if (isLoaded()) return;
                    
                    std::filesystem::path filePath = currentDir / (getRegionName() + ".leaf");
                    if (!std::filesystem::exists(filePath)) return;
                    
                    std::ifstream is(filePath, std::ios::binary);
                    size_t numPoints = 0;
                    is.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
                    
                    points.clear();
                    points.reserve(numPoints);
                    for (size_t i = 0; i < numPoints; ++i) {
                        auto pt = std::make_shared<NodeData>();
                        pt->deserialize(is);
                        points.push_back(pt);
                    }
                    setLoaded(true);
                } else {
                    std::filesystem::path subDir = currentDir / getRegionName();
                    for (auto& child : children) {
                        if (child) child->loadData(subDir);
                    }
                }
            }

            bool intersectsBounds(const PointHigh& bMin, const PointHigh& bMax) const {
                return (bounds.bounds[0][0] <= bMax[0] && bounds.bounds[1][0] >= bMin[0]) &&
                    (bounds.bounds[0][1] <= bMax[1] && bounds.bounds[1][1] >= bMin[1]) &&
                    (bounds.bounds[0][2] <= bMax[2] && bounds.bounds[1][2] >= bMin[2]);
            }

            void offloadRegion(const std::filesystem::path& currentDir, const PointHigh& minB, const PointHigh& maxB) {
                if (!intersectsBounds(minB, maxB)) return;
                
                if (isLeaf() && isLoaded()) {
                    saveData(currentDir); // Write to disk
                    points.clear();       // Free RAM
                    points.shrink_to_fit();
                    setLoaded(false);
                } else if (!isLeaf()) {
                    std::filesystem::path subDir = currentDir / getRegionName();
                    for (auto& child : children) {
                        if (child) child->offloadRegion(subDir, minB, maxB);
                    }
                }
            }

            void loadRegion(const std::filesystem::path& currentDir, const PointHigh& minB, const PointHigh& maxB) {
                if (!intersectsBounds(minB, maxB)) return;
                
                if (isLeaf() && !isLoaded()) {
                    loadData(currentDir);
                } else if (!isLeaf()) {
                    std::filesystem::path subDir = currentDir / getRegionName();
                    for (auto& child : children) {
                        if (child) child->loadRegion(subDir, minB, maxB);
                    }
                }
            }

            bool isEmpty() const {
                return points.empty();
            }

            void split(const std::filesystem::path& currentDir = "") {
                if (!isLeaf()) return;
                if (!currentDir.empty()) loadData(currentDir);

                PointHigh minB = bounds.bounds[0].template cast<high>();
                PointHigh maxB = bounds.bounds[1].template cast<high>();

                for (int i = 0; i < 8; ++i) {
                    PointHigh childMin, childMax;
                    
                    childMin[0] = (i & 1) ? center[0] : minB[0];
                    childMax[0] = (i & 1) ? maxB[0] : center[0];
                    
                    childMin[1] = (i & 2) ? center[1] : minB[1];
                    childMax[1] = (i & 2) ? maxB[1] : center[1];
                    
                    childMin[2] = (i & 4) ? center[2] : minB[2];
                    childMax[2] = (i & 4) ? maxB[2] : center[2];

                    children[i] = std::make_unique<OctreeNode>(childMin, childMax);
                }
                setLeaf(false);
            }

            void insert(const std::shared_ptr<NodeData>& point, const PointHigh& ptAbsCenter, const PointHigh& ptHalfSize, 
                        size_t maxPoints, size_t maxDepth, size_t currentDepth) {
                if (!point) return;
                invalidateLod();

                high childSize = static_cast<high>((bounds.bounds[1] - bounds.bounds[0]).maxCoeff() * 0.5);
                high ptSize = ptHalfSize.maxCoeff() * 2.0;

                if (!isLeaf() && ptSize >= childSize) {
                    points.push_back(point);
                    return;
                }

                if (isLeaf()) {
                    points.push_back(point);

                    if (points.size() > maxPoints && currentDepth < maxDepth) {
                        split();

                        std::vector<std::shared_ptr<NodeData>> keptPoints;
                        auto currentPoints = std::move(points);
                        points.clear();

                        for (const auto& p : currentPoints) {
                            PointHigh pCenter = p->center(this->center).template cast<high>();
                            PointHigh pHalf = p->getHalfSize().template cast<high>();
                            high pSz = pHalf.maxCoeff() * 2.0;

                            if (pSz >= childSize) {
                                keptPoints.push_back(p);
                            } else {
                                PointHigh pMin = pCenter - pHalf;
                                PointHigh pMax = pCenter + pHalf;

                                for (int i = 0; i < 8; ++i) {
                                    if (children[i]->intersectsBounds(pMin, pMax)) {
                                        children[i]->insert(p, pCenter, pHalf, maxPoints, maxDepth, currentDepth + 1);
                                    }
                                }
                            }
                        }
                        points = std::move(keptPoints);
                    }
                } else {
                    PointHigh pMin = ptAbsCenter - ptHalfSize;
                    PointHigh pMax = ptAbsCenter + ptHalfSize;

                    for (int i = 0; i < 8; ++i) {
                        if (children[i]->intersectsBounds(pMin, pMax)) {
                            children[i]->insert(point, ptAbsCenter, ptHalfSize, maxPoints, maxDepth, currentDepth + 1);
                        }
                    }
                }
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
    std::filesystem::path storageBasePath;

    struct SpatialUsage {
        PointHigh sumPos = PointHigh::Zero();
        size_t count = 0;
        PointHigh centroid() const {
            return (count > 0) ? sumPos / static_cast<high>(count) : PointHigh::Zero();
        }
    };

    static uint32_t expandBits(uint32_t v) {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    static uint32_t morton3D(float x, float y, float z) {
        x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
        y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
        z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
        uint32_t xx = expandBits(static_cast<uint32_t>(x));
        uint32_t yy = expandBits(static_cast<uint32_t>(y));
        uint32_t zz = expandBits(static_cast<uint32_t>(z));
        return (xx * 4) + (yy * 2) + zz;
    }

    void accumulateUsage(const OctreeNode* node, 
                         std::map<IndexSize, SpatialUsage>& cUsage,
                         std::map<IndexSize, SpatialUsage>& mUsage) const {
        if (!node) return;
        for (const auto& point : node->points) {
            if (point) {
                PointHigh globalPos = point->center(node->center).template cast<high>();
                cUsage[point->getColorIDX()].sumPos += globalPos;
                cUsage[point->getColorIDX()].count++;
                mUsage[point->getMaterialIDX()].sumPos += globalPos;
                mUsage[point->getMaterialIDX()].count++;
            }
        }
        for (const auto& child : node->children) {
            accumulateUsage(child.get(), cUsage, mUsage);
        }
    }

    void remapTreeIndices(OctreeNode* node,
                          const std::unordered_map<IndexSize, IndexSize>& cRemap,
                          const std::unordered_map<IndexSize, IndexSize>& mRemap) {
        if (!node) return;
        for (auto& point : node->points) {
            if (point) {
                auto cit = cRemap.find(point->getColorIDX());
                if (cit != cRemap.end()) point->setColorIdx(cit->second);

                auto mit = mRemap.find(point->getMaterialIDX());
                if (mit != mRemap.end()) point->setMaterialIDX(mit->second);
            }
        }
        for (auto& child : node->children) {
            remapTreeIndices(child.get(), cRemap, mRemap);
        }
    }
    void checkColorUsage(const OctreeNode* node, std::unordered_set<IndexSize>& usedColors) const {
        if (!node) return;
        for (const auto& point: node->points) {
            if (point) usedColors.insert(point->getColorIDX());
        }
        for (const auto& child: node->children) {
            checkColorUsage(child.get(), usedColors);
        }
    }

    void checkMaterialUsage(const OctreeNode* node, std::unordered_set<IndexSize>& usedMaterials) const {
        if (!node) return;
        for (const auto& point: node->points) {
            if (point) usedMaterials.insert(point->getMaterialIDX());
        }
        for (const auto& child: node->children) {
            checkMaterialUsage(child.get(), usedMaterials);
        }
    }

    void optimizeColorMap() {
        std::unordered_set<IndexSize> usedColors;
        checkColorUsage(root_.get(), usedColors);
        for (auto it = colorMap.begin(); it != colorMap.end(); ) {
            if (usedColors.find(it->first) == usedColors.end()) {
                it = colorMap.erase(it);
            } else {
                ++it;
            }
        }
    }

    void optimizeMaterialMap() {
        std::unordered_set<IndexSize> usedMaterials;
        checkMaterialUsage(root_.get(), usedMaterials);
        for (auto it = materialMap.begin(); it != materialMap.end(); ) {
            if (usedMaterials.find(it->first) == usedMaterials.end()) {
                it = materialMap.erase(it);
            } else {
                ++it;
            }
        }
    }

    IndexSize getNextAvailableColorID() const {
        for (IndexSize i = 0; i < MAX_INDEX; ++i) {
            if (colorMap.find(i) == colorMap.end()) return i;
        }
        return MAX_INDEX;
    }

    IndexSize getNextAvailableMaterialId() const {
        for (IndexSize i = 0; i < MAX_INDEX; ++i) {
            if (materialMap.find(i) == materialMap.end()) return i;
        }
        return MAX_INDEX;
    }

    inline float calculateMaterialDistance(const Material& a, const Material& b) const {
        float dist = 0.0f;
        dist += std::abs(a.ior - b.ior);
        dist += std::abs(a.dispersion - b.dispersion);
        dist += std::abs(a.chromaticity - b.chromaticity) / 500.0f;
        dist += std::abs(a.bandwidth - b.bandwidth) / 100.0f;
        dist += std::abs(a.transmission - b.transmission);
        dist += std::abs(a.roughness - b.roughness);
        dist += std::abs(a.emittance - b.emittance);
        dist += std::abs(a.density - b.density) / 1000.0f;
        dist += std::abs(a.speedOfSound - b.speedOfSound) / 343.0f;
        dist += std::abs(a.audioAbsorption - b.audioAbsorption);
        if (a.light != b.light) dist += 10.0f;
        
        dist += (a.rgb.template cast<float>() - b.rgb.template cast<float>()).norm();
        return dist;
    }

public:
    void optimize() {
        std::unique_lock<std::shared_mutex> lock_c(colormutex, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock_m(materialmutex, std::defer_lock);
        std::lock(lock_c, lock_m);

        std::map<IndexSize, SpatialUsage> cUsage, mUsage;
        accumulateUsage(root_.get(), cUsage, mUsage);

        PointHigh cMin = PointHigh::Constant(std::numeric_limits<high>::max());
        PointHigh cMax = PointHigh::Constant(std::numeric_limits<high>::lowest());
        for (const auto& kv : cUsage) {
            if (kv.second.count > 0) {
                PointHigh c = kv.second.centroid();
                cMin = cMin.cwiseMin(c);
                cMax = cMax.cwiseMax(c);
            }
        }
        PointHigh cRange = cMax - cMin;
        if (cRange.x() == 0) cRange.x() = 1;
        if (cRange.y() == 0) cRange.y() = 1;
        if (cRange.z() == 0) cRange.z() = 1;

        std::vector<std::pair<IndexSize, uint32_t>> cSort;
        for (const auto& kv : cUsage) {
            if (kv.second.count > 0) {
                PointHigh norm = (kv.second.centroid() - cMin).cwiseQuotient(cRange);
                cSort.push_back({kv.first, morton3D(norm.x(), norm.y(), norm.z())});
            }
        }
        std::sort(cSort.begin(), cSort.end(), [](const auto& a, const auto& b){ return a.second < b.second; });

        std::unordered_map<IndexSize, IndexSize> cRemap;
        std::map<IndexSize, Eigen::Vector3f> newColorMap;
        IndexSize nextC = 0;
        for (const auto& item : cSort) {
            cRemap[item.first] = nextC;
            newColorMap[nextC] = colorMap[item.first];
            nextC++;
        }

        PointHigh mMin = PointHigh::Constant(std::numeric_limits<high>::max());
        PointHigh mMax = PointHigh::Constant(std::numeric_limits<high>::lowest());
        for (const auto& kv : mUsage) {
            if (kv.second.count > 0) {
                PointHigh m = kv.second.centroid();
                mMin = mMin.cwiseMin(m);
                mMax = mMax.cwiseMax(m);
            }
        }
        PointHigh mRange = mMax - mMin;
        if (mRange.x() == 0) mRange.x() = 1;
        if (mRange.y() == 0) mRange.y() = 1;
        if (mRange.z() == 0) mRange.z() = 1;

        std::vector<std::pair<IndexSize, uint32_t>> mSort;
        for (const auto& kv : mUsage) {
            if (kv.second.count > 0) {
                PointHigh norm = (kv.second.centroid() - mMin).cwiseQuotient(mRange);
                mSort.push_back({kv.first, morton3D(norm.x(), norm.y(), norm.z())});
            }
        }
        std::sort(mSort.begin(), mSort.end(), [](const auto& a, const auto& b){ return a.second < b.second; });

        std::unordered_map<IndexSize, IndexSize> mRemap;
        std::map<IndexSize, Material> newMaterialMap;
        IndexSize nextM = 0;
        for (const auto& item : mSort) {
            mRemap[item.first] = nextM;
            newMaterialMap[nextM] = materialMap[item.first];
            nextM++;
        }

        remapTreeIndices(root_.get(), cRemap, mRemap);

        colorMap = std::move(newColorMap);
        materialMap = std::move(newMaterialMap);
    }

    inline IndexSize getColorIndex(const Eigen::Vector3f& color) {
        IndexSize closestIdx = 0;
        float minDistance = std::numeric_limits<float>::max();
        {
            std::shared_lock<std::shared_mutex> read_lock(colormutex);
            
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
        }
        std::unique_lock<std::shared_mutex> write_lock(colormutex);
        for (const auto& pair : colorMap) {
            float dist = (pair.second - color).norm();
            if (dist < EPSILON) {
                return pair.first;
            }
        }
        if (colorMap.size() == MAX_INDEX) {
            optimizeColorMap();
        }

        if (colorMap.size() >= MAX_INDEX) {
            std::cerr << "Warning: Color palette limits reached! All indices utilized in Tree. Using nearest color.\n";
            return closestIdx;
        }
        IndexSize newIdx = getNextAvailableColorID();
        colorMap[newIdx] = color;
        return newIdx;
    }

    inline IndexSize getMaterialIndex(const Material& mat) {
        IndexSize closestIdx = 0;
        float minDistance = std::numeric_limits<float>::max();
        
        {
            std::shared_lock<std::shared_mutex> read_lock(materialmutex);
            for (const auto& pair : materialMap) {
                float dist = calculateMaterialDistance(pair.second, mat);
                if (dist < EPSILON) {
                    return pair.first;
                }
                if (dist < minDistance) {
                    minDistance = dist;
                    closestIdx = pair.first;
                }
            }
        }

        std::unique_lock<std::shared_mutex> write_lock(materialmutex);

        for (const auto& pair : materialMap) {
            float dist = calculateMaterialDistance(pair.second, mat);
            if (dist < EPSILON) return pair.first;
        }

        if (materialMap.size() == MAX_INDEX) {
            optimizeMaterialMapLocked();
        }

        if (materialMap.size() >= MAX_INDEX) {
            std::cerr << "Warning: Material map limit reached! All indices utilized in Tree. Using nearest material.\n";
            return closestIdx;
        }
        
        IndexSize newIdx = getNextAvailableMaterialId();
        materialMap[newIdx] = mat;
        return newIdx;
    }

    void setStoragePath(const std::string& path) {
        storageBasePath = path;
    }

    void saveMaps(const std::filesystem::path& path) {
        std::ofstream os(path, std::ios::binary);
        if (!os) return;

        std::shared_lock<std::shared_mutex> lock(colormutex);
        size_t cSize = colorMap.size();
        os.write(reinterpret_cast<const char*>(&cSize), sizeof(cSize));
        for (const auto& pair : colorMap) {
            os.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(float) * 3);
        }
        std::shared_lock<std::shared_mutex> lock(materialmutex);
        size_t mSize = materialMap.size();
        os.write(reinterpret_cast<const char*>(&mSize), sizeof(mSize));
        for (const auto& pair : materialMap) {
            os.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            os.write(reinterpret_cast<const char*>(&pair.second), sizeof(Material));
        }
    }

    void loadMaps(const std::filesystem::path& path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) return;

        std::unique_lock<std::shared_mutex> lock(colormutex);
        colorMap.clear();
        size_t cSize = 0;
        is.read(reinterpret_cast<char*>(&cSize), sizeof(cSize));
        for (size_t i = 0; i < cSize; ++i) {
            IndexSize idx;
            Eigen::Vector3f color;
            is.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            is.read(reinterpret_cast<char*>(color.data()), sizeof(float) * 3);
            colorMap[idx] = color;
        }
        std::unique_lock<std::shared_mutex> lock(materialmutex);
        materialMap.clear();
        size_t mSize = 0;
        is.read(reinterpret_cast<char*>(&mSize), sizeof(mSize));
        for (size_t i = 0; i < mSize; ++i) {
            IndexSize idx;
            Material mat;
            is.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            is.read(reinterpret_cast<char*>(&mat), sizeof(Material));
            materialMap[idx] = mat;
        }
    }
    void save() {
        if (storageBasePath.empty()) return;
        std::filesystem::create_directories(storageBasePath);
        
        saveMaps(storageBasePath / "maps.bin");
        
        if (root_) {
            std::ofstream os(storageBasePath / "tree_struct.bin", std::ios::binary);
            root_->saveStructure(os);
            os.close();
            
            root_->saveData(storageBasePath);
        }
    }

    void loadAll(bool loadPayloadsIntoMemory = false) {
        if (storageBasePath.empty()) return;
        
        loadMaps(storageBasePath / "maps.bin");
        
        std::ifstream is(storageBasePath / "tree_struct.bin", std::ios::binary);
        if (is) {
            root_ = std::make_unique<OctreeNode>(PointHigh::Zero(), PointHigh::Zero());
            root_->loadStructure(is);
            is.close();
            if (loadPayloadsIntoMemory) {
                root_->loadData(storageBasePath);
            }
        }
    }

    void offloadRegion(const PointHigh& minBounds, const PointHigh& maxBounds) {
        if (root_ && !storageBasePath.empty()) {
            root_->offloadRegion(storageBasePath, minBounds, maxBounds);
        }
    }

    void loadRegion(const PointHigh& minBounds, const PointHigh& maxBounds) {
        if (root_ && !storageBasePath.empty()) {
            root_->loadRegion(storageBasePath, minBounds, maxBounds);
        }
    }
};

#endif