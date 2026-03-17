#include "nodeData.hpp"
#include <array>
#include <vector>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>

template<typename T, typename IndexSize = uint16_t, typename high = double, typename medium = float, typename low = Eigen::half>
class Octree;

class BoundingBox;
class OBoundingBox : BoundingBox;
class Ray;

static constexpr uint8_t NODE_LEAF_BIT = 1 << 0;
static constexpr uint8_t NODE_LOADED_BIT = 1 << 1;
static constexpr uint8_t NODE_LOD_VALID_BIT = 1 << 2;

template<typename T, typename IndexSize, typename high, typename medium, typename low>
struct OctreeNode_ {
    using PointMax = Eigen::Matrix<long double, 3, 1>;
    using PointHigh = Eigen::Matrix<high, 3, 1>;
    using PointMedium = Eigen::Matrix<medium, 3, 1>;
    using PointLow = Eigen::Matrix<low, 3, 1>;

    private:
        typename Octree<T, IndexSize, high, medium, low>::BoundingBox bounds;
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
        std::vector<std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>> points;
        std::array<std::unique_ptr<OctreeNode_>, 8> children;
        uint8_t flags;
        
        mutable std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>> lodData;
        mutable std::mutex lodMutex; 

        OctreeNode_(const PointHigh& min, const PointHigh& max) : flags(NODE_LEAF_BIT | NODE_LOADED_BIT), lodData(nullptr) {
            bounds.bounds[0] = min.template cast<medium>();
            bounds.bounds[1] = max.template cast<medium>();
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

        inline bool isStatic() const {
            for (const auto& pt : points) {
                if (pt && !pt->isStatic()) return false;
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && !children[i]->isStatic()) return false;
                }
            }
            return true;
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
            
            if (!isEmpty()) {
                ///TODO: average children recursively, average points in this node.
                ///ignore unless active and visible
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
                        children[i] = std::make_unique<OctreeNode_>(PointHigh::Zero(), PointHigh::Zero());
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
                    auto pt = std::make_shared<NodeData_<T, IndexSize, high, medium, low>>();
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
                saveData(currentDir); 
                points.clear();      
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
            if (!points.empty()) return false;
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && !children[i]->isEmpty()) return false;
                }
            }
            return true;
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

                children[i] = std::make_unique<OctreeNode_>(childMin, childMax);
            }
            setLeaf(false);
        }

        void insert(const std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>& point, const PointHigh& ptAbsCenter, const PointHigh& ptHalfSize, 
                    size_t maxPoints, size_t maxDepth, size_t currentDepth) {
            if (!point) return;
            invalidateLod();

            high childSize = static_cast<high>((bounds.bounds[1] - bounds.bounds[0]).maxCoeff() * 0.5);
            high ptSize = ptHalfSize.maxCoeff() * 2.0;

            if (!isLeaf() && ptSize >= childSize) {
                point->setPosition((ptAbsCenter - center).template cast<low>());
                points.push_back(point);
                return;
            }

            if (isLeaf()) {
                point->setPosition((ptAbsCenter - center).template cast<low>());
                points.push_back(point);

                if (points.size() > maxPoints && currentDepth < maxDepth) {
                    split();

                    std::vector<std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>> keptPoints;
                    auto currentPoints = std::move(points);
                    points.clear();

                    for (const auto& p : currentPoints) {
                        PointHigh pCenter = p->center(this->center).template cast<high>();
                        PointHigh pHalf = p->getHalfSize().template cast<high>();
                        high pSz = pHalf.maxCoeff() * 2.0;

                        if (pSz >= childSize) {
                            p->setPosition((pCenter - center).template cast<low>());
                            keptPoints.push_back(p);
                        } else {
                            PointHigh pMin = pCenter - pHalf;
                            PointHigh pMax = pCenter + pHalf;
                            bool inserted = false;

                            for (int i = 0; i < 8; ++i) {
                                if (children[i]->intersectsBounds(pMin, pMax)) {
                                    children[i]->insert(p, pCenter, pHalf, maxPoints, maxDepth, currentDepth + 1);
                                    inserted = true;
                                    break;
                                }
                            }
                            if (!inserted) {
                                p->setPosition((pCenter - center).template cast<low>());
                                keptPoints.push_back(p);
                            }
                        }
                    }
                    points = std::move(keptPoints);
                }
            } else {
                PointHigh pMin = ptAbsCenter - ptHalfSize;
                PointHigh pMax = ptAbsCenter + ptHalfSize;
                bool inserted = false;

                for (int i = 0; i < 8; ++i) {
                    if (children[i]->intersectsBounds(pMin, pMax)) {
                        children[i]->insert(point, ptAbsCenter, ptHalfSize, maxPoints, maxDepth, currentDepth + 1);
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) {
                    point->setPosition((ptAbsCenter - center).template cast<low>());
                    points.push_back(point);
                }
            }
        }

        std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>> find(const PointHigh& pos, int objectId = -1) const {
            if (!contains(pos)) return nullptr;
            
            for (const auto& pt : points) {
                if (objectId == -1 || pt->getObjectId() == objectId) {
                    if ((pt->center(this->center).template cast<high>() - pos).norm() < 1e-4) {
                        return pt;
                    }
                }
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && children[i]->contains(pos)) {
                        auto res = children[i]->find(pos, objectId);
                        if (res) return res;
                    }
                }
            }
            return nullptr;
        }

        void findRadius(const PointHigh& centerPos, high radius, std::vector<std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>>& result) const {
            // if (!intersectsSphere(centerPos, radius)) return;
            
            high r2 = radius * radius;
            for (const auto& pt : points) {
                if ((pt->center(this->center).template cast<high>() - centerPos).squaredNorm() <= r2) {
                    result.push_back(pt);
                }
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->findRadius(centerPos, radius, result);
                }
            }
        }

        void findObject(int objectId, std::vector<std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>>& result) const {
            for (const auto& pt : points) {
                if (pt->getObjectId() == objectId) {
                    result.push_back(pt);
                }
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->findObject(objectId, result);
                }
            }
        }

        std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>> remove(const PointHigh& pos, int objectId = -1) {
            if (!contains(pos)) return nullptr;
            
            for (auto it = points.begin(); it != points.end(); ++it) {
                if (objectId == -1 || (*it)->getObjectId() == objectId) {
                    if (((*it)->center(this->center).template cast<high>() - pos).norm() < 1e-4) {
                        auto pt = *it;
                        points.erase(it);
                        return pt;
                    }
                }
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i] && children[i]->contains(pos)) {
                        auto res = children[i]->remove(pos, objectId);
                        if (res) return res;
                    }
                }
            }
            return nullptr;
        }

        void removeObject(int objectId, std::vector<std::pair<std::shared_ptr<NodeData_<T, IndexSize, high, medium, low>>, PointHigh>>& removed) {
            for (auto it = points.begin(); it != points.end(); ) {
                if ((*it)->getObjectId() == objectId) {
                    removed.push_back({*it, (*it)->center(this->center).template cast<high>()});
                    it = points.erase(it);
                } else {
                    ++it;
                }
            }
            if (!isLeaf()) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i]) children[i]->removeObject(objectId, removed);
                }
            }
        }
};