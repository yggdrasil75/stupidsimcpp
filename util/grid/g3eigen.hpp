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
#include "nodeData.hpp"
#include "octreeNode.hpp"

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

    struct CelestialBody {
        PointMedium direction;
        float angularRadius;
        float cosAngularRadius;
        uint8_t r, g, b, emittance;
        bool baked;
        struct PixelBackup {
            size_t x, y;
            std::vector<uint8_t> data;
        };
        std::vector<PixelBackup> backup;

        CelestialBody() : angularRadius(0), cosAngularRadius(1), r(255), g(255), b(255), emittance(255), baked(false) {}
    };

    struct Skybox {
        frame skybox;
        std::map<int, CelestialBody> bodies;
        Eigen::Quaternion<medium> skyRotation;

        Skybox(size_t w = 1024, size_t h = 1024) : skybox(w, h, frame::colormap::RGBA), skyRotation(Eigen::Quaternion<medium>::Identity()) { }

        void dirToUV(const PointMedium& dir, float& u, float& v) const {
            PointMedium d = dir.normalized();
            u = 0.5f + (std::atan2(d.z(), d.x()) / (2.0f * M_PI));
            v = 0.5f - (std::asin(d.y()) / M_PI);
        }

        PointMedium uvToDir(float u, float v) const {
            float θ = (u - 0.5f) * 2.0f * M_PI;
            float φ = (0.5f - v) * M_PI;
            float y = std::sin(φ);
            float cosφ = std::cos(φ);
            float x = std::cos(θ) * cosφ;
            float z = std::sin(θ) * cosφ;
            return PointMedium(x, y, z);
        }
        
        std::vector<uint8_t> sample(const PointMedium& dir) {
            PointMedium rotatedDir = skyRotation * dir;
            for (auto it = bodies.rbegin(); it != bodies.rend(); ++it) {
                if (!it->second.baked) {
                    if (rotatedDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                        return getFormattedColor(it->second.r, it->second.g, it->second.b, it->second.emittance);
                    }
                }
            }

            float u, v;
            dirToUV(rotatedDir, u, v);

            u = std::clamp(u, 0.0f, 0.9999f);
            v = std::clamp(v, 0.0f, 0.9999f);
            size_t x = static_cast<size_t>(u * skybox.getWidth());
            size_t y = static_cast<size_t>(v * skybox.getHeight());

            return skybox.getPixel(x, y);
        }

        void setBackground(float r, float g, float b, float e) {
            size_t w = skybox.getWidth();
            size_t h = skybox.getHeight();
            std::vector<float> data(w * h * 4);

            for (size_t i = 0; i < data.size(); i += 4) {
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = e;
            }
            skybox.setData(data);
        }
        
        void addBody(int id, const PointMedium& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance) {
            removeBody(id);
            CelestialBody body;
            body.direction = dir.normalized();
            body.angularRadius = angularRadius;
            body.cosAngularRadius = std::cos(angularRadius);
            body.r = r;
            body.g = g;
            body.b = b;
            body.emittance = emittance;
            body.baked = false;
            bodies[id] = std::move(body);
        }

        void removeBody(int id) {
            auto it = bodies.find(id);
            if (it != bodies.end()) {
                if (it->second.baked) {
                    resetBody(id);
                }
                bodies.erase(it);
            }
        }

        void moveBody(int id, const PointMedium& newDir) {
            auto it = bodies.find(id);
            if (it != bodies.end()) {
                bool wasBaked = it->second.baked;
                if (wasBaked) resetBody(id);
                
                it = bodies.find(id);
                it->second.direction = newDir.normalized();
                
                if (wasBaked) bakeBody(id);
            }
        }

        void bakeBody(int id) {
            auto it = bodies.find(id);
            if (it == bodies.end() || it->second.baked) return;

            if (skybox.getCompressionType() != frame::compresstype::RAW) {
                skybox.decompress();
            }

            size_t w = skybox.getWidth();
            size_t h = skybox.getHeight();
            std::vector<uint8_t> newColor = getFormattedColor(it->second.r, it->second.g, it->second.b, it->second.emittance);
            
            it->second.backup.clear();

            for (size_t y = 0; y < h; ++y) {
                float v = (static_cast<float>(y) + 0.5f) / h; 
                for (size_t x = 0; x < w; ++x) {
                    float u = (static_cast<float>(x) + 0.5f) / w;
                    PointMedium pixelDir = uvToDir(u, v);
                    
                    if (pixelDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                        CelestialBody::PixelBackup backup;
                        backup.x = x;
                        backup.y = y;
                        backup.data = skybox.getPixel(x, y);
                        it->second.backup.push_back(std::move(backup));
                        
                        skybox.setPixel(x, y, newColor);
                    }
                }
            }
            it->second.baked = true;
        }

        void resetBody(int id) {
            auto it = bodies.find(id);
            if (it == bodies.end() || !it->second.baked) return;

            if (skybox.getCompressionType() != frame::compresstype::RAW) {
                skybox.decompress();
            }

            for (const auto& backup : it->second.backup) {
                skybox.setPixel(backup.x, backup.y, backup.data);
            }
            
            it->second.backup.clear();
            it->second.backup.shrink_to_fit();
            it->second.baked = false;
        }
    };

private:
    std::unique_ptr<OctreeNode<T, IndexSize, high, medium, low>> root_;
    size_t maxDepth = 16;
    size_t size = 0;
    size_t maxPointsPerNode = 8;
    float lodFalloffRate = 0.01f;
    float lodMinDistance = 1000.0f;
    float maxDistance = lodMinDistance * 10;
    
    std::map<IndexSize, Eigen::Vector3f> colorMap;
    std::shared_mutex colormutex;
    std::map<IndexSize, Material> materialMap;
    std::shared_mutex materialmutex;
    Material globalMat;
    Eigen::Vector3f globalColor;
    float globalIntensity;
    std::filesystem::path storageBasePath;
    Skybox skybox;

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

    void accumulateUsage(const OctreeNode<T, IndexSize, high, medium, low>* node, 
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
    void checkColorUsage(const OctreeNode<T, IndexSize, high, medium, low>* node, std::unordered_set<IndexSize>& usedColors) const {
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
            optimizeMaterialMap();
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
        saveSkybox();
    }

    void loadAll(bool loadPayloadsIntoMemory = false) {
        if (storageBasePath.empty()) return;
        
        loadMaps(storageBasePath / "maps.bin");
        
        std::ifstream is(storageBasePath / "tree_struct.bin", std::ios::binary);
        if (is) {
            root_ = std::make_unique<OctreeNode<T, IndexSize, high, medium, low>>(PointHigh::Zero(), PointHigh::Zero());
            root_->loadStructure(is);
            is.close();
            if (loadPayloadsIntoMemory) {
                root_->loadData(storageBasePath);
            }
        }
        loadSkybox();
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

    bool set(const PointHigh& pos, const T& data, float ptSize = 0.01f, 
             const Eigen::Vector3f& color = Eigen::Vector3f(1,1,1), 
             const Material& mat = Material(), bool visible = true, 
             bool active = true, int objectId = -1, bool staticNode = false,
             const Eigen::Quaternion<low>& orientation = Eigen::Quaternion<low>::Identity()) {
        if (!root_) return false;
        IndexSize cIdx = getColorIndex(color);
        IndexSize mIdx = getMaterialIndex(mat);
        
        auto node = std::make_shared<NodeData<T, IndexSize, high, medium, low>>(
            data, PointLow::Zero(), visible, cIdx, ptSize, active, objectId, mIdx, staticNode);
        node->setOrientation(orientation);
        
        PointHigh halfSize = PointHigh::Constant(ptSize * 0.5);
        root_->insert(node, pos, halfSize, maxPointsPerNode, maxDepth, 0);
        this->size++;
        return true;
    }

    bool update(const PointHigh& pos, const T& newData) {
        auto pt = find(pos);
        if (pt) {
            pt->setData(newData);
            return true;
        }
        return false;
    }

    bool update(const PointHigh& pos, const T& newData, float ptSize, 
                const Eigen::Vector3f& color, const Material& mat, 
                bool visible, bool active, int objectId, bool staticNode,
                const Eigen::Quaternion<low>& orientation) {
        auto pt = find(pos);
        if (pt) {
            pt->setData(newData);
            pt->setColorIdx(getColorIndex(color));
            pt->setMaterialIDX(getMaterialIndex(mat));
            pt->setVisible(visible);
            pt->setActive(active);
            pt->setObjectId(objectId);
            pt->setStatic(staticNode);
            pt->setOrientation(orientation);
            
            if (std::abs(pt->getHalfSize().x() * 2.0f - ptSize) > 1e-5) {
                pt->setSize(ptSize);
                remove(pos);
                PointHigh halfSize = PointHigh::Constant(ptSize * 0.5);
                root_->insert(pt, pos, halfSize, maxPointsPerNode, maxDepth, 0);
                this->size++;
            }
            return true;
        }
        return false;
    }

    bool setDebugColor(const PointHigh& pos, const Eigen::Vector3f& color) {
        auto pt = find(pos);
        if (pt) {
            pt->setColorIdx(getColorIndex(color));
            return true;
        }
        return false;
    }

    bool setColor(const PointHigh& pos, const Eigen::Vector3f& color, const Material& mat) {
        auto pt = find(pos);
        if (pt) {
            pt->setColorIdx(getColorIndex(color));
            pt->setMaterialIDX(getMaterialIndex(mat));
            return true;
        }
        return false;
    }

    bool setColor(const PointHigh& pos, const Eigen::Vector3f& color) {
        auto pt = find(pos);
        if (pt) {
            pt->setColorIdx(getColorIndex(color));
            return true;
        }
        return false;
    }

    bool setMaterial(const PointHigh& pos, const Material& mat) {
        auto pt = find(pos);
        if (pt) {
            pt->setMaterialIDX(getMaterialIndex(mat));
            return true;
        }
        return false;
    }

    bool move(const PointHigh& oldPos, const PointHigh& newPos) {
        if (!root_) return false;
        auto pt = root_->remove(oldPos);
        if (pt) {
            PointHigh halfSize = pt->getHalfSize().template cast<high>();
            root_->insert(pt, newPos, halfSize, maxPointsPerNode, maxDepth, 0);
            return true;
        }
        return false;
    }

    bool remove(const PointHigh& pos, int objectId = -1) {
        if (!root_) return false;
        auto pt = root_->remove(pos, objectId);
        if (pt) {
            this->size--;
            return true;
        }
        return false;
    }

    bool setVisible(const PointHigh& pos, bool visible) {
        auto pt = find(pos);
        if (pt) {
            pt->setVisible(visible);
            return true;
        }
        return false;
    }

    bool setActive(const PointHigh& pos, bool active) {
        auto pt = find(pos);
        if (pt) {
            pt->setActive(active);
            return true;
        }
        return false;
    }
    
    bool resize(const PointHigh& pos, float newSize) {
        if (!root_) return false;
        auto pt = root_->remove(pos);
        if (pt) {
            pt->setSize(newSize);
            PointHigh halfSize = PointHigh::Constant(newSize * 0.5);
            root_->insert(pt, pos, halfSize, maxPointsPerNode, maxDepth, 0);
            return true;
        }
        return false;
    }

    bool rotate(const PointHigh& pos, const Eigen::Quaternion<low>& newRot) {
        auto pt = find(pos);
        if (pt) {
            pt->setOrientation(newRot);
            return true;
        }
        return false;
    }

    std::shared_ptr<NodeData<T, IndexSize, high, medium, low>> find(const PointHigh& pos) const {
        if (!root_) return nullptr;
        return root_->find(pos);
    }

    std::shared_ptr<NodeData<T, IndexSize, high, medium, low>> find(const PointHigh& pos, int objectId) const {
        if (!root_) return nullptr;
        return root_->find(pos, objectId);
    }

    std::vector<std::shared_ptr<NodeData<T, IndexSize, high, medium, low>>> findradius(const PointHigh& center, high radius) const {
        std::vector<std::shared_ptr<NodeData<T, IndexSize, high, medium, low>>> result;
        if (root_) root_->findRadius(center, radius, result);
        return result;
    }

    std::vector<std::shared_ptr<NodeData<T, IndexSize, high, medium, low>>> findObject(int objectId) const {
        std::vector<std::shared_ptr<NodeData<T, IndexSize, high, medium, low>>> result;
        if (root_) root_->findObject(objectId, result);
        return result;
    }

    bool moveObjectBy(int objectId, const PointHigh& offset) {
        if (!root_) return false;
        std::vector<std::pair<std::shared_ptr<NodeData<T, IndexSize, high, medium, low>>, PointHigh>> pts;
        root_->removeObject(objectId, pts);
        if (pts.empty()) return false;
        
        for (auto& pair : pts) {
            PointHigh newPos = pair.second + offset;
            PointHigh halfSize = pair.first->getHalfSize().template cast<high>();
            root_->insert(pair.first, newPos, halfSize, maxPointsPerNode, maxDepth, 0);
        }
        return true;
    }

    bool saveSkybox() {
        if (storageBasePath.empty()) return false;
        std::filesystem::path p = storageBasePath / "skybox.bin";
        std::ofstream os(p, std::ios::binary);
        if (!os) return false;

        size_t numBodies = skybox.bodies.size();
        os.write(reinterpret_cast<const char*>(&numBodies), sizeof(numBodies));
        for (const auto& kv : skybox.bodies) {
            os.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
            os.write(reinterpret_cast<const char*>(&kv.second.direction), sizeof(PointMedium));
            os.write(reinterpret_cast<const char*>(&kv.second.angularRadius), sizeof(float));
            os.write(reinterpret_cast<const char*>(&kv.second.r), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.g), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.b), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(&kv.second.emittance), sizeof(uint8_t));
        }
        os.write(reinterpret_cast<const char*>(&skybox.skyRotation), sizeof(skybox.skyRotation));
        return true;
    }

    bool loadSkybox() {
        if (storageBasePath.empty()) return false;
        std::filesystem::path p = storageBasePath / "skybox.bin";
        if (!std::filesystem::exists(p)) return false;
        
        std::ifstream is(p, std::ios::binary);
        if (!is) return false;

        size_t numBodies = 0;
        is.read(reinterpret_cast<char*>(&numBodies), sizeof(numBodies));
        skybox.bodies.clear();
        for (size_t i = 0; i < numBodies; i++) {
            int id;
            PointMedium dir;
            float angRad;
            uint8_t r, g, b, e;
            is.read(reinterpret_cast<char*>(&id), sizeof(id));
            is.read(reinterpret_cast<char*>(&dir), sizeof(PointMedium));
            is.read(reinterpret_cast<char*>(&angRad), sizeof(float));
            is.read(reinterpret_cast<char*>(&r), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&g), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&b), sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(&e), sizeof(uint8_t));
            skybox.addBody(id, dir, angRad, r, g, b, e);
        }
        is.read(reinterpret_cast<char*>(&skybox.skyRotation), sizeof(skybox.skyRotation));
        return true;
    }

    bool clear() {
        size = 0;
        colorMap.clear();
        materialMap.clear();
        return true;
    }

    void setLODFalloff(float rate) {
        lodFalloffRate = rate;
    }
    void setLODMinDistance(float dist) {
        lodMinDistance = dist;
    }
    void setMaxDistance(float dist) {
        maxDistance = dist;
    }

    void addSkyboxBody(int id, const PointMedium& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance) {
        skybox.addBody(id, dir, angularRadius, r, g, b, emittance);
    }
    
    void removeSkyboxBody(int id) {
        skybox.removeBody(id);
    }
    
    void moveSkyboxBody(int id, const PointMedium& newDir) {
        skybox.moveBody(id, newDir);
    }
    
    void bakeSkyboxBody(int id) {
        skybox.bakeBody(id);
    }
    
    void setSkyboxBackground(float r, float g, float b, float e) {
        skybox.setBackground(r, g, b, e);
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) return;
        size_t totalNodes = 0, leafNodes = 0, lodGeneratedNodes = 0, maxTreeDepth = 0, actualPoints = 0;
        size_t minPointsInLeaf = std::numeric_limits<size_t>::max();
        size_t maxPointsInLeaf = 0;
        
        auto collectStats = [&](auto& self, const OctreeNode<T, IndexSize, high, medium, low>* node, size_t depth) -> void {
            if (!node) return;
            totalNodes++;
            maxTreeDepth = std::max(maxTreeDepth, depth);
            if (node->isLeaf()) {
                leafNodes++;
                actualPoints += node->points.size();
                minPointsInLeaf = std::min(minPointsInLeaf, node->points.size());
                maxPointsInLeaf = std::max(maxPointsInLeaf, node->points.size());
            }
            if (node->isLodValid()) lodGeneratedNodes++;
            for (int i = 0; i < 8; ++i) {
                self(self, node->children[i].get(), depth + 1);
            }
        };
        
        collectStats(collectStats, root_.get(), 0);
        
        if (minPointsInLeaf == std::numeric_limits<size_t>::max()) minPointsInLeaf = 0;
        double avgPointsPerLeaf = leafNodes > 0 ? static_cast<double>(actualPoints) / leafNodes : 0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode<T, IndexSize, high, medium, low>);
        size_t dataMem = actualPoints * sizeof(NodeData<T, IndexSize, high, medium, low>);
        size_t mapMem = colorMap.size() * (sizeof(IndexSize) + sizeof(Eigen::Vector3f)) + materialMap.size() * (sizeof(IndexSize) + sizeof(Material));

        os << "========================================\n";
        os << "             OCTREE STATS               \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
        os << "  LOD Falloff Rate  : " << lodFalloffRate << "\n";
        os << "  LOD Min Distance  : " << lodMinDistance << "\n";
        os << "Structure:\n";
        os << "  Total Nodes       : " << totalNodes << "\n";
        os << "  Leaf Nodes        : " << leafNodes << "\n";
        os << "  Non-Leaf Nodes    : " << (totalNodes - leafNodes) << "\n";
        os << "  LODs Generated    : " << lodGeneratedNodes << "\n";
        os << "  Tree Height       : " << maxTreeDepth << "\n";
        os << "Data:\n";
        os << "  Total Points      : " << size << " (Tracked) / " << actualPoints << " (Counted)\n";
        os << "  Points/Leaf (Avg) : " << std::fixed << std::setprecision(2) << avgPointsPerLeaf << "\n";
        os << "  Points/Leaf (Min) : " << minPointsInLeaf << "\n";
        os << "  Points/Leaf (Max) : " << maxPointsInLeaf << "\n";
        os << "Maps:\n";
        os << "  Unique Colors     : " << colorMap.size() << "/" << MAX_INDEX << "\n";
        os << "  Unique Materials  : " << materialMap.size() << "/" << MAX_INDEX << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.bounds[0].transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.bounds[1].transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "  Dictionary Maps   : " << (mapMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }
};

#endif