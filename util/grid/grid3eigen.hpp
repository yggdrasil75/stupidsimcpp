#ifndef g3eigen
#define g3eigen

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "camera.hpp"
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>

#ifdef SSE
#include <immintrin.h>
#endif

constexpr int Dim = 3;
constexpr int maxBounces = 4;

template<typename T>
class Octree {
public:
    using PointType = Eigen::Matrix<float, Dim, 1>;
    using BoundingBox = std::pair<PointType, PointType>;
    struct NodeData {
        T data;
        PointType position;
        bool active;
        bool visible;
        float size;
        Eigen::Vector3f color;
        bool light;
        float emittance;
        float refraction; 
        float reflection;

        NodeData(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size = 0.01f,
                 bool active = true, bool light = false, float emittance = 0.0f, float refraction = 0.0f, 
                 float reflection = 0.0f) : data(data), position(pos), active(active), visible(visible), 
                 color(color), size(size), light(light), emittance(emittance), refraction(refraction), 
                 reflection(reflection) {}
    };

    struct OctreeNode {
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::unique_ptr<OctreeNode>, 8> children;
        PointType center;
        bool isLeaf;

        OctreeNode(const PointType& min, const PointType& max) : bounds(min,max), isLeaf(true) {
            for (std::unique_ptr<OctreeNode>& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5;
        }

        bool contains(const PointType& point) const {
            for (int i = 0; i < Dim; ++i) {
                if (point[i] < bounds.first[i] || point[i] > bounds.second[i]) {
                    return false;
                }
            }
            return true;
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

    uint8_t getOctant(const PointType& point, const PointType& center) const {
        int octant = 0;
        for (int i = 0; i < Dim; ++i) {
            if (point[i] >= center[i]) octant |= (1 << i);
        }
        return octant;
    }

    BoundingBox createChildBounds(const OctreeNode* node, uint8_t octant) const {
        PointType childMin, childMax;
        PointType center = node->center;
        for (int i = 0; i < Dim; ++i) {
            bool high = (octant >> i) & 1;
            childMin[i] = high ? center[i] : node->bounds.first[i];
            childMax[i] = high ? node->bounds.second[i] : center[i];
        }
        return {childMin, childMax};
    }

    void splitNode(OctreeNode* node, int depth) {
        if (depth >= maxDepth) return;
        for (int i = 0; i < 8; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<OctreeNode>(childBounds.first, childBounds.second);
        }

        for (const auto& pointData : node->points) {
            int octant = getOctant(pointData->position, node->center);
            node->children[octant]->points.emplace_back(pointData);
        }

        node->points.clear();
        node->isLeaf = false;

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(OctreeNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        if (!node->contains(pointData->position)) return false;

        if (node->isLeaf) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            return true;
        } else {
            int octant = getOctant(pointData->position, node->center);
            if (node->children[octant]) {
                return insertRecursive(node->children[octant].get(), pointData, depth + 1);
            }
        }
        return false;
    }

    void bitonic_sort_8(std::array<std::pair<int, float>, 8>& arr) const noexcept {
        #ifdef SSE
            alignas(32) float values[8];
            alignas(32) uint32_t indices[8];
            
            for (int i = 0; i < 8; i++) {
                values[i] = arr[i].second;
                indices[i] = arr[i].first;
            }
            
            __m256 val = _mm256_load_ps(values);
            __m256i idx = _mm256_load_si256((__m256i*)indices);
            
            __m256 swapped1 = _mm256_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
            __m256i swapped_idx1 = _mm256_shuffle_epi32(idx, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 mask1 = _mm256_cmp_ps(val, swapped1, _CMP_GT_OQ);
            val = _mm256_blendv_ps(val, swapped1, mask1);
            idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(idx),
                _mm256_castsi256_ps(swapped_idx1),
                mask1));
            
            __m256 swapped2 = _mm256_permute2f128_ps(val, val, 0x01);
            __m256i swapped_idx2 = _mm256_permute2f128_si256(idx, idx, 0x01);
            __m256 mask2 = _mm256_cmp_ps(val, swapped2, _CMP_GT_OQ);
            val = _mm256_blendv_ps(val, swapped2, mask2);
            idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(idx),
                _mm256_castsi256_ps(swapped_idx2),
                mask2));
            
            __m256 swapped3 = _mm256_shuffle_ps(val, val, _MM_SHUFFLE(1, 0, 3, 2));
            __m256i swapped_idx3 = _mm256_shuffle_epi32(idx, _MM_SHUFFLE(1, 0, 3, 2));
            __m256 mask3 = _mm256_cmp_ps(val, swapped3, _CMP_GT_OQ);
            val = _mm256_blendv_ps(val, swapped3, mask3);
            idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(idx),
                _mm256_castsi256_ps(swapped_idx3),
                mask3));
            
            __m256 swapped4 = _mm256_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
            __m256i swapped_idx4 = _mm256_shuffle_epi32(idx, _MM_SHUFFLE(2, 3, 0, 1));
            __m256 mask4 = _mm256_cmp_ps(val, swapped4, _CMP_GT_OQ);
            val = _mm256_blendv_ps(val, swapped4, mask4);
            idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(idx),
                _mm256_castsi256_ps(swapped_idx4),
                mask4));
            
            _mm256_store_ps(values, val);
            _mm256_store_si256((__m256i*)indices, idx);
            
            for (int i = 0; i < 8; i++) {
                arr[i].second = values[i];
                arr[i].first = (uint8_t)indices[i];
            }
        #else
            auto a0 = arr[0], a1 = arr[1], a2 = arr[2], a3 = arr[3];
            auto a4 = arr[4], a5 = arr[5], a6 = arr[6], a7 = arr[7];
            
            if (a0.second > a1.second) std::swap(a0, a1);
            if (a2.second < a3.second) std::swap(a2, a3);
            if (a4.second > a5.second) std::swap(a4, a5);
            if (a6.second < a7.second) std::swap(a6, a7);
            
            if (a0.second > a2.second) std::swap(a0, a2);
            if (a1.second > a3.second) std::swap(a1, a3);
            if (a0.second > a1.second) std::swap(a0, a1);
            if (a2.second > a3.second) std::swap(a2, a3);
            
            if (a4.second < a6.second) std::swap(a4, a6);
            if (a5.second < a7.second) std::swap(a5, a7);
            if (a4.second < a5.second) std::swap(a4, a5);
            if (a6.second < a7.second) std::swap(a6, a7);
            
            if (a0.second > a4.second) std::swap(a0, a4);
            if (a1.second > a5.second) std::swap(a1, a5);
            if (a2.second > a6.second) std::swap(a2, a6);
            if (a3.second > a7.second) std::swap(a3, a7);
            
            if (a0.second > a2.second) std::swap(a0, a2);
            if (a1.second > a3.second) std::swap(a1, a3);
            if (a4.second > a6.second) std::swap(a4, a6);
            if (a5.second > a7.second) std::swap(a5, a7);
            
            if (a0.second > a1.second) std::swap(a0, a1);
            if (a2.second > a3.second) std::swap(a2, a3);
            if (a4.second > a5.second) std::swap(a4, a5);
            if (a6.second > a7.second) std::swap(a6, a7);
            
            arr[0] = a0; arr[1] = a1; arr[2] = a2; arr[3] = a3;
            arr[4] = a4; arr[5] = a5; arr[6] = a6; arr[7] = a7;
        #endif
    }

    bool rayBoxIntersect(const PointType& origin, const PointType& dir, const BoundingBox& box,
                            float& tMin, float& tMax) const {
        tMin = 0.0f;
        tMax = std::numeric_limits<float>::max();

        for (int i = 0; i < Dim; ++i) {
            if (std::abs(dir[i]) < std::numeric_limits<float>::epsilon()) {
                if (origin[i] < box.first[i] || origin[i] > box.second[i]) return false;
            } else {
                float invD = 1.f / dir[i];
                float t1 = (box.first[i] - origin[i]) * invD;
                float t2 = (box.second[i] - origin[i]) * invD;
                
                if (t1 > t2) std::swap(t1, t2);
                tMin = std::max(tMin, t1);
                tMax = std::min(tMax, t2);
                if (tMin > tMax) return false;
            }
        }
        return true;
    }

    float randomValueNormalDistribution(uint32_t& state) {
        std::mt19937 gen(state);
        state = gen();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float θ = 2 * M_PI * dist(gen);
        float ρ = sqrt(-2 * log(dist(gen)));
        return ρ * cos(θ);
    }

    PointType randomInHemisphere(const PointType& normal, uint32_t& state) {
        float x = randomValueNormalDistribution(state);
        float y = randomValueNormalDistribution(state);
        float z = randomValueNormalDistribution(state);
        return PointType(x,y,z).normalized();
    }

    float rgbToGrayscale(const Eigen::Vector3f& color) const {
        return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
    }
public:
    Octree(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=16, size_t maxDepth = 16) :
        root_(std::make_unique<OctreeNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
        maxDepth(maxDepth), size(0) {}
    
    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector3f color, float size, bool active,
             bool light = false, float emittance = 0.0f, float refraction = 0.0f, float reflection = 0.0f) {
        auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, 
                                                    light, emittance, refraction, reflection);
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        }
        return false;
    }

    std::vector<std::shared_ptr<NodeData>> voxelTraverse(const PointType& origin, const PointType& direction,
                                         float maxDist, bool stopAtFirstHit) {
        std::vector<std::shared_ptr<NodeData>> hits;
        
        if (empty()) return hits;

        std::function<void(OctreeNode*, const PointType&, const PointType&, float, float)> traverseNode = 
                        [&](OctreeNode* node, const PointType& origin, const PointType& dir, float tMin, float tMax) {
            if (!node || tMin > tMax) return;
            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    PointType toPoint = pointData->position - origin;
                    float projection = toPoint.dot(dir);
                    if (projection >= 0 && projection <= maxDist) {
                        PointType closestPoint = origin + dir * projection;
                        float distSq = (pointData->position - closestPoint).squaredNorm();
                        if (distSq < pointData->size * pointData->size) {
                            hits.emplace_back(pointData);
                            if (stopAtFirstHit) return;
                        }
                    }
                }
            } else {
                std::array<std::pair<int, float>, 8> childOrder;
                PointType center = node->center;
                for (int i = 0; i < 8; ++i) {
                    if (node->children[i]) {
                        PointType childCenter = node->children[i]->center;
                        float dist = (childCenter - origin).dot(dir);
                        childOrder[i] = {i, dist};
                    } else {
                        childOrder[i] = {i, std::numeric_limits<float>::lowest()};
                    }
                }

                bitonic_sort_8(childOrder);

                for (int i = 0; i < 8; ++i) {
                    int childIdx = childOrder[i].first;
                    if (node->children[childIdx]) {
                        const auto& childBounds = node->children[childIdx]->bounds;
                        
                        float childtMin = tMin;
                        float childtMax = tMax;
                        if (rayBoxIntersect(origin, dir, childBounds, childtMin, childtMax)) {
                            traverseNode(node->children[childIdx].get(), origin, dir, childtMin, childtMax);
                            if (stopAtFirstHit && !hits.empty()) return;
                        }
                    }
                }
            }
        };
        float tMin, tMax;
        if (rayBoxIntersect(origin, direction, root_->bounds, tMin, tMax)) {
            tMax = std::min(tMax, maxDist);
            if (tMin <= tMax) {
                traverseNode(root_.get(), origin, direction, tMin, tMax);
            }
        }
        return hits;
    }
    
    frame renderFrame(const Camera& cam, int height, int width, frame::colormap colorformat = frame::colormap::RGB) {
        PointType origin = cam.origin;
        PointType dir = cam.direction.normalized();
        PointType up = cam.up.normalized();
        PointType right = cam.right();
        
        frame outFrame(width, height, colorformat);
        std::vector<uint8_t> colorBuffer;
        int channels;
        if (colorformat == frame::colormap::B) {
            channels = 1;
        } else if (colorformat == frame::colormap::RGB || colorformat == frame::colormap::BGR) {
            channels = 3;
        } else { //BGRA and RGBA
            channels = 4;
        }
        colorBuffer.resize(width * height * channels);

        float aspect = static_cast<float>(width) / height;
        float fovRad = cam.fovRad();
        float tanHalfFov = tan(fovRad * 0.5f);
        float tanfovy = tanHalfFov;
        float tanfovx = tanHalfFov * aspect;

        const Eigen::Vector3f defaultColor(0.01f, 0.01f, 0.01f); 
        float rayLength = std::numeric_limits<float>::max();
        std::function<Eigen::Vector3f(const PointType&, const PointType&, int, uint32_t&)> traceRay = 
            [&](const PointType& rayOrig, const PointType& rayDir, int bounces, uint32_t& rngState) -> Eigen::Vector3f {

            if (bounces > maxBounces) return {0,0,0};

            auto hits = voxelTraverse(rayOrig, rayDir, rayLength, true);
            if (hits.empty() && bounces < 1) {
                return defaultColor; 
            } else if (hits.empty()) {
                return {0,0,0};
            }

            auto obj = hits[0];
            
            PointType center = obj->position;
            float radius = obj->size;
            PointType L_vec = center - rayOrig;
            float tca = L_vec.dot(rayDir);
            float d2 = L_vec.dot(L_vec) - tca * tca;
            float radius2 = radius * radius;

            float t = tca;
            if (d2 <= radius2) {
                float thc = std::sqrt(radius2 - d2);
                t = tca - thc; 
                if (t < 0.001f) t = tca + thc;
            }

            PointType hitPoint = rayOrig + rayDir * t;
            PointType normal = (hitPoint - center).normalized();

            Eigen::Vector3f finalColor = {0,0,0};

            if (obj->light) {
                return obj->color * obj->emittance;
            }

            float refl = obj->reflection;
            float refr = obj->refraction;
            float diffuseProb = 1.0f - refl - refr;

            if (diffuseProb > 0.001f) {
                PointType scatterDir = randomInHemisphere(normal, rngState);
                
                Eigen::Vector3f incomingLight = traceRay(hitPoint + normal * 0.002f, scatterDir, bounces + 1, rngState);
                
                finalColor += obj->color.cwiseProduct(incomingLight) * diffuseProb;
            }

            if (refl > 0.001f) {
                PointType rDir = (rayDir - 2.0f * rayDir.dot(normal) * normal).normalized();
                finalColor += traceRay(hitPoint + normal * 0.002f, rDir, bounces + 1, rngState) * refl;
            }

            if (refr > 0.001f) {
                float ior = 1.45f;
                float η = 1.0f / ior;
                float cosI = -normal.dot(rayDir);
                PointType n_eff = normal;

                if (cosI < 0) {
                    cosI = -cosI;
                    n_eff = -normal;
                    η = ior;
                }

                float k = 1.0f - η * η * (1.0f - cosI * cosI);
                PointType nextDir;
                if (k >= 0.0f) {
                    nextDir = (η * rayDir + (η * cosI - std::sqrt(k)) * n_eff).normalized();
                    finalColor += traceRay(hitPoint - n_eff * 0.002f, nextDir, bounces + 1, rngState) * refr;
                } else {
                    nextDir = (rayDir - 2.0f * rayDir.dot(n_eff) * n_eff).normalized();
                    finalColor += traceRay(hitPoint + n_eff * 0.002f, nextDir, bounces + 1, rngState) * refr;
                }
            }

            return finalColor;
        };
        
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float px = (2.0f * (x + 0.5f) / width - 1.0f) * tanfovx;
                float py = (1.0f - 2.0f * (y + 0.5f) / height) * tanfovy;
                
                PointType rayDir = dir + (right * px) + (up * py);
                rayDir.normalize();
                
                int pidx = (y * width + x);
                uint32_t seed = pidx * 1973 + 9277;
                int idx = pidx * channels;
                
                Eigen::Vector3f color = traceRay(origin, rayDir, 0, seed);
                
                color = color.cwiseMax(0.0f).cwiseMin(1.0f);
                
                
                switch(colorformat) {
                    case frame::colormap::B:
                        colorBuffer[idx    ] = static_cast<uint8_t>(rgbToGrayscale(color) * 255.0f);
                        break;
                    case frame::colormap::RGB:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[2] * 255.0f);
                        break;
                    case frame::colormap::BGR:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[0] * 255.0f);
                        break;
                    case frame::colormap::RGBA:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 3] = 255;
                        break;
                    case frame::colormap::BGRA:
                        colorBuffer[idx    ] = static_cast<uint8_t>(color[2] * 255.0f);
                        colorBuffer[idx + 1] = static_cast<uint8_t>(color[1] * 255.0f);
                        colorBuffer[idx + 2] = static_cast<uint8_t>(color[0] * 255.0f);
                        colorBuffer[idx + 3] = 255;
                        break;
                    
                }
            }
        }
        
        outFrame.setData(colorBuffer);
        return outFrame;
    }

    void printStats(std::ostream& os = std::cout) const {
        if (!root_) {
            os << "[Octree Stats] Tree is null/empty." << std::endl;
            return;
        }

        size_t totalNodes = 0;
        size_t leafNodes = 0;
        size_t actualPoints = 0;
        size_t maxTreeDepth = 0;
        size_t maxPointsInLeaf = 0;
        size_t minPointsInLeaf = std::numeric_limits<size_t>::max();
        
        // Recursive lambda to gather stats
        std::function<void(const OctreeNode*, size_t)> traverse = 
            [&](const OctreeNode* node, size_t depth) {
            if (!node) return;
            
            totalNodes++;
            maxTreeDepth = std::max(maxTreeDepth, depth);

            if (node->isLeaf) {
                leafNodes++;
                size_t pts = node->points.size();
                actualPoints += pts;
                maxPointsInLeaf = std::max(maxPointsInLeaf, pts);
                minPointsInLeaf = std::min(minPointsInLeaf, pts);
            } else {
                for (const auto& child : node->children) {
                    traverse(child.get(), depth + 1);
                }
            }
        };

        traverse(root_.get(), 0);

        if (leafNodes == 0) minPointsInLeaf = 0;
        double avgPointsPerLeaf = leafNodes > 0 ? (double)actualPoints / leafNodes : 0.0;
        
        size_t nodeMem = totalNodes * sizeof(OctreeNode);
        size_t dataMem = actualPoints * (sizeof(NodeData) + 16); 

        os << "========================================\n";
        os << "             OCTREE STATS               \n";
        os << "========================================\n";
        os << "Config:\n";
        os << "  Max Depth Allowed : " << maxDepth << "\n";
        os << "  Max Pts Per Node  : " << maxPointsPerNode << "\n";
        os << "Structure:\n";
        os << "  Total Nodes       : " << totalNodes << "\n";
        os << "  Leaf Nodes        : " << leafNodes << "\n";
        os << "  Non-Leaf Nodes    : " << (totalNodes - leafNodes) << "\n";
        os << "  Tree Height       : " << maxTreeDepth << "\n";
        os << "Data:\n";
        os << "  Total Points      : " << size << " (Tracked) / " << actualPoints << " (Counted)\n";
        os << "  Points/Leaf (Avg) : " << std::fixed << std::setprecision(2) << avgPointsPerLeaf << "\n";
        os << "  Points/Leaf (Min) : " << minPointsInLeaf << "\n";
        os << "  Points/Leaf (Max) : " << maxPointsInLeaf << "\n";
        os << "Bounds:\n";
        os << "  Min               : [" << root_->bounds.first.transpose() << "]\n";
        os << "  Max               : [" << root_->bounds.second.transpose() << "]\n";
        os << "Memory (Approx):\n";
        os << "  Node Structure    : " << (nodeMem / 1024.0) << " KB\n";
        os << "  Point Data        : " << (dataMem / 1024.0) << " KB\n";
        os << "========================================\n" << std::defaultfloat;
    }

    bool empty() const { return size == 0; }
};

#endif
