#ifndef GRID2_HPP
#define GRID2_HPP

#include "../../eigen/Eigen/Dense"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp" 
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <random>

#ifdef SSE
#include <immintrin.h>
#endif

constexpr int Dim2 = 2;

template<typename T>
class Grid2 {
public:
    using PointType = Eigen::Matrix<float, Dim2, 1>; // Eigen::Vector2f
    using BoundingBox = std::pair<PointType, PointType>;
    
    // Shape for 2D is usually a Circle or a Square (AABB)
    enum class Shape {
        CIRCLE,
        SQUARE
    };
    
    struct NodeData {
        T data;
        PointType position;
        int objectId;
        bool active;
        bool visible;
        float size; // Radius or half-width
        Eigen::Vector4f color; // RGBA
        
        // Physics properties
        float temperature;
        float conductivity;
        float specific_heat;
        float density;
        float next_temperature; // For double-buffering simulation

        Shape shape;

        NodeData(const T& data, const PointType& pos, bool visible, Eigen::Vector4f color, float size = 1.0f,
                 bool active = true, int objectId = -1, Shape shape = Shape::SQUARE) 
                : data(data), position(pos), objectId(objectId), active(active), visible(visible), 
                color(color), size(size), shape(shape),
                temperature(0.0f), conductivity(1.0f), specific_heat(1.0f), density(1.0f), next_temperature(0.0f) {}
        
        NodeData() : objectId(-1), active(false), visible(false), size(0.0f), 
                     color(0,0,0,0), shape(Shape::SQUARE),
                     temperature(0.0f), conductivity(1.0f), specific_heat(1.0f), density(1.0f), next_temperature(0.0f) {}

        // Helper for Square bounds
        BoundingBox getSquareBounds() const {
            PointType halfSize(size * 0.5f, size * 0.5f);
            return {position - halfSize, position + halfSize};
        }
    };

    struct QuadNode {
        BoundingBox bounds;
        std::vector<std::shared_ptr<NodeData>> points;
        std::array<std::unique_ptr<QuadNode>, 4> children; // 4 quadrants
        PointType center;
        bool isLeaf;

        QuadNode(const PointType& min, const PointType& max) : bounds(min,max), isLeaf(true) {
            for (auto& child : children) {
                child = nullptr;
            }
            center = (bounds.first + bounds.second) * 0.5f;
        }

        bool contains(const PointType& point) const {
            return (point.x() >= bounds.first.x() && point.x() <= bounds.second.x() &&
                    point.y() >= bounds.first.y() && point.y() <= bounds.second.y());
        }
        
        // Check intersection with a query box
        bool intersects(const BoundingBox& other) const {
            return (bounds.first.x() <= other.second.x() && bounds.second.x() >= other.first.x() &&
                    bounds.first.y() <= other.second.y() && bounds.second.y() >= other.first.y());
        }
    };

private:
    std::unique_ptr<QuadNode> root_;
    size_t maxDepth;
    size_t size;
    size_t maxPointsPerNode;
    
    Eigen::Vector4f backgroundColor_ = {0.0f, 0.0f, 0.0f, 0.0f};
    PNoise2 noisegen;

    // Determine quadrant: 0:SW, 1:SE, 2:NW, 3:NE
    uint8_t getQuadrant(const PointType& point, const PointType& center) const {
        uint8_t quad = 0;
        if (point.x() >= center.x()) quad |= 1; // Right
        if (point.y() >= center.y()) quad |= 2; // Top
        return quad;
    }

    BoundingBox createChildBounds(const QuadNode* node, uint8_t quad) const {
        PointType childMin, childMax;
        PointType center = node->center;
        
        // X axis
        if (quad & 1) { // Right
            childMin.x() = center.x();
            childMax.x() = node->bounds.second.x();
        } else { // Left
            childMin.x() = node->bounds.first.x();
            childMax.x() = center.x();
        }

        // Y axis
        if (quad & 2) { // Top
            childMin.y() = center.y();
            childMax.y() = node->bounds.second.y();
        } else { // Bottom
            childMin.y() = node->bounds.first.y();
            childMax.y() = center.y();
        }
        
        return {childMin, childMax};
    }

    void splitNode(QuadNode* node, int depth) {
        if (depth >= maxDepth) return;
        
        for (int i = 0; i < 4; ++i) {
            BoundingBox childBounds = createChildBounds(node, i);
            node->children[i] = std::make_unique<QuadNode>(childBounds.first, childBounds.second);
        }

        for (const auto& pointData : node->points) {
            int quad = getQuadrant(pointData->position, node->center);
            node->children[quad]->points.emplace_back(pointData);
        }

        node->points.clear();
        node->isLeaf = false;

        for (int i = 0; i < 4; ++i) {
            if (node->children[i]->points.size() > maxPointsPerNode) {
                splitNode(node->children[i].get(), depth + 1);
            }
        }
    }

    bool insertRecursive(QuadNode* node, const std::shared_ptr<NodeData>& pointData, int depth) {
        if (!node->contains(pointData->position)) return false;

        if (node->isLeaf) {
            node->points.emplace_back(pointData);
            if (node->points.size() > maxPointsPerNode && depth < maxDepth) {
                splitNode(node, depth);
            }
            return true;
        } else {
            int quad = getQuadrant(pointData->position, node->center);
            if (node->children[quad]) {
                return insertRecursive(node->children[quad].get(), pointData, depth + 1);
            }
        }
        return false;
    }

    // --- Serialization Helpers ---
    template<typename V>
    void writeVal(std::ofstream& out, const V& val) const {
        out.write(reinterpret_cast<const char*>(&val), sizeof(V));
    }

    template<typename V>
    void readVal(std::ifstream& in, V& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(V));
    }

    void writeVec2(std::ofstream& out, const Eigen::Vector2f& vec) const {
        writeVal(out, vec.x());
        writeVal(out, vec.y());
    }

    void readVec2(std::ifstream& in, Eigen::Vector2f& vec) {
        float x, y;
        readVal(in, x); readVal(in, y);
        vec = Eigen::Vector2f(x, y);
    }
    
    void writeVec4(std::ofstream& out, const Eigen::Vector4f& vec) const {
        writeVal(out, vec[0]); writeVal(out, vec[1]); writeVal(out, vec[2]); writeVal(out, vec[3]);
    }

    void readVec4(std::ifstream& in, Eigen::Vector4f& vec) {
        float x, y, z, w;
        readVal(in, x); readVal(in, y); readVal(in, z); readVal(in, w);
        vec = Eigen::Vector4f(x, y, z, w);
    }

    void serializeNode(std::ofstream& out, const QuadNode* node) const {
        writeVal(out, node->isLeaf);

        if (node->isLeaf) {
            size_t pointCount = node->points.size();
            writeVal(out, pointCount);
            for (const auto& pt : node->points) {
                writeVal(out, pt->data);
                writeVec2(out, pt->position);
                writeVal(out, pt->objectId);
                writeVal(out, pt->active);
                writeVal(out, pt->visible);
                writeVal(out, pt->size);
                writeVec4(out, pt->color);
                // Physics
                writeVal(out, pt->temperature);
                writeVal(out, pt->conductivity);
                writeVal(out, pt->specific_heat);
                writeVal(out, pt->density);
                writeVal(out, static_cast<int>(pt->shape));
            }
        } else {
            uint8_t childMask = 0;
            for (int i = 0; i < 4; ++i) {
                if (node->children[i] != nullptr) childMask |= (1 << i);
            }
            writeVal(out, childMask);

            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) serializeNode(out, node->children[i].get());
            }
        }
    }

    void deserializeNode(std::ifstream& in, QuadNode* node) {
        bool isLeaf;
        readVal(in, isLeaf);
        node->isLeaf = isLeaf;

        if (isLeaf) {
            size_t pointCount;
            readVal(in, pointCount);
            node->points.reserve(pointCount);
            
            for (size_t i = 0; i < pointCount; ++i) {
                auto pt = std::make_shared<NodeData>();
                readVal(in, pt->data);
                readVec2(in, pt->position);
                readVal(in, pt->objectId);
                readVal(in, pt->active);
                readVal(in, pt->visible);
                readVal(in, pt->size);
                readVec4(in, pt->color);
                readVal(in, pt->temperature);
                readVal(in, pt->conductivity);
                readVal(in, pt->specific_heat);
                readVal(in, pt->density);
                int shapeInt;
                readVal(in, shapeInt);
                pt->shape = static_cast<Shape>(shapeInt);
                node->points.push_back(pt);
            }
        } else {
            uint8_t childMask;
            readVal(in, childMask);
            PointType center = node->center;

            for (int i = 0; i < 4; ++i) {
                if ((childMask >> i) & 1) {
                    PointType childMin, childMax;
                    // Logic matches createChildBounds
                    bool right = i & 1;
                    bool top = i & 2;
                    
                    childMin.x() = right ? center.x() : node->bounds.first.x();
                    childMax.x() = right ? node->bounds.second.x() : center.x();
                    childMin.y() = top ? center.y() : node->bounds.first.y();
                    childMax.y() = top ? node->bounds.second.y() : center.y();
                    
                    node->children[i] = std::make_unique<QuadNode>(childMin, childMax);
                    deserializeNode(in, node->children[i].get());
                } else {
                    node->children[i] = nullptr;
                }
            }
        }
    }

public:
    Grid2(const PointType& minBound, const PointType& maxBound, size_t maxPointsPerNode=16, size_t maxDepth = 16) :
        root_(std::make_unique<QuadNode>(minBound, maxBound)), maxPointsPerNode(maxPointsPerNode),
        maxDepth(maxDepth), size(0) {}
    
    Grid2() : root_(nullptr), maxPointsPerNode(16), maxDepth(16), size(0) {}

    void setBackgroundColor(const Eigen::Vector4f& color) { 
        backgroundColor_ = color; 
    }

    bool set(const T& data, const PointType& pos, bool visible, Eigen::Vector4f color, float size = 1.0f,
             bool active = true, int objectId = -1, Shape shape = Shape::SQUARE) {
        auto pointData = std::make_shared<NodeData>(data, pos, visible, color, size, active, objectId, shape);
        if (insertRecursive(root_.get(), pointData, 0)) {
            this->size++;
            return true;
        }
        return false;
    }

    // --- Standard Grid Operations ---

    bool save(const std::string& filename) const {
        if (!root_) return false;
        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        uint32_t magic = 0x47524944; // GRID
        writeVal(out, magic);
        writeVal(out, maxDepth);
        writeVal(out, maxPointsPerNode);
        writeVal(out, size);
        writeVec4(out, backgroundColor_);
        writeVec2(out, root_->bounds.first);
        writeVec2(out, root_->bounds.second);

        serializeNode(out, root_.get());
        out.close();
        std::cout << "Successfully saved Grid2 to " << filename << std::endl;
        return true;
    }

    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        uint32_t magic;
        readVal(in, magic);
        if (magic != 0x47524944) {
            std::cerr << "Invalid Grid2 file format" << std::endl;
            return false;
        }

        readVal(in, maxDepth);
        readVal(in, maxPointsPerNode);
        readVal(in, size);
        readVec4(in, backgroundColor_);
        
        PointType minBound, maxBound;
        readVec2(in, minBound);
        readVec2(in, maxBound);

        root_ = std::make_unique<QuadNode>(minBound, maxBound);
        deserializeNode(in, root_.get());
        in.close();
        return true;
    }

    std::shared_ptr<NodeData> find(const PointType& pos, float tolerance = 0.0001f) {
        std::function<std::shared_ptr<NodeData>(QuadNode*)> searchNode = [&](QuadNode* node) -> std::shared_ptr<NodeData> {
            if (!node->contains(pos)) return nullptr;
            
            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    if (!pointData->active) continue;
                    float distSq = (pointData->position - pos).squaredNorm();
                    if (distSq <= tolerance * tolerance) return pointData;
                }
                return nullptr;
            } else {
                int quad = getQuadrant(pos, node->center);
                if (node->children[quad]) return searchNode(node->children[quad].get());
            }
            return nullptr;
        };
        return searchNode(root_.get());
    }

    std::vector<std::shared_ptr<NodeData>> findInRadius(const PointType& center, float radius) const {
        std::vector<std::shared_ptr<NodeData>> results;
        if (!root_) return results;
        float radiusSq = radius * radius;
        BoundingBox queryBox(center - PointType(radius, radius), center + PointType(radius, radius));

        std::function<void(QuadNode*)> searchNode = [&](QuadNode* node) {
            if (!node->intersects(queryBox)) return;

            if (node->isLeaf) {
                for (const auto& pointData : node->points) {
                    if (!pointData->active) continue;
                    if ((pointData->position - center).squaredNorm() <= radiusSq) {
                        results.emplace_back(pointData);
                    }
                }
            } else {
                for (const auto& child : node->children) {
                    if (child) searchNode(child.get());
                }
            }
        };
        searchNode(root_.get());
        return results;
    }

    // --- Noise Generation Features ---

    Grid2& noiseGenGrid(float minX, float minY, float maxX, float maxY, float minChance = 0.1f, 
                        float maxChance = 1.0f, bool color = true, int noiseSeed = 42, float densityScale = 1.0f) {
        TIME_FUNCTION;
        noisegen = PNoise2(noiseSeed);
        std::cout << "Generating noise grid (" << minX << "," << minY << ") to (" << maxX << "," << maxY << ")" << std::endl;

        // Iterate through integer coordinates (or stepped float coords based on density)
        // Adjust step size based on grid density scaling
        float step = 1.0f / densityScale;

        for (float x = minX; x < maxX; x += step) {
            for (float y = minY; y < maxY; y += step) {
                // Normalize for noise input
                float nx = (x + noiseSeed) / (maxX + 0.00001f) * 10.0f;
                float ny = (y + noiseSeed) / (maxY + 0.00001f) * 10.0f;
                
                // PNoise2 usually takes a struct or Vec2. We'll reconstruct one here 
                // or assume PNoise2 has been updated to take floats or Eigen.
                // Assuming PNoise2::permute takes Vec2(x,y) where Vec2 is from legacy or adapter.
                // Here we pass a legacy-compatible Vec2 just in case, or floats if adapter exists.
                // For this implementation, we assume we can pass floats to a helper or construct a temporary.
                float alpha = noisegen.permute(Vec2(nx, ny)); // Using external Vec2 for compatibility with PNoise2 header
                
                if (alpha > minChance && alpha < maxChance) {
                    PointType pos(x, y);
                    Eigen::Vector4f col;
                    float temp = 0.0f;

                    if (color) {
                        float r = noisegen.permute(Vec2(nx * 0.3f, ny * 0.3f));
                        float g = noisegen.permute(Vec2(nx * 0.6f, ny * 0.06f));
                        float b = noisegen.permute(Vec2(nx * 0.9f, ny * 0.9f));
                        col = Eigen::Vector4f(r, g, b, 1.0f);
                        temp = noisegen.permute(Vec2(nx * 0.2f + 1, ny * 0.1f + 2));
                    } else {
                        col = Eigen::Vector4f(alpha, alpha, alpha, 1.0f);
                        temp = alpha;
                    }

                    // Create node
                    auto pointData = std::make_shared<NodeData>(T(), pos, true, col, step, true, -1, Shape::SQUARE);
                    pointData->temperature = temp * 100.0f; // Scale temp
                    insertRecursive(root_.get(), pointData, 0);
                    this->size++;
                }
            }
        }
        return *this;
    }

    // --- Thermal Simulation ---

    void setMaterialProperties(const PointType& pos, float cond, float sh, float dens) {
        auto node = find(pos);
        if (node) {
            node->conductivity = cond;
            node->specific_heat = sh;
            node->density = dens;
        }
    }

    void setTemp(const PointType& pos, float temp) {
        auto node = find(pos);
        if (node) {
            node->temperature = temp;
        } else {
            // Create invisible thermal point if it doesn't exist?
            // For now, only set if exists to match sparse grid logic
        }
    }

    // Standard Heat Diffusion (Explicit Euler)
    void diffuseTemps(float deltaTime, float neighborRadius = 1.5f) {
        TIME_FUNCTION;
        if (!root_) return;

        // 1. Collect all active nodes (could optimize by threading traversing)
        std::vector<NodeData*> activeNodes;
        std::function<void(QuadNode*)> collect = [&](QuadNode* node) {
            if (node->isLeaf) {
                for (auto& pt : node->points) {
                    if (pt->active) activeNodes.push_back(pt.get());
                }
            } else {
                for (auto& child : node->children) {
                    if (child) collect(child.get());
                }
            }
        };
        collect(root_.get());

        // 2. Calculate Laplacian / Heat flow
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < activeNodes.size(); ++i) {
            NodeData* curr = activeNodes[i];
            
            // Find neighbors
            auto neighbors = findInRadius(curr->position, neighborRadius);
            
            float laplacian = 0.0f;
            float totalWeight = 0.0f;
            
            for (const auto& nb : neighbors) {
                if (nb.get() == curr) continue;
                
                float distSq = (nb->position - curr->position).squaredNorm();
                float dist = std::sqrt(distSq);
                
                // Simple weight based on distance (1/r)
                if (dist > 0.0001f) {
                    float weight = 1.0f / dist;
                    // Heat transfer: k * (T_neighbor - T_current)
                    laplacian += weight * (nb->temperature - curr->temperature);
                    totalWeight += weight;
                }
            }

            // Normalizing factor (simplified physics model for grid)
            // Diffusivity alpha = k / (rho * cp)
            float alpha = curr->conductivity / (curr->density * curr->specific_heat);
            
            // dT/dt = alpha * Laplacian
            // Scaling Laplacian by arbitrary grid constant or 1/area approx
            float change = 0.0f;
            if (totalWeight > 0) {
                 change = alpha * laplacian * deltaTime;
            }
            
            curr->next_temperature = curr->temperature + change;
        }

        // 3. Apply updates
        #pragma omp parallel for
        for (size_t i = 0; i < activeNodes.size(); ++i) {
            activeNodes[i]->temperature = activeNodes[i]->next_temperature;
        }
    }

    // --- Rendering ---
    
    // Rasterize the quadtree onto a 2D frame buffer
    frame renderFrame(const PointType& minView, const PointType& maxView, 
                      int width, int height, 
                      frame::colormap colorformat = frame::colormap::RGBA) {
        TIME_FUNCTION;
        
        frame outFrame(width, height, colorformat);
        std::vector<uint8_t> buffer;
        int channels = (colorformat == frame::colormap::RGBA || colorformat == frame::colormap::BGRA) ? 4 : 3;
        if (colorformat == frame::colormap::B) channels = 1;

        buffer.resize(width * height * channels, 0);
        
        // Fill background
        // Optimizing background fill requires iterating pixels, doing it implicitly during traversal is harder for gaps.
        // So we fill first.
        #pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            int idx = i * channels;
            if (channels == 4) {
                buffer[idx] = static_cast<uint8_t>(backgroundColor_[0] * 255);
                buffer[idx+1] = static_cast<uint8_t>(backgroundColor_[1] * 255);
                buffer[idx+2] = static_cast<uint8_t>(backgroundColor_[2] * 255);
                buffer[idx+3] = static_cast<uint8_t>(backgroundColor_[3] * 255);
            } else if (channels == 3) {
                buffer[idx] = static_cast<uint8_t>(backgroundColor_[0] * 255);
                buffer[idx+1] = static_cast<uint8_t>(backgroundColor_[1] * 255);
                buffer[idx+2] = static_cast<uint8_t>(backgroundColor_[2] * 255);
            }
        }

        Eigen::Vector2f viewSize = maxView - minView;
        Eigen::Vector2f scale(width / viewSize.x(), height / viewSize.y());

        // Recursive render function
        // We traverse nodes that overlap the view. If leaf, we project points to pixels.
        // Painter's algorithm isn't strictly necessary for 2D unless overlapping, 
        // but sorting isn't implemented here. 
        std::function<void(QuadNode*)> renderNode = [&](QuadNode* node) {
            // Cull if node is outside view
            if (node->bounds.second.x() < minView.x() || node->bounds.first.x() > maxView.x() ||
                node->bounds.second.y() < minView.y() || node->bounds.first.y() > maxView.y()) {
                return;
            }

            if (node->isLeaf) {
                for (const auto& pt : node->points) {
                    if (!pt->visible) continue;

                    // Project world to screen
                    float sx = (pt->position.x() - minView.x()) * scale.x();
                    float sy = (height - 1) - (pt->position.y() - minView.y()) * scale.y(); // Flip Y for image coords

                    int px = static_cast<int>(sx);
                    int py = static_cast<int>(sy);
                    
                    // Size in pixels
                    int pSize = static_cast<int>(pt->size * scale.x());
                    if (pSize < 1) pSize = 1;

                    // Simple Splatting
                    for (int dy = -pSize/2; dy <= pSize/2; ++dy) {
                        for (int dx = -pSize/2; dx <= pSize/2; ++dx) {
                            int nx = px + dx;
                            int ny = py + dy;

                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int idx = (ny * width + nx) * channels;
                                
                                // Blend logic (Alpha Blending)
                                float alpha = pt->color[3];
                                float invAlpha = 1.0f - alpha;

                                if (colorformat == frame::colormap::RGBA) {
                                    buffer[idx]   = static_cast<uint8_t>(pt->color[0] * 255 * alpha + buffer[idx] * invAlpha);
                                    buffer[idx+1] = static_cast<uint8_t>(pt->color[1] * 255 * alpha + buffer[idx+1] * invAlpha);
                                    buffer[idx+2] = static_cast<uint8_t>(pt->color[2] * 255 * alpha + buffer[idx+2] * invAlpha);
                                    buffer[idx+3] = 255;
                                } else if (colorformat == frame::colormap::RGB) {
                                    buffer[idx]   = static_cast<uint8_t>(pt->color[0] * 255 * alpha + buffer[idx] * invAlpha);
                                    buffer[idx+1] = static_cast<uint8_t>(pt->color[1] * 255 * alpha + buffer[idx+1] * invAlpha);
                                    buffer[idx+2] = static_cast<uint8_t>(pt->color[2] * 255 * alpha + buffer[idx+2] * invAlpha);
                                }
                            }
                        }
                    }
                }
            } else {
                for (auto& child : node->children) {
                    if (child) renderNode(child.get());
                }
            }
        };

        renderNode(root_.get());

        outFrame.setData(buffer);
        return outFrame;
    }

    // Heatmap rendering
    frame renderTempFrame(const PointType& minView, const PointType& maxView, 
                          int width, int height, 
                          float minTemp = 0.0f, float maxTemp = 100.0f) {
        // Reuse render logic but override color with heatmap gradient
        // Simplification: We iterate active nodes, map temp to color, then call similar render logic
        // For brevity, a dedicated loop over active points:
        
        frame outFrame(width, height, frame::colormap::RGB);
        std::vector<uint8_t> buffer(width * height * 3, 0); // Black background
        
        Eigen::Vector2f scale(width / (maxView.x() - minView.x()), height / (maxView.y() - minView.y()));

        auto activePts = findInRadius((minView + maxView) * 0.5f, (maxView - minView).norm()); // Broad phase

        for(const auto& pt : activePts) {
            float sx = (pt->position.x() - minView.x()) * scale.x();
            float sy = (height - 1) - (pt->position.y() - minView.y()) * scale.y();
            
            int px = static_cast<int>(sx);
            int py = static_cast<int>(sy);
            
            if (px >= 0 && px < width && py >= 0 && py < height) {
                float tNorm = (pt->temperature - minTemp) / (maxTemp - minTemp);
                tNorm = std::clamp(tNorm, 0.0f, 1.0f);
                
                // Simple Blue -> Red heatmap
                uint8_t r = static_cast<uint8_t>(tNorm * 255);
                uint8_t b = static_cast<uint8_t>((1.0f - tNorm) * 255);
                uint8_t g = 0; 

                int idx = (py * width + px) * 3;
                buffer[idx] = r;
                buffer[idx+1] = g;
                buffer[idx+2] = b;
            }
        }
        
        outFrame.setData(buffer);
        return outFrame;
    }

    void clear() {
        if (!root_) return;
        PointType min = root_->bounds.first;
        PointType max = root_->bounds.second;
        root_ = std::make_unique<QuadNode>(min, max);
        size = 0;
    }

    size_t getSize() const { return size; }
};

#endif