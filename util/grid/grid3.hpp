#ifndef GRID3_HPP
#define GRID3_HPP

#include <unordered_map>
#include <fstream>
#include <cstring>
#include <memory>
#include <array>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include "../vecmat/mat4.hpp"
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"

//constexpr char magic[4] = {'Y', 'G', 'G', '3'};
static constexpr int CHUNK_THRESHOLD = 16; //at this size, subdivide.

Mat4f lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
    Vec3f const f = (center - eye).normalized();
    Vec3f const s = f.cross(up).normalized();
    Vec3f const u = s.cross(f);

    Mat4f Result = Mat4f::identity();
    Result(0, 0) = s.x;
    Result(1, 0) = s.y;
    Result(2, 0) = s.z;
    Result(3, 0) = -s.dot(eye);
    Result(0, 1) = u.x;
    Result(1, 1) = u.y;
    Result(2, 1) = u.z;
    Result(3, 1) = -u.dot(eye);
    Result(0, 2) = -f.x;
    Result(1, 2) = -f.y;
    Result(2, 2) = -f.z;
    Result(3, 2) = f.dot(eye);
    return Result;
}

Mat4f perspective(float fovy, float aspect, float zNear, float zfar) {
    float const tanhalfF = tan(fovy / 2);
    Mat4f Result = 0;
    Result(0,0) = 1 / (aspect * tanhalfF);
    Result(1,1) = 1 / tanhalfF;
    Result(2,2) = zfar / (zNear - zfar);
    Result(2,3) = -1;
    Result(3,2) = -(zfar * zNear) / (zfar - zNear);
    return Result;
}

struct Voxel {
    float weight = 1.0;
    bool active = false;
    float alpha = 0.0;
    Vec3ui8 color = Vec3ui8(0,0,0);
    Voxel() = default;
    Voxel(float weight, bool active, float alpha, Vec3ui8 color) : weight(weight), active(active), alpha(alpha), color(color) {}
    // TODO: add curving and similar for water and glass and so on.
    auto members() const -> std::tuple<const float&, const bool&, const float&, const Vec3ui8&> {
        return std::tie(weight, active, alpha, color);
    }
    
    auto members() -> std::tuple<float&, bool&, float&, Vec3ui8&> {
        return std::tie(weight, active, alpha, color);
    }
};

struct Camera {
    Ray3f posfor;
    Vec3f up;
    float fov;
    Camera(Vec3f pos, Vec3f viewdir, Vec3f up, float fov = 80) : posfor(Ray3f(pos, viewdir)), up(up), fov(fov) {}
    
    void rotateYaw(float angle) {
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        Vec3f right = posfor.direction.cross(up).normalized();
        posfor.direction = posfor.direction * cosA + right * sinA;
        posfor.direction = posfor.direction.normalized();
    }
    
    void rotatePitch(float angle) {
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        Vec3f right = posfor.direction.cross(up).normalized();
        posfor.direction = posfor.direction * cosA + up * sinA;
        posfor.direction = posfor.direction.normalized();
        
        up = right.cross(posfor.direction).normalized();
    }

    Vec3f forward() const {
        return (posfor.direction - posfor.origin).normalized();
    }

    Vec3f right() const {
        return forward().cross(up).normalized();
    }

    float fovRad() const {
        return fov * (M_PI / 180);
    }
};

struct Vertex {
    Vec3f position;
    Vec3f normal;
    Vec3ui8 color;
    Vec2f texCoord;
    
    Vertex() = default;
    Vertex(Vec3f pos, Vec3f norm, Vec3ui8 colr, Vec2f tex = Vec2f(0,0)) : position(pos), normal(norm), color(colr), texCoord(tex) {}
};

struct Chunk {
    Voxel reprVoxel; //average of all voxels in chunk for LOD rendering
    //std::vector<Voxel> voxels; //list of all voxels in chunk.
    std::vector<Chunk> children; //list of all chunks in chunk
    bool active; //active if any child chunk or child voxel is active. used to efficiently find active voxels by only going down when an active chunk is found.
    int chunkSize; //should be (CHUNK_THRESHOLD/2) * 2 ^ depth I think. (ie: 1 depth will be (16/2)*(2^1) or 16, second will be (16/2)*(2^2) or 8*4=32)
    Vec3i minCorner; //position of chunk in world space.
    int depth; //number of parent/child traversals to get here. 
};

class VoxelGrid {
private:
    Vec3i gridSize;
    std::vector<Voxel> voxels;
    std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunkList;
    std::unordered_map<Vec3i, bool, Vec3i::Hash> activeChunks;
    
    float radians(float rads) {
        return rads * (M_PI / 180);
    }

    Vec3i getChunkCoord(const Vec3i& voxelPos) const {
        return Vec3i(voxelPos.x / CHUNK_THRESHOLD, voxelPos.y / CHUNK_THRESHOLD, voxelPos.z / CHUNK_THRESHOLD);
    }

    void updateChunkStatus(const Vec3i& pos, bool isActive) {
        Vec3i chunkCoord = getChunkCoord(pos);

        if (isActive) {
            chunkList[chunkCoord].active = true;
            activeChunks[chunkCoord] = true;
        }
    }

public:
    double binSize = 1;
    VoxelGrid() : gridSize(0,0,0) {
        std::cout << "creating empty grid." << std::endl;
    }

    VoxelGrid(int w, int h, int d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
    }
    
    bool serializeToFile(const std::string& filename);
    
    static std::unique_ptr<VoxelGrid> deserializeFromFile(const std::string& filename);
    
    Voxel& get(int x, int y, int z) {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    const Voxel& get(int x, int y, int z) const {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    Voxel& get(const Vec3i& xyz) {
        return get(xyz.x, xyz.y, xyz.z);
    }

    const Voxel& get(const Vec3i& xyz) const {
        return get(xyz.x, xyz.y, xyz.z);
    }

    void resize(int newW, int newH, int newD) {
        std::vector<Voxel> newVoxels(newW * newH * newD);
        
        std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunklist;
        std::unordered_map<Vec3i, bool, Vec3i::Hash> newActiveChunks;

        int copyW = std::min(static_cast<int>(gridSize.x), newW);
        int copyH = std::min(static_cast<int>(gridSize.y), newH);
        int copyD = std::min(static_cast<int>(gridSize.z), newD);
        for (int z = 0; z < copyD; ++z) {
            for (int y = 0; y < copyH; ++y) {
                int oldRowStart = z * gridSize.x * gridSize.y + y * gridSize.x;
                int newRowStart = z * newW * newH + y * newW;
                
                std::copy(
                    voxels.begin() + oldRowStart, 
                    voxels.begin() + oldRowStart + copyW, 
                    newVoxels.begin() + newRowStart
                );

                for (int x = 0; x < copyW; ++x) {
                    if (voxels[oldRowStart + x].active) {
                        Vec3i cc(x / CHUNK_THRESHOLD, y / CHUNK_THRESHOLD, z / CHUNK_THRESHOLD);
                        newActiveChunks[cc] = true;
                    }
                }
            }
        }
        voxels = std::move(newVoxels);
        activeChunks = std::move(newActiveChunks);
        gridSize = Vec3i(newW, newH, newD);
    }

    void resize(Vec3i newsize) {
        resize(newsize.x, newsize.y, newsize.z);
    }

    void set(int x, int y, int z, bool active, Vec3ui8 color) {
        set(Vec3i(x,y,z), active, color);
    }

    void set(Vec3i pos, bool active, Vec3ui8 color) {
        if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0) {
            if (!(pos.x < gridSize.x)) {
                resize(pos.x, gridSize.y, gridSize.z);
            }
            else if (!(pos.y < gridSize.y)) {
                resize(gridSize.x, pos.y, gridSize.z);
            }
            else if (!(pos.z < gridSize.z)) {
                resize(gridSize.x, gridSize.y, pos.z);
            }

            Voxel& v = get(pos);
            v.active = active;
            v.color = color;
            updateChunkStatus(pos, active);
        }
    }

    void set(Vec3i pos, Vec4ui8 rgbaval) {
        set(pos, static_cast<float>(rgbaval.a / 255), rgbaval.toVec3());
    }

    template<typename T>
    bool inGrid(Vec3<T> voxl) const {
        return (voxl >= 0 && voxl.x < gridSize.x && voxl.y < gridSize.y && voxl.z < gridSize.z);
    }

    void voxelTraverse(const Vec3d& origin, const Vec3d& end, std::vector<Vec3i>& visitedVoxel, int maxDist = 10000000) const {
        Vec3i cv = (origin / binSize).floorToI();
        Vec3i lv = (end / binSize).floorToI();
        Vec3d ray = end - origin;
        Vec3f step = Vec3f(ray.x >= 0 ? 1 : -1, ray.y >= 0 ? 1 : -1, ray.z >= 0 ? 1 : -1);
        Vec3d nextVox = cv.toDouble() + step * binSize;
        
        Vec3d tMax = Vec3d(ray.x != 0 ? (nextVox.x - origin.x) / ray.x : INF,
                           ray.y != 0 ? (nextVox.y - origin.y) / ray.y : INF,
                           ray.z != 0 ? (nextVox.z - origin.z) / ray.z : INF);
        Vec3d tDelta = Vec3d(ray.x != 0 ? binSize / ray.x * step.x : INF,
                             ray.y != 0 ? binSize / ray.y * step.y : INF,
                             ray.z != 0 ? binSize / ray.z * step.z : INF);

        float dist = 0;
        
        while (lv != cv && visitedVoxel.size() < 10 && dist < maxDist) {
            
            Vec3i currentChunk = getChunkCoord(cv);
            
            auto it = activeChunks.find(currentChunk);
            bool isChunkActive = (it != activeChunks.end() && it->second);

            if (!isChunkActive) {
                while (getChunkCoord(cv) == currentChunk && inGrid(cv)) {
                    Vec3f chunkStep = step * CHUNK_THRESHOLD / 2;
                    Vec3d chunkDelta = tDelta * CHUNK_THRESHOLD / 2;
                    if (tMax.x < tMax.y) {
                        if (tMax.x < tMax.z) {
                            dist += chunkDelta.x;
                            cv.x += chunkStep.x;
                            tMax.x += chunkDelta.x;
                        } else {
                            dist += chunkDelta.z;
                            cv.z += chunkStep.z;
                            tMax.z += chunkDelta.z;
                        }
                    } else {
                        if (tMax.y < tMax.z) {
                            dist += chunkDelta.y;
                            cv.y += chunkStep.y;
                            tMax.y += chunkDelta.y;
                        } else {
                            dist += chunkDelta.z;
                            cv.z += chunkStep.z;
                            tMax.z += chunkDelta.z;
                        }
                    }
                }
                
                continue;
            }
            if (get(cv).active) {
                visitedVoxel.push_back(cv);
            }

            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    dist += tDelta.x;
                    cv.x += step.x;
                    tMax.x += tDelta.x;
                } else {
                    dist += tDelta.z;
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            } else {
                if (tMax.y < tMax.z) {
                    dist += tDelta.y;
                    cv.y += step.y;
                    tMax.y += tDelta.y;
                } else {
                    dist += tDelta.z;
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            }
        }
        return;
    }

    int getWidth() const {
        return gridSize.x;
    }
    int getHeight() const {
        return gridSize.y;
    }
    int getDepth() const {
        return gridSize.z;
    }
    
    frame renderFrame(const Camera& cam, Vec2i resolution, frame::colormap colorformat = frame::colormap::RGB) const {
        TIME_FUNCTION;
        Vec3f forward = cam.forward();
        Vec3f right = cam.right();
        Vec3f upCor = right.cross(forward).normalized();
        float aspect = resolution.aspect();
        float fovRad = cam.fovRad();
        float viewH = 2 * tan(fovRad / 2);
        float viewW = viewH * aspect;
        float maxDist = std::sqrt(gridSize.lengthSquared()) * binSize;
        frame outFrame(resolution.x, resolution.y, frame::colormap::RGB);
        std::vector<uint8_t> colorBuffer(resolution.x * resolution.y * 3);
        #pragma omp parallel for
        for (int y = 0; y < resolution.x; y++) {
            float v = (static_cast<float>(y) / static_cast<float>(resolution.x - 1)) - 0.5f;    
            for (int x = 0; x < resolution.y; x++) {
                std::vector<Vec3i> hitVoxels;
                float u = (static_cast<float>(x) / static_cast<float>(resolution.y - 1)) - 0.5f;
                Vec3f rayDirWorld = (forward + right * (u * viewW) + upCor * (v * viewH)).normalized();
                Vec3f rayEnd = cam.posfor.origin + rayDirWorld * maxDist;
                Vec3d rayStartGrid = cam.posfor.origin.toDouble() / binSize;
                Vec3d rayEndGrid = rayEnd.toDouble() / binSize;
                voxelTraverse(rayStartGrid, rayEndGrid, hitVoxels);
                Vec3ui8 hitColor(10, 10, 255);
                for (const Vec3i& voxelPos : hitVoxels) {
                    if (inGrid(voxelPos)) {
                        const Voxel& voxel = get(voxelPos);
                        if (voxel.active) {
                            hitColor = voxel.color;
                            
                            break;
                        }
                    }
                }
                hitVoxels.clear();
                hitVoxels.shrink_to_fit();
                // Set pixel color in buffer
                switch (colorformat) {
                    case frame::colormap::BGRA: {
                        int idx = (y * resolution.y + x) * 4;
                        colorBuffer[idx + 3] = hitColor.x;
                        colorBuffer[idx + 2] = hitColor.y;
                        colorBuffer[idx + 1] = hitColor.z;
                        colorBuffer[idx + 0] = 255;
                        break;
                    }
                    case frame::colormap::RGB:
                    default: {
                        int idx = (y * resolution.y + x) * 3;
                        colorBuffer[idx + 0] = hitColor.x;
                        colorBuffer[idx + 1] = hitColor.y;
                        colorBuffer[idx + 2] = hitColor.z;
                        break;
                    }
                }
            }
        }
        
        outFrame.setData(colorBuffer);
        return outFrame;
    }

    void printStats() const {
        int totalVoxels = gridSize.x * gridSize.y * gridSize.z;
        int activeVoxels = 0;
        
        // Count active voxels
        for (const Voxel& voxel : voxels) {
            if (voxel.active) {
                activeVoxels++;
            }
        }
        
        float activePercentage = (totalVoxels > 0) ? 
            (static_cast<float>(activeVoxels) / static_cast<float>(totalVoxels)) * 100.0f : 0.0f;
        
        std::cout << "=== Voxel Grid Statistics ===" << std::endl;
        std::cout << "Grid dimensions: " << gridSize.x << " x " << gridSize.y << " x " << gridSize.z << std::endl;
        std::cout << "Total voxels: " << totalVoxels << std::endl;
        std::cout << "Active voxels: " << activeVoxels << std::endl;
        std::cout << "Inactive voxels: " << (totalVoxels - activeVoxels) << std::endl;
        std::cout << "Active chunks (map size): " << activeChunks.size() << std::endl;
        std::cout << "Active percentage: " << activePercentage << "%" << std::endl;
        std::cout << "Memory usage (approx): " << (voxels.size() * sizeof(Voxel)) / 1024 << " KB" << std::endl;
        std::cout << "============================" << std::endl;
    }

private:
    // Helper function to check if a voxel is on the surface
    bool isSurfaceVoxel(int x, int y, int z) const {
        if (!inGrid(Vec3i(x, y, z))) return false;
        if (!get(x, y, z).active) return false;
        
        // Check all 6 neighbors
        static const std::array<Vec3i, 6> neighbors = {{
            Vec3i(1, 0, 0), Vec3i(-1, 0, 0),
            Vec3i(0, 1, 0), Vec3i(0, -1, 0),
            Vec3i(0, 0, 1), Vec3i(0, 0, -1)
        }};
        
        for (const auto& n : neighbors) {
            Vec3i neighborPos(x + n.x, y + n.y, z + n.z);
            if (!inGrid(neighborPos) || !get(neighborPos).active) {
                return true; // At least one empty neighbor means this is a surface voxel
            }
        }
        
        return false;
    }
    
    // Get normal for a surface voxel
    Vec3f calculateVoxelNormal(int x, int y, int z) const {
        Vec3f normal(0, 0, 0);
        
        // Simple gradient-based normal calculation
        if (inGrid(Vec3i(x+1, y, z)) && !get(x+1, y, z).active) normal.x += 1;
        if (inGrid(Vec3i(x-1, y, z)) && !get(x-1, y, z).active) normal.x -= 1;
        if (inGrid(Vec3i(x, y+1, z)) && !get(x, y+1, z).active) normal.y += 1;
        if (inGrid(Vec3i(x, y-1, z)) && !get(x, y-1, z).active) normal.y -= 1;
        if (inGrid(Vec3i(x, y, z+1)) && !get(x, y, z+1).active) normal.z += 1;
        if (inGrid(Vec3i(x, y, z-1)) && !get(x, y, z-1).active) normal.z -= 1;
        
        if (normal.lengthSquared() > 0) {
            return normal.normalized();
        }
        return Vec3f(0, 1, 0); // Default up normal
    }
public:
    std::vector<frame> genSlices(frame::colormap colorFormat = frame::colormap::RGB) const {
        TIME_FUNCTION;
        int colors;
        std::vector<frame> outframes;
        switch (colorFormat) {
            case frame::colormap::RGBA:
            case frame::colormap::BGRA: {
                colors = 4;
                break;
            }
            case frame::colormap::B: {
                colors = 1;
                break;
            }
            case frame::colormap::RGB:
            case frame::colormap::BGR:
            default: {
                colors = 3;
                break;
            }
        }
        size_t cbsize = gridSize.x * gridSize.y * colors;
        for (int layer = 0; layer < getDepth(); layer++) {
            int layerMult = layer * gridSize.x * gridSize.y;
            frame layerFrame(gridSize.x, gridSize.y, colorFormat);
            std::vector<uint8_t> colorBuffer(cbsize);
            for (int y = 0; y < gridSize.y; y++) {
                int yMult = layerMult + (y * gridSize.x);
                for (int x = 0; x < gridSize.x; x++)  {
                    int vidx = yMult + x;
                    int pidx = (y * gridSize.x + x) * colors;
                    Voxel cv = voxels[vidx];
                    Vec3ui8 cvColor;
                    float cvAlpha;
                    if (cv.active) {
                        cvColor = cv.color;
                        cvAlpha = cv.alpha;
                    } else {
                        cvColor = Vec3ui8(255,255,255);
                        cvAlpha = 255;
                    }
                    switch (colorFormat) {
                        case frame::colormap::RGBA: {
                            colorBuffer[pidx + 0] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 2] = cvColor.z;
                            colorBuffer[pidx + 3] = cvAlpha;
                            break;
                        }
                        case frame::colormap::BGRA: {
                            colorBuffer[pidx + 3] = cvColor.x;
                            colorBuffer[pidx + 2] = cvColor.y;
                            colorBuffer[pidx + 1] = cvColor.z;
                            colorBuffer[pidx + 0] = cvAlpha;
                            break;
                        }
                        case frame::colormap::RGB: {
                            colorBuffer[pidx + 0] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 2] = cvColor.z;
                            break;
                        }
                        case frame::colormap::BGR: {
                            colorBuffer[pidx + 2] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 0] = cvColor.z;
                            break;
                        }
                        case frame::colormap::B: {
                            colorBuffer[pidx] = static_cast<uint8_t>((cvColor.x * 0.299) + (cvColor.y * 0.587) + (cvColor.z * 0.114));
                            break;
                        }
                    }
                }
            }
            layerFrame.setData(colorBuffer);
            //layerFrame.compressFrameLZ78();
            outframes.emplace_back(layerFrame);   
        }
        return outframes;
    }
    
 
};
//#include "g3_serialization.hpp" needed to be usable

#endif
