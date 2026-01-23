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
#include "../vecmat/mat3.hpp"
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"
#include <cfloat> 

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

struct Chunk {
    Voxel reprVoxel; //average of all voxels in chunk for LOD rendering
    std::vector<bool> activeVoxels; //use this to specify active voxels in this chunk. 
    std::vector<Voxel> voxels; //list of all voxels in chunk.
    //std::vector<Chunk> children; //list of all chunks in chunk. for future use.
    bool active; //active if any child chunk or child voxel is active. used to efficiently find active voxels by only going down when an active chunk is found.
    int chunkSize; //should be (CHUNK_THRESHOLD/2) * 2 ^ depth I think. (ie: 1 depth will be (16/2)*(2^1) or 16, second will be (16/2)*(2^2) or 8*4=32)
    Vec3i minCorner; //position of chunk in world space.
    Vec3i maxCorner;
    int depth; //number of parent/child traversals to get here.

    Chunk() : active(false), chunkSize(0), depth(0) {}
    
    Chunk(const Vec3i& minCorner, int chunkSize, int depth = 0) : minCorner(minCorner), chunkSize(chunkSize),
          depth(depth), maxCorner(minCorner + chunkSize), active(false) {
        int voxelCount = chunkSize * chunkSize * chunkSize;
        activeVoxels.resize(voxelCount, false);
        voxels.resize(voxelCount);
    }
    
    // Convert world position to local chunk index
    Vec3i worldToLocal(const Vec3i& worldPos) const {
        return worldPos - minCorner;
    }

    Vec3i localToWorld(const Vec3i& localPos) const {
        return localPos + minCorner;
    }
    
    // Convert local chunk position to index
    size_t mortonIndex(const Vec3i& localPos) const {
        uint8_t x = static_cast<uint8_t>(localPos.x) & 0x0F;
        uint8_t y = static_cast<uint8_t>(localPos.y) & 0x0F;
        uint8_t z = static_cast<uint8_t>(localPos.z) & 0x0F;
        
        // Spread 4 bits using lookup tables or bit operations
        // For 4 bits: x = abcd -> a000b000c000d
        uint16_t xx = x;
        xx = (xx | (xx << 4)) & 0x0F0F;  // 0000abcd -> 0000abcd0000abcd
        xx = (xx | (xx << 2)) & 0x3333;  // -> 00ab00cd00ab00cd
        xx = (xx | (xx << 1)) & 0x5555;  // -> 0a0b0c0d0a0b0c0d
        
        uint16_t yy = y;
        yy = (yy | (yy << 4)) & 0x0F0F;
        yy = (yy | (yy << 2)) & 0x3333;
        yy = (yy | (yy << 1)) & 0x5555;
        
        uint16_t zz = z;
        zz = (zz | (zz << 4)) & 0x0F0F;
        zz = (zz | (zz << 2)) & 0x3333;
        zz = (zz | (zz << 1)) & 0x5555;
        
        // Combine: x in bit 0, y in bit 1, z in bit 2
        return xx | (yy << 1) | (zz << 2);
    }
    
    // Get voxel at world position
    Voxel& getWVoxel(const Vec3i& worldPos) {
        Vec3i local = worldToLocal(worldPos);
        return voxels[mortonIndex(local)];
    }
    
    const Voxel& getWVoxel(const Vec3i& worldPos) const {
        Vec3i local = worldToLocal(worldPos);
        return voxels[mortonIndex(local)];
    }

    Voxel& getLVoxel(const Vec3i& localPos) {
        return voxels[mortonIndex(localPos)];
    }
    
    const Voxel& getLVoxel(const Vec3i& localPos) const {
        return voxels[mortonIndex(localPos)];
    }
    
    // Set voxel at world position
    void setVoxel(const Vec3i& worldPos, const Voxel& voxel) {
        Vec3i local = worldToLocal(worldPos);
        size_t idx = mortonIndex(local);
        voxels[idx] = voxel;
        activeVoxels[idx] = voxel.active;
        
        // Update chunk active status
        if (voxel.active && !active) {
            active = true;
        }
    }
    
    // Check if a world position is inside this chunk
    bool contains(const Vec3i& worldPos) const {
        return worldPos.AllGTE(minCorner) && worldPos.AllLT(maxCorner);
    }
    
    // Check if a point is inside this chunk
    bool contains(const Vec3f& worldPos) const {
        return worldPos.AllGTE(minCorner.toFloat()) && worldPos.AllLT(maxCorner.toFloat());
    }
    
    // Ray bypass - calculate where ray exits this chunk
    bool rayBypass(const Vec3f& rayOrigin, const Vec3f& rayDir, float& tExit) const {
        Vec3f invDir = rayDir.safeInverse();
        Vec3f t1 = (minCorner.toFloat() - rayOrigin) * invDir;
        Vec3f t2 = (maxCorner.toFloat() - rayOrigin) * invDir;
        
        Vec3f tMin = t1.min(t2);
        Vec3f tMax = t1.max(t2);
        
        float tNear = tMin.maxComp();
        tExit = tMax.minComp();
        
        return tMax >= tMin && tMax >= 0.0f;
    }
    
    // Ray traverse within this chunk
    bool rayTraverse(const Vec3f& entryPoint, const Vec3f& exitPoint, 
                    Voxel& outVoxel, std::vector<size_t>& hitIndices) const {
        Vec3f ray = exitPoint - entryPoint;
        
        // Initialize DDA algorithm
        Vec3i cv = entryPoint.floorToI();
        Vec3i lv = exitPoint.floorToI();
        
        // Clamp to chunk bounds
        cv = cv.max(minCorner).min(maxCorner - Vec3i(1, 1, 1));
        lv = lv.max(minCorner).min(maxCorner - Vec3i(1, 1, 1));
        
        Vec3<int8_t> step = Vec3<int8_t>(
            ray.x >= 0 ? 1 : -1,
            ray.y >= 0 ? 1 : -1,
            ray.z >= 0 ? 1 : -1
        );
        
        Vec3f tDelta = Vec3f(
            ray.x != 0 ? std::abs(1.0f / ray.x) : INF,
            ray.y != 0 ? std::abs(1.0f / ray.y) : INF,
            ray.z != 0 ? std::abs(1.0f / ray.z) : INF
        );
        
        // Calculate initial tMax values
        Vec3f tMax;
        if (ray.x > 0) {
            tMax.x = (std::floor(entryPoint.x) + 1.0f - entryPoint.x) / ray.x;
        } else if (ray.x < 0) {
            tMax.x = (entryPoint.x - std::floor(entryPoint.x)) / -ray.x;
        } else {
            tMax.x = INF;
        }
        
        if (ray.y > 0) { 
            tMax.y = (std::floor(entryPoint.y) + 1.0f - entryPoint.y) / ray.y;
        } else if (ray.y < 0) {
            tMax.y = (entryPoint.y - std::floor(entryPoint.y)) / -ray.y;
        } else {
            tMax.y = INF;
        }
        
        if (ray.z > 0) {
            tMax.z = (std::floor(entryPoint.z) + 1.0f - entryPoint.z) / ray.z;
        } else if (ray.z < 0) {
            tMax.z = (entryPoint.z - std::floor(entryPoint.z)) / -ray.z;
        } else {
            tMax.z = INF;
        }
        
        // Clear hit indices
        hitIndices.clear();
        
        // DDA traversal within chunk
        while (cv != lv && contains(cv)) {
            Vec3i local = worldToLocal(cv);
            size_t idx = mortonIndex(local);
            
            if (activeVoxels[idx]) {
                hitIndices.push_back(idx);
            }
            
            // Find next voxel boundary
            int axis = (tMax.x < tMax.y) ? 
                      ((tMax.x < tMax.z) ? 0 : 2) :
                      ((tMax.y < tMax.z) ? 1 : 2);
            
            switch(axis) {
                case 0: 
                    tMax.x += tDelta.x; 
                    cv.x += step.x; 
                    break;
                case 1: 
                    tMax.y += tDelta.y; 
                    cv.y += step.y; 
                    break;
                case 2: 
                    tMax.z += tDelta.z; 
                    cv.z += step.z; 
                    break;
            }
        }
        
        // Check the last voxel
        if (contains(cv)) {
            Vec3i local = worldToLocal(cv);
            size_t idx = mortonIndex(local);
            if (activeVoxels[idx]) {
                hitIndices.push_back(idx);
            }
        }
        
        // Process hits if any
        if (!hitIndices.empty()) {
            outVoxel.alpha = 0.0f;
            outVoxel.active = true;
            
            for (size_t idx : hitIndices) {
                if (outVoxel.alpha >= 1.0f) break;
                
                const Voxel& curVoxel = voxels[idx];
                float remainingOpacity = 1.0f - outVoxel.alpha;
                float contribution = curVoxel.alpha * remainingOpacity;
                
                if (outVoxel.alpha < EPSILON) {
                    outVoxel.color = curVoxel.color;
                } else {
                    // Blend colors
                    outVoxel.color = Vec3ui8(
                        static_cast<uint8_t>(outVoxel.color.x + (curVoxel.color.x * remainingOpacity)),
                        static_cast<uint8_t>(outVoxel.color.y + (curVoxel.color.y * remainingOpacity)),
                        static_cast<uint8_t>(outVoxel.color.z + (curVoxel.color.z * remainingOpacity))
                    );
                }
                outVoxel.alpha += contribution;
            }
            return true;
        }
        
        return false;
    }
    
    // Build representation voxel (average of all active voxels)
    void buildReprVoxel() {
        if (!active) {
            reprVoxel = Voxel();
            return;
        }
        
        int activeCount = 0;
        Vec3f accumColor(0, 0, 0);
        float accumAlpha = 0.0f;
        float accumWeight = 0.0f;
        
        for (size_t i = 0; i < voxels.size(); ++i) {
            if (activeVoxels[i]) {
                const Voxel& v = voxels[i];
                accumColor.x += v.color.x;
                accumColor.y += v.color.y;
                accumColor.z += v.color.z;
                accumAlpha += v.alpha;
                accumWeight += v.weight;
                activeCount++;
            }
        }
        
        if (activeCount > 0) {
            reprVoxel.color = Vec3ui8(
                static_cast<uint8_t>(accumColor.x / activeCount),
                static_cast<uint8_t>(accumColor.y / activeCount),
                static_cast<uint8_t>(accumColor.z / activeCount)
            );
            reprVoxel.alpha = accumAlpha / activeCount;
            reprVoxel.weight = accumWeight / activeCount;
            reprVoxel.active = true;
        } else {
            reprVoxel = Voxel();
        }
    }
};

class VoxelGrid {
private:
    Vec3i gridSize;
    std::vector<Voxel> voxels;
    std::vector<bool> activeVoxels;
    std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunkList;
    int xyPlane;
    
    float radians(float rads) {
        return rads * (M_PI / 180);
    }
    
    Vec3i getChunkCoord(const Vec3i& voxelPos) const {
        return voxelPos / CHUNK_THRESHOLD;
    }

    void updateChunkStatus(const Vec3i& pos, bool isActive) {
        Vec3i chunkCoord = getChunkCoord(pos);

        if (isActive) {
            chunkList[chunkCoord].active = true;
        }
    }
    
    size_t mortonEncode(int x, int y, int z) const {
        //TIME_FUNCTION;
        size_t result = 0;
        uint64_t xx = x & 0x1FFFFF;  // Mask to 21 bits
        uint64_t yy = y & 0x1FFFFF;
        uint64_t zz = z & 0x1FFFFF;
        
        // Spread bits using parallel bit deposit operations
        xx = (xx | (xx << 32)) & 0x1F00000000FFFF;
        xx = (xx | (xx << 16)) & 0x1F0000FF0000FF;
        xx = (xx | (xx << 8)) & 0x100F00F00F00F00F;
        xx = (xx | (xx << 4)) & 0x10C30C30C30C30C3;
        xx = (xx | (xx << 2)) & 0x1249249249249249;
        
        yy = (yy | (yy << 32)) & 0x1F00000000FFFF;
        yy = (yy | (yy << 16)) & 0x1F0000FF0000FF;
        yy = (yy | (yy << 8)) & 0x100F00F00F00F00F;
        yy = (yy | (yy << 4)) & 0x10C30C30C30C30C3;
        yy = (yy | (yy << 2)) & 0x1249249249249249;
        
        zz = (zz | (zz << 32)) & 0x1F00000000FFFF;
        zz = (zz | (zz << 16)) & 0x1F0000FF0000FF;
        zz = (zz | (zz << 8)) & 0x100F00F00F00F00F;
        zz = (zz | (zz << 4)) & 0x10C30C30C30C30C3;
        zz = (zz | (zz << 2)) & 0x1249249249249249;

        result = xx | (yy << 1) | (zz << 2);
        return result;
    }

    bool intersectRayAABB(const Vec3f& origin, const Vec3f& dir, const Vec3f& boxMin, const Vec3f& boxMax, float& tNear, float& tFar) const {
        Vec3f invDir = dir.safeInverse();
        
        Vec3f t1 = (boxMin - origin) * invDir;
        Vec3f t2 = (boxMax - origin) * invDir;
        
        Vec3f tMin = t1.min(t2);
        Vec3f tMax = t1.max(t2);
        
        tNear = tMin.maxComp();
        tFar = tMax.minComp();
        
        return tMax >= tMin && tMax >= 0.0f;
    }

    std::vector<Vec3f> precomputeRayDirs(const Camera& cam, Vec2i res) const {
        std::vector<Vec3f> dirs(res.x * res.y);
        #pragma omp parallel for
        int rest = res.x * res.y;
        for (int i = 0; i < rest; i++) {
            int x = i % res.x;
            int y = i / res.x;
            float aspect = static_cast<float>(res.x) / res.y;
            float tanHalfFOV = tan(cam.fov * 0.5f);
            
            // Convert to normalized device coordinates [-1, 1]
            float ndcX = (2.0f * (x + 0.5f) / res.x - 1.0f) * aspect * tanHalfFOV;
            float ndcY = (1.0f - 2.0f * (y + 0.5f) / res.y) * tanHalfFOV;
            
            Vec3f rayDirCameraSpace = Vec3f(ndcX, ndcY, -1.0f).normalized();
            
            // Transform to world space
            Vec3f forward = cam.forward();
            Vec3f right = cam.right();
            Vec3f up = cam.up;
            
            // Camera-to-world rotation matrix
            Mat3f cameraToWorld(
                right.x, up.x, -forward.x,
                right.y, up.y, -forward.y,
                right.z, up.z, -forward.z
            );
            // Compute ray direction once
            dirs[i] = (cameraToWorld * rayDirCameraSpace).normalized();
        }
        return dirs;
    }

public:
    VoxelGrid() : gridSize(0,0,0) {
        std::cout << "creating empty grid." << std::endl;
    }

    VoxelGrid(int w, int h, int d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
        activeVoxels.resize(w * h * d, false);
    }
    
    bool serializeToFile(const std::string& filename);
    
    static std::unique_ptr<VoxelGrid> deserializeFromFile(const std::string& filename);
    
    Voxel& get(int x, int y, int z) {
        return voxels[mortonEncode(x,y,z)];
    }

    const Voxel& get(int x, int y, int z) const {
        return voxels[mortonEncode(x,y,z)];
    }

    Voxel& get(const Vec3i& xyz) {
        return get(xyz.x, xyz.y, xyz.z);
    }

    const Voxel& get(const Vec3i& xyz) const {
        return get(xyz.x, xyz.y, xyz.z);
    }
    
    bool isActive(int x, int y, int z) const {
        return activeVoxels[mortonEncode(x,y,z)];
    }
    
    bool isActive(const Vec3i& xyz) const {
        return isActive(xyz.x, xyz.y, xyz.z);
    }
    
    bool isActive(size_t index) const {
        return activeVoxels[index];
    }

    void resize(int newW, int newH, int newD) {
        size_t newSize = newW * newH * newD;
        std::vector<Voxel> newVoxels(newSize);
        std::vector<bool> newActiveVoxels(newSize, false);
        
        std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunklist;

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
                
                std::copy(
                    activeVoxels.begin() + oldRowStart, 
                    activeVoxels.begin() + oldRowStart + copyW, 
                    newActiveVoxels.begin() + newRowStart
                );

                for (int x = 0; x < copyW; ++x) {
                    if (activeVoxels[oldRowStart + x]) {
                        Vec3i cc(x / CHUNK_THRESHOLD, y / CHUNK_THRESHOLD, z / CHUNK_THRESHOLD);
                        
                    }
                }
            }
        }
        voxels = std::move(newVoxels);
        activeVoxels = std::move(newActiveVoxels);
        gridSize = Vec3i(newW, newH, newD);
        xyPlane = gridSize.x * gridSize.y;
    }

    void resize(Vec3i newsize) {
        resize(newsize.x, newsize.y, newsize.z);
    }

    void set(int x, int y, int z, bool active, Vec3ui8 color, float alpha = 1) {
        set(Vec3i(x,y,z), active, color, alpha);
    }
    
    void set(Vec3i pos, bool active, Vec3ui8 color, float alpha = 1.f) {
        if (pos.AllGTE(0)) {
            if (pos.AnyGTE(gridSize)) {
                resize(gridSize.max(pos));
            }
            
            size_t idx = mortonEncode(pos.x, pos.y, pos.z);
            Voxel& v = voxels[idx];
            v.active = active;
            v.color = color;
            v.alpha = alpha;
            activeVoxels[idx] = active;
            updateChunkStatus(pos, active);
        }
    }
    
    void setBatch(const std::vector<Vec3i>& positions, bool active, Vec3ui8 color, float alpha = 1.0f) {
        // First, ensure grid is large enough
        Vec3i maxPos(0,0,0);
        for (const auto& pos : positions) {
            maxPos = maxPos.max(pos);
        }
        
        if (maxPos.AnyGTE(gridSize)) {
            resize(maxPos);
        }
        
        // Set all positions
        for (const auto& pos : positions) {
            size_t idx = mortonEncode(pos.x, pos.y, pos.z);
            Voxel& v = voxels[idx];
            v.active = active;
            v.color = color;
            v.alpha = alpha;
            activeVoxels[idx] = active;
            updateChunkStatus(pos, active);
        }
    }

    bool inGrid(Vec3i voxl) const {
        return voxl.AllGTE(0) && voxl.AllLT(gridSize);
    }

    void voxelTraverse(const Vec3f& origin, const Vec3f& end, Voxel& outVoxel, int maxDist = 10000000) const {
        Vec3i cv = origin.floorToI();
        Vec3i lv = end.floorToI();
        Vec3f ray = end - origin;
        Vec3<int8_t> step = Vec3<int8_t>(ray.x >= 0 ? 1 : -1, ray.y >= 0 ? 1 : -1, ray.z >= 0 ? 1 : -1);
        Vec3f tDelta = Vec3f(ray.x != 0 ? std::abs(1.0f / ray.x) : INF,
                            ray.y != 0 ? std::abs(1.0f / ray.y) : INF,
                            ray.z != 0 ? std::abs(1.0f / ray.z) : INF);
        
        Vec3f tMax;
        if (ray.x > 0) {
            tMax.x = (std::floor(origin.x) + 1.0f - origin.x) / ray.x;
        } else if (ray.x < 0) {
            tMax.x = (origin.x - std::floor(origin.x)) / -ray.x;
        } else tMax.x = INF;

        if (ray.y > 0) { 
            tMax.y = (std::floor(origin.y) + 1.0f - origin.y) / ray.y;
        } else if (ray.y < 0) {
            tMax.y = (origin.y - std::floor(origin.y)) / -ray.y;
        } else tMax.y = INF;

        if (ray.z > 0) {
            tMax.z = (std::floor(origin.z) + 1.0f - origin.z) / ray.z;
        } else if (ray.z < 0) {
            tMax.z = (origin.z - std::floor(origin.z)) / -ray.z;
        } else tMax.z = INF;

        std::vector<size_t> activeIndices;
        activeIndices.reserve(16);
        
        while (cv != lv && inGrid(cv) && activeIndices.size() < 16) {
            size_t idx = mortonEncode(cv.x, cv.y, cv.z);
            

            if (voxels[idx].active) {
                activeIndices.push_back(idx);
            }
            
            
            int axis = (tMax.x < tMax.y) ? 
                    ((tMax.x < tMax.z) ? 0 : 2) :
                    ((tMax.y < tMax.z) ? 1 : 2);
            
            switch(axis) {
                case 0: 
                    tMax.x += tDelta.x; 
                    cv.x += step.x; 
                    break;
                case 1: 
                    tMax.y += tDelta.y; 
                    cv.y += step.y; 
                    break;
                case 2: 
                    tMax.z += tDelta.z; 
                    cv.z += step.z; 
                    break;
            }
        }
        
        // Second pass: process only active voxels
        outVoxel.alpha = 0.0f;
        outVoxel.active = !activeIndices.empty();
        
        for (size_t idx : activeIndices) {
            if (outVoxel.alpha >= 1.0f) break;
            
            const Voxel& curVoxel = voxels[idx];
            float remainingOpacity = 1.0f - outVoxel.alpha;
            float contribution = curVoxel.alpha * remainingOpacity;
            
            if (outVoxel.alpha < EPSILON) {
                outVoxel.color = curVoxel.color;
            } else {
                outVoxel.color = outVoxel.color + (curVoxel.color * remainingOpacity);
            }
            outVoxel.alpha += contribution;
        }
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
        float maxDist = std::sqrt(gridSize.lengthSquared());
        Vec3f gridMin(0, 0, 0);

        frame outFrame(resolution.x, resolution.y, colorformat);
        std::vector<uint8_t> colorBuffer;
        int channels;
        if (colorformat == frame::colormap::RGB) {
            channels = 3;
        } else {
            channels = 4;
        }
        colorBuffer.resize(resolution.x * resolution.y * channels);
        std::vector<Vec3f> dirs = precomputeRayDirs(cam, resolution);
        int rest = resolution.x * resolution.y;

        #pragma omp parallel for
        for (int idx = 0; idx < rest; idx++) {
            int x = idx % resolution.x;
            int y = idx / resolution.x;
            int resy = y * resolution.x;
            Voxel outVoxel(0, false, 0.f, Vec3ui8(10, 10, 255));
            Vec3f rayDirWorld = dirs[idx];
            float tNear = 0.0f;
            float tFar = maxDist;
            bool hit = intersectRayAABB(cam.posfor.origin, rayDirWorld, gridMin, gridSize, tNear, tFar);
            if (!hit) {
                Vec3ui8 hitColor = outVoxel.color;
                
                // Set pixel color in buffer
                switch (colorformat) {
                    case frame::colormap::BGRA: {
                        int idx = (resy + x) * 4;
                        colorBuffer[idx + 3] = hitColor.x;
                        colorBuffer[idx + 2] = hitColor.y;
                        colorBuffer[idx + 1] = hitColor.z;
                        colorBuffer[idx + 0] = 255;
                        break;
                    }
                    case frame::colormap::RGB:
                    default: {
                        int idx = (resy + x) * 3;
                        colorBuffer[idx] = hitColor.x;
                        colorBuffer[idx + 1] = hitColor.y;
                        colorBuffer[idx + 2] = hitColor.z;
                        break;
                    }
                }
                continue;
            }

            Vec3f rayStartGrid = cam.posfor.origin;
            Vec3f rayEnd = rayStartGrid + rayDirWorld * tFar;
            Vec3f ray = rayEnd - rayStartGrid;

            voxelTraverse(rayStartGrid, rayEnd, outVoxel, maxDist);
            Vec3ui8 hitColor = outVoxel.color;
            // Set pixel color in buffer
            switch (colorformat) {
                case frame::colormap::BGRA: {
                    int idx = (resy + x) * 4;
                    colorBuffer[idx + 3] = hitColor.x;
                    colorBuffer[idx + 2] = hitColor.y;
                    colorBuffer[idx + 1] = hitColor.z;
                    colorBuffer[idx + 0] = 255;
                    break;
                }
                case frame::colormap::RGB:
                default: {
                    int idx = (resy + x) * 3;
                    colorBuffer[idx] = hitColor.x;
                    colorBuffer[idx + 1] = hitColor.y;
                    colorBuffer[idx + 2] = hitColor.z;
                    break;
                }
            }
        }
        
        outFrame.setData(colorBuffer);
        return outFrame;
    }

    void printStats() const {
        int totalVoxels = gridSize.x * gridSize.y * gridSize.z;
        int activeVoxelsCount = 0;
        
        // Count active voxels using activeVoxels array
        for (bool isActive : activeVoxels) {
            if (isActive) {
                activeVoxelsCount++;
            }
        }
        
        float activePercentage = (totalVoxels > 0) ? 
            (static_cast<float>(activeVoxelsCount) / static_cast<float>(totalVoxels)) * 100.0f : 0.0f;
        
        std::cout << "=== Voxel Grid Statistics ===" << std::endl;
        std::cout << "Grid dimensions: " << gridSize.x << " x " << gridSize.y << " x " << gridSize.z << std::endl;
        std::cout << "Total voxels: " << totalVoxels << std::endl;
        std::cout << "Active voxels: " << activeVoxelsCount << std::endl;
        std::cout << "Inactive voxels: " << (totalVoxels - activeVoxelsCount) << std::endl;
        std::cout << "Active percentage: " << activePercentage << "%" << std::endl;
        std::cout << "Memory usage (approx): " << (voxels.size() * sizeof(Voxel) + activeVoxels.size() * sizeof(bool)) / 1024 << " KB" << std::endl;
        std::cout << "============================" << std::endl;
    }
    
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
        int cbsize = gridSize.x * gridSize.y * colors;
        for (int layer = 0; layer < getDepth(); layer++) {
            int layerMult = layer * gridSize.x * gridSize.y;
            frame layerFrame(gridSize.x, gridSize.y, colorFormat);
            std::vector<uint8_t> colorBuffer(cbsize);
            for (int y = 0; y < gridSize.y; y++) {
                int yMult = layerMult + (y * gridSize.x);
                for (int x = 0; x < gridSize.x; x++)  {
                    int vidx = yMult + x;
                    int pidx = (y * gridSize.x + x) * colors;
                    bool isActive = activeVoxels[vidx];
                    Voxel cv = voxels[vidx];
                    Vec3ui8 cvColor;
                    float cvAlpha;
                    if (isActive) {
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