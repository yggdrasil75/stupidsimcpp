#ifndef GRID3_HPP
#define GRID3_HPP

#include <unordered_map>
#include <fstream>
#include <cstring>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include "../vecmat/mat4.hpp"
//#include "serialization.hpp"
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"

constexpr char magic[4] = {'Y', 'G', 'G', '3'};

struct Voxel {
    //float active;
    bool active;
    //Vec3f position;
    Vec3ui8 color;
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

class VoxelGrid {
private:
    double binSize = 1;
    Vec3i gridSize;
    //int width, height, depth;
    std::vector<Voxel> voxels;
    float radians(float rads) {
        return rads * (M_PI / 180);
    }

    static Mat4f lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
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

    static Mat4f perspective(float fovy, float aspect, float zNear, float zfar) {
        float const tanhalfF = tan(fovy / 2);
        Mat4f Result = 0;
        Result(0,0) = 1 / (aspect * tanhalfF);
        Result(1,1) = 1 / tanhalfF;
        Result(2,2) = zfar / (zNear - zfar);
        Result(2,3) = -1;
        Result(3,2) = -(zfar * zNear) / (zfar - zNear);
        return Result;
    }

public:
    VoxelGrid() : gridSize(0,0,0) {
        std::cout << "creating empty grid." << std::endl;
    }

    VoxelGrid(int w, int h, int d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
    }
    
    //bool serializeToFile(const VoxelGrid grid, const std::string& filename);
    bool serializeToFile(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "failed to open file (serializeToFile): " << filename << std::endl;
            return false;
        }
        
        file.write(magic, 4);
        
        //file.write(reinterpret_cast<const char*>(&binSize), sizeof(binSize));
        
        // Write grid dimensions
        int dims[3] = {gridSize.x, gridSize.y, gridSize.z};
        file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        
        // Write voxel data
        size_t voxelCount = voxels.size();
        file.write(reinterpret_cast<const char*>(&voxelCount), sizeof(voxelCount));
        
        // Write each voxel
        for (const Voxel& voxel : voxels) {
            file.write(reinterpret_cast<const char*>(&voxel.active), sizeof(voxel.active));
            file.write(reinterpret_cast<const char*>(&voxel.color.x), sizeof(voxel.color.x));
            file.write(reinterpret_cast<const char*>(&voxel.color.y), sizeof(voxel.color.y));
            file.write(reinterpret_cast<const char*>(&voxel.color.z), sizeof(voxel.color.z));
        }
        
        file.close();
        return !file.fail();
    }
    
    static std::unique_ptr<VoxelGrid> deserializeFromFile(const std::string& filename) {
        VoxelGrid outgrid;
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
            return nullptr;
        }
        
        // Read and verify magic number
        char filemagic[4];
        file.read(filemagic, 4);
        if (std::strncmp(filemagic, "YGG7", 4) != 0) {
            std::cerr << "Error: Invalid file format or corrupted file" << std::endl;
            return nullptr;
        }
        
        // Read binSize
        //file.read(reinterpret_cast<char*>(&binSize), sizeof(binSize));
        
        // Read grid dimensions
        int dims[3];
        file.read(reinterpret_cast<char*>(dims), sizeof(dims));
        outgrid.resize(Vec3i(dims[0], dims[1], dims[2]));
        //gridSize = Vec3i(dims[0], dims[1], dims[2]);
        
        // Read voxel count
        size_t voxelCount;
        file.read(reinterpret_cast<char*>(&voxelCount), sizeof(voxelCount));
        
        // Verify voxel count matches grid dimensions
        size_t expectedCount = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
        if (voxelCount != expectedCount) {
            std::cerr << "Error: Voxel count mismatch. Expected " << expectedCount 
                     << ", found " << voxelCount << std::endl;
            return nullptr;
        }
        
        // Resize and read voxels
        //voxels.resize(voxelCount);
        grid->voxels.resize(voxelCount);
        for (size_t i = 0; i < voxelCount; ++i) {
            file.read(reinterpret_cast<char*>(&grid->voxels[i].active), sizeof(grid->voxels[i].active));
            file.read(reinterpret_cast<char*>(&grid->voxels[i].color.x), sizeof(grid->voxels[i].color.x));
            file.read(reinterpret_cast<char*>(&grid->voxels[i].color.y), sizeof(grid->voxels[i].color.y));
            file.read(reinterpret_cast<char*>(&grid->voxels[i].color.z), sizeof(grid->voxels[i].color.z));
        }
        
        file.close();
        if (file.fail()) {
            std::cerr << "Error: Failed to read from file: " << filename << std::endl;
            return nullptr;
        }
        
        return grid;
    }
    
    Voxel& get(int x, int y, int z) {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    const Voxel& get(int x, int y, int z) const {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    Voxel& get(const Vec3i& xyz) {
        return voxels[xyz.z * gridSize.x * gridSize.y + xyz.y * gridSize.x + xyz.x];
    }

    void resize(int newW, int newH, int newD) {
        std::vector<Voxel> newVoxels(newW * newH * newD);
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
            }
        }
        voxels = std::move(newVoxels);
        gridSize = Vec3i(newW, newH, newD);
    }

    void resize(Vec3i newsize) {
        resize(newsize.x, newsize.y, newsize.z);
    }

    void set(int x, int y, int z, bool active, Vec3ui8 color) {
        set(Vec3i(x,y,z), active, color);
    }

    void set(Vec3i pos, bool active, Vec3ui8 color) {
        if (pos.x >= 0 || pos.y >= 0 || pos.z >= 0) {
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
            v.active = active; // std::clamp(active, 0.0f, 1.0f);
            v.color = color;
        }
    }

    void set(Vec3i pos, Vec4ui8 rgbaval) {
        set(pos, static_cast<float>(rgbaval.a / 255), rgbaval.toVec3());
    }

    template<typename T>
    bool inGrid(Vec3<T> voxl) {
        return (voxl >= 0 && voxl.x < gridSize.x && voxl.y < gridSize.y && voxl.z < gridSize.z);
    }

    void voxelTraverse(const Vec3d& origin, const Vec3d& end, std::vector<Vec3i>& visitedVoxel) {
        Vec3i cv = (origin / binSize).floorToI();
        Vec3i lv = (end / binSize).floorToI();
        Vec3d ray = end - origin;
        Vec3f step = Vec3f(ray.x >= 0 ? 1 : -1, ray.y >= 0 ? 1 : -1, ray.z >= 0 ? 1 : -1);
        Vec3d nextVox = cv.toDouble() + step * binSize;
        Vec3d tMax = Vec3d(ray.x != 0 ? (nextVox.x - origin.x) / ray.x : INF,
                           ray.y != 0 ? (nextVox.y - origin.y) / ray.y : INF,
                           ray.z != 0 ? (nextVox.z-origin.z) / ray.z : INF);
        Vec3d tDelta = Vec3d(ray.x != 0 ? binSize / ray.x * step.x : INF,
                             ray.y != 0 ? binSize / ray.y * step.y : INF,
                             ray.z != 0 ? binSize / ray.z * step.z : INF);

        Vec3i diff(0,0,0);
        bool negRay = false;
        if (cv.x != lv.x && ray.x < 0) {
            diff.x = diff.x--;
            negRay = true;
        }
        if (cv.y != lv.y && ray.y < 0) {
            diff.y = diff.y--;
            negRay = true;
        }
        if (cv.z != lv.z && ray.z < 0) {
            diff.z = diff.z--;
            negRay = true;
        }

        if (negRay) {
            cv += diff;
            visitedVoxel.push_back(cv);
        }

        while (lv != cv && inGrid(cv)) {
            if (get(cv).active) {
                visitedVoxel.push_back(cv);
            }
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    cv.x += step.x;
                    tMax.x += tDelta.x;
                } else {
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            } else {
                if (tMax.y < tMax.z) {
                    cv.y += step.y;
                    tMax.y += tDelta.y;
                } else {
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            }
        }
        return; // &&visitedVoxel;
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
    
    frame renderFrame(const Camera& cam, Vec2i resolution, frame::colormap colorformat = frame::colormap::RGB) {
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
        std::cout << "Active percentage: " << activePercentage << "%" << std::endl;
        std::cout << "Memory usage (approx): " << (voxels.size() * sizeof(Voxel)) / 1024 << " KB" << std::endl;
        std::cout << "============================" << std::endl;
    }
};

#endif
