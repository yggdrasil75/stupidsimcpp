#ifndef GRID3_HPP
#define GRID3_HPP

#include <unordered_map>
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include "../vecmat/mat4.hpp"
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"

struct Voxel {
    float active;
    //Vec3f position;
    Vec3ui8 color;
};

struct Camera {
    Ray3f posfor;
    Vec3f up;
    float fov;
    Camera(Vec3f pos, Vec3f viewdir, Vec3f up, float fov = 80) : posfor(Ray3f(pos, viewdir)), up(up), fov(fov) {}
};

class VoxelGrid {
private:
    size_t width, height, depth;
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

    std::pair<float,float> rayBoxIntersect(const Vec3f& origin, const Vec3f& direction) {
        Vec3f tBMin = Vec3f(0,0,0);
        Vec3f tBMax = Vec3f(width, height, depth);
        float tmin = 0;
        float tmax = INF;
        Vec3f invDir = direction.safeInverse();
        for (int i = 0; i < 3; ++i) {
            if (abs(direction[i] < EPSILON)) {
                if (origin[i] < tBMin[i] || origin[i] > tBMax[i]) return std::make_pair(0.0f, 0.0f);
                float t1 = tBMin[i] - origin[i] * invDir[i];
                float t2 = tBMax[i] - origin[i] * invDir[i];
                if (t1 > t2) std::swap(t1, t2);
                if (t1 > tmin) tmin = t1;
                if (t2 < tmax) tmax = t2;
                if (tmin > tmax) return std::make_pair(0,0);
            }
        }
        return std::make_pair(tmin, tmax);
    }
    
    //used to prevent division by 0 issues
    bool specialCases(const Vec3f& origin, const Vec3f& direction, float maxDist, Vec3f& hitColor) {
        float stepSize = 0.5;
        int maxSteps = maxDist/stepSize;
        for (int step = 0; step < maxSteps; ++step) {
            float t = step * stepSize;
            Vec3f pos = Vec3f(origin + direction * t);
            Vec3T voxelCoords = pos.floorToT();
            if (inGrid(voxelCoords)) {
                Voxel cv = get(voxelCoords);
                if (cv.active > EPSILON) {
                    hitColor = cv.color.toFloat();
                    std::cout << "hit in special case at: " << voxelCoords << std::endl;
                    return true;
                }
            }
        }
        return false;
    }

public:
    VoxelGrid(size_t w, size_t h, size_t d) : width(w), height(h), depth(d) {
        voxels.resize(w * h * d);
    }

    Voxel& get(size_t x, size_t y, size_t z) {
        return voxels[z * width * height + y * width + x];
    }

    const Voxel& get(size_t x, size_t y, size_t z) const {
        return voxels[z * width * height + y * width + x];
    }

    Voxel& get(const Vec3T& xyz) {
        return voxels[xyz.z*width*height+xyz.y*width+xyz.x];
    }

    void resize() {
        //TODO: proper resizing
    }

    void set(size_t x, size_t y, size_t z, float active, Vec3ui8 color) {
        //expand grid if needed. 
        if (x >= 0 || y >= 0 || z >= 0) {
            if (!(x < width)) {
                //until resizing added:
                return;
                width = x;
                resize();
            }
            else if (!(y < height)) {
                //until resizing added:
                return;
                height = y;
                resize();
            }
            else if (!(z < depth)) {
                //until resizing added:
                return;
                depth = z;
                resize();
            }

            Voxel& v = get(x, y, z);
            v.active = std::clamp(active, 0.0f, 1.0f);
            v.color = color;
        }
    }

    template<typename T>
    bool inGrid(Vec3<T> voxl) {
        return (voxl >= 0 && voxl.x < width && voxl.y < height && voxl.z < depth);
    }

    std::vector<Vec3f> genPixelDirs(const Vec3f& pos, const Vec3f& dir, size_t imgWidth, size_t imgHeight, float fov) {
        std::vector<Vec3f> dirs(imgWidth * imgHeight);
        float fovRad = radians(fov);
        float tanFov = tan(fovRad * 0.5);
        float aspect = static_cast<float>(imgWidth) / static_cast<float>(imgHeight);
        Vec3f worldUp(0, 1, 0);
        Vec3f camRight = worldUp.cross(dir).normalized();
        Vec3f camUp = dir.cross(camRight).normalized();
        
        float imgWidthInv = 1 / (imgWidth - 1);
        float imgHeightInv = 1 / (imgHeight - 1);
        float aspectTanFov = aspect * tanFov;

        for (int y = 0; y < imgHeight; ++y) {
            float ndcY = 1 - (2 * y * imgHeightInv);
            float screenY = ndcY * tanFov;
            for (int x = 0; x < imgWidth; ++x) {
                float ndcX = (2 * x * imgWidthInv) - 1;
                float screenX = ndcX * aspectTanFov;
                Vec3f dir = (camRight * screenX + camUp * screenY + dir).normalized();
                dirs[y*imgWidth+x] = dir;
            }
        }
        return dirs;
    }

    Vec3f perPixelRayDir(size_t x, size_t y, size_t imgWidth, size_t imgHeight, const Camera& cam) const {
        float normedX = (x + 0.5) / imgWidth * 2 - 1;
        float normedY = 1 - (y+0.5) / imgHeight * 2;
        float aspect = imgWidth / imgHeight;

        float fovRad = cam.fov * M_PI / 180;
        float scale = tan(fovRad * 0.5);
        Vec3f rayDirCam = Vec3f(normedX * aspect * scale, normedY * scale, -1).normalized();
        Vec3f eye = cam.posfor.origin;
        Vec3f center = eye + cam.posfor.direction;
        Mat4f viewMat = lookAt(eye, center, cam.up);
        Mat4f invViewMat = viewMat.inverse();
        Vec3f rayDirWorld = invViewMat.transformDirection(rayDirCam);
        return rayDirWorld.normalized();
    }

    bool rayCast(const Vec3f& origin, const Vec3f& direction, float maxDist, Vec3f& hitColor) {
        direction.normalized();
        if (abs(direction.length()) < EPSILON) return false;

        Vec3f invDir = direction.safeInverse();
        Vec3T currentVoxel = origin.floorToT();

        if (direction.x == 0 || direction.y == 0 || direction.z == 0) {
            return specialCases(origin, direction, maxDist, hitColor);
        }

        if (!inGrid(currentVoxel)) {
            std::pair<float,float> re = rayBoxIntersect(origin, direction);
            float tEntry = re.first;
            float tExit = re.second;
            if (tEntry < EPSILON || tExit < EPSILON) return false;
            float tStart = std::max(0.0f, tEntry);
            if (tStart > maxDist) return false;
            Vec3f gridOrig = origin + direction * tStart;
            currentVoxel = gridOrig.floorToT();
        }

        Vec3i8 step = Vec3i8(direction.x >= 0 ? 1 : -1, direction.y >= 0 ? 1 : -1, direction.z >= 0 ? 1 : -1);
        Vec3f tMax;
        for (int i = 0; i < 3; i++) {
            if (step[i] > 0) {
                tMax[i] = ((currentVoxel[i] + 1) - origin[i]) * invDir[i];
            } else {
                tMax[i] = (currentVoxel[i] - origin[i]) * invDir[i];
            }
        }
        Vec3f tDelta = invDir.abs();
        
        float aalpha = 0;

        while (inGrid(currentVoxel) && aalpha < 1) {
            Voxel cv = get(currentVoxel);
            if (cv.active > EPSILON) {
                float alpha = cv.active * (1.0f - aalpha);
                Vec3f voxelColor = cv.color.toFloat() / 255.0f;
                hitColor = hitColor + voxelColor * alpha;
                aalpha += cv.active;
            }
            
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                if (tMax.x > maxDist) break;
                currentVoxel.x += step.x;
                tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) 
            {
                currentVoxel.y += step.y;
                tMax.y += tDelta.y;
            }
            else {
                currentVoxel.z += step.z;
                tMax.z += tDelta.z;
            }
        }
        if (aalpha > EPSILON) {
            //std::cout << "hit in normal case " << " due to any alpha" << std::endl;
            return true;
        } else return false;
    }

    size_t getWidth() const {
        return width;
    }
    size_t getHeight() const {
        return height;
    }
    size_t getDepth() const {
        return depth;
    }

    void renderOut(std::vector<uint8_t>& output, size_t& outwidth, size_t& outheight, const Camera& cam) {
        TIME_FUNCTION; 
        output.resize(outwidth * outheight * 3);
        Vec3f backgroundColor(0.1f, 0.1f, 1.0f);
        float maxDistance = std::sqrt(width*width + height*height + depth*depth) * 2.0f;
        // std::vector<Vec3f> dirs = genPixelDirs(cam.posfor.origin, cam.posfor.direction, outwidth, outheight, cam.fov);

        for (size_t y = 0; y < outheight; y++) {        
            float yout = y * outwidth;
            for (size_t x = 0; x < outwidth; x++) {
                // Vec3f rayDir = dirs[y*outwidth + x];
                // Vec3f hitColor = Vec3f(0,0,0);
                Vec3f rayDir = perPixelRayDir(x, y, outwidth, outheight, cam);
                Ray3f ray(cam.posfor.origin, rayDir);
                Vec3f hitPos;
                Vec3f hitNorm;
                Vec3f hitColor;
                bool hit = rayCast(cam.posfor.origin, rayDir, maxDistance, hitColor);

                Vec3f finalColor;
                if (!hit) {
                    finalColor = backgroundColor;
                } else {
                    finalColor = hitColor;
                }

                finalColor = finalColor.clamp(0, 1);
                size_t pixelIndex = (yout + x) * 3;
                output[pixelIndex + 0] = static_cast<uint8_t>(finalColor.x * 255);
                output[pixelIndex + 1] = static_cast<uint8_t>(finalColor.y * 255);
                output[pixelIndex + 2] = static_cast<uint8_t>(finalColor.z * 255);
            }
        }
    }
};

#endif
