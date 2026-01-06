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
};

class VoxelGrid {
private:
    Vec3T gridSize;
    //size_t width, height, depth;
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
        Vec3f tBMax = gridSize.toFloat();
        float tmin = -INF;
        float tmax = INF;
        Vec3f invDir = direction.safeInverse();
        bool allParallel = true;
        for (int i = 0; i < 3; ++i) {
            if (abs(direction[i]) < EPSILON) {
                if (!(origin[i] < tBMin[i] || origin[i] > tBMax[i])) continue;
            }
            allParallel = false;
            float t1 = (tBMin[i] - origin[i]) * invDir[i];
            float t2 = (tBMax[i] - origin[i]) * invDir[i];
            
            if (t1 > t2) std::swap(t1, t2);
            
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            
            if (tmin > tmax) return std::make_pair(0.0f, 0.0f);
        }
        if (allParallel) {
            return std::make_pair(0.0f, 0.0f);
        }
        if (tmax < 0) return std::make_pair(0.0f, 0.0f);
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
    VoxelGrid(size_t w, size_t h, size_t d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
    }

    Voxel& get(size_t x, size_t y, size_t z) {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    const Voxel& get(size_t x, size_t y, size_t z) const {
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    Voxel& get(const Vec3T& xyz) {
        return voxels[xyz.z * gridSize.x * gridSize.y + xyz.y * gridSize.x + xyz.x];
    }

    void resize() {
        //TODO: proper resizing
    }

    void set(size_t x, size_t y, size_t z, bool active, Vec3ui8 color) {
        set(Vec3T(x,y,z), active, color);
    }

    void set(Vec3T pos, bool active, Vec3ui8 color) {
        if (pos.x >= 0 || pos.y >= 0 || pos.z >= 0) {
            if (!(pos.x < gridSize.x)) {
                //until resizing added:
                return;
                gridSize.x = pos.x;
                resize();
            }
            else if (!(pos.y < gridSize.y)) {
                //until resizing added:
                return;
                gridSize.y = pos.y;
                resize();
            }
            else if (!(pos.z < gridSize.z)) {
                //until resizing added:
                return;
                gridSize.z = pos.z;
                resize();
            }

            Voxel& v = get(pos);
            v.active = active; // std::clamp(active, 0.0f, 1.0f);
            v.color = color;
        }
    }

    void set(Vec3T pos, Vec4ui8 rgbaval) {
        set(pos, static_cast<float>(rgbaval.a / 255), rgbaval.toVec3());
    }

    template<typename T>
    bool inGrid(Vec3<T> voxl) {
        return (voxl >= 0 && voxl.x < gridSize.x && voxl.y < gridSize.y && voxl.z < gridSize.z);
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

    bool castRay(const Vec3f& origin, const Vec3f& direction, float maxDist, Vec3f& hitColor) {
        Vec3f dir = direction.normalized();
        Vec3T pos = origin.floorToT();
        Vec3i step = Vec3i(dir.x > 0 ? -1 : 1, dir.y > 0 ? -1 : 1, dir.z > 0 ? -1 : 1);
        Vec3f tMax;
        Vec3f tDelta;
        if (abs(dir.x) > EPSILON) {
            tMax.x = (static_cast<size_t>(pos.x) + (step.x > 0 ? 1 : 0) - pos.x) / dir.x;
            tDelta.x = step.x / dir.x;
        } else {
            tMax.x = INF;
            tDelta.x = INF;
        }
        if (abs(dir.y) > EPSILON) {
            tMax.y = (static_cast<size_t>(pos.y) + (step.y > 0 ? 1 : 0) - pos.y) / dir.y;
            tDelta.y = step.y / dir.y;
        } else {
            tMax.y = INF;
            tDelta.y = INF;
        }
        if (abs(dir.z) > EPSILON) {
            tMax.z = (static_cast<size_t>(pos.z) + (step.z > 0 ? 1 : 0) - pos.z) / dir.z;
            tDelta.z = step.z / dir.z;
        } else {
            tMax.z = INF;
            tDelta.z = INF;
        }

        auto intersect = rayBoxIntersect(origin, dir);
        float tEntry = intersect.first;
        float tExit = intersect.second;
        if (tEntry <= 0 && tExit <= 0) return false;
        if (tEntry > maxDist) return false;

        float dist = 0;
        while (dist < maxDist) {
            if (inGrid(pos)) {
                Voxel cv = get(pos);
                if (cv.active) {
                    Vec3f norm = Vec3f(0,0,0);
                    if (tMax.x <= tMax.y && tMax.x <= tMax.z) norm.x = -step.x;
                    else if (tMax.y <= tMax.x && tMax.y <= tMax.z) norm.y = -step.y;
                    else norm.z = -step.z;
                    hitColor = cv.color.toFloat() / 255;
                    return true;
                }
            }
            if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                pos.x += step.x;
                dist += tDelta.x;
                //dist = tMax.x;
                tMax.x += tDelta.x;
            } else if (tMax.y <= tMax.x && tMax.y <= tMax.z) {
                pos.y += step.y;
                //dist = tMax.y;
                dist += tDelta.y;
                tMax.y += tDelta.y;
            } else {
                pos.z += step.z;
                //dist = tMax.z;
                dist += tDelta.z;
                tMax.z += tDelta.z;
            }
        }
        return false;
    }

    bool rayCast(const Vec3f& origin, const Vec3f& direction, float maxDist, Vec3f& hitColor) {
        Vec3f dir = direction.normalized();
        if (abs(dir.length()) < EPSILON) return false;

        if (dir.x == 0 || dir.y == 0 || dir.z == 0) {
            return specialCases(origin, dir, maxDist, hitColor);
        }

        float tStart;
        Vec3f invDir = dir.safeInverse();
        Vec3T currentVoxel = origin.floorToT();
        if (!inGrid(currentVoxel)) {
            std::pair<float,float> re = rayBoxIntersect(origin, dir);
            float tEntry = re.first;
            float tExit = re.second;
            if (tEntry < EPSILON || tExit < EPSILON) return false;
            if (tEntry > maxDist) return false;
            tStart = tEntry;
        }

        Vec3f gridOrig = origin + dir * tStart;
        currentVoxel = gridOrig.floorToT();
        Vec3i8 step = Vec3i8(dir.x >= 0 ? 1 : -1, dir.y >= 0 ? 1 : -1, dir.z >= 0 ? 1 : -1);
        Vec3f tMax;
        float tDist = tStart;
        for (int i = 0; i < 3; i++) {
            if (step[i] > 0) {
                tMax[i] = ((currentVoxel[i] + 1) - gridOrig[i]) * invDir[i];
            } else {
                tMax[i] = (currentVoxel[i] - gridOrig[i]) * invDir[i];
            }
        }
        Vec3f tDelta = invDir.abs();
        
        float aalpha = 0;

        while (inGrid(currentVoxel) && aalpha < 1 && tDist <= maxDist) {
            Voxel cv = get(currentVoxel);
            if (cv.active > EPSILON) {
                float alpha = cv.active * (1.0f - aalpha);
                Vec3f voxelColor = cv.color.toFloat() / 255.0f;
                hitColor = hitColor + voxelColor * alpha;
                aalpha += cv.active;
            }
            
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                tDist = tDist + tDelta.x;
                if (tMax.x > maxDist) break;
                currentVoxel.x += step.x;
                tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) {
                tDist = tDist + tDelta.y;
                currentVoxel.y += step.y;
                tMax.y += tDelta.y;
            }
            else {
                tDist = tDist + tDelta.z;
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
        return gridSize.x;
    }
    size_t getHeight() const {
        return gridSize.y;
    }
    size_t getDepth() const {
        return gridSize.z;
    }

    frame renderFrame(const Vec3f& CamPos, const Vec3f& lookAt, const Vec3f& up, float fov, size_t outW, size_t outH) {
        TIME_FUNCTION; 
        Vec3f forward = (lookAt - CamPos).normalized();
        Vec3f right = forward.cross(up).normalized();
        Vec3f upCor = right.cross(forward).normalized();
        float aspect = outW / outH;
        float fovRad = radians(fov);
        float viewH = 2 * tan(fovRad / 2);
        float viewW = viewH * aspect;
        float maxDist = gridSize.lengthSquared() / 2;
        frame outFrame = frame(outH,outW, frame::colormap::RGB);
        std::vector<uint8_t> colorBuffer(outW * outH * 3);
        #pragma omp parallel for
        for (size_t y = 0; y < outH; y++) {
            float v = (y + 0.5) / outH - 0.5;
            for (size_t x = 0; x < outW; x++) {
                float u = (x + 0.5) / outW - 0.5;
                Vec3f rayDir = (forward + right * (u + viewW) + upCor * (v * viewH)).normalized();
                Vec3f hitColor = Vec3f(0,0,0);
                bool hit = castRay(CamPos, rayDir, maxDist, hitColor);
                size_t idx = y*outW+x;
                if (!hit) {
                    hitColor = Vec3f(0.1,0.1,1.0f);
                } else {
                    std::cout << "hit";
                }
                colorBuffer[idx + 0] = static_cast<uint8_t>(hitColor.x * 255);
                colorBuffer[idx + 1] = static_cast<uint8_t>(hitColor.y * 255);
                colorBuffer[idx + 2] = static_cast<uint8_t>(hitColor.z * 255);
            }
        }
        std::cout <<  std::endl;
        outFrame.setData(colorBuffer);
        return outFrame;
    }

    void renderOut(std::vector<uint8_t>& output, size_t& outwidth, size_t& outheight, const Camera& cam) {
        TIME_FUNCTION; 
        Vec3f forward = (cam.posfor.direction - cam.posfor.origin).normalized();
        Vec3f right = forward.cross(cam.up).normalized();
        Vec3f upCor = right.cross(forward).normalized();
        float aspect = outwidth / outheight;
        float fovRad = radians(cam.fov);
        float viewH = 2 * tan(fovRad / 2);
        float viewW = viewH * aspect;
        float maxDist = gridSize.lengthSquared() / 2;
        //frame outFrame = frame(outH,outW, frame::colormap::RGB);
        std::vector<uint8_t> colorBuffer(outwidth * outheight * 3);
        #pragma omp parallel for
        for (size_t y = 0; y < outheight; y++) {
            float v = (y + 0.5) / outheight - 0.5;
            for (size_t x = 0; x < outwidth; x++) {
                float u = (x + 0.5) / outwidth - 0.5;
                Vec3f rayDir = (forward + right * (u + viewW) + upCor * (v * viewH)).normalized();
                Vec3f hitColor = Vec3f(0,0,0);
                bool hit = castRay(cam.posfor.origin, rayDir, maxDist, hitColor);
                size_t idx = y*outwidth+x;
                if (!hit) {
                    hitColor = Vec3f(0.1,0.1,1.0f);
                } else {
                    std::cout << "hit";
                }
                colorBuffer[idx + 0] = static_cast<uint8_t>(hitColor.x * 255);
                colorBuffer[idx + 1] = static_cast<uint8_t>(hitColor.y * 255);
                colorBuffer[idx + 2] = static_cast<uint8_t>(hitColor.z * 255);
            }
        }
        std::cout <<  std::endl;
        output = colorBuffer;
        // TIME_FUNCTION; 
        // output.resize(outwidth * outheight * 3);
        // Vec3f backgroundColor(0.1f, 0.1f, 1.0f);
        // float maxDistance = sqrt(gridSize.lengthSquared()) * 2;
        // //float maxDistance = std::sqrt(width*width + height*height + depth*depth) * 2.0f;
        // // std::vector<Vec3f> dirs = genPixelDirs(cam.posfor.origin, cam.posfor.direction, outwidth, outheight, cam.fov);

        // for (size_t y = 0; y < outheight; y++) {        
        //     float yout = y * outwidth;
        //     for (size_t x = 0; x < outwidth; x++) {
        //         // Vec3f rayDir = dirs[y*outwidth + x];
        //         // Vec3f hitColor = Vec3f(0,0,0);
        //         Vec3f rayDir = perPixelRayDir(x, y, outwidth, outheight, cam);
        //         Ray3f ray(cam.posfor.origin, rayDir);
        //         Vec3f hitPos;
        //         Vec3f hitNorm;
        //         Vec3f hitColor;
        //         bool hit = castRay(cam.posfor.origin, rayDir, maxDistance, hitColor);

        //         Vec3f finalColor;
        //         if (!hit) {
        //             finalColor = backgroundColor;
        //         } else {
        //             finalColor = hitColor;
        //         }

        //         finalColor = finalColor.clamp(0, 1);
        //         size_t pixelIndex = (yout + x) * 3;
        //         output[pixelIndex + 0] = static_cast<uint8_t>(finalColor.x * 255);
        //         output[pixelIndex + 1] = static_cast<uint8_t>(finalColor.y * 255);
        //         output[pixelIndex + 2] = static_cast<uint8_t>(finalColor.z * 255);
        //     }
        // }
    }
};

#endif
