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

    static Mat4f lookAt(Vec3f const& eye, Vec3f const& center, Vec3f const& up) {
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

    std::vector<Vec3f> genPixelDirs(Vec3f pos, Vec3f dir, size_t imgWidth, size_t imgHeight, float fov) {
        std::vector<Vec3f> dirs(imgWidth * imgHeight);
        float fovRad = radians(fov);
        float tanFov = tan(fovRad * 0.5);
        float aspect = static_cast<float>(imgWidth) / static_cast<float>(imgHeight);
        Vec3f worldUp(0, 1, 0);
        Vec3f camRight = worldUp.cross(dir).normalized();
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

    bool rayCast(const Ray3f& ray, float maxDistance, Vec3f hitPos, Vec3f hitNormal, Vec3f& hitColor) {
        hitColor = Vec3f(0,0,0);
        Vec3f rayDir = ray.direction;
        Vec3f rayOrigin = ray.origin;
        Vec3T currentVoxel = rayOrigin.floorToT();
        Vec3i step;
        step.x = (rayDir.x > 0) ? 1 : -1;
        step.y = (rayDir.y > 0) ? 1 : -1;
        step.z = (rayDir.z > 0) ? 1 : -1;
        Vec3f tMax;
        Vec3f tDelta;
        bool startOut = false;

        tDelta.x = std::abs(1.0 / rayDir.x);
        tDelta.y = std::abs(1.0 / rayDir.y);
        tDelta.z = std::abs(1.0 / rayDir.z);
        tMax = mix(((rayOrigin - currentVoxel.toFloat()) / -rayDir).toFloat(), 
                (((currentVoxel.toFloat() + 1) - rayOrigin) / rayDir).toFloat(), 
                rayDir.mask([](float x, float value) { return x > 0; }, 0));
        
        if (!inGrid(rayOrigin)) {
            startOut = true;
            Vec3f tBMin;
            Vec3f tBMax;
            
            tBMin.x = (0.0 - rayOrigin.x) / rayDir.x;
            tBMax.x = (width - rayOrigin.x) / rayDir.x;
            if (tBMin.x > tBMax.x) std::swap(tBMin.x, tBMax.x);
            
            tBMin.y = (0.0 - rayOrigin.y) / rayDir.y;
            tBMax.y = (height - rayOrigin.y) / rayDir.y;
            if (tBMin.y > tBMax.y) std::swap(tBMin.y, tBMax.y);
            
            tBMin.z = (0.0 - rayOrigin.z) / rayDir.z;
            tBMax.z = (depth - rayOrigin.z) / rayDir.z;
            if (tBMin.z > tBMax.z) std::swap(tBMin.z, tBMax.z);
            
            float tEntry = tBMin.maxComp();
            float tExit = tBMax.minComp();
            
            if (tEntry > tExit || tExit < 0.0) return false;
            if (tEntry < 0.0) tEntry = 0.0;

            if (tEntry > 0.0) {
                rayOrigin = rayOrigin + rayDir * tEntry;
                currentVoxel = rayOrigin.floorToT();
                tMax = mix(((currentVoxel.toFloat() + 1) - rayOrigin) / rayDir, 
                        (rayOrigin - currentVoxel) / -rayDir, 
                        rayDir.mask([](float x, float value) { return x > 0; }, 0));
            }
        }
        
        // if (startOut && !inGrid(currentVoxel)) {
        //     std::cout << "grid edge not found. " << currentVoxel << std::endl;
        // }

        float tDist = 0.0;

        // Main DDA loop
        while (inGrid(currentVoxel) && tDist < maxDistance) {
            Voxel& voxel = get(currentVoxel);

            // Ignore alpha - treat any voxel with active > 0 as solid
            if (voxel.active > EPSILON) {
                // Convert color from 0-255 to 0-1 range
                Vec3f voxelColor(
                    static_cast<float>(voxel.color.x / 255.0), 
                    static_cast<float>(voxel.color.y / 255.0), 
                    static_cast<float>(voxel.color.z / 255.0)
                );
                
                // No alpha blending - just take the first solid voxel's color
                hitColor = voxelColor;
                hitPos = rayOrigin + rayDir * tDist;
                
                // Determine which face was hit
                if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                    hitNormal = Vec3f(-step.x, 0.0, 0.0);
                } else if (tMax.y <= tMax.x && tMax.y <= tMax.z) {
                    hitNormal = Vec3f(0.0, -step.y, 0.0);
                } else {
                    hitNormal = Vec3f(0.0, 0.0, -step.z);
                }
                
                return true; // Return immediately on first solid hit
            }

            // Move to next voxel
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    tDist = tMax.x;
                    tMax.x += tDelta.x;
                    currentVoxel.x += step.x;
                } else {
                    tDist = tMax.z;
                    tMax.z += tDelta.z;
                    currentVoxel.z += step.z;
                }
            } else {
                if (tMax.y < tMax.z) {
                    tDist = tMax.y;
                    tMax.y += tDelta.y;
                    currentVoxel.y += step.y;
                } else {
                    tDist = tMax.z;
                    tMax.z += tDelta.z;
                    currentVoxel.z += step.z;
                }
            }
        }
        
        return false;
    }
    // bool rayCast(const Ray3f& ray, float maxDistance, Vec3f hitPos, Vec3f hitNormal, Vec3f& hitColor) {
    //     hitColor = Vec3f(0,0,0);
    //     Vec3f rayDir = ray.direction;
    //     Vec3f rayOrigin = ray.origin;
    //     Vec3T currentVoxel = rayOrigin.floorToT();
    //     Vec3i step;
    //     step.x = (rayDir.x > 0) ? 1 : -1;
    //     step.y = (rayDir.y > 0) ? 1 : -1;
    //     step.z = (rayDir.z > 0) ? 1 : -1;
    //     Vec3f tMax;
    //     Vec3f tDelta;
    //     bool startOut = false;

    //     tDelta.x = std::abs(1.0 / rayDir.x);
    //     tDelta.y = std::abs(1.0 / rayDir.y);
    //     tDelta.z = std::abs(1.0 / rayDir.z);
    //     tMax = mix(((rayOrigin - currentVoxel.toFloat()) / -rayDir).toFloat(), (((currentVoxel.toFloat() + 1) - rayOrigin) / rayDir).toFloat(), rayDir.mask([](float x, float value) { return x > 0; }, 0));
    //     if (!inGrid(rayOrigin)) {
    //         startOut = true;
    //         /*
    //         The initialization phase begins by identifying the voxel in which the ray origin, →
    //         u, is found. If the ray origin is outside the grid, we find the point in which the ray enters the grid and take the adjacent voxel. The integer
    //         variables X and Y are initialized to the starting voxel coordinates. In addition, the variables stepX and
    //         stepY are initialized to either 1 or -1 indicating whether X and Y are incremented or decremented as the
    //         ray crosses voxel boundaries (this is determined by the sign of the x and y components of →
    //         v).
    //         Next, we determine the value of t at which the ray crosses the first vertical voxel boundary and
    //         store it in variable tMaxX. We perform a similar computation in y and store the result in tMaxY. The
    //         minimum of these two values will indicate how much we can travel along the ray and still remain in the
    //         current voxel.
    //         */

    //         Vec3f tBMin;
    //         Vec3f tBMax;
    //         tBMin.x = (0.0 - rayOrigin.x) / rayDir.x;
    //         tBMax.x = (width - rayOrigin.x) / rayDir.x;
    //         if (tBMin.x > tBMax.x) std::swap(tBMin.x, tBMax.x);
    //         tBMin.y = (0.0 - rayOrigin.y) / rayDir.y;
    //         tBMax.y = (height - rayOrigin.y) / rayDir.y;
    //         if (tBMin.y > tBMax.y) std::swap(tBMin.y, tBMax.y);
    //         tBMin.z = (0.0 - rayOrigin.z) / rayDir.z;
    //         tBMax.z = (depth - rayOrigin.z) / rayDir.z;
    //         if (tBMin.z > tBMax.z) std::swap(tBMin.z, tBMax.z);
    //         float tEntry = tBMin.maxComp();
    //         float tExit = tBMax.minComp();
    //         if (tEntry > tExit || tExit < 0.0) return false;
    //         if (tEntry < 0.0) tEntry = 0.0;

    //         if (tEntry > 0.0) {
    //             rayOrigin = rayOrigin + rayDir * tEntry;
    //             currentVoxel = rayOrigin.floorToT();
    //             tMax = mix(((currentVoxel.toFloat() + 1) - rayOrigin) / rayDir, (rayOrigin - currentVoxel) / -rayDir, rayDir.mask([](float x, float value) { return x > 0; }, 0) );
    //         }

    //     }
    //     if (startOut && !inGrid(currentVoxel)) std::cout << "grid edge not found. " << currentVoxel << std::endl;

    //     float aalpha = 0.0;
    //     bool hit = false;
    //     float tDist = 0.0;

    //     /*
    //     Finally, we compute tDeltaX and tDeltaY. TDeltaX indicates how far along the ray we must move
    //     (in units of t) for the horizontal component of such a movement to equal the width of a voxel. Similarly,
    //     we store in tDeltaY the amount of movement along the ray which has a vertical component equal to the
    //     height of a voxel.
    //     */
    //     while (inGrid(currentVoxel) && tDist < maxDistance) {
    //         Voxel& voxel = get(currentVoxel);

    //         if (voxel.active > EPSILON) {
    //             Vec3f voxelColor(static_cast<float>(voxel.color.x / 255.0), static_cast<float>(voxel.color.y / 255.0), static_cast<float>(voxel.color.z / 255.0));
    //             float contribution = voxel.active * (1.0 - aalpha);
    //             hitColor = hitColor + voxelColor * contribution;
    //             aalpha += contribution;
    //             hitPos = rayOrigin + rayDir * tDist;
    //             if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
    //                 hitNormal = Vec3f(-step.x, 0.0, 0.0);
    //             } else if (tMax.y <= tMax.x && tMax.y <= tMax.z) {
    //                 hitNormal = Vec3f(0.0, -step.y, 0.0);
    //             } else {
    //                 hitNormal = Vec3f(0.0, 0.0, -step.z);
    //             }
    //         }
    //         if (aalpha > EPSILON) {
    //             hit = true;
    //         }

    //         if (tMax.x < tMax.y) {
    //             if (tMax.x < tMax.z) {
    //                 tDist = tMax.x;
    //                 tMax.x += tDelta.x;
    //                 currentVoxel.x += step.x;
    //             } else {
    //                 tDist = tMax.z;
    //                 tMax.z += tDelta.z;
    //                 currentVoxel.z += step.z;
    //             }
    //         } else {
    //             if (tMax.y < tMax.z) {
    //                 tDist = tMax.y;
    //                 tMax.y += tDelta.y;
    //                 currentVoxel.y += step.y;
    //             } else {
    //                 tDist = tMax.z;
    //                 tMax.z += tDelta.z;
    //                 currentVoxel.z += step.z;
    //             }
    //         }
    //     }
    //     // if (aalpha > EPSILON) {
    //     //     std::cout << "hit at: " << currentVoxel << " with value of " << aalpha << std::endl;
    //     // }
    //     return hit;
    // }

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
        output.resize(outwidth * outheight * 3);
        Vec3f backgroundColor(0.1f, 0.1f, 0.1f);
        float maxDistance = std::sqrt(width*width + height*height + depth*depth) * 2.0f;
        
        for (size_t y = 0; y < outheight; y++) {
            for (size_t x = 0; x < outwidth; x++) {
                Vec3f rayDir = perPixelRayDir(x, y, outwidth, outheight, cam);
                Ray3f ray(cam.posfor.origin, rayDir);
                Vec3f hitPos;
                Vec3f hitNorm;
                Vec3f hitColor;
                bool hit = rayCast(ray, maxDistance, hitPos, hitNorm, hitColor);

                Vec3f finalColor;
                if (!hit) {
                    finalColor = backgroundColor;
                } else {
                    finalColor = hitColor;
                }

                finalColor = finalColor.clamp(0, 1);
                size_t pixelIndex = (y * outwidth + x) * 3;
                output[pixelIndex + 0] = static_cast<uint8_t>(finalColor.x * 255);
                output[pixelIndex + 1] = static_cast<uint8_t>(finalColor.y * 255);
                output[pixelIndex + 2] = static_cast<uint8_t>(finalColor.z * 255);
            }
        }
    }
};

#endif
