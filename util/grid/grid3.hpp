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

class VoxelGrid {
private:
    size_t width, height, depth;
    std::vector<Voxel> voxels;
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

        //initialization
        tDelta.x = std::abs(1.0 / rayDir.x);
        tDelta.y = std::abs(1.0 / rayDir.y);
        tDelta.z = std::abs(1.0 / rayDir.z);
        tMax = mix((rayOrigin - currentVoxel) / -rayDir, ((currentVoxel + 1) - rayOrigin) / rayDir, rayDir > 0);
        if (!inGrid(rayOrigin)) {
            /*
            The initialization phase begins by identifying the voxel in which the ray origin, →
            u, is found. If the ray origin is outside the grid, we find the point in which the ray enters the grid and take the adjacent voxel. The integer
            variables X and Y are initialized to the starting voxel coordinates. In addition, the variables stepX and
            stepY are initialized to either 1 or -1 indicating whether X and Y are incremented or decremented as the
            ray crosses voxel boundaries (this is determined by the sign of the x and y components of →
            v).
            Next, we determine the value of t at which the ray crosses the first vertical voxel boundary and
            store it in variable tMaxX. We perform a similar computation in y and store the result in tMaxY. The
            minimum of these two values will indicate how much we can travel along the ray and still remain in the
            current voxel.
            */

            float tEntry = 0.0;
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
                rayOrigin = rayOrigin + rayDir + tEntry;
                currentVoxel = rayOrigin.floorToT();
                tMax = mix(((currentVoxel + 1) - rayOrigin) / rayDir, (rayOrigin - currentVoxel) / -rayDir, rayDir > 0 );
            }
        }

        float aalpha = 0.0;
        bool hit = false;
        float tDist = 0.0;

        /*
        Finally, we compute tDeltaX and tDeltaY. TDeltaX indicates how far along the ray we must move
        (in units of t) for the horizontal component of such a movement to equal the width of a voxel. Similarly,
        we store in tDeltaY the amount of movement along the ray which has a vertical component equal to the
        height of a voxel.
        */
        while (inGrid(currentVoxel) && tDist < maxDistance) {
            Voxel& voxel = get(currentVoxel);

            if (voxel.active > EPSILON) {
                Vec3f voxelColor(static_cast<float>(voxel.color.x / 255.0), static_cast<float>(voxel.color.y / 255.0), static_cast<float>(voxel.color.z / 255.0));
                float contribution = voxel.active * (1.0 - aalpha);
                hitColor = hitColor + voxelColor * contribution;
                aalpha += contribution;
                hitPos = rayOrigin + rayDir * tDist;
                if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                    hitNormal = Vec3f(-step.x, 0.0, 0.0);
                } else if (tMax.y <= tMax.x && tMax.y <= tMax.z) {
                    hitNormal = Vec3f(0.0, -step.y, 0.0);
                } else {
                    hitNormal = Vec3f(0.0, 0.0, -step.z);
                }
            }
            if (aalpha > EPSILON) {
                hit = true;
            }

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
        return hit;
    }
};

#endif
