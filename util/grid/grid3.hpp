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

    void set(size_t x, size_t y, size_t z, float active, Vec3ui8 color) {
        //expand grid if needed. 
        if (x >= 0 || y >= 0 || z >= 0) {
            if (!(x < width)) {
                width = x;
                voxels.resize(width*height*depth);
            }
            else if (!(y < height)) {
                height = y;
                voxels.resize(width*height*depth);
            }
            else if (!(z < depth)) {
                depth = z;
                voxels.resize(width*height*depth);
            }

            Voxel& v = get(x, y, z);
            v.active = std::clamp(active, 0.0f, 1.0f);
            v.color = color;
        }
    }

    bool inGrid(Vec3T voxl) {
        return (voxl > 0 && voxl.x < width && voxl.y < height && voxl.z < depth);
    }

    bool rayCast(const Ray3f& ray, float maxDistance, Vec3f hitPos, Vec3f hitNormal, Vec3f& hitColor) {
        hitColor = Vec3f(0,0,0);
        Vec3f rayDir = ray.direction;
        Vec3f rayOrigin = ray.origin;
        Vec3T currentVoxel = rayOrigin.floorToT();


        //important note: voxels store a float of "active" which is to be between 0 and 1, with <epsilon being inactive and <1 being transparent.
        //as progression occurs, add to hitColor all passed voxels multiplied by active. once active reaches 1, stop progression.
        //if active doesnt reach 1 before the edge of the grid is reached, then return the total.
        //always return true if any active voxels are hit
        //Ray3T and Vec3T are size_t.
        Vec3f step;
        Vec3f tMax;
        Vec3f tDelta;
        Vec3T cvoxel = ray.origin.floor();
        //initialization
        if (!inGrid(cvoxel)) {
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

            //update to also include z in this
        }

        /*
        Finally, we compute tDeltaX and tDeltaY. TDeltaX indicates how far along the ray we must move
        (in units of t) for the horizontal component of such a movement to equal the width of a voxel. Similarly,
        we store in tDeltaY the amount of movement along the ray which has a vertical component equal to the
        height of a voxel.
        */

        //also include tDeltaZ in this.

        /*loop {
        if(tMaxX < tMaxY) {
        tMaxX= tMaxX + tDeltaX;
        X= X + stepX;
        } else {
        tMaxY= tMaxY + tDeltaY;
        Y= Y + stepY;
        }
        NextVoxel(X,Y);
        }*/

        /*
        We loop until either we find a voxel with a non-empty object list or we fall out of the end of the grid.
        Extending the algorithm to three dimensions simply requires that we add the appropriate z variables and
        find the minimum of tMaxX, tMaxY and tMaxZ during each iteration. This results in:
        list= NIL;
        do {
        if(tMaxX < tMaxY) {
        if(tMaxX < tMaxZ) {
        X= X + stepX;
        if(X == justOutX)
        return(NIL); 
        tMaxX= tMaxX + tDeltaX;
        } else {
        Z= Z + stepZ;
        if(Z == justOutZ)
        return(NIL);
        tMaxZ= tMaxZ + tDeltaZ;
        }
        } else {
        if(tMaxY < tMaxZ) {
        Y= Y + stepY;
        if(Y == justOutY)
        return(NIL);
        tMaxY= tMaxY + tDeltaY;
        } else {
        Z= Z + stepZ;
        if(Z == justOutZ)
        return(NIL);
        tMaxZ= tMaxZ + tDeltaZ;
        }
        }
        list= ObjectList[X][Y][Z];
        } while(list == NIL);
        return(list);*/
    }
};

#endif