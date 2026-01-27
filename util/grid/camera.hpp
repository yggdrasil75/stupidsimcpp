#ifndef camera_hpp
#define camera_hpp

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

#endif