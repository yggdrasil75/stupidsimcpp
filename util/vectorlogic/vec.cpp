#ifndef vec_hpp
#define vec_hpp

#include "Vec4.hpp"
#include "Vec3.hpp"
#include "Vec2.hpp"

Vec4::Vec4(const Vec3& vec3, float w) : x(vec3.x), y(vec3.y), z(vec3.z), w(w) {}
Vec3::Vec3(const Vec2& vec2, float z) : x(vec2.x), y(vec2.y), z(z) {}

Vec3 Vec4::xyz() const {
    return Vec3(x, y, z);
}

Vec3 Vec4::rgb() const {
    return Vec3(r, g, b);
}

#endif