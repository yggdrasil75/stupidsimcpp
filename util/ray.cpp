#ifndef RAY_HPP
#define RAY_HPP

#include "Vec2.hpp"
#include "Vec3.hpp"
#include "Vec4.hpp"
#include "ray2.hpp"
#include "ray3.hpp"
#include "ray4.hpp"

// Stream operators for rays
inline std::ostream& operator<<(std::ostream& os, const Ray2& ray) {
    os << ray.toString();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Ray3& ray) {
    os << ray.toString();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Ray4& ray) {
    os << ray.toString();
    return os;
}

#endif