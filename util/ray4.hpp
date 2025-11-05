#ifndef RAY4_HPP
#define RAY4_HPP

#include "vec4.hpp"

class Ray4 {
public:
    Vec4 origin;
    Vec4 direction;
    
    Ray4() : origin(Vec4()), direction(Vec4(1, 0, 0, 0)) {}
    Ray4(const Vec4& origin, const Vec4& direction) 
        : origin(origin), direction(direction.normalized()) {}
    
    // Get point at parameter t along the ray (in 4D space)
    Vec4 at(float t) const {
        return origin + direction * t;
    }
    
    // Get 3D projection of the ray (homogeneous coordinates)
    // Ray3 projectTo3D() const {
    //     Vec3 projOrigin = origin.homogenized().xyz();
    //     Vec3 projDirection = direction.homogenized().xyz().normalized();
    //     return Ray3(projOrigin, projDirection);
    // }
    
    // Get the distance from a point to this ray in 4D space
    float distanceToPoint(const Vec4& point) const {
        Vec4 pointToOrigin = point - origin;
        float projection = pointToOrigin.dot(direction);
        Vec4 closestPoint = origin + direction * projection;
        return point.distance(closestPoint);
    }
    
    // Check if this 4D ray intersects with a 3D hyperplane
    bool intersectsHyperplane(const Vec4& planePoint, const Vec4& planeNormal, float& t) const {
        float denom = planeNormal.dot(direction);
        
        if (std::abs(denom) < 1e-10f) {
            return false; // Ray is parallel to hyperplane
        }
        
        t = planeNormal.dot(planePoint - origin) / denom;
        return true;
    }
    
    std::string toString() const {
        return "Ray4(origin: " + origin.toString() + ", direction: " + direction.toString() + ")";
    }
};

#endif