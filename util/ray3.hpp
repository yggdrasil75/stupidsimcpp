#ifndef RAY3_HPP
#define RAY3_HPP

#include "vectorlogic/vec3.hpp"

class Ray3 {
public:
    Vec3 origin;
    Vec3 direction;
    
    Ray3() : origin(Vec3()), direction(Vec3(1, 0, 0)) {}
    Ray3(const Vec3& origin, const Vec3& direction) 
        : origin(origin), direction(direction.normalized()) {}
    
    // Get point at parameter t along the ray
    Vec3 at(float t) const {
        return origin + direction * t;
    }
    
    // Reflect ray off a surface with given normal
    Ray3 reflect(const Vec3& point, const Vec3& normal) const {
        Vec3 reflectedDir = direction.reflect(normal);
        return Ray3(point, reflectedDir);
    }
    
    // Check if ray intersects with a sphere
    bool intersectsSphere(const Vec3& center, float radius, float& t1, float& t2) const {
        Vec3 oc = origin - center;
        float a = direction.dot(direction);
        float b = 2.0f * oc.dot(direction);
        float c = oc.dot(oc) - radius * radius;
        
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) {
            return false;
        }
        
        discriminant = std::sqrt(discriminant);
        t1 = (-b - discriminant) / (2.0f * a);
        t2 = (-b + discriminant) / (2.0f * a);
        
        return true;
    }
    
    // Check if ray intersects with a plane (defined by point and normal)
    bool intersectsPlane(const Vec3& planePoint, const Vec3& planeNormal, float& t) const {
        float denom = planeNormal.dot(direction);
        
        if (std::abs(denom) < 1e-10f) {
            return false; // Ray is parallel to plane
        }
        
        t = planeNormal.dot(planePoint - origin) / denom;
        return t >= 0;
    }
    
    // Get the distance from a point to this ray
    float distanceToPoint(const Vec3& point) const {
        Vec3 pointToOrigin = point - origin;
        Vec3 crossProduct = direction.cross(pointToOrigin);
        return crossProduct.length() / direction.length();
    }
    
    // Transform ray by a 4x4 matrix (for perspective/affine transformations)
    Ray3 transform(const class Mat4& matrix) const;
    
    std::string toString() const {
        return "Ray3(origin: " + origin.toString() + ", direction: " + direction.toString() + ")";
    }
};

#endif