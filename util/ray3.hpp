#ifndef RAY3_HPP
#define RAY3_HPP

#include "vectorlogic/vec3.hpp"

template<typename T>
class Ray3 {
public:
    Vec3<T> origin;
    Vec3<T> direction;
    
    Ray3() : origin(Vec3<T>()), direction(Vec3<T>(1, 0, 0)) {}
    Ray3(const Vec3<T>& origin, const Vec3<T>& direction) 
        : origin(origin), direction(direction.normalized()) {}
    
    // Get point at parameter t along the ray
    Vec3<T> at(float t) const {
        return origin + direction * t;
    }
    
    // Reflect ray off a surface with given normal
    Ray3 reflect(const Vec3<T>& point, const Vec3<T>& normal) const {
        Vec3<T> reflectedDir = direction.reflect(normal);
        return Ray3(point, reflectedDir);
    }
    
    // Check if ray intersects with a sphere
    bool intersectsSphere(const Vec3<T>& center, T radius, T& t1, T& t2) const {
        Vec3<T> oc = origin - center;
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
    bool intersectsPlane(const Vec3<T>& planePoint, const Vec3<T>& planeNormal, T& t) const {
        float denom = planeNormal.dot(direction);
        
        if (std::abs(denom) < 1e-10f) {
            return false; // Ray is parallel to plane
        }
        
        t = planeNormal.dot(planePoint - origin) / denom;
        return t >= 0;
    }
    
    // Get the distance from a point to this ray
    float distanceToPoint(const Vec3<T>& point) const {
        Vec3<T> pointToOrigin = point - origin;
        Vec3<T> crossProduct = direction.cross(pointToOrigin);
        return crossProduct.length() / direction.length();
    }
    
    // Transform ray by a 4x4 matrix (for perspective/affine transformations)
    Ray3 transform(const class Mat4& matrix) const;
    
    std::string toString() const {
        return "Ray3(origin: " + origin.toString() + ", direction: " + direction.toString() + ")";
    }
};

using Ray3f = Ray3<float>;

#endif