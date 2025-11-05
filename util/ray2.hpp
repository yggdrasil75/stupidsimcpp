#ifndef RAY2_HPP
#define RAY2_HPP

#include "Vec2.hpp"

class Ray2 {
public:
    Vec2 origin;
    Vec2 direction;
    
    Ray2() : origin(Vec2()), direction(Vec2(1, 0)) {}
    Ray2(const Vec2& origin, const Vec2& direction) 
        : origin(origin), direction(direction.normalized()) {}
    
    // Get point at parameter t along the ray
    Vec2 at(float t) const {
        return origin + direction * t;
    }
    
    // Reflect ray off a surface with given normal
    Ray2 reflect(const Vec2& point, const Vec2& normal) const {
        Vec2 reflectedDir = direction.reflect(normal);
        return Ray2(point, reflectedDir);
    }
    
    // Check if ray intersects with a circle
    bool intersectsCircle(const Vec2& center, float radius, float& t1, float& t2) const {
        Vec2 oc = origin - center;
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
    
    // Get the distance from a point to this ray
    float distanceToPoint(const Vec2& point) const {
        Vec2 pointToOrigin = point - origin;
        float projection = pointToOrigin.dot(direction);
        Vec2 closestPoint = origin + direction * projection;
        return point.distance(closestPoint);
    }
    
    std::string toString() const {
        return "Ray2(origin: " + origin.toString() + ", direction: " + direction.toString() + ")";
    }
};

#endif