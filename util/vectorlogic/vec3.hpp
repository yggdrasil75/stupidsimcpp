#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <algorithm>
#include <string>
#include <ostream>

class Vec3 {
public:
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3(float scalar) : x(scalar), y(scalar), z(scalar) {}

    Vec3(const class Vec2& vec2, float z = 0.0f);

    Vec3& move(const Vec3 newpos) {
        x = newpos.x;
        y = newpos.y;
        z = newpos.z;
        return *this;
    }

    // Arithmetic operations
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }
    
    Vec3 operator/(const Vec3& other) const {
        return Vec3(x / other.x, y / other.y, z / other.z);
    }

    Vec3 operator+(float scalar) const {
        return Vec3(x + scalar, y + scalar, z + scalar);
    }

    Vec3 operator-(float scalar) const {
        return Vec3(x - scalar, y - scalar, z - scalar);
    }
    
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    Vec3& operator=(float scalar) {
        x = y = z = scalar;
        return *this;
    }
    
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    
    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    
    Vec3& operator*=(const Vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }
    
    Vec3& operator/=(const Vec3& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        return *this;
    }
    
    Vec3& operator+=(float scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }
    
    Vec3& operator-=(float scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }
    
    Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    
    Vec3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    float lengthSquared() const {
        return x * x + y * y + z * z;
    }
    
    float distance(const Vec3& other) const {
        return (*this - other).length();
    }
    
    float distanceSquared(const Vec3& other) const {
        Vec3 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

    Vec3 normalized() const {
        float len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }
    
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    bool operator!=(const Vec3& other) const {
        return x != other.x || y != other.y || z != other.z;
    }
    
    bool operator<(const Vec3& other) const {
        return (x < other.x) || 
               (x == other.x && y < other.y) || 
               (x == other.x && y == other.y && z < other.z);
    }
    
    bool operator<=(const Vec3& other) const {
        return (x < other.x) || 
               (x == other.x && y < other.y) || 
               (x == other.x && y == other.y && z <= other.z);
    }
    
    bool operator>(const Vec3& other) const {
        return (x > other.x) || 
               (x == other.x && y > other.y) || 
               (x == other.x && y == other.y && z > other.z);
    }
    
    bool operator>=(const Vec3& other) const {
        return (x > other.x) || 
               (x == other.x && y > other.y) || 
               (x == other.x && y == other.y && z >= other.z);
    }
    
    Vec3 abs() const {
        return Vec3(std::abs(x), std::abs(y), std::abs(z));
    }
    
    Vec3 floor() const {
        return Vec3(std::floor(x), std::floor(y), std::floor(z));
    }
    
    Vec3 ceil() const {
        return Vec3(std::ceil(x), std::ceil(y), std::ceil(z));
    }
    
    Vec3 round() const {
        return Vec3(std::round(x), std::round(y), std::round(z));
    }
    
    Vec3 min(const Vec3& other) const {
        return Vec3(std::min(x, other.x), std::min(y, other.y), std::min(z, other.z));
    }
    
    Vec3 max(const Vec3& other) const {
        return Vec3(std::max(x, other.x), std::max(y, other.y), std::max(z, other.z));
    }
    
    Vec3 clamp(const Vec3& minVal, const Vec3& maxVal) const {
        return Vec3(
            std::clamp(x, minVal.x, maxVal.x),
            std::clamp(y, minVal.y, maxVal.y),
            std::clamp(z, minVal.z, maxVal.z)
        );
    }
    
    Vec3 clamp(float minVal, float maxVal) const {
        return Vec3(
            std::clamp(x, minVal, maxVal),
            std::clamp(y, minVal, maxVal),
            std::clamp(z, minVal, maxVal)
        );
    }
    
    bool isZero(float epsilon = 1e-10f) const {
        return std::abs(x) < epsilon && std::abs(y) < epsilon && std::abs(z) < epsilon;
    }
    
    bool equals(const Vec3& other, float epsilon = 1e-10f) const {
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon;
    }
    
    friend Vec3 operator+(float scalar, const Vec3& vec) {
        return Vec3(scalar + vec.x, scalar + vec.y, scalar + vec.z);
    }
    
    friend Vec3 operator-(float scalar, const Vec3& vec) {
        return Vec3(scalar - vec.x, scalar - vec.y, scalar - vec.z);
    }
    
    friend Vec3 operator*(float scalar, const Vec3& vec) {
        return Vec3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }
    
    friend Vec3 operator/(float scalar, const Vec3& vec) {
        return Vec3(scalar / vec.x, scalar / vec.y, scalar / vec.z);
    }

    Vec3 reflect(const Vec3& normal) const {
        return *this - 2.0f * this->dot(normal) * normal;
    }
    
    Vec3 lerp(const Vec3& other, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        return *this + (other - *this) * t;
    }
    
    Vec3 slerp(const Vec3& other, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        float dot = this->dot(other);
        dot = std::clamp(dot, -1.0f, 1.0f);
        
        float theta = std::acos(dot) * t;
        Vec3 relative = other - *this * dot;
        relative = relative.normalized();
        
        return (*this * std::cos(theta)) + (relative * std::sin(theta));
    }
    
    Vec3 rotateX(float angle) const {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Vec3(x, y * cosA - z * sinA, y * sinA + z * cosA);
    }
    
    Vec3 rotateY(float angle) const {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Vec3(x * cosA + z * sinA, y, -x * sinA + z * cosA);
    }
    
    Vec3 rotateZ(float angle) const {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Vec3(x * cosA - y * sinA, x * sinA + y * cosA, z);
    }
    
    float angleTo(const Vec3& other) const {
        return std::acos(this->dot(other) / (this->length() * other.length()));
    }
    
    float& operator[](int index) {
        return (&x)[index];
    }
    
    const float& operator[](int index) const {
        return (&x)[index];
    }
    
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
    }

    struct Hash {
        std::size_t operator()(const Vec3& v) const {
            return std::hash<float>()(v.x) ^ (std::hash<float>()(v.y) << 1) ^ (std::hash<float>()(v.z) << 2);
        }
    };
};

inline std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    os << vec.toString();
    return os;
}

namespace std {
    template<>
    struct hash<Vec3> {
        size_t operator()(const Vec3& v) const {
            return hash<float>()(v.x) ^ (hash<float>()(v.y) << 1) ^ (hash<float>()(v.z) << 2);
        }
    };
}

#endif