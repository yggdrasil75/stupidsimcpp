#ifndef VEC4_HPP
#define VEC4_HPP

#include "vec3.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <ostream>
#include <cstdint>

class Vec4 {
public:
    union {
        struct { float x, y, z, w; };
        struct { float r, g, b, a; };
        struct { float s, t, p, q; }; // For texture coordinates
    };
    
    // Constructors
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4(float scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}
    
    Vec4(const Vec3& rgb, float w = 1.0f) : x(rgb.x), y(rgb.y), z(rgb.z), w(w) {}
    static Vec4 RGB(float r, float g, float b, float a = 1.0f) { return Vec4(r, g, b, a); }
    static Vec4 RGBA(float r, float g, float b, float a) { return Vec4(r, g, b, a); }
    
    Vec4& recolor(const Vec4 newColor) {
        r = newColor.r;
        g = newColor.g;
        b = newColor.b;
        a = newColor.a;
        return *this;
    }
    
    Vec4 average(const Vec4& other) const {
        return Vec4((x+other.x)/2,(y+other.y)/2,(z+other.z)/2,(w+other.w)/2);
    }
    
    Vec4 operator+(const Vec4& other) const {
        return Vec4(x + other.x, y + other.y, z + other.z, w + other.w);
    }
    
    Vec4 operator-(const Vec4& other) const {
        return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
    }
    
    Vec4 operator*(const Vec4& other) const {
        return Vec4(x * other.x, y * other.y, z * other.z, w * other.w);
    }
    
    Vec4 operator/(const Vec4& other) const {
        return Vec4(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    Vec4 operator+(float scalar) const {
        return Vec4(x + scalar, y + scalar, z + scalar, w + scalar);
    }

    Vec4 operator-(float scalar) const {
        return Vec4(x - scalar, y - scalar, z - scalar, w - scalar);
    }
    
    Vec4 operator-() const {
        return Vec4(-x, -y, -z, -w);
    }

    Vec4 operator*(float scalar) const {
        return Vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    Vec4 operator/(float scalar) const {
        return Vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    Vec4& operator=(float scalar) {
        x = y = z = w = scalar;
        return *this;
    }
        
    Vec4& operator+=(const Vec4& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }
    
    Vec4& operator-=(const Vec4& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }
    
    Vec4& operator*=(const Vec4& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        w *= other.w;
        return *this;
    }
    
    Vec4& operator/=(const Vec4& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        w /= other.w;
        return *this;
    }
    
    Vec4& operator+=(float scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        w += scalar;
        return *this;
    }
    
    Vec4& operator-=(float scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        w -= scalar;
        return *this;
    }
    
    Vec4& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }
    
    Vec4& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    float dot(const Vec4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
    
    // 4D cross product (returns vector perpendicular to 3 given vectors in 4D space)
    Vec4 cross(const Vec4& v1, const Vec4& v2, const Vec4& v3) const {
        float a = v1.y * (v2.z * v3.w - v2.w * v3.z) -
                  v1.z * (v2.y * v3.w - v2.w * v3.y) +
                  v1.w * (v2.y * v3.z - v2.z * v3.y);
                  
        float b = -v1.x * (v2.z * v3.w - v2.w * v3.z) +
                   v1.z * (v2.x * v3.w - v2.w * v3.x) -
                   v1.w * (v2.x * v3.z - v2.z * v3.x);
                   
        float c = v1.x * (v2.y * v3.w - v2.w * v3.y) -
                  v1.y * (v2.x * v3.w - v2.w * v3.x) +
                  v1.w * (v2.x * v3.y - v2.y * v3.x);
                  
        float d = -v1.x * (v2.y * v3.z - v2.z * v3.y) +
                   v1.y * (v2.x * v3.z - v2.z * v3.x) -
                   v1.z * (v2.x * v3.y - v2.y * v3.x);
        
        return Vec4(a, b, c, d);
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z + w * w);
    }
    
    float lengthSquared() const {
        return x * x + y * y + z * z + w * w;
    }
    
    float distance(const Vec4& other) const {
        return (*this - other).length();
    }
    
    float distanceSquared(const Vec4& other) const {
        Vec4 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    Vec4 normalized() const {
        float len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }
    
    // Homogeneous normalization (divide by w)
    Vec4 homogenized() const {
        if (w != 0.0f) {
            return Vec4(x / w, y / w, z / w, 1.0f);
        }
        return *this;
    }
    
    // Clamp values between 0 and 1
    Vec4 clamped() const {
        return Vec4(
            std::clamp(r, 0.0f, 1.0f),
            std::clamp(g, 0.0f, 1.0f),
            std::clamp(b, 0.0f, 1.0f),
            std::clamp(a, 0.0f, 1.0f)
        );
    }
    
    // Convert to Vec3 (ignoring alpha)
    Vec3 toVec3() const {
        return Vec3(r, g, b);
    }
    
    // Convert to 8-bit color values
    void toUint8(uint8_t& red, uint8_t& green, uint8_t& blue, uint8_t& alpha) const {
        red = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255);
        green = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255);
        alpha = static_cast<uint8_t>(std::clamp(a, 0.0f, 1.0f) * 255);
    }
    
    void toUint8(uint8_t& red, uint8_t& green, uint8_t& blue) const {
        red = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255);
        green = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255);
    }
    // Get XYZ components as Vec3
    class Vec3 xyz() const;
    
    // Get RGB components as Vec3
    class Vec3 rgb() const;
    
    bool operator==(const Vec4& other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
    
    bool operator!=(const Vec4& other) const {
        return x != other.x || y != other.y || z != other.z || w != other.w;
    }
    
    bool operator<(const Vec4& other) const {
        return (x < other.x) || 
               (x == other.x && y < other.y) || 
               (x == other.x && y == other.y && z < other.z) ||
               (x == other.x && y == other.y && z == other.z && w < other.w);
    }
    
    bool operator<=(const Vec4& other) const {
        return (x < other.x) || 
               (x == other.x && y < other.y) || 
               (x == other.x && y == other.y && z < other.z) ||
               (x == other.x && y == other.y && z == other.z && w <= other.w);
    }
    
    bool operator>(const Vec4& other) const {
        return (x > other.x) || 
               (x == other.x && y > other.y) || 
               (x == other.x && y == other.y && z > other.z) ||
               (x == other.x && y == other.y && z == other.z && w > other.w);
    }
    
    bool operator>=(const Vec4& other) const {
        return (x > other.x) || 
               (x == other.x && y > other.y) || 
               (x == other.x && y == other.y && z > other.z) ||
               (x == other.x && y == other.y && z == other.z && w >= other.w);
    }
    
    Vec4 abs() const {
        return Vec4(std::abs(x), std::abs(y), std::abs(z), std::abs(w));
    }
    
    Vec4 floor() const {
        return Vec4(std::floor(x), std::floor(y), std::floor(z), std::floor(w));
    }
    
    Vec4 ceil() const {
        return Vec4(std::ceil(x), std::ceil(y), std::ceil(z), std::ceil(w));
    }
    
    Vec4 round() const {
        return Vec4(std::round(x), std::round(y), std::round(z), std::round(w));
    }
    
    Vec4 min(const Vec4& other) const {
        return Vec4(std::min(x, other.x), std::min(y, other.y), 
                   std::min(z, other.z), std::min(w, other.w));
    }
    
    Vec4 max(const Vec4& other) const {
        return Vec4(std::max(x, other.x), std::max(y, other.y), 
                   std::max(z, other.z), std::max(w, other.w));
    }
    
    Vec4 clamp(float minVal, float maxVal) const {
        return Vec4(
            std::clamp(x, minVal, maxVal),
            std::clamp(y, minVal, maxVal),
            std::clamp(z, minVal, maxVal),
            std::clamp(w, minVal, maxVal)
        );
    }
    
    // Color-specific clamping (clamps RGB between 0 and 1)
    Vec4 clampColor() const {
        return Vec4(
            std::clamp(r, 0.0f, 1.0f),
            std::clamp(g, 0.0f, 1.0f),
            std::clamp(b, 0.0f, 1.0f),
            std::clamp(a, 0.0f, 1.0f)
        );
    }
    
    bool isZero(float epsilon = 1e-10f) const {
        return std::abs(x) < epsilon && std::abs(y) < epsilon && 
               std::abs(z) < epsilon && std::abs(w) < epsilon;
    }
    
    bool equals(const Vec4& other, float epsilon = 1e-10f) const {
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon &&
               std::abs(w - other.w) < epsilon;
    }
    
    friend Vec4 operator+(float scalar, const Vec4& vec) {
        return Vec4(scalar + vec.x, scalar + vec.y, scalar + vec.z, scalar + vec.w);
    }
    
    friend Vec4 operator-(float scalar, const Vec4& vec) {
        return Vec4(scalar - vec.x, scalar - vec.y, scalar - vec.z, scalar - vec.w);
    }
    
    friend Vec4 operator*(float scalar, const Vec4& vec) {
        return Vec4(scalar * vec.x, scalar * vec.y, scalar * vec.z, scalar * vec.w);
    }
    
    friend Vec4 operator/(float scalar, const Vec4& vec) {
        return Vec4(scalar / vec.x, scalar / vec.y, scalar / vec.z, scalar / vec.w);
    }

    Vec4 lerp(const Vec4& other, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        return *this + (other - *this) * t;
    }
    
    // Convert to grayscale using standard RGB weights
    float grayscale() const {
        return r * 0.299f + g * 0.587f + b * 0.114f;
    }
    
    // Color inversion (1.0 - color)
    Vec4 inverted() const {
        return Vec4(1.0f - r, 1.0f - g, 1.0f - b, a);
    }
    
    float& operator[](int index) {
        return (&x)[index];
    }
    
    const float& operator[](int index) const {
        return (&x)[index];
    }
    
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + 
               std::to_string(z) + ", " + std::to_string(w) + ")";
    }
    
    std::string toColorString() const {
        return "RGBA(" + std::to_string(r) + ", " + std::to_string(g) + ", " + 
               std::to_string(b) + ", " + std::to_string(a) + ")";
    }
};

inline std::ostream& operator<<(std::ostream& os, const Vec4& vec) {
    os << vec.toString();
    return os;
}

namespace std {
    template<>
    struct hash<Vec4> {
        size_t operator()(const Vec4& v) const {
            return hash<float>()(v.x) ^ (hash<float>()(v.y) << 1) ^ 
                   (hash<float>()(v.z) << 2) ^ (hash<float>()(v.w) << 3);
        }
    };
}

#endif