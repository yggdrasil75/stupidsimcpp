#ifndef VEC4_HPP
#define VEC4_HPP

#include "vec3.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <ostream>
#include <cstdint>

template<typename T>
class Vec4 {
public:
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        struct { T s, t, p, q; }; // For texture coordinates
    };
    
    // Constructors
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    Vec4(T scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}
    
    Vec4(const Vec3<T>& rgb, T w = 1) : x(rgb.x), y(rgb.y), z(rgb.z), w(w) {}
    static Vec4 RGB(T r, T g, T b, T a = 1) { return Vec4(r, g, b, a); }
    static Vec4 RGBA(T r, T g, T b, T a) { return Vec4(r, g, b, a); }
    
    Vec4& recolor(const Vec4& newColor) {
        r = newColor.r;
        g = newColor.g;
        b = newColor.b;
        a = newColor.a;
        return *this;
    }
    
    Vec4 average(const Vec4& other) const {
        return Vec4((x + other.x) / 2, (y + other.y) / 2, (z + other.z) / 2, (w + other.w) / 2);
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

    Vec4 operator+(T scalar) const {
        return Vec4(x + scalar, y + scalar, z + scalar, w + scalar);
    }

    Vec4 operator-(T scalar) const {
        return Vec4(x - scalar, y - scalar, z - scalar, w - scalar);
    }
    
    Vec4 operator-() const {
        return Vec4(-x, -y, -z, -w);
    }

    Vec4 operator*(T scalar) const {
        return Vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    Vec4 operator/(T scalar) const {
        return Vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    Vec4& operator=(T scalar) {
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
    
    Vec4& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        w += scalar;
        return *this;
    }
    
    Vec4& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        w -= scalar;
        return *this;
    }
    
    Vec4& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }
    
    Vec4& operator/=(T scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    T dot(const Vec4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
    
    // 4D cross product (returns vector perpendicular to 3 given vectors in 4D space)
    Vec4 cross(const Vec4& v1, const Vec4& v2, const Vec4& v3) const {
        T a = v1.y * (v2.z * v3.w - v2.w * v3.z) -
              v1.z * (v2.y * v3.w - v2.w * v3.y) +
              v1.w * (v2.y * v3.z - v2.z * v3.y);
                  
        T b = -v1.x * (v2.z * v3.w - v2.w * v3.z) +
               v1.z * (v2.x * v3.w - v2.w * v3.x) -
               v1.w * (v2.x * v3.z - v2.z * v3.x);
                   
        T c = v1.x * (v2.y * v3.w - v2.w * v3.y) -
              v1.y * (v2.x * v3.w - v2.w * v3.x) +
              v1.w * (v2.x * v3.y - v2.y * v3.x);
                  
        T d = -v1.x * (v2.y * v3.z - v2.z * v3.y) +
               v1.y * (v2.x * v3.z - v2.z * v3.x) -
               v1.z * (v2.x * v3.y - v2.y * v3.x);
        
        return Vec4(a, b, c, d);
    }

    T length() const {
        return static_cast<T>(std::sqrt(static_cast<double>(x * x + y * y + z * z + w * w)));
    }
    
    T lengthSquared() const {
        return x * x + y * y + z * z + w * w;
    }
    
    T distance(const Vec4& other) const {
        return (*this - other).length();
    }
    
    T distanceSquared(const Vec4& other) const {
        Vec4 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    Vec4 normalized() const {
        T len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }
    
    // Homogeneous normalization (divide by w)
    Vec4 homogenized() const {
        if (w != 0) {
            return Vec4(x / w, y / w, z / w, 1);
        }
        return *this;
    }
    
    // Clamp values between 0 and 1
    Vec4 clamped() const {
        return Vec4(
            std::clamp(r, static_cast<T>(0), static_cast<T>(1)),
            std::clamp(g, static_cast<T>(0), static_cast<T>(1)),
            std::clamp(b, static_cast<T>(0), static_cast<T>(1)),
            std::clamp(a, static_cast<T>(0), static_cast<T>(1))
        );
    }
    
    // Convert to Vec3 (ignoring alpha)
    Vec3<T> toVec3() const {
        return Vec3<T>(r, g, b);
    }
    
    // Convert to 8-bit color values
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value>::type
    toUint8(uint8_t& red, uint8_t& green, uint8_t& blue, uint8_t& alpha) const {
        red = static_cast<uint8_t>(std::clamp(r, static_cast<T>(0), static_cast<T>(1)) * 255);
        green = static_cast<uint8_t>(std::clamp(g, static_cast<T>(0), static_cast<T>(1)) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, static_cast<T>(0), static_cast<T>(1)) * 255);
        alpha = static_cast<uint8_t>(std::clamp(a, static_cast<T>(0), static_cast<T>(1)) * 255);
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value>::type
    toUint8(uint8_t& red, uint8_t& green, uint8_t& blue) const {
        red = static_cast<uint8_t>(std::clamp(r, static_cast<T>(0), static_cast<T>(1)) * 255);
        green = static_cast<uint8_t>(std::clamp(g, static_cast<T>(0), static_cast<T>(1)) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, static_cast<T>(0), static_cast<T>(1)) * 255);
    }
    
    // Get XYZ components as Vec3
    Vec3<T> xyz() const {
        return Vec3<T>(x, y, z);
    }
    
    // Get RGB components as Vec3
    Vec3<T> rgb() const {
        return Vec3<T>(r, g, b);
    }
    
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
    
    Vec4 clamp(T minVal, T maxVal) const {   
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
    
    bool isZero(T epsilon = static_cast<T>(1e-10)) const {
        return std::abs(x) < epsilon && std::abs(y) < epsilon && 
               std::abs(z) < epsilon && std::abs(w) < epsilon;
    }
    
    bool equals(const Vec4& other, T epsilon = static_cast<T>(1e-10)) const {
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon &&
               std::abs(w - other.w) < epsilon;
    }
    
    friend Vec4 operator+(T scalar, const Vec4& vec) {
        return Vec4(scalar + vec.x, scalar + vec.y, scalar + vec.z, scalar + vec.w);
    }
    
    friend Vec4 operator-(T scalar, const Vec4& vec) {
        return Vec4(scalar - vec.x, scalar - vec.y, scalar - vec.z, scalar - vec.w);
    }
    
    friend Vec4 operator*(T scalar, const Vec4& vec) {
        return Vec4(scalar * vec.x, scalar * vec.y, scalar * vec.z, scalar * vec.w);
    }
    
    friend Vec4 operator/(T scalar, const Vec4& vec) {
        return Vec4(scalar / vec.x, scalar / vec.y, scalar / vec.z, scalar / vec.w);
    }

    Vec4 lerp(const Vec4& other, T t) const {
        t = std::clamp(t, static_cast<T>(0), static_cast<T>(1));
        return *this + (other - *this) * t;
    }
    
    // Convert to grayscale using standard RGB weights (only valid for float/double)
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, T>::type grayscale() const {
        return r * static_cast<T>(0.299) + g * static_cast<T>(0.587) + b * static_cast<T>(0.114);
    }
    
    // Color inversion (1.0 - color) (only valid for float/double)
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, Vec4>::type
    inverted() const {
        return Vec4(static_cast<T>(1) - r, static_cast<T>(1) - g, 
                   static_cast<T>(1) - b, a);
    }
    
    T& operator[](int index) {
        return (&x)[index];
    }
    
    const T& operator[](int index) const {
        return (&x)[index];
    }
    
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + 
               std::to_string(z) + ", " + std::to_string(w) + ")";
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, std::string>::type
    toColorString() const {
        return "RGBA(" + std::to_string(r) + ", " + std::to_string(g) + ", " + 
               std::to_string(b) + ", " + std::to_string(a) + ")";
    }
    
    struct Hash {
        std::size_t operator()(const Vec4& v) const {
            return std::hash<T>()(v.x) ^ (std::hash<T>()(v.y) << 1) ^ 
                   (std::hash<T>()(v.z) << 2) ^ (std::hash<T>()(v.w) << 3);
        }
    };
};

// Type aliases for common use cases
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int>;
using Vec4u = Vec4<unsigned int>;
using Vec4ui8 = Vec4<uint8_t>;

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec4<T>& vec) {
    os << vec.toString();
    return os;
}

namespace std {
    template<typename T>
    struct hash<Vec4<T>> {
        size_t operator()(const Vec4<T>& v) const {
            return hash<T>()(v.x) ^ (hash<T>()(v.y) << 1) ^ 
                   (hash<T>()(v.z) << 2) ^ (hash<T>()(v.w) << 3);
        }
    };
}

#endif