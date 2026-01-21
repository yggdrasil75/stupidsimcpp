#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <algorithm>
#include <string>
#include <ostream>
#include <cstdint>
#include <stdfloat>
#include <cstring>
#include "vec2.hpp"
#include "../basicdefines.hpp"

#ifdef __SSE__
#include <xmmintrin.h>
#endif

template<typename T>
class alignas(16) Vec3 {
public:
    struct{ T x, y, z; };
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
    Vec3(T scalar) : x(scalar), y(scalar), z(scalar) {}
    Vec3(float acd[3]) : x(acd[0]), y(acd[1]), z(acd[2]) {}
    template<typename U>
    Vec3(const Vec3<U>& other) : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), z(static_cast<T>(other.z)) {}
    
    template<typename U>
    Vec3(const class Vec2<U>& vec2, U z = 0) : x(static_cast<T>(vec2.x)), y(static_cast<T>(vec2.y)), z(static_cast<T>(z)) {}
    
    Vec3& move(const Vec3& newpos) {
        x = newpos.x;
        y = newpos.y;
        z = newpos.z;
        return *this;
    }

    // Arithmetic operations
    template<typename U>
    Vec3 operator+(const Vec3<U>& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    template<typename U>
    Vec3 operator-(const Vec3<U>& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    template<typename U>
    Vec3 operator*(const Vec3<U>& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }
    
    template<typename U>
    Vec3 operator/(const Vec3<U>& other) const {
        return Vec3(x / other.x, y / other.y, z / other.z);
    }

    Vec3 operator+(T scalar) const {
        return Vec3(x + scalar, y + scalar, z + scalar);
    }

    Vec3 operator-(T scalar) const {
        return Vec3(x - scalar, y - scalar, z - scalar);
    }
    
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    Vec3 operator*(T scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    Vec3 operator/(T scalar) const {
        T invScalar = T(1) / scalar;
        return Vec3(x * invScalar, y * invScalar, z * invScalar);
    }

    Vec3& operator=(T scalar) {
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
    
    Vec3& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }
    
    Vec3& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }
    
    Vec3& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    
    Vec3& operator/=(T scalar) {
        T invScalar = T(1) / scalar;
        x *= invScalar;
        y *= invScalar;
        z *= invScalar;
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

    T length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    // Fast inverse length (Quake III algorithm)
    T invLength() const {
        const T lenSq = x * x + y * y + z * z;
        if (lenSq == 0) return 0;
        
        // Fast inverse square root approximation
        const T half = T(0.5) * lenSq;
        T o = lenSq;
        
        // Type punning for float/double
        if constexpr (std::is_same_v<T, float>) {
            long i = *(long*)&o;
            i = 0x5f3759df - (i >> 1);
            o = *(float*)&i;
        } else if constexpr (std::is_same_v<T, double>) {
            long long i = *(long long*)&o;
            i = 0x5fe6eb50c7b537a9 - (i >> 1);
            o = *(double*)&i;
        }
        
        o = o * (T(1.5) - (half * o * o));
        return o;
    }
    
    T lengthSquared() const {
        return x * x + y * y + z * z;
    }
    
    T distance(const Vec3& other) const {
        return (*this - other).length();
    }
    
    T distanceSquared(const Vec3& other) const {
        Vec3 diff = *this - other;
        return diff.lengthSquared();
    }

    // Normalized with SSE optimization
    Vec3 normalized() const {
        const T invLen = invLength();
        if (invLen > 0) {
            #ifdef __SSE__
                if constexpr (std::is_same_v<T, float>) {
                    __m128 vec = _mm_set_ps(0.0f, z, y, x);
                    __m128 inv = _mm_set1_ps(invLen);
                    __m128 result = _mm_mul_ps(vec, inv);
                    
                    alignas(16) float components[4];
                    _mm_store_ps(components, result);
                    return Vec3(components[0], components[1], components[2]);
                } else
            #endif
            {
                // Fallback to scalar operations
                return Vec3(x * invLen, y * invLen, z * invLen);
            }
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
        return (lengthSquared() < other.lengthSquared());
    }

    bool operator<(T scalar) const {
        return (x < scalar && y < scalar && z < scalar);
    }
    
    bool operator<=(const Vec3& other) const {
        return (lengthSquared() <= other.lengthSquared());
    }
    
    bool operator<=(T scalar) const {
        return (x <= scalar && y <= scalar && z <= scalar);
    }
    
    bool operator>(const Vec3& other) const {
        return (lengthSquared() > other.lengthSquared());
    }

    bool operator>(T scalar) const {
        return (x > scalar && y > scalar && z > scalar);
    }
    
    bool operator>=(const Vec3& other) const {
        return (lengthSquared() >= other.lengthSquared());
    }
    
    bool operator>=(T scalar) const {
        return (x >= scalar && y >= scalar && z >= scalar);
    }
    
    bool AllLT(const Vec3& other) const {
        return x < other.x && y < other.y && z < other.z;
    }
    
    bool AllGT(const Vec3& other) const {
        return x > other.x && y > other.y && z > other.z;
    }
    
    bool AllLTE(const Vec3& other) const {
        return x <= other.x && y <= other.y && z <= other.z;
    }
    
    bool AllGTE(const Vec3& other) const {
        return x >= other.x && y >= other.y && z >= other.z;
    }
    
    bool AnyLT(const Vec3& other) const {
        return x < other.x || y < other.y || z < other.z;
    }
    
    bool AnyGT(const Vec3& other) const {
        return x > other.x || y > other.y || z > other.z;
    }
    
    bool AnyLTE(const Vec3& other) const {
        return x <= other.x || y <= other.y || z <= other.z;
    }
    
    bool AnyGTE(const Vec3& other) const {
        return x >= other.x || y >= other.y || z >= other.z;
    }

    template<typename CompareFunc>
    Vec3<bool> mask(CompareFunc comp, T value) const {
        return Vec3<bool>(comp(x, value), comp(y, value), comp(z, value));
    }

    template<typename CompareFunc>
    Vec3<bool> mask(CompareFunc comp, const Vec3& other) const {
        return Vec3<bool>(comp(x, other.x), comp(y, other.y), comp(z, other.z));
    }

    Vec3 abs() const {
        return Vec3(std::abs(x), std::abs(y), std::abs(z));
    }
    
    Vec3 floor() const {
        return Vec3(std::floor(x), std::floor(y), std::floor(z));
    }

    Vec3<int> floorToI() const {
        return Vec3<int>(static_cast<int>(std::floor(x)), static_cast<int>(std::floor(y)), static_cast<int>(std::floor(z)));
    }

    Vec3<uint8_t> floorToI8() const {
        return Vec3<uint8_t>(static_cast<uint8_t>(std::max(T(0), std::floor(x))), static_cast<uint8_t>(std::max(T(0), std::floor(y))), static_cast<uint8_t>(std::max(T(0), std::floor(z))));
    }
    
    Vec3<size_t> floorToT() const {
        return Vec3<size_t>(static_cast<size_t>(std::max(T(0), std::floor(x))), static_cast<size_t>(std::max(T(0), std::floor(y))), static_cast<size_t>(std::max(T(0), std::floor(z))));
    }

    Vec3<float> toFloat() const {
        return Vec3<float>(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }

    Vec3<double> toDouble() const {
        return Vec3<double>(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
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
        return this->max(minVal).min(maxVal);
    }
    
    Vec3 clamp(T minVal, T maxVal) const {
        return this->max(Vec3(minVal)).min(Vec3(maxVal));
    }
    
    bool isZero() const {
        return length() < EPSILON;
        //return std::abs(x) < epsilon && std::abs(y) < epsilon && std::abs(z) < epsilon;
    }
    
    bool equals(const Vec3& other, float epsilon = 1e-10f) const {
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon;
    }
    
    // Template friend operators to allow different scalar types
    // template<typename S>
    // friend Vec3<T> operator+(S scalar, const Vec3<T>& vec) {
    //     return Vec3<T>(static_cast<T>(scalar + vec.x), 
    //                   static_cast<T>(scalar + vec.y), 
    //                   static_cast<T>(scalar + vec.z));
    // }
    
    // template<typename S>
    // friend Vec3<T> operator-(S scalar, const Vec3<T>& vec) {
    //     return Vec3<T>(static_cast<T>(scalar - static_cast<S>(vec.x)),
    //                   static_cast<T>(scalar - static_cast<S>(vec.y)),
    //                   static_cast<T>(scalar - static_cast<S>(vec.z)));
    // }
    
    // template<typename S>
    // friend Vec3<T> operator*(S scalar, const Vec3<T>& vec) {
    //     return Vec3<T>(static_cast<T>(scalar * vec.x), 
    //                   static_cast<T>(scalar * vec.y), 
    //                   static_cast<T>(scalar * vec.z));
    // }
    //previously was commented because Vec3<S> was treated as friend S instead.
    friend Vec3<T> operator/(float scalar, const Vec3<T>& vec) {
        return Vec3<T>(static_cast<T>(scalar / vec.x), 
                      static_cast<T>(scalar / vec.y), 
                      static_cast<T>(scalar / vec.z));
    }

    Vec3 reflect(const Vec3& normal) const {
        return *this - 2.0f * this->dot(normal) * normal;
    }
    
    Vec3 lerp(const Vec3& other, T t) const {
        t = std::clamp(t, T(0), T(1));
        return *this + (other - *this) * t;
    }
    
    Vec3 fastLerp(const Vec3& other, T t) const {
        return *this + (other - *this) * t;
    }

    Vec3 fmaLerp(const Vec3& other, T t) const {
        return Vec3(
            std::fma(t, other.x - x, x),
            std::fma(t, other.y - y, y),
            std::fma(t, other.z - z, z)
        );
    }
    
    Vec3 slerp(const Vec3& other, T t) const {
        t = std::clamp(t, T(0), T(1));
        T dotVal = this->dot(other);
        dotVal = std::clamp(dotVal, T(-1), T(1));
        
        T theta = std::acos(dotVal) * t;
        Vec3 relative = other - *this * dotVal;
        relative = relative.normalized();
        
        return (*this * std::cos(theta)) + (relative * std::sin(theta));
    }
    
    Vec3 rotateX(T angle) const {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Vec3(x, y * cosA - z * sinA, y * sinA + z * cosA);
    }
    
    Vec3 rotateY(T angle) const {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Vec3(x * cosA + z * sinA, y, -x * sinA + z * cosA);
    }
    
    Vec3 rotateZ(T angle) const {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Vec3(x * cosA - y * sinA, x * sinA + y * cosA, z);
    }

    float angle() const {
        float r = length();
        if (r == 0) return 0;
        float θ = std::acos(z / r);
        return θ;
    }

    float azimuth() const {
        float φ = std::atan2(y, x);
        return φ;
    }

    std::pair<float, float> sphericalAngles() const {
        float r = length();
        if (r == 0) return {0, 0};
        
        float θ = std::acos(z / r);
        float φ = std::atan2(y, x);
        
        return {θ, φ};
    }

    float angleTo(const Vec3& other) const {
        return std::acos(this->dot(other) / (this->length() * other.length()));
    }
    
    float directionTo(const Vec3& other) const {
        Vec3 direction = other - *this;
        return direction.angleTo(other);
    }

    T& operator[](int index) {
        return (&x)[index];
    }
    
    const T& operator[](int index) const {
        return (&x)[index];
    }
    
    Vec3 safeInverse() const {
        return Vec3(
            1 / (std::abs(x) < EPSILON ? std::copysign(EPSILON, x) : x),
            1 / (std::abs(y) < EPSILON ? std::copysign(EPSILON, y) : y),
            1 / (std::abs(z) < EPSILON ? std::copysign(EPSILON, z) : z)
        );
    }

    uint8_t calculateOctantMask() const {
        uint8_t mask = 0;
        if (x > 0.f) mask |= 1;
        if (y > 0.f) mask |= 2;
        if (z > 0.f) mask |= 4;
        return mask;
    }
    
    T maxComp() const { 
        return std::max({x, y, z}); 
    }
    
    T minComp() const { 
        return std::min({x, y, z}); 
    }

    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
    }

    struct Hash {
        std::size_t operator()(const Vec3& v) const {
            return std::hash<float>()(v.x) ^ (std::hash<float>()(v.y) << 1) ^ (std::hash<float>()(v.z) << 2);
        }
    };

    Vec2<T> toLatLon() const {
        T r = length();
        if (r == T(0)) return Vec2<T>(0, 0);
        T θ = std::acos(z / r);
        T lat = static_cast<T>(M_PI/2.0) - θ;

        T lon = std::atan2(y, x);
        return Vec2<T>(lat, lon);
    }

    Vec2<T> toLatLon(const Vec3& center) const {
        Vec3 relative = *this - center;
        return relative.toLatLon();
    }

    T toElevation() const {
        return length();
    }

    T toElevation(const Vec3& center) const {
        return distance(center);
    }
};

#ifdef __SSE__
// SSE-optimized version for float types
template<>
inline Vec3<float> Vec3<float>::normalized() const {
    float lenSq = lengthSquared();
    if (lenSq > 0.0f) {
        // Load vector into SSE register
        __m128 vec = _mm_set_ps(0.0f, z, y, x);  // w=0, z, y, x
        
        // Fast inverse square root using SSE
        __m128 lenSq128 = _mm_set1_ps(lenSq);
        
        // Quake III fast inverse sqrt SSE version
        __m128 half = _mm_mul_ps(lenSq128, _mm_set1_ps(0.5f));
        __m128 three = _mm_set1_ps(1.5f);
        
        __m128 y = lenSq128;
        __m128i i = _mm_castps_si128(y);
        i = _mm_sub_epi32(_mm_set1_epi32(0x5f3759df), 
                          _mm_srai_epi32(i, 1));
        y = _mm_castsi128_ps(i);
        
        y = _mm_mul_ps(y, _mm_sub_ps(three, _mm_mul_ps(half, _mm_mul_ps(y, y))));
        
        // Multiply vector by inverse length
        __m128 invLen128 = y;
        __m128 result = _mm_mul_ps(vec, invLen128);
        
        // Extract results
        alignas(16) float resultArr[4];
        _mm_store_ps(resultArr, result);
        
        return Vec3<float>(resultArr[0], resultArr[1], resultArr[2]);
    }
    return *this;
};
#endif

//use a smaller format first instead of larger format.
//#ifdef std::float16_t
//using Vec3f = Vec3<std::float16_t>;
//#else 
using Vec3f = Vec3<float>;
//#endif
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;
using Vec3i32 = Vec3<uint32_t>;
using Vec3i8 = Vec3<int8_t>;
using Vec3ui8 = Vec3<uint8_t>;
using Vec3T = Vec3<size_t>;
using Vec3b = Vec3<bool>;

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec3<T>& vec) {
    os << vec.toString();
    return os;
}

namespace std {
    template<typename T>
    struct hash<Vec3<T>> {
        size_t operator()(const Vec3<T>& v) const {
            return hash<T>()(v.x) ^ (hash<T>()(v.y) << 1) ^ (hash<T>()(v.z) << 2);
        }
    };
}

template<typename T>
Vec3<T> max(Vec3<T> a, Vec3<T> b) {
    return a.max(b);
}

template<typename T>
Vec3<T> min(Vec3<T> a, Vec3<T> b) {
    return a.min(b);
}

template<typename T>
Vec3<T> mix(const Vec3<T>& a, const Vec3<T>& b, const Vec3<bool>& mask) {
    return Vec3<T>(
        mask.x ? b.x : a.x,
        mask.y ? b.y : a.y,
        mask.z ? b.z : a.z
    );
}

#endif