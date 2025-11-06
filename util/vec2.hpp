#ifndef VEC2_HPP
#define VEC2_HPP

#include <cmath>
#include <algorithm>
#include <string>

class Vec2 {
    public:
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}
    
    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }
    
    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }
    
    Vec2 operator*(const Vec2& other) const {
        return Vec2(x * other.x, y * other.y);
    }
    
    Vec2 operator/(const Vec2& other) const {
        return Vec2(x / other.x, y / other.y);
    }

    Vec2 operator+(float scalar) const {
        return Vec2(x + scalar, y + scalar);
    }

    Vec2 operator-(float scalar) const {
        return Vec2(x - scalar, y - scalar);
    }
    
    Vec2 operator-() const {
        return Vec2(-x, -y);
    }

    Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    Vec2 operator/(float scalar) const {
        return Vec2(x / scalar, y / scalar);
    }

    Vec2& operator=(float scalar) {
        x = y = scalar;
        return *this;
    }
    
    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    
    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    
    Vec2& operator*=(const Vec2& other) {
        x *= other.x;
        y *= other.y;
        return *this;
    }
    
    Vec2& operator/=(const Vec2& other) {
        x /= other.x;
        y /= other.y;
        return *this;
    }
    
    Vec2& operator+=(float scalar) {
        x += scalar;
        y += scalar;
        return *this;
    }
    
    Vec2& operator-=(float scalar) {
        x -= scalar;
        y -= scalar;
        return *this;
    }
    
    Vec2& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    
    Vec2& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    float dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }

    float length() const {
        return std::sqrt(x * x + y * y);
    }
    
    float lengthSquared() const {
        return x * x + y * y;
    }
    
    float distance(const Vec2& other) const {
        return (*this - other).length();
    }
    
    float distanceSquared(const Vec2& other) const {
        Vec2 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y;
    }

    Vec2 normalized() const {
        float len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }
    
    bool operator==(const Vec2& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const Vec2& other) const {
        return x != other.x || y != other.y;
    }
    
    bool operator<(const Vec2& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
    
    bool operator<=(const Vec2& other) const {
        return (x < other.x) || (x == other.x && y <= other.y);
    }
    
    bool operator>(const Vec2& other) const {
        return (x > other.x) || (x == other.x && y > other.y);
    }
    
    bool operator>=(const Vec2& other) const {
        return (x > other.x) || (x == other.x && y >= other.y);
    }
    
    Vec2 abs() const {
        return Vec2(std::abs(x), std::abs(y));
    }
    
    Vec2 floor() const {
        return Vec2(std::floor(x), std::floor(y));
    }
    
    Vec2 ceil() const {
        return Vec2(std::ceil(x), std::ceil(y));
    }
    
    Vec2 round() const {
        return Vec2(std::round(x), std::round(y));
    }
    
    Vec2 min(const Vec2& other) const {
        return Vec2(std::min(x, other.x), std::min(y, other.y));
    }
    
    Vec2 max(const Vec2& other) const {
        return Vec2(std::max(x, other.x), std::max(y, other.y));
    }
    
    Vec2 clamp(const Vec2& minVal, const Vec2& maxVal) const {
        return Vec2(
            std::clamp(x, minVal.x, maxVal.x),
            std::clamp(y, minVal.y, maxVal.y)
        );
    }
    
    Vec2 clamp(float minVal, float maxVal) const {
        return Vec2(
            std::clamp(x, minVal, maxVal),
            std::clamp(y, minVal, maxVal)
        );
    }
    
    bool isZero(float epsilon = 1e-10f) const {
        return std::abs(x) < epsilon && std::abs(y) < epsilon;
    }
    
    bool equals(const Vec2& other, float epsilon = 1e-10f) const {
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon;
    }
    
    friend Vec2 operator+(float scalar, const Vec2& vec) {
        return Vec2(scalar + vec.x, scalar + vec.y);
    }
    
    friend Vec2 operator-(float scalar, const Vec2& vec) {
        return Vec2(scalar - vec.x, scalar - vec.y);
    }
    
    friend Vec2 operator*(float scalar, const Vec2& vec) {
        return Vec2(scalar * vec.x, scalar * vec.y);
    }
    
    friend Vec2 operator/(float scalar, const Vec2& vec) {
        return Vec2(scalar / vec.x, scalar / vec.y);
    }

    Vec2 perpendicular() const {
        return Vec2(-y, x);
    }
    
    Vec2 reflect(const Vec2& normal) const {
        return *this - 2.0f * this->dot(normal) * normal;
    }
    
    Vec2 lerp(const Vec2& other, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        return *this + (other - *this) * t;
    }
    
    Vec2 slerp(const Vec2& other, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        float dot = this->dot(other);
        dot = std::clamp(dot, -1.0f, 1.0f);
        
        float theta = std::acos(dot) * t;
        Vec2 relative = other - *this * dot;
        relative = relative.normalized();
        
        return (*this * std::cos(theta)) + (relative * std::sin(theta));
    }
    
    Vec2 rotate(float angle) const {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Vec2(x * cosA - y * sinA, x * sinA + y * cosA);
    }
    
    float angle() const {
        return std::atan2(y, x);
    }
    
    float angleTo(const Vec2& other) const {
        return std::acos(this->dot(other) / (this->length() * other.length()));
    }
    
    float& operator[](int index) {
        return (&x)[index];
    }
    
    const float& operator[](int index) const {
        return (&x)[index];
    }
    
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
    
};

inline std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
    os << vec.toString();
    return os;
}

namespace std {
    template<>
    struct hash<Vec2> {
        size_t operator()(const Vec2& v) const {
            return hash<float>()(v.x) ^ (hash<float>()(v.y) << 1);
        }
    };
}

#endif