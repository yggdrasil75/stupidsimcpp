#ifndef VEC2_HPP
#define VEC2_HPP

#include <cmath>
#include <algorithm>
#include <string>

template<typename T>
class Vec2 {
public:
    T x, y;
    
    Vec2() : x(0), y(0) {}
    Vec2(T x, T y) : x(x), y(y) {}
    
    template<typename U>
    explicit Vec2(const Vec2<U>& other) : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)) {}
    
    Vec2& move(const Vec2 newpos) {
        x = newpos.x;
        y = newpos.y;
        return *this;
    }
    
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
    
    template<typename U>
    Vec2 operator+(U scalar) const {
        return Vec2(x + scalar, y + scalar);
    }
    
    template<typename U>
    Vec2 operator-(U scalar) const {
        return Vec2(x - scalar, y - scalar);
    }
    
    Vec2 operator-() const {
        return Vec2(-x, -y);
    }
    
    template<typename U>
    Vec2 operator*(U scalar) const {
        return Vec2(x * scalar, y * scalar);
    }
    
    template<typename U>
    Vec2 operator/(U scalar) const {
        return Vec2(x / scalar, y / scalar);
    }
    
    template<typename U>
    Vec2& operator=(U scalar) {
        x = y = static_cast<T>(scalar);
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
    
    template<typename U>
    Vec2& operator+=(U scalar) {
        x += scalar;
        y += scalar;
        return *this;
    }
    
    template<typename U>
    Vec2& operator-=(U scalar) {
        x -= scalar;
        y -= scalar;
        return *this;
    }
    
    template<typename U>
    Vec2& operator*=(U scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    
    template<typename U>
    Vec2& operator/=(U scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    
    T dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }
    
    template<typename U = float>
    U length() const {
        return std::sqrt(static_cast<U>(x * x + y * y));
    }
    
    T lengthSquared() const {
        return x * x + y * y;
    }
    
    template<typename U = float>
    U distance(const Vec2& other) const {
        return (*this - other).template length<U>();
    }
    
    T distanceSquared(const Vec2& other) const {
        Vec2 diff = *this - other;
        return diff.x * diff.x + diff.y * diff.y;
    }
    
    template<typename U = float>
    Vec2<U> normalized() const {
        auto len = length<U>();
        if (len > 0) {
            return Vec2<U>(static_cast<U>(x) / len, static_cast<U>(y) / len);
        }
        return Vec2<U>(static_cast<U>(x), static_cast<U>(y));
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
    
    template<typename U>
    Vec2 clamp(U minVal, U maxVal) const {
        return Vec2(
            std::clamp(x, static_cast<T>(minVal), static_cast<T>(maxVal)),
            std::clamp(y, static_cast<T>(minVal), static_cast<T>(maxVal))
        );
    }
    
    template<typename U = float>
    bool isZero(U epsilon = static_cast<U>(1e-10)) const {
        return std::abs(static_cast<U>(x)) < epsilon && 
               std::abs(static_cast<U>(y)) < epsilon;
    }
    
    template<typename U = float>
    bool equals(const Vec2& other, U epsilon = static_cast<U>(1e-10)) const {
        return std::abs(static_cast<U>(x - other.x)) < epsilon && 
               std::abs(static_cast<U>(y - other.y)) < epsilon;
    }
    
    Vec2 perpendicular() const {
        return Vec2(-y, x);
    }
    
    template<typename U = float>
    Vec2<U> reflect(const Vec2<U>& normal) const {
        auto this_f = Vec2<U>(static_cast<U>(x), static_cast<U>(y));
        return this_f - static_cast<U>(2.0) * this_f.dot(normal) * normal;
    }
    
    template<typename U = float>
    Vec2<U> lerp(const Vec2<U>& other, U t) const {
        t = std::clamp(t, static_cast<U>(0.0), static_cast<U>(1.0));
        auto this_f = Vec2<U>(static_cast<U>(x), static_cast<U>(y));
        return this_f + (other - this_f) * t;
    }
    
    template<typename U = float>
    Vec2<U> slerp(const Vec2<U>& other, U t) const {
        t = std::clamp(t, static_cast<U>(0.0), static_cast<U>(1.0));
        auto this_f = Vec2<U>(static_cast<U>(x), static_cast<U>(y));
        U dot = this_f.dot(other);
        dot = std::clamp(dot, static_cast<U>(-1.0), static_cast<U>(1.0));
        
        U theta = std::acos(dot) * t;
        auto relative = other - this_f * dot;
        relative = relative.normalized();
        
        return (this_f * std::cos(theta)) + (relative * std::sin(theta));
    }
    
    template<typename U = float>
    Vec2<U> rotate(U angle) const {
        U cosA = std::cos(angle);
        U sinA = std::sin(angle);
        return Vec2<U>(
            static_cast<U>(x) * cosA - static_cast<U>(y) * sinA,
            static_cast<U>(x) * sinA + static_cast<U>(y) * cosA
        );
    }
    
    template<typename U = float>
    U angle() const {
        return std::atan2(static_cast<U>(y), static_cast<U>(x));
    }
    
    template<typename U = float>
    U angleTo(const Vec2<U>& other) const {
        auto this_f = Vec2<U>(static_cast<U>(x), static_cast<U>(y));
        return std::acos(this_f.dot(other) / (this_f.length() * other.length()));
    }
    
    template<typename U = float>
    U directionTo(const Vec2<U>& other) const {
        auto this_f = Vec2<U>(static_cast<U>(x), static_cast<U>(y));
        auto direction = other - this_f;
        return direction.angle();
    }
    
    T& operator[](int index) {
        return (&x)[index];
    }
    
    const T& operator[](int index) const {
        return (&x)[index];
    }
    
    std::string toString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
    
    std::ostream& operator<<(std::ostream& os) {
        os << toString();
        return os;
    }
    
    struct Hash {
        std::size_t operator()(const Vec2& v) const {
            return std::hash<T>()(v.x) ^ (std::hash<T>()(v.y) << 1);
        }
    };

    float aspect() {
        return static_cast<float>(x) / static_cast<float>(y);
    }
    
    Vec2<float> toFloat() {
        return Vec2<float>(static_cast<float>(x), static_cast<float>(y));
    }
    
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec2<T>& vec) {
    os << vec.toString();
    return os;
}

template<typename T, typename U>
auto operator+(U scalar, const Vec2<T>& vec) -> Vec2<decltype(scalar + vec.x)> {
    using ResultType = decltype(scalar + vec.x);
    return Vec2<ResultType>(scalar + vec.x, scalar + vec.y);
}

template<typename T, typename U>
auto operator-(U scalar, const Vec2<T>& vec) -> Vec2<decltype(scalar - vec.x)> {
    using ResultType = decltype(scalar - vec.x);
    return Vec2<ResultType>(scalar - vec.x, scalar - vec.y);
}

template<typename T, typename U>
auto operator*(U scalar, const Vec2<T>& vec) -> Vec2<decltype(scalar * vec.x)> {
    using ResultType = decltype(scalar * vec.x);
    return Vec2<ResultType>(scalar * vec.x, scalar * vec.y);
}

template<typename T, typename U>
auto operator/(U scalar, const Vec2<T>& vec) -> Vec2<decltype(scalar / vec.x)> {
    using ResultType = decltype(scalar / vec.x);
    return Vec2<ResultType>(scalar / vec.x, scalar / vec.y);
}

namespace std {
    template<typename T>
    struct hash<Vec2<T>> {
        size_t operator()(const Vec2<T>& v) const {
            return hash<T>()(v.x) ^ (hash<T>()(v.y) << 1);
        }
    };
}

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<unsigned int>;

#endif