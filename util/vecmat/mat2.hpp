#ifndef MAT2_HPP
#define MAT2_HPP

#include "Vec2.hpp"
#include <array>
#include <cmath>

class Mat2 {
public:
    union {
        struct { float m00, m01, m10, m11; };
        struct { float a, b, c, d; };
        float data[4];
        float m[2][2];
    };
    
    // Constructors
    Mat2() : m00(1), m01(0), m10(0), m11(1) {}
    Mat2(float scalar) : m00(scalar), m01(scalar), m10(scalar), m11(scalar) {}
    Mat2(float m00, float m01, float m10, float m11) : m00(m00), m01(m01), m10(m10), m11(m11) {}
    
    // Identity matrix
    static Mat2 identity() { return Mat2(1, 0, 0, 1); }
    
    // Zero matrix
    static Mat2 zero() { return Mat2(0, 0, 0, 0); }
    
    // Rotation matrix
    static Mat2 rotation(float angle) {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Mat2(cosA, -sinA, sinA, cosA);
    }
    
    // Scaling matrix
    static Mat2 scaling(const Vec2& scale) {
        return Mat2(scale.x, 0, 0, scale.y);
    }
    
    // Arithmetic operations
    Mat2 operator+(const Mat2& other) const {
        return Mat2(m00 + other.m00, m01 + other.m01,
                   m10 + other.m10, m11 + other.m11);
    }
    
    Mat2 operator-(const Mat2& other) const {
        return Mat2(m00 - other.m00, m01 - other.m01,
                   m10 - other.m10, m11 - other.m11);
    }
    
    Mat2 operator*(const Mat2& other) const {
        return Mat2(
            m00 * other.m00 + m01 * other.m10,
            m00 * other.m01 + m01 * other.m11,
            m10 * other.m00 + m11 * other.m10,
            m10 * other.m01 + m11 * other.m11
        );
    }
    
    Mat2 operator*(float scalar) const {
        return Mat2(m00 * scalar, m01 * scalar,
                   m10 * scalar, m11 * scalar);
    }
    
    Mat2 operator/(float scalar) const {
        return Mat2(m00 / scalar, m01 / scalar,
                   m10 / scalar, m11 / scalar);
    }
    
    Vec2 operator*(const Vec2& vec) const {
        return Vec2(
            m00 * vec.x + m01 * vec.y,
            m10 * vec.x + m11 * vec.y
        );
    }
    
    Mat2& operator+=(const Mat2& other) {
        m00 += other.m00; m01 += other.m01;
        m10 += other.m10; m11 += other.m11;
        return *this;
    }
    
    Mat2& operator-=(const Mat2& other) {
        m00 -= other.m00; m01 -= other.m01;
        m10 -= other.m10; m11 -= other.m11;
        return *this;
    }
    
    Mat2& operator*=(const Mat2& other) {
        *this = *this * other;
        return *this;
    }
    
    Mat2& operator*=(float scalar) {
        m00 *= scalar; m01 *= scalar;
        m10 *= scalar; m11 *= scalar;
        return *this;
    }
    
    Mat2& operator/=(float scalar) {
        m00 /= scalar; m01 /= scalar;
        m10 /= scalar; m11 /= scalar;
        return *this;
    }
    
    bool operator==(const Mat2& other) const {
        return m00 == other.m00 && m01 == other.m01 &&
               m10 == other.m10 && m11 == other.m11;
    }
    
    bool operator!=(const Mat2& other) const {
        return !(*this == other);
    }
    
    // Matrix operations
    float determinant() const {
        return m00 * m11 - m01 * m10;
    }
    
    Mat2 transposed() const {
        return Mat2(m00, m10, m01, m11);
    }
    
    Mat2 inverse() const {
        float det = determinant();
        if (std::abs(det) < 1e-10f) {
            return Mat2(); // Return identity if not invertible
        }
        float invDet = 1.0f / det;
        return Mat2( m11 * invDet, -m01 * invDet,
                    -m10 * invDet,  m00 * invDet);
    }
    
    // Access operators
    float& operator()(int row, int col) {
        return m[row][col];
    }
    
    const float& operator()(int row, int col) const {
        return m[row][col];
    }
    
    float& operator[](int index) {
        return data[index];
    }
    
    const float& operator[](int index) const {
        return data[index];
    }
    
    std::string toString() const {
        return "Mat2([" + std::to_string(m00) + ", " + std::to_string(m01) + "],\n" +
               "      [" + std::to_string(m10) + ", " + std::to_string(m11) + "])";
    }
};

inline std::ostream& operator<<(std::ostream& os, const Mat2& mat) {
    os << mat.toString();
    return os;
}

inline Mat2 operator*(float scalar, const Mat2& mat) {
    return mat * scalar;
}

#endif