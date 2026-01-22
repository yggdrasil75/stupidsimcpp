#ifndef MAT3_HPP
#define MAT3_HPP

#include "../vectorlogic/vec3.hpp"
#include <array>
#include <cmath>
#include <string>
#include <iostream>

template<typename T>
class Mat3 {
public:
    union {
        struct {
            T m00, m01, m02,
              m10, m11, m12,
              m20, m21, m22;
        };
        T data[9];
        T m[3][3];
    };
    
    // Constructors
    Mat3() : m00(1), m01(0), m02(0),
             m10(0), m11(1), m12(0),
             m20(0), m21(0), m22(1) {}
             
    Mat3(T scalar) : m00(scalar), m01(scalar), m02(scalar),
                     m10(scalar), m11(scalar), m12(scalar),
                     m20(scalar), m21(scalar), m22(scalar) {}
    
    Mat3(T m00, T m01, T m02,
         T m10, T m11, T m12,
         T m20, T m21, T m22) : 
         m00(m00), m01(m01), m02(m02),
         m10(m10), m11(m11), m12(m12),
         m20(m20), m21(m21), m22(m22) {}
    
    // Identity matrix
    static Mat3 identity() {
        return Mat3(1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
    }
    
    // Zero matrix
    static Mat3 zero() { return Mat3(0); }
    
    // Rotation matrices
    static Mat3 rotationX(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat3(1, 0, 0,
                   0, cosA, -sinA,
                   0, sinA, cosA);
    }
    
    static Mat3 rotationY(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat3(cosA, 0, sinA,
                   0, 1, 0,
                   -sinA, 0, cosA);
    }
    
    static Mat3 rotationZ(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat3(cosA, -sinA, 0,
                   sinA, cosA, 0,
                   0, 0, 1);
    }
    
    // Scaling matrix
    static Mat3 scaling(const Vec3<T>& scale) {
        return Mat3(scale.x, 0, 0,
                   0, scale.y, 0,
                   0, 0, scale.z);
    }
    
    // Arithmetic operations
    Mat3 operator+(const Mat3& other) const {
        return Mat3(m00 + other.m00, m01 + other.m01, m02 + other.m02,
                   m10 + other.m10, m11 + other.m11, m12 + other.m12,
                   m20 + other.m20, m21 + other.m21, m22 + other.m22);
    }
    
    Mat3 operator-(const Mat3& other) const {
        return Mat3(m00 - other.m00, m01 - other.m01, m02 - other.m02,
                   m10 - other.m10, m11 - other.m11, m12 - other.m12,
                   m20 - other.m20, m21 - other.m21, m22 - other.m22);
    }
    
    Mat3 operator*(const Mat3& other) const {
        return Mat3(
            m00 * other.m00 + m01 * other.m10 + m02 * other.m20,
            m00 * other.m01 + m01 * other.m11 + m02 * other.m21,
            m00 * other.m02 + m01 * other.m12 + m02 * other.m22,
            
            m10 * other.m00 + m11 * other.m10 + m12 * other.m20,
            m10 * other.m01 + m11 * other.m11 + m12 * other.m21,
            m10 * other.m02 + m11 * other.m12 + m12 * other.m22,
            
            m20 * other.m00 + m21 * other.m10 + m22 * other.m20,
            m20 * other.m01 + m21 * other.m11 + m22 * other.m21,
            m20 * other.m02 + m21 * other.m12 + m22 * other.m22
        );
    }
    
    Mat3 operator*(T scalar) const {
        return Mat3(m00 * scalar, m01 * scalar, m02 * scalar,
                   m10 * scalar, m11 * scalar, m12 * scalar,
                   m20 * scalar, m21 * scalar, m22 * scalar);
    }
    
    Mat3 operator/(T scalar) const {
        return Mat3(m00 / scalar, m01 / scalar, m02 / scalar,
                   m10 / scalar, m11 / scalar, m12 / scalar,
                   m20 / scalar, m21 / scalar, m22 / scalar);
    }
    
    Vec3<T> operator*(const Vec3<T>& vec) const {
        return Vec3<T>(
            m00 * vec.x + m01 * vec.y + m02 * vec.z,
            m10 * vec.x + m11 * vec.y + m12 * vec.z,
            m20 * vec.x + m21 * vec.y + m22 * vec.z
        );
    }
    
    Mat3& operator+=(const Mat3& other) {
        *this = *this + other;
        return *this;
    }
    
    Mat3& operator-=(const Mat3& other) {
        *this = *this - other;
        return *this;
    }
    
    Mat3& operator*=(const Mat3& other) {
        *this = *this * other;
        return *this;
    }
    
    Mat3& operator*=(T scalar) {
        *this = *this * scalar;
        return *this;
    }
    
    Mat3& operator/=(T scalar) {
        *this = *this / scalar;
        return *this;
    }
    
    bool operator==(const Mat3& other) const {
        for (int i = 0; i < 9; ++i) {
            if (data[i] != other.data[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const Mat3& other) const {
        return !(*this == other);
    }
    
    // Matrix operations
    T determinant() const {
        return m00 * (m11 * m22 - m12 * m21)
             - m01 * (m10 * m22 - m12 * m20)
             + m02 * (m10 * m21 - m11 * m20);
    }
    
    Mat3 transposed() const {
        return Mat3(m00, m10, m20,
                   m01, m11, m21,
                   m02, m12, m22);
    }
    
    Mat3 inverse() const {
        T det = determinant();
        if (std::abs(det) < static_cast<T>(1e-10)) {
            return Mat3(); // Return identity if not invertible
        }
        
        T invDet = static_cast<T>(1) / det;
        
        return Mat3(
            (m11 * m22 - m12 * m21) * invDet,
            (m02 * m21 - m01 * m22) * invDet,
            (m01 * m12 - m02 * m11) * invDet,
            
            (m12 * m20 - m10 * m22) * invDet,
            (m00 * m22 - m02 * m20) * invDet,
            (m02 * m10 - m00 * m12) * invDet,
            
            (m10 * m21 - m11 * m20) * invDet,
            (m01 * m20 - m00 * m21) * invDet,
            (m00 * m11 - m01 * m10) * invDet
        );
    }
    
    // Access operators
    T& operator()(int row, int col) {
        return m[row][col];
    }
    
    const T& operator()(int row, int col) const {
        return m[row][col];
    }
    
    T& operator[](int index) {
        return data[index];
    }
    
    const T& operator[](int index) const {
        return data[index];
    }
    
    std::string toString() const {
        return "Mat3([" + std::to_string(m00) + ", " + std::to_string(m01) + ", " + std::to_string(m02) + "],\n" +
               "      [" + std::to_string(m10) + ", " + std::to_string(m11) + ", " + std::to_string(m12) + "],\n" +
               "      [" + std::to_string(m20) + ", " + std::to_string(m21) + ", " + std::to_string(m22) + "])";
    }
};

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat3i = Mat3<int>;
using Mat3i8 = Mat3<int8_t>;
using Mat3ui8 = Mat3<uint8_t>;
using Mat3b = Mat3<bool>;

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Mat3<T>& mat) {
    os << mat.toString();
    return os;
}

template<typename T>
inline Mat3<T> operator*(T scalar, const Mat3<T>& mat) {
    return mat * scalar;
}

#endif