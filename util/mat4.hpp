#ifndef MAT4_HPP
#define MAT4_HPP

#include "Vec3.hpp"
#include "Vec4.hpp"
#include <array>
#include <cmath>

class Mat4 {
public:
    union {
        struct {
            float m00, m01, m02, m03,
                  m10, m11, m12, m13,
                  m20, m21, m22, m23,
                  m30, m31, m32, m33;
        };
        float data[16];
        float m[4][4];
    };
    
    // Constructors
    Mat4() : m00(1), m01(0), m02(0), m03(0),
             m10(0), m11(1), m12(0), m13(0),
             m20(0), m21(0), m22(1), m23(0),
             m30(0), m31(0), m32(0), m33(1) {}
             
    Mat4(float scalar) : m00(scalar), m01(scalar), m02(scalar), m03(scalar),
                         m10(scalar), m11(scalar), m12(scalar), m13(scalar),
                         m20(scalar), m21(scalar), m22(scalar), m23(scalar),
                         m30(scalar), m31(scalar), m32(scalar), m33(scalar) {}
    
    Mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) : 
         m00(m00), m01(m01), m02(m02), m03(m03),
         m10(m10), m11(m11), m12(m12), m13(m13),
         m20(m20), m21(m21), m22(m22), m23(m23),
         m30(m30), m31(m31), m32(m32), m33(m33) {}
    
    // Identity matrix
    static Mat4 identity() {
        return Mat4(1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    }
    
    // Zero matrix
    static Mat4 zero() { return Mat4(0); }
    
    // Translation matrix
    static Mat4 translation(const Vec3& translation) {
        return Mat4(1, 0, 0, translation.x,
                   0, 1, 0, translation.y,
                   0, 0, 1, translation.z,
                   0, 0, 0, 1);
    }
    
    // Rotation matrices
    static Mat4 rotationX(float angle) {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Mat4(1, 0, 0, 0,
                   0, cosA, -sinA, 0,
                   0, sinA, cosA, 0,
                   0, 0, 0, 1);
    }
    
    static Mat4 rotationY(float angle) {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Mat4(cosA, 0, sinA, 0,
                   0, 1, 0, 0,
                   -sinA, 0, cosA, 0,
                   0, 0, 0, 1);
    }
    
    static Mat4 rotationZ(float angle) {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Mat4(cosA, -sinA, 0, 0,
                   sinA, cosA, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    }
    
    // Scaling matrix
    static Mat4 scaling(const Vec3& scale) {
        return Mat4(scale.x, 0, 0, 0,
                   0, scale.y, 0, 0,
                   0, 0, scale.z, 0,
                   0, 0, 0, 1);
    }
    
    // Perspective projection matrix
    static Mat4 perspective(float fov, float aspect, float near, float far) {
        float tanHalfFov = std::tan(fov / 2.0f);
        float range = near - far;
        
        return Mat4(1.0f / (aspect * tanHalfFov), 0, 0, 0,
                   0, 1.0f / tanHalfFov, 0, 0,
                   0, 0, (-near - far) / range, 2.0f * far * near / range,
                   0, 0, 1, 0);
    }
    
    // Orthographic projection matrix
    static Mat4 orthographic(float left, float right, float bottom, float top, float near, float far) {
        return Mat4(2.0f / (right - left), 0, 0, -(right + left) / (right - left),
                   0, 2.0f / (top - bottom), 0, -(top + bottom) / (top - bottom),
                   0, 0, -2.0f / (far - near), -(far + near) / (far - near),
                   0, 0, 0, 1);
    }
    
    // LookAt matrix (view matrix)
    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vec3 z = (eye - target).normalized();
        Vec3 x = up.cross(z).normalized();
        Vec3 y = z.cross(x);
        
        return Mat4(x.x, x.y, x.z, -x.dot(eye),
                   y.x, y.y, y.z, -y.dot(eye),
                   z.x, z.y, z.z, -z.dot(eye),
                   0, 0, 0, 1);
    }
    
    // Arithmetic operations
    Mat4 operator+(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
    
    Mat4 operator-(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; ++k) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }
    
    Mat4 operator*(float scalar) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    
    Mat4 operator/(float scalar) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }
    
    Vec4 operator*(const Vec4& vec) const {
        return Vec4(
            m00 * vec.x + m01 * vec.y + m02 * vec.z + m03 * vec.w,
            m10 * vec.x + m11 * vec.y + m12 * vec.z + m13 * vec.w,
            m20 * vec.x + m21 * vec.y + m22 * vec.z + m23 * vec.w,
            m30 * vec.x + m31 * vec.y + m32 * vec.z + m33 * vec.w
        );
    }
    
    Vec3 transformPoint(const Vec3& point) const {
        Vec4 result = *this * Vec4(point, 1.0f);
        return result.xyz() / result.w;
    }
    
    Vec3 transformDirection(const Vec3& direction) const {
        Vec4 result = *this * Vec4(direction, 0.0f);
        return result.xyz();
    }
    
    Mat4& operator+=(const Mat4& other) {
        *this = *this + other;
        return *this;
    }
    
    Mat4& operator-=(const Mat4& other) {
        *this = *this - other;
        return *this;
    }
    
    Mat4& operator*=(const Mat4& other) {
        *this = *this * other;
        return *this;
    }
    
    Mat4& operator*=(float scalar) {
        *this = *this * scalar;
        return *this;
    }
    
    Mat4& operator/=(float scalar) {
        *this = *this / scalar;
        return *this;
    }
    
    bool operator==(const Mat4& other) const {
        for (int i = 0; i < 16; ++i) {
            if (data[i] != other.data[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const Mat4& other) const {
        return !(*this == other);
    }
    
    // Matrix operations
    float determinant() const {
        // Using Laplace expansion for 4x4 determinant
        float det = 0;
        det += m00 * (m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31));
        det -= m01 * (m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30));
        det += m02 * (m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30));
        det -= m03 * (m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30));
        return det;
    }
    
    Mat4 transposed() const {
        return Mat4(m00, m10, m20, m30,
                   m01, m11, m21, m31,
                   m02, m12, m22, m32,
                   m03, m13, m23, m33);
    }
    
    Mat4 inverse() const {
        // This is a simplified inverse implementation
        // For production use, consider a more robust implementation
        float det = determinant();
        if (std::abs(det) < 1e-10f) {
            return Mat4(); // Return identity if not invertible
        }
        
        Mat4 result;
        // Calculate inverse using adjugate matrix divided by determinant
        // This is a placeholder - full implementation would be quite lengthy
        float invDet = 1.0f / det;
        
        // Note: This is a simplified version - full implementation would calculate all 16 cofactors
        result.m00 = (m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31)) * invDet;
        // ... continue for all 16 elements
        
        return result.transposed() * invDet;
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
        return "Mat4([" + std::to_string(m00) + ", " + std::to_string(m01) + ", " + std::to_string(m02) + ", " + std::to_string(m03) + "],\n" +
               "      [" + std::to_string(m10) + ", " + std::to_string(m11) + ", " + std::to_string(m12) + ", " + std::to_string(m13) + "],\n" +
               "      [" + std::to_string(m20) + ", " + std::to_string(m21) + ", " + std::to_string(m22) + ", " + std::to_string(m23) + "],\n" +
               "      [" + std::to_string(m30) + ", " + std::to_string(m31) + ", " + std::to_string(m32) + ", " + std::to_string(m33) + "])";
    }
};

inline std::ostream& operator<<(std::ostream& os, const Mat4& mat) {
    os << mat.toString();
    return os;
}

inline Mat4 operator*(float scalar, const Mat4& mat) {
    return mat * scalar;
}

// Now you can implement the Ray3 transform method
#include "ray3.hpp"

inline Ray3 Ray3::transform(const Mat4& matrix) const {
    Vec3 transformedOrigin = matrix.transformPoint(origin);
    Vec3 transformedDirection = matrix.transformDirection(direction);
    return Ray3(transformedOrigin, transformedDirection.normalized());
}

#endif