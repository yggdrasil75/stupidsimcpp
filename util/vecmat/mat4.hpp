#ifndef MAT4_HPP
#define MAT4_HPP

#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../ray3.hpp"
#include <array>
#include <cmath>

template<typename T>
class Mat4 {
public:
    union {
        struct {
            T m00, m01, m02, m03,
              m10, m11, m12, m13,
              m20, m21, m22, m23,
              m30, m31, m32, m33;
        };
        T data[16];
        T m[4][4];
    };
    
    // Constructors
    Mat4() : m00(1), m01(0), m02(0), m03(0),
             m10(0), m11(1), m12(0), m13(0),
             m20(0), m21(0), m22(1), m23(0),
             m30(0), m31(0), m32(0), m33(1) {}
             
    Mat4(T scalar) : m00(scalar), m01(scalar), m02(scalar), m03(scalar),
                     m10(scalar), m11(scalar), m12(scalar), m13(scalar),
                     m20(scalar), m21(scalar), m22(scalar), m23(scalar),
                     m30(scalar), m31(scalar), m32(scalar), m33(scalar) {}
    
    Mat4(T m00, T m01, T m02, T m03,
         T m10, T m11, T m12, T m13,
         T m20, T m21, T m22, T m23,
         T m30, T m31, T m32, T m33) : 
         m00(m00), m01(m01), m02(m02), m03(m03),
         m10(m10), m11(m11), m12(m12), m13(m13),
         m20(m20), m21(m21), m22(m22), m23(m23),
         m30(m30), m31(m31), m32(m32), m33(m33) {}
    
    // Identity matrix
    static Mat4<T> identity() {
        return Mat4<T>(1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    }
    
    // Zero matrix
    static Mat4 zero() { return Mat4(0); }
    
    // Translation matrix
    static Mat4 translation(const Vec3<T>& translation) {
        return Mat4(1, 0, 0, translation.x,
                   0, 1, 0, translation.y,
                   0, 0, 1, translation.z,
                   0, 0, 0, 1);
    }
    
    // Rotation matrices
    static Mat4 rotationX(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat4(1, 0, 0, 0,
                   0, cosA, -sinA, 0,
                   0, sinA, cosA, 0,
                   0, 0, 0, 1);
    }
    
    static Mat4 rotationY(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat4(cosA, 0, sinA, 0,
                   0, 1, 0, 0,
                   -sinA, 0, cosA, 0,
                   0, 0, 0, 1);
    }
    
    static Mat4 rotationZ(T angle) {
        T cosA = std::cos(angle);
        T sinA = std::sin(angle);
        return Mat4(cosA, -sinA, 0, 0,
                   sinA, cosA, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    }
    
    // Scaling matrix
    static Mat4 scaling(const Vec3<T>& scale) {
        return Mat4(scale.x, 0, 0, 0,
                   0, scale.y, 0, 0,
                   0, 0, scale.z, 0,
                   0, 0, 0, 1);
    }
    
    // Perspective projection matrix
    static Mat4 perspective(T fov, T aspect, T near, T far) {
        T tanHalfFov = std::tan(fov / static_cast<T>(2));
        T range = near - far;
        
        return Mat4(static_cast<T>(1) / (aspect * tanHalfFov), 0, 0, 0,
                   0, static_cast<T>(1) / tanHalfFov, 0, 0,
                   0, 0, (-near - far) / range, static_cast<T>(2) * far * near / range,
                   0, 0, 1, 0);
    }
    
    // Orthographic projection matrix
    static Mat4 orthographic(T left, T right, T bottom, T top, T near, T far) {
        return Mat4(static_cast<T>(2) / (right - left), 0, 0, -(right + left) / (right - left),
                   0, static_cast<T>(2) / (top - bottom), 0, -(top + bottom) / (top - bottom),
                   0, 0, -static_cast<T>(2) / (far - near), -(far + near) / (far - near),
                   0, 0, 0, 1);
    }
    
    // LookAt matrix (view matrix)
    static Mat4 lookAt(const Vec3<T>& eye, const Vec3<T>& target, const Vec3<T>& up) {
        Vec3<T> z = (eye - target).normalized();
        Vec3<T> x = up.cross(z).normalized();
        Vec3<T> y = z.cross(x);
        
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
    
    Mat4 operator*(T scalar) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    
    Mat4 operator/(T scalar) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }
    
    Vec4<T> operator*(const Vec4<T>& vec) const {
        return Vec4<T>(
            m00 * vec.x + m01 * vec.y + m02 * vec.z + m03 * vec.w,
            m10 * vec.x + m11 * vec.y + m12 * vec.z + m13 * vec.w,
            m20 * vec.x + m21 * vec.y + m22 * vec.z + m23 * vec.w,
            m30 * vec.x + m31 * vec.y + m32 * vec.z + m33 * vec.w
        );
    }
    
    Vec3<T> transformPoint(const Vec3<T>& point) const {
        Vec4<T> result = *this * Vec4<T>(point, static_cast<T>(1));
        return result.xyz() / result.w;
    }
    
    Vec3<T> transformDirection(const Vec3<T>& direction) const {
        Vec4<T> result = *this * Vec4<T>(direction, static_cast<T>(0));
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
    
    Mat4& operator*=(T scalar) {
        *this = *this * scalar;
        return *this;
    }
    
    Mat4& operator/=(T scalar) {
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
    T determinant() const {
        // Using Laplace expansion for 4x4 determinant
        T det = 0;
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
        T det = determinant();
        if (std::abs(det) < static_cast<T>(1e-10)) {
            return Mat4(); // Return identity if not invertible
        }
        
        Mat4 result;
        T invDet = static_cast<T>(1) / det;
        
        // Calculate inverse using adjugate matrix
        result.m00 = (m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31)) * invDet;
        result.m01 = (m01 * (m22 * m33 - m23 * m32) - m02 * (m21 * m33 - m23 * m31) + m03 * (m21 * m32 - m22 * m31)) * -invDet;
        result.m02 = (m01 * (m12 * m33 - m13 * m32) - m02 * (m11 * m33 - m13 * m31) + m03 * (m11 * m32 - m12 * m31)) * invDet;
        result.m03 = (m01 * (m12 * m23 - m13 * m22) - m02 * (m11 * m23 - m13 * m21) + m03 * (m11 * m22 - m12 * m21)) * -invDet;
        
        result.m10 = (m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30)) * -invDet;
        result.m11 = (m00 * (m22 * m33 - m23 * m32) - m02 * (m20 * m33 - m23 * m30) + m03 * (m20 * m32 - m22 * m30)) * invDet;
        result.m12 = (m00 * (m12 * m33 - m13 * m32) - m02 * (m10 * m33 - m13 * m30) + m03 * (m10 * m32 - m12 * m30)) * -invDet;
        result.m13 = (m00 * (m12 * m23 - m13 * m22) - m02 * (m10 * m23 - m13 * m20) + m03 * (m10 * m22 - m12 * m20)) * invDet;
        
        result.m20 = (m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30)) * invDet;
        result.m21 = (m00 * (m21 * m33 - m23 * m31) - m01 * (m20 * m33 - m23 * m30) + m03 * (m20 * m31 - m21 * m30)) * -invDet;
        result.m22 = (m00 * (m11 * m33 - m13 * m31) - m01 * (m10 * m33 - m13 * m30) + m03 * (m10 * m31 - m11 * m30)) * invDet;
        result.m23 = (m00 * (m11 * m23 - m13 * m21) - m01 * (m10 * m23 - m13 * m20) + m03 * (m10 * m21 - m11 * m20)) * -invDet;
        
        result.m30 = (m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30)) * -invDet;
        result.m31 = (m00 * (m21 * m32 - m22 * m31) - m01 * (m20 * m32 - m22 * m30) + m02 * (m20 * m31 - m21 * m30)) * invDet;
        result.m32 = (m00 * (m11 * m32 - m12 * m31) - m01 * (m10 * m32 - m12 * m30) + m02 * (m10 * m31 - m11 * m30)) * -invDet;
        result.m33 = (m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)) * invDet;
        
        return result;
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
        return "Mat4([" + std::to_string(m00) + ", " + std::to_string(m01) + ", " + std::to_string(m02) + ", " + std::to_string(m03) + "],\n" +
               "      [" + std::to_string(m10) + ", " + std::to_string(m11) + ", " + std::to_string(m12) + ", " + std::to_string(m13) + "],\n" +
               "      [" + std::to_string(m20) + ", " + std::to_string(m21) + ", " + std::to_string(m22) + ", " + std::to_string(m23) + "],\n" +
               "      [" + std::to_string(m30) + ", " + std::to_string(m31) + ", " + std::to_string(m32) + ", " + std::to_string(m33) + "])";
    }
};

// Stream output operator
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Mat4<T>& mat) {
    os << mat.toString();
    return os;
}

// Scalar multiplication from left
template<typename T>
inline Mat4<T> operator*(T scalar, const Mat4<T>& mat) {
    return mat * scalar;
}

using Mat4f = Mat4<float>;
using Mat4d = Mat4<double>;


Mat4f lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
    Vec3f const f = (center - eye).normalized();
    Vec3f const s = f.cross(up).normalized();
    Vec3f const u = s.cross(f);

    Mat4f Result = Mat4f::identity();
    Result(0, 0) = s.x;
    Result(1, 0) = s.y;
    Result(2, 0) = s.z;
    Result(3, 0) = -s.dot(eye);
    Result(0, 1) = u.x;
    Result(1, 1) = u.y;
    Result(2, 1) = u.z;
    Result(3, 1) = -u.dot(eye);
    Result(0, 2) = -f.x;
    Result(1, 2) = -f.y;
    Result(2, 2) = -f.z;
    Result(3, 2) = f.dot(eye);
    return Result;
}

Mat4f perspective(float fovy, float aspect, float zNear, float zfar) {
    float const tanhalfF = tan(fovy / 2);
    Mat4f Result = 0;
    Result(0,0) = 1 / (aspect * tanhalfF);
    Result(1,1) = 1 / tanhalfF;
    Result(2,2) = zfar / (zNear - zfar);
    Result(2,3) = -1;
    Result(3,2) = -(zfar * zNear) / (zfar - zNear);
    return Result;
}


#endif