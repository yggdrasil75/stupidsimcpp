#ifndef camera_hpp
#define camera_hpp

#include "../../eigen/Eigen/Dense"
#include <cmath>
#include "../basicdefines.hpp"

using Eigen::Vector3f;
using Eigen::Matrix3f;

struct Camera {
    Vector3f origin;
    Vector3f direction;
    Vector3f up;
    float fov;
    
    Camera(const Vector3f& pos, const Vector3f& viewdir, const Vector3f& up, float fov = 80) 
        : origin(pos), direction(viewdir), up(up.normalized()), fov(fov) {}
    
    void rotateYaw(float angle) {
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        Vector3f right = direction.cross(up).normalized();
        
        // Rotate around up vector (yaw)
        Matrix3f rotation;
        rotation = Eigen::AngleAxisf(angle, up);
        direction = rotation * direction;
    }
    
    void rotatePitch(float angle) {
        // Clamp pitch to avoid gimbal lock
        Vector3f right = direction.cross(up).normalized();
        
        // Rotate around right vector (pitch)
        Matrix3f rotation;
        rotation = Eigen::AngleAxisf(angle, right);
        direction = rotation * direction;
        direction.normalize();
        
        // Recalculate up vector to maintain orthogonality
        up = right.cross(direction).normalized();
    }

    Vector3f forward() const {
        return direction.normalized();
    }

    Vector3f right() const {
        return forward().cross(up).normalized();
    }

    float fovRad() const {
        return fov * (M_PI / 180.0f);
    }
    
    // Additional useful methods
    void moveForward(float distance) {
        origin += forward() * distance;
    }
    
    void moveRight(float distance) {
        origin += right() * distance;
    }
    
    void moveUp(float distance) {
        origin += up * distance;
    }
    
    // Get view matrix (lookAt matrix)
    Eigen::Matrix4f getViewMatrix() const {
        Vector3f f = forward();
        Vector3f r = right();
        Vector3f u = up;
        
        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        
        view(0, 0) = r.x(); view(0, 1) = r.y(); view(0, 2) = r.z();
        view(1, 0) = u.x(); view(1, 1) = u.y(); view(1, 2) = u.z();
        view(2, 0) = -f.x(); view(2, 1) = -f.y(); view(2, 2) = -f.z();
        
        view(0, 3) = -r.dot(origin);
        view(1, 3) = -u.dot(origin);
        view(2, 3) = f.dot(origin);
        
        return view;
    }
    
    // Get projection matrix (perspective)
    Eigen::Matrix4f getProjectionMatrix(float aspectRatio, float nearPlane = 0.1f, float farPlane = 100.0f) const {
        float fovrad = fovRad();
        float tanHalfFov = tan(fovrad / 2.0f);
        
        Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
        
        projection(0, 0) = 1.0f / (aspectRatio * tanHalfFov);
        projection(1, 1) = 1.0f / tanHalfFov;
        projection(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        projection(3, 2) = -1.0f;
        projection(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
        
        return projection;
    }
};

#endif