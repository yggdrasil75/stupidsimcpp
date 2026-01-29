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
    float movementSpeed;
    float rotationSpeed;

    Camera() : origin(Vector3f(0,0,0)), direction(Vector3f(0,0,0)), up(Vector3f(0,0,0)), fov(80), movementSpeed(10), rotationSpeed(10) {}
    
    Camera(const Vector3f& pos, const Vector3f& viewdir, const Vector3f& up, float fov = 80,
           float moveSpeed = 1.0f, float rotSpeed = 0.5f) 
        : origin(pos), direction(viewdir.normalized()), up(up.normalized()), fov(fov), movementSpeed(moveSpeed), rotationSpeed(rotSpeed) {}
    
    void rotateYaw(float angle) {
        angle *= rotationSpeed;
        Matrix3f rotation;
        rotation = Eigen::AngleAxisf(angle, up);
        direction = rotation * direction;
        direction.normalize();
    }
    
    void rotatePitch(float angle) {
        angle *= rotationSpeed;
        Vector3f right = direction.cross(up).normalized();
        
        // Rotate around right vector (pitch)
        Matrix3f rotation;
        rotation = Eigen::AngleAxisf(angle, right);
        direction = rotation * direction;
        direction.normalize();
        
        // Recalculate up vector to maintain orthogonality
        up = right.cross(direction).normalized();
    }

    void moveForward(float distance) {
        origin += forward() * distance * movementSpeed;
    }
    
    void moveBackward(float distance) {
        origin -= forward() * distance * movementSpeed;
    }
    
    void moveRight(float distance) {
        origin += right() * distance * movementSpeed;
    }
    
    void moveLeft(float distance) {
        origin -= right() * distance * movementSpeed;
    }
    
    void moveUp(float distance) {
        origin += up * distance * movementSpeed;
    }
    
    void moveDown(float distance) {
        origin -= up * distance * movementSpeed;
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
    
    // Look at a specific point
    void lookAt(const Vector3f& target) {
        direction = (target - origin).normalized();
        
        // Recalculate up vector
        Vector3f worldUp(0, 1, 0);
        if (direction.cross(worldUp).norm() < 0.001f) {
            worldUp = Vector3f(0, 0, 1);
        }
        
        Vector3f right = direction.cross(worldUp).normalized();
        up = right.cross(direction).normalized();
    }
    
    // Set position directly
    void setPosition(const Vector3f& pos) {
        origin = pos;
    }
    
    // Set view direction directly
    void setDirection(const Vector3f& dir) {
        direction = dir.normalized();
        // Recalculate up
        Vector3f worldUp(0, 1, 0);
        Vector3f right = direction.cross(worldUp).normalized();
        up = right.cross(direction).normalized();
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
    Eigen::Matrix4f getProjectionMatrix(float aspectRatio, float nearPlane = 0.1f, float farPlane = 1000.0f) const {
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
    
    void mouseLook(float deltaX, float deltaY) {
        float yaw = -deltaX * 0.001f;
        float pitch = -deltaY * 0.001f;
        
        rotateYaw(yaw);
        rotatePitch(pitch);
    }
};

#endif