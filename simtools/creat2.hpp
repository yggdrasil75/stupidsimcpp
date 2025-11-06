//TODO: implement this.
#ifndef creature_2d
#define creature_2d

#include "../util/vec2.hpp"
#include "../util/ray2.hpp"
#include "../util/grid2.hpp"
#include <random>
#include <cmath>

class Crea2 {
public:
    Vec2 position;
    Vec2 facing;
    float currentSpeed;
    float maxSpeed;
    float currentHealth;
    float maxHealth;
    Grid2 visualRepresentation;
    
    Crea2() 
        : position(Vec2(0, 0)), 
          facing(Vec2(1, 0)), 
          currentSpeed(0), 
          maxSpeed(1), 
          currentHealth(1), 
          maxHealth(1) {
        createDefaultVisual();
    }
    
    Crea2(const Vec2& pos, const Vec2& faceDir, float speed, float health)
        : position(pos), 
          facing(faceDir.normalized()), 
          currentSpeed(speed), 
          maxSpeed(speed), 
          currentHealth(health), 
          maxHealth(health) {
        createDefaultVisual();
    }
    
    // Random initialization method
    void randomInit(int size = 5) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> posDist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> speedDist(0.5f, 3.0f);
        std::uniform_real_distribution<float> healthDist(50.0f, 150.0f);
        
        // Random position
        position = Vec2(posDist(gen), posDist(gen));
        
        // Random facing direction
        float angle = angleDist(gen);
        facing = Vec2(std::cos(angle), std::sin(angle));
        
        // Random speed
        maxSpeed = speedDist(gen);
        currentSpeed = maxSpeed * 0.7f; // Start at 70% of max speed
        
        // Random health
        maxHealth = healthDist(gen);
        currentHealth = maxHealth * 0.8f; // Start at 80% of max health
        
        // Create random visual representation
        createRandomVisual(size);
    }
    
    // Movement methods
    void move(float deltaTime) {
        position += facing * currentSpeed * deltaTime;
    }
    
    void rotate(float angle) {
        facing = facing.rotate(angle).normalized();
        updateVisualRotation();
    }
    
    void setFacing(const Vec2& newFacing) {
        facing = newFacing.normalized();
        updateVisualRotation();
    }
    
    // Health methods
    void takeDamage(float damage) {
        currentHealth = std::max(0.0f, currentHealth - damage);
        updateVisualHealth();
    }
    
    void heal(float amount) {
        currentHealth = std::min(maxHealth, currentHealth + amount);
        updateVisualHealth();
    }
    
    bool isAlive() const {
        return currentHealth > 0;
    }
    
    // Speed methods
    void accelerate(float acceleration, float deltaTime) {
        currentSpeed = std::min(maxSpeed, currentSpeed + acceleration * deltaTime);
    }
    
    void decelerate(float deceleration, float deltaTime) {
        currentSpeed = std::max(0.0f, currentSpeed - deceleration * deltaTime);
    }
    
    // Get ray representing creature's forward direction
    Ray2 getForwardRay() const {
        return Ray2(position, facing);
    }
    
    // Get ray representing creature's view direction
    Ray2 getViewRay(float length = 10.0f) const {
        return Ray2(position, facing * length);
    }
    
    // Get bounding circle radius for collision detection
    float getBoundingRadius() const {
        return 0.5f; // Approximate radius based on visual size
    }
    
    // Check if point is inside creature's bounding area
    bool containsPoint(const Vec2& point) const {
        return position.distance(point) <= getBoundingRadius();
    }
    
    // Get health percentage (0.0 to 1.0)
    float getHealthPercentage() const {
        return currentHealth / maxHealth;
    }
    
private:
    void createDefaultVisual() {
        visualRepresentation.clear();
        
        // Simple triangle shape pointing forward
        visualRepresentation.addPoint(Vec2(0.3f, 0.0f), Vec4(1, 1, 1, 1));  // Nose - white
        visualRepresentation.addPoint(Vec2(-0.2f, 0.2f), Vec4(0, 1, 0, 1)); // Left - green
        visualRepresentation.addPoint(Vec2(-0.2f, -0.2f), Vec4(0, 1, 0, 1)); // Right - green
        visualRepresentation.addPoint(Vec2(-0.1f, 0.0f), Vec4(0, 0.5f, 0, 1)); // Center back - dark green
        
        updateVisualRotation();
        updateVisualHealth();
    }
    
    void createRandomVisual(int size) {
        visualRepresentation.clear();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> colorDist(0.2f, 1.0f);
        std::uniform_real_distribution<float> offsetDist(-0.1f, 0.1f);
        
        // Create a random creature shape
        int points = std::max(3, size); // At least 3 points
        
        // Main body points in a circular pattern
        for (int i = 0; i < points; i++) {
            float angle = 2.0f * M_PI * i / points;
            float radius = 0.2f + offsetDist(gen) + 0.1f * (points / 5.0f);
            
            Vec2 point(std::cos(angle) * radius, std::sin(angle) * radius);
            
            // Shift points slightly backward to create forward direction
            point.x -= 0.1f;
            
            // Random color based on creature properties
            Vec4 color(colorDist(gen), colorDist(gen), colorDist(gen), 1.0f);
            
            // Health affects red component, speed affects blue component
            color.x *= getHealthPercentage();
            color.z *= (currentSpeed / maxSpeed);
            
            visualRepresentation.addPoint(point, color);
        }
        
        // Add a "head" or forward indicator
        visualRepresentation.addPoint(Vec2(0.3f, 0.0f), Vec4(1, 1, 1, 1));
        
        updateVisualRotation();
        updateVisualHealth();
    }
    
    void updateVisualRotation() {
        // Rotate all position points to match facing direction
        float currentAngle = facing.angle();
        
        for (auto& pos : visualRepresentation.positions) {
            pos = pos.rotate(currentAngle);
        }
    }
    
    void updateVisualHealth() {
        // Update colors based on health
        float healthPercent = getHealthPercentage();
        
        for (auto& color : visualRepresentation.colors) {
            // Fade to red as health decreases
            color.x = color.x * healthPercent;
            color.y = color.y * healthPercent;
            color.z = color.z * (1.0f - healthPercent) * 0.5f; // Add some blue when injured
        }
    }
};

#endif