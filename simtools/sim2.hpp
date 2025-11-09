#ifndef SIM2_HPP
#define SIM2_HPP

#include "../util/grid/grid2.hpp"
#include "../util/vec2.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

class PhysicsObject {
protected:
    Vec2 position;
    Vec2 velocity;
    Vec2 acceleration;
    float mass;
    bool isStatic;
    std::shared_ptr<Grid2> shape;
    
public:
    PhysicsObject(const Vec2& pos, float mass = 1.0f, bool isStatic = false)
        : position(pos), velocity(0, 0), acceleration(0, 0), mass(mass), isStatic(isStatic) {}
    
    virtual ~PhysicsObject() = default;
    
    virtual void update(float dt) {
        if (isStatic) return;
        
        // Update velocity and position using basic physics
        velocity += acceleration * dt;
        position += velocity * dt;
        acceleration = Vec2(0, 0); // Reset acceleration for next frame
    }
    
    virtual void applyForce(const Vec2& force) {
        if (!isStatic) {
            acceleration += force / mass;
        }
    }
    
    // Getters
    const Vec2& getPosition() const { return position; }
    const Vec2& getVelocity() const { return velocity; }
    float getMass() const { return mass; }
    bool getIsStatic() const { return isStatic; }
    std::shared_ptr<Grid2> getShape() const { return shape; }
    
    // Setters
    void setPosition(const Vec2& pos) { position = pos; }
    void setVelocity(const Vec2& vel) { velocity = vel; }
    void setMass(float m) { mass = m; }
    void setShape(std::shared_ptr<Grid2> newShape) { shape = newShape; }
    
    // Check if this object contains a world position
    virtual bool contains(const Vec2& worldPos) const {
        if (!shape) return false;
        
        Vec2 localPos = worldPos - position;
        return shape->isOccupied(localPos.round());
    }
    
    // Get all occupied positions in world coordinates
    virtual std::vector<Vec2> getWorldPositions() const {
        std::vector<Vec2> worldPositions;
        if (!shape) return worldPositions;
        
        const auto& localPositions = shape->getPositions();
        for (const auto& localPos : localPositions) {
            worldPositions.push_back(position + localPos);
        }
        return worldPositions;
    }
};

class FallingObject : public PhysicsObject {
private:
    Vec2 gravityForce;
    
public:
    FallingObject(const Vec2& pos, float mass = 1.0f, const Vec2& gravity = Vec2(0, -9.8f))
        : PhysicsObject(pos, mass, false), gravityForce(gravity * mass) {}
    
    void update(float dt) override {
        if (!isStatic) {
            // Apply gravity
            applyForce(gravityForce);
        }
        PhysicsObject::update(dt);
    }
    
    void setGravity(const Vec2& gravity) {
        gravityForce = gravity * mass;
    }
    
    const Vec2 getGravity() const {
        return gravityForce / mass;
    }
};

class SolidObject : public PhysicsObject {
public:
    SolidObject(const Vec2& pos, float mass = 1.0f)
        : PhysicsObject(pos, mass, true) {} // Solids are static by default
    
    // Solids don't move, so override update to do nothing
    void update(float dt) override {
        // Solids don't move
    }
    
    void applyForce(const Vec2& force) override {
        // Solids don't respond to forces
    }
};

class Sim2 : public Grid2 {
private:
    std::vector<std::shared_ptr<PhysicsObject>> objects;
    Vec2 globalGravity;
    float elasticity; // Bounciness factor (0.0 to 1.0)
    float friction;   // Surface friction (0.0 to 1.0)
    Vec2 worldBoundsMin, worldBoundsMax;
    bool useWorldBounds;
    
public:
    Sim2(int width, int height, const Vec2& gravity = Vec2(0, -9.8f))
        : Grid2(width, height), globalGravity(gravity), elasticity(0.7f), 
          friction(0.1f), useWorldBounds(false) {}
    
    // Add objects to simulation
    void addObject(std::shared_ptr<PhysicsObject> object) {
        objects.push_back(object);
    }
    
    // Create a falling ball
    std::shared_ptr<FallingObject> createBall(const Vec2& position, float radius, 
                                            const Vec4& color = Vec4(1.0f, 0.5f, 0.0f, 1.0f), 
                                            float mass = 1.0f) {
        auto ball = std::make_shared<FallingObject>(position, mass, globalGravity);
        auto shape = std::make_shared<Grid2>(static_cast<int>(radius * 2 + 2));
        shape->fillCircle(Vec2(radius, radius), radius, color);
        ball->setShape(shape);
        objects.push_back(ball);
        return ball;
    }
    
    // Create a solid ground
    std::shared_ptr<SolidObject> createGround(const Vec2& position, int width, 
                                            const Vec4& color = Vec4(0.0f, 1.0f, 0.0f, 1.0f)) {
        auto ground = std::make_shared<SolidObject>(position);
        auto shape = std::make_shared<Grid2>(width, 1);
        for (int x = 0; x < width; ++x) {
            shape->addPixel(x, 0, color);
        }
        ground->setShape(shape);
        objects.push_back(ground);
        return ground;
    }
    
    // Create a solid wall
    std::shared_ptr<SolidObject> createWall(const Vec2& position, int height,
                                          const Vec4& color = Vec4(0.3f, 0.3f, 0.7f, 1.0f)) {
        auto wall = std::make_shared<SolidObject>(position);
        auto shape = std::make_shared<Grid2>(1, height);
        for (int y = 0; y < height; ++y) {
            shape->addPixel(0, y, color);
        }
        wall->setShape(shape);
        objects.push_back(wall);
        return wall;
    }
    
    // Update the simulation
    void update(float dt) {
        // Clear the main grid
        clear();
        
        // Update all objects
        for (auto& obj : objects) {
            obj->update(dt);
        }
        
        // Handle collisions
        handleCollisions();
        
        // Handle world bounds
        if (useWorldBounds) {
            handleWorldBounds();
        }
        
        // Render all objects to the main grid
        renderObjects();
    }
    
    // Set world bounds for containment
    void setWorldBounds(const Vec2& min, const Vec2& max) {
        worldBoundsMin = min;
        worldBoundsMax = max;
        useWorldBounds = true;
    }
    
    // Simulation parameters
    void setElasticity(float e) { elasticity = std::clamp(e, 0.0f, 1.0f); }
    void setFriction(float f) { friction = std::clamp(f, 0.0f, 1.0f); }
    void setGlobalGravity(const Vec2& gravity) { globalGravity = gravity; }
    
    float getElasticity() const { return elasticity; }
    float getFriction() const { return friction; }
    const Vec2& getGlobalGravity() const { return globalGravity; }
    
    // Get all objects
    const std::vector<std::shared_ptr<PhysicsObject>>& getObjects() const {
        return objects;
    }
    
    // Clear all objects
    void clearObjects() {
        objects.clear();
        clear();
    }

private:
    void handleCollisions() {
        // Simple collision detection between falling objects and solids
        for (size_t i = 0; i < objects.size(); ++i) {
            auto obj1 = objects[i];
            if (obj1->getIsStatic()) continue; // Skip static objects for collision detection
            
            for (size_t j = 0; j < objects.size(); ++j) {
                if (i == j) continue;
                
                auto obj2 = objects[j];
                if (!obj2->getIsStatic()) continue; // Only check against static objects for now
                
                checkAndResolveCollision(obj1, obj2);
            }
        }
    }
    
    void checkAndResolveCollision(std::shared_ptr<PhysicsObject> dynamicObj, 
                                 std::shared_ptr<PhysicsObject> staticObj) {
        if (!dynamicObj->getShape() || !staticObj->getShape()) return;
        
        // Get all world positions of the dynamic object
        auto dynamicPositions = dynamicObj->getWorldPositions();
        
        for (const auto& dynPos : dynamicPositions) {
            // Check if this position collides with the static object
            if (staticObj->contains(dynPos)) {
                // Simple collision response - bounce
                Vec2 velocity = dynamicObj->getVelocity();
                
                // Determine collision normal (simplified - always bounce up)
                Vec2 normal(0, 1);
                
                // Reflect velocity with elasticity
                Vec2 reflectedVel = velocity.reflect(normal) * elasticity;
                
                // Apply friction
                reflectedVel.x *= (1.0f - friction);
                
                dynamicObj->setVelocity(reflectedVel);
                
                // Move object out of collision
                Vec2 newPos = dynamicObj->getPosition();
                newPos.y += 1.0f; // Move up by 1 pixel
                dynamicObj->setPosition(newPos);
                
                break; // Only handle first collision
            }
        }
    }
    
    void handleWorldBounds() {
        for (auto& obj : objects) {
            if (obj->getIsStatic()) continue;
            
            Vec2 pos = obj->getPosition();
            Vec2 vel = obj->getVelocity();
            bool collided = false;
            
            // Check bounds and bounce
            if (pos.x < worldBoundsMin.x) {
                pos.x = worldBoundsMin.x;
                vel.x = -vel.x * elasticity;
                collided = true;
            } else if (pos.x > worldBoundsMax.x) {
                pos.x = worldBoundsMax.x;
                vel.x = -vel.x * elasticity;
                collided = true;
            }
            
            if (pos.y < worldBoundsMin.y) {
                pos.y = worldBoundsMin.y;
                vel.y = -vel.y * elasticity;
                collided = true;
            } else if (pos.y > worldBoundsMax.y) {
                pos.y = worldBoundsMax.y;
                vel.y = -vel.y * elasticity;
                collided = true;
            }
            
            if (collided) {
                obj->setPosition(pos);
                obj->setVelocity(vel);
            }
        }
    }
    
    void renderObjects() {
        // Render all objects to the main grid
        for (const auto& obj : objects) {
            if (!obj->getShape()) continue;
            
            const auto& shape = obj->getShape();
            const Vec2& objPos = obj->getPosition();
            
            // Copy all pixels from object's shape to main grid at object's position
            const auto& positions = shape->getPositions();
            const auto& colors = shape->getColors();
            const auto& sizes = shape->getSizes();
            
            for (size_t i = 0; i < positions.size(); ++i) {
                Vec2 worldPos = objPos + positions[i];
                Vec2 gridPos = worldPos.round();
                
                // Only add if within grid bounds
                if (gridPos.x >= 0 && gridPos.x < width && 
                    gridPos.y >= 0 && gridPos.y < height) {
                    if (!isOccupied(gridPos)) {
                        addPixel(gridPos, colors[i], sizes[i]);
                    }
                }
            }
        }
    }
};

#endif