#ifndef FLUIDSIM_HPP
#define FLUIDSIM_HPP

#include <map>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>

#include "../util/grid/grid3eigen.hpp"
#include "../util/output/frame.hpp"

struct fluidParticle {
    Eigen::Matrix<float, 3, 1> velocity;
    Eigen::Matrix<float, 3, 1> acceleration;
    Eigen::Matrix<float, 3, 1> forceAccumulator;
    float density = 0.0f;
    float pressure = 0.0f;
    Eigen::Matrix<float, 3, 1> pressureForce;
    float viscosity = 0.5f;
    Eigen::Matrix<float, 3, 1> viscosityForce;
    float restitution = 5.0f;
    float mass;
};

struct gridConfig {
    float gridSizeCube = 8192;
    float SMOOTHING_RADIUS = 1024.0f;
    float REST_DENSITY = 0.00005f;
    float TIMESTEP = 0.016f;
    float G_ATTRACTION = 50.0f;
};

Eigen::Matrix<float, 3, 1> posGen() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1024.0f);
    
    return Eigen::Matrix<float, 3, 1>(dist(gen),dist(gen),dist(gen));
}

Eigen::Matrix<float, 3, 1> velGen() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    
    return Eigen::Matrix<float, 3, 1>(dist(gen),dist(gen),dist(gen));
}

float W_poly6(Eigen::Vector3f rv, float h) {
    float r = rv.squaredNorm();
    
    if (r > h || r < 0) return 0;
    float factor = 315 / (64 * M_PI * pow(h, 9));
    float m = pow(h*h-r, 3);
    return factor * m;
}

Eigen::Vector3f gradW_poly6(Eigen::Vector3f rv, float h) {
    float r = rv.squaredNorm();
    float h2 = h * h;

    if (r > h2 || r < 0) {
        return Eigen::Vector3f::Zero();
    }

    float m = -6 * pow(h*h-r, 2);
    float factor = -945.0f / (32.0f * M_PI * std::pow(h, 9));

    return factor * m * rv;
}

float lapW_poly6(Eigen::Vector3f rv, float h) {
    float r = rv.squaredNorm();
    float h2 = h * h;

    if (r > h || r < 0) return 0;

    float m = h2 - r;
    float term2 = 3.0f * h2 - 7.0f * r;
    float factor = -945.0f / (32.0f * M_PI * std::pow(h, 9));

    return factor * m * term2;
}

float W_spiky(Eigen::Vector3f rv, float h) {
    float r = rv.norm();
    if (r > h || r < 0) return 0;

    float coeff = pow(r-h, 3);
    float factor = 15 / (M_PI * pow(h, 6));

    return factor * coeff;
}

Eigen::Vector3f gradW_spiky(Eigen::Vector3f rv, float h) {
    float r = rv.norm();

    if (r > h || r < 0) {
        return Eigen::Vector3f::Zero();
    }

    float diff = h - r;
    float coeff = -45.0f / (M_PI * std::pow(h, 6));
    
    Eigen::Vector3f direction = rv / r;

    return coeff * std::pow(diff, 2) * direction;
}

float W_visc(Eigen::Vector3f rv, float h) {
    float r = rv.norm();

    if (r > h || r < 0) return 0;

    float r2 = r * r;
    float r3 = r2 * r;
    float h3 = h * h * h;
    
    float coeff = 15.0f / (2.0f * M_PI * h3);
    float term = (-0.5f * r3 / h3) + (r2 / (h * h)) + (h / (2.0f * r)) - 1.0f;

    return coeff * term;
}

float lapW_visc(Eigen::Vector3f rv, float h) {
    float r = rv.norm();

    if (r > h || r < 0) return 0;

    float diff = h - r;
    float coeff = 45.0f / (M_PI * std::pow(h, 6));

    return coeff * diff;
}

Eigen::Vector3f buildGradient(float value, const std::map<float, Eigen::Vector3f>& gradientKeys) {
    auto exactMatch = gradientKeys.find(value);
    if (exactMatch != gradientKeys.end()) {
        return exactMatch->second;
    }
    auto lower = gradientKeys.lower_bound(value);
    if (lower == gradientKeys.begin()) {
        return gradientKeys.begin()->second;
    }
    if (lower == gradientKeys.end()) {
        return gradientKeys.rbegin()->second;
    }
    auto upper = lower;
    lower = std::prev(lower);
    float key1 = lower->first;
    float key2 = upper->first;
    const Eigen::Vector3f& color1 = lower->second;
    const Eigen::Vector3f& color2 = upper->second;
    float t = (value - key1) / (key2 - key1);
    t = std::clamp(t, 0.0f, 1.0f);
    return color1 + t * (color2 - color1);
}

class fluidSim {
private:
    std::unordered_map<size_t, Eigen::Matrix<float, 3, 1>> idposMap;
    float newMass = 1000;
    int nextObjectId = 0;
    std::map<float, Eigen::Vector3f> gradientmap;
public:
    gridConfig config;
    float closeThresh;
    Octree<fluidParticle> grid;

    fluidSim() : grid({-config.gridSizeCube, -config.gridSizeCube, -config.gridSizeCube}, {config.gridSizeCube, config.gridSizeCube, config.gridSizeCube}) {
        closeThresh = 0.01f * config.SMOOTHING_RADIUS;
        //grid.setBackgroundColor({0.1f, 0.1f, 0.2f});
        gradientmap.emplace(0.0, Eigen::Vector3f(1, 0, 0));
        gradientmap.emplace(0.5, Eigen::Vector3f(0, 1, 0));
        gradientmap.emplace(1.0, Eigen::Vector3f(0, 0, 1));
    }

    void reset() {
        grid.clear();
        idposMap.clear();
        nextObjectId = 0;
        newMass = 1000;
    }

    void spawnParticles(fluidParticle toSpawn, int count, bool resize = true) {
        TIME_FUNCTION;
        toSpawn.mass = newMass;
        Eigen::Vector3f color = buildGradient(toSpawn.mass / 1000, gradientmap);
        for (int i = 0; i < count; i++) {
            Eigen::Matrix<float, 3, 1> pos = posGen();
            int id = nextObjectId++;
            
            grid.insert(toSpawn, pos, true, color, 10, true, id, (toSpawn.mass > 100) ? true : false, 1);
            idposMap.emplace(id, pos);
        }
        if (resize){
            newMass *= 0.999f;
        }
    }

    float sphKernel(Eigen::Vector3f rv) {
        float r = rv.norm();
        if (r < closeThresh) return W_spiky(rv, config.G_ATTRACTION);
        else return W_poly6(rv,config.G_ATTRACTION);
    }

    void computeDensities() {
        for (auto& point : idposMap) {
            float densSum = 0;
            auto node = grid.find(point.second);
            if (!node) continue;
            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(point.second, config.SMOOTHING_RADIUS);
            for (auto& neighbor : neighbors) {
                Eigen::Vector3f rv = node->position - neighbor->position;
                float w = sphKernel(rv);
                densSum += neighbor->data.mass * w;
            }
            node->data.density = densSum;
        }
    }

    void applyPressure() {
        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) {
                continue;
            }
            node->data.pressure = node->data.restitution * (node->data.density - config.REST_DENSITY);
        }

        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) {
                continue;
            }
            Eigen::Vector3f pressureForce = Eigen::Vector3f::Zero();

            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(point.second, config.SMOOTHING_RADIUS);
            for (auto& neighbor : neighbors) {
                if (node == neighbor) continue;
                Eigen::Vector3f rv = node->position - neighbor->position;
                Eigen::Vector3f gradW = gradW_spiky(rv, config.SMOOTHING_RADIUS);
                float scalarP = (node->data.pressure + neighbor->data.pressure) / (2.0f * neighbor->data.density);

                pressureForce -= neighbor->data.mass * scalarP * gradW;
            }
            node->data.pressureForce = pressureForce;
        }
    }

    void applyViscosity() {
        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            Eigen::Vector3f viscosityForce = Eigen::Vector3f::Zero();
            if (node->data.velocity == Eigen::Vector3f::Zero()) node->data.velocity = velGen();
            
            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(point.second, config.SMOOTHING_RADIUS);
            for (auto& neighbor : neighbors) {
                Eigen::Vector3f rv = node->position - neighbor->position;
                Eigen::Vector3f velDiff = neighbor->data.velocity - node->data.velocity;
                float lapW = lapW_visc(rv, config.SMOOTHING_RADIUS);
                viscosityForce += node->data.viscosity * neighbor->data.mass * (velDiff / neighbor->data.density) * lapW;
            }
            node->data.viscosityForce = viscosityForce;
        }
    }
    
    Eigen::Vector3f applyMutualGravity(std::shared_ptr<Octree<fluidParticle>::NodeData>& node) {
        Eigen::Vector3f gravityForce = Eigen::Vector3f::Zero();
        std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(node->position, config.SMOOTHING_RADIUS);
        for (auto& neighbor : neighbors) {
            if (node == neighbor) continue;
            Eigen::Vector3f dir = neighbor->position - node->position;
            float dist = dir.norm();
            if (dist > EPSILON) {
                dir.normalize();
                float forceMag = (config.G_ATTRACTION * node->data.mass * neighbor->data.mass) / (dist * dist);
                gravityForce += dir * forceMag;
            }
        }
        return gravityForce;
    }
    
    Eigen::Vector3f applyCenterGravity(std::shared_ptr<Octree<fluidParticle>::NodeData>& node) {
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        Eigen::Vector3f direction = center - node->position;
        float distSq = direction.squaredNorm();
        float dist = std::sqrt(distSq);
        if (dist < 50.0f) {
            dist = 50.0f;
            distSq = 2500.0f; 
        }
        direction /= dist;
        float centerMass = 5000.0f; 
        float forceMag = (config.G_ATTRACTION * node->data.mass * centerMass) / distSq;
        return direction * forceMag;
    }

    void applyForce() {
        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            Eigen::Vector3f internalForces = node->data.pressureForce + node->data.viscosityForce;
            Eigen::Vector3f acceleration = (internalForces / node->data.mass);
            Eigen::Vector3f gravity = applyMutualGravity(node);
            gravity += applyCenterGravity(node);
            node->data.forceAccumulator = (acceleration + gravity) / node->data.mass;
        }
    }

    void replaceLost() {
        std::vector<size_t> idsToRemove;
        int gridHalfSize = config.gridSizeCube / 2;
        for (auto& point : idposMap) { 
            if (std::abs(point.second[0]) > gridHalfSize || 
                std::abs(point.second[1]) > gridHalfSize ||
                std::abs(point.second[2]) > gridHalfSize) {
                idsToRemove.push_back(point.first);
            }
        }

        for (size_t id : idsToRemove) {
            grid.remove(idposMap[id]);
            idposMap.erase(id);
        }

        if (!idsToRemove.empty()) {
            fluidParticle newParticles;
            spawnParticles(newParticles, idsToRemove.size(), false);
        }
    }

    void applyPhysics() {
        computeDensities();
        applyPressure();
        applyViscosity();
        applyForce();

        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            Eigen::Matrix<float, 3, 1> acceleration = node->data.forceAccumulator;
            node->data.velocity += acceleration * config.TIMESTEP;
            Eigen::Matrix<float, 3, 1> newPos = point.second + (node->data.velocity);
            Eigen::Vector3f oldPos = point.second;
            if (grid.move(oldPos, newPos)) {
                idposMap[point.first] = newPos;
            }
        }
        replaceLost();
    }
    
    size_t getParticleCount() const { return idposMap.size(); }
};

#endif