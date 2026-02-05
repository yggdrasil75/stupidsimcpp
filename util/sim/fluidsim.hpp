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

#include "../util/grid/grid3eigen.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/frame.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/noise/pnoise.cpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

struct fluidParticle {
    Eigen::Matrix<float, 3, 1> velocity;
    Eigen::Matrix<float, 3, 1> acceleration;
    Eigen::Matrix<float, 3, 1> forceAccumulator;
    float density = 0.0f;
    float pressure = 0.0f;
    float viscosity = 25.0f;
    float restitution = 500.0f;
    float mass;
};

struct gridConfig {
    float gridSizeCube = 1024;
    const float SMOOTHING_RADIUS = 32.0f;
    const float REST_DENSITY = 0.5f;
    const float TIMESTEP = 0.016f;
    const float G_ATTRACTION = 50.0f;
};

Eigen::Matrix<float, 3, 1> posGen() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1024.0f);
    
    return Eigen::Matrix<float, 3, 1>(dist(gen),dist(gen),dist(gen));
}

// Math Helpers for SPH Kernels
float poly6Kernel(float r, float h) {
    if (r < 0 || r > h) return 0.0f;
    float h2 = h * h;
    float r2 = r * r;
    float diff = h2 - r2;
    float coef = 315.0f / (64.0f * M_PI * pow(h, 9));
    return coef * diff * diff * diff;
}

Eigen::Vector3f spikyGradientKernel(Eigen::Vector3f r_vec, float r, float h) {
    if (r <= 0 || r > h) return Eigen::Vector3f::Zero();
    float diff = h - r;
    float coef = -45.0f / (M_PI * pow(h, 6));
    return r_vec.normalized() * coef * diff * diff;
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
    gridConfig config;
    int nextObjectId = 0;
    std::map<float, Eigen::Vector3f> gradientmap;
public:
    Octree<fluidParticle> grid;

    fluidSim() : grid({-config.gridSizeCube, -config.gridSizeCube, -config.gridSizeCube}, {config.gridSizeCube, config.gridSizeCube, config.gridSizeCube}) {
        grid.setBackgroundColor({0.1f, 0.1f, 0.2f});
        gradientmap.emplace(0.0, Eigen::Vector3f(1, 0, 0));
        gradientmap.emplace(0.5, Eigen::Vector3f(0, 1, 0));
        gradientmap.emplace(1.0, Eigen::Vector3f(0, 0, 1));

    }

    void spawnParticles(fluidParticle toSpawn, int count, bool resize = true) {
        TIME_FUNCTION;
        float size = toSpawn.mass / 10;
        Eigen::Vector3f color = buildGradient(toSpawn.mass / 1000, gradientmap);
        std::cout << "spawning " << count << " particles with a mass of " << toSpawn.mass << " and a size of " << size << std::endl;
        for (int i = 0; i < count; i++) {
            Eigen::Matrix<float, 3, 1> pos = posGen();
            int id = nextObjectId++;
            
            grid.set(toSpawn, pos, true, color, size, true, id, (toSpawn.mass > 100) : true ? false, 1);
            idposMap[id] = pos;
        }
        if (resize){
            newMass -= newMass * .01;
        }
    }

    void applyPressureDensity() {
        #pragma omp parallel for
        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(point.second, config.SMOOTHING_RADIUS * node->data.mass);
            float density = 0.0f;
            for (const auto& neighbor : neighbors) {
                if (node == neighbor) continue;
                float dist = (point.second-neighbor->position).norm();
                density += neighbor->data.mass * poly6Kernel(dist, config.SMOOTHING_RADIUS * node->data.mass);
            }
            node->data.density = density;
            float pressure = node->data.restitution * (density - config.REST_DENSITY);
            node->data.pressure = std::max(pressure, 0.0f);
        }
    }
    
    void applyForce() {
        #pragma omp parallel for
        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            Eigen::Vector3f totalForce = -point.second.normalized() * 10.0f * node->data.mass;
            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(point.second, config.SMOOTHING_RADIUS * node->data.mass);

            for (const auto& neighbor : neighbors) {
                if (node == neighbor) continue;
                Eigen::Vector3f diff = point.second - neighbor->position;
                float dist = diff.norm();
                if (dist < EPSILON) continue;
                float densj = neighbor->data.density;
                if (densj < EPSILON) densj = EPSILON;
                float pressj = neighbor->data.pressure;

                Eigen::Vector3f pressureForceVec = -neighbor->data.mass * ((node->data.pressure + pressj) / (2.0f * densj)) * spikyGradientKernel(diff, dist, config.SMOOTHING_RADIUS);
                totalForce += pressureForceVec;
                Eigen::Vector3f velDiff = neighbor->data.velocity - node->data.velocity;
                float viscositCoef = node->data.viscosity * neighbor->data.mass * (1.0f / densj) * poly6Kernel(dist, config.SMOOTHING_RADIUS);
                totalForce += velDiff * viscositCoef;

                float clampDist = std::max(dist, 5.0f);
                Eigen::Vector3f dirToNeighbor = (neighbor->position - node->position).normalized();
                float attract = (config.G_ATTRACTION * node->data.mass * neighbor->data.mass) / (clampDist * clampDist);
                totalForce += dirToNeighbor * attract;
            }
            node->data.forceAccumulator = totalForce;
        }
    }

    void replaceLost() {
        std::vector<size_t> idsToRemove;
        for (auto& point : idposMap) { 
            if (!grid.inGrid(point.second)) {
                idsToRemove.push_back(point.first);
            }
        }

        for (size_t id : idsToRemove) {
            grid.remove(idposMap[id]);
            idposMap.erase(id);
        }

        if (!idsToRemove.empty()) {
            fluidParticle newParticles;
            newParticles.mass = newMass;
            spawnParticles(newParticles, idsToRemove.size(), false);
        }
    }

    void applyPhysics() {
        TIME_FUNCTION;
        applyPressureDensity();
        applyForce();

        for (auto& point : idposMap) {
            auto node = grid.find(point.second);
            if (!node) continue;
            Eigen::Matrix<float, 3, 1> acceleration = node->data.forceAccumulator / node->data.mass;
            node->data.velocity += acceleration * config.TIMESTEP;
            Eigen::Matrix<float, 3, 1> newPos = point.second + (node->data.velocity * config.TIMESTEP);
            idposMap[point.first] = newPos;
            grid.move(point.second, newPos);
        }
        replaceLost();
    }
};
