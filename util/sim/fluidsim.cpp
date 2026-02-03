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
#include "../util/timing_decorator.cpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/noise/pnoise.cpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"


struct fluidParticle {
    Eigen::Matrix<float, 3, 1> pos;
    Eigen::Matrix<float, 3, 1> velocity;
    Eigen::Matrix<float, 3, 1> acceleration;
    Eigen::Matrix<float, 3, 1> forceAccumulator;
    float pressure;
    float viscosity;
    float restitution;
    float mass;
};

struct gridDefaults {
    float gridSizeCube = 1024;
};

Eigen::Matrix<float, 3, 1> posGen() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1024.0f);
    
    return Eigen::Matrix<float, 3, 1>(
        dist(gen),
        dist(gen),
        dist(gen)
    );
}
class fluidSim {
    std::unordered_map<size_t, Eigen::Matrix<float, 3, 1>> idposMap;
    Octree<fluidParticle> grid;
    
    void spawnParticles(fluidParticle toSpawn, int count) {
        for (int i = 0; i < count; i++) {
            Eigen::Matrix<float, 3, 1> pos = posGen();
            grid.set(toSpawn, pos, true, {toSpawn.mass / 1000, toSpawn.mass / 1000, toSpawn.mass / 1000}, 1, true, 1, false);
            
        }
    }

    void applyPhysics() {
        for (auto& meh : idposMap) {
            std::shared_ptr<Octree<fluidParticle>::NodeData> mehdata = grid.find(meh.second);
            std::vector<std::shared_ptr<Octree<fluidParticle>::NodeData>> neighbors = grid.findInRadius(meh.second, mehdata->data.mass);
            ///TODO: 
            // apply velocity to have particle move
            // use mass of neighboring particles to change direction towards them
            // ressure pushes particles away if they are too close
            // resitution controls how much pressure builds up based on how close particles are
            // have to find a way to have a clump use their combined mass instead of individual mass.
            //if a particle leaves the grid, destroy it and spawn a new one

            for (auto& neighbor : neighbors) {

                
            }
        }
    }
};