#ifndef FLUIDSIM_CPP
#define FLUIDSIM_CPP

#include "../util/sim/fluidsim.hpp"
#include "../util/timing_decorator.cpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"


int main() {
    fluidSim sim;
    
    // Simulation settings
    const int TOTAL_FRAMES = 10000;
    const int WIDTH = 1024;
    const int HEIGHT = 1024;

    // Setup Camera
    Camera cam;
    cam.origin = Eigen::Vector3f(0.0f, 4000.0f, -5800.0f);
    cam.direction = (Eigen::Vector3f(0.0f, 0.0f, 0.0f) - cam.origin).normalized();
    cam.up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    cam.fov = 60.0f;

    // Base particle template
    fluidParticle baseParticle;
    baseParticle.velocity = Eigen::Vector3f::Zero();
    baseParticle.acceleration = Eigen::Vector3f::Zero();
    baseParticle.forceAccumulator = Eigen::Vector3f::Zero();
    baseParticle.pressure = 0.0f;
    baseParticle.viscosity = 8.0f;
    baseParticle.restitution = 200.0f;

    std::cout << "Starting Fluid Simulation..." << std::endl;
    sim.grid.setLODFalloff(0.001);
    sim.grid.setLODMinDistance(4096);

    for (int frameIdx = 0; frameIdx < TOTAL_FRAMES; frameIdx++) {
        
        if (frameIdx < (TOTAL_FRAMES * 0.1f)) {
            if (frameIdx % 1 == 0) {
                //float t = static_cast<float>(frameIdx) / (TOTAL_FRAMES * 0.9f);
                int spawnCount = 2; // + static_cast<int>(t * 4); 
                sim.spawnParticles(baseParticle, spawnCount);
            }
        }

        sim.applyPhysics();

        if (frameIdx % 100 == 0) {
            std::cout << "Rendering Frame " << frameIdx << " / " << TOTAL_FRAMES << std::endl;
            
            frame renderedFrame = sim.grid.renderFrame(cam, HEIGHT, WIDTH, frame::colormap::RGB, 4, 5, true, false);
            
            std::stringstream ss;
            ss << "output/frame_" << std::setw(4) << std::setfill('0') << frameIdx << ".bmp";
            BMPWriter::saveBMP(ss.str(), renderedFrame);
        } else if (frameIdx % 25 == 0) {
            std::cout << "Rendering quick Frame " << frameIdx << " / " << TOTAL_FRAMES << std::endl;
            
            frame renderedFrame = sim.grid.renderFrame(cam, HEIGHT, WIDTH, frame::colormap::RGB, 1, 1, true, false);
            
            std::stringstream ss;
            ss << "output/frame_" << std::setw(4) << std::setfill('0') << frameIdx << ".bmp";
            BMPWriter::saveBMP(ss.str(), renderedFrame);
        } else if (frameIdx % 5 == 0) {
            std::cout << "Rendering ultrafast Frame " << frameIdx << " / " << TOTAL_FRAMES << std::endl;
            
            frame renderedFrame = sim.grid.fastRenderFrame(cam, HEIGHT / 2, WIDTH / 2, frame::colormap::RGB);
            renderedFrame.resize(HEIGHT, WIDTH, frame::interpolation::NEAREST);
            
            std::stringstream ss;
            ss << "output/frame_" << std::setw(4) << std::setfill('0') << frameIdx << ".bmp";
            BMPWriter::saveBMP(ss.str(), renderedFrame);
        }
        
        if (frameIdx % 500 == 0) {
            sim.grid.printStats();
        }
    }

    sim.grid.printStats();
    frame renderedFrame = sim.grid.renderFrame(cam, HEIGHT, WIDTH, frame::colormap::RGB, 5, 7, true, false);
    
    std::stringstream ss;
    ss << "output/frame_" << std::setw(4) << std::setfill('0') << TOTAL_FRAMES << ".bmp";
    BMPWriter::saveBMP(ss.str(), renderedFrame);

    std::cout << "Simulation Complete." << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}

#endif