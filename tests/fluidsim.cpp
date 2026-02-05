#include "../util/sim/fluidsim.hpp"
#include "../util/timing_decorator.cpp"

int main() {
    fluidSim sim;
    
    // Simulation settings
    const int TOTAL_FRAMES = 100000;
    const int WIDTH = 800;
    const int HEIGHT = 600;

    // Setup Camera
    Camera cam;
    cam.origin = Eigen::Vector3f(0.0f, 1000.0f, -1800.0f);
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

    for (int frameIdx = 0; frameIdx < TOTAL_FRAMES; ++frameIdx) {
        
        if (frameIdx % 10 == 0) {
            if (frameIdx < (TOTAL_FRAMES * 0.1f)) {
                //float t = static_cast<float>(frameIdx) / (TOTAL_FRAMES * 0.9f);
                int spawnCount = 2; // + static_cast<int>(t * 4); 
                sim.spawnParticles(baseParticle, spawnCount);
            }
        }

        sim.applyPhysics();

        if (frameIdx % 50 == 0) {
            std::cout << "Rendering Frame " << frameIdx << " / " << TOTAL_FRAMES << std::endl;
            
            frame renderedFrame = sim.grid.renderFrame(cam, HEIGHT, WIDTH, frame::colormap::RGB, 4, 4, true, false);
            
            std::stringstream ss;
            ss << "output/frame_" << std::setw(4) << std::setfill('0') << frameIdx << ".bmp";
            BMPWriter::saveBMP(ss.str(), renderedFrame);
            
        }
        if (frameIdx % 100 == 0) {
            sim.grid.printStats();
        }
    }

    std::cout << "Simulation Complete." << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}