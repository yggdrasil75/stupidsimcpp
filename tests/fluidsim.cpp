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
    cam.origin = Eigen::Vector3f(0.0f, 400.0f, -1800.0f); // Back and slightly up
    cam.direction = (Eigen::Vector3f(0.0f, 0.0f, 0.0f) - cam.origin).normalized(); // Look at center
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
        
        // --- Spawning Logic ---
        // Spawn throughout the simulation (e.g., until 90% completion)
        if (frameIdx % 10 == 0) {
            if (frameIdx < (TOTAL_FRAMES * 0.9f)) {
                float t = static_cast<float>(frameIdx) / (TOTAL_FRAMES * 0.9f);
                
                // Linear interpolation for Mass: Heavy (10.0) -> Light (0.5)
                float currentMass = (1.0f - t) * 10.0f + t * 0.5f;
                
                // Linear interpolation for Size: Large (12.0) -> Small (2.0)
                float currentSize = (1.0f - t) * 12.0f + t * 2.0f;

                // Number of particles to spawn this frame (can increase as they get smaller)
                int spawnCount = 2 + static_cast<int>(t * 4); 

                sim.spawnParticles(baseParticle, spawnCount, currentSize, currentMass);
            }
        }

        // --- Physics Step ---
        sim.applyPhysics();

        // --- Render & Save ---
        if (frameIdx % 50 == 0) {
            std::cout << "Rendering Frame " << frameIdx << " / " << TOTAL_FRAMES << std::endl;
            
            // Generate LODs for faster/better rendering
            sim.grid.generateLODs();

            // Render frame
            frame renderedFrame = sim.grid.renderFrame(cam, HEIGHT, WIDTH, frame::colormap::RGB, 4, 4, true, false);
            
            // Save to BMP
            std::stringstream ss;
            ss << "output/frame_" << std::setw(4) << std::setfill('0') << frameIdx << ".bmp";
            BMPWriter::saveBMP(ss.str(), renderedFrame);
            sim.grid.printStats();
        }
    }

    std::cout << "Simulation Complete." << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}