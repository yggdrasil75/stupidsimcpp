#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

// Include Eigen and project headers
#include "../eigen/Eigen/Dense" 
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3render.cpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/output/frame.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/framewriter.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/y4mwriter.hpp"
#include "../util/timing_decorator.hpp"
#include "../util/timing_decorator.cpp"

float smoothNoise(float x, float y, float z, float scale) {
    float nx = x * scale;
    float ny = y * scale;
    float nz = z * scale;
    // Basic non-repeating trig combinations for pseudo-random organic variance
    float val = std::sin(nx + std::cos(ny)) + std::sin(ny + std::cos(nz)) + std::sin(nz + std::cos(nx));
    // Normalize roughly to 0.0 - 1.0
    return (val + 3.0f) / 6.0f;
}

// Helper function to create a solid volume of voxels with material properties
void createBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, const Eigen::Vector3f& albedo, float emission = 0.0f,
               float roughness = 0.8f, float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f, const Eigen::Vector3f& absorp = Eigen::Vector3f::Zero(), 
               int oid = 0, Grid::BodyType bType = Grid::BodyType::STATIC, float mass = 1.0f, float step = 0.1, const Eigen::Vector3f& sellB = Eigen::Vector3f::Zero(), const Eigen::Vector3f& sellC = Eigen::Vector3f::Zero(), bool useSell = false) {
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                
                octree.insert(1, pos, true, albedo, step, true, oid, emission, roughness, metallic, transmission, ior, absorp, bType, mass);
                if (useSell) {
                    octree.setSellmeier(pos, sellB.cast<Eigen::half>(), sellC.cast<Eigen::half>());
                }
            }
        }
    }
}

void createTarnishedBrassBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, float step, int oid, Grid::BodyType bType, float mass) {
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    
    Eigen::Vector3f cleanBrass(0.78f, 0.69f, 0.22f);
    Eigen::Vector3f tarnishColor(0.25f, 0.30f, 0.15f); // Dark greenish-brown oxidization

    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                // Lower frequency noise for more gradual sweeps of tarnish
                float noise = smoothNoise(x, y, z, 4.0f);
                
                // Smooth interpolation (smoothstep equivalent) between noise 0.3 and 0.7
                float t = std::max(0.0f, std::min(1.0f, (noise - 0.3f) / 0.4f));
                float blend = t * t * (3.0f - 2.0f * t); // cubic ease-in ease-out
                
                Eigen::Vector3f albedo = cleanBrass * (1.0f - blend) + tarnishColor * blend;
                float roughness = 0.08f * (1.0f - blend) + 0.8f * blend;
                float metallic = 0.99f * (1.0f - blend) + 0.1f * blend;
                
                Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                octree.insert(1, pos, true, albedo, step, true, oid, 0.0f, roughness, metallic, 0.0f, 1.18f, Eigen::Vector3f::Zero(), bType, mass);
            }
        }
    }
}

void createTexturedBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, const Eigen::Vector3f& baseColor, float step, int oid, Grid::BodyType bType, float mass) {
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                float n = smoothNoise(x, y, z, 25.0f);
                
                // Determine if this voxel is on the outer crust of the box
                float edgeDist = std::min({x - minB.x(), maxB.x() - x,
                                           y - minB.y(), maxB.y() - y,
                                           z - minB.z(), maxB.z() - z});
                
                // Skip outer voxels to create geometric pitting/roughness
                if (edgeDist < step * 1.5f && n > 0.65f) {
                    continue; 
                }
                
                // Slightly perturb the color of the remaining volume for visual texture
                Eigen::Vector3f color = baseColor * (0.7f + 0.5f * n); 
                color = color.cwiseMin(Eigen::Vector3f::Constant(1.0f)).cwiseMax(Eigen::Vector3f::Constant(0.0f));
                
                Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                octree.insert(1, pos, true, color, step, true, oid, 0.0f, 0.6f + 0.4f * n, 0.0f, 0.0f, 1.46f, Eigen::Vector3f::Zero(), bType, mass);
            }
        }
    }
}

// --- NEW: Enum & Function for complex geometric Gem cuts ---
enum class GemCut { OCTAHEDRON, HEXAGONAL_BIPYRAMID };

void createGem(Grid::Octree<int>& octree, const Eigen::Vector3f& center, float radius, GemCut cut, 
               const Eigen::Vector3f& albedo, float step, int oid, Grid::BodyType bType, float mass, 
               float transmission, float ior, const Eigen::Vector3f& absorp, 
               const Eigen::Vector3f& sellB = Eigen::Vector3f::Zero(), const Eigen::Vector3f& sellC = Eigen::Vector3f::Zero(), bool useSell = false) {
    
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);

    float minX = center.x() - radius; float maxX = center.x() + radius;
    float minY = center.y() - radius; float maxY = center.y() + radius;
    float minZ = center.z() - radius; float maxZ = center.z() + radius;

    for (float x = minX; x <= maxX; x += step) {
        for (float y = minY; y <= maxY; y += step) {
            for (float z = minZ; z <= maxZ; z += step) {
                float dx = x - center.x();
                float dy = y - center.y();
                float dz = z - center.z();
                
                bool inside = false;
                
                if (cut == GemCut::OCTAHEDRON) {
                    // Classic dual-pyramid diamond/ruby shape
                    inside = (std::abs(dx) + std::abs(dy) + std::abs(dz)) <= radius;
                } else if (cut == GemCut::HEXAGONAL_BIPYRAMID) {
                    // Crystal cluster look (Hexagon that tapers off at the Z ends)
                    float hexDist = std::max(std::abs(dx), (std::abs(dx) + std::abs(dy) * 1.732f) / 2.0f);
                    inside = (hexDist + std::abs(dz) * 0.6f) <= (radius * 0.85f) && std::abs(dz) <= radius;
                }

                if (inside) {
                    Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                    octree.insert(1, pos, true, albedo, step, true, oid, 0.0f, 0.01f, 0.0f, transmission, ior, absorp, bType, mass);
                    if (useSell) {
                        octree.setSellmeier(pos, sellB.cast<Eigen::half>(), sellC.cast<Eigen::half>());
                    }
                }
            }
        }
    }
}

// Helper function to create a checkerboard pattern volume
void createCheckerBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, 
                      const Eigen::Vector3f& color1, const Eigen::Vector3f& color2, float checkerSize) {
    float step = 0.1f;
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x, y, z);
                
                // Use floor to correctly handle negative coordinates for the repeating pattern
                int cx = static_cast<int>(std::floor(x / checkerSize));
                int cy = static_cast<int>(std::floor(y / checkerSize));
                int cz = static_cast<int>(std::floor(z / checkerSize));
                
                // 3D Checkerboard logic
                bool isEven = ((cx + cy + cz) % 2 == 0);
                Eigen::Vector3f albedo = isEven ? color1 : color2;
                
                octree.insert(1, pos, true, albedo, step, true, 100, 0.0f, 0.01f, 0.0f, 0.0f, 1.486f);
            }
        }
    }
}

size_t createWaterDrop(Grid::Octree<int>& octree, const Eigen::Vector3f& center, float radius,
                       int count, int oid, float totalMass = 0.1f) {
    static std::mt19937 rng(9001);
    std::uniform_real_distribution<float> jitter(-0.15f, 0.15f);

    auto teardropRadiusAt = [](float t) -> float {
        if (t < 0.0f || t > 1.0f) return 0.0f;
        return std::sqrt(std::max(0.0f, 1.0f - t * t)) * (1.0f - t * 0.55f);
    };

    const float halfHeight = radius;
    const float maxWidth = radius * 0.85f;

    float dropVolume = 0.0f;
    {
        const int slices = 512;
        const float dz = (2.0f * halfHeight) / slices;
        for (int i = 0; i < slices; ++i) {
            float t = (i + 0.5f) / slices;
            float r = teardropRadiusAt(t) * maxWidth;
            dropVolume += 3.14159265f * r * r * dz;
        }
    }

    const float voxelSize = std::cbrt(dropVolume / static_cast<float>(count)) * 0.93f;
    const float step = voxelSize;

    struct Cand { Eigen::Vector3f pos; float rank; };
    std::vector<Cand> candidates;

    for (float z = -halfHeight; z <= halfHeight; z += step) {
        float t = (z + halfHeight) / (2.0f * halfHeight);
        float rAtZ = teardropRadiusAt(t) * maxWidth;
        if (rAtZ <= 0.0f) continue;

        for (float x = -rAtZ; x <= rAtZ; x += step) {
            for (float y = -rAtZ; y <= rAtZ; y += step) {
                if (x * x + y * y > rAtZ * rAtZ) continue;
                Eigen::Vector3f local(x + jitter(rng) * step, y + jitter(rng) * step, z + jitter(rng) * step);
                float radial = std::sqrt(x * x + y * y);
                float depth = radial / std::max(rAtZ, 1e-6f);
                candidates.push_back({local, depth});
            }
        }
    }

    if (static_cast<int>(candidates.size()) > count) {
        std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(),
                         [](const Cand& a, const Cand& b) { return a.rank < b.rank; });
        candidates.resize(count);
    }

    const float perVoxelMass = totalMass / std::max<size_t>(1, candidates.size());

    Eigen::Vector3f cWater(0.85f, 0.92f, 1.0f);
    Eigen::Vector3f waterAbsorp(0.06f, 0.02f, 0.01f);

    for (const auto& c : candidates) {
        Eigen::Vector3f pos = center + c.pos;
        octree.insert(1, pos, true, cWater, voxelSize, true, oid, 0.0f,
                      0.02f, 0.0f, 0.92f, 1.333f, waterAbsorp,
                      Grid::BodyType::FLUID, perVoxelMass);
        octree.setSellmeier(pos,
            Eigen::Vector3f(5.684027565e-1f, 1.726177391e-1f, 2.086189578e-2f).cast<Eigen::half>(),
            Eigen::Vector3f(5.101829712e-3f, 1.821153936e-2f, 2.620722293e-2f).cast<Eigen::half>());
    }

    return candidates.size();
}

enum class TargetState {
    FLUID,
    GAS,
    RIGID
};

struct StateEvent {
    int frameTrigger;
    int objectId;
    float mass;
    bool isMoltenMetal;
    TargetState targetState;
    Eigen::Vector3f gasColor      = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    Eigen::Vector3f gasAbsorption = Eigen::Vector3f::Zero();
    float gasMassScale            = 1.0f;
};

int main() {
    std::cout << "Initializing Grid::Octree..." << std::endl;

    // 1. Initialize Grid::Octree bounds
    Eigen::Vector3f minBound(-10.0f, -10.0f, -10.0f);
    Eigen::Vector3f maxBound(10.0f, 10.0f, 10.0f);
    Grid::Octree<int> octree(minBound, maxBound, "output/renderscene", 4);
    
    // Set a dark background to emphasize the PBR light emission
    octree.setBackgroundColor(Eigen::Vector3f(0.02f, 0.02f, 0.02f));
    octree.setSkylight(Eigen::Vector3f(0.01f, 0.01f, 0.01f));
    octree.setphys_gravityCenter(Eigen::Vector3f(0.0f, 0.0f, -1000.0f));
    octree.setPhysicsSmoothingRadius(0.2f);
    octree.setPhysicsGasConstant(100.0f); // Lowered significantly from 2000 for stability
    octree.setPhysicsVelocityDamping(1.0f); // Higher damping stops infinite scattering
    octree.setPhysicsViscosity(15.0f);
    octree.setPhysicsAirDensity(1.225f);
    // octree.setPhysicsSurfaceTension(2000.0f);


    std::cout << "Building scene..." << std::endl;

    // 2a. Build Room (Floor and 4 Walls)
    Eigen::Vector3f cLightGray(0.8f, 0.8f, 0.8f);
    Eigen::Vector3f cDarkGray(0.2f, 0.2f, 0.2f);
    float chkSize = 1.0f;

    // Floor
    createCheckerBox(octree, Eigen::Vector3f(0.0f, 0.0f, -0.6f), Eigen::Vector3f(14.4f, 14.4f, 0.2f), cLightGray, cDarkGray, chkSize);
    
    // Walls
    createCheckerBox(octree, Eigen::Vector3f( 7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // +X
    createCheckerBox(octree, Eigen::Vector3f(-7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // -X
    createCheckerBox(octree, Eigen::Vector3f( 0.0f,  7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // +Y
    createCheckerBox(octree, Eigen::Vector3f( 0.0f, -7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // -Y
    
    const Eigen::Vector3f leakPoint(0.0f, 5.4f, 7.3f);
    const float leakRadius = 0.35f;
    {
        Eigen::Vector3f ceilingCenter(0.0f, 0.0f, 7.4f);
        Eigen::Vector3f ceilingSize(14.4f, 14.4f, 0.2f);
        Eigen::Vector3f lightSize(8.0f, 8.0f, 0.2f);
        float step = 0.5f;
        float lightStep = 0.1f;

        Eigen::Vector3f minCeiling = ceilingCenter - ceilingSize / 2.0f;
        Eigen::Vector3f maxCeiling = ceilingCenter + ceilingSize / 2.0f;
        Eigen::Vector3f minLight = ceilingCenter - lightSize / 2.0f;
        Eigen::Vector3f maxLight = ceilingCenter + lightSize / 2.0f;

        Eigen::Vector3f cBlack(0.01f, 0.01f, 0.01f);
        Eigen::Vector3f cWhite(1.0f, 1.0f, 1.0f);

        auto inLeakHole = [&](float x, float y) {
            float dx = x - leakPoint.x();
            float dy = y - leakPoint.y();
            return (dx * dx + dy * dy) <= (leakRadius * leakRadius);
        };

        for (float x = minCeiling.x(); x <= maxCeiling.x(); x += step) {
            for (float y = minCeiling.y(); y <= maxCeiling.y(); y += step) {
                for (float z = minCeiling.z(); z <= maxCeiling.z(); z += step) {
                    bool isLightArea = (x >= minLight.x() && x <= maxLight.x() &&
                                        y >= minLight.y() && y <= maxLight.y());
                    
                    if (!isLightArea && !inLeakHole(x, y)) {
                        Eigen::Vector3f pos(x, y, z);
                        octree.insert(1, pos, true, cBlack, step, true, 100, 0.0f, 0.8f, 0.2f, 1.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                    }
                }
            }
        }

        {
            Eigen::Vector3f cWetStain(0.16f, 0.13f, 0.10f);
            float rimOuter = leakRadius * 2.2f;
            for (float x = leakPoint.x() - rimOuter; x <= leakPoint.x() + rimOuter; x += lightStep) {
                for (float y = leakPoint.y() - rimOuter; y <= leakPoint.y() + rimOuter; y += lightStep) {
                    float dx = x - leakPoint.x();
                    float dy = y - leakPoint.y();
                    float d = std::sqrt(dx * dx + dy * dy);
                    if (d < leakRadius || d > rimOuter) continue;

                    float wet = 1.0f - (d - leakRadius) / (rimOuter - leakRadius);
                    float n = smoothNoise(x, y, 0.0f, 18.0f);
                    Eigen::Vector3f albedo = cBlack * (1.0f - wet) + cWetStain * wet;
                    float roughness = 0.8f * (1.0f - wet * 0.7f) + 0.05f * n;

                    for (float z = minCeiling.z(); z <= maxCeiling.z(); z += lightStep) {
                        Eigen::Vector3f pos(x, y, z);
                        octree.insert(1, pos, true, albedo, lightStep, true, 100, 0.0f, roughness, 0.1f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                    }
                }
            }
        }
     
        for (float z = minCeiling.z(); z <= maxCeiling.z(); z += lightStep) {
            for (float x = minLight.x(); x <= maxLight.x(); x += lightStep) {
                for (float y = minLight.y(); y <= maxLight.y(); y += lightStep) {
                    Eigen::Vector3f pos(x, y, z);
                    
                    bool isExposedLightLayer = (z < minCeiling.z() + lightStep);

                    if (isExposedLightLayer) {
                        float u = (x - minLight.x()) / lightSize.x();
                        float v = (y - minLight.y()) / lightSize.y();
                        
                        bool inCanton = (u <= 0.4f) && (v >= (1.0f - 7.0f / 13.0f));
                        Eigen::Vector3f flagColor;
                        
                        if (inCanton) {
                            float starU = u / 0.4f; 
                            float cantonVStart = 1.0f - 7.0f / 13.0f;
                            float starV = (v - cantonVStart) / (7.0f / 13.0f);
                            int cx = static_cast<int>(starU * 11.0f);
                            int cy = static_cast<int>(starV * 9.0f);
                            
                            // Simple alternating pattern to represent stars
                            if ((cx + cy) % 2 == 0) {
                                flagColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
                            } else {
                                flagColor = Eigen::Vector3f(0.05f, 0.05f, 0.8f);
                            }
                        } else {
                            // 13 Stripes
                            int stripeIndex = static_cast<int>((1.0f - v) * 13.0f);
                            if (stripeIndex >= 13) stripeIndex = 12; // Safety clamp
                            
                            if (stripeIndex % 2 == 0) {
                                flagColor = Eigen::Vector3f(1.0f, 0.05f, 0.1f); // Red
                            } else {
                                flagColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f); // White
                            }
                        }

                        // Insert the glowing voxel
                        octree.insert(1, pos, true, flagColor, lightStep, true, 10, 2.0f, 0.8f, 0.0f, 1.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                        // Add explicit chromaticity for colored glowing 
                        octree.setEmittance(pos, flagColor, 0.01f);
                    } else {
                        // Fill backing layers with dark static voxels so it has a ceiling body above the light
                        octree.insert(1, pos, true, cBlack, lightStep, true, 100, 0.0f, 0.8f, 0.2f, 1.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                    }
                }
            }
        }
    }

    // 2b. Create the 3x3 material sampler grid inside the room
    Eigen::Vector3f cRed(1.0f, 0.1f, 0.1f);
    Eigen::Vector3f cRuby(0.878, 0.066, 0.3725);
    Eigen::Vector3f cBlue(0.1f, 0.1f, 1.0f);
    Eigen::Vector3f cPurple(0.6f, 0.1f, 0.8f);
    Eigen::Vector3f cAmethyst(0.6,0.4,0.8);
    Eigen::Vector3f size(1.0f, 1.0f, 1.0f);
    Eigen::Vector3f cGold(1.00f, 0.80f, 0.30f);
    Eigen::Vector3f cSilver(0.90f, 0.90f, 0.95f);
    Eigen::Vector3f cBrass(0.78f, 0.69f, 0.22f);

    float sp = 2.0f; // spacing between cubes
    Grid::BodyType initType = Grid::BodyType::STATIC;
    float mass = 1.0f;

    // LAYER 1: Metals
    createBox(octree, Eigen::Vector3f(-sp, -sp, 0.0f), size * 1.5, cGold,   0.0f, 0.08f, 0.99f, 0.0f, 0.47f, Eigen::Vector3f(0,0,0), 1, initType, mass);
    createBox(octree, Eigen::Vector3f(  0, -sp, 0.0f), size * 1.5, cSilver, 0.0f, 0.08f, 0.99f, 0.0f, 0.13f, Eigen::Vector3f(0,0,0), 2, initType, mass);
    createTarnishedBrassBox(octree, Eigen::Vector3f( sp, -sp, 0.0f), size * 1.5, 0.1f, 3, initType, mass);

    // LAYER 2: Opaque
    createTexturedBox(octree, Eigen::Vector3f(-sp,  0,  0.0f), size, cRed,    0.1f, 4, initType, mass);
    createTexturedBox(octree, Eigen::Vector3f(  0,  0,  0.0f), size, cBlue,   0.1f, 5, initType, mass);
    createTexturedBox(octree, Eigen::Vector3f( sp,  0,  0.0f), size, cPurple, 0.1f, 6, initType, mass);

    // LAYER 3: Glass
    createGem(octree, Eigen::Vector3f(-sp,  sp, 0.0f), 0.45f, GemCut::OCTAHEDRON, cRuby, 0.1f, 7, initType, mass, 0.95f, 1.757f, Eigen::Vector3f(0.05f, 0.8f, 0.8f), Eigen::Vector3f(1.4360479f, 0.64583146f, 3.4556846f), Eigen::Vector3f(0.0052998009f, 0.014262926f, 210.80888f), true); 
    createBox(octree, Eigen::Vector3f(  0,  sp, 0.0f), size * 0.5, cBlue,   0.0f, 0.01f, 0.0f, 0.89f, 1.309f, Eigen::Vector3f(0.08f, 0.02f, 0.01f), 8, initType, mass, 0.5, Eigen::Vector3f(0.54727636f, 0.15459328f, 0.13445437f), Eigen::Vector3f(0.0053423668f, 0.019974298f, 10.596549f), true); 
    createGem(octree, Eigen::Vector3f( sp,  sp, 0.0f), 0.45f, GemCut::HEXAGONAL_BIPYRAMID, cAmethyst, 0.1f, 9, initType, mass, 0.97f, 1.534f, Eigen::Vector3f(0.8f, 0.6f, 0.05f), Eigen::Vector3f(0.696,0.407,0.897), Eigen::Vector3f(0.0046,0.013, 97.934), true);

    std::cout << "Optimizing and Generating LODs..." << std::endl;
    // octree.generateLODs();
    octree.setLODMinDistance(1024);
    octree.setLODFalloff(0.01);
    octree.printStats();
    octree.setMaxDistance(4096);

    // 3. Setup rendering loop
    int width = 512;
    int height = 512;
    
    const float fps = 60.0f;
    const float durationPerSegment = 10.0f;
    const int framesPerSegment = static_cast<int>(fps * durationPerSegment);
    const int samples = 10;
    const int blendedsamples = 30;
    const float blendedfactor = 0.65;
    const int videosamples = 500;
    const int bounces = 8;
    const int physicsSubsteps = 10;
    const float physicsDt = 1.0f / fps;
    const float subDt = physicsDt / physicsSubsteps;
    const float fluidDuration = 120.0f;
    const int totalFluidFrames = static_cast<int>(fps * fluidDuration);

    struct View {
        std::string name;
        Eigen::Vector3f origin;
        Eigen::Vector3f up;
    };

    std::vector<View> views = {
        {"+X", Eigen::Vector3f( 6.8f,  0.0f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"+Y", Eigen::Vector3f( 0.0f,  6.8f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"-X", Eigen::Vector3f(-6.8f,  0.0f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"+Z", Eigen::Vector3f( 0.0f,  0.0f,  7.3f), Eigen::Vector3f(0.0f, 1.0f, 0.0f)},
        {"-Y", Eigen::Vector3f( 0.0f, -6.8f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"-Z", Eigen::Vector3f( 0.0f,  0.0f,  -1.3f), Eigen::Vector3f(0.0f, -1.0f, 0.0f)}
    };

    std::vector<StateEvent> timeline = {
        { 20,   5,  1.0f, false, TargetState::FLUID   }, // Center -> Water
        
        { 100,  8,  0.00005f, false, TargetState::GAS, Eigen::Vector3f::Constant(1.0f), Eigen::Vector3f::Constant(0.1), 40.0f }, // Second Blue -> steam
        { 180,  6,  1.2f, false, TargetState::FLUID   }, // Purple 1 -> Heavy Water
        { 260,  9,  0.4f, false, TargetState::RIGID   }, // Purple 2 -> RIGID (Floats lightly in fluid)
        { 340,  3,  8.5f, true,  TargetState::FLUID   }, // Brass -> Molten Brass
        { 420,  2, 10.5f, true,  TargetState::FLUID   }, // Silver -> Molten Silver
        { 500,  1, 19.3f, true,  TargetState::FLUID   }, // Gold -> Molten Gold
        
        { 580,  4,  0.001f, false, TargetState::GAS, Eigen::Vector3f(1.0f, 0.3f, 0.25f), Eigen::Vector3f(0.05f, 0.2f, 0.2f), 20.0f }, // Red 1 -> heavier GAS
        { 660,  7,  0.1f, false, TargetState::FLUID   }  // Red 2 -> fluid
    };

    Eigen::Vector3f target(0.0f, 0.0f, 0.5f);
    Grid::FrameWriter writer(2, 8);


    // for (const auto& view : views) {
    //     ScopedFunctionTimer meh("CPU fast section");
    //     std::cout << "\nRendering view from " << view.name << " direction (CPU Fast Pass)..." << std::endl;
    
    //     Camera cam;
    //     cam.origin = view.origin;
    //     cam.direction = (target - view.origin).normalized();
    //     cam.up = view.up;
    
    //     frame cpuout = octree.fastRenderFrame(cam, height, width, frame::colormap::RGB);
    //     writer.enqueue(std::move(cpuout), "output/fast_cpurender_" + view.name + ".bmp");
    // }
    // writer.drain();
    // FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    {
        ScopedFunctionTimer meh("Fast section");
        Grid::InFlightFrame inflight;
        std::string pendingName;
        bool havePending = false;

        for (const auto& view : views) {
            // std::cout << "\nRendering view from " << view.name << " direction (Fast Pass)..." << std::endl;

            Camera cam;
            cam.origin = view.origin;
            cam.direction = (target - view.origin).normalized();
            cam.up = view.up;

            Grid::InFlightFrame next = octree.beginFastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
            if (havePending) {
                frame prev = octree.endFastRenderFrameVulkan(inflight);
                writer.enqueue(std::move(prev), "output/fast_vulkanrender_" + pendingName + ".bmp");
            }
            inflight = next;
            pendingName = view.name;
            havePending = true;
        }
        if (havePending) {
            frame prev = octree.endFastRenderFrameVulkan(inflight);
            writer.enqueue(std::move(prev), "output/fast_vulkanrender_" + pendingName + ".bmp");
        }
    }
    writer.drain();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    {
        ScopedFunctionTimer meh("Gamestyle section");
        Grid::InFlightFrame inflight;
        std::string pendingName;
        bool havePending = false;

        for (const auto& view : views) {
            // std::cout << "\nRendering view from " << view.name << " direction (Gamestyle Pass)..." << std::endl;

            Camera cam;
            cam.origin = view.origin;
            cam.direction = (target - view.origin).normalized();
            cam.up = view.up;

            Grid::InFlightFrame next = octree.beginGameStyleRenderFrame(cam, height, width, frame::colormap::RGB);
            if (havePending) {
                frame prev = octree.endGameStyleRenderFrame(inflight);
                writer.enqueue(std::move(prev), "output/gameready_vulkanrender_" + pendingName + ".bmp");
            }
            inflight = next;
            pendingName = view.name;
            havePending = true;
        }
        if (havePending) {
            frame prev = octree.endGameStyleRenderFrame(inflight);
            writer.enqueue(std::move(prev), "output/gameready_vulkanrender_" + pendingName + ".bmp");
        }
    }
    writer.drain();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    {
        ScopedFunctionTimer meh("Slow Section");
        Grid::InFlightFrame inflight;
        std::string pendingName;
        bool havePending = false;

        for (const auto& view : views) {
            // std::cout << "\nRendering view from " << view.name << " direction (Slow " << samples << " Samples Pass)..." << std::endl;

            Camera cam;
            cam.origin = view.origin;
            cam.direction = (target - view.origin).normalized();
            cam.up = view.up;

            Grid::InFlightFrame next = octree.beginRenderFrameVulkan(cam, height, width, frame::colormap::RGB, samples, bounces, false, true);
            if (havePending) {
                frame prev = octree.endRenderFrameVulkan(inflight);
                writer.enqueue(std::move(prev), "output/slow_vulkanrender_" + pendingName + ".bmp");
            }
            inflight = next;
            pendingName = view.name;
            havePending = true;
            // std::cout << "slow submitted" << std::endl;
        }
        if (havePending) {
            frame prev = octree.endRenderFrameVulkan(inflight);
            writer.enqueue(std::move(prev), "output/slow_vulkanrender_" + pendingName + ".bmp");
        }
    }
    writer.drain();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    {
        ScopedFunctionTimer meh("Superblend Section");
        Grid::InFlightFrame inflight;
        std::string pendingName;
        bool havePending = false;

        for (const auto& view : views) {
            // std::cout << "\nRendering view from " << view.name << " direction (Superblend Pass)..." << std::endl;

            Camera cam;
            cam.origin = view.origin;
            cam.direction = (target - view.origin).normalized();
            cam.up = view.up;

            Grid::InFlightFrame next = octree.beginSuperBlendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, blendedsamples, bounces, false, true);
            if (havePending) {
                frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
                writer.enqueue(std::move(prev), "output/slow_superblendrender_" + pendingName + ".bmp");
            }
            inflight = next;
            pendingName = view.name;
            havePending = true;
            // std::cout << "super blended submitted" << std::endl;
        }
        if (havePending) {
            frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
            writer.enqueue(std::move(prev), "output/slow_superblendrender_" + pendingName + ".bmp");
        }
    }
    writer.drain();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    {
        ScopedFunctionTimer meh("Blend Section");
        Grid::InFlightFrame inflight;
        std::string pendingName;
        bool havePending = false;

        for (const auto& view : views) {
            // std::cout << "\nRendering view from " << view.name << " direction (Blend Pass)..." << std::endl;

            Camera cam;
            cam.origin = view.origin;
            cam.direction = (target - view.origin).normalized();
            cam.up = view.up;

            Grid::InFlightFrame next = octree.beginBlendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, blendedsamples, bounces, false, true);
            if (havePending) {
                frame prev = octree.endBlendedRenderFrameVulkan(inflight);
                writer.enqueue(std::move(prev), "output/slow_blendrender_" + pendingName + ".bmp");
            }
            inflight = next;
            pendingName = view.name;
            havePending = true;
            // std::cout << "blended submitted" << std::endl;
        }
        if (havePending) {
            frame prev = octree.endBlendedRenderFrameVulkan(inflight);
            writer.enqueue(std::move(prev), "output/slow_blendrender_" + pendingName + ".bmp");
        }
    }
    writer.drain();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    // std::vector<frame> videoFrames;
    // const int totalFrames = framesPerSegment * views.size();
    // videoFrames.reserve(totalFrames);
    // int frameCounter = 0;

    // std::cout << "\nStarting video render..." << std::endl;
    // std::cout << "Total frames to render: " << totalFrames << std::endl;

    // for (size_t i = 0; i < views.size(); ++i) {
    //     ScopedFunctionTimer meh("Video");
    //     const View& startView = views[i];
    //     const View& endView = views[(i + 1) % views.size()]; // Loop back to the first view at the end
    //     Grid::InFlightFrame inflight;
    //     std::string pendingName;
    //     bool havePending = false;

    //     std::cout << "\nAnimating segment: " << startView.name << " -> " << endView.name << std::endl;

    //     for (int j = 0; j < framesPerSegment; ++j) {
    //         if (frameCounter < -1) {
    //             frameCounter++;
    //             continue;
    //         }
    //         frameCounter++;
    //         float t = static_cast<float>(j) / static_cast<float>(framesPerSegment);

    //         Eigen::Vector3f currentOrigin = startView.origin * (1.0f - t) + endView.origin * t;
            
    //         Eigen::Vector3f currentUp = (startView.up * (1.0f - t) + endView.up * t).normalized();
            
    //         Camera cam;
    //         cam.origin = currentOrigin;
    //         cam.up = currentUp;
    //         cam.direction = (target - cam.origin).normalized();
            
    //         // std::cout << "Rendering video frame " << frameCounter << "/" << totalFrames << "..." << std::endl;
    //         // frame out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
    //         Grid::InFlightFrame next = octree.beginSuperBlendedRenderFrameVulkan(cam, height * 2, width * 2, blendedfactor, frame::colormap::RGB, videosamples, bounces, false);
    //         if (havePending) {
    //             frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
    //             writer.enqueue(std::move(prev), "output/materialframes/debug_material_" + pendingName + ".bmp");
    //         }
    //         inflight = next;
    //         pendingName = std::to_string(frameCounter);
    //         havePending = true;

    //         // frame out = octree.superBlendedRenderFrameVulkan(cam, height * 2, width * 2, blendedfactor, frame::colormap::RGB, videosamples, bounces, false);
    //         // frame out = octree.renderFrameVulkan(cam, height, width, frame::colormap::RGB, videosamples, bounces, false, true);
    //         // videoFrames.push_back(std::move(out));
    //         // writer.enqueue(std::move(out), "output/materialframes/debug_material_" + std::to_string(frameCounter) + ".bmp");
    //     }
    //     if (havePending) {
    //         frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
    //         writer.enqueue(std::move(prev), "output/materialframes/debug_material_" + pendingName + ".bmp");
    //     }
    // }
    // writer.drain();
    // FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    std::cout << "\nStarting LEAKY CEILING drip simulation..." << std::endl;

    const int   dripObjectId   = 200;
    const int   dripVoxelCount = 100;
    const float dripRadius     = 0.5f;
    const float dripMass       = 0.08f;

    Eigen::Vector3f dripSpawn(leakPoint.x(), leakPoint.y(), leakPoint.z() - dripRadius * 0.9f);

    size_t placed = createWaterDrop(octree, dripSpawn, dripRadius, dripVoxelCount, dripObjectId, dripMass);
    std::cout << "Spawned water drop with " << placed << " voxels at ("
              << dripSpawn.x() << ", " << dripSpawn.y() << ", " << dripSpawn.z() << ")" << std::endl;

    octree.markPhysicsCollidersDirty();

    const float dripDuration   = 8.0f;
    const int totalDripFrames = static_cast<int>(fps * dripDuration);
    const int releaseFrame = static_cast<int>(fps * 2.5f);

    std::vector<std::weak_ptr<Grid::Octree<int>::NodeData>> dropVoxels =
        octree.getWeakNodesByObjectId(dripObjectId);

    Camera dripCam;
    dripCam.fov = 70;
    dripCam.origin = Eigen::Vector3f(3.2f, 2.0f, 4.2f);
    dripCam.up = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f dripTarget(leakPoint.x(), leakPoint.y(), 3.0f);
    dripCam.direction = (dripTarget - dripCam.origin).normalized();

    {
        Grid::InFlightFrame dripInflight;
        std::string dripPending;
        bool dripHavePending = false;

        for (int f = 1; f <= totalDripFrames; ++f) {
            if (f < releaseFrame) {
                for (auto& wp : dropVoxels) {
                    if (auto sp = wp.lock()) {
                        sp->physics.velocity = Eigen::Vector3f::Zero();
                        sp->physics.force = Eigen::Vector3f::Zero();
                        sp->size += 0.00035f;
                    }
                }
            } else {
                if (f == releaseFrame) {
                    std::cout << ">>> Drop released from ceiling at frame " << f << std::endl;
                }
                for (int s = 0; s < physicsSubsteps; ++s) {
                    octree.stepPhysics(subDt);
                }
            }

            Grid::InFlightFrame next = octree.beginSuperBlendedRenderFrameVulkan(
                dripCam, height * 2, width * 2, blendedfactor, frame::colormap::RGB, videosamples, bounces, false);

            if (dripHavePending) {
                frame prev = octree.endSuperBlendedRenderFrameVulkan(dripInflight);
                writer.enqueue(std::move(prev), "output/dripframes/debug_drip_" + dripPending + ".bmp");
            }
            dripInflight = next;
            dripPending = std::to_string(f);
            dripHavePending = true;
        }

        if (dripHavePending) {
            frame prev = octree.endSuperBlendedRenderFrameVulkan(dripInflight);
            writer.enqueue(std::move(prev), "output/dripframes/debug_drip_" + dripPending + ".bmp");
        }
    }

    writer.drain();
    std::cout << "\nDrip simulation complete." << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    writer.shutdown();
    std::cout << "frames written: " << writer.writtenCount() << std::endl;
    return 0;
}