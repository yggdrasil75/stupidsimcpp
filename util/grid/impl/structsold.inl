#pragma once
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <memory>

namespace Grid{

constexpr int Dim = 3;

static constexpr uint8_t ACTIVE_BIT = 1 << 0;
static constexpr uint8_t VISIBLE_BIT = 1 << 1;
static constexpr uint8_t STATIC_BIT = 1 << 7;

static constexpr uint8_t LEAF_BIT = 1 << 0;
static constexpr uint8_t LOADED_BIT = 1 << 1;
static constexpr uint8_t DIRTY_BIT = 1 << 2;
static constexpr uint8_t LOADQUEUED = 1 << 3;
static constexpr uint8_t SAVEDQUEUED = 1 << 4;
static constexpr uint8_t KEEPLOADED_BIT = 1 << 5;
static constexpr uint8_t FAT_BIT = 1 << 6;

static constexpr uint8_t OBJ_ALLOW_PARTIAL_UNLOAD_BIT = 1 << 0;

static constexpr uint8_t WORKER_ON = 1 << 0;
static constexpr uint8_t AUTO_OPTIMIZE = 1 << 1;
static constexpr uint8_t QUEUE_STREAMING = 1 << 2;
static constexpr uint8_t PHYSICS_COLLIDER_DIRTY = 1 << 3;

template<typename> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
using PointType = Eigen::Matrix<float, Dim, 1>;
using BoundingBox = std::pair<PointType, PointType>;
namespace fs = std::filesystem;



struct SPHIntegratePC {
    float dt;
    float velocityDamping;
    uint32_t numParticles;
};

struct SPHDensityPC {
    float h;
    float h2;
    float poly6_k;
    float restDensity;
    float gasConstant;
    uint32_t numParticles;
};

struct SPHForcePC {
    float h;
    float spiky_k;
    float visc_l_k;
    float viscosity;
    float gravX;
    float gravY;
    float gravZ;
    float gravStrength;
    float gravCX;
    float gravCY;
    float gravCZ;
    uint32_t useGravityPoint;
    uint32_t numParticles;
    float airDensity;
};

template<typename GasT>
struct EulerianGasState_ {
    GasT data{};
    Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};
    float density = 0.0f;
    float pressure = 0.0f;
    
    int objectId = -1;
    uint16_t renderMatIdx = 0;
    Eigen::Vector3f color{1.0f, 1.0f, 1.0f};
};

}