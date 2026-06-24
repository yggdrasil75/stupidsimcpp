#pragma once
#ifdef VULKAN_SUPPORT
#include <vulkan/vulkan.h>
#include "../../noise/bluetilepool.hpp"
#endif

namespace Grid {

struct RenderData_ {
    PointType position;
    float size;
    Eigen::Vector4f color;
    uint32_t materialIdx;
    PointType boundsMin;
    PointType boundsMax;
    int objectId;
    uint32_t isGas;
};
struct GPUGasField {
    Eigen::Vector4f boundsMin;
    Eigen::Vector4f boundsMax;
    Eigen::Vector4f cellSize;
    uint32_t res;
    uint32_t cellOffset;
    uint32_t slotCount;
    uint32_t pad0;
    uint32_t slotToGlobal[8];
};

template<typename T, typename IndexType>
struct RenderNode_ {
    PointType boundsMin;
    PointType boundsMax;
    PointType center;
    float nodeSize;
    bool isLeaf;
    bool isLoaded;
    uint8_t childMask;
    
    uint32_t firstPoint;
    uint32_t pointCount;
    int32_t lodPoint;
    uint32_t firstChild;
    
    OctreeNode_<T, IndexType>* originalNode;
};

template<typename T, typename IndexType>
struct RenderBuffer_ {
    std::vector<RenderNode_<T, IndexType>> nodes;
    std::vector<RenderData_> points;
    std::vector<Material_> materials;
    std::unordered_map<int, uint32_t> objMaterialOffsets;
    uint32_t defaultMatIdx;
    uint32_t gasMaterialOffset = 0;

    std::vector<GPUGasField> gasFields;
    std::vector<float> gasCells;
    
    void clear() {
        nodes.clear();
        points.clear();
        materials.clear();
        objMaterialOffsets.clear();
        gasMaterialOffset = 0;
        gasFields.clear();
        gasCells.clear();
    }
};

static PointType sampleGGX(const PointType& n, float roughness, uint32_t& state) {
    float alpha = std::max(EPSILON, roughness * roughness);
    float r1 = float(rand_r(&state)) / float(RAND_MAX);
    float r2 = float(rand_r(&state)) / float(RAND_MAX);
    
    float phi = 2.0f * M_PI * r1;
    float denom = 1.0f + (alpha * alpha - 1.0f) * r2;
    denom = std::max(denom, EPSILON);
    float cosTheta = std::sqrt(std::max(0.0f, (1.0f - r2) / denom));
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    
    PointType h;
    h[0] = sinTheta * std::cos(phi);
    h[1] = sinTheta * std::sin(phi);
    h[2] = cosTheta;
    
    PointType up = std::abs(n.z()) < 0.999f ? PointType(0,0,1) : PointType(1,0,0);
    PointType tangent = up.cross(n).normalized();
    PointType bitangent = n.cross(tangent);
    
    return (tangent * h[0] + bitangent * h[1] + n * h[2]).normalized();
}

static PointType sampleCosineHemisphere(const PointType& n, uint32_t& state) {
    float r1 = float(rand_r(&state)) / float(RAND_MAX);
    float r2 = float(rand_r(&state)) / float(RAND_MAX);
    float phi = 2.0f * M_PI * r1;
    float r = std::sqrt(r2);
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.0f, 1.0f - x*x - y*y));
    
    PointType up = std::abs(n.z()) < 0.999f ? PointType(0,0,1) : PointType(1,0,0);
    PointType tangent = up.cross(n).normalized();
    PointType bitangent = n.cross(tangent);
    
    return (tangent * x + bitangent * y + n * z).normalized();
}

static inline float nextFloat(uint32_t& state) {
    if (state == 0) state = 123456789;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state & 0xFFFFFF) / 16777216.0f;
}

#ifdef VULKAN_SUPPORT
static PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;

struct alignas(16) GPURenderNode {
    Eigen::Vector3f boundsMin;
    float padding1;
    Eigen::Vector3f boundsMax;
    float padding2;
    Eigen::Vector3f center;
    float nodeSize;
    uint32_t isLeaf;
    uint32_t isLoaded;
    uint32_t childMask;
    uint32_t firstPoint;
    uint32_t pointCount;
    int32_t  lodPoint;
    uint32_t firstChild;
    uint32_t padding3;
};

struct alignas(16) GPUMaterial {
    uint32_t emittance;
    uint32_t materialProps;
    uint32_t absorption;
    uint32_t albedo;
};

struct alignas(16) GPUFastRenderData {
    Eigen::Vector3f position;
    float size;
    uint32_t color;
    uint32_t materialIdx;
    int objectId;
    uint32_t isGas;
};

struct alignas(16) GPUPBRRenderData {
    Eigen::Vector3f position;
    float size;
    uint32_t color;
    uint32_t materialIdx;
    int objectId;
    uint32_t isGas;
};

struct alignas(16) GPUCameraData {
    Eigen::Vector3f origin;
    float lodMinDist;
    Eigen::Vector3f dir;
    float invLodf;
    Eigen::Vector3f up;
    float minVisibility;
    Eigen::Vector3f right;
    float maxDist;
    Eigen::Vector3f skylight;
    float tanfovx;
    Eigen::Vector3f bgColor;
    float tanfovy;
    int width;
    int height;
    int maxBounces;
    int useLod;
    float invFogRange;
    uint32_t frameCount;
    int skyWidth;
    int skyHeight;
    int currentSampleOffset;
    int dispatchSamples;
    int globalIllumination;
    uint32_t nodeCount;
    uint32_t pointCount;
    int tileOffsetX;
    int tileOffsetY;
    int emissiveCount;
    int targetSamples;
    int sellWidth;
    int sellSecondary;
    uint32_t gasFieldCount;
    uint32_t blueFrameSeed;
    uint32_t gasPad1;
    uint32_t gasPad2;
};

struct alignas(16) GPUParticle {
    Eigen::Vector4f pos_mass;
    Eigen::Vector4f vel_density;
    Eigen::Vector4f force_press;
    Eigen::Vector4i type_pad;
};

struct WavefrontRay {
    Eigen::Vector3f origin;
    uint32_t pixelIndex;
    Eigen::Vector3f dir;
    uint32_t rng_state;
    Eigen::Vector3f throughput;
    uint32_t bounce;
    int active;
    float primaryDepth;
    int primaryObjId;
    float padding1, padding2, padding3;
};

struct WavefrontHit {
    Eigen::Vector3f normal;
    float t;
    int hitIndex;
    int hitFound;
    Eigen::Vector3f hitPoint;
    float padding1;
};

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence renderFence = VK_NULL_HANDLE;
    
    VkShaderModule fastShader = VK_NULL_HANDLE;
    VkShaderModule pbrShader = VK_NULL_HANDLE;
    VkShaderModule smoothShader = VK_NULL_HANDLE;
    VkShaderModule blendShader = VK_NULL_HANDLE;
    VkPipelineLayout fastPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout pbrPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout smoothPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout blendPipelineLayout = VK_NULL_HANDLE;
    VkPipeline fastPipeline = VK_NULL_HANDLE;
    VkPipeline pbrPipeline = VK_NULL_HANDLE;
    VkPipeline smoothPipeline = VK_NULL_HANDLE;
    VkPipeline blendPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout fastDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout smoothDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout blendDescLayout = VK_NULL_HANDLE;
    VkBuffer fastGBuffer = VK_NULL_HANDLE;
    VkDeviceMemory fastGBufferMem = VK_NULL_HANDLE;
    size_t currentFastGCap = 0;

    VkDescriptorSetLayout wfDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet       wfDescSet    = VK_NULL_HANDLE;
    VkPipelineLayout      wfPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule wfInitShader = VK_NULL_HANDLE, wfArgsShader = VK_NULL_HANDLE,
                   wfExtendShader = VK_NULL_HANDLE, wfShadeShader = VK_NULL_HANDLE,
                   wfShadowShader = VK_NULL_HANDLE, wfFinalizeShader = VK_NULL_HANDLE;
    VkPipeline wfInitPipe = VK_NULL_HANDLE, wfArgsPipe = VK_NULL_HANDLE,
               wfExtendPipe = VK_NULL_HANDLE, wfShadePipe = VK_NULL_HANDLE,
               wfShadowPipe = VK_NULL_HANDLE, wfFinalizePipe = VK_NULL_HANDLE;
    VkBuffer wfPathBuf = VK_NULL_HANDLE, wfPathHitBuf = VK_NULL_HANDLE,
             wfExtendABuf = VK_NULL_HANDLE,
             wfExtendBBuf = VK_NULL_HANDLE, wfShadeBuf = VK_NULL_HANDLE,
             wfShadowBuf = VK_NULL_HANDLE, wfCounterBuf = VK_NULL_HANDLE,
             wfArgsBuf = VK_NULL_HANDLE;
    VkDeviceMemory wfPathMem = VK_NULL_HANDLE, wfPathHitMem = VK_NULL_HANDLE,
                   wfExtendAMem = VK_NULL_HANDLE,
                   wfExtendBMem = VK_NULL_HANDLE, wfShadeMem = VK_NULL_HANDLE,
                   wfShadowMem = VK_NULL_HANDLE, wfCounterMem = VK_NULL_HANDLE,
                   wfArgsMem = VK_NULL_HANDLE;
    size_t wfPathCap = 0;
    VkBuffer       blueTileBuf = VK_NULL_HANDLE;
    VkDeviceMemory blueTileMem = VK_NULL_HANDLE;
    size_t         blueTileCap = 0;
    bool           bluePoolBuilt = false;
    bluetile::Pool      bluePool;
    bluetile::Assembler blueAsm{bluePool};
    std::vector<float>  blueFrameTiles;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet fastDescSet = VK_NULL_HANDLE;
    VkDescriptorSet pbrDescSet = VK_NULL_HANDLE;
    VkDescriptorSet smoothDescSet = VK_NULL_HANDLE;
    VkDescriptorSet blendDescSet = VK_NULL_HANDLE;
    
    VkShaderModule sphDensityShader = VK_NULL_HANDLE;
    VkShaderModule sphForceShader = VK_NULL_HANDLE;
    VkShaderModule sphIntegrateShader = VK_NULL_HANDLE;
    VkPipelineLayout sphPipelineLayout = VK_NULL_HANDLE;
    VkPipeline sphDensityPipeline = VK_NULL_HANDLE;
    VkPipeline sphForcePipeline = VK_NULL_HANDLE;
    VkPipeline sphIntegratePipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout sphDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet sphDescSet = VK_NULL_HANDLE;
    VkBuffer particleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory particleMem = VK_NULL_HANDLE;
    size_t currentParticleCap = 0;
    VkBuffer physicsAabbBuffer = VK_NULL_HANDLE;
    VkDeviceMemory physicsAabbMem = VK_NULL_HANDLE;
    size_t currentPhysicsAabbCap = 0;
    
    VkBuffer physicsAsInstanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory physicsAsInstanceMem = VK_NULL_HANDLE;
    VkAccelerationStructureKHR physicsBlas = VK_NULL_HANDLE;
    VkBuffer physicsBlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory physicsBlasMem = VK_NULL_HANDLE;
    VkAccelerationStructureKHR physicsTlas = VK_NULL_HANDLE;
    VkBuffer physicsTlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory physicsTlasMem = VK_NULL_HANDLE;

    VkBuffer nodeBuffer = VK_NULL_HANDLE;
    VkBuffer outBuffer = VK_NULL_HANDLE;
    VkBuffer uboBuffer = VK_NULL_HANDLE;
    VkBuffer fastPointBuffer = VK_NULL_HANDLE;
    VkBuffer pbrPointBuffer = VK_NULL_HANDLE;
    VkBuffer skyboxBuffer = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;
    VkBuffer finalOutBuffer = VK_NULL_HANDLE;
    VkBuffer lowResOutBuffer = VK_NULL_HANDLE;
    VkBuffer adaptiveBuffer = VK_NULL_HANDLE;
    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceMemory nodeMem = VK_NULL_HANDLE;
    VkBuffer sellmeierBuffer = VK_NULL_HANDLE;
    VkBuffer gasFieldBuffer = VK_NULL_HANDLE;     // GPUGasField headers
    VkDeviceMemory gasFieldMem = VK_NULL_HANDLE;
    size_t currentGasFieldCap = 0;
    VkBuffer gasCellBuffer = VK_NULL_HANDLE;       // flattened cell densities
    VkDeviceMemory gasCellMem = VK_NULL_HANDLE;
    size_t currentGasCellCap = 0;
    uint32_t gasFieldCount = 0;
    VkDeviceMemory outMem = VK_NULL_HANDLE;
    VkDeviceMemory uboMem = VK_NULL_HANDLE;
    VkDeviceMemory fastPointMem = VK_NULL_HANDLE;
    VkDeviceMemory pbrPointMem = VK_NULL_HANDLE;
    VkDeviceMemory skyboxMem = VK_NULL_HANDLE;
    VkDeviceMemory lightMem = VK_NULL_HANDLE;
    VkDeviceMemory finalOutMem = VK_NULL_HANDLE;
    VkDeviceMemory lowResOutMem = VK_NULL_HANDLE;
    VkDeviceMemory adaptiveMem = VK_NULL_HANDLE;
    VkDeviceMemory materialMem = VK_NULL_HANDLE;
    VkDeviceMemory sellmeierMem = VK_NULL_HANDLE;
    
    size_t currentNodesCap = 0;
    size_t currentOutCap = 0;
    size_t currentFastPointsCap = 0;
    size_t currentPBRPointsCap = 0;
    size_t currentSkyboxCap = 0;
    size_t currentLightCap = 0;
    size_t currentFinalOutCap = 0;
    size_t currentLowResOutCap = 0;
    size_t currentAdaptiveCap = 0;
    size_t currentMaterialCap = 0;
    size_t currentSellmeierCap = 0;
    uint32_t sellmeierWidth = 0;   // wavelength samples per row
    uint32_t sellmeierRows = 0;    // total rows (materials * secondary)

    bool initialized = false;
    bool hasHardwareRT = false;
    
    VkBuffer aabbBuffer = VK_NULL_HANDLE;
    VkDeviceMemory aabbMem = VK_NULL_HANDLE;
    VkBuffer asInstanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory asInstanceMem = VK_NULL_HANDLE;
    
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkBuffer blasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory blasMem = VK_NULL_HANDLE;
    
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkBuffer tlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tlasMem = VK_NULL_HANDLE;

    size_t currentAabbCap = 0;
    VkBuffer asScratchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory asScratchMem = VK_NULL_HANDLE;
    size_t currentScratchCap = 0;
    uint32_t lastBlasPrimCount = 0;       // primitive count of the live BLAS topology
    uint32_t framesSinceFullBuild = 0;    // refit counter
    uint32_t refitInterval = 16;          // force a clean rebuild every N frames
    bool blasTopologyValid = false;       // is there a refit-able BLAS in place?

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return 0;
    }

    uint32_t findMemoryTypePreferred(uint32_t typeFilter, VkMemoryPropertyFlags required,
                                     VkMemoryPropertyFlags preferred, bool& gotPreferred) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        // First pass: required + preferred.
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = memProperties.memoryTypes[i].propertyFlags;
            if ((typeFilter & (1 << i)) && ((f & (required | preferred)) == (required | preferred))) {
                gotPreferred = true;
                return i;
            }
        }
        // Second pass: required only.
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = memProperties.memoryTypes[i].propertyFlags;
            if ((typeFilter & (1 << i)) && ((f & required) == required)) {
                gotPreferred = false;
                return i;
            }
        }
        gotPreferred = false;
        return 0;
    }
    bool outMemCoherent = true;

    void createReadbackBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool& coherentOut) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        bool gotCached = false;
        uint32_t typeIdx = findMemoryTypePreferred(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            gotCached);

        // Determine coherence of the chosen type so we know whether to invalidate.
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        coherentOut = (memProps.memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = typeIdx;
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }


    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                        VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkShaderModule createShaderModule(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "FAILED TO LOAD " << path << "!\n";
            return VK_NULL_HANDLE;
        }
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = buffer.size();
        moduleInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
        VkShaderModule shaderModule;
        vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule);
        return shaderModule;
    }

    void init() {
        if (initialized) return;
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.apiVersion = VK_API_VERSION_1_2;

        uint32_t extCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &appInfo;
        vkCreateInstance(&createInfo, nullptr, &instance);

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[0]; //need to set a flag for this at some point.

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availableExts.data());

        bool supportsRaytracingExtensions = false;
        bool supportRQ = false;
        bool supportAS = false;
        bool supportDHO = false;
        for (const auto& ext : availableExts) {
            if (strcmp(ext.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0) supportRQ = true;
            if (strcmp(ext.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) supportAS = true;
            if (strcmp(ext.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0) supportDHO = true;
        }
        supportsRaytracingExtensions = supportRQ && supportAS && supportDHO;

        VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rqFeatures};
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &asFeatures};
        VkPhysicalDeviceFeatures2 deviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features12};

        hasHardwareRT = false;
        if (supportsRaytracingExtensions) {
            vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);
            if (asFeatures.accelerationStructure && rqFeatures.rayQuery && features12.bufferDeviceAddress) {
                hasHardwareRT = true;
                std::cout << "Hardware Ray Tracing is supported and will be enabled." << std::endl;
            }
        }
        
        if (!hasHardwareRT) {
            std::cout << "Hardware Ray Tracing not supported. Falling back to software." << std::endl;
            memset(&rqFeatures, 0, sizeof(rqFeatures));
            rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            memset(&asFeatures, 0, sizeof(asFeatures));
            asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            asFeatures.pNext = &rqFeatures;
        }
        std::vector<const char*> deviceExtensions;

        deviceFeatures2.pNext = &features12;
        if (hasHardwareRT) {
            deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            features12.pNext = &asFeatures;
            asFeatures.pNext = &rqFeatures;
            asFeatures.accelerationStructure = VK_TRUE;
            rqFeatures.rayQuery = VK_TRUE;
            features12.bufferDeviceAddress = VK_TRUE;
        }

        VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreateInfo.pNext = &deviceFeatures2;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        if (hasHardwareRT) {
            pfn_vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
            pfn_vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
            pfn_vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
            pfn_vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
            pfn_vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
        }

        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkDescriptorPoolSize poolSizes[] = { 
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 60},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 6}
        };
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.poolSizeCount = 3;
        poolCreateInfo.pPoolSizes = poolSizes;
        poolCreateInfo.maxSets = 8;
        vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);

        VkDescriptorSetLayoutBinding bindings[9] = {};
        for(int i=0; i<6; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        
        bindings[6].binding = 6;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[7].binding = 7;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[8].binding = 8;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 9;
        layoutInfo.pBindings = bindings;

        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &fastDescLayout);
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &pbrDescLayout);

        VkDescriptorSetLayoutBinding smBindings[2] = {};
        for(int i=0; i<2; i++) {
            smBindings[i].binding = i;
            smBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            smBindings[i].descriptorCount = 1;
            smBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo smLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, smBindings};
        vkCreateDescriptorSetLayout(device, &smLayoutInfo, nullptr, &smoothDescLayout);

        VkDescriptorSetLayoutBinding blBindings[3] = {};
        for(int i=0; i<3; i++) {
            blBindings[i].binding = i;
            blBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            blBindings[i].descriptorCount = 1;
            blBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo blLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, blBindings};
        vkCreateDescriptorSetLayout(device, &blLayoutInfo, nullptr, &blendDescLayout);

        VkDescriptorSetAllocateInfo allocSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &fastDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &fastDescSet);
        allocSetInfo.pSetLayouts = &pbrDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &pbrDescSet);
        allocSetInfo.pSetLayouts = &smoothDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &smoothDescSet);
        allocSetInfo.pSetLayouts = &blendDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &blendDescSet);

        if (hasHardwareRT) {
            std::cout << "using _hw versions" << std::endl;
            fastShader = createShaderModule("./bin/fast_raytrace_hw.spv");
            pbrShader = createShaderModule("./bin/pbr_raytrace_hw.spv");
        } else {
            std::cout << "using software versions" << std::endl;
            fastShader = createShaderModule("./bin/fast_raytrace.spv");
            pbrShader = createShaderModule("./bin/pbr_raytrace.spv");
        }
        smoothShader = createShaderModule("./bin/smooth.spv");
        blendShader = createShaderModule("./bin/blend.spv");
        uint32_t sphBindingCount = hasHardwareRT ? 3 : 1;
        VkDescriptorSetLayoutBinding sphBindings[3] = {};
        sphBindings[0].binding = 0;
        sphBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sphBindings[0].descriptorCount = 1;
        sphBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        if (hasHardwareRT) {
            sphBindings[1].binding = 1;
            sphBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            sphBindings[1].descriptorCount = 1;
            sphBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            sphBindings[2].binding = 2;
            sphBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sphBindings[2].descriptorCount = 1;
            sphBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo sphLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, sphBindingCount, sphBindings};
        vkCreateDescriptorSetLayout(device, &sphLayoutInfo, nullptr, &sphDescLayout);
        
        VkDescriptorSetAllocateInfo sphAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sphAllocInfo.descriptorPool = descriptorPool;
        sphAllocInfo.descriptorSetCount = 1;
        sphAllocInfo.pSetLayouts = &sphDescLayout;
        vkAllocateDescriptorSets(device, &sphAllocInfo, &sphDescSet);

        sphDensityShader = createShaderModule("./bin/sph_density.spv");
        sphForceShader = createShaderModule("./bin/sph_force.spv");
        if (hasHardwareRT) {
            sphIntegrateShader = createShaderModule("./bin/sph_integrate.spv");
        }
        
        VkPushConstantRange sphPushInfo{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SPHForcePC)}; // Use the larger struct size
        VkPipelineLayoutCreateInfo sphPipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        sphPipelineLayoutInfo.setLayoutCount = 1;
        sphPipelineLayoutInfo.pSetLayouts = &sphDescLayout;
        sphPipelineLayoutInfo.pushConstantRangeCount = 1;
        sphPipelineLayoutInfo.pPushConstantRanges = &sphPushInfo;
        vkCreatePipelineLayout(device, &sphPipelineLayoutInfo, nullptr, &sphPipelineLayout);

        VkComputePipelineCreateInfo sphComputeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        sphComputeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sphComputeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        sphComputeInfo.stage.pName = "main";
        sphComputeInfo.layout = sphPipelineLayout;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &fastDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &fastPipelineLayout);
        pipelineLayoutInfo.pSetLayouts = &pbrDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pbrPipelineLayout);

        VkPushConstantRange smPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 3};
        pipelineLayoutInfo.pSetLayouts = &smoothDescLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &smPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &smoothPipelineLayout);

        VkPushConstantRange blPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 5 + sizeof(float)};
        pipelineLayoutInfo.pSetLayouts = &blendDescLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &blPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &blendPipelineLayout);

        VkComputePipelineCreateInfo computePipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computePipelineInfo.stage.pName = "main";

        if (fastShader) {
            computePipelineInfo.layout = fastPipelineLayout;
            computePipelineInfo.stage.module = fastShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &fastPipeline);
        }
        if (pbrShader) {
            computePipelineInfo.layout = pbrPipelineLayout;
            computePipelineInfo.stage.module = pbrShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &pbrPipeline);
        }
        if (smoothShader) {
            computePipelineInfo.layout = smoothPipelineLayout;
            computePipelineInfo.stage.module = smoothShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &smoothPipeline);
        }
        if (blendShader) {
            computePipelineInfo.layout = blendPipelineLayout;
            computePipelineInfo.stage.module = blendShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &blendPipeline);
        }
        
        if (sphDensityShader) {
            sphComputeInfo.stage.module = sphDensityShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &sphComputeInfo, nullptr, &sphDensityPipeline);
        }
        if (sphForceShader) {
            sphComputeInfo.stage.module = sphForceShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &sphComputeInfo, nullptr, &sphForcePipeline);
        }
        if (hasHardwareRT && sphIntegrateShader) {
            sphComputeInfo.stage.module = sphIntegrateShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &sphComputeInfo, nullptr, &sphIntegratePipeline);
        }

        if (hasHardwareRT) {
            initWavefront();
        }

        initialized = true;
    }
    
    void executeSingleTimeCommands(std::function<void(VkCommandBuffer)> action) {
        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &allocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        action(cmd);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        executeSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, srcBuffer, dstBuffer, 1, &copyRegion);
        });
    }

    void updateDeviceLocalBuffer(VkBuffer& buffer, VkDeviceMemory& memory, size_t& currentCap, 
                                 const void* data, size_t dataSize, size_t allocSize, VkBufferUsageFlags usage) {
        if (allocSize > currentCap) {
            if (buffer) {
                vkDestroyBuffer(device, buffer, nullptr);
                vkFreeMemory(device, memory, nullptr);
            }
            createBuffer(allocSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
            currentCap = allocSize;
        }

        if (data && dataSize > 0) {
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMem;
            createBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                         stagingBuffer, stagingMem);

            void* mappedData;
            vkMapMemory(device, stagingMem, 0, dataSize, 0, &mappedData);
            memcpy(mappedData, data, dataSize);
            vkUnmapMemory(device, stagingMem);

            copyBuffer(stagingBuffer, buffer, dataSize);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        }
    }

    void createBufferWithAddress(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                 VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, buffer, &memReqs);

        VkMemoryAllocateFlagsInfo allocFlagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.pNext = &allocFlagsInfo;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
        
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }

    void buildPhysicsAccelerationStructures(const std::vector<VkAabbPositionsKHR>& aabbs) {
        if (!hasHardwareRT || aabbs.empty()) return;

        size_t aabbSize = aabbs.size() * sizeof(VkAabbPositionsKHR);
        if (aabbSize > currentPhysicsAabbCap) {
            if (physicsAabbBuffer) {
                vkDestroyBuffer(device, physicsAabbBuffer, nullptr);
                vkFreeMemory(device, physicsAabbMem, nullptr);
            }
            createBufferWithAddress(aabbSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicsAabbBuffer, physicsAabbMem);
            currentPhysicsAabbCap = aabbSize;
            
            // Re-bind AABB storage buffer to desc set
            VkDescriptorBufferInfo bInfo{physicsAabbBuffer, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = sphDescSet;
            write.dstBinding = 2;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void* data;
        vkMapMemory(device, physicsAabbMem, 0, aabbSize, 0, &data);
        memcpy(data, aabbs.data(), aabbSize);
        vkUnmapMemory(device, physicsAabbMem);

        VkAccelerationStructureGeometryKHR blasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        blasGeom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        blasGeom.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        blasGeom.geometry.aabbs.data.deviceAddress = getBufferDeviceAddress(physicsAabbBuffer);
        blasGeom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

        VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasBuildInfo.geometryCount = 1;
        blasBuildInfo.pGeometries = &blasGeom;

        uint32_t numPrimitives = static_cast<uint32_t>(aabbs.size());
        VkAccelerationStructureBuildSizesInfoKHR blasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo, &numPrimitives, &blasSizeInfo);

        if (physicsBlas) {
            pfn_vkDestroyAccelerationStructureKHR(device, physicsBlas, nullptr);
            vkDestroyBuffer(device, physicsBlasBuffer, nullptr);
            vkFreeMemory(device, physicsBlasMem, nullptr);
        }
        createBufferWithAddress(blasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, 
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicsBlasBuffer, physicsBlasMem);
        
        VkAccelerationStructureCreateInfoKHR blasCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        blasCreateInfo.buffer = physicsBlasBuffer;
        blasCreateInfo.size = blasSizeInfo.accelerationStructureSize;
        blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        pfn_vkCreateAccelerationStructureKHR(device, &blasCreateInfo, nullptr, &physicsBlas);

        VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        blasAddrInfo.accelerationStructure = physicsBlas;
        VkDeviceAddress blasAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &blasAddrInfo);

        VkAccelerationStructureInstanceKHR tlasInstance{};
        tlasInstance.transform = { 1.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f, 0.0f };
        tlasInstance.instanceCustomIndex = 0;
        tlasInstance.mask = 0xFF;
        tlasInstance.instanceShaderBindingTableRecordOffset = 0;
        tlasInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tlasInstance.accelerationStructureReference = blasAddress;

        if (physicsAsInstanceBuffer) {
            vkDestroyBuffer(device, physicsAsInstanceBuffer, nullptr);
            vkFreeMemory(device, physicsAsInstanceMem, nullptr);
        }
        createBufferWithAddress(sizeof(VkAccelerationStructureInstanceKHR), 
                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicsAsInstanceBuffer, physicsAsInstanceMem);
        
        vkMapMemory(device, physicsAsInstanceMem, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &data);
        memcpy(data, &tlasInstance, sizeof(VkAccelerationStructureInstanceKHR));
        vkUnmapMemory(device, physicsAsInstanceMem);

        VkAccelerationStructureGeometryKHR tlasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
        tlasGeom.geometry.instances.data.deviceAddress = getBufferDeviceAddress(physicsAsInstanceBuffer);

        VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuildInfo.geometryCount = 1;
        tlasBuildInfo.pGeometries = &tlasGeom;

        uint32_t numInstances = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo, &numInstances, &tlasSizeInfo);

        if (physicsTlas) {
            pfn_vkDestroyAccelerationStructureKHR(device, physicsTlas, nullptr);
            vkDestroyBuffer(device, physicsTlasBuffer, nullptr);
            vkFreeMemory(device, physicsTlasMem, nullptr);
        }
        createBufferWithAddress(tlasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, 
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicsTlasBuffer, physicsTlasMem);

        VkAccelerationStructureCreateInfoKHR tlasCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        tlasCreateInfo.buffer = physicsTlasBuffer;
        tlasCreateInfo.size = tlasSizeInfo.accelerationStructureSize;
        tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        pfn_vkCreateAccelerationStructureKHR(device, &tlasCreateInfo, nullptr, &physicsTlas);

        VkBuffer scratchBuffer;
        VkDeviceMemory scratchMem;
        VkDeviceSize scratchSize = std::max(blasSizeInfo.buildScratchSize, tlasSizeInfo.buildScratchSize);
        createBufferWithAddress(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMem);

        executeSingleTimeCommands([&](VkCommandBuffer cmd) {
            blasBuildInfo.dstAccelerationStructure = physicsBlas;
            blasBuildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);
            VkAccelerationStructureBuildRangeInfoKHR blasOffset{};
            blasOffset.primitiveCount = numPrimitives;
            VkAccelerationStructureBuildRangeInfoKHR* pBlasOffset = &blasOffset;
            pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blasBuildInfo, &pBlasOffset);

            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);

            tlasBuildInfo.dstAccelerationStructure = physicsTlas;
            tlasBuildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);
            VkAccelerationStructureBuildRangeInfoKHR tlasOffset{};
            tlasOffset.primitiveCount = numInstances;
            VkAccelerationStructureBuildRangeInfoKHR* pTlasOffset = &tlasOffset;
            pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuildInfo, &pTlasOffset);
        });

        vkDestroyBuffer(device, scratchBuffer, nullptr);
        vkFreeMemory(device, scratchMem, nullptr);

        // Bind the physics TLAS to descriptor set binding 1
        VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures = &physicsTlas;

        VkWriteDescriptorSet asWrite{};
        asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        asWrite.pNext = &descASInfo;
        asWrite.dstSet = sphDescSet;
        asWrite.dstBinding = 1;
        asWrite.descriptorCount = 1;
        asWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        
        vkUpdateDescriptorSets(device, 1, &asWrite, 0, nullptr);
    }

    template<typename RenderDataType>
    void buildHardwareAccelerationStructures(const std::vector<RenderDataType>& points) {
        if (!hasHardwareRT || points.empty()) return;

        const uint32_t numPrimitives = static_cast<uint32_t>(points.size());
        bool doFullBuild = (!blasTopologyValid)
                        || (numPrimitives != lastBlasPrimCount)
                        || (framesSinceFullBuild >= refitInterval);
        std::vector<VkAabbPositionsKHR> aabbs(numPrimitives);
        for (uint32_t i = 0; i < numPrimitives; ++i) {
            float halfSize = points[i].size * 0.5f;
            aabbs[i].minX = points[i].position.x() - halfSize;
            aabbs[i].minY = points[i].position.y() - halfSize;
            aabbs[i].minZ = points[i].position.z() - halfSize;
            aabbs[i].maxX = points[i].position.x() + halfSize;
            aabbs[i].maxY = points[i].position.y() + halfSize;
            aabbs[i].maxZ = points[i].position.z() + halfSize;
        }

        size_t aabbSize = aabbs.size() * sizeof(VkAabbPositionsKHR);
        if (aabbSize > currentAabbCap) {
            if (aabbBuffer) {
                vkDestroyBuffer(device, aabbBuffer, nullptr);
                vkFreeMemory(device, aabbMem, nullptr);
            }
            createBufferWithAddress(aabbSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, aabbBuffer, aabbMem);
            currentAabbCap = aabbSize;
        }

        void* data;
        vkMapMemory(device, aabbMem, 0, aabbSize, 0, &data);
        memcpy(data, aabbs.data(), aabbSize);
        vkUnmapMemory(device, aabbMem);

        VkAccelerationStructureGeometryKHR blasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        blasGeom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        blasGeom.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        blasGeom.geometry.aabbs.data.deviceAddress = getBufferDeviceAddress(aabbBuffer);
        blasGeom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

        VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        blasBuildInfo.mode = doFullBuild ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        blasBuildInfo.geometryCount = 1;
        blasBuildInfo.pGeometries = &blasGeom;

        VkAccelerationStructureBuildSizesInfoKHR blasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo, &numPrimitives, &blasSizeInfo);


        if (doFullBuild) {
            if (blas) {
                pfn_vkDestroyAccelerationStructureKHR(device, blas, nullptr);
                vkDestroyBuffer(device, blasBuffer, nullptr);
                vkFreeMemory(device, blasMem, nullptr);
            }
            createBufferWithAddress(blasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blasBuffer, blasMem);

            VkAccelerationStructureCreateInfoKHR blasCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            blasCreateInfo.buffer = blasBuffer;
            blasCreateInfo.size = blasSizeInfo.accelerationStructureSize;
            blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            pfn_vkCreateAccelerationStructureKHR(device, &blasCreateInfo, nullptr, &blas);
        }
        blasBuildInfo.srcAccelerationStructure = doFullBuild ? VK_NULL_HANDLE : blas;
        VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        blasAddrInfo.accelerationStructure = blas;
        VkDeviceAddress blasAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &blasAddrInfo);

        VkAccelerationStructureInstanceKHR tlasInstance{};
        tlasInstance.transform = { 1.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f, 0.0f };
        tlasInstance.instanceCustomIndex = 0;
        tlasInstance.mask = 0xFF;
        tlasInstance.instanceShaderBindingTableRecordOffset = 0;
        tlasInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tlasInstance.accelerationStructureReference = blasAddress;

        if (asInstanceBuffer) {
            vkDestroyBuffer(device, asInstanceBuffer, nullptr);
            vkFreeMemory(device, asInstanceMem, nullptr);
        }
        createBufferWithAddress(sizeof(VkAccelerationStructureInstanceKHR), 
                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, asInstanceBuffer, asInstanceMem);
        
        vkMapMemory(device, asInstanceMem, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &data);
        memcpy(data, &tlasInstance, sizeof(VkAccelerationStructureInstanceKHR));
        vkUnmapMemory(device, asInstanceMem);

        VkAccelerationStructureGeometryKHR tlasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
        tlasGeom.geometry.instances.data.deviceAddress = getBufferDeviceAddress(asInstanceBuffer);

        VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        tlasBuildInfo.mode = doFullBuild ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        tlasBuildInfo.geometryCount = 1;
        tlasBuildInfo.pGeometries = &tlasGeom;

        uint32_t numInstances = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo, &numInstances, &tlasSizeInfo);

        if (doFullBuild) {
            if (tlas) {
                pfn_vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
                vkDestroyBuffer(device, tlasBuffer, nullptr);
                vkFreeMemory(device, tlasMem, nullptr);
            }
            createBufferWithAddress(tlasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlasBuffer, tlasMem);

            VkAccelerationStructureCreateInfoKHR tlasCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            tlasCreateInfo.buffer = tlasBuffer;
            tlasCreateInfo.size = tlasSizeInfo.accelerationStructureSize;
            tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            pfn_vkCreateAccelerationStructureKHR(device, &tlasCreateInfo, nullptr, &tlas);
        }
        tlasBuildInfo.srcAccelerationStructure = doFullBuild ? VK_NULL_HANDLE : tlas;

        VkDeviceSize scratchSize = std::max(blasSizeInfo.buildScratchSize, tlasSizeInfo.buildScratchSize);
        scratchSize = std::max(scratchSize, std::max(blasSizeInfo.updateScratchSize, tlasSizeInfo.updateScratchSize));
        if (scratchSize > currentScratchCap) {
            if (asScratchBuffer) {
                vkDestroyBuffer(device, asScratchBuffer, nullptr);
                vkFreeMemory(device, asScratchMem, nullptr);
            }
            createBufferWithAddress(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asScratchBuffer, asScratchMem);
            currentScratchCap = scratchSize;
        }
        VkBuffer scratchBuffer = asScratchBuffer;

        executeSingleTimeCommands([&](VkCommandBuffer cmd) {
            blasBuildInfo.dstAccelerationStructure = blas;
            blasBuildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);
            VkAccelerationStructureBuildRangeInfoKHR blasOffset{};
            blasOffset.primitiveCount = numPrimitives;
            VkAccelerationStructureBuildRangeInfoKHR* pBlasOffset = &blasOffset;
            
            pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blasBuildInfo, &pBlasOffset);

            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);

            tlasBuildInfo.dstAccelerationStructure = tlas;
            tlasBuildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);
            VkAccelerationStructureBuildRangeInfoKHR tlasOffset{};
            tlasOffset.primitiveCount = numInstances;
            VkAccelerationStructureBuildRangeInfoKHR* pTlasOffset = &tlasOffset;

            pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuildInfo, &pTlasOffset);
        });

        // Scratch buffer is persistent now — do not free it here.

        // Refit bookkeeping.
        if (doFullBuild) {
            lastBlasPrimCount = numPrimitives;
            blasTopologyValid = true;
            framesSinceFullBuild = 0;
        } else {
            framesSinceFullBuild++;
        }

        // The TLAS handle only changes on a full build (UPDATE refits in place),
        // so the descriptor binding only needs rewriting then. On refit frames the
        // shader already points at the same (now-updated) structure.
        if (doFullBuild) {
            VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            descASInfo.accelerationStructureCount = 1;
            descASInfo.pAccelerationStructures = &tlas;

            VkWriteDescriptorSet asWrites[2] = {};
            for (int i = 0; i < 2; i++) {
                asWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                asWrites[i].pNext = &descASInfo;
                asWrites[i].dstSet = (i == 0) ? fastDescSet : pbrDescSet;
                asWrites[i].dstBinding = 6;
                asWrites[i].descriptorCount = 1;
                asWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            }
            vkUpdateDescriptorSets(device, 2, asWrites, 0, nullptr);
        }
    }

    void updateLightBuffer(const std::vector<uint32_t>& lights) {
        size_t allocSize = std::max((size_t)256, lights.size() * sizeof(uint32_t));
        size_t dataSize = lights.size() * sizeof(uint32_t);
        updateDeviceLocalBuffer(lightBuffer, lightMem, currentLightCap, 
                                lights.empty() ? nullptr : lights.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    
    void updateMaterialBuffer(const std::vector<GPUMaterial>& materials) {
        size_t allocSize = std::max((size_t)256, materials.size() * sizeof(GPUMaterial));
        size_t dataSize = materials.size() * sizeof(GPUMaterial);
        updateDeviceLocalBuffer(materialBuffer, materialMem, currentMaterialCap, 
                                materials.empty() ? nullptr : materials.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    void updateSellmeierBuffer(const std::vector<float>& lut, uint32_t width, uint32_t rows) {
        sellmeierWidth = width;
        sellmeierRows = rows;
        size_t dataSize = lut.size() * sizeof(float);
        size_t allocSize = std::max((size_t)256, dataSize);
        updateDeviceLocalBuffer(sellmeierBuffer, sellmeierMem, currentSellmeierCap,
                                lut.empty() ? nullptr : lut.data(), dataSize, allocSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    // Upload Eulerian gas fields (headers) and their flattened cell densities
    // for volumetric raymarching in the wavefront shaders.
    void updateGasBuffers(const std::vector<GPUGasField>& fields, const std::vector<float>& cells) {
        gasFieldCount = static_cast<uint32_t>(fields.size());

        size_t fDataSize = fields.size() * sizeof(GPUGasField);
        size_t fAllocSize = std::max((size_t)256, fDataSize);
        updateDeviceLocalBuffer(gasFieldBuffer, gasFieldMem, currentGasFieldCap,
                                fields.empty() ? nullptr : fields.data(), fDataSize, fAllocSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        size_t cDataSize = cells.size() * sizeof(float);
        size_t cAllocSize = std::max((size_t)256, cDataSize);
        updateDeviceLocalBuffer(gasCellBuffer, gasCellMem, currentGasCellCap,
                                cells.empty() ? nullptr : cells.data(), cDataSize, cAllocSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    uint32_t getGasFieldCount() const { return gasFieldCount; }

    void updateBlueNoise(uint32_t frameSeed) {
        if (!bluePoolBuilt) {
            bluePool.build();
            bluePoolBuilt = true;
        }
        blueAsm.assemble(frameSeed, blueFrameTiles);
        size_t bytes = blueFrameTiles.size() * sizeof(float);
        updateDeviceLocalBuffer(blueTileBuf, blueTileMem, blueTileCap,
                                blueFrameTiles.data(), bytes, bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    void updateCommonBuffers(size_t outSize, GPUCameraData& camData) {
        size_t allocSize = (size_t)256;
        
        // Use device local memory for nodes (critical for tree traversal performance)
        updateDeviceLocalBuffer(nodeBuffer, nodeMem, currentNodesCap, 
                                nullptr, 0, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // outBuffer remains HOST_VISIBLE because it is mapped/read directly in grid3render.cpp
        if(outSize > currentOutCap) {
            if(outBuffer) {
                vkDestroyBuffer(device, outBuffer, nullptr);
                vkFreeMemory(device, outMem, nullptr);
            }
            createReadbackBuffer(outSize, outBuffer, outMem, outMemCoherent);
            currentOutCap = outSize;
        }

        size_t adaptiveSize = outSize; 
        if(adaptiveSize > currentAdaptiveCap) {
            if(adaptiveBuffer) {
                vkDestroyBuffer(device, adaptiveBuffer, nullptr);
                vkFreeMemory(device, adaptiveMem, nullptr);
            }
            createBuffer(adaptiveSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, adaptiveBuffer, adaptiveMem);
            currentAdaptiveCap = adaptiveSize;
        }

        size_t uboSize = sizeof(GPUCameraData);
        if(!uboBuffer) {
            createBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuffer, uboMem);
        }

        void* data;
        vkMapMemory(device, uboMem, 0, uboSize, 0, &data);
        memcpy(data, &camData, uboSize);
        vkUnmapMemory(device, uboMem);
    }

    void updateCameraData(const GPUCameraData& camData) {
        void* data;
        vkMapMemory(device, uboMem, 0, sizeof(GPUCameraData), 0, &data);
        memcpy(data, &camData, sizeof(GPUCameraData));
        vkUnmapMemory(device, uboMem);
    }

    void updateSkyboxBuffer(const std::vector<Eigen::Vector4f>& skyData) {
        size_t allocSize = std::max((size_t)256, skyData.size() * sizeof(Eigen::Vector4f));
        size_t dataSize = skyData.size() * sizeof(Eigen::Vector4f);
        updateDeviceLocalBuffer(skyboxBuffer, skyboxMem, currentSkyboxCap, 
                                skyData.empty() ? nullptr : skyData.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    void updateFastBuffers(const std::vector<GPUFastRenderData>& points) {
        size_t allocSize = std::max((size_t)256, points.size() * sizeof(GPUFastRenderData));
        size_t dataSize = points.size() * sizeof(GPUFastRenderData);
        
        updateDeviceLocalBuffer(fastPointBuffer, fastPointMem, currentFastPointsCap, 
                                points.empty() ? nullptr : points.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (hasHardwareRT) buildHardwareAccelerationStructures(points);

        VkDescriptorBufferInfo bInfos[8] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {fastPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE},
            {lightBuffer, 0, VK_WHOLE_SIZE},
            {adaptiveBuffer, 0, VK_WHOLE_SIZE},
            {materialBuffer, 0, VK_WHOLE_SIZE}
        };
        int updateCount = 8;
        VkWriteDescriptorSet writes[8] = {};
        for(int i=0; i<updateCount; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = fastDescSet;
            writes[i].dstBinding = (i >= 6) ? i + 1 : i; // 6 is handled by AS update
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i==3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, updateCount, writes, 0, nullptr);
    }

    void updatePBRBuffers(const std::vector<GPUPBRRenderData>& points) {
        size_t allocSize = std::max((size_t)256, points.size() * sizeof(GPUPBRRenderData));
        size_t dataSize = points.size() * sizeof(GPUPBRRenderData);
        
        updateDeviceLocalBuffer(pbrPointBuffer, pbrPointMem, currentPBRPointsCap, 
                                points.empty() ? nullptr : points.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (hasHardwareRT) buildHardwareAccelerationStructures(points);

        VkDescriptorBufferInfo bInfos[8] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {pbrPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE},
            {lightBuffer, 0, VK_WHOLE_SIZE},
            {adaptiveBuffer, 0, VK_WHOLE_SIZE},
            {materialBuffer, 0, VK_WHOLE_SIZE}
        };
        int updateCount = 8;
        VkWriteDescriptorSet writes[8] = {};
        for(int i=0; i<updateCount; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pbrDescSet;
            writes[i].dstBinding = (i >= 6) ? i + 1 : i; 
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i==3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, updateCount, writes, 0, nullptr);
    }

    void ensureLowResBuffer(size_t size) {
        if(size > currentLowResOutCap) {
            if(lowResOutBuffer) {
                vkDestroyBuffer(device, lowResOutBuffer, nullptr);
                vkFreeMemory(device, lowResOutMem, nullptr);
            }
            // Retained HOST_VISIBLE so it can be mapped later if necessary
            createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                         lowResOutBuffer, lowResOutMem);
            currentLowResOutCap = size;
        }
    }

    void retainFastGBuffer(size_t fastOutSize) {
        if (fastOutSize > currentFastGCap) {
            if (fastGBuffer) {
                vkDestroyBuffer(device, fastGBuffer, nullptr);
                vkFreeMemory(device, fastGBufferMem, nullptr);
            }
            createBuffer(fastOutSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fastGBuffer, fastGBufferMem);
            currentFastGCap = fastOutSize;
        }
        copyBuffer(outBuffer, fastGBuffer, fastOutSize);
    }

    void dispatchSmooth(int width, int height, int samples) {
        size_t finalSize = width * height * 3 * sizeof(float);
        if(finalSize > currentFinalOutCap) {
            if(finalOutBuffer) {
                vkDestroyBuffer(device, finalOutBuffer, nullptr);
                vkFreeMemory(device, finalOutMem, nullptr);
            }
            createBuffer(finalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, finalOutBuffer, finalOutMem);
            currentFinalOutCap = finalSize;
        }

        VkDescriptorBufferInfo bInfos[2] = { {outBuffer, 0, VK_WHOLE_SIZE}, {finalOutBuffer, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[2] = {};
        for(int i=0; i<2; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = smoothDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipelineLayout, 0, 1, &smoothDescSet, 0, nullptr);
        
        struct { int w, h, s; } pc = {width, height, samples};
        vkCmdPushConstants(commandBuffer, smoothPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vkCreateFence(device, &fenceInfo, nullptr, &fence);
        vkQueueSubmit(queue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device, fence, nullptr);
    }

    void dispatchBlend(int width, int height, int lowW, int lowH, float pbrScale, int samples) {
        size_t finalSize = width * height * 3 * sizeof(float);
        if(finalSize > currentFinalOutCap) {
            if(finalOutBuffer) { 
                vkDestroyBuffer(device, finalOutBuffer, nullptr); 
                vkFreeMemory(device, finalOutMem, nullptr); 
            }
            createBuffer(finalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, finalOutBuffer, finalOutMem);
            currentFinalOutCap = finalSize;
        }

        VkDescriptorBufferInfo bInfos[3] = { {outBuffer, 0, VK_WHOLE_SIZE}, {lowResOutBuffer, 0, VK_WHOLE_SIZE}, {finalOutBuffer, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[3] = {};
        for(int i=0; i<3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = blendDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, blendPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, blendPipelineLayout, 0, 1, &blendDescSet, 0, nullptr);
        
        struct { int w, h, lw, lh; float ps; int s; } pc = {width, height, lowW, lowH, pbrScale, samples};
        vkCmdPushConstants(commandBuffer, blendPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;  
        submitInfo.pCommandBuffers = &commandBuffer;

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; 
        VkFence fence; 
        vkCreateFence(device, &fenceInfo, nullptr, &fence);
        vkQueueSubmit(queue, 1, &submitInfo, fence); 
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX); 
        vkDestroyFence(device, fence, nullptr);
    }
    
    void dispatchPhysics(std::vector<GPUParticle>& particles, const SPHDensityPC& dpc, const SPHForcePC& fpc, const SPHIntegratePC& ipc) {
        if (!initialized || particles.empty() || !sphDensityPipeline || !sphForcePipeline) return;

        size_t size = particles.size() * sizeof(GPUParticle);
        if (size > currentParticleCap) {
            if (particleBuffer) {
                vkDestroyBuffer(device, particleBuffer, nullptr);
                vkFreeMemory(device, particleMem, nullptr);
            }
            createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                         particleBuffer, particleMem);
            currentParticleCap = size;
            
            VkDescriptorBufferInfo bInfo{particleBuffer, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = sphDescSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void* data;
        vkMapMemory(device, particleMem, 0, size, 0, &data);
        memcpy(data, particles.data(), size);
        vkUnmapMemory(device, particleMem);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sphPipelineLayout, 0, 1, &sphDescSet, 0, nullptr);

        uint32_t groupCount = (particles.size() + 255) / 256;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sphDensityPipeline);
        vkCmdPushConstants(commandBuffer, sphPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SPHDensityPC), &dpc);
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sphForcePipeline);
        vkCmdPushConstants(commandBuffer, sphPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SPHForcePC), &fpc);
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        if (hasHardwareRT && sphIntegratePipeline != VK_NULL_HANDLE) {
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
            
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sphIntegratePipeline);
            vkCmdPushConstants(commandBuffer, sphPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SPHIntegratePC), &ipc);
            vkCmdDispatch(commandBuffer, groupCount, 1, 1);
        }

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vkCreateFence(device, &fenceInfo, nullptr, &fence);
        vkQueueSubmit(queue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device, fence, nullptr);

        vkMapMemory(device, particleMem, 0, size, 0, &data);
        memcpy(particles.data(), data, size);
        vkUnmapMemory(device, particleMem);
    }

struct WFPushConstants {
    int parity;
    int stage;
    int sampleIndex;
    int pad;
};

static constexpr size_t WF_PATH_STRIDE   = 6 * 4 * sizeof(float); // hot record (was 9*vec4)
static constexpr size_t WF_PATHHIT_STRIDE= 1 * 4 * sizeof(float); // transient extend->shade hand-off
static constexpr size_t WF_SHADOW_STRIDE = 4 * 4 * sizeof(float);
static constexpr size_t WF_COUNTER_SIZE  = 4 * sizeof(uint32_t);   // hot atomic counts only
static constexpr size_t WF_ARGS_SIZE     = 12 * sizeof(uint32_t);  // 3x uvec4 dispatch args
static constexpr VkDeviceSize WF_OFF_EXTEND_ARGS = 0;
static constexpr VkDeviceSize WF_OFF_SHADE_ARGS  = 16;
static constexpr VkDeviceSize WF_OFF_SHADOW_ARGS = 32;

void initWavefront() {
    VkDescriptorSetLayoutBinding b[20] = {};
    for (int i = 0; i < 20; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 20;
    li.pBindings = b;
    vkCreateDescriptorSetLayout(device, &li, nullptr, &wfDescLayout);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &wfDescLayout;
    vkAllocateDescriptorSets(device, &ai, &wfDescSet);

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(WFPushConstants)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &wfDescLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(device, &pli, nullptr, &wfPipelineLayout);

    wfInitShader     = createShaderModule("./bin/wf_init.spv");
    wfArgsShader     = createShaderModule("./bin/wf_args.spv");
    wfExtendShader   = createShaderModule("./bin/wf_extend.spv");
    wfShadeShader    = createShaderModule("./bin/wf_shade.spv");
    wfShadowShader   = createShaderModule("./bin/wf_shadow.spv");
    wfFinalizeShader = createShaderModule("./bin/wf_finalize.spv");

    auto makePipe = [&](VkShaderModule m, VkPipeline& out) {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.pName = "main";
        ci.stage.module = m;
        ci.layout = wfPipelineLayout;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &out);
    };
    makePipe(wfInitShader,     wfInitPipe);
    makePipe(wfArgsShader,     wfArgsPipe);
    makePipe(wfExtendShader,   wfExtendPipe);
    makePipe(wfShadeShader,    wfShadePipe);
    makePipe(wfShadowShader,   wfShadowPipe);
    makePipe(wfFinalizeShader, wfFinalizePipe);
}

void ensureWavefrontBuffers(size_t maxPaths) {
    if (maxPaths <= wfPathCap && wfPathBuf) return;
    auto destroy = [&](VkBuffer& bf, VkDeviceMemory& mm) {
        if (bf) {
            vkDestroyBuffer(device, bf, nullptr);
            vkFreeMemory(device, mm, nullptr);
            bf = VK_NULL_HANDLE;
        }
    };
    destroy(wfPathBuf, wfPathMem);
    destroy(wfPathHitBuf, wfPathHitMem);
    destroy(wfExtendABuf, wfExtendAMem);
    destroy(wfExtendBBuf, wfExtendBMem);
    destroy(wfShadeBuf, wfShadeMem);
    destroy(wfShadowBuf, wfShadowMem);
    destroy(wfCounterBuf, wfCounterMem);
    destroy(wfArgsBuf, wfArgsMem);

    const VkBufferUsageFlags store = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(maxPaths * WF_PATH_STRIDE,    store, devLocal, wfPathBuf,    wfPathMem);
    createBuffer(maxPaths * WF_PATHHIT_STRIDE, store, devLocal, wfPathHitBuf, wfPathHitMem);
    createBuffer(maxPaths * sizeof(uint32_t), store, devLocal, wfExtendABuf, wfExtendAMem);
    createBuffer(maxPaths * sizeof(uint32_t), store, devLocal, wfExtendBBuf, wfExtendBMem);
    createBuffer(maxPaths * sizeof(uint32_t), store, devLocal, wfShadeBuf,   wfShadeMem);
    createBuffer(maxPaths * WF_SHADOW_STRIDE, store, devLocal, wfShadowBuf,  wfShadowMem);
    createBuffer(WF_COUNTER_SIZE, store, devLocal, wfCounterBuf, wfCounterMem);
    createBuffer(WF_ARGS_SIZE, store | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, devLocal, wfArgsBuf, wfArgsMem);
    wfPathCap = maxPaths;
}

void writeWavefrontDescriptors() {
    VkDescriptorBufferInfo bi[20] = {};
    bi[0]  = {uboBuffer,      0, VK_WHOLE_SIZE};
    bi[1]  = {pbrPointBuffer, 0, VK_WHOLE_SIZE};
    bi[2]  = {materialBuffer, 0, VK_WHOLE_SIZE};
    bi[3]  = {skyboxBuffer,   0, VK_WHOLE_SIZE};
    bi[4]  = {lightBuffer,    0, VK_WHOLE_SIZE};
    bi[6]  = {outBuffer,      0, VK_WHOLE_SIZE};
    bi[7]  = {adaptiveBuffer, 0, VK_WHOLE_SIZE};
    bi[8]  = {wfPathBuf,      0, VK_WHOLE_SIZE};
    bi[9]  = {wfExtendABuf,   0, VK_WHOLE_SIZE};
    bi[10] = {wfExtendBBuf,   0, VK_WHOLE_SIZE};
    bi[11] = {wfShadeBuf,     0, VK_WHOLE_SIZE};
    bi[12] = {wfShadowBuf,    0, VK_WHOLE_SIZE};
    bi[13] = {wfCounterBuf,   0, VK_WHOLE_SIZE};
    bi[14] = {wfPathHitBuf,   0, VK_WHOLE_SIZE};
    bi[15] = {sellmeierBuffer ? sellmeierBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[16] = {gasFieldBuffer ? gasFieldBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[17] = {gasCellBuffer  ? gasCellBuffer  : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[18] = {blueTileBuf    ? blueTileBuf    : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[19] = {wfArgsBuf,      0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet w[20] = {};
    int n = 0;
    for (int i = 0; i < 20; ++i) {
        if (i == 5) continue;
        w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[n].dstSet = wfDescSet;
        w[n].dstBinding = i;
        w[n].descriptorCount = 1;
        w[n].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[n].pBufferInfo = &bi[i];
        ++n;
    }
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;
    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[n].pNext = &asInfo;
    w[n].dstSet = wfDescSet;
    w[n].dstBinding = 5;
    w[n].descriptorCount = 1;
    w[n].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    ++n;

    vkUpdateDescriptorSets(device, n, w, 0, nullptr);
}

void wfBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void wfBind(VkCommandBuffer cmd, VkPipeline pipe) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
}
void wfPush(VkCommandBuffer cmd, int parity, int stage, int sampleIndex) {
    WFPushConstants pc{parity, stage, sampleIndex, 0};
    vkCmdPushConstants(cmd, wfPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
}

void dispatchWavefront(int tileW, int tileH, int maxBounces, int samplesPerPixel) {
    size_t maxPaths = size_t(tileW) * size_t(tileH);
    if (maxPaths == 0) return;
    ensureWavefrontBuffers(maxPaths);
    writeWavefrontDescriptors();

    const uint32_t WG = 64;
    uint32_t pathGroups = uint32_t((maxPaths + WG - 1) / WG);
    int maxIters = maxBounces + 12;

    const int samplesPerSubmit = 4;

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fi, nullptr, &fence);

    for (int s0 = 0; s0 < samplesPerPixel; s0 += samplesPerSubmit) {
        int s1 = std::min(s0 + samplesPerSubmit, samplesPerPixel);

        ScopedFunctionTimer* _rec = new ScopedFunctionTimer("wf.recordCmd");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &bi);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                wfPipelineLayout, 0, 1, &wfDescSet, 0, nullptr);

        for (int s = s0; s < s1; ++s) {
            wfBind(commandBuffer, wfArgsPipe);
            wfPush(commandBuffer, 0, 4, s);
            vkCmdDispatch(commandBuffer, 1, 1, 1);
            wfBarrier(commandBuffer);
            wfBind(commandBuffer, wfInitPipe);
            wfPush(commandBuffer, 0, 0, s);
            vkCmdDispatch(commandBuffer, pathGroups, 1, 1);
            wfBarrier(commandBuffer);

            int parity = 0;
            for (int it = 0; it < maxIters; ++it) {
                wfBind(commandBuffer, wfArgsPipe);
                wfPush(commandBuffer, parity, 0, s);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfExtendPipe);
                wfPush(commandBuffer, parity, 0, s);
                vkCmdDispatchIndirect(commandBuffer, wfArgsBuf, WF_OFF_EXTEND_ARGS);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfArgsPipe);
                wfPush(commandBuffer, parity, 1, s);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfShadePipe);
                wfPush(commandBuffer, parity, 0, s);
                vkCmdDispatchIndirect(commandBuffer, wfArgsBuf, WF_OFF_SHADE_ARGS);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfArgsPipe);
                wfPush(commandBuffer, parity, 2, s);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfShadowPipe);
                wfPush(commandBuffer, parity, 0, s);
                vkCmdDispatchIndirect(commandBuffer, wfArgsBuf, WF_OFF_SHADOW_ARGS);
                wfBarrier(commandBuffer);
                wfBind(commandBuffer, wfArgsPipe);
                wfPush(commandBuffer, parity, 3, s);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                wfBarrier(commandBuffer);
                parity ^= 1;
            }

            wfBind(commandBuffer, wfFinalizePipe);
            wfPush(commandBuffer, parity, 0, s);
            vkCmdDispatch(commandBuffer, pathGroups, 1, 1);
            wfBarrier(commandBuffer);
        }

        vkEndCommandBuffer(commandBuffer);
        delete _rec;

        ScopedFunctionTimer _sw("wf.submitAndWait");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffer;
        vkResetFences(device, 1, &fence);
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    vkDestroyFence(device, fence, nullptr);
}

};
inline VulkanContext vkCtx;
#endif
}