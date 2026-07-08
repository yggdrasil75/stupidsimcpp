#pragma once
#ifdef VULKAN_SUPPORT
#include <vulkan/vulkan.h>
#include <cstdlib>
#include <chrono>
#endif

namespace Grid {

struct RenderData {
    Vec3 position;
    float size;
    Eigen::Vector4f color;
    uint32_t materialIdx;
    int objectId;

    const Vec3 boundsMin() const {
        return (position - Vec3::Constant(0.5 * size));
    }

    const Vec3 boundsMax() const {
        return (position + Vec3::Constant(0.5 * size));
    }
};

template<typename T>
struct RenderNode_ {
    Vec3 center;
    float nodeSize;
    bool isLeaf;
    bool isLoaded;
    uint8_t childMask;
    
    uint32_t firstPoint;
    uint32_t pointCount;
    int32_t lodPoint;
    uint32_t firstChild;
    
    OctreeNode_<T>* originalNode;

    const Vec3 boundsMin() const {
        return (center - Vec3::Constant(0.5 * nodeSize));
    }

    const Vec3 boundsMax() const {
        return (center + Vec3::Constant(0.5 * nodeSize));
    }
};

template<typename T>
struct RenderBuffer_ {
    std::vector<RenderNode_<T>> nodes;
    std::vector<RenderData> points;
    std::vector<RenderMaterial> materials;
    uint32_t defaultMatIdx;

    void clear() {
        nodes.clear();
        points.clear();
        materials.clear();
    }
};

#ifdef VULKAN_SUPPORT
static PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
static constexpr uint32_t VCT_RES = 128;
static constexpr uint32_t WF_PATH_STRIDE   = 6 * 4 * sizeof(float);
static constexpr uint32_t WF_PATHHIT_STRIDE= 1 * 4 * sizeof(float);
static constexpr uint32_t WF_SHADOW_STRIDE = 4 * 4 * sizeof(float);
static constexpr uint32_t WF_COUNTER_SIZE  = 16 * sizeof(uint32_t);
static constexpr VkDeviceSize WF_OFF_EXTEND_ARGS = 16;
static constexpr VkDeviceSize WF_OFF_SHADE_ARGS  = 32;
static constexpr VkDeviceSize WF_OFF_SHADOW_ARGS = 48;

struct alignas(16) GPURenderData {
    Vec3 position;
    float size;
    uint32_t color;
    uint32_t materialIdx;
    int objectId;
};

struct alignas(16) GPUCameraData {
    Vec3 origin;
    float lodMinDist;
    Vec3 dir;
    float invLodf;
    Vec3 up;
    float minVisibility;
    Vec3 right;
    float maxDist;
    Vec3 skylight;
    float tanfovx;
    Vec3 bgColor;
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
    int fogVolumeCount;
    int pad0;
};

//change:
//remove minb/maxb.
//set position and radius.
//make fog spherical volume?
struct alignas(16) GPUFogVolume {
    Vec3 minB;
    float density;
    Vec3 maxB;
    float pad0;
    Vec3 scatter;
    float pad1;
    Vec3 absorb;
    float pad2;
};

struct alignas(16) VCTParams {
    Vec3 volMin;
    float voxelSize;
    Vec3 volExtent;
    float invVoxelSize;
    Eigen::Vector3i gridRes;
    int maxMip;
    Vec3 lightDir;
    float enabled;
};

struct VCTMipPush {
    int dstRes[3];
    int pad;
};

static uint32_t vctMipCount(uint32_t res) {
    uint32_t m = 1;
    while (res > 1) {
        res >>= 1;
        ++m;
    }
    return m;
}

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VkDevice> activeDevices;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice primaryDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceType deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence renderFence = VK_NULL_HANDLE;
    VkCommandBuffer wfCmd[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence wfFence[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    
    VkShaderModule fastShader = VK_NULL_HANDLE;
    VkShaderModule smoothShader = VK_NULL_HANDLE;
    VkShaderModule blendShader = VK_NULL_HANDLE;
    VkShaderModule guidedCoeffShader = VK_NULL_HANDLE;
    VkShaderModule wfInitShader = VK_NULL_HANDLE;
    VkShaderModule wfArgsShader = VK_NULL_HANDLE;
    VkShaderModule wfExtendShader = VK_NULL_HANDLE;
    VkShaderModule wfShadeShader = VK_NULL_HANDLE;
    VkShaderModule wfShadowShader = VK_NULL_HANDLE;
    VkShaderModule wfFinalizeShader = VK_NULL_HANDLE;

    VkPipelineLayout fastPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout pbrPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout smoothPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout blendPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout guidedCoeffPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout wfPipelineLayout = VK_NULL_HANDLE;

    VkPipeline fastPipeline = VK_NULL_HANDLE;
    VkPipeline pbrPipeline = VK_NULL_HANDLE;
    VkPipeline smoothPipeline = VK_NULL_HANDLE;
    VkPipeline blendPipeline = VK_NULL_HANDLE;
    VkPipeline guidedCoeffPipeline = VK_NULL_HANDLE;
    VkPipeline wfInitPipe = VK_NULL_HANDLE;
    VkPipeline wfArgsPipe = VK_NULL_HANDLE;
    VkPipeline wfExtendPipe = VK_NULL_HANDLE;
    VkPipeline wfShadePipe = VK_NULL_HANDLE;
    VkPipeline wfShadowPipe = VK_NULL_HANDLE;
    VkPipeline wfFinalizePipe = VK_NULL_HANDLE;

    VkBuffer fastGBuffer = VK_NULL_HANDLE;
    VkBuffer wfPathBuf = VK_NULL_HANDLE;
    VkBuffer wfPathHitBuf = VK_NULL_HANDLE;
    VkBuffer wfExtendABuf = VK_NULL_HANDLE;
    VkBuffer wfExtendBBuf = VK_NULL_HANDLE;
    VkBuffer wfShadeBuf = VK_NULL_HANDLE;
    VkBuffer wfShadowBuf = VK_NULL_HANDLE;
    VkBuffer wfCounterBuf = VK_NULL_HANDLE;
    VkBuffer guidedCoeffBuffer = VK_NULL_HANDLE;
    VkBuffer nodeBuffer = VK_NULL_HANDLE;
    VkBuffer outBuffer = VK_NULL_HANDLE;
    VkBuffer uboBuffer = VK_NULL_HANDLE;
    VkBuffer fastPointBuffer = VK_NULL_HANDLE;
    VkBuffer pbrPointBuffer = VK_NULL_HANDLE;
    VkBuffer skyboxBuffer = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;
    VkBuffer fogBuffer = VK_NULL_HANDLE;
    VkBuffer finalOutBuffer = VK_NULL_HANDLE;
    VkBuffer lowResOutBuffer = VK_NULL_HANDLE;
    VkBuffer adaptiveBuffer = VK_NULL_HANDLE;
    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkBuffer asScratchBuffer = VK_NULL_HANDLE;
    VkBuffer outStagingBuffer = VK_NULL_HANDLE;
    VkBuffer xferStagingBuffer = VK_NULL_HANDLE;
    VkBuffer aabbBuffer = VK_NULL_HANDLE;
    VkBuffer asInstanceBuffer = VK_NULL_HANDLE;
    VkBuffer blasBuffer = VK_NULL_HANDLE;
    VkBuffer tlasBuffer = VK_NULL_HANDLE;
    VkBuffer smoothScratchBuffer = VK_NULL_HANDLE;
    VkBuffer sellmeierBuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout fastDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout smoothDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout blendDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout guidedCoeffDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout wfDescLayout = VK_NULL_HANDLE;

    VkDescriptorSet wfDescSet    = VK_NULL_HANDLE;
    VkDescriptorSet fastDescSet = VK_NULL_HANDLE;
    VkDescriptorSet pbrDescSet = VK_NULL_HANDLE;
    VkDescriptorSet smoothDescSet = VK_NULL_HANDLE;
    VkDescriptorSet blendDescSet = VK_NULL_HANDLE;
    VkDescriptorSet guidedCoeffDescSet = VK_NULL_HANDLE;

    VkDeviceMemory fastGBufferMem = VK_NULL_HANDLE;
    VkDeviceMemory wfPathMem = VK_NULL_HANDLE;
    VkDeviceMemory wfPathHitMem = VK_NULL_HANDLE;
    VkDeviceMemory wfExtendAMem = VK_NULL_HANDLE;
    VkDeviceMemory wfExtendBMem = VK_NULL_HANDLE;
    VkDeviceMemory wfShadeMem = VK_NULL_HANDLE;
    VkDeviceMemory wfShadowMem = VK_NULL_HANDLE;
    VkDeviceMemory wfCounterMem = VK_NULL_HANDLE;
    VkDeviceMemory guidedCoeffMem = VK_NULL_HANDLE;
    VkDeviceMemory nodeMem = VK_NULL_HANDLE;
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
    VkDeviceMemory fogMem = VK_NULL_HANDLE;
    VkDeviceMemory asScratchMem = VK_NULL_HANDLE;
    VkDeviceMemory outStagingMem = VK_NULL_HANDLE;
    VkDeviceMemory xferStagingMem = VK_NULL_HANDLE;
    VkDeviceMemory aabbMem = VK_NULL_HANDLE;
    VkDeviceMemory asInstanceMem = VK_NULL_HANDLE;
    VkDeviceMemory blasMem = VK_NULL_HANDLE;
    VkDeviceMemory tlasMem = VK_NULL_HANDLE;
    VkDeviceMemory smoothScratchMem = VK_NULL_HANDLE;
    VkDeviceMemory sellmeierMem = VK_NULL_HANDLE;
    
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;

    uint32_t currentFastGCap = 0;
    uint32_t currentFastPointsCap = 0;
    uint32_t wfPathCap = 0;
    uint32_t currentGuidedCoeffCap = 0;
    uint32_t currentFogCap = 0;
    uint32_t currentNodesCap = 0;
    uint32_t currentOutCap = 0;
    uint32_t currentPBRPointsCap = 0;
    uint32_t currentSkyboxCap = 0;
    uint32_t currentLightCap = 0;
    uint32_t currentFinalOutCap = 0;
    uint32_t currentLowResOutCap = 0;
    uint32_t currentAdaptiveCap = 0;
    uint32_t currentMaterialCap = 0;
    uint32_t currentAabbCap = 0;
    uint32_t currentScratchCap = 0;
    uint32_t lastBlasPrimCount = 0;
    uint32_t framesSinceFullBuild = 0;
    uint32_t currentOutStagingCap = 0;
    uint32_t currentXferStagingCap = 0;
    uint32_t currentSmoothScratchCap = 0;
    uint32_t currentSellmeierCap = 0;
    uint32_t sellmeierWidth = 0;
    uint32_t sellmeierRows = 0;

    bool initialized = false;
    bool ownsInstance = true;
    bool blasTopologyValid = false;
    bool outMemCoherent = true;
    bool outStagingCoherent = true;
    bool xferStagingCoherent = true;
    bool vctReady = false;

    int lastBlasOrderingTag = -1;

    void* outStagingMapped = nullptr;
    void* xferStagingMapped = nullptr;


    uint32_t findMemoryType(VkPhysicalDevice& phDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(phDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return 0;
    }

    uint32_t findMemoryTypePreferred(VkPhysicalDevice& phDevice, uint32_t typeFilter, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, bool& gotPreferred) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(phDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = memProperties.memoryTypes[i].propertyFlags;
            if ((typeFilter & (1 << i)) && ((f & (required | preferred)) == (required | preferred))) {
                gotPreferred = true;
                return i;
            }
        }
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

    void createReadbackBuffer(VkDevice& device, VkPhysicalDevice& PhDevice, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool& coherentOut) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        bool gotCached = false;
        uint32_t typeIdx = findMemoryTypePreferred(PhDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
             VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, gotCached);

        // Determine coherence of the chosen type so we know whether to invalidate.
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(PhDevice, &memProps);
        coherentOut = (memProps.memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = typeIdx;
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createBuffer(VkDevice& device, VkPhysicalDevice& PhDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
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
        allocInfo.memoryTypeIndex = findMemoryType(PhDevice, memRequirements.memoryTypeBits, properties);
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkShaderModule createShaderModule(VkDevice& device, const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "FAILED TO LOAD " << path << "!\n";
            return VK_NULL_HANDLE;
        }
        uint32_t fileSize = (uint32_t) file.tellg();
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

    static int scorePhysicalDevice(VkPhysicalDevice pd) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qProps.data());
        bool hasCompute = false;
        for (const auto& q : qProps) {
            if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                hasCompute = true;
                break;
            }
        }
        if (!hasCompute) return -1;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
            default: return 1;
        }
    }
    
    void createAllShaderModules() {
        fastShader = createShaderModule(device, "./bin/fast_raytrace_hw.spv");
        smoothShader = createShaderModule(device, "./bin/smooth.spv");
        blendShader = createShaderModule(device, "./bin/blend.spv");
        guidedCoeffShader = createShaderModule(device, "./bin/guided_coeff.spv");
        wfInitShader = createShaderModule(device, "./bin/wf_init.spv");
        wfArgsShader = createShaderModule(device, "./bin/wf_args.spv");
        wfExtendShader = createShaderModule(device, "./bin/wf_extend.spv");
        wfShadeShader = createShaderModule(device, "./bin/wf_shade.spv");
        wfShadowShader = createShaderModule(device, "./bin/wf_shadow.spv");
        wfFinalizeShader = createShaderModule(device, "./bin/wf_finalize.spv");
    }

    void init(VkPhysicalDevice PhDevice = VK_NULL_HANDLE) {
        uint32_t extCount;
        if (initialized) return;
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.apiVersion = VK_API_VERSION_1_2;

        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &appInfo;
        vkCreateInstance(&createInfo, nullptr, &instance);

        if (PhDevice != VK_NULL_HANDLE) {
            primaryDevice = PhDevice;
        } else {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
            // Pick the best-scoring device (discrete GPU > integrated > virtual > cpu)
            // instead of blindly taking devices[0].
            int bestScore = -1;
            for (auto d : devices) {
                int s = scorePhysicalDevice(d);
                if (s > bestScore) {
                    bestScore = s;
                    primaryDevice = d;
                }
            }
            if (primaryDevice == VK_NULL_HANDLE && deviceCount > 0) primaryDevice = devices[0];
        }

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(primaryDevice, &props);
        deviceType = props.deviceType;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(primaryDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(primaryDevice, &queueFamilyCount, queueFamilies.data());
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
        
        vkEnumerateDeviceExtensionProperties(primaryDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(primaryDevice, nullptr, &extCount, availableExts.data());

        VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rqFeatures};
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &asFeatures};
        VkPhysicalDeviceFeatures2 deviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features12};
        
        std::vector<const char*> deviceExtensions;

        deviceFeatures2.pNext = &features12;
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        features12.pNext = &asFeatures;
        asFeatures.pNext = &rqFeatures;
        asFeatures.accelerationStructure = VK_TRUE;
        rqFeatures.rayQuery = VK_TRUE;
        features12.bufferDeviceAddress = VK_TRUE;

        VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreateInfo.pNext = &deviceFeatures2;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        vkCreateDevice(primaryDevice, &deviceCreateInfo, nullptr, &device);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        pfn_vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
        pfn_vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        pfn_vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        pfn_vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
        pfn_vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");

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
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 12},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 6},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},          // VCT mip + voxelize views
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}   // VCT volume sampler in fast/pbr sets
        };
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.poolSizeCount = 5;
        poolCreateInfo.pPoolSizes = poolSizes;
        poolCreateInfo.maxSets = 40;
        vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);

        VkDescriptorSetLayoutBinding bindings[11] = {};
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

        // VCT: mipmapped radiance volume sampler + params UBO
        bindings[9].binding = 9;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[10].binding = 10;
        bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[10].descriptorCount = 1;
        bindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 11;
        layoutInfo.pBindings = bindings;

        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &fastDescLayout);
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &pbrDescLayout);

        VkDescriptorSetLayoutBinding smBindings[3] = {};
        for(int i=0; i<3; i++) {
            smBindings[i].binding = i;
            smBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            smBindings[i].descriptorCount = 1;
            smBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo smLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, smBindings};
        vkCreateDescriptorSetLayout(device, &smLayoutInfo, nullptr, &smoothDescLayout);

        VkDescriptorSetLayoutBinding blBindings[4] = {};
        for(int i=0; i<4; i++) {
            blBindings[i].binding = i;
            blBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            blBindings[i].descriptorCount = 1;
            blBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo blLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, blBindings};
        vkCreateDescriptorSetLayout(device, &blLayoutInfo, nullptr, &blendDescLayout);

        // Guided-filter coefficient pass: guide (full), PT (low), coeff out.
        VkDescriptorSetLayoutCreateInfo gcLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, blBindings};
        vkCreateDescriptorSetLayout(device, &gcLayoutInfo, nullptr, &guidedCoeffDescLayout);

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
        allocSetInfo.pSetLayouts = &guidedCoeffDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &guidedCoeffDescSet);

        createAllShaderModules();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &fastDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &fastPipelineLayout);
        pipelineLayoutInfo.pSetLayouts = &pbrDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pbrPipelineLayout);

        VkPushConstantRange smPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 5};
        pipelineLayoutInfo.pSetLayouts = &smoothDescLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &smPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &smoothPipelineLayout);

        VkPushConstantRange blPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 5 + sizeof(float)};
        pipelineLayoutInfo.pSetLayouts = &blendDescLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &blPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &blendPipelineLayout);

        VkPushConstantRange gcPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 5};
        pipelineLayoutInfo.pSetLayouts = &guidedCoeffDescLayout;
        pipelineLayoutInfo.pPushConstantRanges = &gcPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &guidedCoeffPipelineLayout);

        VkComputePipelineCreateInfo computePipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computePipelineInfo.stage.pName = "main";

        if (fastShader) {
            computePipelineInfo.layout = fastPipelineLayout;
            computePipelineInfo.stage.module = fastShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &fastPipeline);
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
        if (guidedCoeffShader) {
            computePipelineInfo.layout = guidedCoeffPipelineLayout;
            computePipelineInfo.stage.module = guidedCoeffShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &guidedCoeffPipeline);
        }

        initWavefront();

        vctInit();

        initialized = true;
    }
    
    void destroyBuffer(VkDevice& device, VkBuffer& bf, VkDeviceMemory& mm) {
        if (bf) {
            vkDestroyBuffer(device, bf, nullptr);
            vkFreeMemory(device, mm, nullptr);
            bf = VK_NULL_HANDLE;
        }
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

    void copyBuffer(VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        executeSingleTimeCommands([&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, srcBuffer, dstBuffer, 1, &copyRegion);
        });
    }

    void ensureXferStaging(uint32_t size) {
        if (size <= currentXferStagingCap) return;
        if (xferStagingBuffer) {
            vkUnmapMemory(device, xferStagingMem);
            vkDestroyBuffer(device, xferStagingBuffer, nullptr);
            vkFreeMemory(device, xferStagingMem, nullptr);
            xferStagingMapped = nullptr;
        }
        createReadbackBuffer(device, primaryDevice, size, xferStagingBuffer, xferStagingMem, xferStagingCoherent);
        vkMapMemory(device, xferStagingMem, 0, VK_WHOLE_SIZE, 0, &xferStagingMapped);
        currentXferStagingCap = size;
    }

    void flushXferStaging(uint32_t size) {
        if (xferStagingCoherent) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = xferStagingMem;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device, 1, &range);
    }

    void invalidateXferStaging() {
        if (xferStagingCoherent) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = xferStagingMem;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device, 1, &range);
    }

    void uploadToBuffer(VkBuffer dst, const void* src, uint32_t size) {
        if (!src || size == 0) return;
        ensureXferStaging(size);
        memcpy(xferStagingMapped, src, size);
        flushXferStaging(size);
        copyBuffer(device, xferStagingBuffer, dst, size);
    }

    void downloadFromBuffer(VkBuffer src, void* dst, uint32_t size) {
        if (!dst || size == 0) return;
        ensureXferStaging(size);
        copyBuffer(device, src, xferStagingBuffer, size);
        invalidateXferStaging();
        memcpy(dst, xferStagingMapped, size);
    }

    void updateDeviceLocalBuffer(VkBuffer& buffer, VkDeviceMemory& memory, uint32_t& currentCap, 
                                 const void* data, uint32_t dataSize, uint32_t allocSize, VkBufferUsageFlags usage) {
        if (allocSize > currentCap) {
            if (buffer) {
                vkDestroyBuffer(device, buffer, nullptr);
                vkFreeMemory(device, memory, nullptr);
            }
            createBuffer(device, primaryDevice, allocSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
            currentCap = allocSize;
        }

        if (data && dataSize > 0) {
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMem;
            createBuffer(device, primaryDevice, dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                         stagingBuffer, stagingMem);

            void* mappedData;
            vkMapMemory(device, stagingMem, 0, dataSize, 0, &mappedData);
            memcpy(mappedData, data, dataSize);
            vkUnmapMemory(device, stagingMem);

            copyBuffer(device, stagingBuffer, buffer, dataSize);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
        }
    }

    void createBufferWithAddress(VkDevice& device, VkDeviceSize size, VkBufferUsageFlags usage, 
                                 VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
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
        allocInfo.memoryTypeIndex = findMemoryType(primaryDevice, memReqs.memoryTypeBits, properties);
        
        vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }

    void buildHardwareAccelerationStructures(const std::vector<GPURenderData>& points, int orderingTag = 0) {
        if (points.empty()) return;

        const uint32_t numPrimitives = static_cast<uint32_t>(points.size());
        bool doFullBuild = (!blasTopologyValid) || (numPrimitives != lastBlasPrimCount)
                        || (orderingTag != lastBlasOrderingTag);
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

        uint32_t aabbSize = numPrimitives * sizeof(VkAabbPositionsKHR);
        if (aabbSize > currentAabbCap) {
            if (aabbBuffer) {
                vkDestroyBuffer(device, aabbBuffer, nullptr);
                vkFreeMemory(device, aabbMem, nullptr);
            }
            createBufferWithAddress(device, aabbSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 
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
            createBufferWithAddress(device, blasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
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
        createBufferWithAddress(device, sizeof(VkAccelerationStructureInstanceKHR), 
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
            createBufferWithAddress(device, tlasSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
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
            createBufferWithAddress(device, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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

        if (doFullBuild) {
            lastBlasPrimCount = numPrimitives;
            lastBlasOrderingTag = orderingTag;
            blasTopologyValid = true;
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
            framesSinceFullBuild = 0;
        } else {
            framesSinceFullBuild++;
        }
    }

    void updateFogBuffer(const std::vector<GPUFogVolume>& vols) {
        size_t allocSize = std::max((size_t)256, vols.size() * sizeof(GPUFogVolume));
        size_t dataSize = vols.size() * sizeof(GPUFogVolume);
        updateDeviceLocalBuffer(fogBuffer, fogMem, currentFogCap,
                                vols.empty() ? nullptr : vols.data(), dataSize, allocSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
            createBuffer(device, primaryDevice, outSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuffer, outMem);
            outMemCoherent = true; // no host mapping of outMem anymore
            currentOutCap = outSize;

            if (outStagingBuffer) {
                vkUnmapMemory(device, outStagingMem);
                vkDestroyBuffer(device, outStagingBuffer, nullptr);
                vkFreeMemory(device, outStagingMem, nullptr);
                outStagingMapped = nullptr;
            }
            createReadbackBuffer(device, primaryDevice, outSize, outStagingBuffer, outStagingMem, outStagingCoherent);
            vkMapMemory(device, outStagingMem, 0, VK_WHOLE_SIZE, 0, &outStagingMapped);
            currentOutStagingCap = outSize;
        }

        size_t adaptiveSize = outSize; 
        if(adaptiveSize > currentAdaptiveCap) {
            if(adaptiveBuffer) {
                vkDestroyBuffer(device, adaptiveBuffer, nullptr);
                vkFreeMemory(device, adaptiveMem, nullptr);
            }
            createBuffer(device, primaryDevice, adaptiveSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, adaptiveBuffer, adaptiveMem);
            currentAdaptiveCap = adaptiveSize;
        }

        size_t uboSize = sizeof(GPUCameraData);
        if(!uboBuffer) {
            createBuffer(device, primaryDevice, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuffer, uboMem);
        }

        void* data;
        vkMapMemory(device, uboMem, 0, uboSize, 0, &data);
        memcpy(data, &camData, uboSize);
        vkUnmapMemory(device, uboMem);
    }

    const float* readbackOut(VkDeviceSize size) {
        copyBuffer(device, outBuffer, outStagingBuffer, size);
        if (!outStagingCoherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = outStagingMem;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(device, 1, &range);
        }
        return static_cast<const float*>(outStagingMapped);
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

    void updateFastBuffers(const std::vector<GPURenderData>& points) {
        size_t allocSize = std::max((size_t)256, points.size() * sizeof(GPURenderData));
        size_t dataSize = points.size() * sizeof(GPURenderData);
        
        updateDeviceLocalBuffer(fastPointBuffer, fastPointMem, currentFastPointsCap, 
                                points.empty() ? nullptr : points.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        buildHardwareAccelerationStructures(points, 1);

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

        vctWriteFastDescriptors();
    }

    void updatePBRBuffers(const std::vector<GPURenderData>& points) {
        size_t allocSize = std::max((size_t)256, points.size() * sizeof(GPURenderData));
        size_t dataSize = points.size() * sizeof(GPURenderData);
        
        updateDeviceLocalBuffer(pbrPointBuffer, pbrPointMem, currentPBRPointsCap, 
                                points.empty() ? nullptr : points.data(), dataSize, allocSize, 
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        buildHardwareAccelerationStructures(points, 2);

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

        if (vctReady) {
            VkDescriptorImageInfo si{vctSampler, vctSampleView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo bi{vctParamBuf, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet vw[2]{};
            vw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; vw[0].dstSet = pbrDescSet;
            vw[0].dstBinding = 9; vw[0].descriptorCount = 1;
            vw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; vw[0].pImageInfo = &si;
            vw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; vw[1].dstSet = pbrDescSet;
            vw[1].dstBinding = 10; vw[1].descriptorCount = 1;
            vw[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; vw[1].pBufferInfo = &bi;
            vkUpdateDescriptorSets(device, 2, vw, 0, nullptr);
        }
    }

    void ensureLowResBuffer(uint32_t size) {
        if(size > currentLowResOutCap) {
            if(lowResOutBuffer) {
                vkDestroyBuffer(device, lowResOutBuffer, nullptr);
                vkFreeMemory(device, lowResOutMem, nullptr);
            }
            // Retained HOST_VISIBLE so it can be mapped later if necessary
            createBuffer(device, primaryDevice, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                         lowResOutBuffer, lowResOutMem);
            currentLowResOutCap = size;
        }
    }

    void retainFastGBuffer(uint32_t fastOutSize) {
        if (fastOutSize > currentFastGCap) {
            if (fastGBuffer) {
                vkDestroyBuffer(device, fastGBuffer, nullptr);
                vkFreeMemory(device, fastGBufferMem, nullptr);
            }
            createBuffer(device, primaryDevice, fastOutSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fastGBuffer, fastGBufferMem);
            currentFastGCap = fastOutSize;
        }
        copyBuffer(device, outBuffer, fastGBuffer, fastOutSize);
    }

    void dispatchSmoothPasses(int width, int height, int samples, int iters, bool toFinal) {
        uint32_t finalSize = width * height * 3 * sizeof(float);
        if(finalSize > currentFinalOutCap) {
            if(finalOutBuffer) {
                vkDestroyBuffer(device, finalOutBuffer, nullptr);
                vkFreeMemory(device, finalOutMem, nullptr);
            }
            createBuffer(device, primaryDevice, finalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, finalOutBuffer, finalOutMem);
            currentFinalOutCap = finalSize;
        }

        // Ping-pong scratch for the a-trous iterations (5 floats per pixel).
        uint32_t scratchSize = width * height * 5 * sizeof(float);
        if (scratchSize > currentSmoothScratchCap) {
            if (smoothScratchBuffer) {
                vkDestroyBuffer(device, smoothScratchBuffer, nullptr);
                vkFreeMemory(device, smoothScratchMem, nullptr);
            }
            createBuffer(device, primaryDevice, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, smoothScratchBuffer, smoothScratchMem);
            currentSmoothScratchCap = scratchSize;
        }

        VkBuffer src = outBuffer;
        VkBuffer dst = smoothScratchBuffer;

        for (int it = 0; it < iters; ++it) {
            bool finalPass = toFinal && (it == iters - 1);
            int step = 1 << it;
            VkBuffer outBuf = finalPass ? finalOutBuffer : dst;

            VkDescriptorBufferInfo bInfos[3] = {
                {src, 0, VK_WHOLE_SIZE},
                {outBuf, 0, VK_WHOLE_SIZE},
                {adaptiveBuffer, 0, VK_WHOLE_SIZE}
            };
            VkWriteDescriptorSet writes[3] = {};
            for(int i=0; i<3; i++) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = smoothDescSet;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &bInfos[i];
            }
            vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipelineLayout, 0, 1, &smoothDescSet, 0, nullptr);

            struct { int w, h, s, step, finalPass; } pc = {
                width, height, samples, step, finalPass ? 1 : 0
            };
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

            if (!finalPass) { VkBuffer tmp = src; src = dst; dst = tmp; }
        }
    }

    void dispatchSmooth(int width, int height, int samples) {
        dispatchSmoothPasses(width, height, samples, 4, true);
    }

    void dispatchBlend(int width, int height, int lowW, int lowH, float pbrScale, int samples) {
        uint32_t finalSize = width * height * 3 * sizeof(float);
        if(finalSize > currentFinalOutCap) {
            if(finalOutBuffer) { 
                vkDestroyBuffer(device, finalOutBuffer, nullptr); 
                vkFreeMemory(device, finalOutMem, nullptr); 
            }
            createBuffer(device, primaryDevice, finalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, finalOutBuffer, finalOutMem);
            currentFinalOutCap = finalSize;
        }

        // Per-low-pixel guided-filter coefficients (a.rgb, b.rgb).
        uint32_t coeffSize = uint32_t(lowW) * uint32_t(lowH) * 6 * sizeof(float);
        if (coeffSize > currentGuidedCoeffCap) {
            if (guidedCoeffBuffer) {
                vkDestroyBuffer(device, guidedCoeffBuffer, nullptr);
                vkFreeMemory(device, guidedCoeffMem, nullptr);
            }
            createBuffer(device, primaryDevice, coeffSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, guidedCoeffBuffer, guidedCoeffMem);
            currentGuidedCoeffCap = coeffSize;
        }

        VkDescriptorBufferInfo gcInfos[3] = { {outBuffer, 0, VK_WHOLE_SIZE}, {lowResOutBuffer, 0, VK_WHOLE_SIZE}, {guidedCoeffBuffer, 0, VK_WHOLE_SIZE} };
        VkDescriptorBufferInfo bInfos[4]  = { {outBuffer, 0, VK_WHOLE_SIZE}, {lowResOutBuffer, 0, VK_WHOLE_SIZE}, {finalOutBuffer, 0, VK_WHOLE_SIZE}, {guidedCoeffBuffer, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[7] = {};
        for(int i=0; i<3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = guidedCoeffDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &gcInfos[i];
        }
        for(int i=0; i<4; i++) {
            writes[3 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3 + i].dstSet = blendDescSet;
            writes[3 + i].dstBinding = i;
            writes[3 + i].descriptorCount = 1;
            writes[3 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3 + i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // Pass 1: fit PT ~= a*guide + b per low-res pixel.
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, guidedCoeffPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, guidedCoeffPipelineLayout, 0, 1, &guidedCoeffDescSet, 0, nullptr);
        struct { int lw, lh, fw, fh, s; } gpc = {lowW, lowH, width, height, samples};
        vkCmdPushConstants(commandBuffer, guidedCoeffPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpc), &gpc);
        vkCmdDispatch(commandBuffer, (lowW + 15) / 16, (lowH + 15) / 16, 1);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Pass 2: out_full = a * guide_full + b.
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

struct WFPushConstants {
    int parity;
    int stage;
    int sampleIndex;
    int pad;
};


void initWavefront() {

    VkDescriptorSetLayoutBinding b[17] = {};
    for (int i = 0; i < 17; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 17;
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

    wfInitShader     = createShaderModule(device, "./bin/wf_init.spv");
    wfArgsShader     = createShaderModule(device, "./bin/wf_args.spv");
    wfExtendShader   = createShaderModule(device, "./bin/wf_extend.spv");
    wfShadeShader    = createShaderModule(device, "./bin/wf_shade.spv");
    wfShadowShader   = createShaderModule(device, "./bin/wf_shadow.spv");
    wfFinalizeShader = createShaderModule(device, "./bin/wf_finalize.spv");

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

void ensureWavefrontBuffers(uint32_t maxPaths) {
    TIME_FUNCTION; //this probably shouldnt be called every tile.
    if (maxPaths <= wfPathCap && wfPathBuf) return;
    destroyBuffer(device, wfPathBuf, wfPathMem);
    destroyBuffer(device, wfPathHitBuf, wfPathHitMem);
    destroyBuffer(device, wfExtendABuf, wfExtendAMem);
    destroyBuffer(device, wfExtendBBuf, wfExtendBMem);
    destroyBuffer(device, wfShadeBuf, wfShadeMem);
    destroyBuffer(device, wfShadowBuf, wfShadowMem);
    destroyBuffer(device, wfCounterBuf, wfCounterMem);

    const VkBufferUsageFlags store = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    createBuffer(device, primaryDevice, maxPaths * WF_PATH_STRIDE,    store, devLocal, wfPathBuf,    wfPathMem);
    createBuffer(device, primaryDevice, maxPaths * WF_PATHHIT_STRIDE, store, devLocal, wfPathHitBuf, wfPathHitMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfExtendABuf, wfExtendAMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfExtendBBuf, wfExtendBMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfShadeBuf,   wfShadeMem);
    createBuffer(device, primaryDevice, maxPaths * WF_SHADOW_STRIDE, store, devLocal, wfShadowBuf,  wfShadowMem);
    createBuffer(device, primaryDevice, WF_COUNTER_SIZE, store | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, devLocal, wfCounterBuf, wfCounterMem);
    wfPathCap = maxPaths;
}

void writeWavefrontDescriptors() {
    TIME_FUNCTION; //this probably shouldnt be called every tile.
    VkDescriptorBufferInfo bi[17] = {};
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
    bi[16] = {fogBuffer ? fogBuffer : materialBuffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet w[17] = {};
    int n = 0;
    for (int i = 0; i < 17; ++i) {
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

void dispatchWavefront(int tileW, int tileH, int maxBounces, int samplesPerPixel, int sampleStart = 0) {
    TIME_FUNCTION;
    uint32_t maxPaths = uint32_t(tileW) * uint32_t(tileH);
    if (maxPaths == 0 || samplesPerPixel <= 0) return;
    ensureWavefrontBuffers(maxPaths);
    writeWavefrontDescriptors();

    const uint32_t WG = 64;
    uint32_t pathGroups = uint32_t((maxPaths + WG - 1) / WG);
    int maxIters = maxBounces + 24;

    const int samplesPerSubmit = 2;

    if (wfCmd[0] == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = commandPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 2;
        vkAllocateCommandBuffers(device, &cai, wfCmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fi, nullptr, &wfFence[0]);
        vkCreateFence(device, &fi, nullptr, &wfFence[1]);
    }

    int slot = 0;
    int sampleEnd = sampleStart + samplesPerPixel;
    for (int s0 = sampleStart; s0 < sampleEnd; s0 += samplesPerSubmit, slot ^= 1) {
        int s1 = std::min(s0 + samplesPerSubmit, sampleEnd);
        VkCommandBuffer cmd = wfCmd[slot];
        {
            // ScopedFunctionTimer _sw("wf.waitSlot");
            vkWaitForFences(device, 1, &wfFence[slot], VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &wfFence[slot]);
        }

        // ScopedFunctionTimer* _rec = new ScopedFunctionTimer("wf.recordCmd");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                wfPipelineLayout, 0, 1, &wfDescSet, 0, nullptr);

        for (int s = s0; s < s1; ++s) {
            // Clear all counters, then seed the primary-ray queue.
            wfBind(cmd, wfArgsPipe);
            wfPush(cmd, 0, 4, s);
            vkCmdDispatch(cmd, 1, 1, 1);
            wfBarrier(cmd);
            wfBind(cmd, wfInitPipe);
            wfPush(cmd, 0, 0, s);
            vkCmdDispatch(cmd, pathGroups, 1, 1);
            wfBarrier(cmd);
            // Publish extend args for the first iteration (old stage 0).
            wfBind(cmd, wfArgsPipe);
            wfPush(cmd, 0, 0, s);
            vkCmdDispatch(cmd, 1, 1, 1);
            wfBarrier(cmd);

            int parity = 0;
            for (int it = 0; it < maxIters; ++it) {
                // extend: trace the current ray queue
                wfBind(cmd, wfExtendPipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_EXTEND_ARGS);
                wfBarrier(cmd);
                wfBind(cmd, wfArgsPipe);
                wfPush(cmd, parity, 1, s);
                vkCmdDispatch(cmd, 1, 1, 1);
                wfBarrier(cmd);
                wfBind(cmd, wfShadePipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADE_ARGS);
                wfBarrier(cmd);
                wfBind(cmd, wfArgsPipe);
                wfPush(cmd, parity, 5, s);
                vkCmdDispatch(cmd, 1, 1, 1);
                wfBarrier(cmd);
                wfBind(cmd, wfShadowPipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADOW_ARGS);
                wfBarrier(cmd);
                parity ^= 1;
            }

            wfBind(cmd, wfFinalizePipe);
            wfPush(cmd, parity, 0, s);
            vkCmdDispatch(cmd, pathGroups, 1, 1);
            wfBarrier(cmd);
        }

        vkEndCommandBuffer(cmd);
        // delete _rec;

        // ScopedFunctionTimer _sw("wf.submit");
        vkWaitForFences(device, 1, &wfFence[slot ^ 1], VK_TRUE, UINT64_MAX);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, wfFence[slot]);
    }

    // Drain both slots before returning: callers read back / reuse the
    // output buffers immediately after this function.
    {
        // ScopedFunctionTimer _sw("wf.submitAndWait");
        vkWaitForFences(device, 2, wfFence, VK_TRUE, UINT64_MAX);
    }

}

#include "vct_host.inl"

};
inline VulkanContext vkCtx;

struct GPUManager {
    std::vector<VulkanContext*> contexts;              // [0] == &vkCtx
    std::vector<std::unique_ptr<VulkanContext>> extras; // owned secondary contexts
    bool initialized = false;

    size_t count() const { return contexts.size(); }
    VulkanContext& ctx(size_t i) { return *contexts[i]; }

    void init() {
        if (initialized) return;
        initialized = true;

        bool multiEnabled = true;

        // If something already initialized vkCtx (e.g. a fast-path render ran
        // first), adopt its instance; otherwise create one via vkCtx below.
        VkInstance instance = vkCtx.initialized ? vkCtx.instance : VK_NULL_HANDLE;
        if (instance == VK_NULL_HANDLE) {
            VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            appInfo.apiVersion = VK_API_VERSION_1_2;
            VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            createInfo.pApplicationInfo = &appInfo;
            vkCreateInstance(&createInfo, nullptr, &instance);
        }

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> all(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, all.data());

        struct Cand { VkPhysicalDevice dev; int score; uint32_t idx; };
        std::vector<Cand> cands;
        for (uint32_t i = 0; i < deviceCount; ++i) {
            int s = VulkanContext::scorePhysicalDevice(all[i]);
            if (s < 0) continue;
            cands.push_back({all[i], s, i});
        }

        std::stable_sort(cands.begin(), cands.end(),
                            [](const Cand& a, const Cand& b) { return a.score > b.score; });
        bool haveGPU = !cands.empty() && cands.front().score >= 2;
        if (haveGPU) {
            cands.erase(std::remove_if(cands.begin(), cands.end(),
                        [](const Cand& c) { return c.score <= 1; }), cands.end());
        }

        if (!multiEnabled && cands.size() > 1) cands.resize(1);

        // Primary context. If vkCtx was already initialized before the manager
        // ran, keep whatever device it has; otherwise give it the best device.
        if (!vkCtx.initialized) {
            vkCtx.init(cands.empty() ? VK_NULL_HANDLE : cands.front().dev);
        }
        contexts.push_back(&vkCtx);

        // Secondary contexts for the remaining devices.
        for (size_t i = 0; i < cands.size(); ++i) {
            if (cands[i].dev == vkCtx.primaryDevice) continue;
            if (contexts.size() == 1 && i == 0 && vkCtx.initialized && vkCtx.primaryDevice == cands[i].dev) continue;
            auto c = std::make_unique<VulkanContext>();
            c->init(cands[i].dev);
            contexts.push_back(c.get());
            extras.push_back(std::move(c));
        }
    }
};
inline GPUManager gpuMgr;
#endif
}