#pragma once
#ifdef VULKAN_SUPPORT
#include <vulkan/vulkan.h>
#endif

namespace Grid {

template<typename T, typename IndexType, GridStoragePath StoragePath>
struct RenderData_ {
    PointType position;
    float size;
    Eigen::Vector3f color;
    Material_<T, IndexType, StoragePath> material;
    PointType boundsMin;
    PointType boundsMax;
    int objectId;
};

template<typename T, typename IndexType, GridStoragePath StoragePath>
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
    
    OctreeNode_<T, IndexType, StoragePath>* originalNode;
};

template<typename T, typename IndexType, GridStoragePath StoragePath>
struct RenderBuffer_ {
    std::vector<RenderNode_<T, IndexType, StoragePath>> nodes;
    std::vector<RenderData_<T, IndexType, StoragePath>> points;
    
    void clear() {
        nodes.clear();
        points.clear();
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

struct alignas(16) GPUFastRenderData {
    Eigen::Vector3f position;
    float size;
    uint32_t color;
    float emittance;
    int objectId;
    uint32_t padding2;
};

struct alignas(16) GPUPBRRenderData {
    Eigen::Vector3f position;
    float size;
    uint32_t color;
    float emittance;
    uint32_t materialProps;
    uint32_t absorption;
    int objectId;
    uint32_t padding1;
    uint32_t padding2;
    uint32_t padding3;
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
    int padding2;
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
    
    VkShaderModule fastShader = VK_NULL_HANDLE;
    VkShaderModule pbrShader = VK_NULL_HANDLE;
    VkPipelineLayout fastPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout pbrPipelineLayout = VK_NULL_HANDLE;
    VkPipeline fastPipeline = VK_NULL_HANDLE;
    VkPipeline pbrPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout fastDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet fastDescSet = VK_NULL_HANDLE;
    VkDescriptorSet pbrDescSet = VK_NULL_HANDLE;

    VkBuffer nodeBuffer = VK_NULL_HANDLE;
    VkBuffer outBuffer = VK_NULL_HANDLE;
    VkBuffer uboBuffer = VK_NULL_HANDLE;
    VkBuffer fastPointBuffer = VK_NULL_HANDLE;
    VkBuffer pbrPointBuffer = VK_NULL_HANDLE;
    VkBuffer skyboxBuffer = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;

    VkDeviceMemory nodeMem = VK_NULL_HANDLE;
    VkDeviceMemory outMem = VK_NULL_HANDLE;
    VkDeviceMemory uboMem = VK_NULL_HANDLE;
    VkDeviceMemory fastPointMem = VK_NULL_HANDLE;
    VkDeviceMemory pbrPointMem = VK_NULL_HANDLE;
    VkDeviceMemory skyboxMem = VK_NULL_HANDLE;
    VkDeviceMemory lightMem = VK_NULL_HANDLE;

    size_t currentNodesCap = 0;
    size_t currentOutCap = 0;
    size_t currentFastPointsCap = 0;
    size_t currentPBRPointsCap = 0;
    size_t currentSkyboxCap = 0;
    size_t currentLightCap = 0;

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

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return 0;
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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2}
        };
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.poolSizeCount = 3;
        poolCreateInfo.pPoolSizes = poolSizes;
        poolCreateInfo.maxSets = 3;
        vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);

        uint32_t bindingCount = hasHardwareRT ? 7 : 6;
        VkDescriptorSetLayoutBinding bindings[7] = {};
        for(int i=0; i<6; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        if (hasHardwareRT) {
            bindings[6].binding = 6;
            bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            bindings[6].descriptorCount = 1;
            bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = bindingCount;
        layoutInfo.pBindings = bindings;

        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &fastDescLayout);
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &pbrDescLayout);

        VkDescriptorSetAllocateInfo allocSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &fastDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &fastDescSet);
        allocSetInfo.pSetLayouts = &pbrDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &pbrDescSet);

        if (hasHardwareRT) {
            std::cout << "using _hw versions" << std::endl;
            fastShader = createShaderModule("./bin/fast_raytrace_hw.spv");
            pbrShader = createShaderModule("./bin/pbr_raytrace_hw.spv");
        } else {
            std::cout << "using software versions" << std::endl;
            fastShader = createShaderModule("./bin/fast_raytrace.spv");
            pbrShader = createShaderModule("./bin/pbr_raytrace.spv");
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &fastDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &fastPipelineLayout);
        pipelineLayoutInfo.pSetLayouts = &pbrDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pbrPipelineLayout);

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

    template<typename RenderDataType>
    void buildHardwareAccelerationStructures(const std::vector<RenderDataType>& points) {
        if (!hasHardwareRT || points.empty()) return;

        std::vector<VkAabbPositionsKHR> aabbs(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
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
        blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasBuildInfo.geometryCount = 1;
        blasBuildInfo.pGeometries = &blasGeom;

        uint32_t numPrimitives = static_cast<uint32_t>(points.size());
        VkAccelerationStructureBuildSizesInfoKHR blasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo, &numPrimitives, &blasSizeInfo);

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
        tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuildInfo.geometryCount = 1;
        tlasBuildInfo.pGeometries = &tlasGeom;

        uint32_t numInstances = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo, &numInstances, &tlasSizeInfo);

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

        VkBuffer scratchBuffer;
        VkDeviceMemory scratchMem;
        VkDeviceSize scratchSize = std::max(blasSizeInfo.buildScratchSize, tlasSizeInfo.buildScratchSize);
        createBufferWithAddress(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMem);

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

        vkDestroyBuffer(device, scratchBuffer, nullptr);
        vkFreeMemory(device, scratchMem, nullptr);

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

    void updateLightBuffer(const std::vector<uint32_t>& lights) {
        size_t size = std::max((size_t)256, lights.size() * sizeof(uint32_t));
        if(size > currentLightCap) {
            if(lightBuffer) {
                vkDestroyBuffer(device, lightBuffer, nullptr);
                vkFreeMemory(device, lightMem, nullptr);
            }
            createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightBuffer, lightMem);
            currentLightCap = size;
        }
        if(!lights.empty()) {
            void* data;
            vkMapMemory(device, lightMem, 0, size, 0, &data);
            memcpy(data, lights.data(), lights.size() * sizeof(uint32_t));
            vkUnmapMemory(device, lightMem);
        }
    }

    void updateCommonBuffers(const std::vector<GPURenderNode>& nodes, size_t outSize, GPUCameraData& camData) {
        size_t nodeSize = std::max((size_t)256, nodes.size() * sizeof(GPURenderNode));
        size_t uboSize = sizeof(GPUCameraData);

        if(nodeSize > currentNodesCap) {
            if(nodeBuffer) {
                vkDestroyBuffer(device, nodeBuffer, nullptr);
                vkFreeMemory(device, nodeMem, nullptr);
            }
            createBuffer(nodeSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nodeBuffer, nodeMem);
            currentNodesCap = nodeSize;
        }
        if(outSize > currentOutCap) {
            if(outBuffer) {
                vkDestroyBuffer(device, outBuffer, nullptr);
                vkFreeMemory(device, outMem, nullptr);
            }
            createBuffer(outSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, outBuffer, outMem);
            currentOutCap = outSize;
        }
        if(!uboBuffer) {
            createBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuffer, uboMem);
        }

        void* data;
        if(!nodes.empty()) { 
            vkMapMemory(device, nodeMem, 0, nodeSize, 0, &data);
            memcpy(data, nodes.data(), nodes.size() * sizeof(GPURenderNode));
            vkUnmapMemory(device, nodeMem);
        }
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
        size_t size = std::max((size_t)256, skyData.size() * sizeof(Eigen::Vector4f));
        if(size > currentSkyboxCap) {
            if(skyboxBuffer) {
                vkDestroyBuffer(device, skyboxBuffer, nullptr);
                vkFreeMemory(device, skyboxMem, nullptr);
            }
            createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, skyboxBuffer, skyboxMem);
            currentSkyboxCap = size;
        }
        void* data;
        if(!skyData.empty()) {
            vkMapMemory(device, skyboxMem, 0, size, 0, &data);
            memcpy(data, skyData.data(), skyData.size() * sizeof(Eigen::Vector4f));
            vkUnmapMemory(device, skyboxMem);
        }
    }

    void updateFastBuffers(const std::vector<GPUFastRenderData>& points) {
        size_t pointSize = std::max((size_t)256, points.size() * sizeof(GPUFastRenderData));
        if(pointSize > currentFastPointsCap) {
            if(fastPointBuffer) {
                vkDestroyBuffer(device, fastPointBuffer, nullptr);
                vkFreeMemory(device, fastPointMem, nullptr);
            }
            createBuffer(pointSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, fastPointBuffer, fastPointMem);
            currentFastPointsCap = pointSize;
        }
        if (hasHardwareRT) buildHardwareAccelerationStructures(points);
        
        void* data;
        if(!points.empty()) {
            vkMapMemory(device, fastPointMem, 0, pointSize, 0, &data);
            memcpy(data, points.data(), points.size() * sizeof(GPUFastRenderData));
            vkUnmapMemory(device, fastPointMem);
        }

        VkDescriptorBufferInfo bInfos[6] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {fastPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE},
            {lightBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[6] = {};
        for(int i=0; i<6; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = fastDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    void updatePBRBuffers(const std::vector<GPUPBRRenderData>& points) {
        size_t pointSize = std::max((size_t)256, points.size() * sizeof(GPUPBRRenderData));
        if(pointSize > currentPBRPointsCap) {
            if(pbrPointBuffer) {
                vkDestroyBuffer(device, pbrPointBuffer, nullptr);
                vkFreeMemory(device, pbrPointMem, nullptr);
            }
            createBuffer(pointSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pbrPointBuffer, pbrPointMem);
            currentPBRPointsCap = pointSize;
        }
        if (hasHardwareRT) buildHardwareAccelerationStructures(points);
        
        void* data;
        if(!points.empty()) {
            vkMapMemory(device, pbrPointMem, 0, pointSize, 0, &data);
            memcpy(data, points.data(), points.size() * sizeof(GPUPBRRenderData));
            vkUnmapMemory(device, pbrPointMem);
        }

        VkDescriptorBufferInfo bInfos[6] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {pbrPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE},
            {lightBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[6] = {};
        for(int i=0; i<6; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pbrDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }
};
inline VulkanContext vkCtx;
#endif
}