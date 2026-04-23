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
    Eigen::Vector3f color;
    float emittance;
    Eigen::Vector3f boundsMin;
    int objectId;
    Eigen::Vector3f boundsMax;
    float padding2;
};

struct alignas(16) GPUPBRRenderData {
    Eigen::Vector3f position;
    float size;
    Eigen::Vector3f color;
    float emittance;
    Eigen::Vector3f boundsMin;
    float roughness;
    Eigen::Vector3f boundsMax;
    float metallic;
    Eigen::Vector3f absorption;
    float transmission;
    float ior;
    int objectId;
    float padding2;
    float padding3;
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
    int padding1;
    int padding2;
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
    VkDeviceMemory nodeMem = VK_NULL_HANDLE;
    VkDeviceMemory outMem = VK_NULL_HANDLE;
    VkDeviceMemory uboMem = VK_NULL_HANDLE;
    VkDeviceMemory fastPointMem = VK_NULL_HANDLE;
    VkDeviceMemory pbrPointMem = VK_NULL_HANDLE;
    VkDeviceMemory skyboxMem = VK_NULL_HANDLE;

    size_t currentNodesCap = 0;
    size_t currentOutCap = 0;
    size_t currentFastPointsCap = 0;
    size_t currentPBRPointsCap = 0;
    size_t currentSkyboxCap = 0;

    bool initialized = false;

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
        file.seekg(0); file.read(buffer.data(), fileSize); file.close();
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
        appInfo.apiVersion = VK_API_VERSION_1_0;
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
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamilyIndex = i; break; }
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2}
        };
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.poolSizeCount = 2;
        poolCreateInfo.pPoolSizes = poolSizes;
        poolCreateInfo.maxSets = 2;
        vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);

        VkDescriptorSetLayoutBinding bindings[5] = {};
        for(int i=0; i<5; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 5;
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

        fastShader = createShaderModule("./bin/fast_raytrace.spv");
        pbrShader = createShaderModule("./bin/pbr_raytrace.spv");

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
            if(skyboxBuffer) { vkDestroyBuffer(device, skyboxBuffer, nullptr); vkFreeMemory(device, skyboxMem, nullptr); }
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
            if(fastPointBuffer) { vkDestroyBuffer(device, fastPointBuffer, nullptr); vkFreeMemory(device, fastPointMem, nullptr); }
            createBuffer(pointSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, fastPointBuffer, fastPointMem);
            currentFastPointsCap = pointSize;
        }
        
        void* data;
        if(!points.empty()) {
            vkMapMemory(device, fastPointMem, 0, pointSize, 0, &data);
            memcpy(data, points.data(), points.size() * sizeof(GPUFastRenderData));
            vkUnmapMemory(device, fastPointMem);
        }

        VkDescriptorBufferInfo bInfos[5] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {fastPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[5] = {};
        for(int i=0; i<5; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = fastDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
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
        
        void* data;
        if(!points.empty()) {
            vkMapMemory(device, pbrPointMem, 0, pointSize, 0, &data);
            memcpy(data, points.data(), points.size() * sizeof(GPUPBRRenderData));
            vkUnmapMemory(device, pbrPointMem);
        }

        VkDescriptorBufferInfo bInfos[5] = { 
            {nodeBuffer, 0, VK_WHOLE_SIZE}, 
            {pbrPointBuffer, 0, VK_WHOLE_SIZE}, 
            {outBuffer, 0, VK_WHOLE_SIZE}, 
            {uboBuffer, 0, VK_WHOLE_SIZE},
            {skyboxBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[5] = {};
        for(int i=0; i<5; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pbrDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i==3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }
};
inline VulkanContext vkCtx;
#endif
}