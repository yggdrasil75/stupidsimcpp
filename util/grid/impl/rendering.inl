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
    uint32_t extent = EXTENT_UNIT;

    const Vec3 boundsMin() const {
        return (position - Vec3::Constant(0.5f * size));
    }

    const Vec3 boundsMax() const {
        return (position + Vec3::Constant(0.5f * size)
                + Vec3::Constant(size).cwiseProduct(unpackExtent(extent) - Vec3::Ones()));
    }

    bool isMerged() const { return (extent & ~(EXTENT_STATIC_BIT | EXTENT_REUSE_BIT)) != EXTENT_UNIT; }
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
    
    uint32_t originalNode = INVALID_IDX;

    const Vec3 boundsMin() const {
        return (center - Vec3::Constant(0.5 * nodeSize));
    }

    const Vec3 boundsMax() const {
        return (center + Vec3::Constant(0.5 * nodeSize));
    }
};

struct MergeCacheEntry {
    std::vector<RenderData> boxes;
    uint32_t sourceCount = 0;
    uint32_t reuseSignature = 0;
};

template<typename T>
struct RenderBuffer_ {
    std::vector<RenderNode_<T>> nodes;
    std::vector<RenderData> points;
    std::vector<RenderMaterial> materials;
    uint32_t defaultMatIdx;

    std::unordered_map<uint32_t, MergeCacheEntry> mergeCache;

    void clear() {
        nodes.clear();
        points.clear();
        materials.clear();
    }

    void clearMergeCache() { mergeCache.clear(); }
};

struct InFlightFrame {
    int width = 0;
    int height = 0;
    size_t outSize = 0;
    frame::colormap colorformat = frame::colormap::RGB;
    bool pending = false;
    uint32_t slot = 0;
};

#ifdef VULKAN_SUPPORT
static PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
static constexpr uint32_t VCT_RES = 128;
static constexpr uint32_t WF_PATH_STRIDE = 6 * 4 * sizeof(float);
static constexpr uint32_t WF_PATHHIT_STRIDE= 2 * 4 * sizeof(float);
static constexpr uint32_t WF_SHADOW_STRIDE = 4 * 4 * sizeof(float);
static constexpr uint32_t WF_COUNTER_SIZE = 16 * sizeof(uint32_t);
static constexpr VkDeviceSize WF_OFF_EXTEND_ARGS = 16;
static constexpr VkDeviceSize WF_OFF_SHADE_ARGS  = 32;
static constexpr VkDeviceSize WF_OFF_SHADOW_ARGS = 48;

struct alignas(16) GPURenderData {
    Vec3 position;
    float size;
    uint32_t color;
    uint32_t materialIdx;
    int objectId;
    uint32_t extent = EXTENT_UNIT;
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
    int tileW;
    int tileH;
    int wcEnabled;
    uint32_t wcCapacity;
    uint32_t wcFrame;
    int wcMaxAge;
    Vec3 wcOrigin;
    float wcInvCellSize;
    Vec3 ddgiOrigin;
    float ddgiEnabled;
    Vec3 ddgiSpacing;
    float ddgiNormalBias;
    int ddgiProbesX;
    int ddgiProbesY;
    int ddgiProbesZ;
    int ddgiIrrRes;
    int ddgiDepthRes;
    float ddgiDepthSharpness;
};
#include "dispatchprobe.inl"

struct alignas(16) GPUWorldCacheEntry {
    uint32_t key;
    uint32_t frame;
    uint32_t sampleCount;
    uint32_t pad0;
    Vec3 irradiance;
    float pad1;
};

struct alignas(16) GPUReservoir {
    uint32_t key;
    uint32_t lightIdx;
    float wSum;
    float M;
    float W;
    float targetPdf;
    uint32_t frame;
    uint32_t nrmPacked;
    float posX;
    float posY;
    float posZ;
    float roughness;
    uint32_t albedoPacked;
    uint32_t metalPacked;
    float varLight;
    float varBsdf;
};

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

struct GpuContext {
    VkInstance instance = VK_NULL_HANDLE;
    int score = -1;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice primaryDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceType deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence postFence = VK_NULL_HANDLE;
    VkCommandBuffer wfCmd[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence wfFence[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    static constexpr uint32_t FRAME_SLOTS = 2;
    VkCommandBuffer frameCmd[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence frameFence[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    void* frameStagingMapped[FRAME_SLOTS] = {nullptr, nullptr};
    bool frameStagingCoherent[FRAME_SLOTS] = {true, true};
    uint32_t frameStagingCap[FRAME_SLOTS] = {0, 0};
    bool frameSlotInFlight[FRAME_SLOTS] = {false, false};
    uint32_t nextFrameSlot = 0;
    VkDeviceSize gbufferCap = 0;
    VkDeviceSize accumCap = 0;
    
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
    VkShaderModule aabbBuildShader = VK_NULL_HANDLE;
    VkShaderModule ddgiUpdateShader = VK_NULL_HANDLE;

    VkPipelineLayout fastPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout pbrPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout smoothPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout blendPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout guidedCoeffPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout wfPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout aabbBuildPipeLayout = VK_NULL_HANDLE;

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
    VkPipeline aabbBuildPipe = VK_NULL_HANDLE;
    VkPipeline ddgiUpdatePipe = VK_NULL_HANDLE;

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
    VkBuffer accumBuffer = VK_NULL_HANDLE;
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
    VkBuffer worldCacheBuffer = VK_NULL_HANDLE;
    VkBuffer ddgiIrradianceBuffer = VK_NULL_HANDLE;
    VkBuffer ddgiDepthBuffer = VK_NULL_HANDLE;
    VkBuffer frameStaging[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBuffer gbufferBuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout fastDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout smoothDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout blendDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout guidedCoeffDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout wfDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout aabbBuildLayout = VK_NULL_HANDLE;

    VkDescriptorSet wfDescSet = VK_NULL_HANDLE;
    VkDescriptorSet fastDescSet = VK_NULL_HANDLE;
    VkDescriptorSet pbrDescSet = VK_NULL_HANDLE;
    VkDescriptorSet smoothDescSet = VK_NULL_HANDLE;
    VkDescriptorSet blendDescSet = VK_NULL_HANDLE;
    VkDescriptorSet guidedCoeffDescSet = VK_NULL_HANDLE;
    VkDescriptorSet aabbBuildSet = VK_NULL_HANDLE;

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
    VkDeviceMemory worldCacheMem = VK_NULL_HANDLE;
    VkDeviceMemory ddgiIrradianceMem = VK_NULL_HANDLE;
    VkDeviceMemory ddgiDepthMem = VK_NULL_HANDLE;
    VkDeviceMemory frameStagingMem[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory gbufferMem = VK_NULL_HANDLE;
    VkDeviceMemory accumMem = VK_NULL_HANDLE;
    
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
    uint32_t worldCacheCap = 0;
    uint32_t ddgiProbeCap = 0;
    VkBuffer reservoirBuffer = VK_NULL_HANDLE;
    VkDeviceMemory reservoirMem = VK_NULL_HANDLE;
    uint32_t reservoirCap = 0;
    static constexpr int SVGF_HIST_STRIDE = 12;
    VkShaderModule svgfReprojectShader = VK_NULL_HANDLE;
    VkShaderModule svgfMomentsShader = VK_NULL_HANDLE;
    VkPipelineLayout svgfReprojectPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout svgfMomentsPipeLayout = VK_NULL_HANDLE;
    VkPipeline svgfReprojectPipe = VK_NULL_HANDLE;
    VkPipeline svgfMomentsPipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout svgfReprojectDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout svgfMomentsDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet svgfReprojectDescSet = VK_NULL_HANDLE;
    VkDescriptorSet svgfMomentsDescSet = VK_NULL_HANDLE;
    VkBuffer svgfHistBuffer[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory svgfHistMem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBuffer svgfVarBuffer[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory svgfVarMem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t svgfHistCap = 0;
    uint32_t svgfVarCap = 0;
    int svgfHistSlot = 0;
    bool svgfHistValid = false;
    uint32_t svgfWidth = 0, svgfHeight = 0;
    struct {
        float origin[3] = {0, 0, 0};
        float invDir[3] = {0, 0, 1};
        float invRight[3] = {1, 0, 0};
        float invUp[3] = {0, 1, 0};
        float tanfovx = 1.0f;
        float tanfovy = 1.0f;
    } svgfPrevCam;

    float svgfAlpha = 0.1f;
    float svgfMomentsAlpha = 0.1f;
    int svgfMaxHistory = 32;
    int svgfIterations = 7;
    bool svgfEnabled = true;

    bool initialized = false;
    bool blasTopologyValid = false;
    bool pbrPointsResident = false;
    bool fastPointsResident = false;
    bool postPassInFlight = false;
    bool outMemCoherent = true;
    bool outStagingCoherent = true;
    bool xferStagingCoherent = true;
    bool vctReady = false;
    bool aabbBuildReady = false;
    bool aabbBufferHostVisible = false;

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
    void ensureFrameSlot(uint32_t slot, VkDeviceSize outSize) {
        if (frameCmd[slot] == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = commandPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(device, &ai, &frameCmd[slot]);
        }
        if (frameFence[slot] == VK_NULL_HANDLE) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(device, &fi, nullptr, &frameFence[slot]);
        }
        if (outSize > frameStagingCap[slot]) {
            if (frameSlotInFlight[slot]) {
                vkWaitForFences(device, 1, &frameFence[slot], VK_TRUE, UINT64_MAX);
                frameSlotInFlight[slot] = false;
            }
            if (frameStaging[slot]) {
                vkUnmapMemory(device, frameStagingMem[slot]);
                vkDestroyBuffer(device, frameStaging[slot], nullptr);
                vkFreeMemory(device, frameStagingMem[slot], nullptr);
                frameStagingMapped[slot] = nullptr;
            }
            createReadbackBuffer(device, primaryDevice, outSize, frameStaging[slot],
                                 frameStagingMem[slot], frameStagingCoherent[slot]);
            vkMapMemory(device, frameStagingMem[slot], 0, VK_WHOLE_SIZE, 0, &frameStagingMapped[slot]);
            frameStagingCap[slot] = (uint32_t)outSize;
        }
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
        svgfReprojectShader = createShaderModule(device, "./bin/svgf_reproject.spv");
        svgfMomentsShader = createShaderModule(device, "./bin/svgf_moments.spv");
    }

    void init(VkPhysicalDevice PhDevice = VK_NULL_HANDLE, VkInstance sharedInstance = VK_NULL_HANDLE) {
        uint32_t extCount;
        if (initialized) return;

        if (sharedInstance != VK_NULL_HANDLE) {
            instance = sharedInstance;
        } else if (instance == VK_NULL_HANDLE) {
            VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            appInfo.apiVersion = VK_API_VERSION_1_2;
            vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
            VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            createInfo.pApplicationInfo = &appInfo;
            vkCreateInstance(&createInfo, nullptr, &instance);
        }

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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 12},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 6},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}
        };
        VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolCreateInfo.poolSizeCount = 5;
        poolCreateInfo.pPoolSizes = poolSizes;
        poolCreateInfo.maxSets = 48;
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

        VkDescriptorSetLayoutBinding smBindings[7] = {};
        for(int i=0; i<7; i++) {
            smBindings[i].binding = i;
            smBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            smBindings[i].descriptorCount = 1;
            smBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo smLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, smBindings};
        vkCreateDescriptorSetLayout(device, &smLayoutInfo, nullptr, &smoothDescLayout);

        VkDescriptorSetLayoutBinding blBindings[5] = {};
        for(int i=0; i<5; i++) {
            blBindings[i].binding = i;
            blBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            blBindings[i].descriptorCount = 1;
            blBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo blLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, blBindings};
        vkCreateDescriptorSetLayout(device, &blLayoutInfo, nullptr, &blendDescLayout);

        // Guided-filter coefficient pass: guide (full), PT (low), coeff out, gbuffer.
        VkDescriptorSetLayoutCreateInfo gcLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, blBindings};
        vkCreateDescriptorSetLayout(device, &gcLayoutInfo, nullptr, &guidedCoeffDescLayout);

        // SVGF reproject: 7 storage buffers + the current camera UBO at binding 7.
        VkDescriptorSetLayoutBinding svBindings[8] = {};
        for(int i=0; i<8; i++) {
            svBindings[i].binding = i;
            svBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            svBindings[i].descriptorCount = 1;
            svBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        svBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        VkDescriptorSetLayoutCreateInfo srLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 8, svBindings};
        vkCreateDescriptorSetLayout(device, &srLayoutInfo, nullptr, &svgfReprojectDescLayout);

        svBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorSetLayoutCreateInfo smtLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 6, svBindings};
        vkCreateDescriptorSetLayout(device, &smtLayoutInfo, nullptr, &svgfMomentsDescLayout);

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
        allocSetInfo.pSetLayouts = &svgfReprojectDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &svgfReprojectDescSet);
        allocSetInfo.pSetLayouts = &svgfMomentsDescLayout;
        vkAllocateDescriptorSets(device, &allocSetInfo, &svgfMomentsDescSet);

        createAllShaderModules();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &fastDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &fastPipelineLayout);
        pipelineLayoutInfo.pSetLayouts = &pbrDescLayout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pbrPipelineLayout);

        VkPushConstantRange smPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 8};
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

        VkPushConstantRange srPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 16 + sizeof(int) * 5};
        pipelineLayoutInfo.pSetLayouts = &svgfReprojectDescLayout;
        pipelineLayoutInfo.pPushConstantRanges = &srPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &svgfReprojectPipeLayout);

        VkPushConstantRange smtPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 2 + sizeof(float) * 3};
        pipelineLayoutInfo.pSetLayouts = &svgfMomentsDescLayout;
        pipelineLayoutInfo.pPushConstantRanges = &smtPush;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &svgfMomentsPipeLayout);

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
        if (svgfReprojectShader) {
            computePipelineInfo.layout = svgfReprojectPipeLayout;
            computePipelineInfo.stage.module = svgfReprojectShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &svgfReprojectPipe);
        }
        if (svgfMomentsShader) {
            computePipelineInfo.layout = svgfMomentsPipeLayout;
            computePipelineInfo.stage.module = svgfMomentsShader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &svgfMomentsPipe);
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

    void destroyFrameSlots() {
        for (uint32_t i = 0; i < FRAME_SLOTS; ++i) {
            if (frameSlotInFlight[i] && frameFence[i] != VK_NULL_HANDLE) {
                vkWaitForFences(device, 1, &frameFence[i], VK_TRUE, UINT64_MAX);
                frameSlotInFlight[i] = false;
            }
            if (frameStaging[i]) {
                vkUnmapMemory(device, frameStagingMem[i]);
                vkDestroyBuffer(device, frameStaging[i], nullptr);
                vkFreeMemory(device, frameStagingMem[i], nullptr);
                frameStaging[i] = VK_NULL_HANDLE;
                frameStagingMapped[i] = nullptr;
                frameStagingCap[i] = 0;
            }
            if (frameFence[i] != VK_NULL_HANDLE) {
                vkDestroyFence(device, frameFence[i], nullptr);
                frameFence[i] = VK_NULL_HANDLE;
            }
            if (frameCmd[i] != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, commandPool, 1, &frameCmd[i]);
                frameCmd[i] = VK_NULL_HANDLE;
            }
        }
        nextFrameSlot = 0;
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

    void initAabbBuildPipeline() {
        if (aabbBuildReady) return;

        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, b};
        vkCreateDescriptorSetLayout(device, &li, nullptr, &aabbBuildLayout);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &aabbBuildLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(device, &pl, nullptr, &aabbBuildPipeLayout);

        aabbBuildShader = createShaderModule(device, "./bin/aabb_build.spv");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.pName = "main";
        ci.stage.module = aabbBuildShader;
        ci.layout = aabbBuildPipeLayout;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &aabbBuildPipe);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &aabbBuildLayout;
        vkAllocateDescriptorSets(device, &ai, &aabbBuildSet);

        aabbBuildReady = true;
    }

    ///@brief Records the AABB fill into cmd; the caller submits it with the AS build.
    void recordAabbBuild(VkCommandBuffer cmd, VkBuffer srcPoints, uint32_t numPrimitives) {
        initAabbBuildPipeline();

        VkDescriptorBufferInfo bInfos[2] = {
            {srcPoints,  0, VK_WHOLE_SIZE},
            {aabbBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[2]{};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = aabbBuildSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aabbBuildPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aabbBuildPipeLayout,
                                0, 1, &aabbBuildSet, 0, nullptr);
        vkCmdPushConstants(cmd, aabbBuildPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(uint32_t), &numPrimitives);
        vkCmdDispatch(cmd, (numPrimitives + 63) / 64, 1, 1);

        // AABB writes must land before the AS build reads them.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void buildHardwareAccelerationStructures(const std::vector<GPURenderData>& points, int orderingTag = 0,
                                             VkBuffer srcPointBuffer = VK_NULL_HANDLE) {
        if (points.empty()) return;

        const uint32_t numPrimitives = static_cast<uint32_t>(points.size());
        bool doFullBuild = (!blasTopologyValid) || (numPrimitives != lastBlasPrimCount)
                        || (orderingTag != lastBlasOrderingTag);

        uint32_t aabbSize = numPrimitives * sizeof(VkAabbPositionsKHR);
        const bool gpuFill = (srcPointBuffer != VK_NULL_HANDLE);
        if (aabbSize > currentAabbCap || aabbBufferHostVisible == gpuFill || !aabbBuffer) {
            if (aabbBuffer) {
                vkDestroyBuffer(device, aabbBuffer, nullptr);
                vkFreeMemory(device, aabbMem, nullptr);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VkMemoryPropertyFlags props = gpuFill
                ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            createBufferWithAddress(device, aabbSize, usage, props, aabbBuffer, aabbMem);
            currentAabbCap = aabbSize;
            aabbBufferHostVisible = !gpuFill;
            blasTopologyValid = false;
            doFullBuild = true;
        }

        if (!gpuFill) {
            std::vector<VkAabbPositionsKHR> aabbs(numPrimitives);
            for (uint32_t i = 0; i < numPrimitives; ++i) {
                const float halfSize = points[i].size * 0.5f;
                const Vec3 lo = points[i].position - Vec3::Constant(halfSize);
                const Vec3 hi = points[i].position + Vec3::Constant(halfSize)
                    + Vec3::Constant(points[i].size).cwiseProduct(
                        unpackExtent(points[i].extent) - Vec3::Ones());
                aabbs[i].minX = lo.x();
                aabbs[i].minY = lo.y();
                aabbs[i].minZ = lo.z();
                aabbs[i].maxX = hi.x();
                aabbs[i].maxY = hi.y();
                aabbs[i].maxZ = hi.z();
            }
            void* data;
            vkMapMemory(device, aabbMem, 0, aabbSize, 0, &data);
            memcpy(data, aabbs.data(), aabbSize);
            vkUnmapMemory(device, aabbMem);
        }

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
        
        void* data;
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
            if (gpuFill) recordAabbBuild(cmd, srcPointBuffer, numPrimitives);
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

    ///@brief Records and submits the fast pipeline over the whole frame, no wait.
    ///@return The slot index this frame was submitted on; pass it to awaitFastFullFrame.
    uint32_t submitFastFullFrame(int width, int height, VkDeviceSize outSize) {
        const uint32_t slot = nextFrameSlot;
        nextFrameSlot = (nextFrameSlot + 1) % FRAME_SLOTS;

        ensureFrameSlot(slot, outSize);

        if (frameSlotInFlight[slot]) {
            vkWaitForFences(device, 1, &frameFence[slot], VK_TRUE, UINT64_MAX);
            frameSlotInFlight[slot] = false;
        }
        vkResetFences(device, 1, &frameFence[slot]);

        VkCommandBuffer cmd = frameCmd[slot];
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fastPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fastPipelineLayout,
                                0, 1, &fastDescSet, 0, nullptr);
        vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

        VkBufferMemoryBarrier toXfer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        toXfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toXfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toXfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toXfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toXfer.buffer = outBuffer;
        toXfer.offset = 0;
        toXfer.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &toXfer, 0, nullptr);

        VkBufferCopy copyRegion{};
        copyRegion.size = outSize;
        vkCmdCopyBuffer(cmd, outBuffer, frameStaging[slot], 1, &copyRegion);

        VkBufferMemoryBarrier toHost{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.buffer = frameStaging[slot];
        toHost.offset = 0;
        toHost.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0,
                             0, nullptr, 1, &toHost, 0, nullptr);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submitInfo, frameFence[slot]);
        frameSlotInFlight[slot] = true;
        return slot;
    }

    void awaitFastFullFrame(uint32_t slot) {
        if (slot >= FRAME_SLOTS || !frameSlotInFlight[slot]) return;
        vkWaitForFences(device, 1, &frameFence[slot], VK_TRUE, UINT64_MAX);
        frameSlotInFlight[slot] = false;
    }

    void awaitAllFastFrames() {
        for (uint32_t i = 0; i < FRAME_SLOTS; ++i) awaitFastFullFrame(i);
    }

    const float* readbackSlot(uint32_t slot) {
        if (slot >= FRAME_SLOTS || !frameStagingMapped[slot]) return nullptr;
        if (!frameStagingCoherent[slot]) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = frameStagingMem[slot];
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(device, 1, &range);
        }
        return static_cast<const float*>(frameStagingMapped[slot]);
    }

    void dispatchFastFullFrame(int width, int height, VkDeviceSize outSize) {
        awaitFastFullFrame(submitFastFullFrame(width, height, outSize));
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
        ensureGBuffer(uint32_t(camData.width), uint32_t(camData.height));
        
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

    ///@brief Ensure accumBuffer can hold `size` bytes (device-local, copy src/dst).
    void ensureAccumBuffer(VkDeviceSize size) {
        if (size <= accumCap && accumBuffer) return;
        if (accumBuffer) {
            vkDestroyBuffer(device, accumBuffer, nullptr);
            vkFreeMemory(device, accumMem, nullptr);
        }
        createBuffer(device, primaryDevice, size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accumBuffer, accumMem);
        accumCap = size;
    }

    ///@brief Park the raw radiance sum currently in outBuffer for reuse next frame.
    void saveAccum(VkDeviceSize size) {
        ensureAccumBuffer(size);
        copyBuffer(device, outBuffer, accumBuffer, size);
    }

    ///@brief Restore a previously parked sum into outBuffer so wf_finalize adds onto it.
    void restoreAccum(VkDeviceSize size) {
        if (!accumBuffer || size > accumCap) return;
        copyBuffer(device, accumBuffer, outBuffer, size);
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

        buildHardwareAccelerationStructures(points, 1, fastPointBuffer);

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
            vw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            vw[0].dstSet = pbrDescSet;
            vw[0].dstBinding = 9;
            vw[0].descriptorCount = 1;
            vw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            vw[0].pImageInfo = &si;
            vw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            vw[1].dstSet = pbrDescSet;
            vw[1].dstBinding = 10;
            vw[1].descriptorCount = 1;
            vw[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            vw[1].pBufferInfo = &bi;
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

    void dispatchSmoothPasses(int width, int height, int samples, int iters, bool toFinal, bool deferFinalWait = false, int useAlbedo = 1,
                              int varMode = 0, VkBuffer histFeedback = VK_NULL_HANDLE) {
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

        // Variance ping-pong for the variance-guided luminance weight.
        uint32_t varSize = width * height * sizeof(float);
        if (varSize > svgfVarCap) {
            for (int i = 0; i < 2; ++i) {
                destroyBuffer(device, svgfVarBuffer[i], svgfVarMem[i]);
                createBuffer(device, primaryDevice, varSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, svgfVarBuffer[i], svgfVarMem[i]);
            }
            svgfVarCap = varSize;
        }

        VkBuffer src = outBuffer;
        VkBuffer dst = smoothScratchBuffer;

        for (int it = 0; it < iters; ++it) {
            bool finalPass = toFinal && (it == iters - 1);
            int step = 1 << it;
            VkBuffer outBuf = finalPass ? finalOutBuffer : dst;

            VkDescriptorBufferInfo bInfos[7] = {
                {src, 0, VK_WHOLE_SIZE},
                {outBuf, 0, VK_WHOLE_SIZE},
                {adaptiveBuffer, 0, VK_WHOLE_SIZE},
                {gbufferBuffer ? gbufferBuffer : adaptiveBuffer, 0, VK_WHOLE_SIZE},
                {svgfVarBuffer[it % 2], 0, VK_WHOLE_SIZE},
                {svgfVarBuffer[(it + 1) % 2], 0, VK_WHOLE_SIZE},
                {histFeedback ? histFeedback : adaptiveBuffer, 0, VK_WHOLE_SIZE}
            };
            VkWriteDescriptorSet writes[7] = {};
            for(int i=0; i<7; i++) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = smoothDescSet;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &bInfos[i];
            }
            vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, smoothPipelineLayout, 0, 1, &smoothDescSet, 0, nullptr);

            struct { int w, h, s, step, finalPass, useAlbedo, varMode, feedbackHist; } pc = {
                width, height, samples, step, finalPass ? 1 : 0, useAlbedo,
                varMode, (histFeedback && it == 0) ? 1 : 0
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
            if (finalPass && deferFinalWait) {
                awaitPostPass();
                if (postFence != VK_NULL_HANDLE) vkDestroyFence(device, postFence, nullptr);
                postFence = fence;
                postPassInFlight = true;
            } else {
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
                vkDestroyFence(device, fence, nullptr);
            }

            if (!finalPass) {
                VkBuffer tmp = src;
                src = dst;
                dst = tmp;
            }
        }
    }

    void dispatchSmooth(int width, int height, int samples) {
        dispatchSmoothPasses(width, height, samples, 4, true);
    }

    ///@brief dispatchSmooth that returns before the final pass completes.
    void submitSmooth(int width, int height, int samples) {
        dispatchSmoothPasses(width, height, samples, 4, true, true);
    }

    void ensureSVGFBuffers(int width, int height) {
        uint32_t histSize = width * height * SVGF_HIST_STRIDE * sizeof(float);
        if (svgfWidth != uint32_t(width) || svgfHeight != uint32_t(height)) {
            svgfHistValid = false;
            svgfWidth = width;
            svgfHeight = height;
        }
        if (histSize <= svgfHistCap) return;
        for (int i = 0; i < 2; ++i) {
            destroyBuffer(device, svgfHistBuffer[i], svgfHistMem[i]);
            createBuffer(device, primaryDevice, histSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, svgfHistBuffer[i], svgfHistMem[i]);
        }
        svgfHistCap = histSize;
        svgfHistValid = false;
        // Zeroed history means historyLength == 0, which every validity test rejects.
        executeSingleTimeCommands([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, svgfHistBuffer[0], 0, histSize, 0u);
            vkCmdFillBuffer(cmd, svgfHistBuffer[1], 0, histSize, 0u);
        });
    }

    ///@brief Drops the temporal history, e.g. on a scene or camera cut.
    void resetSVGF() { svgfHistValid = false; }

    ///@brief Runs a compute pass on `commandBuffer` and waits for it.
    void runSVGFPass(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                     const void* push, uint32_t pushSize, int width, int height) {
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, push);
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

    ///@brief Temporal half of SVGF: reprojection and moment accumulation.
    ///
    /// Leaves filtered colour in outBuffer and variance in svgfVarBuffer[0],
    /// then hands off to dispatchSmoothPasses for the a-trous iterations.
    ///@return false if the SVGF pipelines are unavailable and nothing was done.
    bool submitSVGF(int width, int height, int samples, const GPUCameraData& camData) {
        if (!svgfReprojectPipe || !svgfMomentsPipe || !smoothPipeline) return false;
        if (width <= 0 || height <= 0) return false;

        int iters = std::clamp(svgfIterations, 1, 8);
        dispatchSmoothPasses(width, height, samples, 0, false); // allocate scratch/variance
        ensureSVGFBuffers(width, height);
        if (!smoothScratchBuffer) return false;

        int histOutSlot = svgfHistSlot;
        int histInSlot = 1 - svgfHistSlot;
        VkBuffer histIn = svgfHistBuffer[histInSlot];
        VkBuffer histOut = svgfHistBuffer[histOutSlot];
        VkBuffer gbuf = gbufferBuffer ? gbufferBuffer : adaptiveBuffer;

        auto writeSet = [&](VkDescriptorSet set, const VkDescriptorBufferInfo* infos, int count, int uboBinding) {
            VkWriteDescriptorSet w[8] = {};
            for (int i = 0; i < count; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = set;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = (i == uboBinding) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, count, w, 0, nullptr);
        };

        // Reproject reads outBuffer and writes the scratch; it samples neighbours
        // so it cannot write outBuffer in place.
        VkDescriptorBufferInfo reInfos[8] = {
            {outBuffer, 0, VK_WHOLE_SIZE},
            {gbuf, 0, VK_WHOLE_SIZE},
            {histIn, 0, VK_WHOLE_SIZE},
            {histOut, 0, VK_WHOLE_SIZE},
            {smoothScratchBuffer, 0, VK_WHOLE_SIZE},
            {adaptiveBuffer, 0, VK_WHOLE_SIZE},
            {svgfVarBuffer[1], 0, VK_WHOLE_SIZE},
            {uboBuffer, 0, VK_WHOLE_SIZE}
        };
        writeSet(svgfReprojectDescSet, reInfos, 8, 7);

        struct {
            float prevOx, prevOy, prevOz;
            float prevIDx, prevIDy, prevIDz;
            float prevIRx, prevIRy, prevIRz;
            float prevIUx, prevIUy, prevIUz;
            float prevTanX, prevTanY, alpha, momentsAlpha;
            int w, h, s, reset, maxHist;
        } rp = {
            svgfPrevCam.origin[0], svgfPrevCam.origin[1], svgfPrevCam.origin[2],
            svgfPrevCam.invDir[0], svgfPrevCam.invDir[1], svgfPrevCam.invDir[2],
            svgfPrevCam.invRight[0], svgfPrevCam.invRight[1], svgfPrevCam.invRight[2],
            svgfPrevCam.invUp[0], svgfPrevCam.invUp[1], svgfPrevCam.invUp[2],
            svgfPrevCam.tanfovx, svgfPrevCam.tanfovy, svgfAlpha, svgfMomentsAlpha,
            width, height, samples, svgfHistValid ? 0 : 1, std::max(1, svgfMaxHistory)
        };
        runSVGFPass(svgfReprojectPipe, svgfReprojectPipeLayout, svgfReprojectDescSet,
                    &rp, sizeof(rp), width, height);

        // Moments moves colour back into outBuffer so the a-trous chain starts
        // where dispatchSmoothPasses always starts.
        VkDescriptorBufferInfo moInfos[6] = {
            {smoothScratchBuffer, 0, VK_WHOLE_SIZE},
            {outBuffer, 0, VK_WHOLE_SIZE},
            {histOut, 0, VK_WHOLE_SIZE},
            {gbuf, 0, VK_WHOLE_SIZE},
            {svgfVarBuffer[1], 0, VK_WHOLE_SIZE},
            {svgfVarBuffer[0], 0, VK_WHOLE_SIZE}
        };
        writeSet(svgfMomentsDescSet, moInfos, 6, -1);

        struct { int w, h; float phiNormal, phiDepth, histThreshold; } mp = {
            width, height, 64.0f, 1.0f, 4.0f
        };
        runSVGFPass(svgfMomentsPipe, svgfMomentsPipeLayout, svgfMomentsDescSet,
                    &mp, sizeof(mp), width, height);

        // samples == 1: reproject already divided by the sample count.
        dispatchSmoothPasses(width, height, 1, iters, true, true, 1, 1, histOut);

        // Retain the dual basis of this frame's camera for next frame.
        Vec3 d = camData.dir, r = camData.right, u = camData.up;
        Vec3 rXu = r.cross(u);
        float det = d.dot(rXu);
        bool ok = std::abs(det) > 1e-12f;
        if (ok) {
            float inv = 1.0f / det;
            Vec3 iD = rXu * inv, iR = u.cross(d) * inv, iU = d.cross(r) * inv;
            for (int i = 0; i < 3; ++i) {
                svgfPrevCam.origin[i] = camData.origin[i];
                svgfPrevCam.invDir[i] = iD[i];
                svgfPrevCam.invRight[i] = iR[i];
                svgfPrevCam.invUp[i] = iU[i];
            }
            svgfPrevCam.tanfovx = camData.tanfovx;
            svgfPrevCam.tanfovy = camData.tanfovy;
        }
        svgfHistSlot = histInSlot;
        svgfHistValid = ok;
        return true;
    }

    ///@brief Guided-filter blend of the PT and guide buffers into finalOutBuffer.
    ///@param deferWait Return before the pass completes; collect with awaitPostPass
    void dispatchBlend(int width, int height, int lowW, int lowH, float pbrScale, int samples,
                       bool deferWait = false) {
        uint32_t finalSize = width * height * 3 * sizeof(float);
        if(finalSize > currentFinalOutCap) {
            if(finalOutBuffer) { 
                vkDestroyBuffer(device, finalOutBuffer, nullptr);
                vkFreeMemory(device, finalOutMem, nullptr);
            }
            createBuffer(device, primaryDevice, finalSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, finalOutBuffer, finalOutMem);
            currentFinalOutCap = finalSize;
        }

        uint32_t coeffSize = uint32_t(lowW) * uint32_t(lowH) * 24 * sizeof(float);
        if (coeffSize > currentGuidedCoeffCap) {
            if (guidedCoeffBuffer) {
                vkDestroyBuffer(device, guidedCoeffBuffer, nullptr);
                vkFreeMemory(device, guidedCoeffMem, nullptr);
            }
            createBuffer(device, primaryDevice, coeffSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, guidedCoeffBuffer, guidedCoeffMem);
            currentGuidedCoeffCap = coeffSize;
        }

        VkBuffer gbuf = gbufferBuffer ? gbufferBuffer : adaptiveBuffer;
        VkDescriptorBufferInfo gcInfos[4] = { {outBuffer, 0, VK_WHOLE_SIZE}, {lowResOutBuffer, 0, VK_WHOLE_SIZE}, {guidedCoeffBuffer, 0, VK_WHOLE_SIZE}, {gbuf, 0, VK_WHOLE_SIZE} };
        VkDescriptorBufferInfo bInfos[5]  = { {outBuffer, 0, VK_WHOLE_SIZE}, {lowResOutBuffer, 0, VK_WHOLE_SIZE}, {finalOutBuffer, 0, VK_WHOLE_SIZE}, {guidedCoeffBuffer, 0, VK_WHOLE_SIZE}, {gbuf, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[9] = {};
        for(int i=0; i<4; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = guidedCoeffDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &gcInfos[i];
        }
        for(int i=0; i<5; i++) {
            writes[4 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4 + i].dstSet = blendDescSet;
            writes[4 + i].dstBinding = i;
            writes[4 + i].descriptorCount = 1;
            writes[4 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[4 + i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);

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

        if (!deferWait) {
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence fence;
            vkCreateFence(device, &fenceInfo, nullptr, &fence);
            vkQueueSubmit(queue, 1, &submitInfo, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            return;
        }

        if (postFence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(device, &fenceInfo, nullptr, &postFence);
        } else {
            awaitPostPass();
            vkResetFences(device, 1, &postFence);
        }
        vkQueueSubmit(queue, 1, &submitInfo, postFence);
        postPassInFlight = true;
    }

    ///@brief Waits on the trailing smooth/blend submit, if one is outstanding.
    void awaitPostPass() {
        if (!postPassInFlight) return;
        vkWaitForFences(device, 1, &postFence, VK_TRUE, UINT64_MAX);
        postPassInFlight = false;
    }

struct WFPushConstants {
    int parity;
    int stage;
    int sampleIndex;
    int pad;
};


void initWavefront() {
    VkDescriptorSetLayoutBinding b[22] = {};
    for (int i = 0; i < 22; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 22;
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

    wfInitShader = createShaderModule(device, "./bin/wf_init.spv");
    wfArgsShader = createShaderModule(device, "./bin/wf_args.spv");
    wfExtendShader = createShaderModule(device, "./bin/wf_extend.spv");
    wfShadeShader = createShaderModule(device, "./bin/wf_shade.spv");
    wfShadowShader = createShaderModule(device, "./bin/wf_shadow.spv");
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

    ddgiUpdateShader = createShaderModule(device, "./bin/ddgi_update.spv");
    if (ddgiUpdateShader != VK_NULL_HANDLE) makePipe(ddgiUpdateShader, ddgiUpdatePipe);
}

void ensureWorldCache(uint32_t capacity) {
    uint32_t cap = 1u;
    while (cap < capacity) cap <<= 1;
    if (worldCacheBuffer && cap <= worldCacheCap) return;
    destroyBuffer(device, worldCacheBuffer, worldCacheMem);
    const VkDeviceSize bytes = VkDeviceSize(cap) * sizeof(GPUWorldCacheEntry);
    createBuffer(device, primaryDevice, bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, worldCacheBuffer, worldCacheMem);
    worldCacheCap = cap;
    clearWorldCache();
}

void clearWorldCache() {
    if (!worldCacheBuffer) return;
    const VkDeviceSize bytes = VkDeviceSize(worldCacheCap) * sizeof(GPUWorldCacheEntry);
    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, worldCacheBuffer, 0, bytes, 0u);
    });
}

void ensureReservoirs(uint32_t capacity) {
    uint32_t cap = 1u;
    while (cap < capacity) cap <<= 1;
    if (reservoirBuffer && cap <= reservoirCap) return;
    destroyBuffer(device, reservoirBuffer, reservoirMem);
    const VkDeviceSize bytes = VkDeviceSize(cap) * sizeof(GPUReservoir);
    createBuffer(device, primaryDevice, bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, reservoirBuffer, reservoirMem);
    reservoirCap = cap;
    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, reservoirBuffer, 0, bytes, 0u);
    });
}

void ensureGBuffer(uint32_t fullW, uint32_t fullH) {
    VkDeviceSize bytes = VkDeviceSize(fullW) * VkDeviceSize(fullH) * 8 * sizeof(float);
    if (bytes == 0) return;
    if (gbufferBuffer && bytes <= gbufferCap) return;
    destroyBuffer(device, gbufferBuffer, gbufferMem);
    createBuffer(device, primaryDevice, bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gbufferBuffer, gbufferMem);
    gbufferCap = bytes;
    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, gbufferBuffer, 0, bytes, 0u);
    });
}

void ensureDDGIBuffers(uint32_t probeCount) {
    if (ddgiIrradianceBuffer && probeCount <= ddgiProbeCap) return;
    destroyBuffer(device, ddgiIrradianceBuffer, ddgiIrradianceMem);
    destroyBuffer(device, ddgiDepthBuffer, ddgiDepthMem);

    const VkDeviceSize irrBytes = VkDeviceSize(probeCount) * DDGI_IRR_RES * DDGI_IRR_RES * 4 * sizeof(float);
    const VkDeviceSize depthBytes = VkDeviceSize(probeCount) * DDGI_DEPTH_RES * DDGI_DEPTH_RES * 2 * sizeof(float);
    const VkBufferUsageFlags use = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    createBuffer(device, primaryDevice, irrBytes, use, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 ddgiIrradianceBuffer, ddgiIrradianceMem);
    createBuffer(device, primaryDevice, depthBytes, use, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 ddgiDepthBuffer, ddgiDepthMem);
    ddgiProbeCap = probeCount;

    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, ddgiIrradianceBuffer, 0, irrBytes, 0u);
        vkCmdFillBuffer(cmd, ddgiDepthBuffer, 0, depthBytes, 0u);
    });
}

void dispatchDDGIUpdate(uint32_t probeCount) {
    if (!ddgiUpdatePipe || probeCount == 0) return;
    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ddgiUpdatePipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                wfPipelineLayout, 0, 1, &wfDescSet, 0, nullptr);
        WFPushConstants pc{0, 0, 0, 0};
        vkCmdPushConstants(cmd, wfPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, probeCount, 1, 1);
    });
}

void ensureWavefrontBuffers(uint32_t maxPaths) {
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

    createBuffer(device, primaryDevice, maxPaths * WF_PATH_STRIDE, store, devLocal, wfPathBuf, wfPathMem);
    createBuffer(device, primaryDevice, maxPaths * WF_PATHHIT_STRIDE, store, devLocal, wfPathHitBuf, wfPathHitMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfExtendABuf, wfExtendAMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfExtendBBuf, wfExtendBMem);
    createBuffer(device, primaryDevice, maxPaths * sizeof(uint32_t), store, devLocal, wfShadeBuf, wfShadeMem);
    createBuffer(device, primaryDevice, maxPaths * WF_SHADOW_STRIDE, store, devLocal, wfShadowBuf, wfShadowMem);
    createBuffer(device, primaryDevice, WF_COUNTER_SIZE, store | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, devLocal, wfCounterBuf, wfCounterMem);
    wfPathCap = maxPaths;
}

void writeWavefrontDescriptors() {
    VkDescriptorBufferInfo bi[22] = {};
    bi[0]  = {uboBuffer, 0, VK_WHOLE_SIZE};
    bi[1]  = {pbrPointBuffer, 0, VK_WHOLE_SIZE};
    bi[2]  = {materialBuffer, 0, VK_WHOLE_SIZE};
    bi[3]  = {skyboxBuffer, 0, VK_WHOLE_SIZE};
    bi[4]  = {lightBuffer, 0, VK_WHOLE_SIZE};
    bi[6]  = {outBuffer, 0, VK_WHOLE_SIZE};
    bi[7]  = {adaptiveBuffer, 0, VK_WHOLE_SIZE};
    bi[8]  = {wfPathBuf, 0, VK_WHOLE_SIZE};
    bi[9]  = {wfExtendABuf, 0, VK_WHOLE_SIZE};
    bi[10] = {wfExtendBBuf, 0, VK_WHOLE_SIZE};
    bi[11] = {wfShadeBuf, 0, VK_WHOLE_SIZE};
    bi[12] = {wfShadowBuf, 0, VK_WHOLE_SIZE};
    bi[13] = {wfCounterBuf, 0, VK_WHOLE_SIZE};
    bi[14] = {wfPathHitBuf, 0, VK_WHOLE_SIZE};
    bi[15] = {sellmeierBuffer ? sellmeierBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[16] = {fogBuffer ? fogBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[17] = {worldCacheBuffer ? worldCacheBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[18] = {ddgiIrradianceBuffer ? ddgiIrradianceBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[19] = {ddgiDepthBuffer ? ddgiDepthBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[20] = {reservoirBuffer ? reservoirBuffer : materialBuffer, 0, VK_WHOLE_SIZE};
    bi[21] = {gbufferBuffer ? gbufferBuffer : materialBuffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet w[22] = {};
    int n = 0;
    for (int i = 0; i < 22; ++i) {
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

void wfBufBarrier(VkCommandBuffer cmd,
                  const VkBuffer* buffers, uint32_t count,
                  VkAccessFlags src, VkAccessFlags dst,
                  VkPipelineStageFlags dstStage) {
    VkBufferMemoryBarrier bmb[12];
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && n < 12; ++i) {
        if (buffers[i] == VK_NULL_HANDLE) continue;
        bmb[n] = VkBufferMemoryBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb[n].srcAccessMask = src;
        bmb[n].dstAccessMask = dst;
        bmb[n].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb[n].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb[n].buffer = buffers[i];
        bmb[n].offset = 0;
        bmb[n].size = VK_WHOLE_SIZE;
        ++n;
    }
    if (n == 0) return;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStage,
        0, 0, nullptr, n, bmb, 0, nullptr);
}

void wfBarrierArgs(VkCommandBuffer cmd) {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = wfCounterBuf;
    bmb.offset = 0;
    bmb.size = VK_WHOLE_SIZE;
    if (wfCounterBuf == VK_NULL_HANDLE) return;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);
}

void wfBarrierData(VkCommandBuffer cmd) {
    const VkBuffer bufs[9] = {
        wfPathBuf,  wfPathHitBuf,  wfExtendABuf, wfExtendBBuf,
        wfShadeBuf, wfShadowBuf,   gbufferBuffer,
        outBuffer,  adaptiveBuffer
    };
    wfBufBarrier(cmd, bufs, 9,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkBuffer ctr[1] = { wfCounterBuf };
    wfBufBarrier(cmd, ctr, 1,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
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
    // TIME_FUNCTION;
    uint32_t maxPaths = uint32_t(tileW) * uint32_t(tileH);
    if (maxPaths == 0 || samplesPerPixel <= 0) return;
    ensureWavefrontBuffers(maxPaths);
    ensureWorldCache(WC_CAPACITY);
    ensureReservoirs(WC_CAPACITY);
    ensureDDGIBuffers(uint32_t(DDGI_PROBES_X * DDGI_PROBES_Y * DDGI_PROBES_Z));
    writeWavefrontDescriptors();

    const uint32_t WG = 64;
    uint32_t pathGroups = uint32_t((maxPaths + WG - 1) / WG);
    int maxIters = maxBounces + 2;

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
        vkWaitForFences(device, 1, &wfFence[slot], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &wfFence[slot]);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                wfPipelineLayout, 0, 1, &wfDescSet, 0, nullptr);

        for (int s = s0; s < s1; ++s) {
            wfBind(cmd, wfArgsPipe);
            wfPush(cmd, 0, 4, s);
            vkCmdDispatch(cmd, 1, 1, 1);
            wfBarrierData(cmd);
            wfBind(cmd, wfInitPipe);
            wfPush(cmd, 0, 0, s);
            vkCmdDispatch(cmd, pathGroups, 1, 1);
            wfBarrierData(cmd);
            wfBind(cmd, wfArgsPipe);
            wfPush(cmd, 0, 0, s);
            vkCmdDispatch(cmd, 1, 1, 1);
            wfBarrierArgs(cmd);

            int parity = 0;
            for (int it = 0; it < maxIters; ++it) {
                wfBind(cmd, wfExtendPipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_EXTEND_ARGS);
                wfBarrierData(cmd);
                wfBind(cmd, wfArgsPipe);
                wfPush(cmd, parity, 1, s);
                vkCmdDispatch(cmd, 1, 1, 1);
                wfBarrierArgs(cmd);
                wfBind(cmd, wfShadePipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADE_ARGS);
                wfBarrierData(cmd);
                wfBind(cmd, wfArgsPipe);
                wfPush(cmd, parity, 5, s);
                vkCmdDispatch(cmd, 1, 1, 1);
                wfBarrierArgs(cmd);
                wfBind(cmd, wfShadowPipe);
                wfPush(cmd, parity, 0, s);
                vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADOW_ARGS);
                wfBarrierData(cmd);
                parity ^= 1;
            }

            wfBind(cmd, wfFinalizePipe);
            wfPush(cmd, parity, 0, s);
            vkCmdDispatch(cmd, pathGroups, 1, 1);
            wfBarrierData(cmd);
        }

        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, wfFence[slot]);
    }
    vkWaitForFences(device, 2, wfFence, VK_TRUE, UINT64_MAX);

}

TileProfile tileProfile;

double probeTileEdge(int edge, const GPUCameraData& cam) {
    GPUCameraData cd = cam;
    cd.tileOffsetX = 0;
    cd.tileOffsetY = 0;
    cd.tileW = edge;
    cd.tileH = edge;
    cd.currentSampleOffset = 0;
    cd.dispatchSamples = 1;
    updateCameraData(cd);

    for (int i = 0; i < PROBE_WARMUP; ++i) {
        dispatchWavefront(edge, edge, PROBE_BOUNCES, 1, 0);
    }

    const double mpix = static_cast<double>(edge) * static_cast<double>(edge) / 1.0e6;
    double best = 0.0;
    for (int i = 0; i < PROBE_REPEATS; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        dispatchWavefront(edge, edge, PROBE_BOUNCES, 1, 0);
        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        best = std::max(best, mpix / std::max(ms, 1.0e-3));
    }
    return best;
}

void probeTileTarget(const GPUCameraData& cam) {
    if (tileProfile.probed) return;

    const uint64_t key = deviceKeyOf(primaryDevice);
    TileProfile cached;
    if (tileProfileStore.find(key, cached)) {
        tileProfile = cached;
        return;
    }

    const int seed = seedTileTarget(primaryDevice);
    const int ceiling = memoryCeilingTileEdge(primaryDevice);
    const int hiEdge = quantizeTileEdge(std::min(seed * 2, ceiling));

    int lo = TILE_MIN / TILE_QUANTUM;
    int hi = std::max(hiEdge / TILE_QUANTUM, lo + 1);
    int bestQ = lo;
    double bestT = 0.0;

    while (hi - lo > PROBE_BRACKET) {
        const int a = lo + static_cast<int>((hi - lo) * PROBE_LO_SPLIT);
        const int b = lo + static_cast<int>((hi - lo) * PROBE_HI_SPLIT);
        if (a == b) break;
        const double ta = probeTileEdge(a * TILE_QUANTUM, cam);
        const double tb = probeTileEdge(b * TILE_QUANTUM, cam);
        if (ta > bestT) {
            bestT = ta;
            bestQ = a;
        }
        if (tb > bestT) {
            bestT = tb;
            bestQ = b;
        }
        if (ta < tb) lo = a;
        else hi = b;
    }
    for (int q = lo; q <= hi; ++q) {
        const double t = probeTileEdge(q * TILE_QUANTUM, cam);
        if (t > bestT) {
            bestT = t;
            bestQ = q;
        }
    }

    tileProfile.deviceKey = key;
    tileProfile.tileTarget = quantizeTileEdge(bestQ * TILE_QUANTUM);
    tileProfile.mpixPerMs = bestT;
    tileProfile.probed = true;
    tileProfileStore.put(tileProfile);
    tileProfileStore.save();
}

int tileTarget() const {
    if (tileProfile.probed) return tileProfile.tileTarget;
    return seedTileTarget(primaryDevice);
}

#include "vct_host.inl"

};
inline GpuContext vkCtx;

struct GpuFleet {
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<GpuContext*> gpus;
    std::vector<std::unique_ptr<GpuContext>> owned;
    bool initialized = false;
    bool multiEnabled = true;

    size_t count() const { return gpus.size(); }
    GpuContext& ctx(size_t i) { return *gpus[i]; }

    std::vector<VkDevice> devices() const {
        std::vector<VkDevice> v;
        v.reserve(gpus.size());
        for (auto* g : gpus) v.push_back(g->device);
        return v;
    }
    std::vector<VkPhysicalDevice> physicalDevices() const {
        std::vector<VkPhysicalDevice> v;
        v.reserve(gpus.size());
        for (auto* g : gpus) v.push_back(g->primaryDevice);
        return v;
    }

    VkInstance ensureInstance() {
        if (instance != VK_NULL_HANDLE) return instance;
        if (vkCtx.initialized && vkCtx.instance != VK_NULL_HANDLE) {
            instance = vkCtx.instance;
            return instance;
        }
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &appInfo;
        vkCreateInstance(&createInfo, nullptr, &instance);
        return instance;
    }

    void init() {
        if (initialized) return;
        initialized = true;

        ensureInstance();

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> all(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, all.data());

        struct Cand {
            VkPhysicalDevice dev;
            int score;
        };
        std::vector<Cand> cands;
        for (auto dev : all) {
            int s = GpuContext::scorePhysicalDevice(dev);
            if (s >= 0) cands.push_back({dev, s});
        }
        std::stable_sort(cands.begin(), cands.end(),
                         [](const Cand& a, const Cand& b) { return a.score > b.score; });

        if (!cands.empty() && cands.front().score >= 2) {
            cands.erase(std::remove_if(cands.begin(), cands.end(),
                        [](const Cand& c) { return c.score <= 1; }), cands.end());
        }
        if (!multiEnabled && cands.size() > 1) cands.resize(1);

        if (!vkCtx.initialized) {
            vkCtx.init(cands.empty() ? VK_NULL_HANDLE : cands.front().dev, instance);
        }
        if (!cands.empty()) vkCtx.score = cands.front().score;
        gpus.push_back(&vkCtx);

        for (auto& c : cands) {
            if (c.dev == vkCtx.primaryDevice) continue;
            auto g = std::make_unique<GpuContext>();
            g->init(c.dev, instance);
            g->score = c.score;
            gpus.push_back(g.get());
            owned.push_back(std::move(g));
        }
    }
    void probeTileTargets(const GPUCameraData& cam) {
        if (tileProbed) return;
        tileProbed = true;
        tileProfileStore.load();
        for (size_t g = 0; g < gpus.size(); ++g) {
            gpus[g]->probeTileTarget(cam);
        }
    }

    bool tileProbed = false;
};
inline GpuFleet gpuFleet;
#endif
}