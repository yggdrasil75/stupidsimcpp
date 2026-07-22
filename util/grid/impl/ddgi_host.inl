static constexpr int DDGI_IRR_RES    = 8;
static constexpr int DDGI_VIS_RES    = 16;
static constexpr int DDGI_IRR_BORDER = DDGI_IRR_RES + 2;
static constexpr int DDGI_VIS_BORDER = DDGI_VIS_RES + 2;
static constexpr int DDGI_PROBE_RAYS = 128;
static constexpr int DDGI_RAY_STRIDE = 2;

struct alignas(16) GPUDDGIVolume {
    Vec3 origin;
    float spacing;
    int32_t countX;
    int32_t countY;
    int32_t countZ;
    int32_t probeCount;
    float hysteresis;
    float depthSharpness;
    float normalBias;
    float viewBias;
    int32_t frameIndex;
    int32_t raysPerProbe;
    int32_t enabled;
    float fireflyClamp;
};

VkBuffer ddgiIrradianceBuf = VK_NULL_HANDLE;
VkDeviceMemory ddgiIrradianceMem = VK_NULL_HANDLE;
VkBuffer ddgiVisibilityBuf = VK_NULL_HANDLE;
VkDeviceMemory ddgiVisibilityMem = VK_NULL_HANDLE;
VkBuffer ddgiRayBuf = VK_NULL_HANDLE;
VkDeviceMemory ddgiRayMem = VK_NULL_HANDLE;
VkBuffer ddgiVolumeBuf = VK_NULL_HANDLE;
VkDeviceMemory ddgiVolumeMem = VK_NULL_HANDLE;

VkShaderModule ddgiTraceShader = VK_NULL_HANDLE;
VkShaderModule ddgiBlendShader = VK_NULL_HANDLE;
VkShaderModule ddgiGatherShader = VK_NULL_HANDLE;
VkPipeline ddgiTracePipe = VK_NULL_HANDLE;
VkPipeline ddgiBlendPipe = VK_NULL_HANDLE;
VkPipeline ddgiGatherPipe = VK_NULL_HANDLE;

VkCommandBuffer ddgiCmd = VK_NULL_HANDLE;
VkFence ddgiFence = VK_NULL_HANDLE;
bool ddgiSubmitted = false;
GPUDDGIVolume ddgiVolume{};
uint32_t ddgiProbeCap = 0;
bool ddgiResident = false;
Vec3 ddgiPrevCamPos = Vec3::Zero();
bool ddgiPrevCamValid = false;
int ddgiProbeBounces = 2;

///@brief Fits a probe grid to a world bounding box at the requested spacing
///@param minB Volume minimum corner
///@param maxB Volume maximum corner
///@param spacing Uniform probe spacing in world units
void ddgiConfigure(const Vec3& minB, const Vec3& maxB, float spacing) {
    Vec3 extent = maxB - minB;
    int cx = std::max(2, static_cast<int>(std::ceil(extent.x() / spacing)) + 1);
    int cy = std::max(2, static_cast<int>(std::ceil(extent.y() / spacing)) + 1);
    int cz = std::max(2, static_cast<int>(std::ceil(extent.z() / spacing)) + 1);

    ddgiVolume.origin = minB;
    ddgiVolume.spacing = spacing;
    ddgiVolume.countX = cx;
    ddgiVolume.countY = cy;
    ddgiVolume.countZ = cz;
    ddgiVolume.probeCount = cx * cy * cz;
    ddgiVolume.hysteresis = 0.97f;
    ddgiVolume.depthSharpness = 50.0f;
    ddgiVolume.normalBias = spacing * 0.15f;
    ddgiVolume.viewBias = spacing * 0.1f;
    ddgiVolume.frameIndex = 0;
    ddgiVolume.raysPerProbe = DDGI_PROBE_RAYS;
    ddgiVolume.enabled = 1;
    ddgiVolume.fireflyClamp = 8.0f;
    ddgiResident = false;
}

void ddgiDestroyBuffers() {
    destroyBuffer(device, ddgiIrradianceBuf, ddgiIrradianceMem);
    destroyBuffer(device, ddgiVisibilityBuf, ddgiVisibilityMem);
    destroyBuffer(device, ddgiRayBuf, ddgiRayMem);
}

void ddgiEnsureBuffers() {
    uint32_t probes = static_cast<uint32_t>(ddgiVolume.probeCount);
    if (probes == 0) return;
    if (probes <= ddgiProbeCap && ddgiIrradianceBuf) return;

    ddgiDestroyBuffers();

    const VkBufferUsageFlags store = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkDeviceSize irrSize = VkDeviceSize(probes) * DDGI_IRR_BORDER * DDGI_IRR_BORDER * 4 * sizeof(float);
    VkDeviceSize visSize = VkDeviceSize(probes) * DDGI_VIS_BORDER * DDGI_VIS_BORDER * 4 * sizeof(float);
    VkDeviceSize raySize = VkDeviceSize(probes) * DDGI_PROBE_RAYS * DDGI_RAY_STRIDE * 4 * sizeof(float);

    createBuffer(device, primaryDevice, irrSize, store, devLocal, ddgiIrradianceBuf, ddgiIrradianceMem);
    createBuffer(device, primaryDevice, visSize, store, devLocal, ddgiVisibilityBuf, ddgiVisibilityMem);
    createBuffer(device, primaryDevice, raySize, store, devLocal, ddgiRayBuf, ddgiRayMem);

    if (ddgiVolumeBuf == VK_NULL_HANDLE) {
        createBuffer(device, primaryDevice, sizeof(GPUDDGIVolume), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     ddgiVolumeBuf, ddgiVolumeMem);
    }

    ddgiProbeCap = probes;
    ddgiVolume.frameIndex = 0;

    VkCommandBufferBeginInfo zbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    zbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &zbi);
    vkCmdFillBuffer(commandBuffer, ddgiIrradianceBuf, 0, irrSize, 0);
    vkCmdFillBuffer(commandBuffer, ddgiVisibilityBuf, 0, visSize, 0);
    vkCmdFillBuffer(commandBuffer, ddgiRayBuf, 0, raySize, 0);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo zsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    zsi.commandBufferCount = 1;
    zsi.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &zsi, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
}

void ddgiUploadVolume() {
    if (ddgiVolumeBuf == VK_NULL_HANDLE) return;
    void* data;
    vkMapMemory(device, ddgiVolumeMem, 0, sizeof(GPUDDGIVolume), 0, &data);
    memcpy(data, &ddgiVolume, sizeof(GPUDDGIVolume));
    vkUnmapMemory(device, ddgiVolumeMem);
}

void ddgiInit() {
    ddgiTraceShader  = createShaderModule(device, "./bin/ddgi_trace.spv");
    ddgiBlendShader  = createShaderModule(device, "./bin/ddgi_blend.spv");
    ddgiGatherShader = createShaderModule(device, "./bin/ddgi_gather.spv");

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.pName = "main";
    ci.layout = wfPipelineLayout;

    ci.stage.module = ddgiTraceShader;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &ddgiTracePipe);
    ci.stage.module = ddgiBlendShader;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &ddgiBlendPipe);
    ci.stage.module = ddgiGatherShader;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &ddgiGatherPipe);
}
void ddgiSyncRenderSettings(int maxBounces, float clampMax) {
    ddgiVolume.fireflyClamp = clampMax;
    ddgiProbeBounces = std::max(1, maxBounces);
}

///@brief Traces and blends one probe update frame
///@param maxBounces Bounce budget for probe rays (2 is usually enough)
void ddgiUpdateProbes(int maxBounces, const Vec3& camPos) {
    TIME_FUNCTION;
    if (!ddgiVolume.enabled || ddgiVolume.probeCount == 0) return;

    ddgiEnsureBuffers();
    float warmup = (ddgiVolume.frameIndex < 30) ? 0.85f : 0.97f;

    float motionH = 0.97f;
    if (ddgiPrevCamValid) {
        float moved = (camPos - ddgiPrevCamPos).norm();
        float t = std::clamp(moved / std::max(1e-4f, ddgiVolume.spacing), 0.0f, 1.0f);
        motionH = 0.97f + (0.80f - 0.97f) * t;
    }
    ddgiPrevCamPos = camPos;
    ddgiPrevCamValid = true;

    ddgiVolume.hysteresis = std::min(warmup, motionH);

    ddgiUploadVolume();

    uint32_t rayCount = uint32_t(ddgiVolume.probeCount) * uint32_t(ddgiVolume.raysPerProbe);
    ensureWavefrontBuffers(rayCount);
    writeWavefrontDescriptors();

    const uint32_t WG = 64;
    uint32_t rayGroups = (rayCount + WG - 1) / WG;
    int maxIters = ddgiProbeBounces + 4;

    if (ddgiCmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = commandPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cai, &ddgiCmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fi, nullptr, &ddgiFence);
    }

    vkWaitForFences(device, 1, &ddgiFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &ddgiFence);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkCommandBuffer cmd = ddgiCmd;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            wfPipelineLayout, 0, 1, &wfDescSet, 0, nullptr);

    wfBind(cmd, wfArgsPipe);
    wfPush(cmd, 0, 4, 0);
    vkCmdDispatch(cmd, 1, 1, 1);
    wfBarrier(cmd);

    wfBind(cmd, ddgiTracePipe);
    wfPush(cmd, 0, 0, 0);
    vkCmdDispatch(cmd, rayGroups, 1, 1);
    wfBarrier(cmd);

    wfBind(cmd, wfArgsPipe);
    wfPush(cmd, 0, 0, 0);
    vkCmdDispatch(cmd, 1, 1, 1);
    wfBarrier(cmd);

    int parity = 0;
    for (int it = 0; it < maxIters; ++it) {
        wfBind(cmd, wfExtendPipe);
        wfPush(cmd, parity, 0, 0);
        vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_EXTEND_ARGS);
        wfBarrier(cmd);
        wfBind(cmd, wfArgsPipe);
        wfPush(cmd, parity, 1, 0);
        vkCmdDispatch(cmd, 1, 1, 1);
        wfBarrier(cmd);
        wfBind(cmd, wfShadePipe);
        wfPush(cmd, parity, 0, 0);
        vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADE_ARGS);
        wfBarrier(cmd);
        wfBind(cmd, wfArgsPipe);
        wfPush(cmd, parity, 5, 0);
        vkCmdDispatch(cmd, 1, 1, 1);
        wfBarrier(cmd);
        wfBind(cmd, wfShadowPipe);
        wfPush(cmd, parity, 0, 0);
        vkCmdDispatchIndirect(cmd, wfCounterBuf, WF_OFF_SHADOW_ARGS);
        wfBarrier(cmd);
        parity ^= 1;
    }

    wfBind(cmd, ddgiGatherPipe);
    wfPush(cmd, parity, 0, 0);
    vkCmdDispatch(cmd, rayGroups, 1, 1);
    wfBarrier(cmd);

    wfBind(cmd, ddgiBlendPipe);
    wfPush(cmd, 0, 0, 0);
    vkCmdDispatch(cmd, uint32_t(ddgiVolume.probeCount), 1, 1);
    wfBarrier(cmd);
    wfPush(cmd, 0, 1, 0);
    vkCmdDispatch(cmd, uint32_t(ddgiVolume.probeCount), 1, 1);
    wfBarrier(cmd);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, ddgiFence);
    ddgiSubmitted = true;

    ddgiVolume.frameIndex++;
    ddgiResident = true;
    ddgiUploadVolume();
}

void ddgiAwaitProbes() {
    if (!ddgiSubmitted || ddgiFence == VK_NULL_HANDLE) return;
    vkWaitForFences(device, 1, &ddgiFence, VK_TRUE, UINT64_MAX);
    ddgiSubmitted = false;
}
