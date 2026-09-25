// GPU side of the physics engine. Owns the buffers and compute pipelines the
// phys_*.comp shaders consume and records one command buffer per frame:
// [raster] then steps x (predict, scan, scatter, iterations, finalize, xsph).
// Uses the shared vkCtx device, queue and command pool from rendering.inl.
#ifdef VULKAN_SUPPORT
namespace Grid {

constexpr uint32_t PhysGpu_NO_OBJ = 0xFFFFFFFFu;
constexpr uint32_t PhysGpu_TYPE_FLUID = 1u;
constexpr uint32_t PhysGpu_TYPE_RIGID = 2u;
constexpr uint32_t PhysGpu_TYPE_SOFT = 3u;

///@brief Per rigid/soft object record, matches Obj in phys_common.glsl
struct PhysObjGpu {
    uint32_t start = 0;
    uint32_t count = 0;
    uint32_t jointBonds = 0;
    uint32_t fiberBonds = 0;
    float alpha = 1.0f;
    float cm[3] = {0.0f, 0.0f, 0.0f};
    float R[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float dv[3] = {0.0f, 0.0f, 0.0f};
    float dw[3] = {0.0f, 0.0f, 0.0f};
    ///@brief Total particle mass of the body
    float mass = 1.0f;
};

///@brief Intra-object bond checked for fracture on the GPU
struct PhysIBondGpu {
    uint32_t a;
    uint32_t b;
    float rest;
    float tension;
    float compression;
    float stiffness;
    uint32_t broken;
    uint32_t pad;
};

///@brief Cross-object distance constraint (anchor, joint or muscle fibre)
struct PhysXBondGpu {
    uint32_t other;
    uint32_t isFixed;
    float rest;
    float stiffness;
    float fixedPos[3];
    uint32_t kind;
};

constexpr uint32_t PhysGpu_XBOND_JOINT = 0u;
constexpr uint32_t PhysGpu_XBOND_FIBER = 1u;
constexpr uint32_t PhysGpu_FLAG_GRAVITY_POINT = 1u;
constexpr uint32_t PhysGpu_FLAG_SOLID_BOUNDARY = 2u;
///@brief floats per static primitive in PhysFrameInput::statics (3 x vec4, see phys_raster.comp)
constexpr size_t PHYS_STATIC_FLOATS = 12;

///@brief Push constants, matches PC in phys_common.glsl
struct PhysPush {
    float gridLo[3];
    float occCell;
    uint32_t gridDim[3];
    uint32_t numParticles;
    float gravity[3];
    float gravityStrength;
    float domLo[3];
    float dt;
    float domHi[3];
    float h;
    float hashCellSize;
    uint32_t numHashCells;
    float damping;
    float xsph;
    float surfaceTension;
    uint32_t count;
    uint32_t iteration;
    uint32_t flags;
};

///@brief Everything the CPU packs for one frame
struct PhysFrameInput {
    std::vector<float> pos;
    std::vector<float> vel;
    std::vector<uint32_t> info;
    std::vector<float> rest;
    std::vector<PhysObjGpu> objs;
    std::vector<PhysIBondGpu> ibonds;
    std::vector<PhysXBondGpu> xbonds;
    std::vector<uint32_t> xoff;
    std::vector<float> statics; // PHYS_STATIC_FLOATS per entry, see collectStaticVoxels
    bool rebuildOcc = false;
    PhysPush pc{};
    int steps = 1;
    int iterations = 4;

    uint32_t numParticles() const {
        return (uint32_t)(pos.size() / 4);
    }
};

///@brief A device buffer with its memory and allocated capacity
struct PhysBufferGpu {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize cap = 0;
};

///@brief One staging to device copy queued for the frame
struct PhysUploadGpu {
    int binding;
    const void* src;
    VkDeviceSize size;
    VkDeviceSize offset;
};

struct PhysGpu {
    static constexpr uint32_t BINDINGS = 17;
    enum Binding {
        B_POS, B_VEL, B_PRED, B_DISP, B_INFO, B_REST, B_COUNT, B_SORT, B_OBJ,
        B_IBOND, B_XBOND, B_XOFF, B_OCC, B_STAT, B_CNORM, B_VELTMP, B_START
    };
    enum Pipeline {
        P_RASTER, P_PREDICT, P_SCAN, P_SCATTER, P_LAMBDA, P_DELTA, P_XBOND,
        P_APPLY, P_STRAIN, P_SMREDUCE, P_SMAPPLY, P_BOUNCE, P_FINALIZE, P_XSPH, P_COUNT
    };
    static constexpr const char* SPV[P_COUNT] = {
        "./bin/phys_raster.spv",
        "./bin/phys_predict.spv",
        "./bin/phys_scan.spv",
        "./bin/phys_scatter.spv",
        "./bin/phys_lambda.spv",
        "./bin/phys_delta.spv",
        "./bin/phys_xbond.spv",
        "./bin/phys_apply.spv",
        "./bin/phys_strain.spv",
        "./bin/phys_sm_reduce.spv",
        "./bin/phys_sm_apply.spv",
        "./bin/phys_bounce.spv",
        "./bin/phys_finalize.spv",
        "./bin/phys_xsph.spv"
    };

    PhysBufferGpu bufs[BINDINGS];
    PhysBufferGpu staging;
    PhysBufferGpu readback;
    void* stagingMapped = nullptr;
    void* readbackMapped = nullptr;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipes[P_COUNT] = {};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool initialized = false;
    bool descriptorsDirty = true;

    bool init() {
        if (initialized) return true;
        vkCtx.init();
        if (vkCtx.device == VK_NULL_HANDLE) return false;
        VkDevice dev = vkCtx.device;

        VkDescriptorSetLayoutBinding bindings[BINDINGS] = {};
        for (uint32_t i = 0; i < BINDINGS; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = BINDINGS;
        layoutInfo.pBindings = bindings;
        vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &layout);

        VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PhysPush)};
        VkPipelineLayoutCreateInfo pipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &layout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pushRange;
        vkCreatePipelineLayout(dev, &pipeLayoutInfo, nullptr, &pipeLayout);

        for (int p = 0; p < P_COUNT; ++p) {
            VkShaderModule module = vkCtx.createShaderModule(dev, SPV[p]);
            if (module == VK_NULL_HANDLE) return false;
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeInfo.stage = stage;
            pipeInfo.layout = pipeLayout;
            vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipes[p]);
            vkDestroyShaderModule(dev, module, nullptr);
        }

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BINDINGS};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        vkCreateDescriptorPool(dev, &poolInfo, nullptr, &pool);
        VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setInfo.descriptorPool = pool;
        setInfo.descriptorSetCount = 1;
        setInfo.pSetLayouts = &layout;
        vkAllocateDescriptorSets(dev, &setInfo, &set);

        VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdInfo.commandPool = vkCtx.commandPool;
        cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cmdInfo, &cmd);
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(dev, &fenceInfo, nullptr, &fence);
        initialized = true;
        return true;
    }

    ///@brief Grows a buffer to at least size bytes, keeping it mapped if requested
    void ensure(PhysBufferGpu& bf, VkDeviceSize size, VkMemoryPropertyFlags props, void** mapped = nullptr) {
        size = std::max<VkDeviceSize>(size, 16);
        if (bf.cap >= size) return;
        if (bf.buf != VK_NULL_HANDLE) {
            if (mapped && *mapped) {
                vkUnmapMemory(vkCtx.device, bf.mem);
                *mapped = nullptr;
            }
            vkDestroyBuffer(vkCtx.device, bf.buf, nullptr);
            vkFreeMemory(vkCtx.device, bf.mem, nullptr);
        }
        VkDeviceSize cap = size + size / 2;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCtx.createBuffer(vkCtx.device, vkCtx.primaryDevice, cap, usage, props, bf.buf, bf.mem);
        bf.cap = cap;
        if (mapped) vkMapMemory(vkCtx.device, bf.mem, 0, VK_WHOLE_SIZE, 0, mapped);
        descriptorsDirty = true;
    }

    void ensureDevice(int binding, VkDeviceSize size) {
        ensure(bufs[binding], size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void writeDescriptors() {
        VkDescriptorBufferInfo infos[BINDINGS];
        VkWriteDescriptorSet writes[BINDINGS] = {};
        for (uint32_t i = 0; i < BINDINGS; ++i) {
            infos[i].buffer = bufs[i].buf;
            infos[i].offset = 0;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(vkCtx.device, BINDINGS, writes, 0, nullptr);
        descriptorsDirty = false;
    }

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void dispatch(int p, uint32_t groups, const PhysPush& pc) {
        if (groups == 0) return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p]);
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PhysPush), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
        barrier();
    }

    static uint32_t groupsFor(uint32_t n) {
        return (n + 255) / 256;
    }

    ///@brief Runs one frame. outPos and outVel get a vec4 per particle, outBroken the ibond flags.
    bool run(PhysFrameInput& in, std::vector<float>& outPos, std::vector<float>& outVel, std::vector<uint32_t>& outBroken) {
        if (!init()) return false;
        VkDevice dev = vkCtx.device;
        const uint32_t N = in.numParticles();
        if (N == 0 && !in.rebuildOcc) return true;

        uint32_t numCells = 1024;
        while (numCells < 2 * N) numCells <<= 1;
        in.pc.numHashCells = numCells;

        const uint64_t occCells = (uint64_t)in.pc.gridDim[0] * in.pc.gridDim[1] * in.pc.gridDim[2];
        const VkDeviceSize occBytes = (occCells + 31) / 32 * 4;
        const uint32_t numStatics = (uint32_t)(in.statics.size() / PHYS_STATIC_FLOATS);
        const VkDeviceSize objBytes = in.objs.size() * sizeof(PhysObjGpu);
        const VkDeviceSize ibondBytes = in.ibonds.size() * sizeof(PhysIBondGpu);
        const VkDeviceSize xbondBytes = in.xbonds.size() * sizeof(PhysXBondGpu);

        ensureDevice(B_POS, N * 16);
        ensureDevice(B_VEL, N * 16);
        ensureDevice(B_PRED, N * 16);
        ensureDevice(B_DISP, N * 16);
        ensureDevice(B_INFO, N * 16);
        ensureDevice(B_REST, N * 16);
        ensureDevice(B_COUNT, (numCells + 1) * 4);
        ensureDevice(B_START, (numCells + 1) * 4);
        ensureDevice(B_SORT, N * 4);
        ensureDevice(B_CNORM, N * 16);
        ensureDevice(B_VELTMP, N * 16);
        ensureDevice(B_OBJ, objBytes);
        ensureDevice(B_IBOND, ibondBytes);
        ensureDevice(B_XBOND, xbondBytes);
        ensureDevice(B_XOFF, (N + 1) * 4);
        ensureDevice(B_OCC, occBytes);
        if (in.rebuildOcc) ensureDevice(B_STAT, in.statics.size() * 4);

        std::vector<PhysUploadGpu> uploads;
        uploads.push_back({B_POS, in.pos.data(), N * 16, 0});
        uploads.push_back({B_VEL, in.vel.data(), N * 16, 0});
        uploads.push_back({B_INFO, in.info.data(), N * 16, 0});
        uploads.push_back({B_REST, in.rest.data(), N * 16, 0});
        uploads.push_back({B_OBJ, in.objs.data(), objBytes, 0});
        uploads.push_back({B_IBOND, in.ibonds.data(), ibondBytes, 0});
        uploads.push_back({B_XBOND, in.xbonds.data(), xbondBytes, 0});
        uploads.push_back({B_XOFF, in.xoff.data(), in.xoff.size() * 4, 0});
        if (in.rebuildOcc) uploads.push_back({B_STAT, in.statics.data(), in.statics.size() * 4, 0});
        VkDeviceSize total = 0;
        for (auto& u : uploads) {
            u.offset = total;
            total += (u.size + 255) & ~(VkDeviceSize)255;
        }
        VkMemoryPropertyFlags hostProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ensure(staging, total, hostProps, &stagingMapped);
        for (const auto& u : uploads) {
            if (u.size == 0) continue;
            std::memcpy((char*)stagingMapped + u.offset, u.src, u.size);
        }

        const VkDeviceSize rbPos = 0;
        const VkDeviceSize rbVel = (VkDeviceSize)N * 16;
        const VkDeviceSize rbBond = (VkDeviceSize)N * 32;
        ensure(readback, rbBond + ibondBytes, hostProps, &readbackMapped);

        if (descriptorsDirty) writeDescriptors();

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        for (const auto& u : uploads) {
            if (u.size == 0) continue;
            VkBufferCopy region{u.offset, 0, u.size};
            vkCmdCopyBuffer(cmd, staging.buf, bufs[u.binding].buf, 1, &region);
        }
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &set, 0, nullptr);
        PhysPush pc = in.pc;
        if (in.rebuildOcc) {
            vkCmdFillBuffer(cmd, bufs[B_OCC].buf, 0, occBytes, 0);
            barrier();
            pc.count = numStatics;
            dispatch(P_RASTER, groupsFor(numStatics), pc);
        } else {
            barrier();
        }
        const uint32_t G = groupsFor(N);
        for (int s = 0; s < in.steps && N > 0; ++s) {
            vkCmdFillBuffer(cmd, bufs[B_COUNT].buf, 0, (numCells + 1) * 4, 0);
            barrier();
            dispatch(P_PREDICT, G, pc);
            dispatch(P_SCAN, 1, pc);
            dispatch(P_SCATTER, G, pc);
            for (int it = 0; it < in.iterations; ++it) {
                pc.iteration = (uint32_t)it;
                dispatch(P_LAMBDA, G, pc);
                dispatch(P_DELTA, G, pc);
                dispatch(P_APPLY, G, pc);
                if (!in.ibonds.empty()) {
                    pc.count = (uint32_t)in.ibonds.size();
                    dispatch(P_STRAIN, groupsFor((uint32_t)in.ibonds.size()), pc);
                }
                if (!in.objs.empty()) {
                    if (!in.xbonds.empty()) dispatch(P_XBOND, G, pc);
                    dispatch(P_SMREDUCE, (uint32_t)in.objs.size(), pc);
                    dispatch(P_SMAPPLY, G, pc);
                }
            }
            if (!in.objs.empty()) dispatch(P_BOUNCE, (uint32_t)in.objs.size(), pc);
            dispatch(P_FINALIZE, G, pc);
            dispatch(P_XSPH, G, pc);
        }
        if (N > 0) {
            VkBufferCopy posRegion{0, rbPos, (VkDeviceSize)N * 16};
            VkBufferCopy velRegion{0, rbVel, (VkDeviceSize)N * 16};
            vkCmdCopyBuffer(cmd, bufs[B_POS].buf, readback.buf, 1, &posRegion);
            vkCmdCopyBuffer(cmd, bufs[B_VEL].buf, readback.buf, 1, &velRegion);
            if (!in.ibonds.empty()) {
                VkBufferCopy bondRegion{0, rbBond, ibondBytes};
                vkCmdCopyBuffer(cmd, bufs[B_IBOND].buf, readback.buf, 1, &bondRegion);
            }
        }
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkResetFences(dev, 1, &fence);
        vkQueueSubmit(vkCtx.queue, 1, &submitInfo, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

        outPos.resize((size_t)N * 4);
        outVel.resize((size_t)N * 4);
        outBroken.resize(in.ibonds.size());
        if (N == 0) return true;
        std::memcpy(outPos.data(), (char*)readbackMapped + rbPos, (size_t)N * 16);
        std::memcpy(outVel.data(), (char*)readbackMapped + rbVel, (size_t)N * 16);
        const PhysIBondGpu* bonds = (const PhysIBondGpu*)((char*)readbackMapped + rbBond);
        for (size_t i = 0; i < in.ibonds.size(); ++i) {
            outBroken[i] = bonds[i].broken;
        }
        return true;
    }
};

}
#endif
