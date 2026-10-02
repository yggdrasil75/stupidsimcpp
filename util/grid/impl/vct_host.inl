static constexpr uint32_t VCT_RES = 128;

VkImage        vctImage      = VK_NULL_HANDLE;
VkDeviceMemory vctImageMem   = VK_NULL_HANDLE;
VkImageView    vctSampleView = VK_NULL_HANDLE;
std::vector<VkImageView> vctMipViews;
VkSampler      vctSampler    = VK_NULL_HANDLE;
uint32_t       vctMipLevels  = 1;

VkBuffer       vctParamBuf   = VK_NULL_HANDLE;
VkDeviceMemory vctParamMem   = VK_NULL_HANDLE;
VCTParams      vctParams{};

VkShaderModule       vctBrickVoxShader   = VK_NULL_HANDLE;
VkDescriptorSetLayout vctBrickVoxLayout  = VK_NULL_HANDLE;
VkPipelineLayout     vctBrickVoxPipeLayout = VK_NULL_HANDLE;
VkPipeline           vctBrickVoxPipe     = VK_NULL_HANDLE;
VkDescriptorSet      vctBrickVoxSet      = VK_NULL_HANDLE;

VkShaderModule       vctClearShader   = VK_NULL_HANDLE;
VkDescriptorSetLayout vctClearLayout  = VK_NULL_HANDLE;
VkPipelineLayout     vctClearPipeLayout = VK_NULL_HANDLE;
VkPipeline           vctClearPipe     = VK_NULL_HANDLE;
VkDescriptorSet      vctClearSet      = VK_NULL_HANDLE;

///@brief Bricks to voxelize this build (full build: every brick); one per frame slot so a
///       frame still in flight never sees the next frame's list
VkBuffer       vctBrickListBuf[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VkDeviceMemory vctBrickListMem[FRAME_SLOTS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
uint32_t       vctBrickListCap[FRAME_SLOTS] = {0, 0};

///@brief The volume currently on the GPU: valid contents, and the bounds they were built for
bool vctHasContents = false;
Vec3 vctBuiltMin = Vec3::Zero();
Vec3 vctBuiltExtent = Vec3::Zero();

///@brief Work queued by vctBuildVolumeBricks, recorded into the next fast frame's command
///       buffer by vctRecordPending so the volume build never waits on its own fence.
struct VCTPending {
    bool active = false;
    bool full = false;
    Eigen::Vector3i regionMin = Eigen::Vector3i::Zero();
    Eigen::Vector3i regionSize = Eigen::Vector3i::Zero();
    std::vector<uint32_t> list;
} vctPending;

VkShaderModule       vctMipShader   = VK_NULL_HANDLE;
VkDescriptorSetLayout vctMipLayout  = VK_NULL_HANDLE;
VkPipelineLayout     vctMipPipeLayout = VK_NULL_HANDLE;
VkPipeline           vctMipPipe     = VK_NULL_HANDLE;
std::vector<VkDescriptorSet> vctMipSets;

void vctCreateImage() {
    vctMipLevels = vctMipCount(VCT_RES);

    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_3D;
    ic.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ic.extent = { VCT_RES, VCT_RES, VCT_RES };
    ic.mipLevels = vctMipLevels;
    ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ic.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device, &ic, nullptr, &vctImage);

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device, vctImage, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(primaryDevice, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &ai, nullptr, &vctImageMem);
    vkBindImageMemory(device, vctImage, vctImageMem, 0);

    // full-chain sampling view
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = vctImage;
    vc.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vc.format = ic.format;
    vc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, vctMipLevels, 0, 1 };
    vkCreateImageView(device, &vc, nullptr, &vctSampleView);

    // per-mip storage views
    vctMipViews.resize(vctMipLevels);
    for (uint32_t m = 0; m < vctMipLevels; ++m) {
        VkImageViewCreateInfo mvc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        mvc.image = vctImage;
        mvc.viewType = VK_IMAGE_VIEW_TYPE_3D;
        mvc.format = ic.format;
        mvc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1 };
        vkCreateImageView(device, &mvc, nullptr, &vctMipViews[m]);
    }

    VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sc.magFilter = VK_FILTER_LINEAR;
    sc.minFilter = VK_FILTER_LINEAR;
    sc.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sc.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sc.minLod = 0.0f;
    sc.maxLod = float(vctMipLevels);
    vkCreateSampler(device, &sc, nullptr, &vctSampler);
}

void vctInit() {
    vctCreateImage();

    createBuffer(device, primaryDevice, sizeof(VCTParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vctParamBuf, vctParamMem);

    {
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b};
        vkCreateDescriptorSetLayout(device, &li, nullptr, &vctClearLayout);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VCTRegionPush)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &vctClearLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(device, &pl, nullptr, &vctClearPipeLayout);

        vctClearShader = createShaderModule(device, "./bin/vct_clear_region.spv");
        if (vctClearShader) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.pName = "main";
            ci.stage.module = vctClearShader;
            ci.layout = vctClearPipeLayout;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &vctClearPipe);
        }

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &vctClearLayout;
        vkAllocateDescriptorSets(device, &ai, &vctClearSet);
    }

    {
        // brick voxelizer: 0 headers, 1 mat words, 2 occupancy, 3 palette, 4 materials,
        // 5 volume image, 6 VCT params, 7 brick params, 8 brick index list
        VkDescriptorSetLayoutBinding b[9]{};
        for (uint32_t i = 0; i < 5; ++i) {
            b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[7] = {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 9, b};
        vkCreateDescriptorSetLayout(device, &li, nullptr, &vctBrickVoxLayout);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &vctBrickVoxLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(device, &pl, nullptr, &vctBrickVoxPipeLayout);

        vctBrickVoxShader = createShaderModule(device, "./bin/vct_voxelize_brick.spv");
        if (vctBrickVoxShader) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.pName = "main";
            ci.stage.module = vctBrickVoxShader;
            ci.layout = vctBrickVoxPipeLayout;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &vctBrickVoxPipe);
        }

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &vctBrickVoxLayout;
        if (vkAllocateDescriptorSets(device, &ai, &vctBrickVoxSet) != VK_SUCCESS) {
            std::cerr << "[vct] failed to allocate the brick voxelizer descriptor set" << std::endl;
            vctBrickVoxSet = VK_NULL_HANDLE;
        }
    }

    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, b};
        vkCreateDescriptorSetLayout(device, &li, nullptr, &vctMipLayout);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VCTMipPush)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &vctMipLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(device, &pl, nullptr, &vctMipPipeLayout);

        vctMipShader = createShaderModule(device, "./bin/vct_mip.spv");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.pName = "main";
        ci.stage.module = vctMipShader;
        ci.layout = vctMipPipeLayout;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &vctMipPipe);

        vctMipSets.resize(vctMipLevels > 0 ? vctMipLevels - 1 : 0);
        for (size_t i = 0; i < vctMipSets.size(); ++i) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &vctMipLayout;
            vkAllocateDescriptorSets(device, &ai, &vctMipSets[i]);

            VkDescriptorImageInfo srcI{VK_NULL_HANDLE, vctMipViews[i],   VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo dstI{VK_NULL_HANDLE, vctMipViews[i+1], VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2]{};
            w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[0].dstSet = vctMipSets[i];
            w[0].dstBinding = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo = &srcI;
            w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[1].dstSet = vctMipSets[i];
            w[1].dstBinding = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo = &dstI;
            vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
        }
    }

    {
        VkDescriptorImageInfo imgI{VK_NULL_HANDLE, vctMipViews[0], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = vctClearSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgI;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    {
        VkDescriptorImageInfo imgI{VK_NULL_HANDLE, vctMipViews[0], VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo uboI{vctParamBuf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = vctBrickVoxSet;
        w[0].dstBinding = 5;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo = &imgI;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = vctBrickVoxSet;
        w[1].dstBinding = 6;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[1].pBufferInfo = &uboI;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }

    vctReady = true;
}

void vctWriteFastDescriptors() {
    if (!vctReady) return;
    VkDescriptorImageInfo si{vctSampler, vctSampleView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo bi{vctParamBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = fastDescSet;
    w[0].dstBinding = 9;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &si;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = fastDescSet;
    w[1].dstBinding = 10;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[1].pBufferInfo = &bi;
    vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
}

void vctImageBarrier(VkCommandBuffer cmd, VkImageLayout oldL, VkImageLayout newL,
                     VkAccessFlags src, VkAccessFlags dst,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                     uint32_t baseMip, uint32_t mipCount) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = oldL;
    b.newLayout = newL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = vctImage;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1 };
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

///@brief Mip level 0 cell range covered by a world-space box (clamped to the grid)
void vctCellRange(const Vec3& lo, const Vec3& hi, Eigen::Vector3i& cmin, Eigen::Vector3i& cmax) const {
    for (int a = 0; a < 3; ++a) {
        cmin[a] = std::clamp(int(std::floor((lo[a] - vctParams.volMin[a]) * vctParams.invVoxelSize)), 0, int(VCT_RES) - 1);
        cmax[a] = std::clamp(int(std::floor((hi[a] - vctParams.volMin[a]) * vctParams.invVoxelSize)), 0, int(VCT_RES) - 1);
    }
}

void vctUploadBrickList(uint32_t slot, const std::vector<uint32_t>& list) {
    const VkDeviceSize bytes = std::max<VkDeviceSize>(list.size() * sizeof(uint32_t), sizeof(uint32_t));
    if (bytes > vctBrickListCap[slot]) {
        destroyBuffer(device, vctBrickListBuf[slot], vctBrickListMem[slot]);
        createBuffer(device, primaryDevice, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vctBrickListBuf[slot], vctBrickListMem[slot]);
        vctBrickListCap[slot] = uint32_t(bytes);
    }
    if (list.empty()) return;
    void* p;
    vkMapMemory(device, vctBrickListMem[slot], 0, bytes, 0, &p);
    memcpy(p, list.data(), list.size() * sizeof(uint32_t));
    vkUnmapMemory(device, vctBrickListMem[slot]);
}

///@brief Queue a radiance volume update from the brick buffers.
///       Full build when the bounds changed or the volume has no contents yet; otherwise only
///       the region covered by `dirtyBricks` is cleared and re-voxelized (with every brick
///       that touches that region, since a volume cell can hold contributions from several
///       bricks) and only that region's mip chain is rebuilt. Nothing is queued when
///       nothing changed. The work is recorded by vctRecordPending inside the next fast frame.
void vctBuildVolumeBricks(const Vec3& aabbMin, const Vec3& aabbMax, const Vec3& lightDir, bool enabled,
                          const BrickGPUData& bricks, bool bricksRebuilt, const std::vector<uint32_t>& dirtyBricks) {
    if (!vctReady) return;
    if (!vctBrickVoxPipe || !vctBrickVoxSet || !vctClearPipe || brickCount == 0) enabled = false;

    Vec3 ext = aabbMax - aabbMin;
    for (int i = 0; i < 3; ++i) if (ext[i] <= 1e-6f) ext[i] = 1.0f;
    Vec3 pad = ext * 0.02f;
    Vec3 vmin = aabbMin - pad;
    Vec3 vext = ext + pad * 2.0f;

    const bool boundsChanged = !vctHasContents || (vmin - vctBuiltMin).cwiseAbs().maxCoeff() > 1e-5f
                            || (vext - vctBuiltExtent).cwiseAbs().maxCoeff() > 1e-5f;

    vctParams.volMin = vmin;
    vctParams.volExtent = vext;
    vctParams.voxelSize = vext.x() / float(VCT_RES);
    vctParams.invVoxelSize = 1.0f / vctParams.voxelSize;
    vctParams.gridRes = Eigen::Vector3i(VCT_RES, VCT_RES, VCT_RES);
    vctParams.maxMip = int(vctMipLevels) - 1;
    vctParams.lightDir = lightDir.normalized();
    vctParams.enabled = enabled ? 1.0f : 0.0f;

    void* pdata;
    vkMapMemory(device, vctParamMem, 0, sizeof(VCTParams), 0, &pdata);
    memcpy(pdata, &vctParams, sizeof(VCTParams));
    vkUnmapMemory(device, vctParamMem);

    if (!enabled) return;

    if (boundsChanged || bricksRebuilt) {
        vctPending.list.resize(brickCount);
        for (uint32_t i = 0; i < brickCount; ++i) vctPending.list[i] = i;
        vctPending.active = true;
        vctPending.full = true;
        vctBuiltMin = vmin;
        vctBuiltExtent = vext;
        return;
    }
    if (dirtyBricks.empty()) return;

    // region: cells covered by the dirty bricks, grown by one cell so a voxel straddling
    // the boundary is re-evaluated on both sides
    Eigen::Vector3i rmin = Eigen::Vector3i::Constant(int(VCT_RES));
    Eigen::Vector3i rmax = Eigen::Vector3i::Constant(-1);
    for (uint32_t i : dirtyBricks) {
        const GPUAabb& a = bricks.aabbs[i];
        Eigen::Vector3i cmin, cmax;
        vctCellRange(Vec3(a.minX, a.minY, a.minZ), Vec3(a.maxX, a.maxY, a.maxZ), cmin, cmax);
        rmin = rmin.cwiseMin(cmin);
        rmax = rmax.cwiseMax(cmax);
    }
    rmin = (rmin - Eigen::Vector3i::Ones()).cwiseMax(Eigen::Vector3i::Zero());
    rmax = (rmax + Eigen::Vector3i::Ones()).cwiseMin(Eigen::Vector3i::Constant(int(VCT_RES) - 1));

    std::vector<uint32_t>& touching = vctPending.list;
    touching.clear();
    touching.reserve(dirtyBricks.size() * 4);
    for (uint32_t i = 0; i < brickCount; ++i) {
        const GPUAabb& a = bricks.aabbs[i];
        Eigen::Vector3i cmin, cmax;
        vctCellRange(Vec3(a.minX, a.minY, a.minZ), Vec3(a.maxX, a.maxY, a.maxZ), cmin, cmax);
        if ((cmax.array() < rmin.array()).any() || (cmin.array() > rmax.array()).any()) continue;
        touching.push_back(i);
    }
    vctPending.active = true;
    vctPending.full = false;
    vctPending.regionMin = rmin;
    vctPending.regionSize = rmax - rmin + Eigen::Vector3i::Ones();
}

void vctWriteBrickVoxDescriptors(uint32_t slot) {
    const VkBuffer buffers[5] = {
        brickHeaders.buffer, brickMats.buffer, brickOccs.buffer, brickPalette.buffer, materialBuffer,
    };
    VkDescriptorBufferInfo infos[7]{};
    VkWriteDescriptorSet w[7]{};
    for (uint32_t i = 0; i < 5; ++i) {
        infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = vctBrickVoxSet;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &infos[i];
    }
    infos[5] = {brickParamBuffer, 0, VK_WHOLE_SIZE};
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[5].dstSet = vctBrickVoxSet;
    w[5].dstBinding = 7;
    w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[5].pBufferInfo = &infos[5];
    infos[6] = {vctBrickListBuf[slot], 0, VK_WHOLE_SIZE};
    w[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[6].dstSet = vctBrickVoxSet;
    w[6].dstBinding = 8;
    w[6].descriptorCount = 1;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[6].pBufferInfo = &infos[6];
    vkUpdateDescriptorSets(device, 7, w, 0, nullptr);
}

///@brief Record the queued volume update into the frame slot's command buffer, before the
///       pass that samples it. The slot's fence guarantees its list buffer is free.
void vctRecordPending(VkCommandBuffer cmd, uint32_t slot) {
    if (!vctPending.active) return;
    vctPending.active = false;
    vctUploadBrickList(slot, vctPending.list);
    vctWriteBrickVoxDescriptors(slot);
    const uint32_t listCount = uint32_t(vctPending.list.size());

    const VkImageLayout from = vctHasContents ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    vctImageBarrier(cmd, from, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, vctMipLevels);

    Eigen::Vector3i regionMin = vctPending.regionMin;
    Eigen::Vector3i regionSize = vctPending.regionSize;
    if (vctPending.full) {
        VkClearColorValue clr{};
        VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, vctMipLevels, 0, 1};
        vkCmdClearColorImage(cmd, vctImage, VK_IMAGE_LAYOUT_GENERAL, &clr, 1, &all);
        regionMin = Eigen::Vector3i::Zero();
        regionSize = Eigen::Vector3i::Constant(int(VCT_RES));
    } else {
        VCTRegionPush pc{{regionMin.x(), regionMin.y(), regionMin.z()}, 0,
                         {regionSize.x(), regionSize.y(), regionSize.z()}, 0};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctClearPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctClearPipeLayout, 0, 1, &vctClearSet, 0, nullptr);
        vkCmdPushConstants(cmd, vctClearPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (regionSize.x() + 3) / 4, (regionSize.y() + 3) / 4, (regionSize.z() + 3) / 4);
    }

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctBrickVoxPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctBrickVoxPipeLayout, 0, 1, &vctBrickVoxSet, 0, nullptr);
    vkCmdPushConstants(cmd, vctBrickVoxPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &listCount);
    const uint32_t threads = listCount * BRICK_VOXELS;
    vkCmdDispatch(cmd, (threads + 63) / 64, 1, 1);

    // mip chain over the region only; each level halves the region's cell range
    uint32_t res = VCT_RES;
    Eigen::Vector3i lo = regionMin;
    Eigen::Vector3i hi = regionMin + regionSize - Eigen::Vector3i::Ones();
    for (uint32_t i = 0; i + 1 < vctMipLevels; ++i) {
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);

        uint32_t dstRes = std::max(1u, res >> 1);
        lo = lo / 2;
        hi = hi / 2;
        Eigen::Vector3i size = hi - lo + Eigen::Vector3i::Ones();
        VCTMipPush pc{{int(dstRes), int(dstRes), int(dstRes)}, 0, {lo.x(), lo.y(), lo.z()}, 0};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctMipPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vctMipPipeLayout, 0, 1, &vctMipSets[i], 0, nullptr);
        vkCmdPushConstants(cmd, vctMipPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VCTMipPush), &pc);
        vkCmdDispatch(cmd, (size.x() + 3) / 4, (size.y() + 3) / 4, (size.z() + 3) / 4);
        res = dstRes;
    }

    vctImageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, vctMipLevels);
    vctHasContents = true;
}
