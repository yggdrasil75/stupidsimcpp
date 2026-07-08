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

VkShaderModule       vctVoxShader   = VK_NULL_HANDLE;
VkDescriptorSetLayout vctVoxLayout  = VK_NULL_HANDLE;
VkPipelineLayout     vctVoxPipeLayout = VK_NULL_HANDLE;
VkPipeline           vctVoxPipe     = VK_NULL_HANDLE;
VkDescriptorSet      vctVoxSet      = VK_NULL_HANDLE;

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
        VkDescriptorSetLayoutBinding b[4]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[3] = {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, b};
        vkCreateDescriptorSetLayout(device, &li, nullptr, &vctVoxLayout);

        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &vctVoxLayout;
        vkCreatePipelineLayout(device, &pl, nullptr, &vctVoxPipeLayout);

        vctVoxShader = createShaderModule(device, "./bin/vct_voxelize.spv");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.pName = "main";
        ci.stage.module = vctVoxShader;
        ci.layout = vctVoxPipeLayout;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &vctVoxPipe);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &vctVoxLayout;
        vkAllocateDescriptorSets(device, &ai, &vctVoxSet);
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
        VkDescriptorBufferInfo uboI{vctParamBuf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = vctVoxSet;
        w[0].dstBinding = 2;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo = &imgI;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = vctVoxSet;
        w[1].dstBinding = 3;
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

void vctBuildVolume(VkBuffer pointBuf, uint32_t pointCount,
                    const Vec3& aabbMin, const Vec3& aabbMax,
                    const Vec3& lightDir, bool enabled) {
    if (!vctReady) return;

    Vec3 ext = aabbMax - aabbMin;
    for (int i = 0; i < 3; ++i) if (ext[i] <= 1e-6f) ext[i] = 1.0f;
    Vec3 pad = ext * 0.02f;
    Vec3 vmin = aabbMin - pad;
    Vec3 vext = ext + pad * 2.0f;

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

    {
        VkDescriptorBufferInfo pI{pointBuf, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mI{materialBuffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = vctVoxSet;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &pI;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = vctVoxSet;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &mI;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(commandBuffer, &bi);

    vctImageBarrier(commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, vctMipLevels);

    VkClearColorValue clr{};

    clr.float32[0]=clr.float32[1]=clr.float32[2]=clr.float32[3]=0.0f;
    VkImageSubresourceRange all{ VK_IMAGE_ASPECT_COLOR_BIT, 0, vctMipLevels, 0, 1 };
    vkCmdClearColorImage(commandBuffer, vctImage, VK_IMAGE_LAYOUT_GENERAL, &clr, 1, &all);

    vctImageBarrier(commandBuffer, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, vctMipLevels);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vctVoxPipe);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vctVoxPipeLayout, 0, 1, &vctVoxSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, (pointCount + 63) / 64, 1, 1);

    uint32_t res = VCT_RES;
    for (uint32_t i = 0; i + 1 < vctMipLevels; ++i) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);

        uint32_t dstRes = std::max(1u, res >> 1);
        VCTMipPush pc{ {int(dstRes), int(dstRes), int(dstRes)}, 0 };
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vctMipPipe);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                vctMipPipeLayout, 0, 1, &vctMipSets[i], 0, nullptr);
        vkCmdPushConstants(commandBuffer, vctMipPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(VCTMipPush), &pc);
        vkCmdDispatch(commandBuffer, (dstRes + 3) / 4, (dstRes + 3) / 4, (dstRes + 3) / 4);
        res = dstRes;
    }

    vctImageBarrier(commandBuffer, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, vctMipLevels);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    vkCreateFence(device, &fi, nullptr, &f);
    vkQueueSubmit(queue, 1, &si, f);
    vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, f, nullptr);

    vctWriteFastDescriptors();
}
