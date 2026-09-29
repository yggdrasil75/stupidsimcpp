struct alignas(16) GPUBrickParams {
    float voxelSize;
    uint32_t brickCount;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
};

///@brief A device-local buffer that grows on demand
struct BrickBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t capacity = 0;
};

BrickBuffer brickHeaders;
BrickBuffer brickMats;
BrickBuffer brickOccs;
BrickBuffer brickPalette;
BrickBuffer brickAabbs;
VkBuffer brickParamBuffer = VK_NULL_HANDLE;
VkDeviceMemory brickParamMemory = VK_NULL_HANDLE;
uint32_t brickCount = 0;

// acceleration structures, separate from the per-voxel ones so both paths coexist
VkAccelerationStructureKHR brickBlas = VK_NULL_HANDLE;
VkAccelerationStructureKHR brickTlas = VK_NULL_HANDLE;
BrickBuffer brickBlasStorage;
BrickBuffer brickTlasStorage;
BrickBuffer brickScratch;
VkBuffer brickInstanceBuffer = VK_NULL_HANDLE;
VkDeviceMemory brickInstanceMemory = VK_NULL_HANDLE;
bool brickAsValid = false;
uint32_t brickAsPrimCount = 0;

bool brickInitialized = false;

void destroyBrickBuffer(BrickBuffer& b) {
    if (!b.buffer) return;
    vkDestroyBuffer(device, b.buffer, nullptr);
    vkFreeMemory(device, b.memory, nullptr);
    b.buffer = VK_NULL_HANDLE;
    b.memory = VK_NULL_HANDLE;
    b.capacity = 0;
}

///@brief Ensure `b` holds at least `bytes`. Reallocation invalidates the AS and descriptors.
void ensureBrickBuffer(BrickBuffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage, bool withAddress) {
    bytes = std::max<VkDeviceSize>(bytes, 256);
    if (b.buffer && bytes <= b.capacity) return;
    destroyBrickBuffer(b);
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (withAddress) {
        createBufferWithAddress(device, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b.buffer, b.memory);
    } else {
        createBuffer(device, primaryDevice, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b.buffer, b.memory);
    }
    b.capacity = static_cast<uint32_t>(bytes);
    brickAsValid = false;
}

///@brief One region copy into a device buffer through a temporary staging buffer
struct BrickCopy {
    VkBuffer dst;
    VkDeviceSize dstOffset;
    const void* src;
    VkDeviceSize size;
};

///@brief Upload several regions in one staging buffer and one submit
void uploadBrickRegions(const std::vector<BrickCopy>& copies) {
    VkDeviceSize total = 0;
    for (const BrickCopy& c : copies) total += c.size;
    if (total == 0) return;

    VkBuffer staging;
    VkDeviceMemory stagingMemory;
    createBuffer(device, primaryDevice, total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMemory);

    uint8_t* mapped = nullptr;
    vkMapMemory(device, stagingMemory, 0, total, 0, reinterpret_cast<void**>(&mapped));
    std::vector<VkBufferCopy> regions(copies.size());
    VkDeviceSize offset = 0;
    for (size_t i = 0; i < copies.size(); ++i) {
        memcpy(mapped + offset, copies[i].src, copies[i].size);
        regions[i] = VkBufferCopy{offset, copies[i].dstOffset, copies[i].size};
        offset += copies[i].size;
    }
    vkUnmapMemory(device, stagingMemory);

    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        for (size_t i = 0; i < copies.size(); ++i) {
            vkCmdCopyBuffer(cmd, staging, copies[i].dst, 1, &regions[i]);
        }
    });
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
}

void initBrickBuffers() {
    if (brickInitialized) return;
    createBuffer(device, primaryDevice, sizeof(GPUBrickParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 brickParamBuffer, brickParamMemory);
    brickInitialized = true;
}

///@brief Full upload of a flattened brick world + (re)build of its BLAS/TLAS.
void uploadBricksFull(const BrickGPUData& g) {
    initBrickBuffers();
    if (!brickInitialized || g.headers.empty()) return;

    const VkDeviceSize headerBytes = g.headers.size() * sizeof(GPUBrickHeader);
    const VkDeviceSize matBytes = g.matWords.size() * sizeof(uint32_t);
    const VkDeviceSize occBytes = g.occWords.size() * sizeof(uint32_t);
    const VkDeviceSize paletteBytes = g.palette.size() * sizeof(GPUPaletteEntry);
    const VkDeviceSize aabbBytes = g.aabbs.size() * sizeof(GPUAabb);
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags asInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | storage;

    ensureBrickBuffer(brickHeaders, headerBytes, storage, false);
    ensureBrickBuffer(brickMats, matBytes, storage, false);
    ensureBrickBuffer(brickOccs, occBytes, storage, false);
    ensureBrickBuffer(brickPalette, paletteBytes, storage, false);
    ensureBrickBuffer(brickAabbs, aabbBytes, asInput, true);

    uploadBrickRegions({
        {brickHeaders.buffer, 0, g.headers.data(), headerBytes},
        {brickMats.buffer, 0, g.matWords.data(), matBytes},
        {brickOccs.buffer, 0, g.occWords.data(), occBytes},
        {brickPalette.buffer, 0, g.palette.data(), paletteBytes},
        {brickAabbs.buffer, 0, g.aabbs.data(), aabbBytes},
    });

    GPUBrickParams params{g.voxelSize, static_cast<uint32_t>(g.headers.size())};
    void* mapped;
    vkMapMemory(device, brickParamMemory, 0, sizeof(params), 0, &mapped);
    memcpy(mapped, &params, sizeof(params));
    vkUnmapMemory(device, brickParamMemory);

    brickCount = static_cast<uint32_t>(g.headers.size());
    buildBrickAccelerationStructures(brickCount);
}

///@brief Partial upload: only the bricks listed in `touched` (from brickFlattenDirty).  No AS rebuild:
///       brick AABBs are fixed, only their contents changed.
void uploadBricksDirty(const BrickGPUData& g, const std::vector<uint32_t>& touched) {
    if (!brickInitialized || touched.empty()) return;
    if (g.headers.size() != brickCount) {
        uploadBricksFull(g);
        return;
    }
    std::vector<BrickCopy> copies;
    copies.reserve(touched.size() * 3);
    for (uint32_t i : touched) {
        const GPUBrickHeader& h = g.headers[i];
        copies.push_back({brickHeaders.buffer, i * sizeof(GPUBrickHeader), &h, sizeof(GPUBrickHeader)});
        copies.push_back({brickMats.buffer, h.matWordOffset * sizeof(uint32_t),
                          &g.matWords[h.matWordOffset], BRICK_MAT_WORDS * sizeof(uint32_t)});
        copies.push_back({brickOccs.buffer, h.occWordOffset * sizeof(uint32_t),
                          &g.occWords[h.occWordOffset], BRICK_OCC_WORDS * 2 * sizeof(uint32_t)});
    }
    uploadBrickRegions(copies);
}

void createBrickAccelerationStructure(VkAccelerationStructureTypeKHR type, VkDeviceSize size,
                                      BrickBuffer& storage, VkAccelerationStructureKHR& as) {
    if (as) {
        pfn_vkDestroyAccelerationStructureKHR(device, as, nullptr);
        as = VK_NULL_HANDLE;
    }
    destroyBrickBuffer(storage);
    createBufferWithAddress(device, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storage.buffer, storage.memory);
    storage.capacity = static_cast<uint32_t>(size);

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = storage.buffer;
    createInfo.size = size;
    createInfo.type = type;
    pfn_vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &as);
}

///@brief Build a BLAS with one AABB per brick and a single-instance TLAS over it
void buildBrickAccelerationStructures(uint32_t numPrimitives) {
    if (numPrimitives == 0) return;

    VkAccelerationStructureGeometryKHR blasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    blasGeom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blasGeom.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    blasGeom.geometry.aabbs.data.deviceAddress = getBufferDeviceAddress(brickAabbs.buffer);
    blasGeom.geometry.aabbs.stride = sizeof(GPUAabb);

    VkAccelerationStructureBuildGeometryInfoKHR blasBuild{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blasBuild.geometryCount = 1;
    blasBuild.pGeometries = &blasGeom;

    VkAccelerationStructureBuildSizesInfoKHR blasSizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &blasBuild, &numPrimitives, &blasSizes);
    createBrickAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                     blasSizes.accelerationStructureSize, brickBlasStorage, brickBlas);

    VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    blasAddrInfo.accelerationStructure = brickBlas;

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = {1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f};
    instance.mask = 0xFF;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &blasAddrInfo);
    if (!brickInstanceBuffer) {
        createBufferWithAddress(device, sizeof(instance),
                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                brickInstanceBuffer, brickInstanceMemory);
    }
    void* mapped;
    vkMapMemory(device, brickInstanceMemory, 0, sizeof(instance), 0, &mapped);
    memcpy(mapped, &instance, sizeof(instance));
    vkUnmapMemory(device, brickInstanceMemory);

    VkAccelerationStructureGeometryKHR tlasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.data.deviceAddress = getBufferDeviceAddress(brickInstanceBuffer);

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuild.geometryCount = 1;
    tlasBuild.pGeometries = &tlasGeom;

    uint32_t numInstances = 1;
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfn_vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &tlasBuild, &numInstances, &tlasSizes);
    createBrickAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                     tlasSizes.accelerationStructureSize, brickTlasStorage, brickTlas);

    VkDeviceSize scratchBytes = std::max(blasSizes.buildScratchSize, tlasSizes.buildScratchSize);
    ensureBrickBuffer(brickScratch, scratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    VkDeviceAddress scratchAddress = getBufferDeviceAddress(brickScratch.buffer);

    executeSingleTimeCommands([&](VkCommandBuffer cmd) {
        blasBuild.dstAccelerationStructure = brickBlas;
        blasBuild.scratchData.deviceAddress = scratchAddress;
        VkAccelerationStructureBuildRangeInfoKHR blasRange{};
        blasRange.primitiveCount = numPrimitives;
        VkAccelerationStructureBuildRangeInfoKHR* pBlasRange = &blasRange;
        pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blasBuild, &pBlasRange);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        tlasBuild.dstAccelerationStructure = brickTlas;
        tlasBuild.scratchData.deviceAddress = scratchAddress;
        VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
        tlasRange.primitiveCount = numInstances;
        VkAccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;
        pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuild, &pTlasRange);
    });

    brickAsValid = true;
    brickAsPrimCount = numPrimitives;
}

void writeFastDescriptors() {
    const VkBuffer buffers[14] = {
        nodeBuffer, fastPointBuffer, outBuffer, uboBuffer, skyboxBuffer, lightBuffer, adaptiveBuffer, materialBuffer,
        vctParamBuf, brickHeaders.buffer, brickMats.buffer, brickOccs.buffer, brickPalette.buffer, brickParamBuffer,
    };
    const uint32_t bindingIds[14] = {0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15};
    const VkDescriptorType bindingTypes[14] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    VkDescriptorBufferInfo bufferInfos[14] = {};
    VkWriteDescriptorSet writes[16] = {};
    for (int i = 0; i < 14; ++i) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = fastDescSet;
        writes[i].dstBinding = bindingIds[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindingTypes[i];
        writes[i].pBufferInfo = &bufferInfos[i];
    }

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &brickTlas;
    writes[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[14].pNext = &asInfo;
    writes[14].dstSet = fastDescSet;
    writes[14].dstBinding = 6;
    writes[14].descriptorCount = 1;
    writes[14].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorImageInfo vctImageInfo{vctSampler, vctSampleView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[15].dstSet = fastDescSet;
    writes[15].dstBinding = 9;
    writes[15].descriptorCount = 1;
    writes[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[15].pImageInfo = &vctImageInfo;

    vkUpdateDescriptorSets(device, 16, writes, 0, nullptr);
}
