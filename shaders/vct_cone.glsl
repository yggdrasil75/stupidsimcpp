// vct_cone.glsl  -- shared voxel-cone-tracing sampling routines.
// Include AFTER declaring:
//   - a sampler3D bound as `voxelTex` (the mipmapped radiance volume, linear filtered)
//   - the VoxelParams UBO as `vp` with: volMin, voxelSize, volExtent, invVoxelSize,
//     gridRes, maxMip, lightDir
// Coordinates: the volume covers world AABB [volMin, volMin+volExtent].

#ifndef VCT_CONE_GLSL
#define VCT_CONE_GLSL

// World position -> normalized [0,1] texture coordinate in the volume.
vec3 worldToVoxelUV(vec3 p) {
    return (p - vp.volMin) / vp.volExtent;
}

// Sample the volume at a world position and a cone diameter (in world units).
// The diameter selects the mip LOD so wider cones read coarser, pre-averaged data.
vec4 sampleVoxelCone(vec3 worldPos, float diameter) {
    vec3 uv = worldToVoxelUV(worldPos);
    if (any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0))))
        return vec4(0.0);
    // LOD from cone diameter relative to a single mip0 cell.
    float lod = clamp(log2(max(diameter * vp.invVoxelSize, 1.0)), 0.0, float(vp.maxMip));
    return textureLod(voxelTex, uv, lod);
}

// March a single cone from `origin` along `dir`, accumulating front-to-back.
// aperture = tan(halfAngle)*2 ; larger = softer/wider (more occlusion blur).
// Returns rgb = incoming radiance, a = accumulated occlusion [0,1].
vec4 traceCone(vec3 origin, vec3 dir, float aperture, float maxDist) {
    vec4 acc = vec4(0.0);
    // Start a bit out so we don't sample the originating surface voxel.
    float dist = vp.voxelSize;             // one cell offset
    float startBias = vp.voxelSize * 1.0;
    dist = startBias;

    while (dist < maxDist && acc.a < 0.99) {
        float diameter = max(vp.voxelSize, aperture * dist);
        vec3 p = origin + dir * dist;
        vec4 s = sampleVoxelCone(p, diameter);

        // front-to-back compositing
        float w = (1.0 - acc.a);
        acc.rgb += w * s.rgb;
        acc.a   += w * s.a;

        // step proportional to cone diameter -> roughly constant sample overlap
        dist += diameter * 0.5;
    }
    return acc;
}

// Six-cone diffuse hemisphere gather around a surface normal.
// Returns indirect diffuse radiance; `ao` outputs ambient occlusion (1 = open).
vec3 coneTraceDiffuse(vec3 worldPos, vec3 normal, float maxDist, out float ao) {
    // Build a tangent basis.
    vec3 up = abs(normal.y) > 0.95 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 t = normalize(cross(up, normal));
    vec3 b = cross(normal, t);

    // 60-degree aperture cones: one straight up, five in a ring tilted 60 deg.
    const float aperture = 0.577; // tan(30deg)*2 ~ approximates a 60deg cone
    vec3 diffuse = vec3(0.0);
    float occlusion = 0.0;

    // weights: center cone gets more, ring cones share the rest
    const float wCenter = 0.25;
    const float wSide   = 0.15; // x5 = 0.75

    // offset origin off the surface to avoid self-occlusion
    vec3 o = worldPos + normal * vp.voxelSize * 1.5;

    vec4 c = traceCone(o, normal, aperture, maxDist);
    diffuse += wCenter * c.rgb;
    occlusion += wCenter * c.a;

    const float SIN60 = 0.866025;
    const float COS60 = 0.5;
    for (int i = 0; i < 5; ++i) {
        float ang = float(i) * (6.2831853 / 5.0);
        vec3 dirT = cos(ang) * t + sin(ang) * b;
        vec3 d = normalize(COS60 * normal + SIN60 * dirT);
        vec4 sc = traceCone(o, d, aperture, maxDist);
        diffuse   += wSide * sc.rgb;
        occlusion += wSide * sc.a;
    }

    ao = 1.0 - clamp(occlusion, 0.0, 1.0);
    return diffuse;
}

// Single tight specular cone along the reflection vector.
vec3 coneTraceSpecular(vec3 worldPos, vec3 normal, vec3 viewDir, float roughness, float maxDist) {
    vec3 refl = reflect(-viewDir, normal);
    // roughness widens the cone; clamp so mirror-ish surfaces stay sharp.
    float aperture = clamp(roughness, 0.05, 0.6);
    vec3 o = worldPos + normal * vp.voxelSize * 1.5;
    return traceCone(o, refl, aperture, maxDist).rgb;
}

#endif // VCT_CONE_GLSL
