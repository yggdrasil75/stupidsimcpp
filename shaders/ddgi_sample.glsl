#ifndef DDGI_SAMPLE_GLSL
#define DDGI_SAMPLE_GLSL

vec3 ddgiSampleIrrTile(int probeIdx, vec3 dir) {
    vec2 oct = dirToOct(normalize(dir));
    vec2 uv = (oct * 0.5 + 0.5) * float(DDGI_IRR_RES) + 1.0;
    vec2 f = fract(uv - 0.5);
    ivec2 b = ivec2(floor(uv - 0.5));
    int base = ddgiIrrBase(probeIdx);
    vec3 acc = vec3(0.0);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            ivec2 t = clamp(b + ivec2(i, j), ivec2(0), ivec2(DDGI_IRR_BORDER - 1));
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            acc += ddgiIrradiance[base + t.y * DDGI_IRR_BORDER + t.x].rgb * w;
        }
    }
    return acc;
}

vec2 ddgiSampleVisTile(int probeIdx, vec3 dir) {
    vec2 oct = dirToOct(normalize(dir));
    vec2 uv = (oct * 0.5 + 0.5) * float(DDGI_VIS_RES) + 1.0;
    vec2 f = fract(uv - 0.5);
    ivec2 b = ivec2(floor(uv - 0.5));
    int base = ddgiVisBase(probeIdx);
    vec2 acc = vec2(0.0);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            ivec2 t = clamp(b + ivec2(i, j), ivec2(0), ivec2(DDGI_VIS_BORDER - 1));
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            acc += ddgiVisibility[base + t.y * DDGI_VIS_BORDER + t.x].xy * w;
        }
    }
    return acc;
}

vec3 ddgiSampleIrradiance(vec3 P, vec3 N, vec3 V, DDGIVolume vol) {
    if (vol.state.z == 0 || vol.counts.w <= 0) return vec3(0.0);

    float spacing = vol.originSpacing.w;
    vec3 biased = P + N * vol.params.z + V * vol.params.w;

    vec3 gridF = (biased - vol.originSpacing.xyz) / spacing;
    ivec3 baseC = ivec3(floor(gridF));
    vec3 alpha = clamp(gridF - vec3(baseC), vec3(0.0), vec3(1.0));

    vec3 sumIrr = vec3(0.0);
    float sumW = 0.0;

    for (int i = 0; i < 8; ++i) {
        ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 c = clamp(baseC + off, ivec3(0), vol.counts.xyz - 1);
        vec3 probeP = ddgiProbePosition(c, vol);
        int pIdx = ddgiProbeIndex(c, vol.counts.xyz);

        vec3 trilinear = mix(1.0 - alpha, alpha, vec3(off));
        float weight = trilinear.x * trilinear.y * trilinear.z;
        if (weight <= 1e-6) continue;

        vec3 toProbe = probeP - P;
        float distToProbe = length(toProbe);
        vec3 dirToProbe = distToProbe > 1e-6 ? toProbe / distToProbe : N;

        float wrap = (dot(dirToProbe, N) + 1.0) * 0.5;
        weight *= wrap * wrap + 0.2;

        vec2 vis = ddgiSampleVisTile(pIdx, -dirToProbe);
        float meanDist = vis.x;
        float meanDist2 = vis.y;
        if (distToProbe > meanDist) {
            float variance = max(1e-5, meanDist2 - meanDist * meanDist);
            float d = distToProbe - meanDist;
            float cheb = variance / (variance + d * d);
            cheb = max(0.0, cheb * cheb * cheb);
            weight *= cheb;
        }

        if (weight < 1e-6) continue;

        vec3 irr = ddgiSampleIrrTile(pIdx, N);
        sumIrr += irr * weight;
        sumW += weight;
    }

    if (sumW <= 1e-6) return vec3(0.0);
    vec3 result = sumIrr / sumW;
    return result * result;
}

#endif
