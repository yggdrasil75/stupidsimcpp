#ifndef WF_COMMON_GLSL
#define WF_COMMON_GLSL

#define PI 3.14159265359
const float EPS  = 1e-8;
const float LARGE = 1e30;
const float GAP_EPSILON  = 1e-2;
const float DIST_EPSILON = 1e-4;

const int MAX_TRANSPARENT_BOUNCES = 12;
const int MAX_VOLUMETRIC_BOUNCES  = 8;

struct GPUMaterial {
    uint emittance;
    uint  materialProps;
    uint  absorption;
    uint  albedo;
};

struct GPUGasField {
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 cellSize;
    uint res;
    uint cellOffset;
    uint slotCount;
    uint pad0;
    uint slotToGlobal[8];
};

const uint MAX_GAS_SPECIES = 8u;
const uint GAS_INVALID_SLOT = 0xFFFFFFFFu;

struct GPUPBRRenderData {
    vec3 position;
    float size;
    uint color;
    uint materialIdx;
    int  objectId;
    uint isGas;
};

struct PathHot {
    vec4 o_tmax;
    vec4 dir_rough;
    vec4 thp;
    vec4 rad;
    uvec4 u0;
    ivec4 i1;
};

struct PathHit {
    vec4 hit;
};

#define PC_GET_BOUNCE(p)  (int((p) & 0xFFu))
#define PC_GET_TRANS(p)   (int(((p) >> 8u) & 0xFFu))
#define PC_GET_VOLUM(p)   (int(((p) >> 16u) & 0xFFu))
#define PC_PACK(b,t,v)    ((uint(b) & 0xFFu) | ((uint(t) & 0xFFu) << 8u) | ((uint(v) & 0xFFu) << 16u))

#define FLAG_SPECULAR 1
#define FLAG_HITFOUND 2
#define FLAG_ALIVE    4
#define HERO_SHIFT 5
#define HERO_MASK  (3 << HERO_SHIFT)
#define GET_HERO(f)    (((f) & HERO_MASK) >> HERO_SHIFT)
#define SET_HERO(f, h) (((f) & ~HERO_MASK) | (((h) & 3) << HERO_SHIFT))


struct ShadowRay {
    vec4 o_tmax;
    vec4 dir_slot;
    vec4 contrib;
    vec4 thp;
};

struct Counters {
    uint extendCount;
    uint shadeCount;
    uint shadowCount;
    uint nextExtendCount;
};
struct DispatchArgs {
    uvec4 extendArgs;
    uvec4 shadeArgs;
    uvec4 shadowArgs;
};

layout(local_size_x = 64) in;

layout(binding = 0) uniform CameraData {
    vec3 origin;
    float lodMinDist;
    vec3 dir;
    float invLodf;
    vec3 up;
    float minVisibility;
    vec3 right;
    float maxDist;
    vec3 skylight;
    float tanfovx;
    vec3 bgColor;
    float tanfovy;
    int width;
    int height;
    int maxBounces;
    int useLod;
    float invFogRange;
    uint frameCount;
    int skyWidth;
    int skyHeight;
    int currentSampleOffset;
    int dispatchSamples;
    int globalIllumination;
    uint nodeCount;
    uint pointCount;
    int tileOffsetX;
    int tileOffsetY;
    int emissiveCount;
    int targetSamples;
    int sellWidth;
    int sellSecondary;
    uint gasFieldCount;
    uint blueFrameSeed;
    uint gasPad1;
    uint gasPad2;
} cam;

layout(std430, binding = 1) readonly buffer PointBuffer    { GPUPBRRenderData points[]; };
layout(std430, binding = 2) readonly buffer MaterialBuffer { GPUMaterial materials[]; };
layout(std430, binding = 3) readonly buffer SkyboxBuffer   { vec4 skyPixels[]; };
layout(std430, binding = 4) readonly buffer LightBuffer    { uint emissiveIndices[]; };
layout(binding = 5) uniform accelerationStructureEXT tlas;
layout(std430, binding = 6) buffer OutputBuffer   { float pixels[]; };
layout(std430, binding = 7) buffer AdaptiveBuffer { float adaptiveData[]; };
layout(std430, binding = 8) buffer PathHotBuffer  { PathHot pathsHot[]; };
layout(std430, binding = 9)  buffer ExtendA   { uint extendA[]; };
layout(std430, binding = 10) buffer ExtendB   { uint extendB[]; };
layout(std430, binding = 11) buffer ShadeQ    { uint shadeQueue[]; };
layout(std430, binding = 12) buffer ShadowQ   { ShadowRay shadowQueue[]; };
layout(std430, binding = 13) buffer CounterBuf{ Counters ctr; };
layout(std430, binding = 14) buffer PathHitBuffer { PathHit pathsHit[]; };
layout(std430, binding = 15) readonly buffer SellmeierBuffer { float sellmeierLUT[]; };
layout(std430, binding = 16) readonly buffer GasFieldBuffer { GPUGasField gasFields[]; };
layout(std430, binding = 17) readonly buffer GasCellBuffer  { float gasCells[]; };
layout(std430, binding = 19) buffer DispatchArgsBuf { DispatchArgs args; };

layout(push_constant) uniform PC {
    int parity;
    int stage;
    int sampleIndex;
    int pad;
} pc;

vec3 unpackRGB8(uint c) {
    return vec3(float(c & 0xFF) / 255.0, float((c >> 8) & 0xFF) / 255.0, float((c >> 16) & 0xFF) / 255.0);
}

vec4 unpackRGBA8(uint c) {
    return vec4(float(c & 0xFF) / 255.0, float((c >> 8) & 0xFF) / 255.0, float((c >> 16) & 0xFF) / 255.0, float((c >> 24) & 0xFF) / 255.0);
}

vec3 unpackRGB9E5(uint c) {
    if (c == 0u) return vec3(0.0);
    int e = int(c >> 27) - 15;
    float scale = exp2(float(e - 9));
    float r = float(c & 0x1FFu) * scale;
    float g = float((c >> 9) & 0x1FFu) * scale;
    float b = float((c >> 18) & 0x1FFu) * scale;
    return vec3(r, g, b);
}

bool gasRayBox(vec3 ro, vec3 invD, vec3 bmin, vec3 bmax, float maxT, out float t0, out float t1) {
    vec3 ta = (bmin - ro) * invD;
    vec3 tb = (bmax - ro) * invD;
    vec3 tmin = min(ta, tb);
    vec3 tmax = max(ta, tb);
    t0 = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    t1 = min(min(tmax.x, tmax.y), min(tmax.z, maxT));
    return t1 > t0;
}

void marchGasFields(vec3 ro, vec3 rd, float segLen, inout vec3 throughput, inout vec3 radiance) {
    if (cam.gasFieldCount == 0u) return;
    vec3 invD = 1.0 / mix(rd, vec3(1e-8), equal(rd, vec3(0.0)));

    for (uint fi = 0u; fi < cam.gasFieldCount; ++fi) {
        GPUGasField gf = gasFields[fi];
        float t0, t1;
        if (!gasRayBox(ro, invD, gf.boundsMin.xyz, gf.boundsMax.xyz, segLen, t0, t1)) continue;

        float cellMin = min(gf.cellSize.x, min(gf.cellSize.y, gf.cellSize.z));
        cellMin = max(cellMin, 1e-4);
        int steps = int(clamp((t1 - t0) / cellMin, 1.0, 128.0));
        float dt = (t1 - t0) / float(steps);
        float R = float(gf.res);

        for (int s = 0; s < steps; ++s) {
            float tc = t0 + (float(s) + 0.5) * dt;
            vec3 wp = ro + rd * tc;
            vec3 rel = (wp - gf.boundsMin.xyz) / gf.cellSize.xyz;
            ivec3 ci = ivec3(floor(rel));
            if (any(lessThan(ci, ivec3(0))) || any(greaterThanEqual(ci, ivec3(int(gf.res))))) continue;

            uint cellIdx = gf.cellOffset + uint((ci.z * int(R) + ci.y) * int(R) + ci.x);
            uint base = cellIdx * MAX_GAS_SPECIES;

            vec3 sigma_t = vec3(0.0);
            vec3 sigma_s = vec3(0.0);
            vec3 emission = vec3(0.0);
            for (uint sl = 0u; sl < gf.slotCount; ++sl) {
                float dens = gasCells[base + sl];
                if (dens <= 1e-5) continue;
                uint mat = gf.slotToGlobal[sl];
                if (mat == GAS_INVALID_SLOT) continue;
                GPUMaterial gm = materials[mat];
                vec3 absorp = unpackRGB8(gm.absorption);
                vec3 alb    = unpackRGB8(gm.albedo);
                sigma_t += (absorp + absorp * alb) * dens;
                sigma_s += absorp * alb * dens;
                if (gm.emittance != 0u) emission += unpackRGB9E5(gm.emittance) * alb * dens;
            }

            if (max(sigma_t.x, max(sigma_t.y, sigma_t.z)) <= 1e-6) continue;

            vec3 stepTrans = exp(-sigma_t * dt);
            vec3 inscatter = emission + sigma_s * cam.skylight;
            radiance += throughput * inscatter * (vec3(1.0) - stepTrans) / max(sigma_t, vec3(1e-5));
            throughput *= stepTrans;

            if (max(throughput.x, max(throughput.y, throughput.z)) < 0.003) return;
        }
    }
}

const float SELL_LMIN = 0.380; // um
const float SELL_LMAX = 0.720; // um
const float HERO_LAMBDA_R = 0.610;
const float HERO_LAMBDA_G = 0.550;
const float HERO_LAMBDA_B = 0.465;

float lambdaToU(float lambdaUm) {
    return clamp((lambdaUm - SELL_LMIN) / (SELL_LMAX - SELL_LMIN), 0.0, 1.0);
}

float sampleIor(uint row, float lambdaUm) {
    int w = cam.sellWidth;
    if (w <= 0) return 1.45;
    float u = lambdaToU(lambdaUm) * float(w - 1);
    int i0 = int(floor(u));
    int i1 = min(i0 + 1, w - 1);
    float f = u - float(i0);
    uint base = row * uint(w);
    float a = sellmeierLUT[base + uint(i0)];
    float b = sellmeierLUT[base + uint(i1)];
    return mix(a, b, f);
}

void unpackMaterial(uint m, out float roughness, out float metallic, out uint sellRow) {
    roughness = float(m & 0xFF) / 255.0;
    metallic  = float((m >> 8) & 0xFF) / 255.0;
    sellRow   = (m >> 16) & 0xFFFFu;
}

float heroLambda(int hero) {
    if (hero == 1) return HERO_LAMBDA_R;
    if (hero == 3) return HERO_LAMBDA_B;
    return HERO_LAMBDA_G;
}

vec3 heroMask(int hero) {
    if (hero == 1) return vec3(1.0, 0.0, 0.0);
    if (hero == 3) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 1.0, 0.0);
}

bool isInvalid(float v) { return isnan(v) || isinf(v); }
bool isInvalid(vec3 v)  { return any(isnan(v)) || any(isinf(v)); }

vec3 safe_invDir(vec3 d) {
    return vec3(
        abs(d.x) < EPS ? (d.x < 0.0 ? -LARGE : LARGE) : 1.0 / d.x,
        abs(d.y) < EPS ? (d.y < 0.0 ? -LARGE : LARGE) : 1.0 / d.y,
        abs(d.z) < EPS ? (d.z < 0.0 ? -LARGE : LARGE) : 1.0 / d.z);
}

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float nextFloat(inout uint state) {
    if (state == 0u) state = 123456789u;
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return float(state & 0xFFFFFFu) / 16777216.0;
}

float smith(float cost, float alpha2) {
    float cost2 = cost * cost;
    float tanT2 = (1.0 - cost2) / max(1e-5, cost2);
    if (tanT2 <= 0.0) return 1.0;
    return 2.0 / (1.0 + sqrt(1.0 + alpha2 * tanT2));
}

float ggxD(float NdotH, float alpha2) {
    float d = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(1e-12, PI * d * d);
}

float bsdfPdfW(vec3 N, vec3 V, vec3 L, float alpha2, float pspec, float pdiff) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return 0.0;
    vec3 H = normalize(V + L);
    float NdotH = max(0.0, dot(N, H));
    float VdotH = max(1e-5, dot(V, H));
    float pdfSpec = ggxD(NdotH, alpha2) * NdotH / (4.0 * VdotH);
    float pdfDiff = NdotL / PI;
    return pspec * pdfSpec + pdiff * pdfDiff;
}

float misWeight(float a, float b) {
    return (a <= 0.0) ? 0.0 : a / (a + b);
}

vec3 sampleSkybox(vec3 d) {
    float u = 0.5 + (atan(d.z, d.x) / (2.0 * PI));
    float v = 0.5 - (asin(clamp(d.y, -1.0, 1.0)) / PI);
    u = clamp(u, 0.0, 0.9999);
    v = clamp(v, 0.0, 0.9999);
    int x = int(u * float(cam.skyWidth));
    int y = int(v * float(cam.skyHeight));
    int idx = y * cam.skyWidth + x;
    if (idx < 0 || idx >= cam.skyWidth * cam.skyHeight) return cam.bgColor;
    return skyPixels[idx].rgb;
}

bool rayCubeIntersect(vec3 ro, vec3 rd, vec3 invD, GPUPBRRenderData pt,
                      out float t, out vec3 normal, out vec3 hitPoint, out float tExit) {
    vec3 bMin = pt.position - pt.size * 0.5;
    vec3 bMax = pt.position + pt.size * 0.5;
    vec3 t0 = (bMin - ro) * invD;
    vec3 t1 = (bMax - ro) * invD;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    float tMin = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tMax = min(min(tmax3.x, tmax3.y), tmax3.z);
    tExit = tMax;
    if (tMax < max(0.0, tMin)) {
        t = 0.0;
        normal = vec3(0.0);
        hitPoint = ro;
        return false;
    }
    bool inside = tMin < 0.0;
    t = inside ? tMax : tMin;
    hitPoint = ro + rd * t;

    vec3 sgn = vec3(rd.x < 0.0 ? 1.0 : -1.0,
                    rd.y < 0.0 ? 1.0 : -1.0,
                    rd.z < 0.0 ? 1.0 : -1.0);
    vec3 slab = inside ? tmax3 : tmin3;
    float key = inside ? tMax : tMin;
    vec3 mask;
    if (key == slab.x)      mask = vec3(1.0, 0.0, 0.0);
    else if (key == slab.y) mask = vec3(0.0, 1.0, 0.0);
    else                    mask = vec3(0.0, 0.0, 1.0);
    normal = inside ? (sgn * mask) : (-sgn * mask);
    return true;
}

int voxelTraverse(vec3 ro, vec3 rd, vec3 invD, float maxDist,
                  out int hitIndex, out float outT, out vec3 outNormal, out vec3 outHitPoint) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF, ro, 0.0, rd, maxDist);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int ptIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            float t;
            vec3 n;
            vec3 hp;
            float tEx;
            if (rayCubeIntersect(ro, rd, invD, points[ptIdx], t, n, hp, tEx) && t >= 0.0 && t <= maxDist) {
                rayQueryGenerateIntersectionEXT(rq, t);
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionGeneratedEXT) {
        hitIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        outT = rayQueryGetIntersectionTEXT(rq, true);
        float dummyT, tEx;
        rayCubeIntersect(ro, rd, invD, points[hitIndex], dummyT, outNormal, outHitPoint, tEx);
        return 1;
    }
    hitIndex = -1;
    return 0;
}

int getMediumVoxelAt(vec3 hitPoint, vec3 normal, int targetObjectId, int ignoreIdx) {
    if (targetObjectId == -1) return -1;
    rayQueryEXT rq;
    vec3 ro = hitPoint - normal * DIST_EPSILON;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF, ro, 0.0, normal, GAP_EPSILON);
    int foundIdx = -1;
    float minDist = 1e30;
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int ptIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            
            if (ptIdx != ignoreIdx && points[ptIdx].objectId == targetObjectId) {
                foundIdx = ptIdx;
                break;
            }
        }
    }
    return foundIdx;
}

vec3 shadowTransmit(vec3 ro, vec3 rd, vec3 invD, float maxDist, int lightPtIdx) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, ro, 0.0, rd, maxDist);
    vec3 transmittance = vec3(1.0);
    if (cam.invFogRange > 0.0) transmittance *= exp(-vec3(cam.invFogRange) * maxDist);

    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int ptIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            if (ptIdx == lightPtIdx) continue;
            GPUPBRRenderData pt = points[ptIdx];
            vec3 bMin = pt.position - pt.size * 0.5;
            vec3 bMax = pt.position + pt.size * 0.5;
            vec3 t0 = (bMin - ro) * invD;
            vec3 t1 = (bMax - ro) * invD;
            vec3 tmin3 = min(t0, t1);
            vec3 tmax3 = max(t0, t1);
            float tEntry = max(max(tmin3.x, tmin3.y), tmin3.z);
            float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
            if (tExit >= max(0.0, tEntry) && tEntry <= maxDist) {
                GPUMaterial tMat = materials[pt.materialIdx];
                float r, m;
                uint sellRow;
                unpackMaterial(tMat.materialProps, r, m, sellRow);
                bool isMedGas = pt.isGas != 0;
                vec4 albColor = unpackRGBA8(pt.color);
                float ptOpacity = albColor.a;
                float ptTransmission = 1.0 - ptOpacity;
                if (isMedGas || ptTransmission > 0.01) {
                    vec3 absColor = unpackRGB8(tMat.absorption);
                    float actualTEntry = max(0.0, tEntry);
                    float actualTExit  = min(maxDist, tExit);
                    float thickness = max(0.0, actualTExit - actualTEntry);
                    if (isMedGas) {
                        float densityProxy = max(0.001, ptOpacity);
                        vec3 compColor = vec3(1.0) - albColor.rgb;
                        vec3 sigma_a = (absColor + compColor * 2.0) * densityProxy * 5.0;
                        vec3 sigma_s = albColor.rgb * densityProxy * 5.0;
                        transmittance *= exp(-(sigma_a + sigma_s) * thickness);
                    } else {
                        transmittance *= exp(-absColor * thickness);
                    }
                } else {
                    float tHit = tEntry < 0.0 ? tExit : tEntry;
                    if (tHit >= 0.0 && tHit <= maxDist) rayQueryGenerateIntersectionEXT(rq, tHit);
                }
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionGeneratedEXT)
        return vec3(0.0);
    return transmittance;
}

#endif
