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
    float emittance;
    uint  materialProps;
    uint  absorption;
    uint  padding;
};

struct GPUPBRRenderData {
    vec3 position;
    float size;
    uint color;
    uint materialIdx;
    int  objectId;
    uint isGas;
};

struct Path {
    vec4 o_tmax;
    vec4 dir_rough;
    vec4 thp;
    vec4 rad;
    vec4 hitN_t;
    vec4 hitP_idx;
    uvec4 u0;
    ivec4 i1;
    vec4 misc;
};

#define PC_GET_BOUNCE(p)  (int((p) & 0xFFu))
#define PC_GET_TRANS(p)   (int(((p) >> 8u) & 0xFFu))
#define PC_GET_VOLUM(p)   (int(((p) >> 16u) & 0xFFu))
#define PC_PACK(b,t,v)    ((uint(b) & 0xFFu) | ((uint(t) & 0xFFu) << 8u) | ((uint(v) & 0xFFu) << 16u))

#define FLAG_SPECULAR 1
#define FLAG_HITFOUND 2
#define FLAG_ALIVE    4

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
} cam;

layout(std430, binding = 1) readonly buffer PointBuffer    { GPUPBRRenderData points[]; };
layout(std430, binding = 2) readonly buffer MaterialBuffer { GPUMaterial materials[]; };
layout(std430, binding = 3) readonly buffer SkyboxBuffer   { vec4 skyPixels[]; };
layout(std430, binding = 4) readonly buffer LightBuffer    { uint emissiveIndices[]; };
layout(binding = 5) uniform accelerationStructureEXT tlas;
layout(std430, binding = 6) buffer OutputBuffer   { float pixels[]; };
layout(std430, binding = 7) buffer AdaptiveBuffer { float adaptiveData[]; };
layout(std430, binding = 8) buffer PathBuffer     { Path paths[]; };
layout(std430, binding = 9)  buffer ExtendA   { uint extendA[]; };
layout(std430, binding = 10) buffer ExtendB   { uint extendB[]; };
layout(std430, binding = 11) buffer ShadeQ    { uint shadeQueue[]; };
layout(std430, binding = 12) buffer ShadowQ   { ShadowRay shadowQueue[]; };
layout(std430, binding = 13) buffer CounterBuf{ Counters ctr; };

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

void unpackMaterial(uint m, out float roughness, out float metallic, out float ior) {
    roughness = float(m & 0xFF) / 255.0;
    metallic  = float((m >> 8) & 0xFF) / 255.0;
    ior       = (float((m >> 16) & 0xFF) / 255.0) * 1.5 + 1.0;
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
    t = tMin < 0.0 ? tMax : tMin;
    hitPoint = ro + rd * t;
    vec3 dMin = abs(hitPoint - bMin);
    vec3 dMax = abs(hitPoint - bMax);
    vec3 minmin = min(dMin, dMax);
    float mind = min(min(minmin.x, minmin.y), minmin.z);
    normal = vec3(0.0);
    if      (mind == dMin.x) normal.x = -1.0;
    else if (mind == dMax.x) normal.x =  1.0;
    else if (mind == dMin.y) normal.y = -1.0;
    else if (mind == dMax.y) normal.y =  1.0;
    else if (mind == dMin.z) normal.z = -1.0;
    else                     normal.z =  1.0;
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
                float r, m, ior;
                unpackMaterial(tMat.materialProps, r, m, ior);
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
