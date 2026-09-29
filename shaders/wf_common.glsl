#ifndef WF_COMMON_GLSL
#define WF_COMMON_GLSL

#define PI 3.14159265359
const float EPS  = 1e-8;
const float LARGE = 1e30;
const float GAP_EPSILON  = 1e-2;
const float DIST_EPSILON = 1e-4;

const int MAX_TRANSPARENT_BOUNCES = 12;
const int MAX_VOLUMETRIC_BOUNCES  = 8;
#define DDGI_RAYS 1024
const float DDGI_HYSTERESIS = 0.80;

struct GPUMaterial {
    uint chromaticity;
    uint materialProps;
    uint absorption;
    uint albedo;
};

struct GPURenderData {
    vec3 position;
    float size;
    uint color;
    uint materialIdx;
    int  objectId;
    uint extent;
};

vec3 unpackExtent(uint e) {
    return vec3(float((e & 0x3FFu) + 1u), float(((e >> 10) & 0x3FFu) + 1u), float(((e >> 20) & 0x3FFu) + 1u));
}

const uint EXTENT_STATIC_BIT = 1u << 30;
const uint EXTENT_REUSE_BIT  = 1u << 31;
bool extentIsStatic(uint e) { return (e & EXTENT_STATIC_BIT) != 0u; }
bool extentIsReusable(uint e) { return (e & EXTENT_REUSE_BIT) != 0u; }
vec3 ptBoundsMin(GPURenderData p) { return p.position - p.size * 0.5; }
vec3 ptBoundsMax(GPURenderData p) { return p.position + p.size * 0.5 + p.size * (unpackExtent(p.extent) - vec3(1.0)); }


struct PathHot {
    vec4 o_tmax;
    vec4 dir_rough;
    vec4 thp;
    vec4 rad;
    uvec4 u0;
    ivec4 i1;
};

struct PathHit {
    float t;
    uint  hitIndex;
    float misc;
    float pad;
    uint  neeSlot;
    float varLight;
    float varBsdf;
    float pad2;
};
const uint WF_NO_SLOT = 0xFFFFFFFFu;
const uint WF_NO_HIT = 0xFFFFFFFFu;

#define PC_GET_BOUNCE(p) (int((p) & 0xFFu))
#define PC_GET_TRANS(p) (int(((p) >> 8u) & 0xFFu))
#define PC_GET_VOLUM(p) (int(((p) >> 16u) & 0xFFu))
#define PC_PACK(b,t,v) ((uint(b) & 0xFFu) | ((uint(t) & 0xFFu) << 8u) | ((uint(v) & 0xFFu) << 16u))

#define FLAG_SPECULAR 1
#define FLAG_HITFOUND 2
#define FLAG_ALIVE 4
#define HERO_SHIFT 5
#define HERO_MASK (3 << HERO_SHIFT)
#define GET_HERO(f) (((f) & HERO_MASK) >> HERO_SHIFT)
#define SET_HERO(f, h) (((f) & ~HERO_MASK) | (((h) & 3) << HERO_SHIFT))

struct ShadowRay {
    vec4 o_tmax;
    vec3 dir;
    uint slot;
    vec3 contrib;
    uint lightIdx;
    vec3 thp;
    uint pad;
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
    int sellWidth;
    int sellSecondary;
    int fogVolumeCount;
    int tileW;
    int tileH;
    int wcEnabled;
    uint wcCapacity;
    uint wcFrame;
    int wcMaxAge;
    vec3 wcOrigin;
    float wcInvCellSize;
    vec3 ddgiOrigin;
    float ddgiEnabled;
    vec3 ddgiSpacing;
    float ddgiNormalBias;
    int ddgiProbesX;
    int ddgiProbesY;
    int ddgiProbesZ;
    int ddgiIrrRes;
    int ddgiDepthRes;
    float ddgiDepthSharpness;
} cam;
bool adaptiveEnabled() { return cam.dispatchSamples >= cam.targetSamples; }

layout(std430, binding = 1) readonly buffer PointBuffer { GPURenderData points[]; };
layout(std430, binding = 2) readonly buffer MaterialBuffer { GPUMaterial materials[]; };
layout(std430, binding = 3) readonly buffer SkyboxBuffer { vec4 skyPixels[]; };
layout(std430, binding = 4) readonly buffer LightBuffer { uint emissiveIndices[]; };
layout(binding = 5) uniform accelerationStructureEXT tlas;
layout(std430, binding = 6) buffer OutputBuffer { float pixels[]; };
layout(std430, binding = 7) buffer AdaptiveBuffer { float adaptiveData[]; };
layout(std430, binding = 8) buffer PathHotBuffer { PathHot pathsHot[]; };
layout(std430, binding = 9)  buffer ExtendA { uint extendA[]; };
layout(std430, binding = 10) buffer ExtendB { uint extendB[]; };
layout(std430, binding = 11) buffer ShadeQ { uint shadeQueue[]; };
layout(std430, binding = 12) buffer ShadowQ { ShadowRay shadowQueue[]; };
layout(std430, binding = 13) buffer CounterBuf{ Counters ctr; };
layout(std430, binding = 14) buffer PathHitBuffer { PathHit pathsHit[]; };
layout(std430, binding = 15) readonly buffer SellmeierBuffer { float sellmeierLUT[]; };

struct FogVolume {
    vec4 minB;
    vec4 maxB;
    vec4 scatter;
    vec4 absorb;
};
layout(std430, binding = 16) readonly buffer FogVolumeBuffer { FogVolume fogVolumes[]; };

struct WorldCacheEntry {
    uint key;
    uint frame;
    uint sampleCount;
    uint pad0;
    vec3 irradiance;
    float pad1;
};
layout(std430, binding = 17) buffer WorldCacheBuffer { WorldCacheEntry worldCache[]; };

const uint WC_INVALID_KEY = 0u;

uint wcQuantizeNormal(vec3 n) {
    vec3 a = abs(n);
    uint major = 0u;
    if (a.y > a.x && a.y >= a.z) major = 1u;
    else if (a.z > a.x && a.z > a.y) major = 2u;
    return major * 2u + (n[major] < 0.0 ? 1u : 0u);
}

uint wcKey(ivec3 c, uint nb) {
    uint h = uint(c.x) * 73856093u;
    h ^= uint(c.y) * 19349663u;
    h ^= uint(c.z) * 83492791u;
    h ^= nb * 0x9e3779b9u;
    h ^= h >> 16;
    return h == WC_INVALID_KEY ? 1u : h;
}

ivec3 wcCell(vec3 p) {
    return ivec3(floor((p - cam.wcOrigin) * cam.wcInvCellSize));
}

uint wcSlot(uint key) {
    return key & (cam.wcCapacity - 1u);
}

bool wcLookup(vec3 p, vec3 n, out vec3 outIrradiance) {
    outIrradiance = vec3(0.0);
    if (cam.wcEnabled == 0 || cam.wcCapacity == 0u) return false;
    uint key = wcKey(wcCell(p), wcQuantizeNormal(n));
    WorldCacheEntry e = worldCache[wcSlot(key)];
    if (e.key != key || e.sampleCount == 0u) return false;
    if (int(cam.wcFrame - e.frame) > cam.wcMaxAge) return false;
    outIrradiance = e.irradiance;
    return true;
}

void wcStore(vec3 p, vec3 n, vec3 radiance) {
    if (cam.wcEnabled == 0 || cam.wcCapacity == 0u) return;
    uint key = wcKey(wcCell(p), wcQuantizeNormal(n));
    uint slot = wcSlot(key);
    uint prev = atomicCompSwap(worldCache[slot].key, WC_INVALID_KEY, key);
    if (prev != WC_INVALID_KEY && prev != key) {
        if (int(cam.wcFrame - worldCache[slot].frame) <= cam.wcMaxAge) return;
        atomicExchange(worldCache[slot].key, key);
        worldCache[slot].sampleCount = 0u;
        worldCache[slot].irradiance = vec3(0.0);
    }
    uint c = atomicAdd(worldCache[slot].sampleCount, 1u) + 1u;
    float w = 1.0 / float(min(c, 64u));
    worldCache[slot].irradiance = mix(worldCache[slot].irradiance, radiance, w);
    worldCache[slot].frame = cam.wcFrame;
}

layout(std430, binding = 18) buffer DDGIIrradianceBuffer { vec4 ddgiIrradiance[]; };
layout(std430, binding = 19) buffer DDGIDepthBuffer { vec2 ddgiDepth[]; };

vec2 octEncode(vec3 d) {
    d /= (abs(d.x) + abs(d.y) + abs(d.z));
    vec2 o = d.xy;
    if (d.z < 0.0) {
        o = (1.0 - abs(d.yx)) * vec2(d.x >= 0.0 ? 1.0 : -1.0, d.y >= 0.0 ? 1.0 : -1.0);
    }
    return o * 0.5 + 0.5;
}

vec3 octDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 d = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-d.z, 0.0);
    d.x += (d.x >= 0.0) ? -t : t;
    d.y += (d.y >= 0.0) ? -t : t;
    return normalize(d);
}

int ddgiProbeIndex(ivec3 c) {
    return (c.z * cam.ddgiProbesY + c.y) * cam.ddgiProbesX + c.x;
}

vec3 ddgiProbePosition(ivec3 c) {
    return cam.ddgiOrigin + vec3(c) * cam.ddgiSpacing;
}

vec3 ddgiSampleIrradiance(int probe, vec3 dir) {
    vec2 uv = octEncode(dir) * float(cam.ddgiIrrRes - 1);
    ivec2 t = ivec2(clamp(uv, vec2(0.0), vec2(float(cam.ddgiIrrRes - 1))));
    int base = probe * cam.ddgiIrrRes * cam.ddgiIrrRes;
    return ddgiIrradiance[base + t.y * cam.ddgiIrrRes + t.x].rgb;
}

vec2 ddgiSampleDepth(int probe, vec3 dir) {
    vec2 uv = octEncode(dir) * float(cam.ddgiDepthRes - 1);
    ivec2 t = ivec2(clamp(uv, vec2(0.0), vec2(float(cam.ddgiDepthRes - 1))));
    int base = probe * cam.ddgiDepthRes * cam.ddgiDepthRes;
    return ddgiDepth[base + t.y * cam.ddgiDepthRes + t.x];
}

vec3 ddgiIrradianceAt(vec3 p, vec3 n, vec3 viewDir) {
    if (cam.ddgiProbesX <= 0 || cam.ddgiProbesY <= 0 || cam.ddgiProbesZ <= 0) return vec3(0.0);

    vec3 biased = p + (-viewDir) * cam.ddgiNormalBias;

    vec3 grid = (biased - cam.ddgiOrigin) / cam.ddgiSpacing;
    ivec3 baseCell = ivec3(floor(grid));
    vec3 frac = grid - vec3(baseCell);

    ivec3 maxCell = ivec3(cam.ddgiProbesX, cam.ddgiProbesY, cam.ddgiProbesZ) - 1;
    if (any(lessThan(baseCell, ivec3(-1))) || any(greaterThan(baseCell, maxCell))) return vec3(0.0);

    vec3 sum = vec3(0.0);
    float wSum = 0.0;

    for (int i = 0; i < 8; ++i) {
        ivec3 offs = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 c = clamp(baseCell + offs, ivec3(0), maxCell);
        vec3 probePos = ddgiProbePosition(c);
        vec3 toProbe = probePos - biased;
        float dist = length(toProbe);
        vec3 dirToProbe = (dist > 1e-6) ? (toProbe / dist) : n;

        vec3 tri = mix(1.0 - frac, frac, vec3(offs));
        float w = tri.x * tri.y * tri.z;

        float ndl = dot(n, dirToProbe);
        w *= max(0.0, (ndl + 1.0) * 0.5);

        vec2 md = ddgiSampleDepth(ddgiProbeIndex(c), -dirToProbe);
        float mean = md.x;
        float variance = abs(md.x * md.x - md.y);
        if (dist > mean) {
            float d = dist - mean;
            float cheb = variance / (variance + d * d);
            cheb = max(cheb * cheb * cheb, 0.0);
            w *= cheb;
        }

        if (w < 1e-4) continue;
        sum += ddgiSampleIrradiance(ddgiProbeIndex(c), n) * w;
        wSum += w;
    }

    if (wSum <= 0.0) return vec3(0.0);
    return sum / wSum;
}

bool fogClip(vec3 ro, vec3 invD, float tMax, vec4 minB, vec4 maxB, out float t0, out float t1) {
    vec3 tA = (minB.xyz - ro) * invD;
    vec3 tB = (maxB.xyz - ro) * invD;
    vec3 tMin3 = min(tA, tB);
    vec3 tMax3 = max(tA, tB);
    t0 = max(0.0, max(tMin3.x, max(tMin3.y, tMin3.z)));
    t1 = min(tMax, min(tMax3.x, min(tMax3.y, tMax3.z)));
    return t1 > t0;
}

vec3 fogTransmittance(vec3 ro, vec3 rd, vec3 invD, float dist) {
    vec3 tau = vec3(0.0);
    for (int i = 0; i < cam.fogVolumeCount; ++i) {
        FogVolume fv = fogVolumes[i];
        float t0, t1;
        if (!fogClip(ro, invD, dist, fv.minB, fv.maxB, t0, t1)) continue;
        vec3 sigma_t = fv.minB.w * (fv.scatter.rgb + fv.absorb.rgb);
        tau += sigma_t * (t1 - t0);
    }
    return exp(-tau);
}

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
    if (hero == 2) return vec3(0.0, 1.0, 0.0);
    if (hero == 3) return vec3(0.0, 0.0, 1.0);
    return vec3(1.0, 1.0, 1.0);
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
    return float(pcg_hash(state) >> 8u) / 16777216.0;
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

void branchlessONB(vec3 n, out vec3 t, out vec3 b) {
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float ab = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * ab, -s * n.x);
    b = vec3(ab, s + n.y * n.y * a, -n.y);
}

vec3 sampleGGXVNDFLocal(vec3 Ve, float alpha, float u1, float u2) {
    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    float phi = 2.0 * PI * u1;
    float z = fma(1.0 - u2, 1.0 + Vh.z, -Vh.z);
    float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    vec3 c = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    vec3 Nh = c + Vh;
    return normalize(vec3(alpha * Nh.x, alpha * Nh.y, Nh.z));
}

float vndfReflectPdf(float NdotV, float NdotH, float alpha2) {
    return smith(NdotV, alpha2) * ggxD(NdotH, alpha2) / max(1e-6, 4.0 * NdotV);
}

float bsdfPdfW(vec3 N, vec3 V, vec3 L, float alpha2, float pspec, float pdiff) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return 0.0;
    float NdotV = max(1e-4, dot(N, V));
    vec3 H = normalize(V + L);
    float NdotH = max(0.0, dot(N, H));
    float pdfSpec = vndfReflectPdf(NdotV, NdotH, alpha2);
    float pdfDiff = NdotL / PI;
    return pspec * pdfSpec + pdiff * pdfDiff;
}

float misWeight(float a, float b) {
    return (a <= 0.0) ? 0.0 : a / (a + b);
}

#define RESTIR_VARIANCE_AWARE_MIS 1

float misWeightVA(float pA, float vA, float pB, float vB) {
#if RESTIR_VARIANCE_AWARE_MIS
    if (pA <= 0.0) return 0.0;
    if (pB <= 0.0) return 1.0;
    float vFloor = max(max(vA, vB) * 1e-3, 1e-8);
    float a = pA / max(vA, vFloor);
    float b = pB / max(vB, vFloor);
    float wVA = a / (a + b);
    float wBal = misWeight(pA, pB);
    const float VA_TRUST = 0.5;
    return mix(wBal, wVA, VA_TRUST);
#else
    return misWeight(pA, pB);
#endif
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

vec3 sampleCosHemisphere(vec3 N, float r1, float r2, out float pdfW) {
    float phi = 2.0 * PI * r1;
    float cosT = sqrt(max(0.0, 1.0 - r2));
    float sinT = sqrt(r2);
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tang = normalize(cross(up, N));
    vec3 bitang = cross(N, tang);
    vec3 d = tang * (sinT * cos(phi)) + bitang * (sinT * sin(phi)) + N * cosT;
    pdfW = cosT / PI;
    return normalize(d);
}

const uint WC_TICK_STRIDE = 1024u;
uint wcTick() {
    return cam.wcFrame * WC_TICK_STRIDE + uint(clamp(pc.sampleIndex, 0, int(WC_TICK_STRIDE) - 1));
}

void lightDominantFace(GPURenderData lp, vec3 refPoint, out int axis, out float sgn) {
    vec3 bMin = ptBoundsMin(lp);
    vec3 bMax = ptBoundsMax(lp);
    vec3 ext  = max(bMax - bMin, vec3(1e-6));
    vec3 d    = refPoint - (bMin + bMax) * 0.5;
    vec3 ad = abs(d) / ext;
    axis = (ad.x >= ad.y && ad.x >= ad.z) ? 0 : ((ad.y >= ad.z) ? 1 : 2);
    sgn  = (d[axis] >= 0.0) ? 1.0 : -1.0;
}

void lightSampleFace(GPURenderData lp, vec3 refPoint, float u1, float u2,
                     out vec3 pos, out vec3 nrm, out float area) {
    vec3 bMin = ptBoundsMin(lp);
    vec3 bMax = ptBoundsMax(lp);
    vec3 ext  = max(bMax - bMin, vec3(1e-6));
    int axis;
    float sgn;
    lightDominantFace(lp, refPoint, axis, sgn);
    int a1 = (axis + 1) % 3;
    int a2 = (axis + 2) % 3;

    nrm = vec3(0.0);
    nrm[axis] = sgn;
    pos = vec3(0.0);
    pos[axis] = (sgn > 0.0) ? bMax[axis] : bMin[axis];
    pos[a1] = bMin[a1] + u1 * ext[a1];
    pos[a2] = bMin[a2] + u2 * ext[a2];
    area = max(ext[a1] * ext[a2], 1e-12);
}

float lightFaceArea(GPURenderData lp, vec3 refPoint, out vec3 nrm) {
    vec3 bMin = ptBoundsMin(lp);
    vec3 bMax = ptBoundsMax(lp);
    vec3 ext  = max(bMax - bMin, vec3(1e-6));
    int axis;
    float sgn;
    lightDominantFace(lp, refPoint, axis, sgn);
    nrm = vec3(0.0);
    nrm[axis] = sgn;
    return max(ext[(axis + 1) % 3] * ext[(axis + 2) % 3], 1e-12);
}

float lightPdfW(float area, float distSq, float cosAtLight, int lightCount) {
    if (cosAtLight <= 1e-6 || area <= 0.0 || lightCount <= 0) return 0.0;
    return distSq / (cosAtLight * area * float(lightCount));
}

const float RESTIR_G_MAX = 1e6;
float geometryTerm(float ndl, float cosAtLight, float distSq) {
    if (ndl <= 0.0 || cosAtLight <= 0.0) return 0.0;
    return min(ndl * cosAtLight / max(distSq, 1e-8), RESTIR_G_MAX);
}

struct Reservoir {
    uint key;
    uint lightIdx;
    float wSum;
    float M;
    float W;
    float targetPdf;
    uint frame;
    uint nrmPacked;

    float posX;
    float posY;
    float posZ;
    float roughness;

    uint albedoPacked;
    uint metalPacked;
    float varLight;
    float varBsdf;
};
layout(std430, binding = 20) buffer ReservoirBuffer { Reservoir reservoirs[]; };

const int GBUF_STRIDE = 8;
layout(std430, binding = 21) buffer GBufferBuffer { float gbuf[]; };

const float RESTIR_M_CAP = 32.0;
const int RESTIR_CANDIDATES = 8;
const int RESTIR_SPATIAL_TAPS = 3;
const int RESTIR_MAX_SOURCES  = 2 + RESTIR_SPATIAL_TAPS;

uint packNormalOct(vec3 n) {
    vec2 e = octEncode(normalize(n));
    return (uint(clamp(e.x, 0.0, 1.0) * 65535.0) & 0xFFFFu)
         | ((uint(clamp(e.y, 0.0, 1.0) * 65535.0) & 0xFFFFu) << 16u);
}

vec3 unpackNormalOct(uint p) {
    vec2 e = vec2(float(p & 0xFFFFu), float((p >> 16u) & 0xFFFFu)) / 65535.0;
    return octDecode(e);
}

uint packRGBA8f(vec4 c) {
    uvec4 q = uvec4(clamp(c, vec4(0.0), vec4(1.0)) * 255.0 + 0.5);
    return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

struct SurfCtx {
    vec3  pos;
    vec3  n;
    vec3  V;
    vec3  albedo;
    vec3  F0;
    float alpha2;
    float opacity;
    float metallic;
    float valid;
};

SurfCtx surfFromReservoir(Reservoir r) {
    SurfCtx s;
    s.pos = vec3(r.posX, r.posY, r.posZ);
    s.n   = unpackNormalOct(r.nrmPacked);
    vec4 a = unpackRGBA8(r.albedoPacked);
    s.albedo   = a.rgb;
    s.opacity  = a.a;
    s.metallic = float(r.metalPacked & 0xFFu) / 255.0;
    s.F0 = mix(vec3(0.04), s.albedo, s.metallic);
    float alpha = max(1e-5, r.roughness * r.roughness);
    s.alpha2 = alpha * alpha;
    s.V = s.n;
    s.opacity = a.a;
    s.valid = 1.0;
    return s;
}

float reservoirTargetAt(SurfCtx s, uint lightIdx) {
    if (s.valid == 0.0) return 0.0;
    if (lightIdx >= uint(points.length())) return 0.0;
    GPURenderData lp = points[lightIdx];
    GPUMaterial  lm  = materials[lp.materialIdx];
    if (lm.chromaticity == 0u) return 0.0;

    vec3 lnrm;
    float area = lightFaceArea(lp, s.pos, lnrm);
    vec3 bMin = ptBoundsMin(lp), bMax = ptBoundsMax(lp);
    vec3 lpos = (bMin + bMax) * 0.5 + lnrm * (0.5 * abs(dot(bMax - bMin, lnrm)));

    vec3 to = lpos - s.pos;
    float d2 = dot(to, to);
    if (d2 < 1e-12) return 0.0;
    float d = sqrt(d2);
    vec3 L = to / d;

    float ndl = dot(s.n, L);
    if (ndl <= 0.0) return 0.0;
    float cosAtLight = max(0.0, dot(lnrm, -L));
    float G = geometryTerm(ndl, cosAtLight, d2);
    if (G <= 0.0) return 0.0;

    vec4 la = unpackRGBA8(lp.color);
    vec3 Le = la.rgb * unpackRGB9E5(lm.chromaticity);

    float NdotV = max(1e-4, dot(s.n, s.V));
    vec3 H = normalize(s.V + L);
    float NdotH = max(0.0, dot(s.n, H));
    float VdotH = max(0.0, dot(s.V, H));
    vec3 F = s.F0 + (vec3(1.0) - s.F0) * pow(1.0 - min(1.0, VdotH), 5.0);
    vec3 diff = s.albedo * (vec3(1.0) - F) / PI * s.opacity * (1.0 - s.metallic);
    float D = ggxD(NdotH, s.alpha2);
    float Gs = smith(NdotV, s.alpha2) * smith(ndl, s.alpha2);
    vec3 spec = F * (D * Gs / max(1e-5, 4.0 * NdotV * ndl));

    return dot(Le * (diff + spec) * G, vec3(0.2126, 0.7152, 0.0722));
}

float reservoirTarget(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

bool reservoirUpdate(inout Reservoir r, uint lightIdx, float w, float pHat, inout uint rng) {
    r.M += 1.0;
    if (w <= 0.0) return false;
    r.wSum += w;
    if (nextFloat(rng) < w / r.wSum) {
        r.lightIdx = lightIdx;
        r.targetPdf = pHat;
        return true;
    }
    return false;
}

void reservoirFinalize(inout Reservoir r) {
    if (r.targetPdf <= 0.0 || r.M <= 0.0) {
        r.W = 0.0;
        return;
    }
    r.W = r.wSum / (r.M * r.targetPdf);
}

void reservoirInit(out Reservoir r) {
    r.key = 0u;
    r.lightIdx = 0u;
    r.wSum = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.targetPdf = 0.0;
    r.frame = 0u;
    r.nrmPacked = 0u;
    r.posX = 0.0;
    r.posY = 0.0;
    r.posZ = 0.0;
    r.roughness = 0.0;
    r.albedoPacked = 0u;
    r.metalPacked = 0u;
    r.varLight = 0.0;
    r.varBsdf = 0.0;
}

uint reservoirSlot(vec3 p, vec3 n, out uint outKey) {
    outKey = wcKey(wcCell(p), wcQuantizeNormal(n));
    return wcSlot(outKey);
}

bool reservoirFetch(vec3 p, vec3 n, vec3 planeP, vec3 planeN, float planeTol,
                    out Reservoir r) {
    reservoirInit(r);
    if (cam.wcCapacity == 0u) return false;
    uint key;
    uint slot = reservoirSlot(p, n, key);
    Reservoir e = reservoirs[slot];
    if (e.key != key) return false;
    if (e.M <= 0.0 || e.W <= 0.0 || e.targetPdf <= 0.0) return false;
    if (int((wcTick() - e.frame) / WC_TICK_STRIDE) > cam.wcMaxAge) return false;
    if (e.lightIdx >= uint(points.length())) return false;
    vec3 en = unpackNormalOct(e.nrmPacked);
    if (dot(en, planeN) < 0.906) return false;
    vec3 ep = vec3(e.posX, e.posY, e.posZ);
    if (abs(dot(ep - planeP, planeN)) > planeTol) return false;
    r = e;
    return true;
}

void reservoirCombineGRIS(out Reservoir dst,
                          Reservoir srcR[RESTIR_MAX_SOURCES],
                          SurfCtx srcS[RESTIR_MAX_SOURCES],
                          int count, inout uint rng) {
    reservoirInit(dst);
    if (count <= 0) return;
    float mConf[RESTIR_MAX_SOURCES];
    float mCap = max(srcR[0].M, 1.0);
    for (int i = 0; i < count; ++i) {
        mConf[i] = min(srcR[i].M, mCap);
    }

    for (int j = 0; j < count; ++j) {
        if (srcR[j].M <= 0.0) continue;
        uint X = srcR[j].lightIdx;

        float pvals[RESTIR_MAX_SOURCES];
        float denom = 0.0;
        for (int i = 0; i < count; ++i) {
            pvals[i] = (srcR[i].M > 0.0) ? reservoirTargetAt(srcS[i], X) : 0.0;
            denom += mConf[i] * pvals[i];
        }

        dst.M += srcR[j].M;
        if (denom <= 0.0 || srcR[j].W <= 0.0) continue;

        float mj = mConf[j] * pvals[j] / denom;
        float w  = mj * pvals[0] * srcR[j].W;
        if (w <= 0.0) continue;

        dst.wSum += w;
        if (nextFloat(rng) < w / dst.wSum) {
            dst.lightIdx  = X;
            dst.targetPdf = pvals[0];
        }
    }

    dst.W = (dst.targetPdf > 0.0) ? dst.wSum / dst.targetPdf : 0.0;
}

const float VA_EMA = 0.25;
const float VA_FLOOR = 1e-6;

void vaUpdateLight(uint slot, float moment) {
    if (slot == WF_NO_SLOT || cam.wcCapacity == 0u) return;
    float prev = max(reservoirs[slot].varLight, VA_FLOOR);
    reservoirs[slot].varLight = mix(prev, max(moment, VA_FLOOR), VA_EMA);
}

void vaUpdateBsdf(uint slot, float moment) {
    if (slot == WF_NO_SLOT || cam.wcCapacity == 0u) return;
    float prev = max(reservoirs[slot].varBsdf, VA_FLOOR);
    reservoirs[slot].varBsdf = mix(prev, max(moment, VA_FLOOR), VA_EMA);
}

bool rayCubeIntersect(vec3 ro, vec3 rd, vec3 invD, GPURenderData pt,
                      out float t, out vec3 normal, out vec3 hitPoint, out float tExit) {
    vec3 bMin = ptBoundsMin(pt);
    vec3 bMax = ptBoundsMax(pt);
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
    if (key == slab.x) mask = vec3(1.0, 0.0, 0.0);
    else if (key == slab.y) mask = vec3(0.0, 1.0, 0.0);
    else mask = vec3(0.0, 0.0, 1.0);
    normal = (inside ? -sgn : sgn) * mask;
    return true;
}

int voxelTraverse(vec3 ro, vec3 rd, vec3 invD, float maxDist,
                  out int hitIndex, out float outT) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF, ro, 0.0, rd, maxDist);
    float tBest = maxDist;
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int ptIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            GPURenderData cand = points[ptIdx];
            vec3 t0 = (ptBoundsMin(cand) - ro) * invD;
            vec3 t1 = (ptBoundsMax(cand) - ro) * invD;
            vec3 tmin3 = min(t0, t1);
            vec3 tmax3 = max(t0, t1);
            float tMin = max(max(tmin3.x, tmin3.y), tmin3.z);
            float tMax = min(min(tmax3.x, tmax3.y), tmax3.z);
            if (tMax < max(0.0f, tMin)) continue;
            float t = (tMin < 0.0f) ? tMax : tMin;
            if (t >= 0.0f && t < tBest) {
                rayQueryGenerateIntersectionEXT(rq, t);
                tBest = t;
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionGeneratedEXT) {
        hitIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        outT = rayQueryGetIntersectionTEXT(rq, true);
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
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF, ro, 0.0f, rd, maxDist);
    float tBest = maxDist;
    vec3 transmittance = vec3(1.0);
    if (cam.invFogRange > 0.0) transmittance *= exp(-vec3(cam.invFogRange) * maxDist);

    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int ptIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            if (ptIdx == lightPtIdx) continue;
            GPURenderData pt = points[ptIdx];
            vec3 t0 = (ptBoundsMin(pt) - ro) * invD;
            vec3 t1 = (ptBoundsMax(pt) - ro) * invD;
            vec3 tmin3 = min(t0, t1);
            vec3 tmax3 = max(t0, t1);
            float tEntry = max(max(tmin3.x, tmin3.y), tmin3.z);
            float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
            if (tExit >= max(0.0f, tEntry) && tEntry <= maxDist) {
                GPUMaterial tMat = materials[pt.materialIdx];
                float r, m;
                uint sellRow;
                unpackMaterial(tMat.materialProps, r, m, sellRow);
                vec4 albColor = unpackRGBA8(pt.color);
                float ptOpacity = albColor.a;
                float ptTransmission = 1.0 - ptOpacity;
                if (ptTransmission > 0.01) {
                    vec3 absColor = unpackRGB9E5(tMat.absorption);
                    float actualTEntry = max(0.0, tEntry);
                    float actualTExit  = min(maxDist, tExit);
                    float thickness = max(0.0, actualTExit - actualTEntry);
                    transmittance *= exp(-absColor * thickness);
                } else {
                    float tHit = tEntry < 0.0 ? tExit : tEntry;
                    if (tHit >= 0.0 && tHit < tBest) {
                        rayQueryGenerateIntersectionEXT(rq, tHit);
                        tBest = tHit;
                    }
                }
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionGeneratedEXT)
        return vec3(0.0);
    return transmittance;
}

#endif
