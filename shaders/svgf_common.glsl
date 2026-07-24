// Shared declarations for the SVGF denoiser passes.
//
// Buffer strides used across the SVGF passes:
//   PIX_STRIDE   (outBuffer)      : [0..2] radiance sum, [3] depth sum, [4] objectId
//   GBUF_STRIDE  (gbufferBuffer)  : [0..2] demodulation albedo, [3..5] normal, [6..7] unused
//   HIST_STRIDE  (svgfHist*)      : [0..2] history colour (demodulated irradiance)
//                                   [3] first luminance moment, [4] second luminance moment
//                                   [5] history length (frames), [6] depth, [7] objectId
//                                   [8..10] normal, [11] unused
//   variance     (svgfVar*)       : 1 float per pixel
#ifndef SVGF_COMMON_GLSL
#define SVGF_COMMON_GLSL

const int PIX_STRIDE  = 5;
const int GBUF_STRIDE = 8;
const int HIST_STRIDE = 12;
// Colour travels in the PIX_STRIDE layout; variance rides in its own buffer.

// Anything at or beyond this distance is treated as background/sky.
const float SVGF_SKY_DEPTH = 999000.0;

float svgfLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

bool svgfIsSky(float depth) { return depth >= SVGF_SKY_DEPTH; }

///@brief Guards against NaN/Inf leaking out of the path tracer into the history.
bool svgfInvalid(vec3 c) {
    return any(isnan(c)) || any(isinf(c)) || any(lessThan(c, vec3(-1e-3)));
}

///@brief Albedo used to demodulate radiance. Floored so dark surfaces do not explode.
vec3 svgfAlbedoFloor(vec3 a) { return max(a, vec3(0.02)); }

#endif
