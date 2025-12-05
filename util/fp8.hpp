// fp8.hpp
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

class fp8_e4m3 {
private:
    uint8_t data;

public:
    // Constructors
    __host__ __device__ fp8_e4m3() : data(0) {}
    
    __host__ __device__ explicit fp8_e4m3(uint8_t val) : data(val) {}
    
    // Conversion from float32
    __host__ __device__ explicit fp8_e4m3(float f) {
        #ifdef __CUDACC__
        data = float_to_fp8(f);
        #else
        data = cpu_float_to_fp8(f);
        #endif
    }
    
    // Conversion from float16 (CUDA only)
    #ifdef __CUDACC__
    __host__ __device__ explicit fp8_e4m3(__half h) {
        data = half_to_fp8(h);
    }
    #endif
    
    // Conversion to float32
    __host__ __device__ operator float() const {
        #ifdef __CUDACC__
        return fp8_to_float(data);
        #else
        return cpu_fp8_to_float(data);
        #endif
    }
    
    // Arithmetic operators
    __host__ __device__ fp8_e4m3 operator+(const fp8_e4m3& other) const {
        return fp8_e4m3(float(*this) + float(other));
    }
    
    __host__ __device__ fp8_e4m3 operator-(const fp8_e4m3& other) const {
        return fp8_e4m3(float(*this) - float(other));
    }
    
    __host__ __device__ fp8_e4m3 operator*(const fp8_e4m3& other) const {
        return fp8_e4m3(float(*this) * float(other));
    }
    
    __host__ __device__ fp8_e4m3 operator/(const fp8_e4m3& other) const {
        return fp8_e4m3(float(*this) / float(other));
    }
    
    // Compound assignment operators
    __host__ __device__ fp8_e4m3& operator+=(const fp8_e4m3& other) {
        *this = fp8_e4m3(float(*this) + float(other));
        return *this;
    }
    
    __host__ __device__ fp8_e4m3& operator-=(const fp8_e4m3& other) {
        *this = fp8_e4m3(float(*this) - float(other));
        return *this;
    }
    
    __host__ __device__ fp8_e4m3& operator*=(const fp8_e4m3& other) {
        *this = fp8_e4m3(float(*this) * float(other));
        return *this;
    }
    
    __host__ __device__ fp8_e4m3& operator/=(const fp8_e4m3& other) {
        *this = fp8_e4m3(float(*this) / float(other));
        return *this;
    }
    
    // Comparison operators
    __host__ __device__ bool operator==(const fp8_e4m3& other) const {
        // Handle NaN and ±0.0 cases
        if ((data & 0x7F) == 0x7F) return false; // NaN
        if (data == other.data) return true;
        return false;
    }
    
    __host__ __device__ bool operator!=(const fp8_e4m3& other) const {
        return !(*this == other);
    }
    
    __host__ __device__ bool operator<(const fp8_e4m3& other) const {
        return float(*this) < float(other);
    }
    
    __host__ __device__ bool operator>(const fp8_e4m3& other) const {
        return float(*this) > float(other);
    }
    
    __host__ __device__ bool operator<=(const fp8_e4m3& other) const {
        return float(*this) <= float(other);
    }
    
    __host__ __device__ bool operator>=(const fp8_e4m3& other) const {
        return float(*this) >= float(other);
    }
    
    // Get raw data
    __host__ __device__ uint8_t get_raw() const { return data; }
    
    // Special values
    __host__ __device__ static fp8_e4m3 zero() { return fp8_e4m3(0x00); }
    __host__ __device__ static fp8_e4m3 one() { return fp8_e4m3(0x3C); } // 1.0
    __host__ __device__ static fp8_e4m3 nan() { return fp8_e4m3(0x7F); }
    __host__ __device__ static fp8_e4m3 inf() { return fp8_e4m3(0x78); } // +inf
    __host__ __device__ static fp8_e4m3 neg_inf() { return fp8_e4m3(0xF8); } // -inf
    
    // Memory operations
    __host__ __device__ static void memcpy(void* dst, const void* src, size_t count) {
        ::memcpy(dst, src, count);
    }
    
    __host__ __device__ static void memset(void* ptr, int value, size_t count) {
        ::memset(ptr, value, count);
    }
    
private:
    // CPU implementation (fast bit manipulation)
    __host__ __device__ static uint8_t cpu_float_to_fp8(float f) {
        uint32_t f_bits;
        memcpy(&f_bits, &f, sizeof(float));
        
        uint32_t sign = (f_bits >> 31) & 0x1;
        int32_t exp = ((f_bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = f_bits & 0x7FFFFF;
        
        // Handle special cases
        if (exp == 128) { // NaN or Inf
            return (sign << 7) | 0x7F; // Preserve sign for NaN/Inf
        }
        
        // Denormal handling
        if (exp < -6) {
            return sign << 7; // Underflow to zero
        }
        
        // Clamp exponent to e4m3 range [-6, 7]
        if (exp > 7) {
            return (sign << 7) | 0x78; // Overflow to inf
        }
        
        // Convert to fp8 format
        uint32_t fp8_exp = (exp + 6) & 0xF; // Bias: -6 -> 0, 7 -> 13
        uint32_t fp8_mant = mantissa >> 20; // Keep top 3 bits
        
        // Round to nearest even
        uint32_t rounding_bit = (mantissa >> 19) & 1;
        uint32_t sticky_bits = (mantissa & 0x7FFFF) ? 1 : 0;
        if (rounding_bit && (fp8_mant & 1 || sticky_bits)) {
            fp8_mant++;
            if (fp8_mant > 0x7) { // Mantissa overflow
                fp8_mant = 0;
                fp8_exp++;
                if (fp8_exp > 0xF) { // Exponent overflow
                    return (sign << 7) | 0x78; // Infinity
                }
            }
        }
        
        return (sign << 7) | (fp8_exp << 3) | (fp8_mant & 0x7);
    }
    
    __host__ __device__ static float cpu_fp8_to_float(uint8_t fp8) {
        uint32_t sign = (fp8 >> 7) & 0x1;
        uint32_t exp = (fp8 >> 3) & 0xF;
        uint32_t mant = fp8 & 0x7;
        
        // Handle special cases
        if (exp == 0xF) { // NaN or Inf
            uint32_t f_bits = (sign << 31) | (0xFF << 23) | (mant << 20);
            float result;
            memcpy(&result, &f_bits, sizeof(float));
            return result;
        }
        
        if (exp == 0) {
            // Denormal/subnormal
            if (mant == 0) return sign ? -0.0f : 0.0f;
            // Convert denormal
            exp = -6;
            mant = mant << 1;
        } else {
            exp -= 6; // Remove bias
        }
        
        // Convert to float32
        uint32_t f_exp = (exp + 127) & 0xFF;
        uint32_t f_mant = mant << 20;
        uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mant;
        
        float result;
        memcpy(&result, &f_bits, sizeof(float));
        return result;
    }
    
    // CUDA implementation (using intrinsics when available)
    #ifdef __CUDACC__
    __device__ static uint8_t float_to_fp8(float f) {
        #if __CUDA_ARCH__ >= 890  // Hopper+ has native FP8 support
        return __float_to_fp8_rn(f);
        #else
        return cpu_float_to_fp8(f);
        #endif
    }
    
    __device__ static float fp8_to_float(uint8_t fp8) {
        #if __CUDA_ARCH__ >= 890
        return __fp8_to_float(fp8);
        #else
        return cpu_fp8_to_float(fp8);
        #endif
    }
    
    __device__ static uint8_t half_to_fp8(__half h) {
        return float_to_fp8(__half2float(h));
    }
    #else
    // For non-CUDA, use CPU versions
    __host__ __device__ static uint8_t float_to_fp8(float f) {
        return cpu_float_to_fp8(f);
    }
    
    __host__ __device__ static float fp8_to_float(uint8_t fp8) {
        return cpu_fp8_to_float(fp8);
    }
    #endif
};

// Vectorized operations for performance
namespace fp8_ops {
    // Convert array of floats to fp8 (efficient batch conversion)
    static void convert_float_to_fp8(uint8_t* dst, const float* src, size_t count) {
        #pragma omp parallel for simd if(count > 1024)
        for (size_t i = 0; i < count; ++i) {
            dst[i] = fp8_e4m3(src[i]).get_raw();
        }
    }
    
    // Convert array of fp8 to floats
    static void convert_fp8_to_float(float* dst, const uint8_t* src, size_t count) {
        #pragma omp parallel for simd if(count > 1024)
        for (size_t i = 0; i < count; ++i) {
            dst[i] = fp8_e4m3(src[i]);
        }
    }
    
    // Direct memory operations
    static void memset_fp8(void* ptr, fp8_e4m3 value, size_t count) {
        uint8_t val = value.get_raw();
        ::memset(ptr, val, count);
    }
}