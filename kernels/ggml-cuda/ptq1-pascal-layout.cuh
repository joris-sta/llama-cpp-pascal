#pragma once
#include "common.cuh"

// Internal GEMV-only operand: 32 planes of Q8 words followed by four planes
// of (FP16 scale, exact signed INT16 quant sum). K-block lanes read adjacent
// words. Neither the GGUF weights nor the generic/MMQ Q8 layout changes.
static __host__ __device__ inline int ptq1_pascal_pitch(int k) { return ((k/128+31)/32)*32; }
static inline size_t ptq1_pascal_bytes(int k, int columns) {
    return size_t(36) * ptq1_pascal_pitch(k) * columns * sizeof(uint32_t);
}

static __device__ __forceinline__ void ptq1_pascal_store_q8(
        uint32_t * dst, int pitch, int column, int element, int q, float d) {
    const int lane = threadIdx.x & 31;
    const int word = ((element & 127) >> 2);
    const int kb = element / 128;
    const uint32_t packed = (uint32_t(q) & 255u) |
        ((uint32_t(__shfl_down_sync(0xffffffff, q, 1)) & 255u) << 8) |
        ((uint32_t(__shfl_down_sync(0xffffffff, q, 2)) & 255u) << 16) |
        ((uint32_t(__shfl_down_sync(0xffffffff, q, 3)) & 255u) << 24);
    int sum = warp_reduce_sum(q);
    if ((lane & 3) == 0) dst[(size_t(column)*36 + word)*pitch + kb] = packed;
    if (lane == 0) {
        const uint32_t meta = uint32_t(__half_as_ushort(__float2half(d))) | (uint32_t(uint16_t(sum)) << 16);
        dst[(size_t(column)*36 + 32 + ((element & 127) >> 5))*pitch + kb] = meta;
    }
}

void quantize_ptq1_pascal_cuda(const float * x, uint32_t * dst,
        int k, int columns, int64_t stride, cudaStream_t stream);
