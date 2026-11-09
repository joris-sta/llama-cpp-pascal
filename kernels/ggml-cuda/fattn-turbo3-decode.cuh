#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Pascal register attention: each warp owns a shape-selected group of query
// heads. K/V are decoded once per warp and shared by those heads. Query columns
// are independent CUDA blocks, including short resumed prompt suffixes.
static __device__ __forceinline__ float turbo3_decode_coefficient(const char * row, int j) {
    const auto & block = reinterpret_cast<const block_turbo3_0 *>(row)[j / QK_TURBO3];
    const int p = j % QK_TURBO3;
    const int code = ((block.qs[p/4] >> (2*(p%4))) & 3) | (((block.signs[p/8] >> (p%8)) & 1) << 2);
    return d_turbo_centroids_3bit_fattn[code] * __half2float(block.norm);
}

template<int D, int G, int Warps, bool Softcap>
__launch_bounds__(32*Warps, 2)
static __global__ void flash_attn_turbo3_register(
        const char * Q_ptr, const char * K_ptr, const char * V_ptr,
        const char * mask_ptr, const char * sinks_ptr, const int * KV_max_ptr,
        float * dst, float2 * dst_meta,
        const float scale, const float max_bias, const float m0, const float m1,
        const uint32_t n_head_log2, const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
        const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
        const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    GGML_UNUSED_VARS(KV_max_ptr,max_bias,m0,m1,n_head_log2,ne00,ne03,ne10,ne13,ne31,ne32,ne33,nb32);
    const int lane = threadIdx.x, warp = threadIdx.y, column = blockIdx.x;
    constexpr int J = D/32;
    const int groups = ne02/G, seq = blockIdx.z/groups, head0 = (blockIdx.z%groups)*G;
    const int kv_head = head0/(ne02/ne12);
    const char * kr = K_ptr + int64_t(seq)*nb13 + kv_head*nb12;
    const char * vr = V_ptr + int64_t(seq)*nb23 + kv_head*nb22;
    const half * mask = reinterpret_cast<const half *>(mask_ptr + int64_t(seq)*nb33 + int64_t(column)*nb31);
    float query[G][J], output[G][J] = {};
    float maximum[G];
    float denominator[G] = {};
#pragma unroll
    for (int h = 0; h < G; ++h) {
        maximum[h] = -INFINITY;
        const float * qp = reinterpret_cast<const float *>(Q_ptr + int64_t(seq)*nb03 + (head0+h)*nb02 + int64_t(column)*nb01);
#pragma unroll
        for (int j = 0; j < J; ++j) query[h][j] = qp[lane+32*j];
    }
    for (int pos = blockIdx.y*Warps+warp; pos < ne11; pos += gridDim.y*Warps) {
        const float mask_value = __half2float(mask[pos]);
        if (mask_value == -INFINITY) continue;
        float key[J], value[J];
#pragma unroll
        for (int j = 0; j < J; ++j) {
            key[j] = turbo3_decode_coefficient(kr + int64_t(pos)*nb11,lane+32*j);
            value[j] = turbo3_decode_coefficient(vr + int64_t(pos)*nb21,lane+32*j);
#if TURBO3_SKEW_EXP >= 2
            key[j] *= d_turbo3_ss_k_fattn[(lane+32*j)%128];
            value[j] *= d_turbo3_ss_v_fattn[(lane+32*j)%128];
#endif
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float score = 0;
#pragma unroll
            for (int j = 0; j < J; ++j) score = fmaf(query[h][j],key[j],score);
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) score += __shfl_xor_sync(0xffffffffu,score,offset);
            score *= scale;
            if constexpr (Softcap) score = logit_softcap*tanhf(score);
            score += mask_value;
            const float next = fmaxf(maximum[h],score);
            const float rescale = next > maximum[h] ? expf(maximum[h]-next) : 1.0f;
            const float weight = expf(score-next);
            maximum[h] = next;
            denominator[h] = fmaf(denominator[h],rescale,weight);
#pragma unroll
            for (int j = 0; j < J; ++j) output[h][j] = fmaf(value[j],weight,output[h][j]*rescale);
        }
    }
    __shared__ float maxima[Warps][G], sums[Warps][G];
    __shared__ float partial[Warps][G][D];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        if (lane == 0) { maxima[warp][h] = maximum[h]; sums[warp][h] = denominator[h]; }
#pragma unroll
        for (int j = 0; j < J; ++j) partial[warp][h][lane+32*j] = output[h][j];
    }
    __syncthreads();
    // Warps write distinct query heads after the CTA-wide merge.
    for (int h = warp; h < G; h += Warps) {
        float top = -INFINITY;
#pragma unroll
        for (int w = 0; w < Warps; ++w) top = fmaxf(top,maxima[w][h]);
        const float sink = sinks_ptr && blockIdx.y == 0 ? reinterpret_cast<const float *>(sinks_ptr)[head0+h] : -INFINITY;
        top = fmaxf(top,sink);
        float sum = sink != -INFINITY ? expf(sink-top) : 0;
        float combined[J] = {};
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float weight = sums[w][h] > 0 ? expf(maxima[w][h]-top) : 0;
            sum += sums[w][h]*weight;
#pragma unroll
            for (int j = 0; j < J; ++j) combined[j] = fmaf(partial[w][h][lane+32*j],weight,combined[j]);
        }
        const int row = ((seq*int(ne01.z)+column)*ne02 + head0+h)*gridDim.y + blockIdx.y;
        const float normalize = gridDim.y == 1 && sum > 0 ? 1.0f/sum : 1.0f;
#pragma unroll
        for (int j = 0; j < J; ++j) dst[row*D+lane+32*j] = combined[j]*normalize;
        if (gridDim.y > 1 && lane == 0) dst_meta[row] = make_float2(top,sum);
    }
}



#endif // NVIDIA CUDA
