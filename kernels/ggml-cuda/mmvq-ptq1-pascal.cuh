#pragma once
#include "ptq1-pascal-layout.cuh"
#include "unary.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <initializer_list>

struct ptq1_pascal_fusion {
    const block_ptq1_0 * gate = nullptr;
    const float * bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op op = GGML_GLU_OP_SWIGLU;
    float limit = 0;
};

// Streaming weights bypass the GP104 texture/L1 cache. Reused activation
// words and metadata use LDG explicitly; no global -dlcm=ca build flag.
static __device__ __forceinline__ uint32_t ptq1_pascal_weight(const uint32_t * p) {
    uint32_t v;
    asm("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template<bool Raw, bool Cache>
static __device__ __forceinline__ float ptq1_pascal_dot(
        const block_ptq1_0 * w, const uint32_t * a, int pitch, int kb) {
    int acc[4] = {};
    const uint32_t * wp = (const uint32_t *) w;
    uint32_t packed = ptq1_pascal_weight(wp);
    // Software lookahead: issue the next packed-weight load before the current
    // word's five multiply/unpack/DP4A steps. Four independent INT32 chains.
#pragma unroll
    for (int g = 0; g < 6; ++g) {
        const uint32_t next = ptq1_pascal_weight(wp + g + 1);
        uint32_t lo = __byte_perm(packed, 0, 0x4140);
        uint32_t hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t wl = lo * 3, wh = hi * 3;
            lo = wl & 0x00ff00ff; hi = wh & 0x00ff00ff;
            int trits = __byte_perm(wl, wh, 0x7531);
            if constexpr (!Raw) trits = __vsub4(trits, 0x01010101);
            const int e = g < 4 ? t*16 + 4*g : 80 + t*8 + 4*(g-4);
            const uint32_t * ap = a + (e/4)*pitch + kb;
            const int q = Cache ? __ldg(ap) : ptq1_pascal_weight(ap);
            acc[e/32] = ggml_cuda_dp4a(trits, q, acc[e/32]);
        }
        packed = next;
    }
    uint32_t v = (packed & 255u) | ((packed & 0xff00u) << 8);
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v*3; v = w0 & 0x00ff00ff;
        const uint32_t w1 = v*3; v = w1 & 0x00ff00ff;
        int trits = __byte_perm(w0, w1, 0x7531);
        if constexpr (!Raw) trits = __vsub4(trits, 0x01010101);
        const uint32_t * ap = a + (30+t/2)*pitch + kb;
        const int q = Cache ? __ldg(ap) : ptq1_pascal_weight(ap);
        acc[3] = ggml_cuda_dp4a(trits, q, acc[3]);
    }
    float result = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t * ap = a + (32+j)*pitch + kb;
        const uint32_t meta = Cache ? __ldg(ap) : ptq1_pascal_weight(ap);
        if constexpr (Raw) acc[j] -= int(int16_t(meta >> 16));
        result += __half2float(__ushort_as_half(meta & 65535)) * float(acc[j]);
    }
    return __half2float(__ushort_as_half(packed >> 16)) * result;
}

// Flat distributes the complete (row, K-block) space; warp ownership gives
// each warp a row and strides lanes through K. Both retain deterministic
// per-row reduction order independent of CTA geometry and column count.
template<int Threads, int Rows, bool Flat, bool Raw = true, bool Cache = true>
__launch_bounds__(Threads, 2)
static __global__ void mul_mat_vec_ptq1_pascal(const block_ptq1_0 * weights,
        const uint32_t * activations, float * dst, int kblocks, int rows,
        int row_stride, int dst_stride, int pitch, ptq1_pascal_fusion fusion) {
    extern __shared__ float partial[];
    float * pg = partial + Rows*kblocks;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int row0 = blockIdx.x*Rows;
    const uint32_t * a = activations + size_t(blockIdx.y)*36*pitch;
    if constexpr (Flat) {
        for (int work = tid; work < Rows*kblocks; work += Threads) {
            const int r = work/kblocks, kb = work%kblocks;
            float value = 0, gate = 0;
            if (row0+r < rows) {
                value = ptq1_pascal_dot<Raw,Cache>(weights + size_t(row0+r)*row_stride + kb, a, pitch, kb);
                if (fusion.gate) gate = ptq1_pascal_dot<Raw,Cache>(fusion.gate + size_t(row0+r)*row_stride + kb, a, pitch, kb);
            }
            partial[r*kblocks+kb] = value;
            if (fusion.gate) pg[r*kblocks+kb] = gate;
        }
    } else {
        for (int r = warp; r < Rows; r += Threads/32) {
            for (int kb = lane; kb < kblocks; kb += 32) {
                float value = 0, gate = 0;
                if (row0+r < rows) {
                    value = ptq1_pascal_dot<Raw,Cache>(weights + size_t(row0+r)*row_stride + kb, a, pitch, kb);
                    if (fusion.gate) gate = ptq1_pascal_dot<Raw,Cache>(fusion.gate + size_t(row0+r)*row_stride + kb, a, pitch, kb);
                }
                partial[r*kblocks+kb] = value;
                if (fusion.gate) pg[r*kblocks+kb] = gate;
            }
        }
    }
    __syncthreads();
    for (int r = warp; r < Rows; r += Threads/32) {
        float value = 0, gate = 0;
        for (int kb = lane; kb < kblocks; kb += 32) {
            value += partial[r*kblocks+kb];
            if (fusion.gate) gate += pg[r*kblocks+kb];
        }
        value = warp_reduce_sum(value);
        gate = warp_reduce_sum(gate);
        if (lane == 0 && row0+r < rows) {
            const int row = row0+r;
            if (fusion.bias) value += __ldg(fusion.bias+row);
            if (fusion.gate) {
                if (fusion.gate_bias) gate += __ldg(fusion.gate_bias+row);
                switch (fusion.op) {
                    case GGML_GLU_OP_SWIGLU: value *= ggml_cuda_op_silu_single(gate); break;
                    case GGML_GLU_OP_GEGLU: value *= ggml_cuda_op_gelu_single(gate); break;
                    case GGML_GLU_OP_SWIGLU_OAI: value = ggml_cuda_op_swiglu_oai_single(gate,value); break;
                    case GGML_GLU_OP_SWIGLU_CLAMP: value = ggml_cuda_op_swiglu_clamp_single(gate,value,fusion.limit); break;
                    default: value *= gate; break;
                }
            }
            dst[size_t(blockIdx.y)*dst_stride+row] = value;
        }
    }
}

// Reuse the expensive base-3 weight decode across a small verification batch.
// Each column retains the single-column integer accumulation and reduction order.
template<int Columns>
static __device__ __forceinline__ void ptq1_pascal_dot_multi(
        const block_ptq1_0 * w, const uint32_t * a, int pitch, int kb, float (&result)[Columns]) {
    int acc[Columns][4] = {};
    const uint32_t * wp = (const uint32_t *) w;
    uint32_t packed = ptq1_pascal_weight(wp);
#pragma unroll
    for (int g = 0; g < 6; ++g) {
        const uint32_t next = ptq1_pascal_weight(wp + g + 1);
        uint32_t lo = __byte_perm(packed, 0, 0x4140);
        uint32_t hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t wl = lo * 3, wh = hi * 3;
            lo = wl & 0x00ff00ff; hi = wh & 0x00ff00ff;
            const int trits = __byte_perm(wl, wh, 0x7531);
            const int e = g < 4 ? t*16 + 4*g : 80 + t*8 + 4*(g-4);
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                const int q = __ldg(a + (size_t(c)*36 + e/4)*pitch + kb);
                acc[c][e/32] = ggml_cuda_dp4a(trits, q, acc[c][e/32]);
            }
        }
        packed = next;
    }
    uint32_t v = (packed & 255u) | ((packed & 0xff00u) << 8);
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v*3; v = w0 & 0x00ff00ff;
        const uint32_t w1 = v*3; v = w1 & 0x00ff00ff;
        const int trits = __byte_perm(w0, w1, 0x7531);
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const int q = __ldg(a + (size_t(c)*36 + 30+t/2)*pitch + kb);
            acc[c][3] = ggml_cuda_dp4a(trits, q, acc[c][3]);
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        result[c] = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t meta = __ldg(a + (size_t(c)*36 + 32+j)*pitch + kb);
            acc[c][j] -= int(int16_t(meta >> 16));
            result[c] += __half2float(__ushort_as_half(meta & 65535)) * float(acc[c][j]);
        }
        result[c] *= __half2float(__ushort_as_half(packed >> 16));
    }
}

template<int Columns>
__launch_bounds__(96, 2)
static __global__ void mul_mat_vec_ptq1_pascal_multi(const block_ptq1_0 * weights,
        const uint32_t * activations, float * dst, int kblocks, int rows,
        int row_stride, int dst_stride, int pitch, const float * bias) {
    extern __shared__ float partial[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int row0 = blockIdx.x*4;
    for (int work = tid; work < 4*kblocks; work += 96) {
        const int r = work/kblocks, kb = work%kblocks;
        float value[Columns] = {};
        if (row0+r < rows) ptq1_pascal_dot_multi<Columns>(
            weights + size_t(row0+r)*row_stride + kb, activations, pitch, kb, value);
#pragma unroll
        for (int c = 0; c < Columns; ++c) partial[(c*4+r)*kblocks+kb] = value[c];
    }
    __syncthreads();
    for (int cr = warp; cr < 4*Columns; cr += 3) {
        const int c = cr/4, r = cr%4;
        float value = 0;
        for (int kb = lane; kb < kblocks; kb += 32) value += partial[cr*kblocks+kb];
        value = warp_reduce_sum(value);
        if (lane == 0 && row0+r < rows) {
            if (bias) value += __ldg(bias+row0+r);
            dst[size_t(c)*dst_stride+row0+r] = value;
        }
    }
}

struct ptq1_pascal_config { int threads; int rows; bool flat; };

// Explicit variants are retained for reproducible device-local tuning.
static inline ptq1_pascal_config ptq1_pascal_select(int k) {
    static const char * variant = getenv("GGML_CUDA_PTQ1_PASCAL_GEMV");
    ptq1_pascal_config c = k <= 6144 ? ptq1_pascal_config{96,4,true} :
        ptq1_pascal_config{128,4,false};
    if (!variant || !strcmp(variant,"auto") || !strcmp(variant,"on")) return c;
    for (int threads : {96,128,160,192}) for (int rows : {1,2,4}) for (bool flat : {true,false}) {
        char name[32]; snprintf(name,sizeof(name),"%d-%d-%s",threads,rows,flat?"flat":"warp");
        if (!strcmp(name,variant)) return {threads,rows,flat};
    }
    GGML_ABORT("Invalid GGML_CUDA_PTQ1_PASCAL_GEMV: %s",variant);
}

template<int Threads,int Rows,bool Raw=true,bool Cache=true>
static void ptq1_pascal_launch_ownership(bool flat, const block_ptq1_0 * w, const uint32_t * a,
        float * dst,int k,int rows,int columns,int row_stride,int dst_stride,
        ptq1_pascal_fusion fusion,cudaStream_t stream) {
    const size_t shared = size_t(Rows)*(k/128)*sizeof(float)*(fusion.gate?2:1);
    const ggml_cuda_kernel_launch_params lp(dim3((rows+Rows-1)/Rows,columns,1),dim3(Threads,1,1),shared,stream);
    if (flat) ggml_cuda_kernel_launch(mul_mat_vec_ptq1_pascal<Threads,Rows,true,Raw,Cache>,lp,
        w,a,dst,k/128,rows,row_stride,dst_stride,ptq1_pascal_pitch(k),fusion);
    else ggml_cuda_kernel_launch(mul_mat_vec_ptq1_pascal<Threads,Rows,false,Raw,Cache>,lp,
        w,a,dst,k/128,rows,row_stride,dst_stride,ptq1_pascal_pitch(k),fusion);
}

static void ptq1_pascal_launch(ptq1_pascal_config c,const block_ptq1_0 * w,const uint32_t * a,
        float * dst,int k,int rows,int columns,int row_stride,int dst_stride,
        ptq1_pascal_fusion fusion,cudaStream_t stream) {
    static const bool multi_enabled = [] {
        const char * s = getenv("GGML_CUDA_PTQ1_PASCAL_MULTI");
        return s && !strcmp(s,"on");
    }();
    if (multi_enabled && columns >= 2 && columns <= 4 && k <= 6144 && !fusion.gate) {
        const size_t shared = size_t(4)*columns*(k/128)*sizeof(float);
        const ggml_cuda_kernel_launch_params lp(dim3((rows+3)/4,1,1),dim3(96,1,1),shared,stream);
#define PTQ1_MULTI(N) if (columns == N) { \
        ggml_cuda_kernel_launch(mul_mat_vec_ptq1_pascal_multi<N>,lp, \
            w,a,dst,k/128,rows,row_stride,dst_stride,ptq1_pascal_pitch(k),fusion.bias); return; }
        PTQ1_MULTI(2) PTQ1_MULTI(3) PTQ1_MULTI(4)
#undef PTQ1_MULTI
    }
#define PTQ1_CASE(T,R) if(c.threads==T && c.rows==R) { \
    ptq1_pascal_launch_ownership<T,R>(c.flat,w,a,dst,k,rows,columns,row_stride,dst_stride,fusion,stream); return; }
    PTQ1_CASE(96,1) PTQ1_CASE(96,2) PTQ1_CASE(96,4)
    PTQ1_CASE(128,1) PTQ1_CASE(128,2) PTQ1_CASE(128,4)
    PTQ1_CASE(160,1) PTQ1_CASE(160,2) PTQ1_CASE(160,4)
    PTQ1_CASE(192,1) PTQ1_CASE(192,2) PTQ1_CASE(192,4)
#undef PTQ1_CASE
    GGML_ABORT("Unsupported Pascal PTQ1 GEMV geometry");
}
