#include "common.cuh"
#include <atomic>
#include "fattn-tile.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn-turbo3-decode.cuh"

// Overrides apply to every supported head geometry; no model identifiers.
static int turbo3_decode_tile() {
    static const int selected = [] {
        const char * value = getenv("GGML_CUDA_T3_DECODE_TILE");
        if (!value || !*value || strcmp(value,"auto") == 0 || strcmp(value,"default") == 0) return 0;
        if (strcmp(value,"scalar") == 0) return 1;
        if (strcmp(value,"warp1") == 0 || strcmp(value,"reg1") == 0) return 2;
        if (strcmp(value,"warp4") == 0 || strcmp(value,"reg4") == 0) return 3;
        GGML_ABORT("GGML_CUDA_T3_DECODE_TILE must be auto, scalar, warp1 or warp4");
    }();
    return selected;
}

template<int D, int G, int Warps>
static void launch_turbo3_register(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    float softcap = 0; memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    fattn_kernel_t kernel = softcap == 0 ? flash_attn_turbo3_register<D,G,Warps,false> : flash_attn_turbo3_register<D,G,Warps,true>;
    launch_fattn<D,1,G>(ctx,dst,kernel,Warps,0,32,false,false,false,false,32);
#else
    GGML_UNUSED_VARS(ctx,dst);
    GGML_ABORT("T3 register decode requires NVIDIA CUDA");
#endif
}

template<int D, int G>
static bool try_turbo3_register(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    float bias = 0; memcpy(&bias,(const float *)dst->op_params+1,sizeof(float));
    // Mixed/broadcast sequences and ALiBi keep the existing general native tile.
    if (ggml_cuda_info().devices[ctx.device].cc != 610 || !mask || bias != 0 ||
        q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] || q->ne[3] != mask->ne[3]) return false;
    const int selected = turbo3_decode_tile();
    if (selected == 1) return false;
    // Bound live query/output vectors instead of assuming a particular model.
    // Preserve the measured GQA4 crossover; higher register pressure and wide
    // heads prefer a multi-warp CTA. Tiny scans avoid a large fixup grid.
    const bool four = selected == 3 || (selected == 0 &&
        (D == 512 || G*D > 1024 || k->ne[1] <= 512 || (D >= 256 && k->ne[1] > 65536)));
    if (four) launch_turbo3_register<D,G,4>(ctx,dst);
    else launch_turbo3_register<D,G,1>(ctx,dst);
    return true;
}

template<int D, int G>
static void launch_turbo3_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if constexpr (D*G <= 1536) {
        if (try_turbo3_register<D,G>(ctx,dst)) return;
    }
    const int nwarps = ggml_cuda_fattn_tile_get_nthreads(D,D,G,cc)/32;
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(D,D,G,cc);
    float softcap = 0; memcpy(&softcap,(const float *)dst->op_params+2,sizeof(float));
    fattn_kernel_t kernel = softcap == 0 ? flash_attn_tile<D,D,1,G,false,true>
                                       : flash_attn_tile<D,D,1,G,true,true>;
    launch_fattn<D,1,G>(ctx,dst,kernel,nwarps,0,nbatch_fa,false,false,false,false,32);
}

template<int D>
static void dispatch_turbo3_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int64_t ratio = Q->ne[2] / K->ne[2];
    float bias = 0.0f;
    memcpy(&bias, (const float *) dst->op_params + 1, sizeof(float));
    // Multi-head tiles require a mask, no ALiBi and padded KV rows. Otherwise
    // use the native single-head tile, which handles bounds and per-head slopes.
    if (dst->src[3] && bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0) {
        if constexpr (D <= 256) {
            if (ratio%3 == 0 && ratio%6 != 0 && try_turbo3_register<D,3>(ctx,dst)) return;
        }
        if constexpr (D == 256) {
            if (ratio%6 == 0) return launch_turbo3_decode<D, 6>(ctx, dst);
        }
        if (ratio % 8 == 0 && (D <= 128 || turbo3_decode_tile() == 1 || ggml_cuda_info().devices[ctx.device].cc != 610)) return launch_turbo3_decode<D, 8>(ctx, dst);
        if (ratio % 4 == 0 && (D <= 256 || turbo3_decode_tile() == 1 || ggml_cuda_info().devices[ctx.device].cc != 610)) return launch_turbo3_decode<D, 4>(ctx, dst);
        if (ratio % 2 == 0) return launch_turbo3_decode<D, 2>(ctx, dst);
    }
    launch_turbo3_decode<D, 1>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_tile_turbo3_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_TURBO3_0 && dst->src[2]->type == GGML_TYPE_TURBO3_0);
    turbo_vanilla_cb_load_fattn();
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true, std::memory_order_relaxed)) {
        fprintf(stderr, "T3/T3 decode: hard-native fused CUDA tile, D=%lld GQA=%lld variant=%d (no full-KV FP16 materialization)\n",
                (long long) dst->src[0]->ne[0], (long long) (dst->src[0]->ne[2] / dst->src[1]->ne[2]), turbo3_decode_tile());
    }
    switch (dst->src[0]->ne[0]) {
        case 128: return dispatch_turbo3_gqa<128>(ctx, dst);
        case 256: return dispatch_turbo3_gqa<256>(ctx, dst);
        case 512: return dispatch_turbo3_gqa<512>(ctx, dst);
        default: GGML_ABORT("hard-native T3/T3 decode: unsupported head dimension");
    }
}

template<int D, int G, int Cols, bool Softcap>
static void launch_turbo3_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int N = Cols/G;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int nwarps = ggml_cuda_fattn_tile_get_nthreads(D,D,Cols,cc)/32;
    const int batch = ggml_cuda_fattn_tile_get_nbatch_fa(D,D,Cols,cc);
    launch_fattn<D,N,G>(ctx,dst,flash_attn_tile<D,D,N,G,Softcap,true>,nwarps,0,batch,
                       false,false,false,false,32);
}

template<int D, int G>
static void dispatch_turbo3_prefill_cols(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int cols, bool softcap) {
    if (cols == -1) {
        const int columns = int(dst->src[0]->ne[1])*G;
        if constexpr (G <= 2) {
            if (columns <= 2) {
                if (softcap) return launch_turbo3_prefill<D,G,2,true>(ctx,dst);
                return launch_turbo3_prefill<D,G,2,false>(ctx,dst);
            }
        }
        if constexpr (G <= 4) {
            if (columns <= 4) {
                if (softcap) return launch_turbo3_prefill<D,G,4,true>(ctx,dst);
                return launch_turbo3_prefill<D,G,4,false>(ctx,dst);
            }
        }
        if (columns <= 8) {
            if (softcap) return launch_turbo3_prefill<D,G,8,true>(ctx,dst);
            return launch_turbo3_prefill<D,G,8,false>(ctx,dst);
        }
        cols = columns <= 16 ? 16 : 32;
    }
    if (cols == 16) {
        if (softcap) return launch_turbo3_prefill<D,G,16,true>(ctx,dst);
        return launch_turbo3_prefill<D,G,16,false>(ctx,dst);
    }
    if (softcap) return launch_turbo3_prefill<D,G,32,true>(ctx,dst);
    return launch_turbo3_prefill<D,G,32,false>(ctx,dst);
}

template<int D>
static void dispatch_turbo3_prefill_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int cols) {
    float bias = 0, softcap = 0;
    memcpy(&bias,(const float *)dst->op_params+1,sizeof(float));
    memcpy(&softcap,(const float *)dst->op_params+2,sizeof(float));
    const int ratio = dst->src[0]->ne[2]/dst->src[1]->ne[2];
    if (dst->src[3] && bias == 0 && dst->src[1]->ne[1]%FATTN_KQ_STRIDE == 0) {
        if constexpr (D == 256) {
            if (ratio%6 == 0 && dst->src[0]->ne[1] <= 4 && try_turbo3_register<D,6>(ctx,dst)) return;
        }
        if (ratio%8 == 0) return dispatch_turbo3_prefill_cols<D,8>(ctx,dst,cols,softcap != 0);
        if (ratio%4 == 0) return dispatch_turbo3_prefill_cols<D,4>(ctx,dst,cols,softcap != 0);
        if (ratio%2 == 0) return dispatch_turbo3_prefill_cols<D,2>(ctx,dst,cols,softcap != 0);
    }
    dispatch_turbo3_prefill_cols<D,1>(ctx,dst,cols,softcap != 0);
}

void ggml_cuda_flash_attn_ext_tile_turbo3_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int cols) {
    turbo_vanilla_cb_load_fattn();
    switch (dst->src[0]->ne[0]) {
        case 128: return dispatch_turbo3_prefill_gqa<128>(ctx,dst,cols);
        case 256: return dispatch_turbo3_prefill_gqa<256>(ctx,dst,cols);
        default: GGML_ABORT("native T3 prefill: unsupported dimension");
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
