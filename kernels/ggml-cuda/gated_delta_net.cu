#include "gated_delta_net.cuh"
#include "gdn-norm.cuh"
#include "gdn-chunk-pascal.cuh"
#include "gated_delta_net_fla_ptx.cuh"
#include "ggml-cuda/common.cuh"

#include <type_traits>

// RDNA3 wants 1 block/SM here: with 2 blocks/SM the compiler is forced under
// ~128 VGPR/lane and spills on this kernel. 1 block/SM raises the per-lane
// VGPR budget to ~256 and eliminates the spill entirely. CUDA keeps 2.
// See llama.cpp issue #20354 (GATED_DELTA_NET HIP underperforms on RDNA3).
// exp2 is a single SFU instruction (ex2.approx) vs expf's multiply + ex2 + range reduction
#define GDN_EXPF(x) exp2f((x) * 1.442695041f)

#if defined(GGML_USE_HIP)
#define GGML_GDN_MIN_BLOCKS_PER_SM 1
#else
#define GGML_GDN_MIN_BLOCKS_PER_SM 2
#endif

template <int S_v, bool KDA, bool keep_rs_t, bool sum_eps_t, bool indexed_state = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     ggml_cuda_gdn_norm l2_norm, const int32_t * input_rows) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t input_sequence       = indexed_state ? input_rows[sequence] : sequence;
    const int64_t state_in_offset      = input_sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
#if !defined(GGML_USE_HIP)
        // HIP normalizes in the preceding paired kernel and never defers it.
        // Keep this unused branch out of its recurrent token loop.
        // Deferred q/k normalization retains the graph's epsilon convention.
        if (sum_eps_t || l2_norm.eps >= 0.0f) {
            float sum_q = 0.0f;
            float sum_k = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                sum_q += q_reg[r] * q_reg[r];
                sum_k += k_reg[r] * k_reg[r];
            }
            sum_q = warp_reduce_sum<warp_size>(sum_q);
            sum_k = warp_reduce_sum<warp_size>(sum_k);
            const float scale_q = sum_eps_t ? rsqrtf(sum_q + l2_norm.eps) : l2_norm.inverse(sum_q, S_v);
            const float scale_k = sum_eps_t ? rsqrtf(sum_k + l2_norm.eps) : l2_norm.inverse(sum_k, S_v);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_reg[r] = sum_eps_t ? q_reg[r] * scale_q : l2_norm.apply(q_reg[r], scale_q);
                k_reg[r] = sum_eps_t ? k_reg[r] * scale_k : l2_norm.apply(k_reg[r], scale_k);
            }
        }

#else
        GGML_UNUSED(l2_norm);
#endif

        if constexpr (!KDA) {
            const float g_val = GDN_EXPF(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += GDN_EXPF(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = GDN_EXPF(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Chunk-staged recurrent FP32 path for pre-tensor-core GPUs. Each warp reuses
// normalized Q/K across C value columns. Token tiles amortize global input
// loads across the CTA while preserving the exact recurrent/snapshot order.
template<int D, int C, int Tile, bool Keep, bool SumEps>
__global__ __launch_bounds__(128,2) void gated_delta_net_pascal(
        const float * q, const float * k, const float * v, const float * g,
        const float * beta, const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        uint3 neqk1_magic, uint3 rq3_magic, float scale, int64_t state_slot_stride,
        int K, ggml_cuda_gdn_norm norm, const int32_t * input_rows) {
    GGML_UNUSED_VARS(n_seqs,input_rows);
    constexpr int R = D/32, Columns = 4*C;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp*32+lane;
    const int head = blockIdx.x, seq = blockIdx.y, col0 = blockIdx.z*Columns+warp*C;
    const int qhead = fastmodulo(head,neqk1_magic), qseq = fastdiv(seq,rq3_magic);
    const int64_t state_base = (int64_t(seq)*H+head)*D*D;
    float shard[C][R];
#pragma unroll
    for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) shard[c][r] = curr_state[state_base+(col0+c)*D+lane+32*r];
    }
    __shared__ float qs[Tile][D], ks[Tile][D], vs[Tile][Columns];
    __shared__ float gs[Tile], bs[Tile];
    for (int t0 = 0; t0 < n_tokens; t0 += Tile) {
        const int count = int(n_tokens-t0) < Tile ? int(n_tokens-t0) : Tile;
        for (int i = tid; i < count*D; i += 128) {
            const int t = i/D, j = i%D;
            qs[t][j] = q[qseq*sq3+(t0+t)*sq2+qhead*sq1+j];
            ks[t][j] = k[qseq*sq3+(t0+t)*sq2+qhead*sq1+j];
        }
        for (int i = tid; i < count*Columns; i += 128) {
            const int t = i/Columns, c = i%Columns;
            vs[t][c] = v[seq*sv3+(t0+t)*sv2+head*sv1+blockIdx.z*Columns+c];
        }
        if (tid < count) {
            const int64_t offset = seq*sb3+(t0+tid)*sb2+head*sb1;
            gs[tid] = GDN_EXPF(g[offset]); bs[tid] = beta[offset];
        }
        __syncthreads();
        for (int t = 0; t < count; ++t) {
            float qr[R], kr[R];
#pragma unroll
            for (int r = 0; r < R; ++r) { qr[r]=qs[t][lane+32*r]; kr[r]=ks[t][lane+32*r]; }
            if (SumEps || norm.eps >= 0.0f) {
                float qq = 0, kk = 0;
#pragma unroll
                for (int r = 0; r < R; ++r) { qq += qr[r]*qr[r]; kk += kr[r]*kr[r]; }
                qq = warp_reduce_sum<32>(qq); kk = warp_reduce_sum<32>(kk);
                const float qscale = SumEps ? rsqrtf(qq+norm.eps) : norm.inverse(qq,D);
                const float kscale = SumEps ? rsqrtf(kk+norm.eps) : norm.inverse(kk,D);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    qr[r] = SumEps ? qr[r]*qscale : norm.apply(qr[r],qscale);
                    kr[r] = SumEps ? kr[r]*kscale : norm.apply(kr[r],kscale);
                }
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                float product = 0;
#pragma unroll
                for (int r = 0; r < R; ++r) product += shard[c][r]*kr[r];
                product = warp_reduce_sum<32>(product);
                const float delta = (vs[t][warp*C+c]-gs[t]*product)*bs[t];
                float result = 0;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    shard[c][r] = gs[t]*shard[c][r]+kr[r]*delta;
                    result += shard[c][r]*qr[r];
                }
                result = warp_reduce_sum<32>(result);
                if (lane == 0) dst[(int64_t(seq)*n_tokens*H+(t0+t)*H+head)*D+col0+c] = result*scale;
                if constexpr (Keep) {
                    const int slot = int(n_tokens)-1-(t0+t);
                    if (slot < K) {
#pragma unroll
                        for (int r = 0; r < R; ++r)
                            state[slot*state_slot_stride+state_base+(col0+c)*D+lane+32*r] = shard[c][r];
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (!Keep) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) state[state_base+(col0+c)*D+lane+32*r] = shard[c][r];
        }
    }
}
#endif

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, ggml_cuda_gdn_norm l2_norm, cudaStream_t stream,
        const int32_t * input_rows = nullptr) {
    // Recurrent fallback for decode, unsupported shapes and explicit staged comparisons.
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    // Select canonical normalization once, outside the recurrent token loop.
    const auto launch = [&](auto sum_eps_tag) {
        constexpr bool sum_eps_t = decltype(sum_eps_tag)::value;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        static const int staged = [] {
            const char * value = getenv("GGML_CUDA_GDN_PASCAL");
            if (!value || strcmp(value,"auto") == 0) return 0;
            if (strcmp(value,"off") == 0) return -1;
            if (strcmp(value,"staged") == 0) return 1;
            GGML_ABORT("GGML_CUDA_GDN_PASCAL must be auto, off or staged");
        }();
        if constexpr (!KDA) {
            // Width 64 regressed in the measured sweep; width 32 gains were
            // marginal. Keep automatic selection on the robust width-128 win.
            const bool automatic = S_v == 128 && n_tokens >= 64;
            if (staged >= 0 && (staged == 1 || automatic) && !input_rows && n_tokens >= 16 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == 610) {
                const auto parameters = ggml_cuda_kernel_launch_params(dim3(H,n_seqs,S_v/16),dim3(32,4),0,stream);
#define GDN_STAGE(D) ggml_cuda_kernel_launch(gated_delta_net_pascal<D,4,16,keep_rs_t,sum_eps_t>,parameters, \
                    q_d,k_d,v_d,g_d,b_d,s_d,dst_d,state_d,H,n_tokens,n_seqs, \
                    sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,neqk1_magic,rq3_magic, \
                    scale,state_slot_stride,K,l2_norm,input_rows)
                switch (S_v) {
                    case 32: GDN_STAGE(32); return;
                    case 64: GDN_STAGE(64); return;
                    case 128: GDN_STAGE(128); return;
                    default: break;
                }
#undef GDN_STAGE
            }
        }
#endif
        switch (S_v) {
            case 16:
                ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, sum_eps_t>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows);
                break;
            case 32:
                ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, sum_eps_t>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows);
                break;
            case 64: {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, sum_eps_t>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows);
                break;
            }
            case 128: {
                if constexpr (!KDA && keep_rs_t) {
                    if (input_rows) {
                        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, true, sum_eps_t, true>, launch_params,
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows);
                        break;
                    }
                }
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, sum_eps_t>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_norm, input_rows);
                break;
            }
            default:
                GGML_ABORT("fatal error");
                break;
        }
    };
#if defined(GGML_USE_HIP)
    launch(std::false_type{});
#else
    if (l2_norm.sum_eps) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
#endif
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

#if !defined(GGML_USE_HIP)
    const bool defer_q_l2 = ctx.gdn_deferred_l2.erase(src_q->data) != 0;
    const bool defer_k_l2 = ctx.gdn_deferred_l2.erase(src_k->data) != 0;
#else
    const bool defer_q_l2 = false;
    const bool defer_k_l2 = false;
#endif
    GGML_ASSERT(defer_q_l2 == defer_k_l2);

    ggml_cuda_gdn_norm l2_norm, k_norm;
    const ggml_tensor * packed_q = defer_q_l2 ? ggml_cuda_gdn_norm_input(src_q, l2_norm) : src_q;
    const ggml_tensor * packed_k = defer_k_l2 ? ggml_cuda_gdn_norm_input(src_k, k_norm) : src_k;
    GGML_ASSERT(packed_q && packed_k);
    GGML_ASSERT(l2_norm.eps == k_norm.eps && l2_norm.rms == k_norm.rms && l2_norm.sum_eps == k_norm.sum_eps &&
                l2_norm.post_scale == k_norm.post_scale);

    const float * q_d = (const float *) packed_q->data;
    const float * k_d = (const float *) packed_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = cache && cache->input_state ? cache->input_state : (const float *) src_state->data;
    const int32_t * input_rows = cache ? cache->input_rows : nullptr;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = packed_q->nb[1] / sizeof(float);
    int64_t sq2 = packed_q->nb[2] / sizeof(float);
    int64_t sq3 = packed_q->nb[3] / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr && cache->data != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static const int chunk_policy = [] {
        const char * value = getenv("GGML_CUDA_GDN_PASCAL_CHUNK");
        if (!value || strcmp(value, "auto") == 0) return -1;
        if (strcmp(value, "off") == 0) return 0;
        if (strcmp(value, "16") == 0) return 16;
        if (strcmp(value, "32") == 0) return 32;
        GGML_ABORT("GGML_CUDA_GDN_PASCAL_CHUNK must be auto, off, 16 or 32");
    }();
    // Whole-graph measurements include canonical deferred normalization.
    // Timing repeated GDN nodes alone amortizes it away and selects the wrong
    // algorithm. C16 wins on the tested 32/48-value-head canonical graphs;
    // retain the staged recurrence for already-normalized and small-head inputs.
    const int chunk_size = chunk_policy < 0 ?
        (l2_norm.sum_eps && H >= 32 && n_tokens >= 64 && !keep_rs ? 16 : 0) : chunk_policy;
    if (chunk_size && ggml_cuda_info().devices[ctx.device].cc == 610 &&
        !kda && !input_rows && S_v == 128 && n_tokens >= 16) {
        const auto chunk_launch = [&](auto chunk_tag, auto norm_tag, auto keep_tag) {
            constexpr int C = decltype(chunk_tag)::value;
            constexpr bool SumEps = decltype(norm_tag)::value;
            constexpr bool Keep = decltype(keep_tag)::value;
            const int chunks = (n_tokens+C-1)/C;
            ggml_cuda_pool_alloc<float> workspace(ctx.pool(), size_t(H)*n_seqs*chunks*gdn_chunk_layout<C>::floats);
            ggml_cuda_kernel_launch(gdn_chunk_prepare<C,SumEps>,
                ggml_cuda_kernel_launch_params(dim3(H,n_seqs,chunks),dim3(128),0,stream),
                q_d,k_d,v_d,g_d,b_d,workspace.get(),int(H),int(n_tokens),chunks,
                sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,init_fastdiv_values(neqk1),init_fastdiv_values(rq3),l2_norm);
            ggml_cuda_kernel_launch(gdn_chunk_apply<C,16,Keep>,
                ggml_cuda_kernel_launch_params(dim3(H,n_seqs,8),dim3(128),0,stream),
                workspace.get(),s_d,dst_d,state_d,int(H),int(n_tokens),chunks,state_slot_stride,K,scale);
        };
        const auto chunk_norm = [&](auto c) {
            if (l2_norm.sum_eps) {
                if (keep_rs) chunk_launch(c,std::true_type{},std::true_type{});
                else chunk_launch(c,std::true_type{},std::false_type{});
            } else {
                if (keep_rs) chunk_launch(c,std::false_type{},std::true_type{});
                else chunk_launch(c,std::false_type{},std::false_type{});
            }
        };
        if (chunk_size == 16) chunk_norm(std::integral_constant<int,16>{});
        else chunk_norm(std::integral_constant<int,32>{});
        return;
    }
#endif
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!input_rows && ggml_cuda_gdn_fla_ptx_supported(cc, kda, keep_rs, S_v, H, neqk1, n_tokens, n_seqs)) {
        ggml_cuda_gdn_fla_ptx(ctx, cc, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                              n_tokens, H, neqk1, sq1, sq2, sq3, sv1, sv2, sv3,
                              nullptr,
                              l2_norm,
                              cache != nullptr ? cache->rms_weight : nullptr,
                              cache != nullptr ? cache->rms_gate : nullptr,
                              cache != nullptr ? cache->rms_gate_bf16 : false,
                              cache != nullptr ? cache->rms_output : nullptr,
                              cache != nullptr ? cache->rms_output_bf16 : false,
                              cache != nullptr ? cache->rms_output_int8 : false,
                              cache != nullptr ? cache->rms_output_scale : nullptr,
                              cache != nullptr ? cache->rms_eps : 0.0f);
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_norm, stream, input_rows);
        }
    }
}

void ggml_cuda_op_gated_delta_net_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ABORT("tree GDN kernel not yet ported to upstream state layout");
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
