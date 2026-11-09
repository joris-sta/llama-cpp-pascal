#pragma once

#include "gdn-norm.cuh"

// Scalar-gate DeltaNet, C-token WY/UT prefill, independently derived from
// S_i = exp(g_i) S_{i-1} + k_i beta_i (v_i - exp(g_i) k_i^T S_{i-1}).
// L_ij = beta_i exp(sum_{j<u<=i} g_u) k_i^T k_j, j<i;
// T=(I+L)^-1; W=T diag(beta*G) K; U=T diag(beta) V.
// Delta=T(beta V-beta G K S_in); O=G Q S_in+A Delta;
// S_out=G_last S_in+K^T D_last Delta (implicit WY, materialized UT).
// No inverse cumulative gate: negative/reset gates can underflow safely.
// Only the small triangular transform is sequential; the D x D state advances
// once per chunk. FP32 throughout; no tensor cores or model-name dispatch.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template<int C> struct gdn_chunk_layout {
    static constexpr int D = 128;
    static constexpr int matrix = C*D;
    static constexpr int floats = 4*matrix + 3*C*C + C;
    float * q; float * k; float * w; float * u;
    float * a; float * decay; float * prefix; float * triangular;
    __host__ __device__ explicit gdn_chunk_layout(float * p) :
        q(p), k(p+matrix), w(p+2*matrix), u(p+3*matrix),
        a(p+4*matrix), decay(a+C*C), prefix(decay+C*C), triangular(prefix+C) {}
};

__device__ __forceinline__ float gdn_chunk_exp(float x) {
    return exp2f(x*1.442695041f);
}

template<int C, bool SumEps>
static __global__ void __launch_bounds__(128, 2) gdn_chunk_prepare(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        float * work, int H, int tokens, int chunks,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, uint3 head_magic, uint3 seq_magic,
        ggml_cuda_gdn_norm norm) {
    constexpr int D=128;
    const int head=blockIdx.x, seq=blockIdx.y, chunk=blockIdx.z;
    const int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    const int start=chunk*C, count=min(C,tokens-start);
    const int qh=fastmodulo(head,head_magic), qs=fastdiv(seq,seq_magic);
    gdn_chunk_layout<C> out(work+((size_t(seq)*H+head)*chunks+chunk)*gdn_chunk_layout<C>::floats);
    // 29,696 B C32; pad key rows to avoid shared-bank conflicts in UT GEMMs.
    __shared__ float keys[C][D+1], inverse[C][C+1], attention[C][C+1];
    __shared__ float gates[C], betas[C], prefix[C], row[C], decay_matrix[C][C+1];
    ggml_cuda_pdl_sync();
    for (int t=tid; t<C; t+=128) {
        const int64_t gb=seq*sb3+(start+t)*sb2+head*sb1;
        gates[t]=t<count ? gdn_chunk_exp(g[gb]) : 1.0f;
        betas[t]=t<count ? beta[gb] : 0.0f;
    }
    __syncthreads();
    for (int t=warp;t<C;t+=4) {
        float qr[4], kr[4], qsum=0, ksum=0;
#pragma unroll
        for(int r=0;r<4;++r) {
            const int i=lane+32*r;
            qr[r]=t<count ? q[qs*sq3+(start+t)*sq2+qh*sq1+i] : 0.0f;
            kr[r]=t<count ? k[qs*sq3+(start+t)*sq2+qh*sq1+i] : 0.0f;
            qsum=fmaf(qr[r],qr[r],qsum); ksum=fmaf(kr[r],kr[r],ksum);
        }
        if(SumEps || norm.eps>=0) {
            qsum=warp_reduce_sum(qsum); ksum=warp_reduce_sum(ksum);
            const float qi=SumEps ? rsqrtf(qsum+norm.eps) : norm.inverse(qsum,D);
            const float ki=SumEps ? rsqrtf(ksum+norm.eps) : norm.inverse(ksum,D);
#pragma unroll
            for(int r=0;r<4;++r) {
                qr[r]=SumEps ? qr[r]*qi : norm.apply(qr[r],qi);
                kr[r]=SumEps ? kr[r]*ki : norm.apply(kr[r],ki);
                // A padded zero token must remain zero even with eps=0.
                if(t>=count) qr[r]=kr[r]=0;
            }
        }
#pragma unroll
        for(int r=0;r<4;++r) {
            const int i=lane+32*r;
            out.q[t*D+i]=qr[r]; keys[t][i]=kr[r]; out.k[t*D+i]=kr[r];
        }
        if(lane==0) {
            float product=1;
            for(int u=0;u<=t;++u) product*=gates[u];
            out.prefix[t]=prefix[t]=product;
        }
    }
    __syncthreads();
    if(tid<C) {
        float product=1;
        for(int j=tid;j>=0;--j) {
            out.decay[tid*C+j]=decay_matrix[tid][j]=product;
            product*=gates[j];
        }
        for(int j=tid+1;j<C;++j) out.decay[tid*C+j]=decay_matrix[tid][j]=0;
    }
    __syncthreads();
    // Keep the query and current key in registers across a complete causal row.
    // Each warp's four feature fragments are reused instead of reloading them
    // for every pair. The gate products were computed once above.
    for(int i=warp;i<C;i+=4) {
        float qi[4],ki[4];
#pragma unroll
        for(int r=0;r<4;++r) { qi[r]=out.q[i*D+lane+32*r]; ki[r]=keys[i][lane+32*r]; }
        for(int j=0;j<C;++j) {
            float kk=0,qk=0;
            if(j<=i) {
#pragma unroll
                for(int r=0;r<4;++r) {
                    const float kj=keys[j][lane+32*r];
                    kk=fmaf(ki[r],kj,kk); qk=fmaf(qi[r],kj,qk);
                }
                kk=warp_reduce_sum(kk); qk=warp_reduce_sum(qk);
            }
            if(lane==0) {
                const float decay=decay_matrix[i][j];
                inverse[i][j]=j<i ? betas[i]*decay*kk : (j==i ? 1.0f : 0.0f);
                attention[i][j]=decay*qk;
            }
        }
    }
    __syncthreads();
    // Forward substitution for the unit-lower triangular inverse. Cache the
    // original L row before overwriting it; all inverse columns run in parallel.
    for(int i=1;i<C;++i) {
        if(tid<i) row[tid]=inverse[i][tid];
        __syncthreads();
        if(tid<i) {
            float acc=0;
            for(int j=tid;j<i;++j) acc=fmaf(row[j],inverse[j][tid],acc);
            inverse[i][tid]=-acc;
        }
        __syncthreads();
    }
    // Apply UT to the value-sized residual in the state kernel, rather than
    // materializing both W=T beta G K and U=T beta V here. This saves one
    // triangular CxCxD product and keeps the same chunkwise algebra.
    for(int e=tid;e<C*D;e+=128) {
        const int i=e/D, d=e%D;
        out.w[e]=keys[i][d]*(betas[i]*prefix[i]);
        out.u[e]=i<count ? __ldg(v+seq*sv3+(start+i)*sv2+head*sv1+d)*betas[i] : 0;
    }
    for(int e=tid;e<C*C;e+=128) {
        out.a[e]=attention[e/C][e%C]; out.triangular[e]=inverse[e/C][e%C];
    }
    ggml_cuda_pdl_lc();
}

// SIMT matrix products: adjacent threads own adjacent value columns. A
// key-major shared state and padded operand tiles provide broadcast/coalesced
// shared reads. Each lane accumulates several columns, without register copies
// of the entire state. Token rows are computed in parallel within the chunk.
template<int C, int Columns=16, bool Keep=false, int TileK=(C==16 ? 64 : 32)>
static __global__ void __launch_bounds__(128, 4) gdn_chunk_apply(
        const float * work, const float * input, float * dst, float * states,
        int H, int tokens, int chunks, int64_t slot_stride, int snapshots, float scale) {
    constexpr int D=128, Parts=128/C, Acc=Columns/Parts;
    static_assert(D%TileK==0 && Columns==16,"Pascal SIMT matrix tile");
    const int tid=threadIdx.x, head=blockIdx.x, seq=blockIdx.y, col0=blockIdx.z*Columns;
    const int t=tid/Parts, part=tid%Parts;
    const size_t state_base=(size_t(seq)*H+head)*D*D;
    __shared__ float state[D][Columns+1], mk[C][TileK+1], mq[C][TileK+1];
    __shared__ float residual[C][Columns+1], delta[C][Columns+1], coeff[C][C+1], prefix[C];
    ggml_cuda_pdl_sync();
    for(int e=tid;e<Columns*D;e+=128) state[e%D][e/D]=input[state_base+(col0+e/D)*D+e%D];
    __syncthreads();
    for(int chunk=0;chunk<chunks;++chunk) {
        const int start=chunk*C, count=min(C,tokens-start);
        gdn_chunk_layout<C> p(const_cast<float *>(work)+((size_t(seq)*H+head)*chunks+chunk)*gdn_chunk_layout<C>::floats);
        float d[Acc]={},o[Acc]={};
        for(int k0=0;k0<D;k0+=TileK) {
            for(int e=tid;e<C*TileK;e+=128) {
                mk[e/TileK][e%TileK]=__ldg(p.w+(e/TileK)*D+k0+e%TileK);
                mq[e/TileK][e%TileK]=__ldg(p.q+(e/TileK)*D+k0+e%TileK);
            }
            __syncthreads();
#pragma unroll 4
            for(int k=0;k<TileK;++k) {
                const float w=mk[t][k],q=mq[t][k];
#pragma unroll
                for(int a=0;a<Acc;++a) {
                    const float s=state[k0+k][part+a*Parts];
                    d[a]=fmaf(w,s,d[a]); o[a]=fmaf(q,s,o[a]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for(int a=0;a<Acc;++a) residual[t][part+a*Parts]=__ldg(p.u+t*D+col0+part+a*Parts)-d[a];
        for(int e=tid;e<C*C;e+=128) coeff[e/C][e%C]=__ldg(p.triangular+e);
        if(tid<C) prefix[tid]=__ldg(p.prefix+tid);
        __syncthreads();
        float solved[Acc]={};
        for(int j=0;j<=t;++j) {
            const float a=coeff[t][j];
#pragma unroll
            for(int c=0;c<Acc;++c) solved[c]=fmaf(a,residual[j][part+c*Parts],solved[c]);
        }
#pragma unroll
        for(int c=0;c<Acc;++c) delta[t][part+c*Parts]=solved[c];
        __syncthreads();
        for(int e=tid;e<C*C;e+=128) coeff[e/C][e%C]=__ldg(p.a+e);
        __syncthreads();
#pragma unroll
        for(int c=0;c<Acc;++c) o[c]*=prefix[t];
        for(int j=0;j<=t;++j) {
            const float a=coeff[t][j];
#pragma unroll
            for(int c=0;c<Acc;++c) o[c]=fmaf(a,delta[j][part+c*Parts],o[c]);
        }
        if(t<count) {
#pragma unroll
            for(int c=0;c<Acc;++c) dst[((size_t(seq)*tokens+start+t)*H+head)*D+col0+part+c*Parts]=o[c]*scale;
        }
        __syncthreads();
        for(int e=tid;e<C*C;e+=128) coeff[e/C][e%C]=__ldg(p.decay+e);
        __syncthreads();
        // Tiled K^T Delta updates each state element once for this chunk.
        // Snapshot reconstruction uses the incoming state before overwriting it.
        constexpr int Rows=TileK*Columns/128;
        const int key=tid/Columns, col=tid%Columns;
        for(int k0=0;k0<D;k0+=TileK) {
            for(int e=tid;e<C*TileK;e+=128) mk[e/TileK][e%TileK]=__ldg(p.k+(e/TileK)*D+k0+e%TileK);
            __syncthreads();
            float initial[Rows],s[Rows];
#pragma unroll
            for(int r=0;r<Rows;++r) {
                initial[r]=state[k0+key+r*(128/Columns)][col];
                s[r]=prefix[count-1]*initial[r];
            }
            if constexpr(Keep) {
                const int first=max(0,tokens-snapshots-start);
                for(int i=first;i<count;++i) {
                    float snapshot[Rows];
#pragma unroll
                    for(int r=0;r<Rows;++r) snapshot[r]=prefix[i]*initial[r];
                    for(int j=0;j<=i;++j) {
                        const float d=coeff[i][j]*delta[j][col];
#pragma unroll
                        for(int r=0;r<Rows;++r) snapshot[r]=fmaf(mk[j][key+r*(128/Columns)],d,snapshot[r]);
                    }
#pragma unroll
                    for(int r=0;r<Rows;++r) states[size_t(tokens-1-start-i)*slot_stride+state_base+(col0+col)*D+k0+key+r*(128/Columns)]=snapshot[r];
                }
            }
            for(int j=0;j<count;++j) {
                const float d=coeff[count-1][j]*delta[j][col];
#pragma unroll
                for(int r=0;r<Rows;++r) s[r]=fmaf(mk[j][key+r*(128/Columns)],d,s[r]);
            }
#pragma unroll
            for(int r=0;r<Rows;++r) state[k0+key+r*(128/Columns)][col]=s[r];
            __syncthreads();
        }
    }
    if constexpr(!Keep) for(int e=tid;e<Columns*D;e+=128)
        states[state_base+(col0+e/D)*D+e%D]=state[e%D][e/D];
    ggml_cuda_pdl_lc();
}
#endif
