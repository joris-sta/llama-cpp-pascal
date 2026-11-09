#include "gdn-chunk-pascal.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <functional>
#include <algorithm>

static float ms(const std::function<void()> & fn) {
    for(int i=0;i<3;++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a)); for(int i=0;i<10;++i) fn();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float elapsed; CUDA_CHECK(cudaEventElapsedTime(&elapsed,a,b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b)); return elapsed/10;
}

struct sample {
    int tokens, heads, qheads, seqs, qseqs, snapshots, pattern;
    bool norm, permuted;
};

static void check(sample c, bool perf) {
    constexpr int D=128;
    const int T=c.tokens,H=c.heads,N=c.seqs,Q=c.qheads,QN=c.qseqs;
    const int64_t sq1=c.permuted ? D*T : D, sq2=c.permuted ? D : D*Q, sq3=D*Q*T;
    const int64_t sv1=D,sv2=D*H,sv3=D*H*T,sb1=1,sb2=H,sb3=H*T;
    std::mt19937 gen(77+T+c.pattern);
    std::uniform_real_distribution<float> rnd(-1,1);
    std::vector<float> q(D*Q*T*QN),k(q.size()),v(D*H*T*N),g(H*T*N),b(g.size()),s(D*D*H*N);
    for(auto & x:q) x=rnd(gen); for(auto & x:k) x=rnd(gen);
    for(auto & x:v) x=rnd(gen); for(auto & x:s) x=rnd(gen)*.1f;
    for(size_t i=0;i<g.size();++i) {
        g[i]=c.pattern==1 ? -.0001f : -.01f-std::abs(rnd(gen))*2;
        b[i]=std::abs(rnd(gen));
        if(c.pattern==2 && i%(H*17)==0) g[i]=-INFINITY;
        if(c.pattern==3) b[i]=(i%3==0 ? 0 : 1);
    }
    ggml_cuda_gdn_norm norm;
    if(c.norm) { norm.sum_eps=true; norm.eps=1e-6f; }
    for(int seq=0;seq<QN;++seq) for(int h=0;h<Q;++h) for(int t=0;t<T;++t) {
        const size_t pos=seq*sq3+t*sq2+h*sq1;
        if(c.pattern==4) {
            for(int d=0;d<D;++d) { q[pos+d]=t%2==0 ? 0 : q[pos+d]*1e-5f; k[pos+d]=t%3==0 ? 0 : k[pos+d]*1e-5f; }
        }
        if(!c.norm) {
            double nq=0,nk=0;
            for(int d=0;d<D;++d) { nq+=double(q[pos+d])*q[pos+d]; nk+=double(k[pos+d])*k[pos+d]; }
            for(int d=0;d<D;++d) { q[pos+d]*=float(1/sqrt(std::max(nq,1e-12))); k[pos+d]*=float(1/sqrt(std::max(nk,1e-12))); }
        }
    }
    const size_t output=size_t(D)*H*T*N, state_size=s.size();
    // Unwritten older snapshot slots retain their sentinel, as in ggml's kernel.
    std::vector<double> expected(output+state_size*c.snapshots,123.0);
    for(int seq=0;seq<N;++seq) for(int h=0;h<H;++h) {
        const size_t base=(size_t(seq)*H+h)*D*D;
        std::vector<double> state(D*D);
        for(int e=0;e<D*D;++e) state[e]=s[base+e];
        for(int t=0;t<T;++t) {
            const size_t qi=(seq/(N/QN))*sq3+t*sq2+(h%Q)*sq1;
            double qr[D],kr[D],nq=0,nk=0;
            for(int d=0;d<D;++d) { qr[d]=q[qi+d]; kr[d]=k[qi+d]; nq+=qr[d]*qr[d]; nk+=kr[d]*kr[d]; }
            if(c.norm) for(int d=0;d<D;++d) { qr[d]/=sqrt(nq+norm.eps); kr[d]/=sqrt(nk+norm.eps); }
            const size_t gb=(size_t(seq)*T+t)*H+h;
            const double decay=exp(double(g[gb])), beta=b[gb];
            for(int col=0;col<D;++col) {
                double kv=0; for(int d=0;d<D;++d) kv+=kr[d]*state[col*D+d];
                const double delta=beta*(v[gb*D+col]-decay*kv);
                double out=0;
                for(int d=0;d<D;++d) { state[col*D+d]=decay*state[col*D+d]+kr[d]*delta; out+=qr[d]*state[col*D+d]; }
                expected[gb*D+col]=out/sqrt(double(D));
            }
            if((c.snapshots==1 && t==T-1) || (c.snapshots>1 && T-1-t<c.snapshots)) {
                const int slot=c.snapshots==1 ? 0 : T-1-t;
                for(int e=0;e<D*D;++e) expected[output+size_t(slot)*state_size+base+e]=state[e];
            }
        }
    }
    float * dq,*dk,*dv,*dg,*db,*ds,*dd,*work;
    auto copy=[](float ** p,const std::vector<float> & x) { CUDA_CHECK(cudaMalloc(p,x.size()*4)); CUDA_CHECK(cudaMemcpy(*p,x.data(),x.size()*4,cudaMemcpyHostToDevice)); };
    copy(&dq,q);copy(&dk,k);copy(&dv,v);copy(&dg,g);copy(&db,b);copy(&ds,s);
    CUDA_CHECK(cudaMalloc(&dd,expected.size()*4));
    CUDA_CHECK(cudaMalloc(&work,size_t(H)*N*((T+15)/16)*gdn_chunk_layout<32>::floats*4));
    const auto run=[&](auto chunk_tag,auto norm_tag,auto keep_tag,auto geometry_tag) {
        constexpr int C=decltype(chunk_tag)::value;
        constexpr bool Norm=decltype(norm_tag)::value, Keep=decltype(keep_tag)::value;
        constexpr int TileK=decltype(geometry_tag)::value;
        const int chunks=(T+C-1)/C;
        auto prep=[&] { gdn_chunk_prepare<C,Norm><<<dim3(H,N,chunks),128>>>(dq,dk,dv,dg,db,work,H,T,chunks,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,init_fastdiv_values(Q),init_fastdiv_values(N/QN),norm); };
        auto apply=[&] { gdn_chunk_apply<C,16,Keep,TileK><<<dim3(H,N,8),128>>>(work,ds,dd,dd+output,H,T,chunks,state_size,c.snapshots,1/sqrtf(D)); };
        std::vector<float> actual(expected.size(),123);
        CUDA_CHECK(cudaMemcpy(dd,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
        prep(); apply(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(),dd,actual.size()*4,cudaMemcpyDeviceToHost));
        double error[2]={},power[2]={},max_abs[2]={};
        for(size_t i=0;i<actual.size();++i) {
            const int section=i>=output; const double diff=actual[i]-expected[i];
            if(!std::isfinite(actual[i])) GGML_ABORT("non-finite chunk output at %zu",i);
            error[section]+=diff*diff; power[section]+=expected[i]*expected[i];
            max_abs[section]=std::max(max_abs[section],std::abs(diff));
        }
        for(int j=0;j<2;++j) if(error[j]/std::max(power[j],1e-30)>1e-10 || max_abs[j]>2e-5)
            GGML_ABORT("FP64 oracle mismatch C%d section%d nmse=%g abs=%g",C,j,error[j]/std::max(power[j],1e-30),max_abs[j]);
        std::printf("C=%d tile_k=%d tokens=%d H=%d Q=%d seqs=%d K=%d pattern=%d norm=%d permuted=%d output_nmse=%.3g state_nmse=%.3g max_abs=%.3g",
            C,TileK,T,H,Q,N,c.snapshots,c.pattern,c.norm,c.permuted,error[0]/std::max(power[0],1e-30),error[1]/std::max(power[1],1e-30),std::max(max_abs[0],max_abs[1]));
        if(perf) std::printf(" prepare_ms=%.4f apply_ms=%.4f total_ms=%.4f",ms(prep),ms(apply),ms([&]{prep();apply();}));
        std::puts("");
    };
    for(int C : {16,32}) {
        const auto dispatch=[&](auto tag,auto geometry) {
            if(c.norm) { if(c.snapshots>1) run(tag,std::true_type{},std::true_type{},geometry); else run(tag,std::true_type{},std::false_type{},geometry); }
            else { if(c.snapshots>1) run(tag,std::false_type{},std::true_type{},geometry); else run(tag,std::false_type{},std::false_type{},geometry); }
        };
        const auto geometries=[&](auto tag) {
            constexpr int Size=decltype(tag)::value;
            dispatch(tag,std::integral_constant<int,Size==16 ? 64 : 32>{});
            if(perf) {
                dispatch(tag,std::integral_constant<int,16>{});
                if constexpr(Size==16) { dispatch(tag,std::integral_constant<int,32>{}); dispatch(tag,std::integral_constant<int,128>{}); }
                else dispatch(tag,std::integral_constant<int,64>{});
            }
        };
        if(C==16) geometries(std::integral_constant<int,16>{}); else geometries(std::integral_constant<int,32>{});
    }
    CUDA_CHECK(cudaFree(dq));CUDA_CHECK(cudaFree(dk));CUDA_CHECK(cudaFree(dv));CUDA_CHECK(cudaFree(dg));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(ds));CUDA_CHECK(cudaFree(dd));CUDA_CHECK(cudaFree(work));
}

int main(int argc,char ** argv) {
    int count; CUDA_CHECK(cudaGetDeviceCount(&count)); if(count==0) return 77;
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=6 || prop.minor!=1) { std::puts("SKIP: requires sm_61"); return 77; }
    const bool perf=argc>1 && std::string(argv[1])=="--perf";
    if(perf) {
        for(int t : {64,256,512,1024,2048}) check({t,48,16,1,1,1,1,true,false},true);
    } else {
        for(int t : {1,15,16,17,31,32,33,65,257,513}) for(bool norm : {false,true})
            check({t,6,2,2,1,1,0,norm,t%2==1},false);
        for(int pattern : {1,2,3,4}) for(int k : {1,3,35,67})
            check({65,6,2,2,1,k,pattern,true,true},false);
        check({1024,6,2,1,1,3,1,true,false},false);
    }
    std::puts("PASS: chunked FP32 DeltaNet outputs, final state and suffix snapshots match FP64 recurrence");
}
