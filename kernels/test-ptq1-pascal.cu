#include "mmvq-ptq1-pascal.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include <vector>
#include <random>
#include <cstdio>
#include <functional>

static __global__ void reference(const block_ptq1_0 * w,const block_q8_1 * a,
        float * dst,int k,int m,int padded) {
    float v = 0;
    for (int kb = threadIdx.x; kb < k/128; kb += 32)
        v += vec_dot_ptq1_0_q8_1_cuda(w+size_t(blockIdx.x)*(k/128),
            a+size_t(blockIdx.y)*(padded/32),kb,4*kb);
    v = warp_reduce_sum(v);
    if (threadIdx.x == 0) dst[size_t(blockIdx.y)*m+blockIdx.x] = v;
}

static float timing(const std::function<void()> & f) {
    for(int i=0;i<5;++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,end; CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<40;++i) f();
    CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(end));return ms/40;
}

static void shape(int k,int m,int n,bool bench,bool extreme=false) {
    std::mt19937 rng(1234+k+m+n);
    std::vector<block_ptq1_0> hw(size_t(k/128)*m);
    for(auto & b:hw) {
        for(auto & q:b.qs) q=rng()%243;
        for(auto & q:b.qh) q=rng()%81;
        b.d=__float2half(float(1+rng()%16)/32);
    }
    std::vector<float> hx(k*n);
    for(size_t i=0;i<hx.size();++i) hx[i]=extreme ?
        (i/32%3==0 ? 0.0f : (i%32==0 ? 127.0f : (i%3==0 ? -126.0f : 3.25f))) :
        float(int(rng()%20001)-10000)/3333;
    block_ptq1_0 * w;float * x,* dst,* ref;uint32_t * a;block_q8_1 * aos;
    const int padded=GGML_PAD(k,MATRIX_ROW_PADDING);
    CUDA_CHECK(cudaMalloc(&w,hw.size()*sizeof(*w)));CUDA_CHECK(cudaMalloc(&x,hx.size()*4));
    CUDA_CHECK(cudaMalloc(&a,ptq1_pascal_bytes(k,n)));CUDA_CHECK(cudaMalloc(&aos,size_t(padded/32)*n*sizeof(*aos)));
    CUDA_CHECK(cudaMalloc(&dst,size_t(m)*n*4));CUDA_CHECK(cudaMalloc(&ref,size_t(m)*n*4));
    CUDA_CHECK(cudaMemcpy(w,hw.data(),hw.size()*sizeof(*w),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    quantize_ptq1_pascal_cuda(x,a,k,n,k,nullptr);
    quantize_row_q8_1_cuda(x,nullptr,aos,GGML_TYPE_PTQ1_0,k,k,k*n,k*n,padded,n,1,1,nullptr);
    reference<<<dim3(m,n),32>>>(w,aos,ref,k,m,padded);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expected(size_t(m)*n),out(expected.size());
    CUDA_CHECK(cudaMemcpy(expected.data(),ref,expected.size()*4,cudaMemcpyDeviceToHost));
    // Verify every activation byte, FP16 scale and exact integer sum against
    // the existing AoS quantizer. Never use rounded floating-point sums.
    std::vector<uint32_t> ha(ptq1_pascal_bytes(k,n)/4);
    std::vector<block_q8_1> hq(size_t(padded/32)*n);
    CUDA_CHECK(cudaMemcpy(ha.data(),a,ha.size()*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hq.data(),aos,hq.size()*sizeof(*aos),cudaMemcpyDeviceToHost));
    const int pitch=ptq1_pascal_pitch(k);
    for(int col=0;col<n;++col) for(int kb=0;kb<k/128;++kb) for(int j=0;j<4;++j) {
        const auto & b=hq[col*(padded/32)+kb*4+j];int sum=0;
        for(int e=0;e<32;++e) {
            sum+=b.qs[e];
            uint32_t word=ha[(size_t(col)*36+j*8+e/4)*pitch+kb];
            if(int8_t(word>>(8*(e%4)))!=b.qs[e]) GGML_ABORT("Q8 byte mismatch");
        }
        uint32_t meta=ha[(size_t(col)*36+32+j)*pitch+kb];
        uint16_t d;memcpy(&d,&b.ds,2);
        if((meta&65535)!=d || int16_t(meta>>16)!=sum) GGML_ABORT("Q8 scale/sum mismatch");
    }
    // Independent scalar CPU arithmetic on selected output rows.
    const auto * traits=ggml_get_type_traits(GGML_TYPE_PTQ1_0);
    std::vector<float> deq(k);
    for(int row : {0,m/2,m-1}) {
        traits->to_float(hw.data()+size_t(row)*(k/128),deq.data(),k);
        for(int col=0;col<n;++col) {
            double value=0;
            for(int e=0;e<k;++e) {
                const auto & b=hq[col*(padded/32)+e/32];ggml_fp16_t d;memcpy(&d,&b.ds,2);
                value+=double(deq[e])*GGML_FP16_TO_FP32(d)*int(b.qs[e%32]);
            }
            if(fabs(value-expected[size_t(col)*m+row])>1e-4*(1+fabs(value)))
                GGML_ABORT("CPU oracle mismatch k=%d m=%d n=%d row=%d col=%d cpu=%g gpu=%g scale=%g",
                    k,m,n,row,col,value,expected[size_t(col)*m+row],__half2float(hw[0].d));
        }
    }
    double quant_us = bench ? timing([&]{quantize_ptq1_pascal_cuda(x,a,k,n,k,nullptr);})*1000 : 0;
    for(int threads:{96,128,160,192}) for(int rows:{1,2,4}) for(bool flat:{true,false}) {
        auto launch=[&]{ptq1_pascal_launch({threads,rows,flat},w,a,dst,k,m,n,k/128,m,{},nullptr);};
        launch();CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(),dst,out.size()*4,cudaMemcpyDeviceToHost));
        double err=0,norm=0,maxabs=0;
        for(size_t i=0;i<out.size();++i) {double d=out[i]-expected[i];err+=d*d;norm+=double(expected[i])*expected[i];maxabs=fmax(maxabs,fabs(d));}
        if(!std::isfinite(err) || err/fmax(norm,1e-30)>1e-10) GGML_ABORT("Pascal PTQ1 oracle mismatch");
        double us=bench ? timing(launch)*1000 : 0;
        printf("{\"k\":%d,\"m\":%d,\"n\":%d,\"threads\":%d,\"rows\":%d,\"ownership\":\"%s\",\"gemv_us\":%.6f,\"quant_us\":%.6f,\"nmse\":%.12g,\"max_abs\":%.8g}\n",
            k,m,n,threads,rows,flat?"flat":"warp",us,quant_us,err/fmax(norm,1e-30),maxabs);
        fflush(stdout);
    }
    if (bench) {
        const auto c=ptq1_pascal_select(k);
        // Isolate raw-digit correction and GP104 read-only caching while
        // holding geometry, Q8 bytes, and the weight lookahead unchanged.
        double normal=timing([&]{ptq1_pascal_launch(c,w,a,dst,k,m,n,k/128,m,{},nullptr);})*1000;
        double centered,uncached;
        auto verify=[&] {
            CUDA_CHECK(cudaMemcpy(out.data(),dst,out.size()*4,cudaMemcpyDeviceToHost));
            double err=0,norm=0;
            for(size_t i=0;i<out.size();++i) {double d=out[i]-expected[i];err+=d*d;norm+=double(expected[i])*expected[i];}
            if(!std::isfinite(err) || err/fmax(norm,1e-30)>1e-10) GGML_ABORT("Ablation oracle mismatch");
        };
#define PTQ1_ABLATE(T,R) \
        centered=timing([&]{ptq1_pascal_launch_ownership<T,R,false,true>(c.flat,w,a,dst,k,m,n,k/128,m,{},nullptr);})*1000; verify(); \
        uncached=timing([&]{ptq1_pascal_launch_ownership<T,R,true,false>(c.flat,w,a,dst,k,m,n,k/128,m,{},nullptr);})*1000; verify();
        if(c.threads==96 && c.rows==4) {PTQ1_ABLATE(96,4)}
        else if(c.threads==96 && c.rows==2) {PTQ1_ABLATE(96,2)}
        else {PTQ1_ABLATE(128,4)}
#undef PTQ1_ABLATE
        printf("{\"ablation\":true,\"k\":%d,\"m\":%d,\"n\":%d,\"threads\":%d,\"rows\":%d,\"ownership\":\"%s\",\"raw_ldg_us\":%.6f,\"centered_ldg_us\":%.6f,\"raw_l2_us\":%.6f}\n",
            k,m,n,c.threads,c.rows,c.flat?"flat":"warp",normal,centered,uncached);
    }
    CUDA_CHECK(cudaFree(w));CUDA_CHECK(cudaFree(x));CUDA_CHECK(cudaFree(a));CUDA_CHECK(cudaFree(aos));CUDA_CHECK(cudaFree(dst));CUDA_CHECK(cudaFree(ref));
}

int main(int argc,char ** argv) {
    ggml_context * init=ggml_init({1024,nullptr,true});ggml_free(init);
    ggml_backend_t backend=ggml_backend_cuda_init(0);
    if(!backend) return 1;
    const auto & device=ggml_cuda_info().devices[0];
    if(device.cc!=610) {fprintf(stderr,"This sweep targets sm_61; detected %d\n",device.cc);ggml_backend_free(backend);return 77;}
    bool bench=argc>1 && !strcmp(argv[1],"--bench");
    if(bench && argc==5) shape(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),true);
    else if(bench) for(auto s: std::initializer_list<std::pair<int,int>>{{5120,1024},{5120,6144},{5120,10240},{5120,12288},{5120,17408},{6144,5120},{17408,5120},{5120,248320}})
        for(int n:{1,2,4}) shape(s.first,s.second,n,true);
    else {
        for(int k:{128,384,4096,5120,6144,17408}) for(int n:{1,2,3,4}) shape(k,37,n,false);
        shape(5120,37,4,false,true);
    }
    ggml_backend_free(backend);return 0;
}
