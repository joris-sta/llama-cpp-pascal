#include "ggml.h"
#include "arg.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstring>
#include <cstdlib>

// Logical attention Q is [D, query, head, sequence], while model projections
// store [D, head, query, sequence] before the attention permutation.
static std::vector<float> model_query_storage(const std::vector<float> & q,
        int d, int queries, int heads, int sequences) {
    std::vector<float> storage(q.size());
    for (int seq = 0; seq < sequences; ++seq) for (int head = 0; head < heads; ++head)
        for (int col = 0; col < queries; ++col)
            std::copy_n(q.data()+((seq*heads+head)*queries+col)*d, d,
                        storage.data()+((seq*queries+col)*heads+head)*d);
    return storage;
}

// Compare CUDA compressed-cache decode with FP32 CPU attention on the SAME
// decoded centroids, signed-FWHT query rotation, mask and attention sinks.
// This tests the kernel, independently of model quality or quantization loss.
static bool check(ggml_backend_t gpu, int d, int ratio, int kv, bool masked, bool sinks, float bias, float softcap, int heads = 2, int sequences = 2, int queries = 1, int prefix_rows = 0, bool strided_queries = false) {
    std::mt19937 rng(42 + d + ratio + kv);
    std::uniform_real_distribution<float> random(-0.5f, 0.5f);
    std::vector<float> q(d * queries * heads * ratio * sequences);
    std::vector<float> k(d * kv * heads * sequences), v(k.size());
    for (auto & x : q) x = random(rng);
    for (auto & x : k) x = random(rng);
    for (auto & x : v) x = random(rng);
    const auto * trait = ggml_get_type_traits(GGML_TYPE_TURBO3_0);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO3_0, d);
    std::vector<uint8_t> kb(row_bytes * kv * heads * sequences), vb(kb.size());
    for (int row = 0; row < kv * heads * sequences; ++row) {
        trait->from_float_ref(k.data() + row*d, kb.data() + row*row_bytes, d);
        trait->from_float_ref(v.data() + row*d, vb.data() + row*row_bytes, d);
        trait->to_float(kb.data() + row*row_bytes, k.data() + row*d, d);
        trait->to_float(vb.data() + row*row_bytes, v.data() + row*d, d);
    }
    std::vector<ggml_fp16_t> mask(kv * queries * sequences);
    for (int seq = 0; seq < sequences; ++seq) {
        for (int col = 0; col < queries; ++col) for (int i = 0; i < kv; ++i)
            mask[(seq*queries+col)*kv+i] = ggml_fp32_to_fp16(
                (prefix_rows ? i >= prefix_rows : i > kv-queries+col-17) ? -INFINITY : float(-(i%7))*0.1f);
    }
    std::vector<float> sink(heads * ratio, 0.25f);
    const auto run = [&](ggml_backend_t backend, bool compressed) {
        ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
        const bool strided = compressed && strided_queries;
        ggml_tensor * qs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d,
                strided ? heads*ratio : queries, strided ? queries : heads*ratio, sequences);
        ggml_tensor * qt = strided ? ggml_cont(ctx, ggml_permute(ctx, qs, 0, 2, 1, 3)) : qs;
        const ggml_type type = compressed ? GGML_TYPE_TURBO3_0 : GGML_TYPE_F32;
        // K/V views have gaps between heads and between sequences, as real caches do.
        ggml_tensor * ks = ggml_new_tensor_4d(ctx, type, d, 2*kv, heads, sequences);
        ggml_tensor * vs = ggml_new_tensor_4d(ctx, type, d, 2*kv, heads, sequences);
        ggml_tensor * kt = ggml_view_4d(ctx, ks, d, kv, heads, sequences, ks->nb[1], ks->nb[2], ks->nb[3], 0);
        ggml_tensor * vt = ggml_view_4d(ctx, vs, d, kv, heads, sequences, vs->nb[1], vs->nb[2], vs->nb[3], 0);
        ggml_tensor * mt = masked ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, queries, 1, sequences) : nullptr;
        ggml_tensor * st = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads*ratio) : nullptr;
        ggml_tensor * rotated = compressed ? qt : ggml_turbo_wht(ctx, qt, 0);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, rotated, kt, vt, mt, 1.0f/std::sqrt(float(d)), bias, softcap);
        ggml_flash_attn_ext_add_sinks(out, st);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("test tensor allocation failed");
        ggml_backend_buffer_clear(buffer, 0);
        const auto q_storage = strided ? model_query_storage(q,d,queries,heads*ratio,sequences) : q;
        ggml_backend_tensor_set(qs, q_storage.data(), 0, q_storage.size()*sizeof(float));
        for (int seq = 0; seq < sequences; ++seq) for (int head = 0; head < heads; ++head) {
            const int row = (seq*heads+head)*kv;
            const size_t offset = seq*ks->nb[3] + head*ks->nb[2];
            ggml_backend_tensor_set(ks, compressed ? (const void *)(kb.data()+row*row_bytes) : (const void *)(k.data()+row*d), offset, kv*(compressed ? row_bytes : d*sizeof(float)));
            ggml_backend_tensor_set(vs, compressed ? (const void *)(vb.data()+row*row_bytes) : (const void *)(v.data()+row*d), offset, kv*(compressed ? row_bytes : d*sizeof(float)));
        }
        if (mt) ggml_backend_tensor_set(mt, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
        if (st) ggml_backend_tensor_set(st, sink.data(), 0, sink.size()*sizeof(float));
        // Repeated CUDA evaluation exercises graph capture/replay when enabled.
        for (int repeat = 0; repeat < (compressed ? 3 : 1); ++repeat) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("test graph compute failed");
        }
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return result;
    };
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    const auto expected = run(cpu, false), actual = run(gpu, true);
    ggml_backend_free(cpu);
    double error = 0, energy = 0, largest = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite CUDA output");
        const double delta = actual[i]-expected[i];
        error += delta*delta; energy += double(expected[i])*expected[i];
        largest = std::max(largest, std::abs(delta));
    }
    const double nmse = error/std::max(energy, 1e-30);
    const bool ok = nmse < 5e-5 && largest < 1e-3;
    std::printf("D=%d GQA=%d KV=%d Q=%d strided_q=%d mask=%d sinks=%d bias=%.1f softcap=%.1f NMSE=%.3g max_abs=%.3g %s\n", d,ratio,kv,queries,strided_queries,masked,sinks,bias,softcap,nmse,largest,ok?"PASS":"FAIL");
    return ok;
}
// End-to-end CUDA attention microbenchmark with Ornith's TKHE cache strides.
// Keep the model weights out of this test so candidates can be swept cheaply;
// selected variants are subsequently checked with the actual model/snapshot.
static void benchmark(ggml_backend_t gpu, int kv, int d = 256, int ratio = 4, int queries = 1) {
    constexpr int heads = 4;
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    const bool strided = std::getenv("T3_BENCH_STRIDED") != nullptr;
    const bool packed = std::getenv("T3_BENCH_PACK") != nullptr;
    auto * qs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d,
            strided ? heads*ratio : queries, strided ? queries : heads*ratio, 1);
    auto * q = strided ? ggml_permute(ctx, qs, 0, 2, 1, 3) : qs;
    if (packed) q = ggml_cont(ctx,q);
    auto * ks = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO3_0, d, heads, kv, 1);
    auto * vs = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO3_0, d, heads, kv, 1);
    auto * k = ggml_view_4d(ctx, ks, d, kv, heads, 1, ks->nb[2], ks->nb[1], ks->nb[3], 0);
    auto * v = ggml_view_4d(ctx, vs, d, kv, heads, 1, vs->nb[2], vs->nb[1], vs->nb[3], 0);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, queries, 1, 1);
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/std::sqrt(float(d)), 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, out);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buffer) throw std::runtime_error("benchmark allocation failed");
    ggml_backend_buffer_clear(buffer, 0);
    std::mt19937 rng(42); std::uniform_real_distribution<float> random(-0.5f,0.5f);
    std::vector<float> query(d*queries*heads*ratio); for (auto & x : query) x = random(rng);
    const auto q_storage = strided ? model_query_storage(query,d,queries,heads*ratio,1) : query;
    ggml_backend_tensor_set(qs, q_storage.data(), 0, q_storage.size()*sizeof(float));
    const auto * trait = ggml_get_type_traits(GGML_TYPE_TURBO3_0);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO3_0, d);
    std::vector<uint8_t> encoded(row_bytes*heads*kv);
    std::vector<float> values(d);
    for (auto * tensor : {ks, vs}) {
        // A 1024-position random pool, repeated into distinct cache addresses.
        const int pool_rows = std::min(kv,1024)*heads;
        for (int row = 0; row < pool_rows; ++row) {
            for (auto & x : values) x = random(rng);
            trait->from_float_ref(values.data(), encoded.data()+row*row_bytes, d);
        }
        for (int row = pool_rows; row < kv*heads; ++row)
            std::memcpy(encoded.data()+row*row_bytes, encoded.data()+(row%pool_rows)*row_bytes, row_bytes);
        ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    }
    for (int i = 0; i < 10; ++i) if (ggml_backend_graph_compute(gpu,graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("warmup failed");
    const int repeats = queries == 1 ? 64 : 8;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) if (ggml_backend_graph_compute(gpu,graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("benchmark failed");
    const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
    std::vector<float> result(ggml_nelements(out)); ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
    double checksum = 0; for (float value : result) { if (!std::isfinite(value)) throw std::runtime_error("nonfinite benchmark output"); checksum += value; }
    std::printf("BENCH KV=%d D=%d GQA=%d Q=%d strided=%d packed=%d ms=%.6f checksum=%.9g\n",kv,d,ratio,queries,strided,packed,ms,checksum);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--layout-check") == 0) {
        ggml_backend_load_all();
        auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
        if (!gpu) return 2;
        bool ok = true;
        for (const auto geometry : {std::pair<int,int>{128,4}, {128,8}, {256,6}})
            for (int queries : {2,3,4})
                ok = check(gpu,geometry.first,geometry.second,4096,true,false,0,0,1,2,queries,0,true) && ok;
        ok = check(gpu,256,6,65536,true,false,0,0,1,1,2,0,true) && ok;
        ggml_backend_free(gpu);
        std::printf("10 model-query layout numerical cases: %s\n",ok?"PASS":"FAIL");
        return ok ? 0 : 1;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
        ggml_backend_load_all(); auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
        if (!gpu) return 2;
        if (argc >= 3) benchmark(gpu,std::stoi(argv[2]),argc>3?std::stoi(argv[3]):256,argc>4?std::stoi(argv[4]):4,argc>5?std::stoi(argv[5]):1);
        else for (int kv : {32768,65536,131072}) benchmark(gpu,kv);
        ggml_backend_free(gpu); return 0;
    }

    if (common_parse_kv_q8_layers("3,7,3,11-13") != std::vector<int32_t>({3,7,11,12,13})) return 1;
    for (const char * bad : {"", "-1", "x", "3,", "3,,7", "7-3", "3--7", "65536"}) {
        bool rejected = false;
        try { common_parse_kv_q8_layers(bad); } catch (const std::exception &) { rejected = true; }
        if (!rejected) { std::fprintf(stderr, "parser accepted invalid layer list: %s\n", bad); return 1; }
    }
    std::puts("Q8 layer-list parsing: PASS");
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) return 2;
    bool ok = true;
    int cases = 0;
    for (int d : {128,256,512}) {
        for (int ratio : {1,2,3,4,6,8,16,32}) for (int kv : {256,513,2048}) {
            ok = check(gpu,d,ratio,kv,true,false,0,0) && ok; ++cases;
        }
        ok = check(gpu,d,4,513,false,false,0,0) && ok; ++cases;
        ok = check(gpu,d,4,2048,true,true,0,8) && ok; ++cases;
        ok = check(gpu,d,4,513,true,false,2,0) && ok; ++cases;
    }
    // Ratios 12 and 20 exercise partial groups within each KV head.
    // Register head indexing must preserve the real ratio and sequence offsets.
    for (int ratio : {12,20}) for (int kv : {256,513,2048}) {
        ok = check(gpu,256,ratio,kv,true,false,0,0) && ok; ++cases;
    }
    for (int ratio : {4,6}) for (int kv : {16384,65536}) {
        ok = check(gpu,256,ratio,kv,true,false,0,0) && ok; ++cases;
    }
    // Validate the actual 128K depth without allocating multi-gigabyte CPU views.
    for (int ratio : {4,6}) {
        ok = check(gpu,256,ratio,131072,true,false,0,0,1,1) && ok; ++cases;
    }
    // Native tiled prefill: multiple query columns, partial query tiles,
    // odd GQA, non-padded histories, masks, sinks, softcap and ALiBi.
    for (int d : {128,256}) for (int ratio : {1,2,3,4,6,8,12}) for (int queries : {2,17,32}) {
        ok = check(gpu,d,ratio,256,true,false,0,0,1,1,queries) && ok; ++cases;
    }
    for (int d : {128,256}) {
        ok = check(gpu,d,4,513,true,true,0,8,1,2,17) && ok; ++cases;
        ok = check(gpu,d,3,513,false,false,0,0,1,1,17) && ok; ++cases;
        ok = check(gpu,d,4,513,true,false,2,0,1,1,17) && ok; ++cases;
    }
    // Automatic short-suffix dispatch at meaningful attended depths.
    for (int ratio : {4,6,12}) for (int queries : {2,3,4}) {
        ok = check(gpu,256,ratio,4096,true,false,0,0,1,1,queries) && ok; ++cases;
    }
    for (int queries : {2,3,4}) {
        ok = check(gpu,128,8,4096,true,false,0,0,1,1,queries) && ok; ++cases;
    }
    for (int ratio : {4,6}) {
        ok = check(gpu,256,ratio,131072,true,false,0,0,1,1,4) && ok; ++cases;
        ok = check(gpu,256,ratio,65536,true,true,0,0,1,1,1,3) && ok; ++cases;
    }
    for (int d : {128,256}) {
        ok = check(gpu,d,9,2048,true,false,0,0,1,1) && ok; ++cases;
    }
    ok = check(gpu,256,24,2048,true,false,0,0,1,1) && ok; ++cases;
    ggml_backend_free(gpu);
    std::printf("%d numerical cases: %s\n",cases,ok?"PASS":"FAIL");
    return ok ? 0 : 1;
}
