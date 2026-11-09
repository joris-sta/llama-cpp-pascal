# Native Turbo3 decode and selected Q8 layers

This isolated build extends the local Pascal 6:1 fused tile into shape-based
CUDA T3/T3 decode. It does not depend on a Qwen model name, 6:1 GQA, a minimum
context length, or the old `GGML_CUDA_TURBO3_SM61_FUSED` environment variable.

For single-token CUDA decode with `turbo3` K and V, supported equal head
sizes are 128, 256 and 512, with contiguous queries. Query rotation is applied once; compressed K/V
are decoded inside the flash-attention tile (or the existing fused MMA kernel on tensor-core GPUs). No full-cache FP16 materialization
is reachable for that decode branch. Unsupported dimensions/layouts fail
explicitly. This is a decode guarantee; multi-token prefill keeps its existing
implementation. It is not a promise about unknown future head layouts or
non-CUDA backends.

Padded masked caches use tiles sharing decoded rows among 2, 4 or 8 query
heads. D=256 also supports an exact six-head group. Larger GQA groups are
split into divisor-sized tiles. Odd ratios, unmasked attention, ALiBi and
unpadded tails use a native single-head tile, not an FP16 cache fallback.

## Selected Q8 cache layers

`--kv-q8-layers 3,7,11` pins BOTH K and V in those layers to `q8_0`, while
other full-attention layers use `-ctk/-ctv`. Indices are zero-based model-layer
indices, not ordinal indices among full-attention layers. Inclusive ranges
are accepted, for example `3,7,11-15`; duplicates are deduplicated. Every
selected index must refer to an existing full-attention layer. Recurrent
and sliding-window layers are rejected. For Ornith 9B the valid indices
are `3,7,11,15,19,23,27,31`.

The option is for fixed cache types. Dynamic VBR is rejected with selected
Q8 layers so they cannot silently degrade. Q8 layers use the existing Q8
attention implementation; the hard native T3 guarantee applies to T3/T3
layers. Q8 uses more VRAM and can lower the maximum context that fits.

```bash
./build-pascal/bin/llama-server -m MODEL.gguf -ngl 99 -c 65536 -np 1 \
  -fa on -ctk turbo3 -ctv turbo3 --kv-q8-layers 3,31
```

The public context API has `kv_q8_layers` and `n_kv_q8_layers`; selected
indices are copied at context creation. All applications must be rebuilt
against this modified context struct. Do not mix headers/shared libraries
with the previous build.

## Build and verification

```bash
cmake -S . -B build-pascal -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_TURBO_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16'
cmake --build build-pascal --target llama-server llama-bench test-turbo3-native -j1
./build-pascal/bin/test-turbo3-native
```

The dedicated numerical test compares compressed CUDA decode to FP32 CPU
attention over the same decoded centroids and explicit signed-FWHT query
rotation. It covers D=128/256/512, GQA=1/2/3/4/6/8/16/32, short and unpadded
contexts, two sequences, strided K/V views, absent masks, attention sinks,
ALiBi and softcapping. This separates kernel correctness from quantization
quality. Generic CPU Turbo3 flash-attention is not an appropriate oracle:
its cache rows expose rotated centroids and it lacks the required dot kernel.

## Measured result on the GTX 1070

The production launcher retains the default Pascal CUDA-graph gate and sets
`CUDA_LAUNCH_BLOCKING=1`. Asynchronous Pi requests failed during prefill with
both forced and default graph settings, including a fresh cache. Blocking
completed the Pi tool round-trip; it is the same driver-stack workaround used
by the Bonsai launcher. Numerical tests also run with forced CUDA graphs and
repeated graph evaluation. All 85 cases passed. Selected
Q8 layers 3 and 31 were tested at 65,536 context, including an OpenAI chat
request and slot save/erase/restore. Recurrent/out-of-range selections and
dynamic VBR were rejected.

Both builds used the same model, full GPU layer offload, 262,144 allocated
context, flash attention, batch 2,048 / physical batch 512, one slot, and
64 greedy output tokens. This production comparison enables
`CUDA_LAUNCH_BLOCKING=1` on BOTH builds and uses the default Pascal graph settings:

| Occupied prompt | Previous T3 decode | Native T3 decode | Increase |
| --- | ---: | ---: | ---: |
| 11,264 tokens | 17.80 tokens/s | 21.02 tokens/s | 18% |
| 33,792 tokens | 11.97 tokens/s | 16.35 tokens/s | 37% |

Greedy output matched for both comparisons. Prompt processing was 424/325
tokens/s before and 430/328 after. The full 262,144-token cache fits.
The final rebuild passed all 85 numerical cases. A Pi medium-thinking request
passed after restart and cache restore: native D=256/GQA=4 decode, read tool,
then correct `anchorContextTokens=51200` answer.

Earlier experimental runs without launch blocking measured 20.69/13.18
tokens/s on the previous build and 23.78/18.09 on the native build. The native
experiment also forced CUDA graphs, so these are combined changes rather
than an isolated kernel comparison. Those settings are not the production
configuration because asynchronous Pi prefill failed.

Full-window latency and model task quality were not benchmarked. Prefill
retains its FP16 materialization path on Pascal and has not been validated
at a filled 256 Ki window. The decode guarantee does not remove the cost
of scanning a longer cache or Qwen's recurrent layers.

Benchmarks, build logs, numerical logs, mixed-cache checks and launcher
backups were kept in local tuning-state archives during the campaign; only
the conclusions and measured numbers are reproduced here. Rebuilt
applications are server, bench, CLI and perplexity.

Production uses a separate slot-cache profile for this model. The previous
cache files are preserved, and the server and its prompt-cache gateway restart
around cache restore.

## Filled-window measurements

Measured on 2026-10-02 using the production native build and exact cold
prompts of 32/64/128 Ki tokens. Each request used repeated-sentence token IDs,
128 greedy output tokens, `cache_prompt=false`, and no reused tokens. These
are average complete-prefill rates, followed by decode at the filled depth.
The server allocation stayed 262,144, T3/T3, FA on, batch 2,048 / physical
batch 512, full GPU offload, launch blocking, and default Pascal graph settings.

| Prompt tokens | Prefill throughput | Prefill time | Decode throughput |
| ---: | ---: | ---: | ---: |
| 32,768 | 334.66 tokens/s | 97.92 s | 16.54 tokens/s |
| 65,536 | 244.67 tokens/s | 267.86 s | 12.27 tokens/s |
| 131,072 | 162.65 tokens/s | 805.88 s | 8.27 tokens/s |

All three requests completed without CUDA or allocation errors. Long runs
observed roughly 81–83 °C on the GTX 1070 and roughly 7.4 GiB reported
VRAM use during late prefill. These are single synthetic runs, not a model-quality evaluation.
The serving stack's cache gateway was paused during measurement and
restarted afterward; its existing session snapshot was preserved. Full
response JSON, exact configuration and timings were archived locally and are
summarized by the tables in this document.

## Pascal register decode optimization

The D=256/GQA=4 single-query CUDA path on sm_61 now keeps the four queries
and their output accumulators in registers. Each warp decodes K/V and reuses
them across the four heads. The history scan has no shared-memory tile
staging or CTA barriers; partial warp results are merged once at the end and
then use the existing split-KV combine. Softcapping, attention sinks, strided
K/V views and matched sequence batches are supported. Other shapes retain
the general hard-native T3 implementation. Cache encoding and allocation are
unchanged; saved slots from the previous build remain usable.

`GGML_CUDA_T3_DECODE_TILE=auto` is the default. For matching Pascal shapes it
uses one warp through 65,536 attended (padded) KV rows and four warps above
that depth. `scalar`, `warp1` and `warp4` allow reproducible A/B checks. The
launcher exposes the override as `ORNITH_T3_DECODE_TILE`. This does not change
the non-T3 cache layers selected by `--kv-q8-layers`.

The dedicated test now covers 87 numerical cases including 128K depth for
GQA=4 and GQA=6. All passed normally and with forced CUDA graph capture/replay.
The CUDA attention microbenchmark models Ornith's TKHE cache strides:

```bash
CUDA_LAUNCH_BLOCKING=1 ./build-pascal/bin/test-turbo3-native --bench
GGML_CUDA_T3_DECODE_TILE=scalar CUDA_LAUNCH_BLOCKING=1 \
  ./build-pascal/bin/test-turbo3-native --bench 131072
```

The 128K microbenchmark decreased from 8.65 ms to 5.48 ms per attention
operation. A matched actual-model comparison restored the same saved 128K
cache and generated 128 greedy tokens: 8.58 tokens/s before, 11.51 after
(34% increase), with identical output text. Both used launch blocking and
default graph settings. Neither the cooperative shuffle unpacker nor the
shared-memory codebook experiment improved throughput; both were removed.

Opt-in per-operation CUDA event timings are available for diagnosis:

```bash
GGML_CUDA_PROFILE_OPS=1 GGML_CUDA_DISABLE_GRAPHS=1 \
GGML_CUDA_DISABLE_STEP_GRAPHS=1 CUDA_LAUNCH_BLOCKING=1 \
  ./build-pascal/bin/llama-server [normal model/cache arguments]
```

The profiler prints `CUDA_OP_PROFILE` records on stderr and synchronizes
each measured operation. Graph capture/replay must be disabled as shown.
These timings include launch gaps on the stream; do not use profiled
throughput as production throughput. In the restored 128K scalar run,
seven decode steps spent about 77.6 ms per token across eight FA operations.
Operation profiling confirms attention is the main context-dependent cost.

The previous binaries, cache snapshot, successful and rejected experiments,
build logs, numerical checks and actual-model JSON responses were archived
locally during the campaign. The launcher retains the full
262,144 allocation and the established prefill driver workaround.

Actual generation measurements after selecting the register kernel:

| Filled context | Previous native tile | Register decode | Increase |
| ---: | ---: | ---: | ---: |
| 32,768 | 16.54 tokens/s | 18.95 tokens/s | 15% |
| 65,536 | 12.27 tokens/s | 15.22 tokens/s | 24% |
| 131,072 | 8.58 tokens/s | 11.51 tokens/s | 34% |

32K/64K used complete cold prompts followed by 128 greedy output tokens;
prefill measured 334.7/245.1 tokens/s. 128K used the matched restored-cache
comparison described above, avoiding another full prefill. All compared
outputs matched. JSON timings are in `decode-v2/final-summary.json`.

Pi medium-thinking tool use passed after cache restore on the final build: read
`context-policy.json`, then answer `anchorContextTokens=51200`. Server and
gateway are active.
