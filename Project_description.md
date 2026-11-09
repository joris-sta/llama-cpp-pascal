# llama.cpp — Pascal Inference Fork: Project Description

The llama.cpp fork here powers all local inference in this machine's
agent stack (Pi). It is built around a single unglamorous constraint: the
production GPU is a 2016 GTX 1070 — Pascal, SM 6.1, 8 GB VRAM, no tensor
cores — and the workloads are real, sustained agentic use of large models:
Bonsai-2-27B, a ternary-quantized dense model, at an 80K–96K token context;
Ornith-9B, a Qwen3.5-native full-attention + recurrent hybrid, allocating a
262,144-token context; plus OxCoder-9B (with a multimodal projector and MTP
drafting) and NeoHorse-4B, served by the same binary through a model router.
Stock llama.cpp cannot do any of this on this hardware. The fork contributes
four layers of custom engineering, each documented, benchmarked and covered by
dedicated numerical tests: a native compressed-KV attention kernel family for
pre-tensor-core CUDA, ternary-weight matmul kernels, a dynamic variable-bitrate
KV cache system, and a server-side retention engine for persistent prompt
caching. The full engine runs in production today, with the exact binaries in
`build-pascal/bin` driving long-lived agent sessions through cache slots that
survive restarts.

## 1. TurboQuant KV cache codecs

The fork defines a custom KV-cache quantization ladder — `turbo8 → turbo4 →
turbo3 → turbo2 → turbo1`, with combined-TCQ variants — spanning roughly 8 bits
down to 1.25 effective bits per value, compared with stock options of f16,
q8_0, and q4_0. Unlike a naive quantized cache, K and V rows are stored in a
**rotated space** using signed Fast Walsh–Hadamard Transforms (custom
`fwht.cu` CUDA kernels), which flattens outliers before quantization so the
aggressive bit rates stay numerically clean. Each tier has custom packing and
dequantization with its own GGML type ID, and flash attention is forced on
because only the FA paths can decode the rotated-space rows. Quality tiers
were validated with KLD panels against f16 reference logits. Turbo3 — the
production setting for both K and V — cuts the KV footprint to roughly 3.5
bits per value versus 16 for f16, about 3× more context in the same VRAM; this
is what makes 80K–96K token contexts and a full 262K-token allocation fit in
8 GB.

On top of the static tiers, the fork adds **per-layer Q8 pinning**
(`--kv-q8-layers 3,7,11-15`): individual zero-based model layers can hold q8_0
K/V while siblings keep the global Turbo type, with strict validation
(recurrent and SWA layers rejected, no conflict with dynamic VBR, no conflict
with shared/reused cache layers) so nothing degrades silently. Ornith's
production profile pins its two boundary layers this way.

## 2. Hard-native Turbo3 attention (the headline work)

Off-the-shelf llama.cpp cannot read Turbo3 rows on old hardware: Pascal has no
tensor cores and DDR5 bandwidth is the bottleneck, and the stock flow
dequantizes the whole cache into a separate FP16 buffer. The fork writes its
own attention:

- **Native compressed-KV decode tile**: single-token decode attention decodes
  compressed Turbo3 K/V *inside* the flash-attention kernel. There is no
  full-cache FP16 materialization path reachable at all for that branch — the
  dequantize-then-multiply roundtrip is eliminated. Unsupported shapes abort
  explicitly rather than falling back.
- **GQA row sharing**: each decoded KV row is decoded once and shared across
  the whole query group — tiles of 2, 4, 6 or 8 heads, including an exact
  six-head group at D=256, matching real model geometries (Bonsai's 6:1,
  Ornith's 4:1). Dispatch is shape-driven, not model-name-dependent; larger
  groups split into divisor tiles, and odd ratios, unmasked attention, ALiBi
  and unpadded tails use a native single-head tile, never an FP16 fallback.
- **Register-resident decode kernel**: for the production D=256/GQA=4 shape on
  sm_61, the four queries and their accumulators live entirely in registers;
  each warp decodes K/V once and reuses it across all four heads with zero
  shared-memory staging and no CTA barriers during the history scan. The
  heuristic uses one warp through 65,536 attended rows and four warps above,
  with `GGML_CUDA_T3_DECODE_TILE=auto|scalar|warp1|warp4` for reproducible A/B
  checks.
- **Turbo3 prefill tiles**: small multi-token suffix batches (up to 4 rows)
  run the same native tile under `GGML_CUDA_T3_PREFILL`, and the graph layer
  packs their strided queries so even small verification batches never
  materialize KV. Full prefill retains a validated FP16 path on Pascal.
- Query rotation is applied once through the signed FWHT kernels; an opt-in
  per-operation CUDA event profiler (`GGML_CUDA_PROFILE_OPS`) prints
  `CUDA_OP_PROFILE` timings and proved attention dominates context-depth
  decode cost, driving the register-kernel work.

Correctness is asserted by a dedicated test binary, `test-turbo3-native`, that
compares compressed CUDA decode against **FP32 CPU attention over the same
decoded centroids**: 87 numerical cases covering head sizes 128/256/512, GQA
groups of 1–32, chunked and unpadded contexts, multiple sequences, strided
K/V views, absent masks, attention sinks, ALiBi and softcapping — each with
and without forced CUDA graph capture/replay.

**Measured gains (GTX 1070, greedy output identical across comparisons):**

| Change | Isolated kernel | Real model decode |
|---|---|---|
| Native T3/T3 tile vs. previous fused T3 | — | +18% @ 11K ctx (17.8 → 21.0 tok/s); +37% @ 34K (12.0 → 16.4) |
| Register kernel vs. general native tile | 128K attention 8.65 → **5.48 ms** (−37%) | +15% @ 32K (16.5 → 19.0), +24% @ 64K (12.3 → 15.2), +34% @ 128K (8.6 → 11.5 tok/s) |

Filled-window prefill on the production configuration measured 334.7 / 244.7 /
162.7 tok/s at 32K/64K/128K cold prompts, completing without a single CUDA
error at those depths, at 81–83 °C sustained GPU temperature. Two tempting but
regressive experiments (a cooperative shuffle unpacker, a shared-memory
codebook) were measured, found to hurt throughput, and removed — the tuning
log records what did not work as carefully as what did.

## 3. Ternary weight inference (Bonsai-2)

The fork ports PrismML's ternary codec (`PTQ1_0` and `PQ2_0` packings), ports
the blockwise Hadamard activation transform machinery it requires, and then
adds real optimizations beyond the upstream port:

- **Loader-level guarantees**: the model's sign vectors, transform version,
  tensor names and head permutation are validated at load; a graph-coverage
  check rejects folded weights whose consumers bypass the activation
  transform. Stored weights carry blockwise Hadamard rotations folded into
  them, so loading the codec alone is not sufficient — the engine administers
  graph-typed transforms keyed by input, rotation, signs and head permutation.
- **Pascal DP4A matmuls**: MMQ entries lift the sm_61 shared Q8-SRAM tile to
  PTQ1 prefill (each weight block decodes once into the stock tile), and a
  dedicated dense PTQ1-only GEMV path for decode reuses one quantized Q8
  activation row across four PTQ1 output rows per block, fusing quantization
  with the signed-FWHT input rotation when the graph provides one.
- **Fusion kernels**: single-consumer 1024-wide signed Hadamard transforms
  write MMVQ's Q8 activations directly (no intermediate pass), and ternary
  embedding lookup on GPU fuses row lookup + inverse rotation + signs into one
  kernel, sharing a 128-thread Hadamard helper. Wider PTQ MMQ tiles reuse
  unpacked weights across more prompt tokens.

Result: Bonsai-2 27B prefill throughput went from **~747 tok/s (upstream
reference) to ~1,112 tok/s (+49%)** at equal perplexity (8.6272 vs 8.6253;
99.46% top-token agreement) on the PTQ1 packing, verified with 426+ CUDA
kernel-level tests and CUDA memcheck over all 28 whole-graph cases with zero
errors. In production the model serves ~60 tok/s decode with n-gram
speculative decoding over Turbo3 KV.

## 4. Gated delta net — the recurrent memory path

Ornith-9B's Qwen3.5-native geometry mixes full-attention layers with
**gated-delta-net recurrent memory** instead of a KV cache. The fork gives
that layer a hand-written chunked kernel for SM 6.1
(`gdn-chunk-pascal.cuh`, activated by `GGML_CUDA_GDN_PASCAL=auto` and
`GGML_CUDA_GDN_PASCAL_CHUNK=16|32`) — the recurrent equivalent of the Turbo3
attention work:

- Each warp holds a **permanent register shard of the D×D recurrent state**
  across an entire token tile; per-CTA token tiles stream B tokens at a time
  and reuse shared-memory-normalized Q/K across C value columns, amortizing
  global loads.
- The fused delta-rule update `s = g·s + k·β·(v − g·(s·k))`, the output
  projection and on-the-fly L2 normalization (sum-of-squares-epsilon or
  vectorized L2 modes) run inside the same register loop, with optional
  last-K-slot snapshot capture for the server's state-retention machinery.
- Crucially, the kernel preserves the **exact recurrent/snapshot update
  order** of the reference implementation, so numerical results match the
  reference path rather than approximating it.

Coverage comes from `test-gdn-chunk-pascal` plus extended backend-op tests
hitting chunk-crossing token counts (15–65), multi-sequence states, norm
amplitude regimes, suffix snapshots crossing chunk boundaries and state-cache
fusion combinations.

## 5. VBR — variable-bit-rate KV cache

A runtime cache management system rather than a codec: the KV cache starts at
**f16** quality, and a VRAM-budget-aware controller **transcodes individual
layer tensors down the quality ladder in place** (cheapest quality-per-byte
first, with orders measured per-model on KLD panels) as pressure builds.
`llama-server -ctk vbr` is the entire interface; a floor flag controls the
minimum aggregate bits/value, and the advertised context is derived from the
budget's capacity at the floor tier. Host-side prompt caching uses
authenticated, immutable projected artifacts so warmed sessions can hold
tier-typed KV safely across restarts, with typed, explicit live-only fallbacks
for unsupported topologies (media prompts, aLoRA boundaries, classic codecs)
instead of best-effort guesswork. The ladder resets losslessly when the cache
empties, and every fully-degraded run stayed coherent in validation.

## 6. Server retention engine

The production reliability layer for prompt memory:

- **Disk-budgeted persistent resume** (`--resume-max-mib`,
  `--resume-budget-path`): byte-accurate, recursion that refuses to follow
  symlinks, accounting that folds adjacent cache directories into one budget
  (8 GiB in production), eviction of only the store's own entries, and
  never a destructive decision based on an unreadable root. The Pi launcher
  wires the slot-cache file store and resume store together under the budget.
- **Checkpoint authority**: an incoming context checkpoint may replace the
  earliest recovery point of a mandatory retention-turn bucket only when
  replay-boundedness and retention scoring prove coverage — the rule that let
  production move from 8192-token checkpoint spacing to request-boundary
  checkpoints (`--checkpoint-min-step 0`), eliminating long replays after
  tool calls. `--checkpoint-max-mib` bounds each automated capture so large
  MTP/frontier checkpoints cannot balloon the displaced host cache.
- **Cache identity digests**: token count + retention token digest + sequence
  epoch, exposed through the cache control plane so the Pi gateways can verify
  prompt-cache lineage across restarts.
- **Atomic streaming tool calls** (PEG-native parser): partial parses never
  retract an already-streamed SSE prefix and never expose a half-parsed tool;
  tool calls are held back and published atomically after a strict final
  parse, malformed final tool requests degrade to a visible text notice
  instead of hanging or crashing the stream, and the qwen3-coder framing
  grammar can no longer let one unterminated parameter swallow subsequent
  tool calls.

## 7. Rigor and rejected ideas

Every performance claim is anchored to same-output (bit-identical greedy text)
paired comparisons on idle GPUs with three repeats; failed experiments are
recorded alongside successes — including a rejected one-warp fused activation
version of the Hadamard transform (61.53 → 59.70 tok/s regression), two
rejected 128K decode variants, rejected speculative-dispatch restructuring,
and unqualified platforms (Metal/Vulkan/HIP pre-tensor-core paths) marked as
explicitly unqualified rather than silently assumed. Numerical tests separate
kernel correctness from quantization quality (the compressed CUDA tests
compare against FP32 CPU attention on the same decoded centroids, not against
CPU quantized attention). Tests run both ordinarily and with forced CUDA graph
capture/replay, because production uses graphs.

## Result

A 27B ternary model and a 9B hybrid both running sustained multi-hour agentic
workloads on 8 GB of 2016-era VRAM: ~60 tok/s decode at 32K depth, ~330 tok/s
prefill, ~1,100 tok/s prefill on ternary matmuls, 80K–262K contexts that fit
because of custom cache codecs, a full 128K-token window decoded
context-dependently at +34% over the already-custom native tile, and session
state that survives process restarts as authenticated artifacts rather than a
full prefill. Everything is committed as layered, reviewable history with its
own test wheel (`test-turbo3-native`, `test-ptq1-pascal`,
`test-gdn-chunk-pascal`, extended backend-op coverage).
