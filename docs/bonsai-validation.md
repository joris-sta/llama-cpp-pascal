# Bonsai 2 integration validation — 2026-09-18

## Revisions and hardware

- Fork base: `30da59942c012a965c06f501313a35afbbc46ba5`.
- Prism reference: `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`, branch `prism`.
- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`.
- Dorei: one RTX 3090 24 GiB, 62 GiB host RAM; CUDA 13.3.73,
  driver 610.43.03; Release builds targeting SM86.
- Prism and the port were built with the same installed toolchain. The existing
  master regression binary was built with GCC 15.3, versus GCC 16.1 for this port.

The pre-existing master build is an archived dated DFlash release tree. Its
source snapshot matches that master's `src`, `ggml`, `common`, and `tools`
directories. It rejects the PTQ file's wire type 143; this is not already
supported by that master.

Artifacts, source snapshots, reference binaries, models' metadata, and logs
live under a per-campaign artifact root on the validation host (Dorei: one
RTX 3090 24 GiB). `initial-bin` preserves the unoptimized port;
`candidate-bin` contains shared graph transforms and wider PTQ MMQ tiles;
`final-bin` additionally shares the signed group-64/group-128 MMVQ decoder.
Set `LD_LIBRARY_PATH` to a snapshot's `bin` directory when running it: these
builds otherwise have absolute build-directory RPATHs.

## Correctness

- Production CPU codec, dot-product, exact ternary packing equivalence, and
  canonical GGUF type-ID round trips: `test-quantize-fns`, zero failures.
- CPU PTQ matmuls: 65/65 supported cases passed.
- CPU GGUF tests: 92/92 passed.
- CUDA PTQ/PQ/group-64 matmul, indexed matmul, and row lookup tests: 426/426 supported
  cases passed, including partial output tiles and decode/MMQ transitions.
- CPU and CUDA Hadamard checks: 18/18 each, including F16 input and fused
  multiplication's F16 intermediate rounding.
- Server smoke probes cover factual, arithmetic, code, explanatory, and longer
  summarization prompts. Both model packings produce coherent responses.
- Default VBR smoke probes also pass. At this short context the reported entry
  remained F16; this does **not** qualify every lossy VBR tier or pressure-induced
  re-tiering policy for this model.
- Static `turbo8` smoke probes pass with PTQ weights and a reported 8.125 bits
  per KV value. This is a coherence smoke check, not a quantitative T8 quality
  qualification.

For a fixed WikiText-2 panel, four chunks with context 2,048, comparing the
optimized PTQ port against saved Prism PTQ probabilities:

| Metric | Result |
| --- | ---: |
| Prism perplexity | 8.625304 |
| Port perplexity | 8.627174 |
| Mean KLD | 0.000149 ± 0.000004 |
| Highest-probability token agreement | 99.462% |

The final PQ2 comparison against Prism PQ2 gives perplexity 8.627174 versus
8.622612, mean KLD 0.000147 ± 0.000004, and 99.242% top-token agreement.

The numerical comparison is close, not bit-identical. Saved reference
probabilities use llama-perplexity's standard compressed format. This is a
focused compatibility panel, not a general model-quality evaluation.

## Performance methodology

No speculative decoding, all layers offloaded, F16 KV, flash attention enabled,
eight CPU threads. Reference and candidate run serially on the same GPU.

`llama-bench` uses three repeats, batch 2,048, microbatch 512:

```sh
llama-bench -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -p 512,2048 -n 128 -r 3 -ngl 99 -fa 1 -ctk f16 -ctv f16 -t 8 -o json
```

| PTQ build | PP512 tok/s | PP2048 tok/s | TG128 tok/s |
| --- | ---: | ---: | ---: |
| Prism reference | 747.1 | 743.8 | 60.15 |
| Initial port, confirmation run | 795.5 | 792.6 | 60.41 |
| Shared transforms + wider MMQ | 1,131.9 | 1,124.7 | 61.73 |
| Final confirmation | 1,115.9 | 1,109.5 | 60.21 |

For PQ2, Prism measured 1,362.9/1,354.6 tok/s PP512/PP2048 and 72.07 tok/s
TG128. The final port measured 1,476.8/1,470.1 and 73.48, respectively. Before
porting the signed-symbol decoder, the port's PQ2 decode was only 63.49 tok/s.
Small decode changes between repeats should not be interpreted as precise gains;
the GPU uses its normal dynamic clocks.

Server measurements use `scripts/bench-bonsai.py --perf-corpus`, context 8,192,
batch/microbatch 512, one slot, three different corpus slices at each prompt
length, and 256 generated tokens. Probability reporting and prompt reuse are
disabled; every recorded `cache_n` is zero. Values below are the arithmetic mean
of the three server-reported rates, excluding the separate warm-up request.

| Server build | PP512 tok/s | TG after 512 tok/s | PP2048 tok/s | TG after 2048 tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prism PTQ | 677.2 | 57.75 | 715.7 | 56.22 |
| Final PTQ | 884.5 | 60.89 | 1,039.3 | 59.16 |
| Prism PQ2 | 1,214.9 | 68.46 | 1,281.1 | 67.50 |
| Final PQ2 | 1,134.2 | 71.91 | 1,342.7 | 70.62 |

PQ2's short-prompt server prefill is slower despite faster kernel benchmarks;
the comparison retains each server's normal behavior, including the fork's
checkpoint machinery. The 2K result should not be generalized to every prompt
length or workload.

The ordinary Qwen 27B IQ3_XXS regression arm remains approximately 47 tok/s
with F16 KV: master/candidate averages were 47.37/47.39 after 512 prompt tokens
and 46.74/46.72 after 2,048. Prefill was 1,062/1,084 and 1,284/1,297 tok/s,
respectively. These are small single-host differences, not a claimed speedup
for non-Bonsai models.

The final quiet-GPU default-VBR regression arm measured master/port rates of
47.87/47.23 tok/s after a 512-token prompt and 46.61/46.60 after 2,048 tokens.
Prefill was 1,076/1,061 and 1,278/1,277 tok/s. This did not reproduce a material
regression; it also does not establish a zero-percent performance change.

## Reproducing the numerical panel

```sh
prism-bin/llama-perplexity \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf -f /path/to/wiki.test.raw \
  -ngl 99 -c 2048 -b 512 -ub 512 -fa on -ctk f16 -ctv f16 -t 8 \
  --chunks 4 --save-all-logits prism-ptq-2k.logits

build/bin/llama-perplexity \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf -f /path/to/wiki.test.raw \
  -ngl 99 -c 2048 -b 512 -ub 512 -fa on -ctk f16 -ctv f16 -t 8 \
  --chunks 4 --kl-divergence --kl-divergence-base prism-ptq-2k.logits
```

## Limits and discarded runs

### Follow-up fusion experiments

The three additional experiments compare against `final-bin` (the optimized
port above), not against Prism. `both-cta-bin` contains the retained candidates:

- Single-consumer signed Hadamard transforms feed MMVQ's Q8 activations
  directly. Shared Q/K/V or gate/up transforms retain the ordinary reusable
  F32 output. Prefill/MMQ, F16 inputs, other block sizes, split weight buffers,
  and unsupported layouts retain their existing paths.
- CUDA ternary embedding lookup can fuse the inverse Hadamard transform and
  post-transform signs. This requires the embedding lookup to be on CUDA;
  default CPU embedding placement is unchanged.
- Explicit multi-column PTQ unpack reuse passed correctness but produced only
  approximately 1–2% differences in the sampled two-to-four-column matmuls
  (41.20/59.25/75.60 us versus 40.43/58.38/74.77 us). The extra branch was removed
  rather than treating that small result as an established model-level gain.

Both retained kernels share a 128-thread, 1024-element Hadamard implementation.
The initial one-warp fused activation version serialized the Q8 epilogue and
regressed model decode (61.53 to 59.70 tok/s); it was discarded. The initial
one-warp embedding version also regressed small batches and was replaced.

Whole-graph CUDA checks pass 46/46, comprising 28 Bonsai producer/consumer and
fallback cases plus 18 Hadamard cases. CPU checks pass the same 46 cases. The
signed-transform tests now execute whole graphs, so they actually exercise
fusion rather than checking only the separate operators.

The backend performance harness previously repeated only the final node, even
for whole-graph tests. It now measures a whole graph per iteration for those
tests. The `base-fusions-perf.log` / `new-fusions-perf.log` results from before
that correction are not fusion benchmarks. Corrected timings include host
submission overhead; they are not pure kernel timings.

Representative corrected graph latency on Dorei (2048-wide activations):

| Graph | Tokens | Before, us | After, us |
| --- | ---: | ---: | ---: |
| PTQ rotation + projection | 1 | 10.28 | 8.45 |
| PTQ rotation + projection | 3 | 10.80 | 8.32 |
| PQ rotation + projection | 1 | 9.35 | 7.65 |
| PQ rotation + projection | 3 | 10.68 | 8.22 |
| PTQ embedding + inverse rotation | 1 | 10.21 | 7.82 |
| PTQ embedding + inverse rotation | 512 | 32.49 | 15.77 |
| PQ embedding + inverse rotation | 1 | 9.76 | 6.65 |
| PQ embedding + inverse rotation | 512 | 30.65 | 13.45 |

Logs: `base-cta-perf.log`, `both-cta-perf.log`, `both-cta-tests.log`.
The initial CUDA trace (`fusions-profile.nsys-rep`) verified that the activation
fusion executes in the real model, while default embedding placement does not
engage the embedding fusion.

The release snapshot is `fusion-release-bin`: it additionally rejects split
embedding buffers, without changing the measured single-device path. The final
trace, `fusion-embedding-profile.nsys-rep`, confirms both retained kernels run
with `-ot token_embd.weight=CUDA0`. CUDA memcheck reports zero errors for all
28 whole-graph Bonsai cases (`fusion-memcheck.log`). With GPU embeddings, both
packings retain the same four-chunk perplexity/KLD/top-token metrics reported
above (`fusion-{ptq,pq}-embedding-kld.log`).
The final GPU-embedding smoke checks also produce coherent factual, arithmetic,
code, explanatory, and summary responses with PTQ/Turbo8 KV and PQ/F16 KV
(`fusion-ptq-t8-smoke`, `fusion-pq-f16-smoke`).

Full-model results are much smaller than the isolated graph speedups. In a
fresh five-repeat PTQ `llama-bench` comparison, TG128 is 61.53 -> 62.69 tok/s;
PP512 is 1123.2 -> 1135.4 and PP2048 is 1122.8 -> 1125.6. The server comparison
uses the same three distinct prompts and fixed 256-token generations as above:

| Default-placement server, 2048-token prompt | Before PP | After PP | Before TG | After TG |
| --- | ---: | ---: | ---: | ---: |
| PTQ | 1038.7 | 1033.6 | 59.51 | 59.93 |
| PQ | 1351.2 | 1342.9 | 70.91 | 71.61 |

These are small single-host gains, not a general speedup guarantee. All twelve
default-placement generated responses match the pre-fusion build exactly at
temperature zero. Logs are `final-bin-fusion-server`, `both-cta-ptq-server`,
`final-bin-cta-pq-server`, and `both-cta-bin-cta-pq-server`.

With explicit CUDA embeddings, PTQ server PP2048 is 1027.9 -> 1027.2 and TG is
58.43 -> 58.85. Thus faster embedding kernels do **not** establish that moving
embeddings to CUDA improves this model's overall throughput. Placement remains
unchanged. Reproduce this arm by adding
`--override-tensor token_embd.weight=CUDA0` to `scripts/bench-bonsai.py`.

An ordinary Qwen3.8-27B IQ3_XXS `llama-bench` regression pair (F16 KV, no MTP,
same flags, three repeats) measured before/after PP512 1436.3/1425.0, PP2048
1442.3/1431.6, and TG128 49.03/48.69 tok/s. All differences are below 1%; this
is not proof of zero overhead. Logs are `*-fusion-qwen-regression.json`.

### Simplify review

Three independent read-only reviews covered architecture/reuse, hot-path cost,
and correctness. The resulting cleanup:

- Constructs Hadamard descriptors and dummy weight buffers during `no_alloc`
  fitting, so memory estimation measures the real graph and permanent weights.
  One allocation helper owns both real and dry-run rotation/sign tensors.
  Real loads still construct transforms after mmap tensor buffers are bound.
- Places embedding inverses with the first layer, rather than the last weight
  visited in an unordered map.
- Rejects malformed optional inverse metadata and duplicate sign widths.
- Declines unsupported PTQ/PQ Metal row lookups as well as matmuls, allowing
  scheduler fallback instead of requesting nonexistent Metal kernels.
- Removes unused multi-column PTQ dot-product scaffolding and diagnostic graph
  scans. Wire IDs, packing, arithmetic order, and the retained fusions are unchanged.

The new `test-llama-archs -a llama -s 1234` contract compares real/dry model-memory
maps and reserved graph operation types/shapes, checks allocation-free transform
descriptors, and rejects malformed metadata. It passes locally on CPU and on
Dorei with CPU, mixed CPU/GPU, and full GPU layer placement. Metadata ordering is
reversed between the two loads. The mixed arm offloads two layers (including the
output), explicitly verifying that the first repeating layer stays on CPU and
the second moves to GPU. A serialized GGUF/mmap reload in each placement arm
also pins the real-load buffer-binding order.

The cleaned snapshot is `simplify-bin`. CPU checks pass all 92 GGUF tests,
quantization-function tests, 73 PTQ lookup/matmul cases, and 46 Hadamard/fusion
cases. CUDA passes 296 ternary lookup/matmul/expert-matmul cases and 46
Hadamard/fusion cases. All 28 Bonsai whole-graph cases pass CUDA memcheck with
zero errors. Logs: `simplify-loader-final.log`, `simplify-fusion-final.log`,
`simplify-ternary.log`, and `simplify-memcheck.log` under the Dorei artifact root.

Repeated randomized projection checks exposed a pre-existing test tolerance
issue in both snapshots: the old build reached NMSE 1.32e-7 and the cleaned build
1.40e-7 against a pure-float 1e-7 limit. Projection graphs include Q8 activation
rounding after differently accumulated CPU/GPU transforms. Only these projection
cases now allow 5e-7 NMSE; standalone transforms and inverse-embedding graphs
retain their original limit. The comparison runs are saved as
`simplify-check-{fusion-release-bin,simplify-bin}-*.log`.

Fresh serial server comparisons against `fusion-release-bin`, with three
distinct prompts per length and 256 generated tokens per request:

| Packing | Prompt tokens | Before PP | Cleaned PP | Before TG | Cleaned TG |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTQ | 512 | 885.41 | 872.62 | 61.81 | 61.07 |
| PTQ | 2048 | 1040.73 | 1037.48 | 60.66 | 59.93 |
| PQ | 512 | 1135.25 | 1114.93 | 73.60 | 73.07 |
| PQ | 2048 | 1347.78 | 1341.16 | 72.43 | 71.69 |

All twelve generated responses match exactly. All measured differences are
below 2%, but this is not proof of zero overhead: decode is about 1–1.2% lower
in this comparison. A subsequent PTQ reference recheck measured 60.72 tok/s at
2048 tokens. Logs are `simplify-final-{fusion-release-bin,simplify-bin}-*-server`
and `simplify-ptq-reference-recheck`.

The explicit fit smoke (`LLAMA_ARG_FIT=on`, `LLAMA_ARG_LOG_VERBOSITY=4`,
`-ngl auto`, `-ctk turbo8 -ctv turbo8`, `-ot token_embd.weight=CUDA0`) completed
its fit in 0.29 seconds and produced coherent answers to all five smoke prompts.
Artifacts: `simplify-explicit-fit-smoke`, including the full command and server
log. The harness was invoked through its `run()` entry point to supply `auto`;
its CLI's `--gpu-layers` argument currently accepts integers only.

Speculative dispatch restructuring was not adopted: inspection found no
allocation in the fusion predicates and no evidence of material overhead.
Metal fallback changes have source review only; no Metal host was available.

### Qualification limits

The original port qualifies text inference on CPU/CUDA, with hardware validation on one
SM86 GPU. HIP, Metal, Vulkan, multi-GPU operation, adapters, speculative decoding,
and vision were not qualified in that campaign. DFlash2 follow-up results are
below. The portable HIP fallback is not a claim of HIP
performance parity.

The `reuse-*` and `wide-*` experiments are not optimization authorities: source
timestamps preserved across machines caused an incremental build to reuse old
C++ objects; the subsequent mixed-layout build crashed during graph setup.
The affected sources and header dependents were explicitly rebuilt, and the
consistent snapshot is `candidate-bin`. Its tests and timings above supersede
those earlier candidates. New transfers use content comparison without
preserving source timestamps. The initial port and Prism references are intact.

## DFlash2 shared-transform follow-up

Patch base: `63d057134e794349c1a2641c0cc9eba9e8b9575d`. Tested on Dorei with
`Qwen3.8-27B-DFlash2-Q2_K.gguf` from the draft-model cache directory.
The missing pieces were transform descriptors for shared embeddings/output
heads and inverse transforms after DFlash token lookup, not target hidden-state
capture. Sharing now places auxiliary tensors using the existing alias/copy
rules and keys descriptors by the destination tensor identity.

Four server prompts (19, 25, 25, and 1,381 tokens), temperature zero, seed 1234,
up to 256 generated tokens, no prompt reuse; F16 KV, context 8,192, batch/ubatch
512, eight CPU threads, full GPU offload, `--spec-dflash-default`, draft maximum
15. The optimized fork DFlash2 selector was confirmed in the logs.

| Packing | Accepted/generated before | Accepted/generated fixed | Coding TG before/fixed | Explanation TG before/fixed |
| --- | ---: | ---: | ---: | ---: |
| PTQ1_0 | 0/1,261 | 376/1,066 | 24.23 / 67.78 | 23.99 / 49.54 |
| PQ2_0 | 0/1,261 | 379/1,065 | 43.91 / 115.32 | 43.65 / 87.32 |

Rates are tokens/second. These are short serial smoke comparisons, not a broad
performance benchmark or speedups against target-only decoding. In particular,
target-only PTQ ran the explanation at about 62 tok/s, faster than DFlash2.
All fixed responses were coherent; sampled PQ decoding at temperature 1 also
produced coherent responses with nonzero acceptance.

Regression controls using the preserved `simplify-bin` baseline:

- Target-only PTQ: all four output texts/token arrays identical; coding TG
  62.50 -> 62.09, explanation TG 62.16 -> 61.89, long-prompt PP
  965.97 -> 960.97 tok/s.
- Ordinary Qwen3.8-27B IQ3_XXS + the same DFlash2 drafter: all four output
  texts/token arrays and acceptance counts identical; coding TG
  121.07 -> 120.66, explanation TG 78.45 -> 78.27, long-prompt PP
  1010.78 -> 1010.01 tok/s. These single samples do not prove zero overhead.
- `test-draft-shared-tensors` passes CPU/GPU alias/copy and allocation-free
  sharing contracts, tied/independent heads, transform deduplication and metadata,
  and numerical inverse-embedding/forward-head checks including a token view.
- `test-llama-archs -a llama -s 1234` passes the Bonsai loader contracts and
  CPU/CUDA/Meta graph checks (Meta roundtrip is skipped).

Explicit full auto-fit reports `dflash requires ctx_other to be set`, fails its
draft context measurement, and restores pre-fit parameters on **both** baseline
and candidate. Inference then succeeds with the supplied placement. This is a
separate pre-existing limitation, not a passed auto-fit test; the lower-level
allocation-free sharing tests do pass.

Artifacts under the Dorei root above: `bonsai-dflash-repro.py`,
`dflash-{ptq,pq}-{repro,fixed}`, `dflash-plain-ptq-{simplify-bin,build/bin}`,
`dflash-qwen-regression-{simplify-bin,build/bin}`, `dflash-pq-fit-{sampled,debug,baseline}`,
`dflash-fix-shared-tests.log`, and `dflash-fix-arch-tests.log`. Each server run
records its exact command, log, and responses. Candidate binary: `build/bin`.
