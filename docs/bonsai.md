# Bonsai 2 ternary GGUF

Bonsai 2 uses both a ternary weight codec and a metadata-defined activation
transform. Loading the codec alone is not sufficient: the stored weights have
blockwise Hadamard rotations folded into them.

This integration adds CPU/CUDA inference for Prism's `PTQ1_0` packing and maps
the public `PQ2_0` packing onto the existing group-128 ternary implementation.
It does not change the TurboQuant KV-cache codecs or their type IDs.

| File packing | GGUF type ID | Group size | Bytes/group |
| --- | ---: | ---: | ---: |
| PQ2_0 | 142 | 128 | 34 |
| PTQ1_0 | 143 | 128 | 28 |

The GGUF reader/writer translates these public IDs to/from the fork's compact
runtime IDs. Legacy group-128 Bonsai GGUF detection remains supported.
The two packings can represent the same ternary weights without additional
quantization loss; different kernel arithmetic can still change logits slightly.

## Running

Download a packing from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
For an initial text-only correctness/performance baseline:

```sh
./build/bin/llama-server \
  -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -c 8192 -fa on -ctk f16 -ctv f16 \
  --jinja
```

PQ2_0 trades a larger file for simpler decoding. Benchmark both on the target
hardware rather than assuming the smaller packing is faster. This is a dense
model; it does not need MoE expert-cache flags.

### DFlash2

The Qwen3.8-27B DFlash2 Q2_K drafter has been tested with both Bonsai packings.
Add `-md /path/to/Qwen3.8-27B-DFlash2-Q2_K.gguf --spec-dflash-default`
to the command above. Shared embedding/output tensors retain their Hadamard
transforms, and draft token lookups restore the original embedding basis.
This uses the fork's optimized DFlash2 path.

Speedup depends on the prompt and packing; speculation can still be slower than
target-only decoding. See the validation report for measured examples. The
existing DFlash auto-fit probe can warn that `ctx_other` is missing and fall back
to the supplied placement; successful inference after that warning does not
mean auto-fit proved the placement viable.

## Implementation and provenance

The port is based on [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
with reference revision `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`:

- `e0c828b90`: Hadamard metadata, graph transforms, and CPU/CUDA execution.
- `01fd9521c`: PTQ1_0 codec and CPU/CUDA decode kernels.
- `9294043b2`, `7f292be19`: packed CUDA dequantization and quantized prefill.

The model's sign vectors, transform version, tensor names, and head permutation
are validated during loading. A graph coverage check rejects folded weights
whose consumers bypass the activation transform. Reused transforms are local
to a single graph and keyed by the input, rotation, signs, and head permutation;
they are not part of the persistent prompt/KV cache.

Memory-fit probes build the same transform descriptors and graph as real loads,
using allocation-free weight buffers. The embedding inverse is placed with the
first model layer, independently of metadata ordering.

CUDA uses packed MMVQ for decode and MMQ for prefill on supported GPUs. The
portable CPU path is also available. HIP shares fallback code but has not been
hardware-validated here. This port does not add Prism's Metal or Vulkan ternary
kernels; those acceleration paths are not qualified here.

The port shares identical activation transforms within a graph and permits
wider PTQ MMQ tiles to reuse unpacked weights across more prompt tokens.
PQ2 reuses the existing group-64 signed-symbol decoder with a group-128 block
stride, avoiding both duplicate unpacking code and a rounded activation-sum
correction.

For supported CUDA layouts, single-consumer 1024-wide signed transforms write
MMVQ's Q8 activations directly. GPU-resident ternary embeddings fuse row lookup,
inverse rotation, and signs. Both use a shared 128-thread transform helper;
shared activations and unsupported layouts keep the ordinary paths. This does
not change embedding placement or the model's quantization scheme.

Tests cover production codec round trips, canonical GGUF IDs, ternary packing
equivalence, dot products, backend matmuls, signed Hadamard transforms, malformed
metadata, and dry-fit versus real-load graph/memory parity.
Hardware benchmark artifacts and exact commands are recorded separately from
these format contracts in [the validation report](bonsai-validation.md).

For reproducible server smoke checks, use `scripts/bench-bonsai.py`. Add
`--perf-corpus /path/to/wiki.test.raw` for three independent 512-token and
2,048-token prompts with 256 generated tokens each. Performance mode disables
prompt reuse and probability reporting, and verifies that every prompt was
evaluated in full. Run reference and candidate serially on an idle GPU.

```sh
python3 scripts/bench-bonsai.py \
  --binary ./build/bin/llama-server \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --out /tmp/bonsai-smoke
```
