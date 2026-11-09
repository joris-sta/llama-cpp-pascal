#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// Bonsai's 1024-wide transform, directly into the MMVQ activation layout.
void ggml_cuda_fwht_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                         const ggml_tensor * signs, void * dst, bool pascal_ptq1 = false);

// Fuses ternary embedding lookup, inverse rotation, and post-transform signs.
bool ggml_cuda_get_rows_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rows,
                             const ggml_tensor * signs, ggml_tensor * dst);
