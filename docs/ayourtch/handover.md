# ayourtch-may14 — CUDA prefill optimizations

This branch lands 6 CUDA prefill optimizations on top of `0cba357` /
`2a7a5f3`. Per-context prefill is +2-6% across the 2K-32K range, all
existing regression tests are at the baseline pass/fail bar, and one
new GPU unit test is added.

## What's in the branch

```
d52b41e cuda: f16 KV shared memory + __hfma2 dot product + 2× rows in attention
670b73a cuda: fuse inv_rope + f16 conversion, strided output projection GEMM
8c4932a cuda: f16 Q path — Q_B matmul outputs f16, fused norm+rope reads f16
5a4b31e cuda: overlap shared expert with routed MoE via async stream
12247fa cuda: fuse rms_norm + f16 conversion for HC matmul prefill paths
d5b4fce cuda: cache f16 activation conversions across consecutive matmuls
```

All commits build clean, pass the existing test suite at baseline, and
include explicit safety guards documented in each commit message.

## Test results

Run from `ds4_test`:

| Test | Pre-branch baseline | After branch | Verdict |
|------|---------------------|--------------|---------|
| `--logprob-vectors` | 14 failures | 14 failures | unchanged (model-vs-API drift, not introduced by these commits) |
| `--long-context` | 1 failure (Alice 50 vs 52) | 1 failure (same) | unchanged |
| `--tool-call-quality` (fast path) | OK | OK | unchanged |
| `--tool-call-quality` (exact path) | OK | OK | unchanged |
| `--server` | OK | OK | unchanged |
| `tests/cuda_long_context_smoke` | OK | OK (incl. new `rms_norm_matmul_f16_fused_matches_unfused` — bit-exact) | new test added |

Reproduce:

```
DS4_TEST_VECTOR_FILE=tests/test-vectors/official.vec ./ds4_test --logprob-vectors
./ds4_test --long-context
./ds4_test --tool-call-quality
./ds4_test --server
./tests/cuda_long_context_smoke
```

## Per-commit prefill throughput (t/s)

Measured with `./ds4-bench --prompt-file /tmp/bench_prompt.txt
--ctx-start 2048 --ctx-max 32768 --step-mul 2 --gen-tokens 32` on
NVIDIA GB10 (sm_121) with the same `ds4flash.gguf` model.

| ctx | baseline (`2a7a5f3`) | `d5b4fce` | `12247fa` | `5a4b31e` | `8c4932a` | `670b73a` | `d52b41e` (HEAD) | net Δ |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 395.73 | 397.95 | 408.22 | 407.67 | 392.81 | 395.35 | **405.36** | +2.4% |
| 4096 | 395.69 | 398.14 | 406.29 | 407.07 | 410.20 | 414.91 | **417.96** | +5.6% |
| 8192 | 388.37 | 390.52 | 398.46 | 399.75 | 402.40 | 406.80 | **410.68** | +5.7% |
| 16384 | 375.78 | 378.07 | 385.39 | 386.31 | 389.62 | 393.37 | **396.86** | +5.6% |
| 32768 | 357.35 | 359.26 | 365.64 | 366.59 | 369.36 | 373.21 | **375.99** | +5.2% |

The 2K column shows a transient regression at `8c4932a` and `670b73a`
because cuBLAS picks slower f16-out kernel selections at small
n_tokens. This is recovered by the time `d52b41e` lands. The
n_tokens >= 2048 gates on the f16-Q and f16-heads paths exist precisely
because at sub-2K sizes the cuBLAS f16-out kernels also introduce
top-token drift, not just slowdown.

Full 32-point sweep with `--gen-tokens 128 --step-incr 2048` is
available in `/tmp/cherrypick/final_full_sweep.csv` (same shape as
the existing `ayourtch.csv` format).

Decode (gen_tps) is unchanged across all commits; the optimizations
are prefill-only.

## What changed in each commit

### `d5b4fce` — f16 activation cache
Adds `g_xh_cache` (a pointer-equality keyed f16 buffer) and
`cuda_get_f16_activations()` so consecutive matmuls sharing the same
f32 source activation skip the f32→f16 conversion. Refactors two
existing matmul entry points (`cuda_matmul_q8_0_tensor_labeled`,
`ds4_gpu_matmul_f16_tensor`) to use the helper. Internal change; no
public API impact. Verified safe against pointer-equality staleness in
all current call sites — none mutate activations in place between cached
hits.

### `12247fa` — fuse rms_norm + f16 for HC matmul prefill paths
Adds `rms_norm_plain_f16_kernel` (and an unused weighted variant)
that reads f32, normalizes, and writes `__half` directly. New public
API `ds4_gpu_rms_norm_matmul_f16_tensor()` fuses the kernel launch with
`cublasGemmEx`. Wired into three HC pre-processing call sites:
attention HC pre, FFN HC pre, output-head HC. Eliminates ~268 MB of
memory traffic per call at 2K tokens.

Dropped the unrelated experimental `use_tile16_gate` MoE hunks that
referenced a symbol that does not exist on this branch — they were
gated by `DS4_CUDA_MOE_TILE16` and disabled by default anyway.

New unit test
`check_rms_norm_matmul_f16_fused_matches_unfused()` compares the new
fused path against the unfused
`ds4_gpu_rms_norm_plain_rows_tensor() + ds4_gpu_matmul_f16_tensor()`
sequence on synthetic mmap-backed weights. Currently passes bit-exact.
Run with `./tests/cuda_long_context_smoke`.

### `5a4b31e` — overlap shared expert with routed MoE via async stream
Adds a secondary CUDA stream + cuBLAS handle, plus public API
`ds4_gpu_begin_async()` / `ds4_gpu_end_async()` / `ds4_gpu_sync_async()`.
During batch prefill, the shared expert FFN runs on the async stream
while the routed MoE runs on the default stream; event-based
dependencies guard both ends.

`ds4_gpu_begin_async()` short-circuits to return 0 when `g_quality_mode`
is set. Reason: the Q8 fast path on cuBLAS respects the active stream,
but the native Q8 fallback kernels used in quality mode launch on
stream 0 unconditionally — running them under async produced a race
that read partial shared-expert outputs in the combine step and broke
tool-call generation. Quality mode falls back to the original
sequential path. Disable entirely with
`DS4_CUDA_NO_ASYNC_SHARED_EXPERT=1`.

### `8c4932a` — f16 Q path
Adds `ds4_gpu_matmul_q8_0_f16out_tensor` (cuBLAS GemmEx with
`CUDA_R_16F` output) and `head_rms_norm_rope_tail_f16in_kernel`
(reads f16 Q, normalizes/applies RoPE in f32, writes f32). Saves a
128 MB intermediate f32 buffer between Q_B and head norm.

Call site is gated on **both** `!g->quality` and `n_tokens >= 2048`.
- `!g->quality`: `cuda_q8_f16_cache_allowed()` returns 0 in quality
  mode, so the new f16-out matmul has no usable weight pointer and no
  native fallback in this commit. Without the guard the prefill batch
  would fail (no tokens generated → tool-call quality test asserts).
- `n_tokens >= 2048`: at smaller n_tokens cuBLAS picks slower kernel
  variants for f16 output that both regress throughput and shift
  top-token choices on borderline short prompts (observed as
  `short_code_completion` step 0 logprob delta of -5.48 vs the
  reference 0). With the gate, short prefills (e.g. tool-call's
  512-token system prompts) take the original f32 path.

Disable entirely with `DS4_CUDA_NO_Q_F16=1`.

### `670b73a` — fuse inv_rope + f16 + strided output projection GEMM
Adds `ds4_gpu_inv_rope_f16_tensor` which combines inverse-RoPE (was
an in-place f32 pass) with an f32→f16 conversion writing to
`batch_q_f16`. The output projection GEMM A then reads f16 with
strided batched access, eliminating the pack_group_heads_f16
rearrangement.

Same call-site gating as `8c4932a` (`!g->quality && n_tokens >= 2048`)
for the same reasons.

Adds a defensive guard inside `ds4_gpu_attention_output_q8_batch_tensor`:
if `heads_f16` is supplied but execution would otherwise fall through
to a path that consumes the f32 `heads` buffer, the function returns 0
instead. The f32 `heads` buffer has *not* been inverse-RoPE'd in this
mode (the inverse-RoPE is folded into the f16 fused kernel) — projecting
it would silently corrupt output. The guard ensures any env-var
override that knocks out the cuBLAS f16 path
(`DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A`,
`DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN` raised, OOM) propagates as a
clean failure rather than a wrong-math success.

Disable entirely with `DS4_CUDA_NO_HEADS_F16=1`.

### `d52b41e` — f16 KV shared memory + __hfma2 + 2× rows
Three coupled changes inside the existing static / indexed attention
kernels:
1. KV stored in `__half2` (was `float4`) in shared memory.
2. Q·K computed with `__hfma2` pairs instead of scalar f32.
3. Doubled rows per chunk (8→16 static, 2× ROWS_PER_STAGE indexed).

Applied unmodified — no fixes were needed.

The Q·K dot accumulator is now `__half2` rather than f32, with the
final two-lane reduction converting back to f32 for the warp reduce.
For typical RMS-normalized Q/K magnitudes this stays well within f16
range (max representable ~65504), but adversarial activations could in
principle round/saturate a half lane. Not observed on existing
fixtures.

Bisect escape hatches (no unified flag):
- `DS4_CUDA_NO_WINDOW_ATTENTION=1` disables static/window online path
- `DS4_CUDA_NO_INDEXED_HEADS8=1` disables indexed heads8 online path
- `DS4_CUDA_INDEXED_TWOPASS=1` avoids the indexed online kernel entirely

## Files touched

```
ds4.c
ds4_cuda.cu
ds4_gpu.h
tests/cuda_long_context_smoke.c     (new test added)
```

## Env vars added/used

| Variable | Effect |
|----------|--------|
| `DS4_CUDA_NO_ASYNC_SHARED_EXPERT=1` | disable `5a4b31e` async shared expert |
| `DS4_CUDA_NO_Q_F16=1` | disable `8c4932a` f16 Q path |
| `DS4_CUDA_NO_HEADS_F16=1` | disable `670b73a` f16 heads + inv_rope fusion |
| `DS4_CUDA_NO_WINDOW_ATTENTION=1` | disable `d52b41e` static/window online kernel |
| `DS4_CUDA_NO_INDEXED_HEADS8=1` | disable `d52b41e` indexed online kernel |
| `DS4_CUDA_INDEXED_TWOPASS=1` | disable `d52b41e` indexed online kernel (alternate path) |

## Open items

- The numerical effect of the f16 Q·K accumulator (`d52b41e`) on
  adversarial activations is not directly tested. Future work: add a
  GPU unit test that drives `attention_static_mixed_heads8_online_kernel`
  with Q/K spike magnitudes (>1 RMS) and confirms output stays within
  bounded error of a f32 reference.
- `cuda_get_f16_activations()` keys on `(src ptr, count)` only. In
  current call sites no activation tensor is mutated between cache
  hits, but a future caller could trip this. Either expand the cache
  key, or invalidate (`g_xh_cache_src = NULL`) whenever a producing
  kernel writes to a tensor that may be cached.
- `ds4_gpu_matmul_f16_tensor()` is not fully stream-consistent: its
  internal f32→f16 conversion follows `g_active_stream` but the cuBLAS
  GEMM uses the default `g_cublas` handle, and the native fallback
  ignores `g_active_stream`. This is benign with the current call
  sites but would block any future use of `ds4_gpu_begin_async()` /
  `ds4_gpu_end_async()` around it.

## Reproducing the perf numbers

```
make ds4_test ds4-server ds4-bench tests/cuda_long_context_smoke
./ds4-bench --prompt-file /tmp/bench_prompt.txt \
            --ctx-start 2048 --ctx-max 65536 \
            --step-incr 2048 --gen-tokens 128 \
            --csv out.csv
```
