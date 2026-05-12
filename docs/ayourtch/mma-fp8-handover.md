---
name: MMA FP8 tensor core integration handover
description: Detailed state of the in-progress tensor core MMA integration for FP8 attention decode in ds4, including the register layout bug being debugged
type: project
originSessionId: 771453ab-8837-4e69-9402-f80bb764a234
---
## Current Branch & Commits

Branch `ayourtch-fp8` on remote `ayourtch` (https://github.com/ayourtch-llm/ds4.git), 11 commits ahead of main.

Key commits (latest first):
- `2c90d47` Add MMA FP8 correctness test and arch guard
- `31d0734` Add rope contribution to MMA attention scores
- `7b53bb1` Complete MMA score computation with per-block scale application
- `0e99b8c` Defer block scale multiply outside FP8 inner loop
- `fffc8f6` Split FP8 KV dot product into nope/rope phases
- `9b2e7f4` Vectorize comp_kv_load4
- `2dd44ce` Replace FP8 dequant LUT with hardware intrinsic
- `60d0870` Restore q8 fp16 weight cache budget/fallback

## Hardware Setup

- GPU 0: RTX 4090 (sm_89, 24GB) — not used for this model (too small)
- GPU 1: RTX PRO 6000 Blackwell Max-Q (sm_120, 96GB) — primary target
- Model: `./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` (81GB)
- Always use `CUDA_VISIBLE_DEVICES=1` and `CUDA_HOME=/usr/local/cuda-12.9`

## Performance Baseline

Generation speed (t/s) measured with ds4-bench at various context sizes:

| Ctx | Main (f32 KV) | FP8 (current) | FP8 mem savings |
|-----|--------------|---------------|-----------------|
| 2K  | 39.0 | 35.4 (-9%) | -30% |
| 4K  | 38.3 | 35.0 (-9%) | -39% |
| 8K  | 37.5 | 34.0 (-9%) | -46% |
| 16K | 35.5 | 31.7 (-11%) | -50% |
| 32K | 34.2 | 29.1 (-15%) | -53% |
| 64K | 31.8 | 25.1 (-21%) | -54% |

The gap widens with context because decode is compute-bound (FP8→float conversion cost), not bandwidth-bound. Tensor core MMA should eliminate this gap.

## FP8 KV Cache Layout

Each compressed KV row is 640 bytes (`DS4_FP8_ROW_STRIDE`):
- Bytes 0-31: 8 float scales (7 used, one per 64-element nope block)
- Bytes 32-479: 448 FP8 e4m3 nope bytes (`DS4_FP8_NOPE_OFF`)
- Bytes 480-607: 64 fp16 rope values (`DS4_FP8_ROPE_OFF`)
- Bytes 608-639: padding

Scales are powers of 2: `exp2f(ceilf(log2f(amax / 448.0f)))`. Compatible with ue8m0 format.

Model dimensions: 64 heads, 1 KV head (MQA), 512 head_dim (448 nope + 64 rope).

## MMA Function: `comp_kv_mma_scores_16x8()` in ds4_cuda.cu

**Location:** Around line 2588, guarded by `#if __CUDA_ARCH__ >= 1200`.

**Purpose:** Compute 16 KV rows × 8 heads attention scores using tensor core FP8 MMA.

**Approach:**
1. Iterate over 7 scale blocks (64 dims each = 2 K-chunks of 32)
2. For each block: run 2 MMAs into fresh accumulator `p0..p3`, then `d += p * scale`
3. After nope: add rope contribution (scalar 64-dim dot per thread)
4. Store scores to row-major output array

**PTX instruction:** `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (sm_89 syntax works on sm_120; the `kind::f8f6f4` variant does NOT compile for sm_120, only sm_120a)

## THE BUG: A-Matrix Register Packing

**Status:** The MMA instruction executes correctly (all-ones test gives 32.0), but the A/B register packing scrambles data positions.

**Evidence from `/tmp/test_mma_simple.cu`:**
- Test 1 (all ones): PASS — all D elements = 32.0
- Test 2 (row index): FAIL — all rows show 240.0 instead of 32*row_index. 240 = 32 * avg(0..15) = 32 * 7.5

**Register layout (from CUTLASS, verified by research agent):**

For lane L: `tid = L & 3`, `gid = L >> 2`

A matrix (16×32, row-major e4m3):
```
a0 bytes: A[2*tid, gid], A[2*tid, gid+16], A[2*tid+1, gid], A[2*tid+1, gid+16]
a1 bytes: A[2*tid, gid+8], A[2*tid, gid+24], A[2*tid+1, gid+8], A[2*tid+1, gid+24]
a2 bytes: A[2*tid+8, gid], A[2*tid+8, gid+16], A[2*tid+9, gid], A[2*tid+9, gid+16]
a3 bytes: A[2*tid+8, gid+8], A[2*tid+8, gid+24], A[2*tid+9, gid+8], A[2*tid+9, gid+24]
```

B matrix (32×8, col-major e4m3):
```
b0 bytes: B[gid, tid], B[gid+8, tid], B[gid+16, tid], B[gid+24, tid]
b1 bytes: B[gid, tid+4], B[gid+8, tid+4], B[gid+16, tid+4], B[gid+24, tid+4]
```

D matrix (16×8, f32):
```
d0 = D[gid, tid*2], d1 = D[gid, tid*2+1], d2 = D[gid+8, tid*2], d3 = D[gid+8, tid*2+1]
```

**Key insight for KV attention mapping:**
- A = KV rows (M=16 rows, K=32 dims) → rows indexed by `2*tid` and `2*tid+1` (NOT gid!)
- B = Q heads (K=32 dims, N=8 heads) → heads indexed by `tid` and `tid+4` for columns
- D = scores (M=16 rows, N=8 heads) → rows indexed by `gid`, heads by `tid*2`

The MMA redistributes: A-rows are loaded via `tid`, but D-rows are output via `gid`. This is the expected behavior of matrix multiply — it's NOT a bug in the instruction, it's our packing that needs to match.

**RESOLVED: The CUTLASS layout documentation was WRONG for this hardware.**

Empirically verified layout on RTX PRO 6000 Blackwell (sm_120):

```
A matrix: row = gid (NOT 2*tid as CUTLASS says)
  a0, a2: top half (rows 0..7, row = gid)
  a1, a3: bottom half (rows 8..15, row = gid+8)
  tid and byte determine K-column (16 K-positions per thread)
  Each thread holds ALL its K-values for a SINGLE row
```

This means: thread with gid=G loads K-dimension data for row G (top) and row G+8 (bottom).
This is MUCH simpler for our use case — no cross-thread data distribution needed.

The D output layout is confirmed: d0=D[gid][tid*2], d1=D[gid][tid*2+1], d2=D[gid+8][tid*2], d3=D[gid+8][tid*2+1].

## MMA Integration Status (completed)

The register layout was fixed and the MMA kernel is fully integrated:
- `attention_decode_mma_online_kernel` in ds4_cuda.cu
- Phase A: Raw KV via shared memory (unchanged from original)
- Phase B: Comp KV with MMA scores + direct L1-cached FP8 output accumulation
- Gated by runtime SM check (sm_120+) and DS4_CUDA_NO_MMA env var

**Performance finding:** MMA is performance-neutral vs scalar path. The score computation is NOT the bottleneck — the **output accumulation** (FP8 dequant + weighted sum, 512 dims per row per head) dominates.

## nsys Profiling Results (16K context)

The REAL hot kernel is NOT the decode kernel we optimized:

| Kernel | Total Time | Calls | Avg/call |
|--------|-----------|-------|----------|
| attention_indexed_mixed_heads8_online | **2.07s** | 147 | 14.1ms |
| indexer_scores_wmma | **1.09s** | 147 | 7.4ms |
| attention_decode_mixed_heads8_online | 546ms | 154 | 3.5ms |
| attention_indexed_mixed | 358ms | 1344 | 0.3ms |

The indexed attention kernel is **4x more expensive** than the decode kernel. It's structurally identical (same comp_kv_load4 → shared memory → float4 dot → online softmax) but uses a topk index to select which compressed KV rows to attend to. The same MMA + L1 optimization applies directly.

## ncu Profiling: The Definitive Bottleneck

Profiled `attention_indexed_mma_online_kernel` at 16K context:
```
DRAM throughput:  0.17%   — NOT memory-bandwidth bound
L1 throughput:   89.26%   — THIS IS THE BOTTLENECK
SM throughput:   45.17%   — compute half-utilized
Occupancy:       65.70%   — decent
FMA instructions:  90.2B  — dominant op
FMUL instructions: 83.3B  — per-element scale multiply
FADD instructions: 10.2B  — relatively few
```

The kernel is **L1-cache bound**. The FP8 data fits in L1 but we issue too many L1 load requests. The 83B FMUL instructions are the per-element scale multiply during dequantization.

## Approaches Tried and Results

1. **MMA for scores only** → 0% speedup (score computation was never the bottleneck)
2. **Direct L1 output accumulation** (skip shared memory) → 0% speedup (8 warps issue redundant L1 loads)
3. **Hybrid: MMA scores + shared memory output** → 0% speedup (shared memory load is the bottleneck in all variants)
4. **Vectorized uint32 FP8 loads** → 0% speedup (L1 loads were already coalesced across warp)

All approaches are performance-neutral because they all go through `comp_kv_load4` for the output accumulation, which saturates L1 at 89%.

## The Real Optimization Target

The bottleneck is `comp_kv_load4`: FP8 byte load → HW intrinsic fp8→half→float → scale multiply → store to shared memory. This runs for every comp KV row × 512 dims.

**Promising approaches for future work:**
- **Half-precision shared memory**: store dequanted values as fp16 instead of float32 in shared memory. Halves shared memory bandwidth and storage. The output accumulation could use `__hfma2` for 2-wide half FMA.
- **MMA for O = softmax(S) × V**: restructure kernel so the weighted sum is also a matrix multiply. Would require batching: O[8 heads × 512 dims] = W[8 heads × N_rows] × V[N_rows × 512 dims]. But softmax weights are float, not FP8.
- **Skip dequantization entirely**: keep FP8 in shared memory, dequant only during the FMA accumulation. Avoids the store-as-float32 expansion.
- **Reduce scale multiply count**: the 83B FMUL is almost as many instructions as the 90B FMA. Pre-multiplying weight × scale could eliminate one multiply per element.

## Branch Status

Branch `ayourtch-fp8` has ~18 commits on top of main. Key files:
- `ds4_cuda.cu` — all CUDA kernels, MMA functions, and both attention decode variants
- `tests/test_mma_fp8.cu` — MMA correctness test (PASS, max abs error 2.92)
- `tests/test_mma_layout.cu` — empirical register layout probing
- Build: `CUDA_HOME=/usr/local/cuda-12.9 make -j$(nproc)`
- Test: `CUDA_VISIBLE_DEVICES=1 make test-mma`
- Bench: use `/tmp/ds4_ab_bench.py` or `ds4-bench` directly
- Disable MMA: `DS4_CUDA_NO_MMA=1`
- Enable FP8 histogram: `DS4_FP8_HISTOGRAM=1`

## Test Infrastructure

- `tests/test_mma_fp8.cu` — standalone correctness test, build with `make test-mma` (uses `-arch=sm_120`)
- `/tmp/test_mma_simple.cu` — quick debug test for register layout (not committed)
- Baseline binaries: `/tmp/ds4_baseline`, `/tmp/ds4bench_baseline`
- FP8 binaries: `/tmp/ds4_fp8`, `/tmp/ds4bench_fp8`
- A/B benchmark script: `/tmp/ds4_ab_bench.py` (uses ds4-bench CSV output)

## Key Files

- `ds4_cuda.cu` — all CUDA kernels, MMA function at ~line 2588
- `ds4.c` — model constants (line 90: DS4_N_HEAD=64, DS4_N_HEAD_KV=1, DS4_N_HEAD_DIM=512)
- `ds4_gpu.h` — GPU API declarations
- `Makefile` — build targets including `test-mma`
