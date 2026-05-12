# ds4 Optimization Handover — May 12, 2026

## Branch State

- **`ayourtch-may12`** on remote `ayourtch` (https://github.com/ayourtch-llm/ds4.git) — clean fork of `main` at `8809b90`, no changes yet. This is the branch for the matmul optimization work.
- **`ayourtch-fp8`** — FP8 KV cache + MMA tensor core work (complete, ~20 commits). Parked.
- **`main`** — synced to `origin/main` (antirez/ds4) at `8809b90`.

## Hardware

- GPU 0: RTX 4090 (sm_89, 24GB) — not used for this model
- GPU 1: **RTX PRO 6000 Blackwell Max-Q** (sm_120, 96GB) — primary target
- Model: `./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` (81GB)
- Always use `CUDA_VISIBLE_DEVICES=1` and `CUDA_HOME=/usr/local/cuda-12.9`
- Prompt file: `./speed-bench/promessi_sposi.txt` (1.3MB, on `main` branch; was `./bench/` on `ayourtch-fp8`)

## The Opportunity: matmul_q8_0_preq_kernel is 40% of GPU Time

### nsys Profile at 16K Context (from earlier profiling run, data in `/tmp/ds4_nsys_mma.nsys-rep`)

| Kernel | Time | % | Instances | Avg/call |
|--------|------|---|-----------|----------|
| **matmul_q8_0_preq_kernel** | **25.9s** | **40%** | 1,992 | 13.0ms |
| matmul_q8_0_preq_batch_warp8_kernel | 10.5s | 16% | 328 | 32.0ms |
| grouped_q8_0_a_preq_warp8_kernel | 10.2s | 16% | 3,080 | 3.3ms |
| moe_gate_up_mid_expert_tile8 | 5.2s | 8% | 344 | 15.2ms |
| moe_down_expert_tile16 | 3.9s | 6% | 344 | 11.5ms |
| attention_indexed_heads8_online | 2.1s | 3% | 147 | 14.1ms |
| indexer_scores_wmma | 1.1s | 1% | 147 | 7.4ms |
| attention_decode_heads8_online | 0.5s | 1% | 154 | 3.5ms |

**The top 3 matmul kernels account for 72% of total GPU time.** Attention is only ~5%.

### ncu Profile of matmul_q8_0_preq_kernel

```
dram__throughput:   0.13%   — NOT DRAM bound, everything hits L1
l1tex__throughput: 86.51%   — L1 CACHE IS THE BOTTLENECK
sm__throughput:    77.18%   — compute well-utilized but waiting on L1
sm__warps_active:  91.61%   — excellent occupancy
tensor_pipe:       0        — ZERO tensor core usage
ffma:              268M     — float FMA instructions
fmul:              268M     — float MUL (scale multiply, equals FMA count!)
```

**The kernel is L1-throughput bound at 86.5%, uses zero tensor cores, and has as many scale multiplies as FMAs.**

## The Root Cause: No Weight Reuse Across Tokens

### Current Kernel Structure (`ds4_cuda.cu` line 1821)

```cuda
__global__ static void matmul_q8_0_preq_kernel(
        float *out, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok,
        uint64_t blocks, int use_dp4a) {
    uint64_t row = blockIdx.x;   // one output row (weight row)
    uint64_t tok = blockIdx.y;   // one token
    // ...
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        // Load Q8 weight block (34 bytes: 2 byte scale + 32 int8)
        // Load pre-quantized input block (32 int8)
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    // warp reduction → one output value
}
```

Grid: `(out_dim, n_tok)` e.g., `(512, 2048)` for prefill.

**Problem:** Each block computes ONE output value (one weight row × one token). The grid's Y dimension iterates over tokens. Adjacent blocks in Y load THE SAME weight row independently. With 2048 tokens, the same weight data is loaded from L1 **2048 times**.

This is a textbook matmul tiling failure — no data reuse along the token (N) dimension.

### The Q8_0 Format

Each Q8 block is 34 bytes:
- 2 bytes: `__half` scale factor
- 32 bytes: 32 × int8 quantized values

Weight dimensions for typical layers:
- `in_dim`: varies (e.g., 2048, 7168, etc.)
- `out_dim`: varies (e.g., 512, 2048, etc.)
- `blocks = (in_dim + 31) / 32`: number of Q8 blocks per row

Input (`xq`): pre-quantized to int8 with per-block scales (`xscale`).

The inner product uses `dp4a` (4-wide int8 dot product instruction), accumulating into int32, then converting to float with scale multiplication.

### The `dot_i8_block` / `dot_i8x32_dp4a` Functions (line 1727)

```cuda
__device__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}
```

This does 8 `dp4a` instructions per 32-element block. Each `dp4a` does 4 int8 multiplies + 4 additions in one instruction. So 32 multiply-accumulates per block.

### Other Matmul Variants in the File

1. **`matmul_q8_0_preq_warp8_kernel`** (line 1857): single-token decode variant. 8 output rows per block (one per warp). No token tiling needed (n_tok=1).

2. **`matmul_q8_0_preq_batch_warp8_kernel`** (16% of time, line ~1857+): used for batched single-token. Similar structure.

3. **`grouped_q8_0_a_preq_warp8_kernel`** (16% of time): for grouped/expert operations.

## Optimization Plan

### Phase 1: Tile Across Tokens (Biggest Win)

Rewrite `matmul_q8_0_preq_kernel` to process multiple tokens per block:

```
Current:  grid(out_dim, n_tok) — one output per block
Proposed: grid(out_dim, n_tok/TILE_N) — TILE_N outputs per block
```

Approach:
1. Each block loads a tile of weight data into shared memory (e.g., 32 Q8 blocks = 1088 bytes)
2. Loop over TILE_N tokens, loading each token's input and computing dot products against the shared weight tile
3. Weight data loaded once, reused TILE_N times

This reduces L1 traffic by TILE_N×. With TILE_N=8 and 2048 tokens, weight loads drop from 2048× to 256×.

Shared memory budget: 256 threads × 4 bytes + weight tile. A 32-block weight tile is 34×32 = 1088 bytes. With TILE_N=8 tokens × 32 int8 values = 256 bytes per token tile. Total ~2KB per iteration — well within 48KB shared memory.

### Phase 2: Tensor Core INT8 MMA

Replace `dp4a` with INT8 tensor core MMA (`mma.sync.m16n8k32.s32.s8.s8.s32`). This computes 16×8 output tiles from 16×32 × 32×8 input tiles in one instruction.

The mapping:
- M = output rows (weight rows), tile 16 at a time
- N = tokens, tile 8 at a time  
- K = inner dimension, chunk 32 at a time (matches Q8 block size!)

This would give 16×8 = 128 outputs per MMA instruction vs 1 output per `dp4a` chain — massive throughput improvement AND it bypasses the L1 bottleneck.

**Note:** On Blackwell (sm_120), the FP8 MMA we verified uses the same register layout. INT8 MMA uses the same `m16n8k32` tile shape. The empirically-verified layout from our FP8 work applies: row = gid, column = gid for B, K packed consecutively.

### Phase 3: Reduce Scale Multiplies

The ncu shows FMUL count equals FMA count — half the compute is scale multiplication. Each output element does:
```
acc += __half2float(w_scale) * x_scale * (float)dot;
```

Two multiplies per block (w_scale and x_scale). Could pre-multiply `w_scale * x_scale` once per block, or fold into the tensor core block-scaling path.

## Key Files and Locations

- `ds4_cuda.cu` line 1821: `matmul_q8_0_preq_kernel` (the 40% target)
- `ds4_cuda.cu` line 1727: `dot_i8x32_dp4a` (inner product)
- `ds4_cuda.cu` line 1857: `matmul_q8_0_preq_warp8_kernel` (decode variant)
- `ds4_cuda.cu` line ~5220: dispatch logic for matmul
- `ds4_cuda.cu` line ~5224: grid launch `matmul_q8_0_preq_kernel<<<grid, 256>>>`

## Build & Test Commands

```bash
# Build
CUDA_HOME=/usr/local/cuda-12.9 make -j$(nproc)

# Quick test
CUDA_VISIBLE_DEVICES=1 ./ds4 --cuda -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf -p "What is 2+2? One word."

# Benchmark
CUDA_VISIBLE_DEVICES=1 ./ds4-bench --cuda -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf --prompt-file ./speed-bench/promessi_sposi.txt --ctx-start 4096 --ctx-max 4096 --gen-tokens 64

# nsys profiling (run as regular user, NOT root — root breaks kernel capture)
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-12.9/bin/nsys profile --stats=true -o /tmp/ds4_profile -f true ./ds4-bench ...

# ncu profiling (run as root for perf counter access)
sudo /usr/local/cuda-12.9/bin/ncu -k "regex:matmul_q8_0_preq_kernel" -s 5 -c 2 --replay-mode application --metrics <metrics> ./ds4-bench ...
```

## Profiling Notes

- **nsys must run as regular user** (not root) to capture CUDA kernel data on this system. Root sessions produce empty kernel reports.
- **ncu must run as root** for hardware performance counters. GPU profiling was enabled via `/etc/modprobe.d/nvidia-profiling.conf` with `options nvidia NVreg_RestrictProfilingToAdminUsers=0` (takes effect on reboot; currently works via root).
- Lock file `/tmp/ds4.lock` often needs `rm -f` between runs (especially if previous run was root-owned).

## Summary

The single biggest optimization opportunity in ds4 is tiling the Q8 matmul kernel across tokens. It's 40% of GPU time, L1-bound at 86.5%, with zero weight reuse across the token dimension. A properly tiled kernel with shared memory weight caching could reduce L1 traffic by 8-16× and potentially double overall inference throughput.
