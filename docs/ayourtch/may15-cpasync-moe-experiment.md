# ayourtch-may15 — cp.async + double-buffer for MoE gate_up: negative result

This is a **negative result writeup**. The experiment built a new
double-buffered cp.async variant of the routed MoE gate/up kernel,
benchmarked it on GB10 against the existing kernel, and confirmed
that the existing kernel's natural per-thread overlap is hard to beat
on this workload. The experimental kernel is kept in-tree behind the
`DS4_CUDA_MOE_GATE_CPASYNC` env var so future work can revisit it,
but it is **disabled by default and regresses prefill by ~8% when
enabled** on the DeepSeek V4 Flash IQ2_XXS model.

## What we measured

`./ds4-bench` with the standard prompt, ds4flash IQ2_XXS model, GB10
(sm_121), `make cuda-spark` build:

| ctx | baseline (kernel unchanged) | DS4_CUDA_MOE_GATE_CPASYNC=1 | Δ |
|----:|---:|---:|---:|
| 4096  | 417.45 | 380.95 | -8.7% |
| 65536 | 357.95 | 330.18 | -7.8% |

Decode throughput is unchanged in both modes — the new kernel only
sits on the prefill MoE gate/up path (`use_gate_row2048` branch with
`gate_row_span == 1024` and `xq_blocks == 16`).

## Why we tried this

`nsys` on a 65536-token prefill showed routed-MoE gate/up was 29.8%
of GPU time (kernel `moe_gate_up_mid_expert_tile8_rowspan_kernel<1024>`,
55.8 s of 187 s total) — the largest single hotspot. `ncu` on a single
launch of that kernel reported:

- **L1/TEX Cache Throughput: 89.99%** — saturated
- **L2 Cache Throughput: 39.37%** — large headroom
- **L1 bytes used per sector: 16.6 / 32 (52%)** — half the L1 traffic
  wasted on strided per-thread loads
- **Scheduler "no eligible warp" cycles: 54.13%** — over half the
  cycles, no warp ready to issue (memory latency stall)
- **L2 hit rate: 95.84%** — the data lives in L2; DRAM is not the
  bottleneck
- Achieved occupancy 32.95% (16 / 48 warps per SM), capped by
  register count (90/thread) and static SMEM (39.55 KB/block)

The data pointed at two complementary attacks:

1. **`cp.async.cg`** — load weights L2→SMEM via the async copy
   engine, bypassing L1 entirely. Eliminates the 52% sector waste
   (the cooperative load coalesces 32 lanes onto contiguous
   addresses) and removes L1 throughput as a binding constraint.
2. **Double-buffer the SMEM stage** — overlap the load of stage N+1
   with the compute on stage N. Addresses the "54% no-eligible-warp"
   stall directly by giving the scheduler real work to do while loads
   are in flight.

These are exactly the two mechanisms the user proposed in the design
discussion ("can some data come directly from memory and lower the
pressure on L1 cache" / "and if we do proper prefetch — maybe we can
hide the memory latency").

## What was built

New kernel
`moe_gate_up_mid_expert_tile8_rowspan_cpasync_kernel<ROW_SPAN>` in
`ds4_cuda.cu`, dispatched when `DS4_CUDA_MOE_GATE_CPASYNC=1` and
`xq_blocks == 16` and `gate_row_span == 1024`. Falls back to the
original kernel otherwise.

Inner pipeline (per CTA, with `STAGE_ROWS=30`):

```
prologue: cp.async.cg load gate[rr=0] → s_stage[0], wait, sync

for rr in 0..N_RR:
    # in flight: gate[rr] in s_stage[gate_buf]
    cp.async.cg load up[rr] → s_stage[up_buf]
    compute gate dots from s_stage[gate_buf]
    wait_all; sync
    # in flight: nothing new yet
    if rr+1 < N_RR:
        cp.async.cg load gate[rr+1] → s_stage[gate_buf]
    compute up dots from s_stage[up_buf]
    if rr+1 < N_RR: wait_all; sync
    reduce + write outputs
```

Each cooperative load issues 16-byte `cp.async.cg.shared.global`
chunks (the 1056 B per row is 66 chunks). Threads cooperatively
cover `STAGE_ROWS × 66 = 1980` chunks per matrix per rr.

### SMEM budget on GB10

`cudaDevAttrMaxSharedMemoryPerBlockOptin = 101,376` bytes (99 KB) on
sm_121. Achieving full STAGE_ROWS=32 double-buffer required cutting
~7 KB off the existing caches:

- LUTs (iq2_xxs grid + signs) stay in SMEM: 2,176 B — needed because
  per-thread divergent indices into constant memory would serialize
  the warp.
- `cuda_block_q8_K_compact` (new): drop the `bsums[16]` field, keep
  `f32 d + int8 qs[256]` (260 B vs full 292 B). Saves 4 KB on the
  activation SMEM cache. The dot helper never reads bsums anyway.
- Even with that, full STAGE_ROWS=32 doesn't quite fit: static
  (compact sxq 33,280 + LUTs 2,176 = 35,456) + dynamic 2×32×1056 =
  67,584 lands at 103 KB, ~2 KB over budget.
- Compromise: **STAGE_ROWS=30**, which fits (35,456 + 63,360 = 98,816
  bytes). Costs 2 of 32 row_lanes idle per iter (~6.25% compute
  waste) and an uneven rr count (35 iterations: 34 full + 1 partial,
  since 1024/30 = 34.13).

The kernel uses dynamic SMEM via `cudaFuncSetAttribute(
cudaFuncAttributeMaxDynamicSharedMemorySize, 63360)` on first launch.

### Build-time requirement

`cp.async` requires `.target sm_80` or higher. The original
`make cuda-spark` target had `CUDA_ARCH=` (empty), which made nvcc
13.0 default to `sm_75` — fails to compile the new kernel. The
Makefile fix in the companion commit sets `CUDA_ARCH=sm_121`, which
both unblocks cp.async and produces native GB10 code instead of
Turing-baseline PTX that JITs at load time.

## Why it didn't work

Walking through what each prototype told us:

| Variant | 4K prefill | vs baseline | what it confirms |
|---|---:|---:|---|
| original | 418.72 | — | baseline |
| single-buffer cp.async, STAGE_ROWS=32 | 384.84 | -7.6% | cooperative load + sync wait is strictly worse than per-thread async-via-scheduler |
| double-buffer cp.async, STAGE_ROWS=28 | 355.79 | -15.0% | overlap recovers some, but STAGE_ROWS=28 idle penalty (~12.5%) dominates |
| double-buffer cp.async, STAGE_ROWS=30 + compact | 380.95 | -9.0% | each row of STAGE_ROWS ≈ 3% perf; extrapolated STAGE_ROWS=32 would be ~-3 to -5%, still a loss |
| `__ldcg` only (preserves occupancy, no SMEM staging) | 341.18 | **-18.5%** | bypassing L1 with original SMEM footprint is even worse — every load takes L2 latency |

### Mechanism, take 2: the real bottleneck is *occupancy*, not bandwidth

A side-by-side ncu profile of the cp.async double-buffer kernel vs
the original (both at 4K prefill, one launch each) reveals what
actually happens. Key numbers:

| Metric | Original | cp.async DB |
|---|---:|---:|
| Duration | 44.53 ms | 60.00 ms (+35%) |
| L1/TEX Throughput | 89.99% | 73.00% |
| L1 Hit Rate | 74.15% | 19.64% |
| L2 Throughput | 39.37% | 6.58% |
| L2 Hit Rate | 95.84% | 84.30% |
| Mem Pipes Busy | 88.40% | 72.71% |
| Compute (SM) Throughput | 88.99% | 72.65% |
| **Achieved Occupancy** | **32.95%** | **16.46%** |
| Active warps/scheduler | 4.09 | 2.00 |
| "No eligible warp" cycles | 54.13% | 61.04% |
| Block Limit (SMEM) | 2/SM | 1/SM |

**The cp.async kernel ran correctly** — L1 throughput dropped (the
bypass works), no barrier stall (`L2 active / SM active = 63%`
means SMs are *not* sitting in `wait_all`). Double-buffer smoothed
the latency.

**But two stages cost ~64 KB dynamic SMEM, which forces 1 block/SM
instead of 2.** Once occupancy halves, every downstream metric
collapses with it:

1. **Half the warps per scheduler** (2.00 vs 4.09) → less ILP for
   the scheduler to chew on while individual warps wait.
2. **Half the concurrent CTAs across the chip** (48 vs 96) →
   roughly half the chance that two same-expert CTAs (placed
   adjacent by `sorted_pairs`) overlap in time. **That's what
   collapses L2 hit rate from 96% to 84%.**

   The cross-CTA L2 reuse the original kernel "got for free" from
   high occupancy is a hidden lever the metric-level analysis
   missed.
3. **More idle cycles** (61% no-eligible vs 54%) → consequence of
   1+2.

So double-buffer works as advertised; cooperative cp.async works as
advertised. The cost was just in a place we weren't measuring:
*occupancy preservation*.

### Why `__ldcg` alone was even worse

The `__ldcg` variant kept the original SMEM footprint (preserving 2
blocks/SM) and only changed weight reads to bypass L1. Result:
**-18.5%**, the worst of all variants. Reason: the original
kernel's **L1 hit rate of 74% isn't waste, it's a fast path** at
~30 cycle latency. `__ldcg` forces all reads to L2 at ~200 cycle
latency. Average per-load latency triples. The kernel is bound on
*scheduler-can't-find-eligible-warps*; slowing every load makes
that strictly worse, even with occupancy preserved.

### The real lesson

The original kernel sits at a **multi-axis sweet spot**:
- L1 cache catches inter-thread sector reuse → 74% hit rate at ~30
  cycle latency
- Occupancy at 2 blocks/SM → 96 concurrent CTAs → cross-CTA L2
  reuse → 96% L2 hit rate
- Warp scheduler hides L2 misses across 4 active warps/scheduler

Any change that improves one axis at the cost of another loses
because all three are simultaneously near-binding. cp.async fixed
L1 sector waste but broke occupancy. `__ldcg` preserved occupancy
but broke L1's fast path. There is no obvious axis to improve in
isolation.

## Lessons / what to try next

The MoE wall is still there. Memory-access reshuffling is not the
lever on this kernel. The three-way sweet spot (L1 hit rate ×
occupancy × warp-scheduler overlap) is too tightly coupled to
improve one axis without losing on the others.

Directions still worth trying, biased toward changes that
**reduce total work** rather than redistribute existing work:

1. **Co-encode gate + up weights into one packed stream.** Each row
   of gate and up is read together every time; if they were jointly
   quantized into a single iq2-style format whose 2-bit code indexes
   into a shared LUT of (gate, up) value pairs, weight bandwidth
   halves. Requires a new GGUF format and re-quantizing the model,
   but the bytes-per-output reduction is real and orthogonal to the
   cache effects this experiment hit a wall on.
2. **FP8 mma.sync at the inner GEMM.** Dequantize iq2_xxs → FP8 in
   SMEM (small tile), issue `mma.sync.aligned.m16n8k32...f8...`
   instructions. Each mma replaces many `__dp4a` issues per thread;
   could relieve the compute side enough that memory becomes the
   sole bottleneck (currently both are at 89% co-saturation in the
   original kernel). Doesn't change byte traffic, but changes
   instruction count and warp-issue cost.
3. **Different kernel entirely.** The indexer pipeline
   (`indexer_scores_wmma128_kernel` + `attention_indexed_mixed_*`)
   was ~22% of long-ctx prefill time with high per-call variance
   — fresh territory with no analysis yet.
4. **Custom persistent CTAs** that hold expert weights resident in
   L2/SMEM across multiple tile batches, amortizing the per-CTA
   weight read. Direct attack on the **cross-CTA L2 reuse** that
   this experiment showed is critical (and that any naive
   occupancy-reducing change destroys).

What is *not* worth retrying (this experiment establishes it):
- Bypassing L1 with `__ldcg` (loses fast-path latency, worse net)
- Cooperative cp.async with smaller activation cache (would just
  shift the SMEM tax onto activation reuse, same occupancy hit)
- Restructuring warp layout for coalesced SMEM loads — only useful
  if the SMEM staging path turned out to win, which it didn't

## Files touched (this experiment)

- `ds4_cuda.cu`:
  - new `cuda_block_q8_K_compact` struct
  - new `dev_dot_iq2_xxs_q8_K_compact_block8_deq_lut` helper
  - new `dev_dot_iq2_xxs_q8_K_block8_deq_lut_ldcg` helper
  - new `cp_async_16_cg / cp_async_commit_group / cp_async_wait_all` PTX wrappers
  - new `moe_gate_up_mid_expert_tile8_rowspan_cpasync_kernel<ROW_SPAN>` (gated `DS4_CUDA_MOE_GATE_CPASYNC`)
  - new `moe_gate_up_mid_expert_tile8_rowspan_ldcg_kernel<ROW_SPAN>` (gated `DS4_CUDA_MOE_GATE_LDCG`)
  - env-gated dispatch in `routed_moe_gate_up`
- `Makefile`: `cuda-spark` target now passes `CUDA_ARCH=sm_121`
  (separate commit — required to build cp.async at all on this
  project's canonical build path)

## Reproducing

```
make cuda-spark
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 4096 --gen-tokens 16 --csv /tmp/off_4k.csv

DS4_CUDA_MOE_GATE_CPASYNC=1 \
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 4096 --gen-tokens 16 --csv /tmp/on_4k.csv
```

`ncu` profiling (note: **must** use `--replay-mode application` on
GB10 or it crashes the host):

```
sudo /usr/local/cuda/bin/ncu \
  --replay-mode application \
  --target-processes all \
  --kernel-name regex:moe_gate_up_mid_expert_tile8_rowspan_kernel \
  --launch-count 1 --set basic \
  -f -o /tmp/ncu-report \
  ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 4096 --ctx-max 4096 --gen-tokens 1 --csv /tmp/ncu.csv
```

Raw nsys/ncu reports for the analysis are archived under `profiles/`
on this branch.
