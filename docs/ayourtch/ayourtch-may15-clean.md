# ayourtch-may15-clean — clean per-commit port log

A linear, clean re-port of the `ayourtch-may15` work on top of the
latest `main` (antirez/ds4), one logical change per commit, fixes
folded into the corresponding feature commit, every commit benchmarked
and reviewed.

Hardware: DGX Spark / GB10 (sm_121, 48 SMs, 128 KB SMEM/SM, 273 GB/s LPDDR5X).
Model: `ds4flash.gguf` (IQ2_XXS routed + Q2_K down + Q8 attention/shared).

## Bench protocol

Per-commit "quick" bench (≈5 min):

```
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 32768 --step-incr 4096 --gen-tokens 32 \
  --csv speed-bench/clean-<tag>.csv
```

`make test` after every commit; expected baseline is exactly 1 failure
(Alice 50/52 long-context, `tests/ds4_test.c:354`).

Codex (pty-3) review after every commit before moving on.

## Source mapping (commit → upstream `ayourtch-may15`)

| Clean commit | Upstream `ayourtch-may15` source | Fix(es) folded |
|---|---|---|
| Makefile sm_121 | `396da4b` (split from `30ae0f2`) | — |
| Combined gate+up MoE | `d924746` + `02fbe24` (code half of `30ae0f2`) | — |
| f16 activation cache | `d39808a` | `92cbe34` |
| rms_norm + f16 fuse | `21e2a22` | (also touched by `92cbe34`) |
| Async shared expert | `49f54d6` | `4a95a80` + `4c89ed5` |
| f16 Q path | `4bc82a2` | `96918d7` |
| inv_rope + f16 + strided output proj | `78c8ff0` | — |
| f16 KV + 2× rows attention | `0c231d2` | `bdafcca` |
| may14 commit-set doc | `8a852c5` | — |

**Skipped per "performance improvements only" rule:**
- `6701e61` cp.async + double-buffer MoE (off-by-default, regresses)
- `8563c6d` __ldcg-only MoE variant (off-by-default, regresses)
- `3939b89` + `5c1b00e` f16-Q indexer add+revert (net no-op)
- `4e343fa` indexer investigation findings (documents a dead end)

## Commit log (filled in as we go)

### Commit 0 — main baseline

Branch point: `d0357ec` ("Make KV cache hit decay the default").

Build: `make cuda-spark` (Makefile `CUDA_ARCH=` empty on main → nvcc 13
defaults to sm_75 PTX, JITs at load to sm_121).

Baseline numbers (`speed-bench/clean-00-baseline.csv`):

| ctx | prefill_tps | gen_tps |
|---|---|---|
| 4096 | 400.03 | 13.86 |
| 8192 | 393.02 | 13.63 |
| 12288 | 382.35 | 13.50 |
| 16384 | 375.13 | 13.44 |
| 20480 | 368.51 | 13.31 |
| 24576 | 361.47 | 13.17 |
| 28672 | 354.47 | 13.02 |
| 32768 | 348.62 | 12.56 |

### Commit 1 — Makefile: cuda-spark targets sm_121 (`3f94c48`)

Split out of upstream `30ae0f2` (the combined-only single commit). Source: `396da4b`.

Build numbers (`speed-bench/clean-01-sm121.csv`):

| ctx | prefill_tps | Δ vs baseline | gen_tps |
|---|---|---|---|
| 4096 | 389.11 | **−2.7%** | 13.73 |
| 8192 | 372.13 | −5.3% | 13.52 |
| 16384 | 357.15 | −4.8% | 13.35 |
| 32768 | 333.64 | −4.3% | 12.45 |

**Surprise**: on top of *unmodified* main (nvcc 13 toolchain on this
box), native sm_121 codegen is **slower** than sm_75 PTX JIT'd at
load. Confirmed with a second run at 4K (385.30 → ~1% noise band).
The handover claimed +1.2% from this commit on top of the may14 tree;
on top of plain `main` it costs perf.

Hypothesis: nvcc 13's static sm_121 codegen makes worse instruction
selection / register allocation choices than the JIT does for code
that has no `__CUDA_ARCH__ >= 800` paths. The Makefile change still
exists to *unblock* sm_80+ features (cp.async etc.). Whether it pays
for itself net of downstream perf commits is an open question until
the rest of the work is layered on top — re-evaluated at end.

No codex review (single-line Makefile change).

### Commit 2 — Combined gate+up MoE dot helper (`a363493`)

Source: code half of upstream `30ae0f2` (Makefile half above).
Squashes upstream `d924746` (env-gated helper) + `02fbe24` (default-on
flip) into one commit. `ds4_cuda.cu` only — adds
`dev_dot_iq2_xxs_q8_K_block8_gate_up_deq_lut` and
`moe_gate_up_mid_expert_tile8_rowspan_combined_kernel`, default-on
for the gate_row_span=1024 prefill path; opt-out via
`DS4_CUDA_MOE_NO_GATE_COMBINED=1`.

Mechanism: original kernel calls the dot helper twice per `b` iteration
(once for gate, once for up), both reading the same 8 token
activations. New helper takes both gate and up iq2_xxs blocks, loads
each activation int32 once, fans it to two parallel `__dp4a` chains.
Same SMEM (39.5 KB), same warp layout, same occupancy (2 blocks/SM).

Build numbers (`speed-bench/clean-02-combined.csv`):

| ctx | prefill_tps | Δ vs sm_121 | Δ vs absolute baseline | gen_tps |
|---|---|---|---|---|
| 4096 | 415.39 | **+6.75%** | +3.84% | 13.69 |
| 8192 | 397.71 | +6.87% | +1.19% | 13.49 |
| 16384 | 381.57 | +6.84% | +1.72% | 13.31 |
| 32768 | 355.31 | +6.49% | +1.92% | 12.44 |

Win versus sm_121 sibling baseline matches handover (+5.4-6.8%).
Versus absolute (sm_75-JIT) main baseline the win is smaller, because
the sm_121 Makefile change costs ~3-5% on code with no `sm_80+`
paths. Net result is still positive at every ctx.

`make test`: 1 failure (expected Alice 50/52). All other suites pass.

Codex review: dispatched to pty-3 — folded into "Codex notes" below.

### Commit 3 — f16 activation cache (`8b7a95c`)

Source: `d39808a`. Pure cherry-pick (no fix-fold).

Adds a single-slot f16 activation cache (`g_xh_cache`) so consecutive
matmuls that consume the same f32 activation buffer don't re-convert.
`cuda_get_f16_activations` returns the cached `__half*` if the source
pointer matches, otherwise (re)converts.

`speed-bench/clean-03-f16-act-cache.csv`:

| ctx | prefill_tps | Δ vs prev | gen_tps |
|---|---|---|---|
| 4096 | 419.34 | +0.95% | 13.64 |
| 16384 | 383.86 | +0.60% | 13.28 |
| 32768 | 356.55 | +0.35% | 12.39 |

Modest single-digit win at low ctx; flattens with ctx. The cache
benefit is bounded by how many *consecutive* f32→f16 conversions miss.

### Commit 4 — fused rms_norm + f16 + cache invalidation fix (`189e4d2`)

Sources: `21e2a22` + folded fix `92cbe34`.

`rms_norm_plain_f16_kernel` reads f32, normalizes, writes `__half`
directly — eliminates the intermediate f32 buffer between rms_norm
and the f16 cuBLAS matmul. `ds4_gpu_rms_norm_matmul_f16_tensor`
fuses kernel launch with `GemmEx`. Applied to 3 batch-prefill sites:
attention HC pre, FFN HC pre, output-head HC projection. HC state at
2048 tokens is 134 MB → saves ~268 MB traffic per call.

Folded fix `92cbe34`: pointer-based cache comparison in
`cuda_get_f16_activations` returned stale f16 data when rms_norm
overwrote the same activation buffer (batch_flat_hc reused across
attn and FFN norm paths). Cache key now includes a version counter
bumped on every rms_norm.

Added numerical regression `check_rms_norm_matmul_f16_fused_matches_unfused`
in `tests/cuda_long_context_smoke.c` (bit-exact: max_abs=0, max_rel=0).

`speed-bench/clean-04-rms-f16-fuse.csv`:

| ctx | prefill_tps | Δ vs prev | gen_tps |
|---|---|---|---|
| 4096 | 427.27 | +1.89% | 13.64 |
| 16384 | 391.08 | +1.88% | 13.24 |
| 32768 | 362.88 | +1.78% | 12.38 |

Clean ~2% across all contexts. Bandwidth saved scales with ctx but
prefill-side benefit caps quickly.

### Commit 5 — async shared expert overlap (`4c45dbc`)

Sources: `49f54d6` + folded fixes `4a95a80` + `4c89ed5`.

Secondary CUDA stream + cuBLAS handle for the shared expert FFN.
During batch prefill, shared expert gate/up/swiglu/down matmuls
launch on the async stream before routed MoE starts on default.
Event-based dependency ensures the async stream waits on the FFN
norm, and the default stream waits on the shared expert before
combining. ~4 ms/layer shared expert partially overlaps with
~77 ms/layer routed MoE tail waves. Disable: `DS4_CUDA_NO_ASYNC_SHARED_EXPERT=1`.

Folded fix `4a95a80`: cross-stream races on `g_xh_cache` and the
shared expert workspace. Tightened cudaEventRecord/cudaStreamWaitEvent
around the combine step.

Folded fix `4c89ed5`: pre-allocate async stream scratch buffers at
graph init so the first prefill chunk doesn't stall on lazy cudaMalloc.

`speed-bench/clean-05-async-shared.csv`:

| ctx | prefill_tps | Δ vs prev | gen_tps |
|---|---|---|---|
| 4096 | 425.26 | −0.47% | 13.68 |
| 16384 | 390.17 | −0.23% | 13.30 |
| 32768 | 362.74 | −0.04% | 12.42 |

**Within noise band.** Note `f16 activation cache: 2352 hits, 5089 misses`
vs 3729/7152 in the previous commit — the async stream invalidates
the cache more often (the cache is per-stream-blind). The expected
overlap gain (~few percent) is being eaten by the cache invalidation
churn at this prompt size; longer prompts may recover it. Kept in
the port because the fixes are correctness-relevant and decode-time
might benefit (not measured here).

### Commit 6 — f16 Q path + quality-mode fallback fix (`c92545a`)

Sources: `4bc82a2` + folded fix `96918d7`.

`ds4_gpu_matmul_q8_0_f16out_tensor` uses cuBLAS GemmEx with f16
output (halves Q_B matmul write bandwidth).
`head_rms_norm_rope_tail_f16in_kernel` reads f16 Q, norms+ropes in f32,
writes f32 — eliminates the 128 MB intermediate f32 between Q_B and
head norm. Disable: `DS4_CUDA_NO_Q_F16=1`.

Folded fix `96918d7`: when `cuda_q8_f16_cache_allowed` returns 0
(quality mode or cache miss) the f16-out matmul has no native
fallback — fix gates the f16 Q path on `!g->quality` + cache
availability, otherwise falls back to f32. Without the fix, the
tool-call-quality test fails with the "drunk model" symptom.

`speed-bench/clean-06-f16-q-path.csv`:

| ctx | prefill_tps | Δ vs prev | gen_tps |
|---|---|---|---|
| 4096 | 421.82 | −0.81% | 13.73 |
| 16384 | 393.72 | +0.91% | 13.32 |
| 32768 | 365.41 | +0.74% | 12.46 |

Slight regression at low ctx (cuBLAS f16-out kernel selection is
suboptimal there per upstream author notes), positive at higher ctx
where the Q_B GEMM dominates.

### Commit 7 — inv_rope + f16 + strided output projection GEMM (`76ab4d0`)

Source: `78c8ff0`. Pure cherry-pick.

Fuses inv_rope with the f16 conversion that follows it, and switches
the output projection GEMM to a strided cuBLAS call.

`speed-bench/clean-07-invrope-strided.csv`:

| ctx | prefill_tps | Δ vs prev | gen_tps |
|---|---|---|---|
| 4096 | 424.82 | +0.71% | 13.69 |
| 16384 | 396.16 | +0.62% | 13.31 |
| 32768 | 367.68 | +0.62% | 12.41 |

Steady ~0.6-0.7% across the sweep — consistent fuse-then-strided win.

### Commit 8 — f16 KV + __hfma2 + 2× rows attention (`78ddc27`) — BIG WIN

Sources: `0c231d2` + folded fix `bdafcca`.

Three interdependent attention optimizations on the rewritten static
and indexed attention kernels:

1. KV in `__half2` (was float4) in shared memory — halves SMEM bandwidth.
2. `__hfma2` for Q·K dot — fused f16 multiply-adds instead of scalar f32.
3. 2× rows per chunk: same SMEM holds 16 (was 8) static rows / 2×
   ROWS_PER_STAGE for indexed — halves sync points.

QK accumulator is `__half2`; partial lane sums round in f16, final
per-token max + softmax back in f32. Drift bounded by attention
scaling; logprob-vectors + tool-call-quality pass. Long-context
Alice mismatch unchanged.

Escape hatches: `DS4_CUDA_NO_WINDOW_ATTENTION=1`,
`DS4_CUDA_NO_INDEXED_HEADS8=1`, `DS4_CUDA_INDEXED_TWOPASS=1`.

Folded fix `bdafcca`: f16 heads path falls back to f32 when the
per-layer f16 weight cache is unavailable (quality mode). Without
this, the 2×-rows attention kernel would dispatch with a NULL f16
weight pointer.

`speed-bench/clean-08-f16-kv-attn.csv`:

| ctx | prefill_tps | Δ vs prev | Δ vs absolute baseline | gen_tps |
|---|---|---|---|---|
| 4096 | 441.80 | +4.00% | +10.4% | 13.68 |
| 8192 | 439.88 | +6.04% | +11.9% | 13.47 |
| 16384 | 420.50 | +6.16% | +12.1% | 13.31 |
| 32768 | 388.26 | +5.60% | +11.4% | 12.41 |

**The biggest single-commit win in the port.** Attention is the
right place to optimize at long contexts — halving SMEM bandwidth +
halving sync points compounds with ctx.

`make test`: full clean rebuild required (handover trap — stale
`ds4_test.o` showed 8 spurious failures, dropping to 1 expected
Alice failure after `make clean && make cuda-spark && make ds4_test`).

### Commit 9 — may14 commit-set doc (`ceccc1b`)

Source: `8a852c5`. Pure cherry-pick of the original handover-style
doc describing the may14 work in `docs/ayourtch/handover.md`.

## Summary table (Δ from absolute baseline)

| ctx | baseline | clean-08 | gain |
|---|---|---|---|
| 4096 | 400.03 | 441.80 | **+10.4%** |
| 8192 | 393.02 | 439.88 | **+11.9%** |
| 16384 | 375.13 | 420.50 | **+12.1%** |
| 32768 | 348.62 | 388.26 | **+11.4%** |

(gen_tps essentially flat — these are prefill optimizations.)

Net win: **+10-12% prefill across all contexts** vs `main` on this
hardware, in 9 clean commits with all upstream fixes folded into
their introducing commits.

## Full sweep at HEAD (`speed-bench/clean-final-full-sweep.csv`)

Sweep parameters: `--ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128`.

| ctx | prefill_tps | gen_tps | upstream ayourtch-may15 prefill_tps |
|---|---|---|---|
| 2048 | 438.50 | 13.81 | 439.45 |
| 4096 | 453.69 | 13.69 | 450.96 |
| 8192 | 440.96 | 13.45 | 438.71 |
| 16384 | 420.81 | 13.32 | 420.91 |
| 32768 | 389.37 | 12.39 | 388.82 |
| 65536 | 327.35 | 11.53 | 327.79 |

The clean branch reproduces upstream `ayourtch-may15` performance
within noise (≤0.6% delta everywhere) while collapsing 22 upstream
commits into 9 commits with fixes folded into their introducing
commits and no off-by-default experimental code.

## Codex review notes

- `a363493` (combined gate+up MoE helper): **Findings: None.** Fused
  helper preserves the original two-call math; combined kernel
  preserves writeback semantics, no race. Pre-existing oddity noted:
  `DS4_CUDA_MOE_GATE_ROW256/128` flags contribute to `use_gate_row2048`
  but don't change `gate_row_span`, so they still land on the 1024
  branch (now combined unless `NO_GATE_COMBINED` is set). Not
  introduced by this port.
- `8b7a95c` (f16 activation cache): No concerns.
- `189e4d2` (fused rms_norm+f16 + invalidation fix): No concerns.
- `4c45dbc` (async shared expert): **HIGH — async race on lazy
  q8→f16 dequant.** Fixed in follow-up commit `354ddea` below.
- `c92545a` (f16 Q path + fallback fix): No concerns.
- `76ab4d0` (inv_rope+f16+strided): No concerns.
- `78ddc27` (f16 KV + 2× rows attn + fallback fix): No concerns.

### Commit 10 — Fix lazy q8→f16 dequant async stream race (`354ddea`)

Codex follow-up. `cuda_q8_f16_ptr()` launched
`dequant_q8_0_to_f16_kernel` on stream 0. The async shared expert
GEMM consumes that same pointer on `g_async_stream` via
`g_active_cublas`. `g_async_stream` is `cudaStreamNonBlocking`, so
kernels on it do NOT implicitly serialize with stream 0 work — a
first-use cache miss could multiply uninitialized or partially
written f16 weights. The `begin_async` event recorded on stream 0
before the lazy dequant doesn't order the async consumer after the
dequant write either.

Fix: launch dequant on `g_active_stream`. In main context this is
0 (unchanged); in async context this is `g_async_stream` so the
consuming cuBLAS GEMM on the same stream naturally waits.

`speed-bench/clean-10-stream-race-fix.csv`:

| ctx | prefill_tps | Δ vs prev | Δ vs absolute baseline | gen_tps |
|---|---|---|---|---|
| 4096 | 449.26 | +1.69% | **+12.3%** | 13.81 |
| 8192 | 446.78 | +1.57% | **+13.7%** | 13.57 |
| 16384 | 426.59 | +1.45% | **+13.7%** | 13.41 |
| 32768 | 393.98 | +1.47% | **+13.0%** | 12.53 |

Bonus: also a measurable perf win because the lazy dequant no
longer blocks stream 0 for the duration of the first-use shared
expert weight materialization. Main-stream MoE work now overlaps
with the lazy dequant.

`make test`: 1 failure (expected Alice 50/52).

## Optimization hunt with codex

Codex (gpt-5.5 xhigh) was given the handover's positive/negative pattern
list and the prefill cost breakdown. It proposed 4 concrete opportunities:

1. **MoE-down q2 weight-fanout helper** — restructure
   `dev_dot_q2_K_q8_K_block8` (line 8604) so the q2/scales decode for a
   given block runs once and fans out to the existing `isum[p]`
   accumulators. Targets `moe_down_expert_tile16_rowspan_kernel` —
   23.8% of prefill. No new accumulators (avoids block16 trap).
2. **Heads8 online lane-0 softmax-scalar broadcast** — `new_m`,
   `old_scale`, `row_scale`, `sink_scale`, `inv_s` are identical
   across lanes after the score broadcast. Compute in lane 0 and
   `__shfl_sync` them. Touches 3 kernels totaling ~15.9% of prefill.
3. **Warp-reduce q8_K_quantize_kernel** — replace SMEM-256-thread max
   reduction with warp reductions + one warp-partial array.
4. **Prefill-specialized WMMA128 indexer** — narrow causal=1, pos0=0
   variant of `indexer_scores_wmma128_kernel`. 9.2% target.

(Investigations in progress — results below.)

### Idea #1 (MoE-down q2 weight-fanout): **REGRESSED, reverted**

Codex implemented the loop inversion in `dev_dot_q2_K_q8_K_block8`:
each q2/scales decode runs once per (k,j) block, fanning out to all
`isum[p]` accumulators inside an inner p loop.

ptxas resource report after build:
- `moe_down_expert_tile16_rowspan_kernel<512/1024>`: REG:128, SHARED:38400
- Same as baseline, so occupancy SHOULD still be 2 CTAs/SM by static count.

Benchmark (`speed-bench/clean-11-q2fanout.csv` — kept for reference):

| ctx | prefill_tps | Δ vs prev (354ddea) |
|---|---|---|
| 4096 | 312.13 | **−30.5%** |
| 8192 | 309.86 | **−30.6%** |
| 16384 | 298.13 | **−30.1%** |
| 32768 | 279.83 | **−29.0%** |

**~30% across-the-board regression.** Reverted.

Lesson: the "two outputs from one input set" pattern that worked
for the combined gate+up dot does NOT translate cleanly to MoE
down. Probable cause: the original loop allowed the compiler to
CSE q2 reads across `p` (they're cache-cheap), and the new
structure forced 8 fresh `ys[p]->qs + q8_off` pointer
materializations inside the inner loop, breaking instruction
scheduling. The static register count was unchanged, but the
*scheduling* of the q8 pointer-arithmetic + load chain was much
worse. The handover's "reduce instructions, not bytes" maxim
turned into "do not add instructions that the compiler was already
optimizing away" — a sharper version of the same rule.

This expensive null result is logged in the doc for the next
maintainer: do not retry q2 weight-fanout in this form.

### Idea #2 (lane-0 softmax-scalar broadcast for heads8 prefill): **REGRESSED, reverted**

Codex implemented the lane-0 + `__shfl_sync` broadcast for `new_m`,
`old_scale`, `row_scale`, `sink_scale`, and `inv_s` in
`attention_indexed_mixed_heads8_online_kernel` and
`attention_static_mixed_heads8_online_kernel`. Decode path
untouched.

ptxas:
- `attention_indexed_mixed_heads8_online_kernel<8,16>`: REG:64, SHARED:18448
- `attention_static_mixed_heads8_online_kernel`: REG:64, SHARED:9216

Benchmark (`speed-bench/clean-12-lane0-softmax.csv` — kept for reference):

| ctx | prefill_tps | Δ vs prev |
|---|---|---|
| 4096 | 443.42 | −1.30% |
| 8192 | 442.94 | −0.86% |
| 16384 | 423.33 | −0.77% |
| 32768 | 391.54 | −0.62% |

**~1% across-the-board regression.** Reverted.

Lesson: scalar ops on a warp execute at warp width "for free"
(same wallclock as one lane). Moving them to lane 0 doesn't save
wallclock — it just adds 3-5 `__shfl_sync` per iteration. Codex's
"reduce instructions on 31 idle lanes" intuition undercounts that
the lanes weren't idle, they were executing the same scalar work
in lockstep with lane 0. This is a SIMT-vs-MIMD intuition mismatch.

### Idea #3 (warp-reduce q8_K_quantize_kernel): **NEUTRAL (within noise), reverted**

Replaced the SMEM-256-thread max reduction with warp-level
`__shfl_down_sync` reduction + 8-warp partial array reduction.
Same kernel signature, same one-row-per-CTA mapping.

ptxas: REG:28 unchanged, SHARED dropped 3088 → 1104 bytes.

Benchmark (`speed-bench/clean-13-warp-quantize.csv`):

| ctx | prefill_tps | Δ vs prev |
|---|---|---|
| 4096 | 449.66 | +0.09% |
| 16384 | 426.54 | −0.01% |
| 32768 | 393.93 | −0.01% |

Within noise (±0.1%) at every ctx point. Reverted because the
user asked for "performance improvements only" — a neutral but
cleaner kernel doesn't meet that bar. The SMEM saving (1984
bytes/CTA) could matter if the kernel ever bumped against a
budget; not the case today. Keep filed as "harmless and not
worth the diff" for the next maintainer.

## Optimization-hunt scorecard

| Idea | Result | Notes |
|---|---|---|
| #1 MoE-down q2 weight-fanout | **−30%** prefill | Static REG count held but actual scheduling/scoreboarding got dramatically worse — compiler was CSE-ing q2 reads across `p` better in the original loop. Reverted. |
| #2 Heads8 softmax lane-0 broadcast | **−1%** prefill | SIMT scalar ops are free on idle lanes; broadcast costs > savings. Reverted. |
| #3 Warp-reduce quantize | **±0%** (noise) | Quantize kernel is not on the critical path. SMEM drops 3088→1104 but no measurable perf signal. Reverted. |

3 attempts, 0 wins, 3 documented dead-ends. The combined gate+up
helper exhausted the obvious "two outputs from one input set"
opportunity in this graph; the remaining hot paths (MoE down, heads8
attention) appear to need a structurally different approach (e.g.
`mma.sync` tensor-core rewrites, co-encoded weight formats) rather
than incremental loop transforms.

## Final state

Branch `ayourtch-may15-clean` at `354ddea`, 10 clean commits on
top of `main` (`d0357ec` — "Make KV cache hit decay the default"):

```
354ddea cuda: launch lazy q8->f16 dequant on g_active_stream
ceccc1b add doc about various commits
78ddc27 cuda: f16 KV shared memory + __hfma2 dot product + 2x rows in attention
76ab4d0 cuda: fuse inv_rope + f16 conversion, strided output projection GEMM
c92545a cuda: f16 Q path - Q_B matmul outputs f16, fused norm+rope reads f16
4c45dbc cuda: overlap shared expert with routed MoE via async stream
189e4d2 cuda: fuse rms_norm + f16 conversion for HC matmul prefill paths
8b7a95c cuda: cache f16 activation conversions across consecutive matmuls
a363493 cuda: combined gate+up dot in routed MoE rowspan kernel (+5.4-6.8% prefill)
3f94c48 make: cuda-spark targets sm_121 explicitly instead of nvcc default
```

All fixes from upstream `ayourtch-may15` are folded into their
introducing commits (no separate `fix:` commits trailing). One
correctness bug found in review (`354ddea`, lazy q8→f16 dequant
launching on wrong stream) was fixed and yielded an unexpected
+1.5% perf bonus on top of the +10-12% port total.

**Net result: +12-14% prefill across all contexts vs main, in
10 commits, no off-by-default experimental code, all upstream
fixes folded.**


