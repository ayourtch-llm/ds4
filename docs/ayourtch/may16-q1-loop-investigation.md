# Q1 loop investigation and int64 atomicAdd MoE down (`ayourtch-may16`)

Date: 2026-05-17. Branch `ayourtch-may16` HEAD `2a95003` (`cuda: optional int64
fixed-point MoE down accumulator for determinism`).

## TL;DR

- ds4-eval Q1 (GPQA Diamond, LMC astronaut, 201-token prompt) was observed to
  occasionally enter an infinite token loop emitting `1.612 * 10^(-3.5) = ...`
  until the 16 000-token cap on `ayourtch-may16` with `78ddc27` (cuda f16 KV +
  __hfma2) cherry-picked on top.
- Initial reaction was to skip `78ddc27` from the may15-clean port. **That was
  premature.** Pure `antirez/ds4 main` (`d0357ec`) loops the same Q1 with the
  same pattern at the same ~20% rate (1/5 runs in our measurement).
- Root cause confirmed by codex: floating-point `atomicAdd` in the routed-MoE
  expert-tile down kernels. fp32 add is non-associative, expert contributions
  to the same `down_out[tok,row]` arrive in scheduler-dependent order, and
  Q1 lives near a token-decision boundary so the bit drift flips greedy
  decoding into a loop.
- `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` confirms it (Q1 10/10 PASS) but costs
  -26 to -30 % prefill. Not viable as a default.
- This commit adds `DS4_CUDA_MOE_INT64_DOWN=1`, an alternative path that uses
  an int64 fixed-point accumulator (scale 2^32) + post-conversion. Integer
  addition is associative ⇒ deterministic, no perf cost in our measurement
  (within ±1 % of fp32 atomic at 2K-32K prefill). **Q1 stops looping; Q9 can
  still loop** because cuBLAS TF32 GEMM and one other atomicAdd in the
  indexed-attention path (ds4_cuda.cu:3029 on main) remain non-deterministic.

## Background

ds4-eval default decoding is `--temp 0` greedy. Expected behavior: identical
prompt + identical model + identical kernels ⇒ identical output, every run.
We observed instead:

| Build | Q1 ×5 (--questions 1, --plain) | Token counts when passing |
|---|---|---|
| `main` d0357ec (no may15 commits, default nvcc arch) | 4 PASS / 1 LOOP | 2121, 3360, 2121, 2883 |
| `ayourtch-may16` + 78ddc27 vanilla | (similar; ~20 % loop rate inferred) | varies |
| `+ DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` | 10/10 PASS | 2088-2951 |
| `+ DS4_CUDA_MOE_INT64_DOWN=1` (this commit) | 5/5 PASS | 2221-5226 |

The loop pattern is always the Lorentz-factor `√(2.6e-7)` calculation:

```
... 1.612 * 10^(-3.5) = 1.612 * 10^(-3.5) = 1.612 * 10^(-3.5) = ...
```

Same exact substring on every loop, regardless of build. This is because the
question itself reaches a numerically delicate state and any bit-level drift
in attention logits pushes greedy decoding into self-reinforcing repetition
of that fragment.

## Root cause: fp32 atomicAdd in MoE down

Codex review of `main` found (file paths and line numbers on `main` d0357ec):

- `ds4_cuda.cu:9856-9858` — `use_atomic_down` is enabled by default when
  `n_tokens >= 128`. Q1 prefill = 201 tokens, so prefill hits the path.
- `ds4_cuda.cu:9293, 9356, 9426, 9498, 9572` — five
  `atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p])` sites in the
  five `moe_down_expert_tile{4,8,16}_row{32,2048,span}` kernels.

When use_atomic_down is on, the per-(expert, token, row) tile kernels race
to atomically add their contribution into the shared per-token output row.
fp32 addition is not associative, so the final result depends on arrival
order, which in turn depends on SM scheduling, warp occupancy, and many
other non-deterministic factors. The order-induced drift is typically tiny
(ULPs), but Q1 sits near a token-decision boundary and a single ULP flip in
a single logit can swap which token wins, after which generation diverges.

Additional non-determinism sources flagged by codex but not addressed in this
commit:

- `ds4_cuda.cu:3029` (on main) — `atomicAdd(&comp_count, ...)` in the
  generic indexed attention path. Reachable during long generation. Permutes
  `comp_rows`, then attention output sums in non-deterministic order.
- cuBLAS Tensor Core math mode: TF32 is enabled (lines ~1215-1219 on main)
  and `cublasGemmEx` is called with `CUBLAS_GEMM_DEFAULT` (lines 5805,
  6047, 7347 on main). TF32 GEMMs on Ampere+/sm_121 are documented to be
  non-bit-deterministic by default.

## The int64 fixed-point fix

The idea (codex's proposal, validated empirically):

1. Allocate an `unsigned long long` scratch buffer of size `n_tokens *
   out_dim`.
2. Zero it before launching the down kernel.
3. Each expert tile that would have done `atomicAdd((float*)down_out + i,
   acc[p])` instead does
   `atomicAdd((unsigned long long*)(scratch + i), (unsigned long long)
   __double2ll_rn((double)acc[p] * 2^32))`.
4. After all tiles complete, run a one-shot kernel that does
   `out[i] = (float)((double)((long long)scratch[i]) * (1.0 / 2^32))`.

Why this works:

- Integer add is associative mod 2^64. Inter-block arrival order no longer
  affects the bit pattern of the integer sum.
- 2^32 scale gives 32 fractional bits, more than fp32's 24 mantissa bits,
  so no precision is lost in the round-to-nearest conversion of individual
  contributions.
- Worst-case overflow: largest plausible single contribution is O(100), so
  scaled it is O(2^39); sum across `top_k=6` experts is O(2^42), well under
  2^63. Plenty of headroom.

### Scope of this commit

Only `moe_down_expert_tile16_row2048_kernel` is patched. This is the kernel
that runs by default during prefill (`use_down_row2048 && down_row_span ==
2048u`, both default for `n_tokens >= 128`). If the user sets one of the
`DS4_CUDA_MOE_DOWN_ROW512/1024/256/128/64` env vars or
`DS4_CUDA_MOE_NO_DOWN_ROW2048`, a different kernel runs and the int64 path
is silently ignored (the kernel uses fp32 atomic as before).

The other four kernels (`tile4_row32`, `tile8_row32`, `tile16_row32`,
`tile16_rowspan<512>`, `tile16_rowspan<1024>`) are unchanged. Extending the
patch to them is mechanical but adds parameter-passing churn that we did
not want to commit until the prototype validated.

## Measurements

### Determinism (Q1 ×5, --questions 1, --plain)

| Mode | Pass rate | Token counts |
|---|---|---|
| Default (fp32 atomic) | 4/5 | 2121, 3360, 2121, 2883, **LOOP 16000** |
| `NO_ATOMIC_DOWN=1` | 10/10 | 2088-2951 |
| `INT64_DOWN=1` | 5/5 | 2221, 2271, 5226, 3005, 2270 |

Token counts still vary with `INT64_DOWN=1` because cuBLAS TF32 and the
indexed-attn atomic remain non-deterministic. But Q1 no longer loops.

### Prefill throughput (`ds4-bench`, GB10/sm_121, head_dim=512, top_k=6)

| ctx | INT64 prototype | may15-clean (fp32 atomic) | Δ |
|---|---|---|---|
| 2 048 | 436.27 | 444.80 | -1.9 % |
| 10 240 | 440.53 | 436.28 | +1.0 % |
| 18 432 | 420.64 | 420.12 | +0.1 % |
| 26 624 | 404.39 | 402.41 | +0.5 % |
| 32 768 | 391.30 | 390.22 | +0.3 % |

Within noise. The int64 path costs nothing measurable.

### 10-question eval

| Build | --questions 10 result | Notes |
|---|---|---|
| baseline (no 78ddc27) | 9/10, runtime 35 m | Q8 SuperGPQA E-vs-F (baseline failure) |
| 78ddc27 + this commit `INT64_DOWN=1` | 8/10, runtime 39 m | Q1 PASS at 3516 tokens (no loop). Q8 baseline fail. Q9 AIME geometry hit 16000-token cap on `28 = 13` repetition |

Q9 looped on a different question (AIME2025 isosceles trapezoid + triangle
intersection). It is the same fundamental issue — drift pushes greedy
decoding into a self-reinforcing repetition — but on a different question.
The int64 patch fixed the *Q1* manifestation without fully eliminating the
underlying class of failure. Other questions remain at risk until the
indexed-attn atomic and cuBLAS TF32 are also addressed.

### Full eval (75 questions)

Single run of `DS4_CUDA_MOE_INT64_DOWN=1 ./ds4-eval --plain --trace ...`
on `ayourtch-may16` HEAD `2a95003`:

```
ds4-eval: 56/75 passed, 19 failed, runtime 11h:13m
```

Result file: `/tmp/eval-int64-full.log` (and `/tmp/eval-int64-full.trace`).

Breakdown of the 19 failures:

- **Q1 PASSED** (2221 tokens, "B") — the target of this work.
- **Q8 SuperGPQA mouthparts** (F vs E) — known baseline failure on this
  question, present in every run we did.
- **9 questions hit the 16 000-token cap with a wrong answer** — these
  are the same class of failure as the original Q1 loop, just on different
  questions: Q28, Q31, Q33, Q52, Q63, Q66, Q68, Q70, Q72, Q75. Several of
  these wrap up at "0/1/2/4" as a final fallback after running out of
  tokens, suggesting the model entered a degenerate repetition that the
  trace-end heuristic recovered from with the wrong scratch number.
- **3 questions hit 16 000 cap but still PASSED** (Q13, Q58, Q64) —
  the model emitted the correct answer mid-generation and the eval matched
  on it before the cap.
- **The remaining failures** (Q11, Q20, Q26, Q35, Q41, Q67, Q71, Q74) are
  ordinary "wrong answer" outcomes that may or may not be drift-related;
  they did not hit the token cap.

Interpretation:

- The int64 patch demonstrably fixes the *target* question (Q1) — that
  failure mode is gone.
- It does not eliminate the broader class of "long-generation drift into
  repetition" because cuBLAS TF32 and the indexed-attn atomic at line 3029
  remain non-deterministic. Roughly 10/75 ≈ 13 % of questions hit the
  cap in this single run; we expect that rate to be variable across runs.
- A second full-eval run on the same build would likely show a different
  set of cap-hitting questions but a similar total count, because the
  drift is broad-spectrum rather than question-specific.
- We did not run a 75-question baseline (no `INT64_DOWN`, no patches) for
  direct A/B comparison because each run is ~11 hours. A useful future
  experiment would be: 3 full-eval runs at default and 3 with
  `INT64_DOWN=1`, comparing both the pass rate distribution AND the set
  of cap-hitting questions across runs.

## How to test

```bash
# A. Q1 stability check (5 runs, --questions 1).
for i in 1 2 3 4 5; do
  DS4_CUDA_MOE_INT64_DOWN=1 ./ds4-eval --questions 1 --plain 2>&1 \
    | grep -E "PASSED|FAILED|runtime"
done

# B. Prefill cost.
DS4_CUDA_MOE_INT64_DOWN=1 ./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 8192 --gen-tokens 128

# C. Disable the atomic path entirely (slow but fully deterministic on the
# MoE-down axis).
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 ./ds4-eval --questions 1 --plain
```

## Open follow-ups

1. **Cover the other four down kernels.** Mechanical extension of the
   `atomic_i64`+`fixed_scale` parameter pattern. Worth doing before exposing
   the env var to non-experts.
2. **Indexed-attention atomic at `ds4_cuda.cu:3029`** (on main; line number
   shifts on `ayourtch-may16` due to upstream changes). Rewrite to a
   deterministic scan + permute, or apply the same int64 trick.
3. **cuBLAS TF32 GEMM.** Two options:
   - Switch `cublasGemmEx` from `CUBLAS_GEMM_DEFAULT` to
     `CUBLAS_GEMM_ALGO0_TENSOR_OP` (fixed algo) or
     `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and verify the algo is stable across
     runs; OR
   - Disable TF32 globally via `cublasSetMathMode(..., CUBLAS_DEFAULT_MATH)`
     and accept the perf cost.
4. **Make INT64_DOWN the default once #1 lands** if measurement still shows
   ≤1% prefill cost. The current bench is on one workload (Italian-prose
   prefill); chat workloads with `n_tokens` between 128 and 256 may behave
   differently and should be sanity-checked.

## References

- Commit: `2a95003 cuda: optional int64 fixed-point MoE down accumulator for
  determinism`
- Related: [[ayourtch-may15-clean]] (port log for the perf commits)
- Related: [[handover.md]]
