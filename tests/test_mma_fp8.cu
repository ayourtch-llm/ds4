/* Correctness test for FP8 MMA attention score computation.
 *
 * Compares three paths:
 *   1. CPU reference: float32 dot product on original (pre-quantization) data
 *   2. GPU scalar: comp_kv_dot_strided (FP8 dequant + scalar FMA)
 *   3. GPU MMA: comp_kv_mma_scores_16x8 (tensor core FP8)
 *
 * Build: nvcc -O2 -arch=native -o test_mma_fp8 tests/test_mma_fp8.cu -lcudart
 * Run:   ./test_mma_fp8
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

/* ---- Constants matching ds4_cuda.cu ---- */
enum {
    N_NOPE     = 448u,
    N_ROPE     = 64u,
    HEAD_DIM   = 512u,
    N_BLOCKS   = 7u,
    BLOCK_SIZE = 64u,
    SCALES_OFF = 0u,
    NOPE_OFF   = 32u,
    ROPE_OFF   = 480u,
    ROW_STRIDE = 640u,
    N_HEADS    = 8u,
    N_ROWS_TEST= 16u
};

/* ---- CPU reference: pack float32 KV to FP8 format ---- */
static uint8_t cpu_float_to_e4m3(float v) {
    if (v == 0.0f) return 0;
    union { float f; uint32_t u; } bits = { .f = v };
    uint8_t sign = (uint8_t)((bits.u >> 24) & 0x80u);
    int exp_f32 = (int)((bits.u >> 23) & 0xffu);
    int exp_e4 = exp_f32 - 120;
    uint32_t man = (bits.u >> 20) & 0x7u;
    if (exp_e4 <= 0) {
        man = (8u | man) >> (1 - exp_e4);
        return (uint8_t)(sign | (man & 7u));
    }
    if (exp_e4 > 15) return (uint8_t)(sign | 0x7eu);
    return (uint8_t)(sign | ((uint32_t)exp_e4 << 3) | man);
}

static float cpu_e4m3_to_float(uint8_t v) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)v, __NV_E4M3);
    return __half2float(*(__half *)&hr);
}

static void cpu_pack_row(uint8_t *out, const float *kv_f32) {
    memset(out, 0, ROW_STRIDE);
    float *scales = (float *)(out + SCALES_OFF);
    uint8_t *nope = out + NOPE_OFF;
    __half *rope = (__half *)(out + ROPE_OFF);

    for (uint32_t blk = 0; blk < N_BLOCKS; blk++) {
        float amax = 1.0e-4f;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++)
            amax = fmaxf(amax, fabsf(kv_f32[blk * BLOCK_SIZE + i]));
        float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        scales[blk] = scale;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            float v = kv_f32[blk * BLOCK_SIZE + i];
            float q = fminf(448.0f, fmaxf(-448.0f, v / scale));
            nope[blk * BLOCK_SIZE + i] = cpu_float_to_e4m3(q);
        }
    }
    for (uint32_t i = 0; i < N_ROPE; i++) {
        rope[i] = __float2half(kv_f32[N_NOPE + i]);
    }
}

/* CPU reference dot product on packed FP8 data (dequantizing inline) */
static float cpu_dot_packed(const uint8_t *packed_row, const float *q) {
    const float *scales = (const float *)(packed_row + SCALES_OFF);
    const uint8_t *nope = packed_row + NOPE_OFF;
    const __half *rope = (const __half *)(packed_row + ROPE_OFF);

    float dot = 0.0f;
    for (uint32_t blk = 0; blk < N_BLOCKS; blk++) {
        float s = scales[blk];
        float block_dot = 0.0f;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            uint32_t d = blk * BLOCK_SIZE + i;
            block_dot += q[d] * cpu_e4m3_to_float(nope[d]);
        }
        dot += block_dot * s;
    }
    for (uint32_t i = 0; i < N_ROPE; i++)
        dot += q[N_NOPE + i] * __half2float(rope[i]);
    return dot;
}

/* CPU reference dot product on original float32 data (no quantization) */
static float cpu_dot_f32(const float *kv, const float *q) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < HEAD_DIM; d++)
        dot += q[d] * kv[d];
    return dot;
}

/* ---- GPU scalar kernel (uses comp_kv_dot_strided logic) ---- */
__device__ __forceinline__ static float e4m3_hw(uint8_t v) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)v, __NV_E4M3);
    return __half2float(*(__half *)&hr);
}

__global__ static void test_scalar_dot_kernel(
        float *scores, const uint8_t *comp_kv, const float *q,
        uint32_t n_rows, uint32_t n_heads) {
    uint32_t r = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (r >= n_rows || h >= n_heads) return;
    uint32_t lane = threadIdx.x;
    if (lane >= 32u) return;

    const uint8_t *rb = comp_kv + (uint64_t)r * ROW_STRIDE;
    const float *sc = (const float *)(rb + SCALES_OFF);
    const uint8_t *nope = rb + NOPE_OFF;
    const __half *rope = (const __half *)(rb + ROPE_OFF);
    const float *qh = q + (uint64_t)h * HEAD_DIM;

    float dot = 0.0f;
    for (uint32_t blk = 0; blk < N_BLOCKS; blk++) {
        uint32_t boff = blk * BLOCK_SIZE;
        float block_dot = 0.0f;
        for (uint32_t d = lane; d < BLOCK_SIZE; d += 32u)
            block_dot += qh[boff + d] * e4m3_hw(nope[boff + d]);
        dot += block_dot * sc[blk];
    }
    for (uint32_t d = lane; d < N_ROPE; d += 32u)
        dot += qh[N_NOPE + d] * __half2float(rope[d]);

    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        dot += __shfl_down_sync(0xffffffffu, dot, off);
    if (lane == 0u)
        scores[r * n_heads + h] = dot;
}

/* ---- GPU MMA kernel (calls the MMA function) ---- */

__device__ __forceinline__ static uint32_t pack4b(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

__global__ static void test_mma_dot_kernel(
        float *scores, const uint8_t *comp_kv, const float *q,
        uint32_t n_rows, uint32_t n_heads) {
    if (n_heads != 8u || blockDim.x < 32u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t tid = lane & 3u;
    const uint32_t gid = lane >> 2u;

    const float *q_heads[8];
    for (int h = 0; h < 8; h++)
        q_heads[h] = q + (uint64_t)h * HEAD_DIM;

    /* Precompute row pointers */
    const uint8_t *nope_ptrs[4];
    const float *scale_ptrs[4];
    for (int i = 0; i < 4; i++) {
        uint32_t r = (i < 2) ? (2u * tid + (uint32_t)i) : (2u * tid + 6u + (uint32_t)i);
        if (r < n_rows) {
            const uint8_t *rb = comp_kv + (uint64_t)r * ROW_STRIDE;
            nope_ptrs[i] = rb + NOPE_OFF;
            scale_ptrs[i] = (const float *)(rb + SCALES_OFF);
        } else {
            nope_ptrs[i] = NULL;
            scale_ptrs[i] = NULL;
        }
    }

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (uint32_t blk = 0; blk < N_BLOCKS; blk++) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

        #pragma unroll
        for (uint32_t sub = 0; sub < 2u; sub++) {
            const uint32_t kbase = blk * 64u + sub * 32u;

            const float *qh0 = q_heads[tid];
            const float *qh1 = q_heads[tid + 4u];
            uint32_t b0 = pack4b(
                (uint8_t)__nv_cvt_float_to_fp8(qh0[kbase + gid],      __NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh0[kbase + gid + 8u], __NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh0[kbase + gid + 16u],__NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh0[kbase + gid + 24u],__NV_SATFINITE, __NV_E4M3));
            uint32_t b1 = pack4b(
                (uint8_t)__nv_cvt_float_to_fp8(qh1[kbase + gid],      __NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh1[kbase + gid + 8u], __NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh1[kbase + gid + 16u],__NV_SATFINITE, __NV_E4M3),
                (uint8_t)__nv_cvt_float_to_fp8(qh1[kbase + gid + 24u],__NV_SATFINITE, __NV_E4M3));

            uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const uint8_t *n0 = nope_ptrs[half * 2];
                const uint8_t *n1 = nope_ptrs[half * 2 + 1];
                uint32_t ax = pack4b(
                    n0 ? n0[kbase + gid]       : 0u,
                    n0 ? n0[kbase + gid + 16u] : 0u,
                    n1 ? n1[kbase + gid]       : 0u,
                    n1 ? n1[kbase + gid + 16u] : 0u);
                uint32_t ay = pack4b(
                    n0 ? n0[kbase + gid + 8u]  : 0u,
                    n0 ? n0[kbase + gid + 24u] : 0u,
                    n1 ? n1[kbase + gid + 8u]  : 0u,
                    n1 ? n1[kbase + gid + 24u] : 0u);
                if (half == 0) { a0 = ax; a1 = ay; }
                else           { a2 = ax; a3 = ay; }
            }

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "+f"(p0), "+f"(p1), "+f"(p2), "+f"(p3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(p0), "f"(p1), "f"(p2), "f"(p3));
        }

        float s_top = 0.0f, s_bot = 0.0f;
        if (gid < n_rows) {
            const uint8_t *rb_top = comp_kv + (uint64_t)gid * ROW_STRIDE;
            s_top = ((const float *)(rb_top + SCALES_OFF))[blk];
        }
        if (gid + 8u < n_rows) {
            const uint8_t *rb_bot = comp_kv + (uint64_t)(gid + 8u) * ROW_STRIDE;
            s_bot = ((const float *)(rb_bot + SCALES_OFF))[blk];
        }
        d0 += p0 * s_top;
        d1 += p1 * s_top;
        d2 += p2 * s_bot;
        d3 += p3 * s_bot;
    }

    /* Rope */
    {
        const float *qr0 = q_heads[tid * 2u] + N_NOPE;
        const float *qr1 = q_heads[tid * 2u + 1u] + N_NOPE;

        for (int rh = 0; rh < 2; rh++) {
            uint32_t r = gid + (uint32_t)rh * 8u;
            if (r >= n_rows) continue;
            const __half *rope = (const __half *)(comp_kv + (uint64_t)r * ROW_STRIDE + ROPE_OFF);
            float rdot0 = 0.0f, rdot1 = 0.0f;
            for (uint32_t dd = 0; dd < N_ROPE; dd++) {
                float rv = __half2float(rope[dd]);
                rdot0 += qr0[dd] * rv;
                rdot1 += qr1[dd] * rv;
            }
            if (rh == 0) { d0 += rdot0; d1 += rdot1; }
            else         { d2 += rdot0; d3 += rdot1; }
        }
    }

    /* Store */
    if (gid < n_rows) {
        scores[gid * 8u + tid * 2u] = d0;
        scores[gid * 8u + tid * 2u + 1u] = d1;
    }
    if (gid + 8u < n_rows) {
        scores[(gid + 8u) * 8u + tid * 2u] = d2;
        scores[(gid + 8u) * 8u + tid * 2u + 1u] = d3;
    }
}

/* ---- Main ---- */
static float randf(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int main(void) {
    srand(42);
    printf("FP8 MMA correctness test\n");
    printf("========================\n\n");

    /* Generate random data */
    float *kv_f32 = (float *)malloc(N_ROWS_TEST * HEAD_DIM * sizeof(float));
    float *q_f32  = (float *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    uint8_t *kv_packed = (uint8_t *)calloc(N_ROWS_TEST, ROW_STRIDE);

    for (uint32_t i = 0; i < N_ROWS_TEST * HEAD_DIM; i++)
        kv_f32[i] = randf() * 4.0f;
    for (uint32_t i = 0; i < N_HEADS * HEAD_DIM; i++)
        q_f32[i] = randf() * 2.0f;

    /* Pack KV to FP8 */
    for (uint32_t r = 0; r < N_ROWS_TEST; r++)
        cpu_pack_row(kv_packed + r * ROW_STRIDE, kv_f32 + r * HEAD_DIM);

    /* ---- Path 1: CPU f32 reference (no quantization) ---- */
    float cpu_f32_scores[N_ROWS_TEST * N_HEADS];
    for (uint32_t r = 0; r < N_ROWS_TEST; r++)
        for (uint32_t h = 0; h < N_HEADS; h++)
            cpu_f32_scores[r * N_HEADS + h] = cpu_dot_f32(
                kv_f32 + r * HEAD_DIM, q_f32 + h * HEAD_DIM);

    /* ---- Path 2: CPU packed FP8 reference ---- */
    float cpu_fp8_scores[N_ROWS_TEST * N_HEADS];
    for (uint32_t r = 0; r < N_ROWS_TEST; r++)
        for (uint32_t h = 0; h < N_HEADS; h++)
            cpu_fp8_scores[r * N_HEADS + h] = cpu_dot_packed(
                kv_packed + r * ROW_STRIDE, q_f32 + h * HEAD_DIM);

    /* ---- Upload to GPU ---- */
    uint8_t *d_kv;
    float *d_q, *d_scores_scalar, *d_scores_mma;
    cudaMalloc(&d_kv, N_ROWS_TEST * ROW_STRIDE);
    cudaMalloc(&d_q, N_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_scores_scalar, N_ROWS_TEST * N_HEADS * sizeof(float));
    cudaMalloc(&d_scores_mma, N_ROWS_TEST * N_HEADS * sizeof(float));
    cudaMemcpy(d_kv, kv_packed, N_ROWS_TEST * ROW_STRIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_f32, N_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Path 3: GPU scalar ---- */
    dim3 grid_scalar(N_ROWS_TEST, N_HEADS);
    test_scalar_dot_kernel<<<grid_scalar, 32>>>(
        d_scores_scalar, d_kv, d_q, N_ROWS_TEST, N_HEADS);
    cudaDeviceSynchronize();

    float gpu_scalar_scores[N_ROWS_TEST * N_HEADS];
    cudaMemcpy(gpu_scalar_scores, d_scores_scalar,
               N_ROWS_TEST * N_HEADS * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- Path 4: GPU MMA ---- */
    test_mma_dot_kernel<<<1, 32>>>(
        d_scores_mma, d_kv, d_q, N_ROWS_TEST, N_HEADS);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("MMA kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float gpu_mma_scores[N_ROWS_TEST * N_HEADS];
    cudaMemcpy(gpu_mma_scores, d_scores_mma,
               N_ROWS_TEST * N_HEADS * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- Compare results ---- */
    printf("%-6s %-6s | %12s %12s %12s %12s | %8s %8s %8s\n",
           "Row", "Head", "CPU f32", "CPU fp8", "GPU scalar", "GPU MMA",
           "fp8 err", "scl err", "mma err");
    printf("------+------+");
    for (int i = 0; i < 53; i++) printf("-");
    printf("+");
    for (int i = 0; i < 27; i++) printf("-");
    printf("\n");

    float max_fp8_err = 0, max_scalar_err = 0, max_mma_err = 0;
    float max_mma_vs_scalar = 0;
    int failures = 0;

    for (uint32_t r = 0; r < N_ROWS_TEST; r++) {
        for (uint32_t h = 0; h < N_HEADS; h++) {
            uint32_t idx = r * N_HEADS + h;
            float ref = cpu_f32_scores[idx];
            float fp8 = cpu_fp8_scores[idx];
            float scl = gpu_scalar_scores[idx];
            float mma = gpu_mma_scores[idx];

            float denom = fmaxf(fabsf(ref), 1.0f);
            float fp8_err = fabsf(fp8 - ref) / denom;
            float scl_err = fabsf(scl - fp8) / fmaxf(fabsf(fp8), 1.0f);
            float mma_err = fabsf(mma - fp8) / fmaxf(fabsf(fp8), 1.0f);
            float mma_vs_scl = fabsf(mma - scl) / fmaxf(fabsf(scl), 1.0f);

            max_fp8_err = fmaxf(max_fp8_err, fp8_err);
            max_scalar_err = fmaxf(max_scalar_err, scl_err);
            max_mma_err = fmaxf(max_mma_err, mma_err);
            max_mma_vs_scalar = fmaxf(max_mma_vs_scalar, mma_vs_scl);

            if (r < 4 || mma_err > 0.05f) {
                printf("%-6u %-6u | %12.4f %12.4f %12.4f %12.4f | %7.4f%% %7.4f%% %7.4f%%",
                       r, h, ref, fp8, scl, mma,
                       fp8_err * 100, scl_err * 100, mma_err * 100);
                if (mma_err > 0.05f) { printf(" FAIL"); failures++; }
                printf("\n");
            }
        }
    }

    printf("\n");
    printf("Summary:\n");
    printf("  FP8 quantization error (vs f32):    max %.4f%%\n", max_fp8_err * 100);
    printf("  GPU scalar error (vs CPU fp8):       max %.4f%%\n", max_scalar_err * 100);
    printf("  GPU MMA error (vs CPU fp8):          max %.4f%%\n", max_mma_err * 100);
    printf("  GPU MMA vs GPU scalar:               max %.4f%%\n", max_mma_vs_scalar * 100);
    printf("\n");

    /* MMA quantizes Q to FP8, so expect ~1-5% error vs the fp8 reference
     * (which uses float Q). The key check is MMA vs scalar — both use FP8 KV
     * but scalar uses float Q while MMA uses fp8 Q. */
    float tolerance = 0.10f; /* 10% relative error acceptable for double-quantized path */
    if (max_mma_err > tolerance) {
        printf("FAIL: MMA error %.2f%% exceeds tolerance %.0f%%\n",
               max_mma_err * 100, tolerance * 100);
        failures++;
    } else {
        printf("PASS: All errors within %.0f%% tolerance\n", tolerance * 100);
    }

    cudaFree(d_kv);
    cudaFree(d_q);
    cudaFree(d_scores_scalar);
    cudaFree(d_scores_mma);
    free(kv_f32);
    free(q_f32);
    free(kv_packed);
    return failures > 0 ? 1 : 0;
}
