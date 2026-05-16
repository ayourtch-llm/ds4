#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/*
 * Numerical regression for ds4_gpu_rms_norm_matmul_f16_tensor.
 * Compares the new fused path (rms_norm + f16 matmul in one entry) against
 * the established unfused path (ds4_gpu_rms_norm_plain_rows_tensor followed
 * by ds4_gpu_matmul_f16_tensor) on a synthetic weight buffer.  Picks up any
 * divergence introduced by the fused kernel — wrong eps, transposed read,
 * stale activation cache, or mis-strided GEMM.
 */
static int check_rms_norm_matmul_f16_fused_matches_unfused(void) {
    const uint32_t n_tok = 4;
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const float eps = 1e-5f;

    const uint64_t weight_count = (uint64_t)out_dim * in_dim;
    const uint64_t weight_bytes = weight_count * sizeof(uint16_t);
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const size_t page_sz = page_sz_l > 0 ? (size_t)page_sz_l : 4096u;
    const size_t map_bytes = ((weight_bytes + page_sz - 1u) / page_sz) * page_sz;

    void *model_map = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (model_map == MAP_FAILED) {
        fprintf(stderr, "rms_norm_matmul_f16: mmap failed\n");
        return 1;
    }
    uint16_t *weights = (uint16_t *)model_map;
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const float w = ((float)((o * 7u + i * 3u) & 31u) - 15.0f) / 256.0f;
            weights[(uint64_t)o * in_dim + i] = f32_to_f16_bits(w);
        }
    }

    const uint64_t x_count = (uint64_t)n_tok * in_dim;
    float *x_host = (float *)calloc((size_t)x_count, sizeof(float));
    float *out_unfused = (float *)calloc((size_t)n_tok * out_dim, sizeof(float));
    float *out_fused = (float *)calloc((size_t)n_tok * out_dim, sizeof(float));
    if (!x_host || !out_unfused || !out_fused) {
        fprintf(stderr, "rms_norm_matmul_f16: host alloc failed\n");
        munmap(model_map, map_bytes);
        free(x_host); free(out_unfused); free(out_fused);
        return 1;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const float v = sinf((float)(t + 1u) * 0.13f + (float)i * 0.07f);
            x_host[(uint64_t)t * in_dim + i] = v;
        }
    }

    int rc = 1;
    ds4_gpu_tensor *x_dev = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *norm_dev = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *out_dev = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    if (x_dev && norm_dev && out_dev &&
        ds4_gpu_tensor_write(x_dev, 0, x_host, x_count * sizeof(float)) &&
        ds4_gpu_set_model_map(model_map, map_bytes)) {
        const int unfused_ok =
            ds4_gpu_rms_norm_plain_rows_tensor(norm_dev, x_dev, in_dim, n_tok, eps) &&
            ds4_gpu_matmul_f16_tensor(out_dev, model_map, map_bytes, 0,
                                      in_dim, out_dim, norm_dev, n_tok) &&
            ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(out_dev, 0, out_unfused,
                                (uint64_t)n_tok * out_dim * sizeof(float));

        const int fused_ok = unfused_ok &&
            ds4_gpu_rms_norm_matmul_f16_tensor(out_dev, x_dev, model_map, map_bytes, 0,
                                               in_dim, out_dim, n_tok, eps) &&
            ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(out_dev, 0, out_fused,
                                (uint64_t)n_tok * out_dim * sizeof(float));

        if (fused_ok) {
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            for (uint32_t t = 0; t < n_tok; t++) {
                for (uint32_t o = 0; o < out_dim; o++) {
                    const float a = out_unfused[(uint64_t)t * out_dim + o];
                    const float b = out_fused[(uint64_t)t * out_dim + o];
                    const float diff = fabsf(a - b);
                    if (diff > max_abs) max_abs = diff;
                    const float denom = fabsf(a) > 1e-6f ? fabsf(a) : 1e-6f;
                    const float rel = diff / denom;
                    if (rel > max_rel) max_rel = rel;
                }
            }
            (void)f16_bits_to_f32;
            const float tol_abs = 1e-2f;
            const float tol_rel = 5e-2f;
            if (max_abs <= tol_abs || max_rel <= tol_rel) {
                fprintf(stderr,
                        "cuda-regression: rms_norm_matmul_f16 fused vs unfused max_abs=%g max_rel=%g\n",
                        (double)max_abs, (double)max_rel);
                rc = 0;
            } else {
                fprintf(stderr,
                        "rms_norm_matmul_f16 regression: max_abs=%g max_rel=%g (tol_abs=%g tol_rel=%g)\n",
                        (double)max_abs, (double)max_rel,
                        (double)tol_abs, (double)tol_rel);
            }
        } else {
            fprintf(stderr, "rms_norm_matmul_f16: gpu call returned failure\n");
        }
    } else {
        fprintf(stderr, "rms_norm_matmul_f16: setup failed (alloc/write/model_map)\n");
    }

    ds4_gpu_tensor_free(out_dev);
    ds4_gpu_tensor_free(norm_dev);
    ds4_gpu_tensor_free(x_dev);
    free(out_fused);
    free(out_unfused);
    free(x_host);
    munmap(model_map, map_bytes);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_rms_norm_matmul_f16_fused_matches_unfused() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
