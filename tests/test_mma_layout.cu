#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

/* Place 1.0 at a0 byte0 for a SPECIFIC gid only. See which row it maps to. */
__global__ void test_a_gid(float *out, uint32_t target_gid) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tid = lane & 3u;
    uint32_t gid = lane >> 2u;
    (void)tid;

    uint32_t one4 = 0x38383838u;
    uint32_t a0 = (gid == target_gid) ? 0x38u : 0u; /* only byte0 */
    uint32_t a1 = 0, a2 = 0, a3 = 0;

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(one4), "r"(one4),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));

    out[gid * 8 + tid * 2] = d0;
    out[gid * 8 + tid * 2 + 1] = d1;
    out[(gid + 8) * 8 + tid * 2] = d2;
    out[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

/* Place 1.0 at a0 for a SPECIFIC tid only. */
__global__ void test_a_tid(float *out, uint32_t target_tid) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tid = lane & 3u;
    uint32_t gid = lane >> 2u;
    (void)gid;

    uint32_t one4 = 0x38383838u;
    uint32_t a0 = (tid == target_tid) ? 0x38u : 0u;
    uint32_t a1 = 0, a2 = 0, a3 = 0;

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(one4), "r"(one4),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));

    out[gid * 8 + tid * 2] = d0;
    out[gid * 8 + tid * 2 + 1] = d1;
    out[(gid + 8) * 8 + tid * 2] = d2;
    out[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

/* Place different values per (gid, tid, byte) in a0 to fully decode layout */
__global__ void test_a_unique(float *out) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tid = lane & 3u;
    uint32_t gid = lane >> 2u;

    uint32_t one4 = 0x38383838u; /* B = all 1s */

    /* Encode position as fp8: value = (gid+1) for byte0, 0 for rest.
     * This way each gid contributes a unique value. */
    uint8_t v = (uint8_t)__nv_cvt_float_to_fp8((float)(gid + 1u), __NV_SATFINITE, __NV_E4M3);
    uint32_t a0 = (uint32_t)v; /* only byte0 has value, bytes 1-3 = 0 */
    uint32_t a1 = 0, a2 = 0, a3 = 0;

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(one4), "r"(one4),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));

    out[gid * 8 + tid * 2] = d0;
    out[gid * 8 + tid * 2 + 1] = d1;
    out[(gid + 8) * 8 + tid * 2] = d2;
    out[(gid + 8) * 8 + tid * 2 + 1] = d3;
}

int main(void) {
    float *d_out, h_out[16 * 8];
    cudaMalloc(&d_out, 16 * 8 * sizeof(float));

    printf("=== Test: a0 byte0 on specific GID ===\n");
    for (uint32_t g = 0; g < 8; g++) {
        cudaMemset(d_out, 0, 16 * 8 * sizeof(float));
        test_a_gid<<<1, 32>>>(d_out, g);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("gid=%d → ", g);
        for (int r = 0; r < 16; r++) {
            if (h_out[r * 8] != 0.0f) printf("row%d(%.1f) ", r, h_out[r * 8]);
        }
        printf("\n");
    }

    printf("\n=== Test: a0 byte0 on specific TID ===\n");
    for (uint32_t t = 0; t < 4; t++) {
        cudaMemset(d_out, 0, 16 * 8 * sizeof(float));
        test_a_tid<<<1, 32>>>(d_out, t);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("tid=%d → ", t);
        for (int r = 0; r < 16; r++) {
            if (h_out[r * 8] != 0.0f) printf("row%d(%.1f) ", r, h_out[r * 8]);
        }
        printf("\n");
    }

    printf("\n=== Test: a0 byte0 = gid+1, unique per gid ===\n");
    cudaMemset(d_out, 0, 16 * 8 * sizeof(float));
    test_a_unique<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("D[row][col=0]:\n");
    for (int r = 0; r < 16; r++)
        printf("  row %2d: %.1f\n", r, h_out[r * 8]);

    cudaFree(d_out);
    return 0;
}
