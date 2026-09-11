// 4.4(a): B300 cta_group::2 staged TMA pipeline.
// Intended destination: assignment02/cuda/m4_gemm/04a_cta_pair_pipeline.cu

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../common.h"

namespace cg = cooperative_groups;

#ifndef STAGES
#define STAGES 4
#endif

constexpr int BM_CTA = 128;
constexpr int BM_CLUSTER = 256;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int B_ROWS_CTA = BN / 2;
constexpr int NSTAGE = STAGES;

static_assert(NSTAGE >= 2, "pipeline needs at least two stages");

constexpr uint32_t A_STAGE_BYTES =
    BM_CTA * BK * sizeof(__nv_bfloat16);             // 16 KiB
constexpr uint32_t B_STAGE_BYTES =
    B_ROWS_CTA * BK * sizeof(__nv_bfloat16);         // 4 KiB
constexpr uint32_t STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
constexpr uint32_t TMA_BYTES = STAGE_BYTES;          // 20 KiB / CTA / stage

__device__ __forceinline__ uint64_t make_desc_sm100(
    uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n"
            "}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
    }
}

__device__ __forceinline__ bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done != 0;
}

__device__ __forceinline__ void issue_tma_tile(
    uint32_t a_dst, uint32_t b_dst, uint32_t full_mbar,
    const CUtensorMap* tmapA, const CUtensorMap* tmapB,
    int tileK, int tileM, int tileNForThisCta) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(full_mbar), "r"(TMA_BYTES)
        : "memory");

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(a_dst), "l"(tmapA), "r"(tileK), "r"(tileM),
          "r"(full_mbar)
        : "memory");

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(b_dst), "l"(tmapB), "r"(tileK), "r"(tileNForThisCta),
          "r"(full_mbar)
        : "memory");
}

__global__ void gemm_cta2_pipeline(
    const __nv_bfloat16* gA, const __nv_bfloat16* gB,
    float* gD, int M, int N, int K,
    const __grid_constant__ CUtensorMap tmapA,
    const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(smem_raw) + 1023) &
        ~static_cast<uintptr_t>(1023));

    cg::cluster_group cluster = cg::this_cluster();
    int rank = cluster.block_rank();
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    __shared__ __align__(8) uint64_t full[NSTAGE];
    __shared__ __align__(8) uint64_t empty[NSTAGE];
    __shared__ uint32_t s_taddr;

    uint32_t full_addr[NSTAGE];
    uint32_t empty_addr[NSTAGE];
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
        full_addr[s] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&full[s]));
        empty_addr[s] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&empty[s]));
    }

    if (warp == 0) {
        if (lane == 0) {
#pragma unroll
            for (int s = 0; s < NSTAGE; ++s) {
                asm volatile(
                    "mbarrier.init.shared::cta.b64 [%0], 1;"
                    :
                    : "r"(full_addr[s])
                    : "memory");
                asm volatile(
                    "mbarrier.init.shared::cta.b64 [%0], 1;"
                    :
                    : "r"(empty_addr[s])
                    : "memory");
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }

        // Pair allocation: warp 0 in both CTAs must execute this instruction.
        uint32_t dst =
            static_cast<uint32_t>(__cvta_generic_to_shared(&s_taddr));
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :
            : "r"(dst), "r"(BN));
        asm volatile(
            "tcgen05.relinquish_alloc_permit."
            "cta_group::2.sync.aligned;");
    }

    // CUDA lays the two blocks of each 2x1x1 cluster consecutively in x.
    int cluster_m = static_cast<int>(blockIdx.x) / 2;
    int tileM = cluster_m * BM_CLUSTER + rank * BM_CTA;
    int tileN = static_cast<int>(blockIdx.y) * BN;
    int localBTileN = tileN + rank * B_ROWS_CTA;
    int k_tiles = K / BK;

    // Also makes the barrier initialization and pair TMEM allocation visible.
    cluster.sync();

    uint32_t taddr = s_taddr;
    uint32_t idesc =
        (1u << 4) |    // D = f32
        (1u << 7) |    // A = bf16
        (1u << 10) |   // B = bf16
        (8u << 17) |   // N = 64
        (16u << 24);   // M = 256

    int next_to_issue = 0;
    int warmup = k_tiles < NSTAGE ? k_tiles : NSTAGE;

    // Each CTA loads its own A half and its own N/2 slice of B.
    if (tid == 0) {
        for (; next_to_issue < warmup; ++next_to_issue) {
            int s = next_to_issue % NSTAGE;
            uint32_t stage_base = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem + s * STAGE_BYTES));
            issue_tma_tile(
                stage_base, stage_base + A_STAGE_BYTES, full_addr[s],
                &tmapA, &tmapB, next_to_issue * BK, tileM, localBTileN);
        }
    }

    for (int it = 0; it < k_tiles; ++it) {
        if (tid == 0) {
            // Mandatory issue path.
            if (next_to_issue == it) {
                int s = it % NSTAGE;
                int generation = it / NSTAGE;
                mbar_wait(empty_addr[s], (generation - 1) & 1);

                uint32_t stage_base = static_cast<uint32_t>(
                    __cvta_generic_to_shared(smem + s * STAGE_BYTES));
                issue_tma_tile(
                    stage_base, stage_base + A_STAGE_BYTES, full_addr[s],
                    &tmapA, &tmapB, it * BK, tileM, localBTileN);
                ++next_to_issue;
            }

            // Optional deeper prefetch. Each CTA tests its local empty barrier.
            while (next_to_issue < k_tiles) {
                int prefetch_it = next_to_issue;
                int s = prefetch_it % NSTAGE;
                int generation = prefetch_it / NSTAGE;
                if (!mbar_try(empty_addr[s], (generation - 1) & 1)) {
                    break;
                }

                uint32_t stage_base = static_cast<uint32_t>(
                    __cvta_generic_to_shared(smem + s * STAGE_BYTES));
                issue_tma_tile(
                    stage_base, stage_base + A_STAGE_BYTES, full_addr[s],
                    &tmapA, &tmapB, prefetch_it * BK, tileM, localBTileN);
                ++next_to_issue;
            }

            int s = it % NSTAGE;
            int generation = it / NSTAGE;
            mbar_wait(full_addr[s], generation & 1);
        }

        // Both CTAs must have the current A/B fragments before rank 0 issues
        // the pair MMA. This is deliberately simple and correctness-first.
        cluster.sync();

        if (tid == 0 && rank == 0) {
            int s = it % NSTAGE;
            uint32_t stage_base = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem + s * STAGE_BYTES));
            uint32_t a_base = stage_base;
            uint32_t b_base = stage_base + A_STAGE_BYTES;

            asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
            for (int round = 0; round < 4; ++round) {
                int kk = round * 16;
                uint64_t da =
                    make_desc_sm100(a_base + kk * 2, 0, 1024, 2);
                uint64_t db =
                    make_desc_sm100(b_base + kk * 2, 0, 1024, 2);
                uint32_t accumulate = (it != 0 || round != 0);

                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 "
                    "[%0], %1, %2, %3, p;\n"
                    "}"
                    :
                    : "r"(taddr), "l"(da), "l"(db), "r"(idesc),
                      "r"(accumulate)
                    : "memory");
            }

            // The same-offset empty barrier is completed in both CTAs.
            asm volatile(
                "tcgen05.commit.cta_group::2."
                "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                "[%0], %1;"
                :
                : "r"(empty_addr[s]), "h"((uint16_t)0x3)
                : "memory");
        }
    }

    // Each CTA waits on its local copy of the multicast completion barrier.
    if (tid == 0) {
        int last_it = k_tiles - 1;
        int last_s = last_it % NSTAGE;
        int last_generation = last_it / NSTAGE;
        mbar_wait(empty_addr[last_s], last_generation & 1);
    }
    cluster.sync();

    asm volatile("tcgen05.fence::after_thread_sync;");
    int row = warp * 32 + lane;
    for (int c = 0; c < BN; c += 8) {
        uint32_t src =
            taddr + (static_cast<uint32_t>(warp * 32) << 16) + c;
        float r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),
              "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
            : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            gD[static_cast<size_t>(tileM + row) * N + tileN + c + i] = r[i];
        }
    }

    cluster.sync();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :
            : "r"(taddr), "r"(BN));
    }

    (void)gA;
    (void)gB;
    (void)M;
}

int main(int argc, char** argv) {
    int M = argc > 3 ? std::atoi(argv[1]) : 4096;
    int N = argc > 3 ? std::atoi(argv[2]) : 4096;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    if (M <= 0 || N <= 0 || K <= 0 ||
        M % BM_CLUSTER || N % BN || K % BK) {
        std::printf("shape must be positive and aligned to %dx%dx%d\n",
                    BM_CLUSTER, BN, BK);
        return 1;
    }

    size_t nA = static_cast<size_t>(M) * K;
    size_t nB = static_cast<size_t>(N) * K;
    size_t nD = static_cast<size_t>(M) * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16(static_cast<float>(dist(rng)));
    for (auto& v : hB) v = __float2bfloat16(static_cast<float>(dist(rng)));

    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(*dA)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(*dB)));
    CUDA_CHECK(cudaMalloc(&dD, nD * sizeof(*dD)));
    CUDA_CHECK(cudaMalloc(&dRef, nD * sizeof(*dRef)));
    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), nA * sizeof(*dA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), nB * sizeof(*dB), cudaMemcpyHostToDevice));

    alignas(64) CUtensorMap tmapA{};
    alignas(64) CUtensorMap tmapB{};
    uint64_t globalDimA[2] = {
        static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
    uint64_t globalDimB[2] = {
        static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
    uint64_t globalStrides[1] = {
        static_cast<uint64_t>(K) * sizeof(__nv_bfloat16)};
    uint32_t boxDimA[2] = {BK, BM_CTA};
    uint32_t boxDimB[2] = {BK, B_ROWS_CTA};
    uint32_t elementStrides[2] = {1, 1};

    CUresult statusA = cuTensorMapEncodeTiled(
        &tmapA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA,
        globalDimA, globalStrides, boxDimA, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult statusB = cuTensorMapEncodeTiled(
        &tmapB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB,
        globalDimB, globalStrides, boxDimB, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (statusA != CUDA_SUCCESS || statusB != CUDA_SUCCESS) {
        std::printf("tensor-map creation failed: A=%d B=%d\n",
                    static_cast<int>(statusA), static_cast<int>(statusB));
        return 1;
    }

    size_t smemBytes = static_cast<size_t>(NSTAGE) * STAGE_BYTES + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_cta2_pipeline, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smemBytes)));
    int activeBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm, gemm_cta2_pipeline, 128, smemBytes));

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((M / BM_CLUSTER) * 2, N / BN);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smemBytes;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    auto launch = [&] {
        CUDA_CHECK(cudaLaunchKernelEx(
            &cfg, gemm_cta2_pipeline, dA, dB, dD, M, N, K, tmapA, tmapB));
    };

    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, dB, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
        &beta, dRef, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(
        got.data(), dD, nD * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        ref.data(), dRef, nD * sizeof(float), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; ++i) bad += got[i] != ref[i];

    int timing_iters = nD >= static_cast<size_t>(4096) * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, timing_iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha, dB, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                &beta, dRef, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        },
        timing_iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);

    std::printf(
        "[4.4a cta2 S=%d] M=%d N=%d K=%d  %s(bad=%ld)  "
        "smem/CTA=%.1f KiB active-blocks/SM<=%d  %.2f ms  %.1f TFLOPS "
        "(cuBLAS %.1f, %.0f%%)\n",
        NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad,
        smemBytes / 1024.0, activeBlocksPerSm, ms, tflops, cub_tflops,
        100.0 * tflops / cub_tflops);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dRef));
    return bad != 0;
}
