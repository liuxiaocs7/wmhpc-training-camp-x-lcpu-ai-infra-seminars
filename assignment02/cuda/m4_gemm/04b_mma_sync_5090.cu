// 4.4(b): RTX 5090 / SM120 mma.sync implementation of the same
// 128x64x64 tiling. Two shared-memory stages overlap cp.async with WMMA.
// Intended destination: assignment02/cuda/m4_gemm/04b_mma_sync_5090.cu

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../common.h"

namespace wmma = nvcuda::wmma;

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int WARPS = BM / WM;
constexpr int THREADS = WARPS * 32;
constexpr int NSTAGE = 2;

constexpr uint32_t A_STAGE_BYTES = BM * BK * sizeof(__nv_bfloat16);
constexpr uint32_t B_STAGE_BYTES = BN * BK * sizeof(__nv_bfloat16);
constexpr uint32_t STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(smem_addr), "l"(src)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void issue_cp_async_tile(
    uint8_t* stage, const __nv_bfloat16* gA, const __nv_bfloat16* gB,
    int tileM, int tileN, int tileK, int K) {
    uint8_t* sA = stage;
    uint8_t* sB = stage + A_STAGE_BYTES;

    // One 16-byte copy moves eight bf16 values. All 256 threads issue the
    // same number of copies: four for A and two for B.
    constexpr int ELEMENTS_PER_COPY = 16 / sizeof(__nv_bfloat16);
    constexpr int A_COPIES = BM * BK / ELEMENTS_PER_COPY;
    constexpr int B_COPIES = BN * BK / ELEMENTS_PER_COPY;

    for (int copy = threadIdx.x; copy < A_COPIES; copy += blockDim.x) {
        int element = copy * ELEMENTS_PER_COPY;
        int row = element / BK;
        int k = element % BK;
        cp_async_16(
            sA + element * sizeof(__nv_bfloat16),
            gA + static_cast<size_t>(tileM + row) * K + tileK + k);
    }

    for (int copy = threadIdx.x; copy < B_COPIES; copy += blockDim.x) {
        int element = copy * ELEMENTS_PER_COPY;
        int row = element / BK;
        int k = element % BK;
        cp_async_16(
            sB + element * sizeof(__nv_bfloat16),
            gB + static_cast<size_t>(tileN + row) * K + tileK + k);
    }

    cp_async_commit();
}

__global__ void gemm_mma_sync(
    const __nv_bfloat16* gA, const __nv_bfloat16* gB,
    float* gD, int M, int N, int K) {
    extern __shared__ __align__(16) uint8_t smem[];

    int warp = threadIdx.x >> 5;
    int tileM = static_cast<int>(blockIdx.x) * BM;
    int tileN = static_cast<int>(blockIdx.y) * BN;
    int warpM = warp * WM;
    int k_tiles = K / BK;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> accum[BN / WN];
#pragma unroll
    for (int n = 0; n < BN / WN; ++n) {
        wmma::fill_fragment(accum[n], 0.0f);
    }

    // Warm up K0 into stage 0.
    issue_cp_async_tile(smem, gA, gB, tileM, tileN, 0, K);
    cp_async_wait_all();
    __syncthreads();

    for (int it = 0; it < k_tiles; ++it) {
        int read_stage = it & 1;
        int next = it + 1;

        // Once the next stage is known to be free, start loading it before
        // issuing this iteration's MMA. The copy runs while Tensor Cores work.
        if (next < k_tiles) {
            int write_stage = next & 1;
            issue_cp_async_tile(
                smem + write_stage * STAGE_BYTES,
                gA, gB, tileM, tileN, next * BK, K);
        }

        const __nv_bfloat16* sA =
            reinterpret_cast<const __nv_bfloat16*>(
                smem + read_stage * STAGE_BYTES);
        const __nv_bfloat16* sB =
            reinterpret_cast<const __nv_bfloat16*>(
                smem + read_stage * STAGE_BYTES + A_STAGE_BYTES);

#pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK,
                           __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, sA + warpM * BK + kk, BK);

#pragma unroll
            for (int n = 0; n < BN / WN; ++n) {
                wmma::fragment<wmma::matrix_b, WM, WN, WK,
                               __nv_bfloat16, wmma::col_major> b;
                // B is stored as B[n][k]. Interpreting that storage as a
                // column-major KxN matrix gives exactly B^T for GEMM.
                wmma::load_matrix_sync(b, sB + n * WN * BK + kk, BK);
                wmma::mma_sync(accum[n], a, b, accum[n]);
            }
        }

        // No thread may reuse read_stage until every warp has finished its
        // WMMA loads. Then wait for the already-issued next stage.
        __syncthreads();
        if (next < k_tiles) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

#pragma unroll
    for (int n = 0; n < BN / WN; ++n) {
        wmma::store_matrix_sync(
            gD + static_cast<size_t>(tileM + warpM) * N + tileN + n * WN,
            accum[n], N, wmma::mem_row_major);
    }

    (void)M;
}

int main(int argc, char** argv) {
    int M = argc > 3 ? std::atoi(argv[1]) : 4096;
    int N = argc > 3 ? std::atoi(argv[2]) : 4096;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % BK) {
        std::printf("shape must be positive and aligned to %dx%dx%d\n",
                    BM, BN, BK);
        return 1;
    }

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) {
        std::printf("mma.sync bf16 requires SM80 or newer\n");
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

    dim3 grid(M / BM, N / BN);
    size_t smemBytes = NSTAGE * STAGE_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_mma_sync, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smemBytes)));
    int activeBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm, gemm_mma_sync, THREADS, smemBytes));
    auto launch = [&] {
        gemm_mma_sync<<<grid, THREADS, smemBytes>>>(dA, dB, dD, M, N, K);
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
        "[4.4b mma.sync] GPU=%s M=%d N=%d K=%d  %s(bad=%ld)  "
        "smem=%.1f KiB active-blocks/SM=%d  %.2f ms  %.1f TFLOPS "
        "(cuBLAS %.1f, %.0f%%)\n",
        prop.name, M, N, K, bad ? "FAIL" : "PASS", bad,
        smemBytes / 1024.0, activeBlocksPerSm, ms, tflops, cub_tflops,
        100.0 * tflops / cub_tflops);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dRef));
    return bad != 0;
}
