// 问题 4.2(MODIFY):把 4.1 的 staging 换成 TMA,其余不动(仍单缓冲)。
//
// 从你自己的 01_tiled.cu 出发:mma 发射、epilogue、判测口径全部不变,
// 改动集中在两处——host 侧建 tensor map,kernel 侧把 st.shared staging
// 换成 cp.async.bulk.tensor + mbarrier。
//
// 直接告知的事实(工具链与布局配对,不属于考核点):
//   - tensor map 用驱动 API cuTensorMapEncodeTiled 建(Makefile 已链
//     -lcuda);kernel 参数按 const __grid_constant__ CUtensorMap 传
//   - 维度次序:dim0 是最内维(这里是 K,单位为元素数);globalStrides
//     只填外维的字节跨度 {K*2};box 是一次搬运的块 {BK, BM}(B 矩阵
//     {BK, BN});elementStrides 全 1
//   - swizzle 选 CU_TENSOR_MAP_SWIZZLE_128B:TMA 硬件落进 smem 的布局
//     与你 4.1 手工 swz128 摆出来的完全相同,descriptor 一个字段都
//     不用改;interleave/L2 promotion/oob fill 都取 NONE
//   - fence 口径(2.1(b) 在这里兑现):TMA 写 smem 与 tcgen05 读 smem
//     都走 async proxy,fence.proxy.async 不再需要;mbar_wait 之后的
//     tcgen05.fence::after_thread_sync 仍然要
//
// 交付:PASS + 梯子表第二行;回答 handout 4.2 的问题(相对 4.1 的提升
// 为什么这么大——4.1 的 staging 成本由什么构成,用 ncu 佐证)。
//
// 运行:make run/m4_gemm/02_tma;自定形状 ./bin/m4_gemm/02_tma M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int BM = 128, BN = 64, BK = 64;

constexpr uint32_t TMA_BYTES =
    (BM + BN) * BK * sizeof(__nv_bfloat16);
// (128 + 64) * 64 * 2 = 24576 字节

// SM100 smem descriptor(与 4.1 相同;swz128 已经不需要了)。
__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__global__ void gemm_tma(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                         float* gD, int M, int N, int K,
                         const __grid_constant__ CUtensorMap tmapA,
                         const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // TODO:把你 4.1 的 kernel 搬进来,K 循环的 staging 部分改为:
    // (1) 多初始化一组 mbarrier:full(TMA 到达)。4.1 里等 mma 消费
    //     完成的那个继续当 empty 用
    // (2) 每轮:除首轮外先等 empty(smem 可覆写)→ 单线程发 TMA →
    //     等 full → mma(与 4.1 相同)→ commit
    //     发 TMA = 一条 mbarrier.arrive.expect_tx(字节数一次报满
    //     (BM+BN)*BK*2,A、B 两条拷贝共用一个 mbar)+ 两条
    //     cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::
    //     complete_tx::bytes,坐标次序与 tensor map 的维度次序一致:
    //     A 是 {it*BK, tileM},B 是 {it*BK, tileN}
    // (3) 删掉 st.shared staging、swz128、fence.proxy.async(见文件头)
    // full/empty 的 parity 都随轮次翻转,想清楚各自翻转的节奏。

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    // shared memory 地址
    uint8_t* sA = smem;
    uint8_t* sB = smem + BM * BK * 2;

    __shared__ __align__(8) uint64_t full;
    __shared__ __align__(8) uint64_t empty;
    __shared__ uint32_t s_taddr;
    uint32_t fullAddr = (uint32_t)__cvta_generic_to_shared(&full);
    uint32_t emptyAddr = (uint32_t)__cvta_generic_to_shared(&empty);

    if (warp == 0) {
        // 初始化 mbarrier
        if (lane == 0) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], 1;"
                :: "r"(fullAddr)
                : "memory");
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], 1;"
                :: "r"(emptyAddr)
                : "memory");
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        // 分配 tmem
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :: "r"(dst), "r"(BN));
        asm volatile(
            "tcgen05.relinquish_alloc_permit."
            "cta_group::1.sync.aligned;");
    }

    const int tileM = blockIdx.x * BM;
    const int tileN = blockIdx.y * BN;

    __syncthreads();

    uint32_t taddr = s_taddr;
    uint32_t aBase = (uint32_t)__cvta_generic_to_shared(sA);
    uint32_t bBase = (uint32_t)__cvta_generic_to_shared(sB);

    uint32_t idesc =
        (1u << 4)  |  // D = f32
        (1u << 7)  |  // A = bf16
        (1u << 10) |  // B = bf16
        (8u << 17) |  // N = 64
        (8u << 24);   // M = 128

    int iters = K / BK;

    for (int it = 0; it < iters; ++it) { // k维tile级循环
        const int tileK = it * BK;

        // 单缓冲：覆盖 shared memory 前，先等上一轮 MMA 消费完成
        if (it != 0) {
            mbar_wait(emptyAddr, (it - 1) & 1);
        }

        // 关键：所有 warp 都已观察完上一轮，才能重用 barrier/smem
        __syncthreads();

        if (tid == 0) {
            // 告诉 full barrier: 本轮等待 24576 字节
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 "
                "_, [%0], %1;"
                :
                : "r"(fullAddr), "r"(TMA_BYTES)
                : "memory");

            // A[tileM : tileM+BM][tileK : tileK+BK]
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(aBase),
                  "l"(&tmapA),
                  "r"(tileK),
                  "r"(tileM),
                  "r"(fullAddr)
                : "memory");
            
            // B[tileN : tileN+BN][tileK : tileK+BK]
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(bBase),
                  "l"(&tmapB),
                  "r"(tileK),
                  "r"(tileN),
                  "r"(fullAddr)
                : "memory");
        }

        // 等本轮两条 TMA 都完成
        mbar_wait(fullAddr, it & 1);

        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");

            for (int round = 0; round < 4; round++) {
                int kk = round * 16;

                uint64_t da = make_desc_sm100(aBase + kk * 2, 0, 1024, 2);
                uint64_t db = make_desc_sm100(bBase + kk * 2, 0, 1024, 2);

                // 整个 GEMM 只有第一条 MMA 不累加
                uint32_t accumulate = (it != 0 || round != 0);

                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, p;\n"
                    "}"
                    :
                    : "r"(taddr), "l"(da), "l"(db),
                      "r"(idesc), "r"(accumulate)
                    : "memory");
            }
            // 只提交一次，通知 shared memory 已消费完
            asm volatile(
                "tcgen05.commit.cta_group::1."
                "mbarrier::arrive::one.shared::cluster.b64 "
                "[%0];"
                :
                : "r"(emptyAddr)
                : "memory");
        }
    }

    // 最后一轮之后没有下一轮负责等待，因此单独等待
    mbar_wait(emptyAddr, (iters - 1) & 1);

    asm volatile("tcgen05.fence::after_thread_sync;");
    int row = warp * 32 + lane;
    for (int c = 0; c < BN; c += 8) {
        uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + c;
        float r[8];

        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),
              "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
            : "r"(src));

        asm volatile("tcgen05.wait::ld.sync.aligned;");

    #pragma unroll
        for (int i = 0; i < 8; i++) {
            gD[(size_t)(tileM + row) * N + tileN + c + i] = r[i];
        }
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 "
            "%0, %1;"
            :: "r"(taddr), "r"(BN));
    }
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // TODO:cuTensorMapEncodeTiled 建 tmapA/tmapB(参数要点见文件头;
    // 返回值要检查,CUDA_SUCCESS 之外一律报错退出——tensor map 参数错
    // 的典型症状是 kernel 静默读到 0 或越界,而不是启动失败)。
    CUtensorMap tmapA = {}, tmapB = {};

    uint64_t globalDimA[2] = {
        (uint64_t)K,
        (uint64_t)M
    };

    uint64_t globalDimB[2] = {
        (uint64_t)K,
        (uint64_t)N
    };

    // 外层一行的字节跨度
    uint64_t globalStrides[1] = {
        (uint64_t)K * sizeof(__nv_bfloat16)
    };

    uint32_t boxDimA[2] = {BK, BM};
    uint32_t boxDimB[2] = {BK, BN};
    uint32_t elementStrides[2] = {1, 1};

    CUresult statusA = cuTensorMapEncodeTiled(
        &tmapA,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        dA,
        globalDimA,
        globalStrides,
        boxDimA,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (statusA != CUDA_SUCCESS) {
        printf("创建 tmapA 失败，错误码=%d\n", (int)statusA);
        return 1;
    }

    CUresult statusB = cuTensorMapEncodeTiled(
        &tmapB,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        dB,
        globalDimB,
        globalStrides,
        boxDimB,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (statusB != CUDA_SUCCESS) {
        printf("创建 tmapB 失败，错误码=%d\n", (int)statusB);
        return 1;
    }

    dim3 grid(M / BM, N / BN);
    size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_tma,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_tma<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
    };

    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.2 tma] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dRef));
    return bad != 0;
}
