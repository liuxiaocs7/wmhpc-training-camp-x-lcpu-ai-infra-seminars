// 问题 3.2(模块压轴):从零写 tcgen05 单 tile GEMM。
//
// 形状 m128n64k64,bf16 输入,f32 累加,cta_group::1,单 block 128 线程。
// 数据通路:global -> smem(K-major + 128B swizzle)-> tcgen05.mma ->
// TMEM -> tcgen05.ld -> global。判测(main 已给出)用小整数严格对拍。
//
// 给你的材料:课件 F27 的七步流程(下面 kernel 里只留了步骤注释)、
// 你在 2.2 写的 descriptor 编码(SM100 位域)、2.3 的 swizzle_128B
// (staging 布局用它;布局错,结果必错——这里是它的真硬件判测)。
// 其余(TMEM alloc、mbarrier、idesc、tcgen05.mma/ld 的写法)自己查
// PTX ISA 对应章节,课件 C15-C21 讲过每一件的语义,数字换成本题形状。
//
// 两个提醒,直接说明:
// - smem 写完到发射 mma 之间需要 fence.proxy.async(2.1 排序题的答案
//   在这里上真硬件;漏掉的现象自己观察一次,写进报告)
// - tcgen05.ld 每个 warp 只能读自己的 32 条 lane(3.1(a));taddr 高
//   16 bit 是 lane 偏移、低 16 bit 是列偏移;ld 之后要 tcgen05.wait::ld
//
// 运行:make run/m3_tcgen05/02_single_tile;多 seed:./judge_tile.sh
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
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

__global__ void tcgen05_tile(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                             float* gD) {
    // TODO: 按七步实现。
    // (1) mbarrier 初始化 + TMEM 分配(alloc 结果写到 shared,广播)
    // A: 128 × 64 × 2 B = 16 KiB。
    // __align__(1024) 保证 sA 的起始地址按 1024 B 对齐。
    __shared__ __align__(1024) uint8_t sA[M * K * 2];
    // B: 64 × 64 × 2 B = 8 KiB。
    // __align__(1024) 保证 sB 的起始地址按 1024 B 对齐。
    __shared__ __align__(1024) uint8_t sB[N * K * 2];
    // 位于 shared memory 中的 64-bit completion mbarrier 对象。
    __shared__ __align__(8) uint64_t mbar;
    // tcgen05.alloc 将分配得到的 32-bit TMEM 基地址写到这里。
    // alloc 后需要 __syncthreads()，才能让整个 CTA 安全读取。
    __shared__ uint32_t s_taddr[1];
    // block 内的线程编号。本题 blockDim.x == 128，因此 tid 为 0～127。
    const int tid = static_cast<int>(threadIdx.x);
    // 当前线程所属的 warp，取值为 0～3。
    const int warp = tid >> 5;
    // 当前线程在所属 warp 内的 lane 编号，取值为 0～31。
    const int lane = tid & 31;
    // 将 mbar 的 generic pointer 转成 PTX shared-space 使用的 32-bit 地址。
    const uint32_t mbar_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    // ==================================================================
    // (1) 初始化 mbarrier，并分配 64 个 TMEM columns
    // ==================================================================
    if (warp == 0) {
        // 只有 warp 0 的 lane 0 初始化 CTA shared-memory mbarrier
        if (lane == 0) {
            // mbarrier 初始化为 1
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(mbar_u32), "r"(1)
                : "memory");
            // 发布 mbarrier 初始化，使后续硬件异步操作能够正确观察到已初始化的 barrier 状态
            asm volatile(
                "fence.mbarrier_init.release.cluster;"
                :
                :
                : "memory");
        }
        // 将 tcgen05.alloc 的返回结果写到 shared-memory 变量 s_taddr (分配的 TMEM)
        const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(s_taddr));
        // warp 0 的全部 32 个线程协作分配 64 个 TMEM columns
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(dst), "r"(64)
            : "memory");
        // 声明本 CTA 不再 alloc，把配额让给同 SM 的其他 CTA
        // warp 0 放弃后续 TMEM 分配许可，但并不释放刚分配的 TMEM
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            :
            :
            : "memory");
    }

    // (2) 全体线程把 A/B 按 swizzled 布局写进 smem
    // ------------------------------------------------------------------
    // (2) 全 CTA 将 A/B 写成 K-major + 128B swizzle 布局
    // ------------------------------------------------------------------
    for (int i = tid; i < M * K; i += blockDim.x) {
        // 定位到 swizzle atom cell
        const int m = i / K;
        const int k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(
            // 输入 row, colByte -> 物理偏移
            &sA[swz128(m, k * 2)]) = gA[i];
    }
    // 输入 B 的 host/global 布局为 B[n][k]
    for (int i = tid; i < N * K; i += blockDim.x) {
        const int n = i / K;
        const int k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(
            &sB[swz128(n, k * 2)]) = gB[i];
    }
    // (3) fence.proxy.async + __syncthreads
    // ------------------------------------------------------------------
    // (3) 将普通 shared-memory 写入发布给 async proxy，并同步整个 CTA
    // ------------------------------------------------------------------
    // 把当前线程此前执行的 shared-memory 写入发布给 async proxy，使后续 tcgen05.mma 能正确看到这些数据。
    asm volatile(
        "fence.proxy.async.shared::cta;"
        :
        :
        : "memory");

    // 所有 128 个线程都必须无条件执行这个 __syncthreads()。
    // 与上面的 proxy fence 配合，确保：
    // 1. 所有线程完成 A/B staging；
    // 2. staging 数据对 tcgen05 使用的 async proxy 可见；
    // 3. mbarrier 初始化已经完成；
    // 4. TMEM 分配完成，s_taddr[0] 已经写入。

    __syncthreads();
    const uint32_t taddr = s_taddr[0];
    // A、B 在 shared-memory 地址空间中的 32-bit 基地址。
    const uint32_t aBase = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
    const uint32_t bBase = static_cast<uint32_t>(__cvta_generic_to_shared(sB));

    // tcgen05 MMA instruction descriptor
    // D format = f32
    // A format = bf16
    // B format = bf16
    // N / 8 = 64 / 8 = 8
    // M / 16 = 128 / 16 = 8
    const uint32_t idesc = 
        (1u << 4)  | 
        (1u << 7)  |
        (1u << 10) |
        (8u << 17) |
        (8u << 24);

    // (4) 单线程发射 4 条 k16 的 tcgen05.mma(第一条不累加),commit
    // ==================================================================
    // (4) 单线程发射四条 k16 MMA，并 commit 一次
    // ==================================================================
    // tcgen05.mma 不是 warp-collective。
    // 它由单个线程发射，随后由硬件异步执行。
    if (tid == 0) {
        asm volatile(
            "tcgen05.fence::after_thread_sync;"
            :
            :
            : "memory");
        
        for (int round = 0; round < 4; round++) {
            // 每条指令处理 k16
            const int kk = round * 16;
            // bf16 每个元素为 2 字节，因此每轮 K 起点增加 32 B。
            //
            // 128B swizzle descriptor：
            //   LBO    = 0
            //   SBO    = 1024
            //   layout = 2
            const uint64_t da =
                make_desc_sm100(
                    aBase + kk * static_cast<int>(sizeof(__nv_bfloat16)),
                    0,
                    1024,
                    2);
            const uint64_t db =
                make_desc_sm100(
                    bBase + kk * static_cast<int>(sizeof(__nv_bfloat16)),
                    0,
                    1024,
                    2
                );
            // 第一条覆盖 D，后面三条累加到 D
            const uint32_t accumulate = static_cast<uint32_t>(round != 0);
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %4, 0;\n"
                "  tcgen05.mma.cta_group::1.kind::f16 "
                "  [%0], %1, %2, %3, p;\n"
                "}\n"
                :
                : "r"(taddr),
                  "l"(da),
                  "l"(db),
                  "r"(idesc),
                  "r"(accumulate)
                : "memory");
        }
        // commit 不会阻塞等待 MMA 完成。
        //
        // 它将此前发射的 MMA batch 与 mbarrier 关联；
        // 真正完成后硬件才会对 mbarrier 执行 arrive。
        asm volatile(
            "tcgen05.commit.cta_group::1"
            ".mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];"
            :
            : "r"(mbar_u32)
            : "memory");
    }
    
    // (5) mbarrier 等待
    // ==================================================================
    // (5) 等待四条 MMA 全部完成
    // ==================================================================
    // mbarrier 的初始 phase 为 0。
    //
    // 异步 MMA batch 完成后，arrival count 归零并翻转 phase。
    // 因此这里等待离开旧 phase 0。
    mbar_wait(mbar_u32, 0);
    asm volatile(
        "tcgen05.fence::after_thread_sync;"
        :
        :
        : "memory");    
    
    // (6) epilogue:每 warp tcgen05.ld 自己的 32 条 lane,写回 global
    // ==================================================================
    // (6) 每个 warp 读取自己对应的 32 条 TMEM lane
    // ==================================================================
    //
    // warp 0：TMEM lane   0..31  -> D row   0..31
    // warp 1：TMEM lane  32..63  -> D row  32..63
    // warp 2：TMEM lane  64..95  -> D row  64..95
    // warp 3：TMEM lane  96..127 -> D row  96..127
    //
    // taddr：
    //   高 16 bit = lane offset
    //   低 16 bit = column offset
    const int row = warp * 32 + lane;
    const uint32_t warp_lane_offset = static_cast<uint32_t>(warp * 32) << 16;
    for (int c = 0; c < N; c += 8) {
        const uint32_t src = taddr + warp_lane_offset + static_cast<uint32_t>(c);
        float r[8];
        // 每个 warp 只能读取属于自己的 32 条 TMEM lane。
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]),
              "=f"(r[1]),
              "=f"(r[2]),
              "=f"(r[3]),
              "=f"(r[4]),
              "=f"(r[5]),
              "=f"(r[6]),
              "=f"(r[7])
            : "r"(src)
            : "memory");
        // tcgen05.ld 的寄存器结果不能立即使用。
        asm volatile(
            "tcgen05.wait::ld.sync.aligned;"
            :
            :
            : "memory");
        for (int i = 0; i < 8; ++i) {
            gD[row * N + c + i] = r[i];
        }
    }
    
    // (7) __syncthreads 后 dealloc
    // ==================================================================
    // (7) 所有 warp 完成读取后释放 TMEM
    // ==================================================================
    __syncthreads();
    if (warp == 0) {
        // delloc 是 warp-collective，必须由完整的 warp 执行
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(taddr), "r"(64)
            : "memory");
    }
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    tcgen05_tile<<<1, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    printf(bad ? "FAIL seed=%u: %ld / %d\n" : "PASS seed=%u\n", seed,
           bad ? bad : (long)seed, M * N);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));
    return bad != 0;
}
