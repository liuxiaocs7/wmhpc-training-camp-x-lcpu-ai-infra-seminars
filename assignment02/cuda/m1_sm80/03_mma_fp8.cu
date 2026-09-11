#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::printf("FAIL: CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                        cudaGetErrorString(err__));                              \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static_assert(sizeof(__nv_fp8_e4m3) == 1,
              "__nv_fp8_e4m3 must occupy exactly one byte");

// A: [16][32]，行主序
// B: [32][8]，行主序
// D: [16][8]，行主序
__global__ void mma_fp8_tile(const __nv_fp8_e4m3* A,
                             const __nv_fp8_e4m3* B,
                             float* D) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;  // 0~7
    const int tig = lane & 3;     // 0~3

    // 通过 unsigned char 读取 FP8 的原始 8-bit 表示。
    const unsigned char* A_bytes =
        reinterpret_cast<const unsigned char*>(A);
    const unsigned char* B_bytes =
        reinterpret_cast<const unsigned char*>(B);

    // A fragment：
    // 每个 lane 有 16 个 FP8，即 4 个 b32 寄存器。
    //
    // 对寄存器 r=0..3、寄存器内 byte j=0..3：
    //
    // row = group + 8*(r&1)
    // k   = 4*tig + j + 16*(r>>1)
    unsigned a[4];

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = group + 8 * (r & 1);
        const int k_base = 4 * tig + 16 * (r >> 1);

        unsigned packed = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = k_base + j;
            const unsigned byte =
                static_cast<unsigned>(A_bytes[row * 32 + k]);

            packed |= byte << (8 * j);
        }

        a[r] = packed;
    }

    // B fragment：
    // 每个 lane 有 8 个 FP8，即 2 个 b32 寄存器。
    //
    // 对寄存器 r=0..1、寄存器内 byte j=0..3：
    //
    // k = 4*tig + j + 16*r
    // n = group
    unsigned b[2];

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int k_base = 4 * tig + 16 * r;
        const int n = group;

        unsigned packed = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = k_base + j;
            const unsigned byte =
                static_cast<unsigned>(B_bytes[k * 8 + n]);

            packed |= byte << (8 * j);
        }

        b[r] = packed;
    }

    // C 是 FP32 累加器。本题令 C=0，因此 D=A*B。
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d[4];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    // 使程序在不支持 FP8 MMA 的架构上确定地产生 MISMATCH，
    // 而不是使用未初始化的 d。
    d[0] = 1234567.0f;
    d[1] = 1234567.0f;
    d[2] = 1234567.0f;
    d[3] = 1234567.0f;
#endif

    // m16n8 的 D fragment 映射：
    //
    // d[0] -> D[group    ][2*tig    ]
    // d[1] -> D[group    ][2*tig + 1]
    // d[2] -> D[group + 8][2*tig    ]
    // d[3] -> D[group + 8][2*tig + 1]
    D[group * 8 + 2 * tig] = d[0];
    D[group * 8 + 2 * tig + 1] = d[1];
    D[(group + 8) * 8 + 2 * tig] = d[2];
    D[(group + 8) * 8 + 2 * tig + 1] = d[3];
}

int main(int argc, char** argv) {
    unsigned seed = 1;

    if (argc >= 2) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0') {
            std::printf("FAIL: invalid seed: %s\n", argv[1]);
            return 1;
        }

        seed = static_cast<unsigned>(parsed);
    }

    constexpr int M = 16;
    constexpr int N = 8;
    constexpr int K = 32;

    __nv_fp8_e4m3 hA[M * K];
    __nv_fp8_e4m3 hB[K * N];
    float reference[M * N] = {};
    float result[M * N] = {};

    std::mt19937 rng(seed);

    // 仅生成能够被 E4M3 精确表示的小整数。
    //
    // A ∈ {-2,-1,0,1,2}
    // B ∈ {-1,0,1}
    std::uniform_int_distribution<int> dist_a(-2, 2);
    std::uniform_int_distribution<int> dist_b(-1, 1);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float value = static_cast<float>(dist_a(rng));
            hA[m * K + k] = __nv_fp8_e4m3(value);
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const float value = static_cast<float>(dist_b(rng));
            hB[k * N + n] = __nv_fp8_e4m3(value);
        }
    }

    // 使用转换后的 FP8 值计算 CPU 参考结果。
    // 这样对拍的是实际送给 GPU MMA 的输入。
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                const float av = static_cast<float>(hA[m * K + k]);
                const float bv = static_cast<float>(hB[k * N + n]);
                sum += av * bv;
            }

            reference[m * N + n] = sum;
        }
    }

    __nv_fp8_e4m3* dA = nullptr;
    __nv_fp8_e4m3* dB = nullptr;
    float* dD = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(hA)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), sizeof(hB)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dD), sizeof(result)));

    CUDA_CHECK(
        cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    // 一个 block、一个 warp、一个 m16n8k32 tile。
    mma_fp8_tile<<<1, 32>>>(dA, dB, dD);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(result, dD, sizeof(result), cudaMemcpyDeviceToHost));

    long mismatches = 0;
    int first_m = -1;
    int first_n = -1;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const int index = m * N + n;

            // 输入和运算结果均为较小的精确整数，可以严格比较。
            if (result[index] != reference[index]) {
                if (first_m < 0) {
                    first_m = m;
                    first_n = n;
                }
                ++mismatches;
            }
        }
    }

    if (mismatches == 0) {
        std::printf("PASS seed=%u\n", seed);
    } else {
        const int index = first_m * N + first_n;
        std::printf(
            "MISMATCH seed=%u count=%ld first=(%d,%d) "
            "gpu=%.9g cpu=%.9g\n",
            seed, mismatches, first_m, first_n,
            result[index], reference[index]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    return mismatches == 0 ? 0 : 1;
}
