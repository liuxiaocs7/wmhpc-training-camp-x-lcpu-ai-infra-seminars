// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    // TODO: 与你的 quant kernel 同形的访存,xor 直通,无数学。
    const int groupsPerRow = K / NVFP4_GROUP;
    const int64_t totalGroups = (int64_t)M * groupsPerRow;
    const int numKTiles = nvfp4_num_ktiles(K);
    const int64_t first = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * BLOCK;
    for (int64_t linear = first; linear < totalGroups; linear += stride) {
        const int row = (int)(linear / groupsPerRow);
        const int group = (int)(linear - (int64_t)row * groupsPerRow);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(in) +
                              linear * NVFP4_GROUP;
        uint64_t packed = 0;
        uint8_t sfByte = 0;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i += 2) {
            const uint16_t lo = src[i];
            const uint16_t hi = src[i + 1];
            const uint8_t byte =
                (uint8_t)(lo ^ (lo >> 8) ^ hi ^ (hi >> 8) ^ 0x5au);
            packed |= (uint64_t)byte << (i * 4);
            sfByte ^= byte;
        }
        reinterpret_cast<uint64_t*>(dataOut)[linear] = packed;
        sfOut[sf_swizzled_offset(row, group, numKTiles)] = sfByte;
    }
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    // TODO: 启动配置。
    constexpr int BLOCK = 256;
    if (M <= 0 || K < NVFP4_GROUP) return;
    const int64_t totalGroups = (int64_t)M * (K / NVFP4_GROUP);
    const int64_t needed = (totalGroups + BLOCK - 1) / BLOCK;
    const int maxBlocks = (sms > 0 ? sms : 1) * 6;
    const int grid = (int)(needed < maxBlocks ? needed : maxBlocks);
    probe_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // for (const auto& shape :
    //      {std::pair{4096, 7168}, {16384, 4096}, {16384, 8192}}) {

    for (const auto& shape :
      {std::pair{1, 4096},
        {16, 4096},
        {256, 4096},
        {1024, 4096},
        {4096, 4096},
        {16384, 4096},
        {4096, 7168},
        {16384, 7168},
        {4096, 8192},
        {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}
