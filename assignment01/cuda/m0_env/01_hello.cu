// 问题 0.1：第一个 CUDA 程序（模块 8 的编译实验也用它）。
// 编译运行：make run/m0_env/01_hello
#include "common.h"

__global__ void hello() {
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // 启动 4 个 block、每个 block 8 个线程
    hello<<<4, 8>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
