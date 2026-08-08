#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void fill_x(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = ((i % 2048) - 1024) * 0.5f;
    }
}

static inline void fill_y(float *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = (i % 1024) - 512;
    }
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)


__global__ void saxpy_kernel(int n, float a, float *x, float *y, float *out) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < n) {
    //     out[idx] = a * x[idx] + y[idx];
    // }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a * x[i] + y[i];
    }
}

int main(int argc, char *argv[])
{
    int n = 0;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(n * sizeof(float));
    float *h_b = (float *)malloc(n * sizeof(float));
    float *h_c = (float *)malloc(n * sizeof(float));
    fill_x(h_a, n);
    fill_y(h_b, n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 1024;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    saxpy_kernel<<<blocks_per_grid, threads_per_block>>>(n, 2.0f, d_a, d_b, d_c);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += h_c[i];
    }
    printf("SUM=%.0f\n", s);

    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
