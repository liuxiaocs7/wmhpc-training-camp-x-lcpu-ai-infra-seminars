#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 16; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(16384);
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)A_shared)[((k % 3) * 8192)])), (k * 64), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].arrive_and_expect_tx(16384);
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)B_shared)[((k % 3) * 8192)])), (((int)blockIdx.x) * 128), (k * 64));
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)B_shared)[(((k % 3) * 8192) + 4096)])), ((((int)blockIdx.x) * 128) + 64), (k * 64));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)A_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_a, ((k_1 % 3) * 16384));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)B_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((k_1 % 3) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_1 * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(C_local + (i_1 * 64))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      *(float2*)(C + (((((((((((int)blockIdx.y) * 131072) + ((i_2 >> 5) * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + ((i_2 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + (((i_2 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 65536)) = *(float2*)(C_local + (i_2 * 2));
    }
  }
}

