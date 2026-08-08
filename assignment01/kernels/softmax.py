"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + pid * n + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    tl.store(z_ptr + pid * n + offsets, num / denom, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    res = torch.empty_like(x)
    m, n = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (m,)
    softmax_kernel[grid](x, res, n, BLOCK_SIZE=BLOCK_SIZE)
    return res
