
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_add_div_mul_pow_sub_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_azor/wy/cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp1 = in_ptr1[static_cast<long>(0L)];
        auto tmp3 = in_ptr2[static_cast<long>(0L)];
        auto tmp4 = in_ptr3[static_cast<long>(0L)];
        auto tmp6 = in_ptr3[static_cast<long>(1L)];
        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
        auto tmp5 = decltype(tmp4)(tmp4 * tmp4);
        auto tmp7 = decltype(tmp6)(tmp6 * tmp6);
        auto tmp8 = decltype(tmp5)(tmp5 + tmp7);
        auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
        auto tmp10 = decltype(tmp2)(tmp2 - tmp9);
        auto tmp11 = tmp4 / tmp6;
        auto tmp12 = decltype(tmp3)(tmp3 + tmp11);
        auto tmp13 = static_cast<float>(3.0);
        auto tmp14 = decltype(tmp4)(tmp4 * tmp13);
        auto tmp15 = decltype(tmp3)(tmp3 * tmp6);
        auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
        auto tmp17 = decltype(tmp12)(tmp12 * tmp16);
        auto tmp18 = decltype(tmp9)(tmp9 + tmp17);
        auto tmp19 = decltype(tmp6)(tmp6 - tmp1);
        auto tmp20 = decltype(tmp19)(tmp19 * tmp19);
        auto tmp21 = tmp20 / tmp4;
        auto tmp22 = decltype(tmp3)(tmp3 - tmp21);
        auto tmp23 = tmp22 / tmp10;
        auto tmp24 = decltype(tmp18)(tmp18 + tmp23);
        out_ptr0[static_cast<long>(0L)] = tmp10;
        out_ptr1[static_cast<long>(0L)] = tmp24;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (2, ), (1, ))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (), ())
    buf0 = empty_strided_cpu((), (), torch.float32)
    buf1 = empty_strided_cpu((), (), torch.float32)
    cpp_fused_add_div_mul_pow_sub_0(primals_4, primals_3, primals_2, primals_1, buf0, buf1)
    del primals_4
    return (buf1, primals_1, primals_2, primals_3, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
