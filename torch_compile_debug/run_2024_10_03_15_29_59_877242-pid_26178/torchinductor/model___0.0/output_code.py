
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


cpp_fused_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_azor/wy/cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp2 = in_ptr1[static_cast<long>(0L)];
        auto tmp3 = in_ptr2[static_cast<long>(1L)];
        auto tmp4 = in_ptr3[static_cast<long>(0L)];
        auto tmp7 = in_ptr2[static_cast<long>(0L)];
        auto tmp10 = in_ptr4[static_cast<long>(0L)];
        auto tmp1 = decltype(tmp0)(-tmp0);
        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
        auto tmp6 = decltype(tmp5)(tmp5 * tmp5);
        auto tmp8 = tmp6 / tmp7;
        auto tmp9 = decltype(tmp2)(tmp2 - tmp8);
        auto tmp11 = tmp9 / tmp10;
        auto tmp12 = tmp11 / tmp10;
        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
        auto tmp14 = decltype(tmp13)(-tmp13);
        auto tmp15 = decltype(tmp14)(tmp14 + tmp0);
        auto tmp16 = decltype(tmp15)(tmp15 * tmp2);
        auto tmp17 = tmp0 / tmp10;
        auto tmp18 = decltype(tmp17)(-tmp17);
        auto tmp19 = tmp18 / tmp7;
        auto tmp20 = static_cast<float>(2.0);
        auto tmp21 = decltype(tmp5)(tmp5 * tmp20);
        auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
        auto tmp23 = tmp7 / tmp3;
        auto tmp24 = decltype(tmp2)(tmp2 + tmp23);
        auto tmp25 = decltype(tmp0)(tmp0 * tmp24);
        auto tmp26 = decltype(tmp25)(tmp25 * tmp2);
        auto tmp27 = decltype(tmp22)(tmp22 + tmp26);
        auto tmp28 = static_cast<float>(3.0);
        auto tmp29 = decltype(tmp7)(tmp7 * tmp28);
        auto tmp30 = decltype(tmp2)(tmp2 * tmp3);
        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
        auto tmp32 = decltype(tmp0)(tmp0 * tmp31);
        auto tmp33 = decltype(tmp32)(-tmp32);
        auto tmp34 = tmp23 / tmp3;
        auto tmp35 = decltype(tmp33)(tmp33 * tmp34);
        auto tmp36 = decltype(tmp27)(tmp27 + tmp35);
        auto tmp37 = decltype(tmp3)(tmp3 * tmp20);
        auto tmp38 = decltype(tmp16)(tmp16 * tmp37);
        auto tmp39 = decltype(tmp36)(tmp36 + tmp38);
        auto tmp40 = decltype(tmp18)(-tmp18);
        auto tmp41 = tmp8 / tmp7;
        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
        auto tmp43 = decltype(tmp25)(tmp25 * tmp28);
        auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
        auto tmp45 = tmp32 / tmp3;
        auto tmp46 = decltype(tmp44)(tmp44 + tmp45);
        auto tmp47 = decltype(tmp7)(tmp7 * tmp20);
        auto tmp48 = decltype(tmp16)(tmp16 * tmp47);
        auto tmp49 = decltype(tmp46)(tmp46 + tmp48);
        out_ptr1[static_cast<long>(0L)] = tmp39;
        out_ptr2[static_cast<long>(0L)] = tmp49;
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            auto tmp3 = out_ptr1[static_cast<long>(0L)];
            auto tmp8 = out_ptr2[static_cast<long>(0L)];
            auto tmp0 = c10::convert<int>(x0);
            auto tmp1 = static_cast<int>(1);
            auto tmp2 = tmp0 == tmp1;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = tmp2 ? tmp3 : tmp4;
            auto tmp6 = static_cast<int>(0);
            auto tmp7 = tmp0 == tmp6;
            auto tmp9 = tmp7 ? tmp8 : tmp4;
            auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
            out_ptr3[static_cast<long>(x0)] = tmp10;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, ), (1, ))
    assert_size_stride(arg1_1, (), ())
    assert_size_stride(arg2_1, (), ())
    assert_size_stride(arg3_1, (), ())
    assert_size_stride(arg4_1, (), ())
    buf1 = empty_strided_cpu((), (), torch.float32)
    buf2 = empty_strided_cpu((), (), torch.float32)
    buf3 = empty_strided_cpu((2, ), (1, ), torch.float32)
    cpp_fused_0(arg4_1, arg1_1, arg0_1, arg2_1, arg3_1, buf1, buf2, buf3)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    return (buf3, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
