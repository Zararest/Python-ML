
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.generate_intermediate_hooks = True




isolate_fails_code_str = None



# torch version: 2.3.0+cu121
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce GTX 1660 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        select = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1 = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        div = torch.ops.aten.div.Tensor(select, select_1)
        add = torch.ops.aten.add.Tensor(arg1_1, div)
        mul = torch.ops.aten.mul.Tensor(select, 3.0)
        mul_1 = torch.ops.aten.mul.Tensor(arg1_1, select_1)
        add_1 = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        sub = torch.ops.aten.sub.Tensor(select_1, arg2_1);  arg2_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        div_1 = torch.ops.aten.div.Tensor(pow_1, select);  pow_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(arg1_1, div_1)
        div_2 = torch.ops.aten.div.Tensor(sub_1, arg3_1);  sub_1 = None
        div_3 = torch.ops.aten.div.Tensor(div_2, arg3_1);  div_2 = None
        neg = torch.ops.aten.neg.default(arg4_1)
        mul_2 = torch.ops.aten.mul.Tensor(neg, div_3);  neg = div_3 = None
        div_4 = torch.ops.aten.div.Tensor(arg4_1, arg3_1);  arg3_1 = None
        neg_1 = torch.ops.aten.neg.default(mul_2);  mul_2 = None
        add_2 = torch.ops.aten.add.Tensor(neg_1, arg4_1);  neg_1 = None
        neg_2 = torch.ops.aten.neg.default(div_4);  div_4 = None
        div_5 = torch.ops.aten.div.Tensor(div_1, select);  div_1 = None
        neg_3 = torch.ops.aten.neg.default(neg_2)
        mul_3 = torch.ops.aten.mul.Tensor(neg_3, div_5);  neg_3 = div_5 = None
        div_6 = torch.ops.aten.div.Tensor(neg_2, select);  neg_2 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        mul_4 = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_5 = torch.ops.aten.mul.Tensor(div_6, mul_4);  div_6 = mul_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(arg4_1, add);  add = None
        mul_7 = torch.ops.aten.mul.Tensor(arg4_1, add_1);  arg4_1 = add_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, arg1_1)
        add_3 = torch.ops.aten.add.Tensor(mul_5, mul_8);  mul_5 = mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_6, 3.0);  mul_6 = None
        add_4 = torch.ops.aten.add.Tensor(mul_3, mul_9);  mul_3 = mul_9 = None
        div_7 = torch.ops.aten.div.Tensor(div, select_1);  div = None
        neg_4 = torch.ops.aten.neg.default(mul_7)
        mul_10 = torch.ops.aten.mul.Tensor(neg_4, div_7);  neg_4 = div_7 = None
        div_8 = torch.ops.aten.div.Tensor(mul_7, select_1);  mul_7 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, div_8);  add_4 = div_8 = None
        add_6 = torch.ops.aten.add.Tensor(add_3, mul_10);  add_3 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_2, arg1_1);  add_2 = arg1_1 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(select_1, 1.0);  select_1 = None
        mul_12 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, mul_12);  mul_12 = None
        add_7 = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(select, 1.0);  select = None
        mul_14 = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_11, mul_14);  mul_11 = mul_14 = None
        add_8 = torch.ops.aten.add.Tensor(add_5, mul_15);  add_5 = mul_15 = None
        full = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full, add_7, 0, 1);  add_7 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full, add_8, 0, 0);  full = add_8 = None
        add_9 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        return [add_9, None, None, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 8)
    reader.tensor(buf0, (2,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 12)
    reader.tensor(buf1, (), storage_offset=2, is_leaf=True)  # arg1_1
    reader.tensor(buf1, (), storage_offset=1, is_leaf=True)  # arg2_1
    buf2 = reader.storage(None, 4)
    reader.tensor(buf2, (), is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 4)
    reader.tensor(buf3, (), is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
