
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


torch._functorch.config.debug_partitioner = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        select = torch.ops.aten.select.int(primals_1, 0, 0)
        select_1 = torch.ops.aten.select.int(primals_1, 0, 1)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(select, 2)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(select_1, 2)
        add = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        mul = torch.ops.aten.mul.Tensor(primals_2, add);  add = None
        div = torch.ops.aten.div.Tensor(select, select_1)
        add_1 = torch.ops.aten.add.Tensor(primals_2, div);  div = None
        mul_1 = torch.ops.aten.mul.Tensor(select, 3.0)
        mul_2 = torch.ops.aten.mul.Tensor(primals_2, select_1)
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_1, add_2);  add_1 = add_2 = None
        sub = torch.ops.aten.sub.Tensor(select_1, primals_3);  select_1 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        div_1 = torch.ops.aten.div.Tensor(pow_3, select);  pow_3 = select = None
        sub_1 = torch.ops.aten.sub.Tensor(primals_2, div_1);  div_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul, mul_3);  mul_3 = None
        sub_2 = torch.ops.aten.sub.Tensor(primals_4, primals_3);  primals_4 = None
        sub_3 = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        div_2 = torch.ops.aten.div.Tensor(sub_1, sub_3);  sub_1 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, div_2);  add_3 = div_2 = None
        return [add_4, primals_1, primals_2, primals_3, sub_3]
        
def load_args(reader):
    buf0 = reader.storage(None, 8)
    reader.tensor(buf0, (2,), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4)
    reader.tensor(buf1, (), storage_offset=2, is_leaf=True)  # primals_2
    reader.tensor(buf1, (), storage_offset=1, is_leaf=True)  # primals_3
    reader.tensor(buf1, (), is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
