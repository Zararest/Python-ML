class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2]", primals_2: "f32[]", primals_3: "f32[]", primals_4: "f32[]"):
        # File: /home/azor/projects/Sber-ML/model-debug.py:10 in forward, code: w0 = self.w[0]
        select: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 0)
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:11 in forward, code: w1 = self.w[1]
        select_1: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 1)
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:12 in forward, code: r1 = x3 * (w0 ** 2 + w1 ** 2)
        pow_1: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select, 2)
        pow_2: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select_1, 2)
        add: "f32[]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        mul: "f32[]" = torch.ops.aten.mul.Tensor(primals_2, add);  add = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        div: "f32[]" = torch.ops.aten.div.Tensor(select, select_1)
        add_1: "f32[]" = torch.ops.aten.add.Tensor(primals_2, div);  div = None
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(select, 3.0)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(primals_2, select_1)
        add_2: "f32[]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(add_1, add_2);  add_1 = add_2 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:14 in forward, code: r3 = x3 - (w1 - x2) ** 2 / w0
        sub: "f32[]" = torch.ops.aten.sub.Tensor(select_1, primals_3);  select_1 = None
        pow_3: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(pow_3, select);  pow_3 = select = None
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(primals_2, div_1);  div_1 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:15 in forward, code: return r1 + r2 + r3 / (x1 - x2 - r1)
        add_3: "f32[]" = torch.ops.aten.add.Tensor(mul, mul_3);  mul_3 = None
        sub_2: "f32[]" = torch.ops.aten.sub.Tensor(primals_4, primals_3);  primals_4 = None
        sub_3: "f32[]" = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(sub_1, sub_3);  sub_1 = None
        add_4: "f32[]" = torch.ops.aten.add.Tensor(add_3, div_2);  add_3 = div_2 = None
        return [add_4, primals_1, primals_2, primals_3, sub_3]
        