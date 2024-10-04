class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2]", primals_2: "f32[]", primals_3: "f32[]", sub_3: "f32[]", tangents_1: "f32[]"):
        # File: /home/azor/projects/Sber-ML/model-debug.py:10 in forward, code: w0 = self.w[0]
        select: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 0)
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:11 in forward, code: w1 = self.w[1]
        select_1: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 1);  primals_1 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        div: "f32[]" = torch.ops.aten.div.Tensor(select, select_1)
        add_1: "f32[]" = torch.ops.aten.add.Tensor(primals_2, div)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(select, 3.0)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(primals_2, select_1)
        add_2: "f32[]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:14 in forward, code: r3 = x3 - (w1 - x2) ** 2 / w0
        sub: "f32[]" = torch.ops.aten.sub.Tensor(select_1, primals_3);  primals_3 = None
        pow_3: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(pow_3, select);  pow_3 = None
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(primals_2, div_1)
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:15 in forward, code: return r1 + r2 + r3 / (x1 - x2 - r1)
        div_2: "f32[]" = torch.ops.aten.div.Tensor(sub_1, sub_3);  sub_1 = None
        div_4: "f32[]" = torch.ops.aten.div.Tensor(div_2, sub_3);  div_2 = None
        neg: "f32[]" = torch.ops.aten.neg.default(tangents_1)
        mul_4: "f32[]" = torch.ops.aten.mul.Tensor(neg, div_4);  neg = div_4 = None
        div_5: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, sub_3);  sub_3 = None
        neg_1: "f32[]" = torch.ops.aten.neg.default(mul_4);  mul_4 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:15 in forward, code: return r1 + r2 + r3 / (x1 - x2 - r1)
        add_5: "f32[]" = torch.ops.aten.add.Tensor(neg_1, tangents_1);  neg_1 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:14 in forward, code: r3 = x3 - (w1 - x2) ** 2 / w0
        neg_2: "f32[]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        div_7: "f32[]" = torch.ops.aten.div.Tensor(div_1, select);  div_1 = None
        neg_3: "f32[]" = torch.ops.aten.neg.default(neg_2)
        mul_5: "f32[]" = torch.ops.aten.mul.Tensor(neg_3, div_7);  neg_3 = div_7 = None
        div_8: "f32[]" = torch.ops.aten.div.Tensor(neg_2, select);  neg_2 = None
        pow_4: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        mul_6: "f32[]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_7: "f32[]" = torch.ops.aten.mul.Tensor(div_8, mul_6);  div_8 = mul_6 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        mul_8: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, add_1);  add_1 = None
        mul_9: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, add_2);  tangents_1 = add_2 = None
        mul_10: "f32[]" = torch.ops.aten.mul.Tensor(mul_8, primals_2)
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        add_6: "f32[]" = torch.ops.aten.add.Tensor(mul_7, mul_10);  mul_7 = mul_10 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        mul_11: "f32[]" = torch.ops.aten.mul.Tensor(mul_8, 3.0);  mul_8 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        add_7: "f32[]" = torch.ops.aten.add.Tensor(mul_5, mul_11);  mul_5 = mul_11 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        div_10: "f32[]" = torch.ops.aten.div.Tensor(div, select_1);  div = None
        neg_4: "f32[]" = torch.ops.aten.neg.default(mul_9)
        mul_12: "f32[]" = torch.ops.aten.mul.Tensor(neg_4, div_10);  neg_4 = div_10 = None
        div_11: "f32[]" = torch.ops.aten.div.Tensor(mul_9, select_1);  mul_9 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:13 in forward, code: r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        add_8: "f32[]" = torch.ops.aten.add.Tensor(add_7, div_11);  add_7 = div_11 = None
        add_9: "f32[]" = torch.ops.aten.add.Tensor(add_6, mul_12);  add_6 = mul_12 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:12 in forward, code: r1 = x3 * (w0 ** 2 + w1 ** 2)
        mul_13: "f32[]" = torch.ops.aten.mul.Tensor(add_5, primals_2);  add_5 = primals_2 = None
        pow_5: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select_1, 1.0);  select_1 = None
        mul_14: "f32[]" = torch.ops.aten.mul.Scalar(pow_5, 2.0);  pow_5 = None
        mul_15: "f32[]" = torch.ops.aten.mul.Tensor(mul_13, mul_14);  mul_14 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:12 in forward, code: r1 = x3 * (w0 ** 2 + w1 ** 2)
        add_10: "f32[]" = torch.ops.aten.add.Tensor(add_9, mul_15);  add_9 = mul_15 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:12 in forward, code: r1 = x3 * (w0 ** 2 + w1 ** 2)
        pow_6: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select, 1.0);  select = None
        mul_16: "f32[]" = torch.ops.aten.mul.Scalar(pow_6, 2.0);  pow_6 = None
        mul_17: "f32[]" = torch.ops.aten.mul.Tensor(mul_13, mul_16);  mul_13 = mul_16 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:12 in forward, code: r1 = x3 * (w0 ** 2 + w1 ** 2)
        add_11: "f32[]" = torch.ops.aten.add.Tensor(add_8, mul_17);  add_8 = mul_17 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:11 in forward, code: w1 = self.w[1]
        full_default: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        
        # No stacktrace found for following nodes
        select_scatter_default: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, add_10, 0, 1);  add_10 = None
        select_scatter_default_1: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, add_11, 0, 0);  full_default = add_11 = None
        
        # File: /home/azor/projects/Sber-ML/model-debug.py:10 in forward, code: w0 = self.w[0]
        add_12: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        return [add_12, None, None, None]
        