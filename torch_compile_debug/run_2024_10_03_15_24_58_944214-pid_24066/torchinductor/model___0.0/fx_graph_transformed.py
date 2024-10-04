class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2]", arg1_1: "f32[]", arg2_1: "f32[]", arg3_1: "f32[]", arg4_1: "f32[]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(select, select_1)
        add: "f32[]" = torch.ops.aten.add.Tensor(arg1_1, div)
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select, 3.0)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(arg1_1, select_1)
        add_1: "f32[]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        sub: "f32[]" = torch.ops.aten.sub.Tensor(select_1, arg2_1);  arg2_1 = None
        pow_1: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(pow_1, select);  pow_1 = None
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(arg1_1, div_1)
        div_2: "f32[]" = torch.ops.aten.div.Tensor(sub_1, arg3_1);  sub_1 = None
        div_3: "f32[]" = torch.ops.aten.div.Tensor(div_2, arg3_1);  div_2 = None
        neg: "f32[]" = torch.ops.aten.neg.default(arg4_1)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(neg, div_3);  neg = div_3 = None
        div_4: "f32[]" = torch.ops.aten.div.Tensor(arg4_1, arg3_1);  arg3_1 = None
        neg_1: "f32[]" = torch.ops.aten.neg.default(mul_2);  mul_2 = None
        add_2: "f32[]" = torch.ops.aten.add.Tensor(neg_1, arg4_1);  neg_1 = None
        neg_2: "f32[]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        div_5: "f32[]" = torch.ops.aten.div.Tensor(div_1, select);  div_1 = None
        neg_3: "f32[]" = torch.ops.aten.neg.default(neg_2)
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(neg_3, div_5);  neg_3 = div_5 = None
        div_6: "f32[]" = torch.ops.aten.div.Tensor(neg_2, select);  neg_2 = None
        pow_2: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        mul_4: "f32[]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_5: "f32[]" = torch.ops.aten.mul.Tensor(div_6, mul_4);  div_6 = mul_4 = None
        mul_6: "f32[]" = torch.ops.aten.mul.Tensor(arg4_1, add);  add = None
        mul_7: "f32[]" = torch.ops.aten.mul.Tensor(arg4_1, add_1);  arg4_1 = add_1 = None
        mul_8: "f32[]" = torch.ops.aten.mul.Tensor(mul_6, arg1_1)
        add_3: "f32[]" = torch.ops.aten.add.Tensor(mul_5, mul_8);  mul_5 = mul_8 = None
        mul_9: "f32[]" = torch.ops.aten.mul.Tensor(mul_6, 3.0);  mul_6 = None
        add_4: "f32[]" = torch.ops.aten.add.Tensor(mul_3, mul_9);  mul_3 = mul_9 = None
        div_7: "f32[]" = torch.ops.aten.div.Tensor(div, select_1);  div = None
        neg_4: "f32[]" = torch.ops.aten.neg.default(mul_7)
        mul_10: "f32[]" = torch.ops.aten.mul.Tensor(neg_4, div_7);  neg_4 = div_7 = None
        div_8: "f32[]" = torch.ops.aten.div.Tensor(mul_7, select_1);  mul_7 = None
        add_5: "f32[]" = torch.ops.aten.add.Tensor(add_4, div_8);  add_4 = div_8 = None
        add_6: "f32[]" = torch.ops.aten.add.Tensor(add_3, mul_10);  add_3 = mul_10 = None
        mul_11: "f32[]" = torch.ops.aten.mul.Tensor(add_2, arg1_1);  add_2 = arg1_1 = None
        pow_3: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select_1, 1.0);  select_1 = None
        mul_12: "f32[]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_13: "f32[]" = torch.ops.aten.mul.Tensor(mul_11, mul_12);  mul_12 = None
        add_7: "f32[]" = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None
        pow_4: "f32[]" = torch.ops.aten.pow.Tensor_Scalar(select, 1.0);  select = None
        mul_14: "f32[]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_15: "f32[]" = torch.ops.aten.mul.Tensor(mul_11, mul_14);  mul_11 = mul_14 = None
        add_8: "f32[]" = torch.ops.aten.add.Tensor(add_5, mul_15);  add_5 = mul_15 = None
        full: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_scatter_default: "f32[2]" = torch.ops.aten.select_scatter.default(full, add_7, 0, 1);  add_7 = None
        select_scatter_default_1: "f32[2]" = torch.ops.aten.select_scatter.default(full, add_8, 0, 0);  full = add_8 = None
        add_9: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        return [add_9, None, None, None]
        