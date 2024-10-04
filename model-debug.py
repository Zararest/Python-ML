import torch
from torch import nn

class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.950, 0.288], dtype=torch.float32))

    def forward(self, x1, x2, x3):
        w0 = self.w[0]
        w1 = self.w[1]
        r1 = x3 * (w0 ** 2 + w1 ** 2)
        r2 = (x3 + w0 / w1) * (3.0 * w0 + x3 * w1)
        r3 = x3 - (w1 - x2) ** 2 / w0
        return r1 + r2 + r3 / (x1 - x2 - r1)

    def your_forward_backward(self, x1, x2, x3):
        w0 = self.w[0]
        w1 = self.w[1]

        # forward
        t1 = w0 ** 2
        t2 = w1 ** 2
        t3 = t1 + t2
        t1 = t2 = None
        t4 = t3 * x3 #r1
        t3 = None

        t5 = w1 * x3
        t6 = 3.0 * w0
        t7 = t5 + t6
        t5 = t6 = None
        t8 = w0 / w1
        t9 = x3 + t8
        t8 = None
        t10 = t7 * t9 #r2
        
        t11 = w1 - x2 
        t12 = t11 ** 2
        t13 = t12 / w0 
        t14 = x3 - t13 #r3
        t13 = None

        t15 = x1 - x2
        t16 = t15 - t4
        t15 = None
        t17 = t14 / t16
        t18 = t17 + t10
        t17 = None
        t10 = None
        t19 = t18 + t4 #res
        t4 = t18 = None

        # backward
        #dw0
        dt8dw0 = 1.0 / w1
        dt9dw0 = dt8dw0
        dt8dw0 = None
        dt10dt9 = t7
        t7 = None
        dt6dw0 = 3.0
        dt7dt6 = 1.0 
        dt5dw0 = 0.0
        dt7dt5 = 1.0
        dt7dw0 = dt7dt5 * dt5dw0 + dt7dt6 * dt6dw0
        dt10dt7 = t9 
        t9 = None
        dt10dw0 = dt10dt7 * dt7dw0 + dt10dt9 * dt9dw0
        dt9dw0 = dt7dw0 = None

        dt11dw0 = 0.0
        dt12dt11 = 2.0 * t11
        t11 = None
        dt12dw0 = dt12dt11 * dt11dw0
        dt13dt12 = 1.0 / w0
        dt13dw0 = dt13dt12 * dt12dw0 - t12 / (w0 ** 2)
        t12 = dt12dw0 = None
        dt14dt13 = -1.0
        dt14dw0 = dt14dt13 * dt13dw0
        dt13dw0 = None
        dt17dt14 = 1.0 / t16

        dt3dw0 = 2.0 * w0
        dt4dt3 = x3
        dt4dw0 = dt4dt3 * dt3dw0
        dt3dw0 = None
        dt16dt4 = -1.0
        dt15dw0 = 0.0
        dt16dt15 = 1.0
        dt16dw0 = dt16dt15 * dt15dw0 + dt16dt4 * dt4dw0
        dt17dt16 = - t14 / (t16 ** 2)
        t14 = t16 = None
        dt17dw0 = dt17dt16 * dt16dw0 + dt17dt14 * dt14dw0
        dt14dw0 = dt16dw0 = None
        dw0 = dt17dw0 + dt10dw0 + dt4dw0 
        dt10dw0 = dt4dw0 = dt17dw0 = None

        #dw1
        dt8dw1 = w0 * (- 1 / (w1 ** 2))
        dt9dw1 = dt8dw1
        dt8dw1 = None

        dt6dw1 = 0.0
        dt5dw1 = x3
        dt7dw1 = dt7dt5 * dt5dw1 + dt7dt6 * dt6dw1
        dt10dw1 = dt10dt7 * dt7dw1 + dt10dt9 * dt9dw1
        dt10dt9 = dt9dw1 = dt7dw1 = None

        dt3dw1 = 2.0 * w1
        dt4dw1 = dt4dt3 * dt3dw1
        dt3dw1 = None
        dt15dw1 = 0.0
        dt16dw1 = dt16dt15 * dt15dw1 + dt16dt4 * dt4dw1

        dt11dw1 = 1.0
        dt12dw1 = dt12dt11 * dt11dw1
        dt13dw1 = dt13dt12 * dt12dw1
        dt13dt12 = dt12dw1 = None
        dt14dw1 = dt14dt13 * dt13dw1
        dt13dw1 = None
        dt17dw1 = dt17dt14 * dt14dw1 + dt17dt16 * dt16dw1
        dt17dt14 = dt17dt16 = dt16dw1 = dt14dw1 = None
        dw1 = dt17dw1 + dt10dw1 + dt4dw1
        dt10dw1 = dt4dw1 = dt17dw1 = None

        self.w.grad = torch.stack([dw0, dw1])

        return t19
    
model = Graph()
compiled_model = torch.compile(model)

x1, x2, x3 = torch.rand(3)
compiled_model.zero_grad()
y_torch = compiled_model(x1, x2, x3)
y_torch.backward()
grad_torch = compiled_model.w.grad.clone()


compiled_model.zero_grad()
with torch.no_grad():
    y_manual = compiled_model.your_forward_backward(x1, x2, x3)
grad_manual = compiled_model.w.grad.clone()

assert torch.allclose(y_manual, y_torch, rtol=5e-05, atol=1e-7)
assert torch.allclose(grad_manual, grad_torch, rtol=5e-05, atol=1e-7)
