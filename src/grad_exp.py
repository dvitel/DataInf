import torch
from torch.autograd.functional import jacobian

x = torch.tensor([[2,1]], requires_grad=True, dtype=torch.float32)
M = torch.tensor([[1,2],[3,4]], dtype=torch.float32)

y = torch.matmul(x, M)

jacobian()

y.backward(gradient=torch.tensor([[1,0]], dtype=torch.float32), retain_graph=True)
grad1 = x.grad.clone()
x.grad.zero_()
y.backward(gradient=torch.tensor([[0,1]], dtype=torch.float32), retain_graph=True)
grad2 = x.grad.clone()

y.backward(gradient=torch.tensor([[1,1]], dtype=torch.float32), retain_graph=True)
grad3 = x.grad.clone()
pass

# https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html

# We need jacobian 

# https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad
# https://pytorch.org/docs/stable/generated/torch.func.jacrev.html#torch.func.jacrev
# https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html 
# https://pytorch.org/docs/stable/generated/torch.func.jacfwd.html#torch.func.jacfwd