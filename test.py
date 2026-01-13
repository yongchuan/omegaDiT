import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
target = torch.randn(2, 3)  # 形状为(2,3)
target.requires_grad=True
y = x.expand_as(target)     # y形状变为(2,3)
loss = y.sum()
print(loss)
loss.backward()
print(x.grad)  # 输出: tensor([2., 2., 2.])，因为梯度在第0维累加
print(target.grad)
