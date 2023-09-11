import torch


def hook_func(grad):
    print("middle grad: %f" % grad)


# a = torch.tensor(1.0, requires_grad=True)
# b = a.clone()
# e = a.clone()
#
# c = a * 2
# c.backward()
# print("a.grad: %f" % a.grad)
#
# d = b * 3
# b.register_hook(hook_func)
# d.backward()
# print("a.grad: %f" % a.grad)
#
# f = e * 4
# e.register_hook(hook_func)
# f.backward()
# print("a.grad: %f" % a.grad)


a = torch.tensor([1.0], requires_grad=True)
b = a.clone()
e = a.clone()
b[0] = 2

c = a * 2
c.backward()
print("a.grad: %f" % a.grad)

d = b * 3
b.register_hook(hook_func)
d.backward()
print("a.grad: %f" % a.grad)

f = e * 4
e.register_hook(hook_func)
f.backward()
print("a.grad: %f" % a.grad)
