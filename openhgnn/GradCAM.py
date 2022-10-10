import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM(object):
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = []
        self.gradient = []
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _forward_hook_func(self, model, input, output):
        # input就是node embedding 元组只有一个tensor--(num_node x 1024) 从中取出所有预测类型节点
        self.feature = input[0]

    def _backward_hook_func(self, model, input_grad, output_grad):
        self.gradient = input_grad[1]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:  # linear1
                self.handlers.append(module.register_forward_hook(self._forward_hook_func))
                self.handlers.append(module.register_backward_hook(self._backward_hook_func))

    def grad_cam(self, flow, idx):
        self.net.zero_grad()
        output = self.net(flow.hg, flow.model.input_feature())[flow.category][flow.test_idx][idx]  # 测试集中第0个节点的分类score
        index = np.argmax(output.data.numpy())  # 得分最高类别的index
        target = output[index]  # 对该类别的得分求梯度
        target.backward()
        weight = self.gradient[self.net.category_idx[flow.test_idx[idx]]]
        feature = self.feature[self.net.category_idx[flow.test_idx[idx]]]
        return weight, feature


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 4)

    def forward(self, X):
        X = self.linear1(X)
        X = F.relu(X)
        y = self.linear2(X)
        return y

    def forward_hook_func(self, model, input, output):
        self.input = input
        self.output = output

    def backward_hook_func(self, model, input_grad, output_grad):
        self.input_grad = input_grad
        self.output_grad = output_grad


if __name__ == "__main__":
    X = torch.ones(1, 1024, requires_grad=True)
    model = Test()
    model.eval()
    model.zero_grad()
    model.linear1.register_forward_hook(model.forward_hook_func)
    model.linear1.register_backward_hook(model.backward_hook_func)
    y = model(X)
    y_scalar = y[0][1]
    y_scalar.backward()
    print("done")

    # test = Test()
    # test.state_dict()["linear1.weight"].copy_(flow.model.state_dict()["linear1.weight"])  # 把model的参数拷贝进来
    # test.state_dict()["linear1.bias"].copy_(flow.model.state_dict()["linear1.bias"])
    # test.state_dict()["linear2.weight"].copy_(flow.model.state_dict()["linear2.weight"])
    # test.state_dict()["linear2.bias"].copy_(flow.model.state_dict()["linear2.bias"])
    # test_embedding = flow.model.X_[flow.test_idx]  # 测试集集中 GCN的输出embedding
    # X = test_embedding[0].clone().detach().requires_grad_()
    # X = X.view(1, -1)
    # # X = torch.ones(1, 1024, requires_grad=True)
    # test.eval()
    # test.zero_grad()
    # test.linear1.register_forward_hook(test.forward_hook_func)
    # test.linear1.register_backward_hook(test.backward_hook_func)
    # y = test(X)
    # y_scalar = y[0][1]
    # y_scalar.backward()
