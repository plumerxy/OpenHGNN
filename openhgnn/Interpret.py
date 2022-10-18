import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import lime
import lime.lime_tabular
from tqdm import tqdm


class Saliency(object):
    """
    计算每个基于元路径的同质图的节点重要性
    从feature_name中获得待计算梯度的特征矩阵变量，并用hook函数绑定获取中间梯度
    """

    def __init__(self, flow, feature_name):
        self.flow = flow
        self.net = flow.model
        self.feature_name = feature_name  # 要求梯度的变量名
        self.feature = []
        self.gradient = []
        self.net.eval()
        self.handlers = []

        self.output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx]
        self._register_hook()

    def _hook_func(self, grad):
        self.gradient.append(grad)

    def _register_hook(self):
        for name, var in vars(self.net).items():
            if name == self.feature_name:
                self.feature = var
        for i in range(len(self.feature)):
            self.handlers.append(self.feature[i].register_hook(self._hook_func))

    def gen_exp(self, idx):
        """
        为测试集中第idx个样本生成节点重要性解释，默认为概率最高的那个类生成解释
        Parameters
        ----------
        idx: 第idx个样本

        Returns 具有梯度的节点的index，以及相应的重要性。
                idx_list, grad_list  shape  ---> (num_channels, num_important_nodes)
        -------

        """
        self.net.zero_grad()
        self.gradient = []
        # self.handlers = []

        # output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx][
            # idx]  # 测试集中第0个节点的分类score
        output = self.output[idx]
        index = np.argmax(output.data.numpy())  # 得分最高类别的index
        target = output[index]  # 对该类别的得分求梯度
        # self._register_hook()
        target.backward(retain_graph=True)
        grad_list = []
        idx_list = []
        for grad in self.gradient:
            nonzeroindex = torch.nonzero(grad.sum(axis=1)).view(-1)
            idx_list.append(nonzeroindex)
            grad_list.append(grad[nonzeroindex].sum(axis=1))
        grad_list = torch.stack(grad_list)
        idx_list = torch.stack(idx_list)
        grad_list = normalize(grad_list, p=2.0, dim=1)  # 这个获得的就是正则化后的节点重要性

        return idx_list.detach().numpy(), grad_list.detach().numpy()


def divide_weights(weights, hidden_dim, cal_type="sum"):
    """
    将node embedding全部分量的重要性，整合为每个同质图的权重
    Parameters
    ----------
    weights: ndarray, (1, hidden_dim * num_channels)
    hidden_dim: 每个node embedding的维度
    cal_type: 单个同质图的权重通过sum还是mean的方式获得

    Returns pos_w: 正类权重 neg_w: 负类权重
    -------

    """
    pos_w = np.zeros((len(weights)))
    pos_w[np.argwhere(weights > 0)] = weights[np.argwhere(weights > 0)]
    neg_w = np.zeros((len(weights)))
    neg_w[np.argwhere(weights < 0)] = weights[np.argwhere(weights < 0)]
    if cal_type == "sum":
        pos_w = np.sum(pos_w.reshape((-1, hidden_dim)), axis=1)
        neg_w = np.sum(neg_w.reshape((-1, hidden_dim)), axis=1)
    elif cal_type == "mean":
        pos_w = np.mean(pos_w.reshape((-1, hidden_dim)), axis=1)
        neg_w = np.mean(neg_w.reshape((-1, hidden_dim)), axis=1)
    else:
        print("error cal type.")
    return pos_w, neg_w


class GradCAM(object):
    """
    类似grad cam的方法计算每个基于元路径的node embedding的重要性
    """

    def __init__(self, flow, net, layer_name):
        self.flow = flow
        self.net = net
        self.layer_name = layer_name
        self.feature = []
        self.gradient = []
        self.net.eval()
        self.handlers = []
        self._register_hook()

        self.test = flow.model.X_[flow.test_idx].detach().numpy()  # 测试集集中 GCN的输出embedding

    def _forward_hook_func(self, model, input, output):
        # input就是node embedding 元组只有一个tensor--(num_node x 1024) 从中取出所有预测类型节点
        self.feature = input[0]
        # self.feature.append(output)

    def _backward_hook_func(self, model, input_grad, output_grad):
        self.gradient = input_grad[1]
        # self.gradient.append(output_grad)

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:  # linear1
                self.handlers.append(module.register_forward_hook(self._forward_hook_func))
                self.handlers.append(module.register_backward_hook(self._backward_hook_func))

    def gen_exp(self, idx):
        """
        为测试集中第idx个样本生成解释，默认为概率最高的那个类生成解释
        Parameters
        ----------
        idx: 第idx个样本

        Returns tuple(pos_w, neg_w) pos_w: 正类权重 neg_w: 负类权重
        -------

        """
        self.net.zero_grad()
        output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx][
            idx]  # 测试集中第0个节点的分类score
        index = np.argmax(output.data.numpy())  # 得分最高类别的index
        target = output[index]  # 对该类别的得分求梯度
        target.backward()
        weight = self.gradient[self.net.category_idx[self.flow.test_idx[idx]]].detach().numpy()
        feature = self.feature[self.net.category_idx[self.flow.test_idx[idx]]].detach().numpy()

        return divide_weights(weight, self.flow.args.hidden_dim, cal_type="mean")


class Lime(object):
    """
    通过lime的方法，为每个基于元路径的node embedding计算重要性
    """

    def __init__(self, flow):
        """
        处理数据集，生成explainer
        flow为对应的trainer flow
        """
        # y_pred = flow.model(flow.hg, flow.model.input_feature())[flow.category][flow.train_idx].detach().numpy()  # 训练集的分类结果yc
        train = flow.model.X_[flow.train_idx].detach().numpy()  # 训练集中 GCN的输出embedding
        test = flow.model.X_[flow.test_idx].detach().numpy()  # 测试集集中 GCN的输出embedding
        feature_names = ["mp" + str(channel) + '_' + str(dim) for channel in range(1, flow.args.num_channels + 1) for
                         dim in range(1, flow.args.hidden_dim + 1)]  # embedding中每个特征的名称
        class_names = ["class" + str(i) for i in range(flow.num_classes)]  # 类别名称
        """解释生成"""
        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                                           discretize_continuous=False)  # lime解释器
        self.test = test
        self.explainer = explainer
        self.flow = flow

    def gen_exp(self, idx):
        """
        为测试集中第idx个样本生成解释，默认为概率最高的那个类生成解释
        Parameters
        ----------
        idx: 第idx个样本

        Returns tuple(pos_w, neg_w) pos_w: 正类权重 neg_w: 负类权重
        -------

        """
        exp = self.explainer.explain_instance(self.test[idx], self.flow.model.predict_lime,
                                              num_features=self.flow.args.hidden_dim * self.flow.args.num_channels,
                                              top_labels=1)  # 生成单个实例的解释
        local_exp = exp.local_exp[list(exp.local_exp.keys())[0]]  # 特征重要性列表
        weight = np.empty((len(local_exp)))  # 转化为numpy (1, hidden_dim*num_channels)
        for i, attr in local_exp:
            weight[i] = attr
        return divide_weights(weight, self.flow.args.hidden_dim)  # w_pos, w_neg


class InterpretEvaluator(object):
    """
    测试解释方法的指标
    """

    def __init__(self, interpreter):
        """
        要求interpreter里面要包含flow和test 以便于获取测试时候的参数
        Parameters
        ----------
        interpreter
        """
        self.interpreter = interpreter
        self.flow = interpreter.flow
        self.test = interpreter.test

    def test_metrics(self, counts):
        """
        自己写的那个m1和m2指标的测试方法。
        要求interpreter有gen_exp的方法，暂时没有写成继承，如果后期要整合代码，再做基类吧
        Parameters
        ----------
        counts 利用前多少个sample计算该指标

        Returns 指标m1和m2, m1是原始概率-最重要元路径覆盖后概率   m2是原始概率-最不重要元路径覆盖后概率
        -------

        """
        m1_list = []
        m2_list = []
        print("calculate test metrics for %s" % self.interpreter.__class__.__name__)
        for test_idx in tqdm(range(counts)):
            w_pos, w_neg = self.interpreter.gen_exp(test_idx)
            prob = self.flow.model.predict_lime(self.test[test_idx].reshape(1, -1))
            max_idx = np.argmax(w_pos)
            sample = self.test[test_idx].copy()
            sample[max_idx * self.flow.args.hidden_dim: (max_idx + 1) * self.flow.args.hidden_dim] = 0
            prob_max = self.flow.model.predict_lime(sample.reshape(1, -1))
            m1_list.append(np.max(prob, axis=1) - prob_max[0][np.argmax(prob)])
            min_idx = np.argmin(w_pos)
            sample = self.test[test_idx].copy()
            sample[min_idx * self.flow.args.hidden_dim: (min_idx + 1) * self.flow.args.hidden_dim] = 0
            prob_min = self.flow.model.predict_lime(sample.reshape(1, -1))
            m2_list.append(np.max(prob, axis=1) - prob_min[0][np.argmax(prob)])
        m1 = np.mean(m1_list)
        m2 = np.mean(m2_list)
        return m1, m2


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = GraphConv(10, 8, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(8, 4, norm='both', weight=True, bias=True)
        self.linear1 = nn.Linear(4, 2)

    def forward(self, feat, g):
        e1 = self.conv1(g, feat)
        e2 = self.conv2(g, e1)
        y = self.linear1(e2)
        feat.register_hook(self.hook_func)
        return y

    def forward_hook_func(self, model, input, output):
        self.input = input
        self.output = output

    def backward_hook_func(self, model, input_grad, output_grad):
        self.input_grad = input_grad
        self.output_grad = output_grad

    def hook_func(self, grad):
        self.grad = grad


if __name__ == "__main__":
    import dgl
    import numpy as np
    import torch as th
    from dgl.nn import GraphConv

    g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
    g = dgl.add_self_loop(g)
    feat = th.rand(6, 10, requires_grad=True)
    model = Test()
    model.eval()
    model.zero_grad()
    model.conv1.register_forward_hook(model.forward_hook_func)
    model.conv1.register_backward_hook(model.backward_hook_func)
    y = model(feat, g)
    y_scalar = y[4][0]
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
