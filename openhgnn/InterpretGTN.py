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
        self.test = None

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
        为*测试集*中第idx个样本生成节点重要性解释，默认为概率最高的那个类生成解释
        包含总节点重要性和不同同质图的节点重要性
        Parameters
        ----------
        idx: 第idx个样本

        Returns 具有梯度的节点的index，以及相应的重要性。
                nonzeroindex, grad_sum:   shape ---> num_important_nodes
                idx_list, grad_list:  shape  ---> (num_channels, num_important_nodes)
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
        grad_sum = torch.zeros_like(self.gradient[0])
        for grad in self.gradient:
            nonzeroindex = torch.nonzero(grad.sum(axis=1)).view(-1)  # 把非零的节点记下来
            idx_list.append(nonzeroindex)
            grad_list.append(grad[nonzeroindex].sum(axis=1))  # 对应的节点重要性
            grad_sum = grad_sum + grad  # 总节点重要性（梯度方法加起来就是总重要性了）
        grad_list = torch.stack(grad_list)
        idx_list = torch.stack(idx_list)
        grad_list = normalize(grad_list, p=2.0, dim=1)  # 这个获得的就是正则化后的节点重要性

        nonzeroindex = torch.nonzero(grad_sum.sum(axis=1)).view(-1)
        grad_sum = grad_sum[nonzeroindex].sum(axis=1).view(1, -1)
        grad_sum = normalize(grad_sum, p=2.0, dim=1)

        return nonzeroindex.detach().numpy(), grad_sum.detach().numpy(), idx_list.detach().numpy(), grad_list.detach().numpy()


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
    pos_w = normalize(torch.tensor(pos_w).view(1, -1), p=2.0).numpy()
    neg_w = normalize(torch.tensor(neg_w).view(1, -1), p=2.0).numpy()
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

        y_pred = flow.model(flow.hg, flow.model.input_feature())
        self.test = flow.model.X_[
            flow.test_idx].detach().numpy()  # 测试集集中 GCN的输出embedding  TODO test是这个吗？需要看一下，评估方法的时候看这个吧
        self._register_hook()

    def _forward_hook_func(self, model, input, output):
        # input就是node embedding 元组只有一个tensor--(num_node x 1024) 从中取出所有预测类型节点
        self.feature = input[0]
        # self.feature.append(output)

    def _backward_hook_func(self, model, input_grad, output_grad):
        self.gradient = input_grad[1]  # 需要确定梯度是不是这个
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
        weight = self.gradient[self.net.category_idx[self.flow.test_idx[idx]]].detach().numpy()  # 注意考虑index的问题
        feature = self.feature[self.net.category_idx[self.flow.test_idx[idx]]].detach().numpy()

        return divide_weights(weight, self.flow.args.hidden_dim, cal_type="mean")


class Lime(object):
    """
    通过lime的方法，为每个基于元路径的node embedding计算重要性
    """

    def __init__(self, flow, **kwargs):
        """
        处理数据集，生成explainer
        flow为对应的trainer flow
        """
        if len(kwargs) != 0:
            y_pred = flow.model(flow.hg, flow.model.input_feature(), mode=kwargs['mode'], channel=kwargs['channel'],
                                idx=kwargs['idx'])
        else:
            y_pred = flow.model(flow.hg, flow.model.input_feature())
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
        return divide_weights(weight, self.flow.args.hidden_dim, cal_type="mean")  # w_pos, w_neg


class Rise(object):
    """
       计算每个基于元路径的同质图的节点重要性，以及总节点重要性
       利用RISE方法，随机采样，黑盒方法
    """

    def __init__(self, flow, layer_name, N, p, threshold):
        self.flow = flow
        self.net = flow.model
        self.layer_name = layer_name  # 要求梯度的变量名

        self.feature_dict = {}
        self.gradient_dict = {}
        self.feature = []
        self.gradient = []
        self.handlers = []
        self.net.eval()
        self.test = None
        self.meta_path = []
        self.coalesced_graph = {}
        self.coalesced_graph_node_idx = {}  # 分元路径的，涉及到的邻居节点的index
        self.coalesced_graph_node_idx_all = torch.tensor([])  # 全部的邻居节点index
        self.N = N
        self.p = p
        self.threshold = threshold

        self.output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx]

    def generate_masks(self, idx):
        """
        生成掩码，按照定义好的元路径和卷积层数决定节点掩码的大小
        通过元路径先生成同质图，再找到n阶的邻居，就是计算图，计算图即为节点掩码的范围
        （如果有使用异质节点的，之后可以考虑再扩展）
        p: 掩码为1的概率
        N: 生成掩码的数量
        idx: 测试集中的第idx个节点

        Returns
        node_idx: 对应节点的索引
        node_masks: 对应节点的掩码
        """
        num_convs = 1  # 卷积层数 GTN这里目前都是1
        # self.meta_path = self.flow.args.meta_paths_dict  # GTN是自动生成同质图的方法，用不到元路径的概念

        # 利用GTN自动产生的基于元路径的同质图 并找到全部邻居节点




        for mp, mp_value in self.meta_path.items():
            self.coalesced_graph[mp] = dgl.metapath_reachable_graph(self.flow.hg, mp_value)
            idx = torch.where(self.coalesced_graph[mp].ndata["test_mask"] == 1)[0][
                idx]  # 该同质图中待测试节点的index（不是全图，是该类节点中的index）
            self.coalesced_graph_node_idx[mp] = torch.tensor([idx])  # 找邻居节点
            for i in range(num_convs):
                graph = self.coalesced_graph[mp].sample_neighbors(self.coalesced_graph_node_idx[mp], -1)
                self.coalesced_graph_node_idx[mp] = torch.cat(
                    (graph.edges()[0], self.coalesced_graph_node_idx[mp])).unique()
            self.coalesced_graph_node_idx_all = torch.cat(
                (self.coalesced_graph_node_idx[mp], self.coalesced_graph_node_idx_all)).unique()

        # 将全部邻居节点生成节点掩码
        num_of_nodes = len(self.coalesced_graph_node_idx_all)
        node_masks = (np.random.rand(self.N, num_of_nodes) < self.p).astype(float)  # 概率1-p置为0
        return self.coalesced_graph_node_idx_all, node_masks

    def gen_exp(self, idx):
        """
        为测试集中第idx个样本生成节点重要性解释，默认为概率最高的那个类生成解释
        包含总节点重要性和不同同质图的节点重要性
        Parameters
        ----------
        idx: 第idx个样本

        Returns 具有梯度的节点的index，以及相应的重要性。
                nonzeroindex, grad_sum:   shape ---> num_important_nodes
                idx_list, grad_list:  shape  ---> (num_channels, num_important_nodes)
        -------

        """
        self.meta_path = []
        self.coalesced_graph = {}
        self.coalesced_graph_node_idx = {}  # 分元路径的，涉及到的邻居节点的index
        self.coalesced_graph_node_idx_all = torch.tensor([])  # 全部的邻居节点index

        # 生成节点掩码，根据计算图生成 (目前所采到的，应该都是和待预测节点同一个类别的节点，扰动特征矩阵的时候也只考虑同类别节点即可）
        node_idx, node_masks = self.generate_masks(idx)
        node_masks_non = (node_masks == 0).astype(float)  # 对mask取反 看被遮掉特征的重要性
        # 进行扰动
        feat_masked = self.flow.model.input_feature()[self.flow.category].unsqueeze(0).repeat(self.N, 1,
                                                                                              1)  # 对特征矩阵进行扰动，这里只涉及到待预测类型节点的特征矩阵
        node_masks_all = (torch.zeros((self.N, len(feat_masked[0]))).double()  # 需要扩展成完整的节点掩码以对特征矩阵进行扰动
                          .scatter_(1, node_idx.unsqueeze(0).repeat(self.N, 1).long(), torch.tensor(node_masks)))
        feat_masked = torch.mul(node_masks_all.unsqueeze(2), feat_masked)
        # 获取原概率 self.output
        prob_init = F.softmax(self.output[idx])  # 原始概率
        # 计算扰动后的结果以计算权重
        feat_dict = self.flow.model.input_feature().copy()
        weights = []
        for feat_m in tqdm(feat_masked):
            feat_dict[self.flow.category] = feat_m.float()
            output = self.net(self.flow.hg, feat_dict)[self.flow.category][self.flow.test_idx][idx]
            prob = F.softmax(output)
            weight = prob_init - prob
            for i in range(len(weight)):
                if weight[i] <= self.threshold:
                    weight[i] = 0
            weights.append(weight)
            gc.collect()
        weights = torch.stack(weights).detach().numpy()
        attribution = (weights.T.dot(node_masks_non) / self.N / (1 - self.p))[torch.argmax(prob_init)]

        print("--------rise for inx {}, {} nodes important".format(idx, len(node_idx)))
        return node_idx.long().detach().numpy(), torch.tensor(attribution).unsqueeze(0).detach().numpy(), None, None


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
        counts: 利用前多少个sample计算该指标

        Returns 指标m1和m2, m1是原始概率-最重要元路径覆盖后概率   m2是原始概率-最不重要元路径覆盖后概率
        -------

        """
        m1_list = []
        m2_list = []
        print("\ncalculate test metrics for %s" % self.interpreter.__class__.__name__)
        for test_idx in tqdm(range(counts)):
            w_pos, w_neg = self.interpreter.gen_exp(test_idx)
            prob = self.flow.model.predict_lime(self.test[test_idx].reshape(1, -1))
            max_idx = np.argmax(w_pos)
            sample = self.test[test_idx].copy()  # test是基于元路径的node embedding，拿一个出来，直接通过它做预测看结果
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

    def tot_node_importance(self, counts):
        """
        计算总节点重要性的评估指标，将最重要的前5个节点（除样本本身外）的特征全部置零，测量预测概率的改变情况
        Parameters
        ----------
        counts: 利用前多少个sample计算该指标

        Returns metric, 原始概率-置零重要节点后的概率
        -------

        """
        metric = []
        print("calculate total node importance metrics for %s" % self.interpreter.__class__.__name__)
        for test_idx in tqdm(range(counts)):
            # 获取待遮节点的index
            tot_idx, tot_grad, __, __ = self.interpreter.gen_exp(test_idx)  # 获取节点总梯度tot_grad
            chosen_idx = tot_idx[np.flipud(np.argsort(tot_grad)[0])[:5]]
            original_idx = self.flow.model.category_idx[self.flow.test_idx[test_idx]].item()  # 测试样本在全部节点中的idx
            original_inside_idx = np.where(chosen_idx == original_idx)[0]  # 判断待预测节点本身在不在梯度前几中 如果是的话返回其索引
            if original_inside_idx.size == 1:  # 待预测的节点本身不能被遮掉吧
                chosen_idx = np.delete(chosen_idx, original_inside_idx)
            # 遮掉相应节点的特征，置为0
            h = self.flow.model.input_feature()
            ntype_dict = {}
            sum = 0
            for ntype in self.flow.hg.ntypes:
                ntype_dict[ntype] = np.arange(sum, sum + self.flow.hg.num_nodes(ntype))
                sum = sum + self.flow.hg.num_nodes(ntype)
            for i in chosen_idx:
                for ntype, index in ntype_dict.items():
                    if np.argwhere(index == i).size == 1:
                        h[ntype][np.argwhere(index == i)] = 0
            # 计算指标
            output_ori = \
                self.interpreter.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][
                    self.flow.test_idx][test_idx]
            prob_ori = torch.nn.functional.softmax(output_ori)
            output = self.interpreter.net(self.flow.hg, h)[self.flow.category][self.flow.test_idx][
                test_idx]  # 测试集中第0个节点的分类score
            prob = torch.nn.functional.softmax(output)
            metric.append((torch.max(prob_ori) - prob[torch.argmax(prob_ori)]).item())
        metric = np.mean(metric)
        return metric

    def node_importance_metric(self, counts):
        """
        计算各channel的节点重要性的评估指标，将最重要的前5个节点（除样本本身外）的特征全部置零，测量预测概率的改变情况
        Parameters
        ----------
        counts 利用前多少个sample计算该指标

        Returns list1 概率改变情况, list2重要性改变情况, 每个channel的评估结果 len = num_channels
        -------

        """
        metrics1 = [[] for i in range(self.flow.model.num_channels)]
        metrics2 = [[] for i in range(self.flow.model.num_channels)]
        for test_idx in tqdm(range(counts)):
            __, __, idx, grads = self.interpreter.gen_exp(test_idx)  # 获取节点总梯度tot_grad
            original_idx = self.flow.model.category_idx[self.flow.test_idx[test_idx]].item()  # 测试样本在全部节点中的idx
            for i in range(len(grads)):
                max_n_idx = np.flipud(np.argsort(grads[i]))[:5]  # 梯度最大的n个梯度在grads[i]中的index
                grad = grads[i][max_n_idx]
                id = idx[i][max_n_idx]  # 梯度最大的前n个节点在整个图中的index
                original_inside_idx = np.where(id == original_idx)[0]  # 判断待预测节点本身在不在梯度前几中 如果是的话返回其索引
                if original_inside_idx.size == 1:  # 待预测的节点本身不能被遮掉吧
                    id = np.delete(id, original_inside_idx)
                # 计算指标1
                output_ori = self.interpreter.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][
                    self.flow.test_idx][
                    test_idx]
                prob_ori = torch.nn.functional.softmax(output_ori, dim=0)
                output = self.interpreter.net(self.flow.hg, self.flow.model.input_feature(), mode="eva", channel=i,
                                              idx=id)[self.flow.category][self.flow.test_idx][test_idx]
                prob = torch.nn.functional.softmax(output, dim=0)
                metrics1[i].append((torch.max(prob_ori) - prob[torch.argmax(prob_ori)]).item())
                # 计算指标2
                lm = Lime(self.flow)
                w_pos, w_neg = lm.gen_exp(test_idx)
                lm = Lime(self.flow, mode='eva', channel=i, idx=id)
                new_w_pos, new_w_neg = lm.gen_exp(test_idx)
                metrics2[i].append(w_pos[0][i] - new_w_pos[0][i])
        return [np.mean(m) for m in metrics1], [np.mean(m) for m in metrics2]


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
