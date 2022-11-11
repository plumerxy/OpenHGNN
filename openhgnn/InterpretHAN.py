import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import lime
import lime.lime_tabular
from tqdm import tqdm


def divide_weights(allWeights, cal_type="sum"):
    """
    将node embedding全部分量的重要性，整合为每个同质图的权重
    对于每个元路径来说，先将全部节点的embedding的梯度求和，成为一个embedding的梯度形式，再考虑把这个embedding的梯度求和还是求平均值。
    Parameters
    ----------
    allWeights: ndarray, (num_metapath, hidden_dim)
    hidden_dim: 每个node embedding的维度
    cal_type: 单个同质图的权重通过sum还是mean的方式获得

    Returns pos_w: 正类权重 neg_w: 负类权重
    -------

    """
    pos_list = []
    neg_list = []
    for weights in allWeights:
        hidden_dim = len(weights[0])
        weights = weights.reshape(-1)
        pos_w = np.zeros((len(weights)))
        pos_w[np.argwhere(weights > 0)] = weights[np.argwhere(weights > 0)]
        neg_w = np.zeros((len(weights)))
        neg_w[np.argwhere(weights < 0)] = weights[np.argwhere(weights < 0)]
        pos_w = pos_w.reshape(-1, hidden_dim)
        neg_w = neg_w.reshape(-1, hidden_dim)
        if cal_type == "sum":
            pos_w = np.sum(pos_w.reshape((-1, hidden_dim)), axis=1)
            neg_w = np.sum(neg_w.reshape((-1, hidden_dim)), axis=1)
        elif cal_type == "mean":
            pos_w = np.mean(np.sum(pos_w, axis=0))
            neg_w = np.mean(np.sum(neg_w, axis=0))
        else:
            print("error cal type.")
        pos_list.append(pos_w)
        neg_list.append(neg_w)
    pos_w = normalize(torch.tensor(pos_list).view(1, -1), p=2.0).numpy()
    neg_w = normalize(torch.tensor(neg_list).view(1, -1), p=2.0).numpy()
    return pos_w, neg_w


class Lime(object):
    """
    通过lime的方法，为每个基于元路径的node embedding计算重要性
    """

    def __init__(self, flow, layer_name="layers.1.model.mods", **kwargs):
        """
        处理数据集，生成explainer
        flow为对应的trainer flow
        """
        self.flow = flow
        self.layer_name = layer_name
        self.handlers = []
        self.meta_path = []  # 还会记录相应的元路径名称
        self.X_ = []
        self.X_dict = {}
        self._register_hook()
        self.flow.model.eval()

        if len(kwargs) != 0:
            y_pred = flow.model(flow.hg, flow.model.input_feature(), **kwargs)
        else:
            y_pred = flow.model(flow.hg, flow.model.input_feature())
        self.X_ = [self.X_dict[mp] for mp in self.meta_path]
        self.X_ = np.hstack(self.X_)
        for h in self.handlers:
            h.remove()
        train = self.X_[flow.train_idx]  # 训练集中 GCN的输出embedding
        test = self.X_[flow.test_idx]  # 测试集集中 GCN的输出embedding
        feature_names = [mp + '_' + str(dim) for mp in self.meta_path for dim in
                         range(1, (flow.args.hidden_dim * flow.args.num_heads[-1]) + 1)]  # embedding中每个特征的名称
        class_names = ["class" + str(i) for i in range(flow.num_classes)]  # 类别名称

        """解释生成"""
        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                                           discretize_continuous=False)  # lime解释器
        self.test = test
        self.explainer = explainer

    def _forward_hook_func(self, model, input, output):
        for mp in self.meta_path:
            if model is self.flow.model.layers[-1].model.mods[mp]:
                self.X_dict[mp] = output.flatten(1).detach().numpy()

    def _register_hook(self):
        for (name, module) in self.flow.model.named_modules():
            if name == self.layer_name:  # layers.1.model.mods
                for meta_path, mod in module.items():
                    self.meta_path.append(meta_path)
                    self.handlers.append(mod.register_forward_hook(self._forward_hook_func))

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
                                              num_features=len(self.test[idx]),
                                              top_labels=1)  # 生成单个实例的解释
        local_exp = exp.local_exp[list(exp.local_exp.keys())[0]]  # 特征重要性列表
        weight = np.empty((len(local_exp)))  # 转化为numpy (1, hidden_dim*num_channels)
        for i, attr in local_exp:
            weight[i] = attr
        weight = weight.reshape(-1, self.flow.args.hidden_dim * self.flow.args.num_heads[-1])
        weight = [weight[i].reshape(1, -1) for i in range(len(weight))]
        return divide_weights(weight)  # w_pos, w_neg


class GradCAM(object):
    """
    类似grad cam的方法计算每个基于元路径的node embedding的重要性

    确保meta_path和gradient 以及最终权重的顺序是一一对应的
    """

    def __init__(self, flow, net, layer_name):
        self.flow = flow
        self.net = net
        self.layer_name = layer_name
        self.feature = []
        self.feature_dict = {}
        self.gradient = []
        self.gradient_dict = {}
        self.net.eval()
        self.handlers = []
        self.meta_path = []  # 还会记录相应的元路径名称

        self._register_hook()  # 这里会先记录下来元路径的名字，和GATlayer那个model dict的key是对应的。
        y_pred = flow.model(flow.hg, flow.model.input_feature())
        self.feature = [self.feature_dict[mp] for mp in self.meta_path]
        self.test = np.hstack(self.feature)[flow.test_idx]
        # self.test = flow.model.X_[flow.test_idx].detach().numpy()  # 测试集集中 GCN的输出embedding

    def _forward_hook_func(self, model, input, output):
        for mp in self.meta_path:
            if model is self.flow.model.layers[-1].model.mods[mp]:
                self.feature_dict[mp] = output.flatten(1).detach().numpy()

    def _backward_hook_func(self, model, input_grad, output_grad):
        for mp in self.meta_path:
            if model is self.flow.model.layers[-1].model.mods[mp]:
                self.gradient_dict[mp] = output_grad[0].flatten(1).detach().numpy()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:  # layers.1.model.mods
                for meta_path, mod in module.items():
                    self.meta_path.append(meta_path)
                    self.handlers.append(mod.register_forward_hook(self._forward_hook_func))
                    self.handlers.append(mod.register_backward_hook(self._backward_hook_func))

    def gen_exp(self, idx):
        """
        为测试集中第idx个样本生成解释，默认为概率最高的那个类生成解释

        确保meta_path和gradient 以及最终权重的顺序是一一对应的
        Parameters
        ----------
        idx: 第idx个样本

        Returns tuple(pos_w, neg_w) pos_w: 正类权重 neg_w: 负类权重
        -------

        """
        self.gradient = []
        self.gradient_dict = {}
        self.feature = []
        self.feature_dict = {}
        self.net.zero_grad()
        output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx][
            idx]  # 测试集中第idx个节点的分类score
        index = np.argmax(output.data.numpy())  # 得分最高类别的index
        target = output[index]  # 对该类别的得分求梯度
        target.backward()
        self.gradient = [self.gradient_dict[mp] for mp in self.meta_path]
        pos_w, neg_w = divide_weights(self.gradient, cal_type="mean")

        return pos_w, neg_w


class Saliency(object):
    """
    计算每个基于元路径的同质图的节点重要性，以及总节点重要性
    从feature_name中获得待计算梯度的特征矩阵变量，并用hook函数绑定获取中间梯度
    """

    def __init__(self, flow, layer_name):
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

        self._register_hook()
        self.output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx]

    def _forward_hook_func(self, model, input, output):
        for mp in self.meta_path:
            if model is self.flow.model.layers[0].model.mods[mp]:
                self.feature_dict[mp] = input

    def _backward_hook_func(self, model, input_grad, output_grad):
        for mp in self.meta_path:
            if model is self.flow.model.layers[0].model.mods[mp]:
                self.gradient_dict[mp] = input_grad[0].flatten(1).detach()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:  # layers.0.model.mods
                for meta_path, mod in module.items():
                    self.meta_path.append(meta_path)
                    self.handlers.append(mod.register_forward_hook(self._forward_hook_func))
                    self.handlers.append(mod.register_backward_hook(self._backward_hook_func))

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
        self.net.zero_grad()
        self.gradient_dict = {}
        self.gradient = []

        # output = self.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][self.flow.test_idx][
        # idx]  # 测试集中第0个节点的分类score
        output = self.output[idx]
        index = np.argmax(output.data.numpy())  # 得分最高类别的index
        target = output[index]  # 对该类别的得分求梯度
        # self._register_hook()
        target.backward(retain_graph=True)
        self.gradient = [self.gradient_dict[mp] for mp in self.meta_path]
        grad_list = []
        idx_list = []
        grad_sum = torch.zeros_like(self.gradient[0])  # 用于存储所有同质图的总梯度
        for grad in self.gradient:  # 对于每一个基于元路径的同质图
            nonzeroindex = torch.nonzero(grad.sum(axis=1)).view(-1)  # 哪个节点有梯度
            idx_list.append(nonzeroindex)
            grad_list.append(grad[nonzeroindex].sum(axis=1))  # 所有的节点特征维度上求和 作为该节点的重要性
            grad_sum = grad_sum + grad
        grad_list = torch.stack(grad_list)
        idx_list = torch.stack(idx_list)
        grad_list = normalize(grad_list, p=2.0, dim=1)  # 这个获得的就是正则化后的节点重要性

        nonzeroindex = torch.nonzero(grad_sum.sum(axis=1)).view(-1)
        grad_sum = grad_sum[nonzeroindex].sum(axis=1).view(1, -1)
        grad_sum = normalize(grad_sum, p=2.0, dim=1)

        return nonzeroindex.detach().numpy(), grad_sum.detach().numpy(), idx_list.detach().numpy(), grad_list.detach().numpy()


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
        print("calculate test metrics for %s" % self.interpreter.__class__.__name__)
        for test_idx in tqdm(range(counts)):
            w_pos, w_neg = self.interpreter.gen_exp(test_idx)
            prob = self.flow.model.predict_lime(self.test[test_idx].reshape(1, -1))
            max_idx = np.argmax(w_pos)
            sample = self.test[test_idx].copy()
            sample[max_idx * self.flow.args.hidden_dim * self.flow.args.num_heads[-1]: (
                                                                                               max_idx + 1) * self.flow.args.hidden_dim *
                                                                                       self.flow.args.num_heads[-1]] = 0
            prob_max = self.flow.model.predict_lime(sample.reshape(1, -1))
            m1_list.append(np.max(prob, axis=1) - prob_max[0][np.argmax(prob)])
            min_idx = np.argmin(w_pos)
            sample = self.test[test_idx].copy()
            sample[min_idx * self.flow.args.hidden_dim * self.flow.args.num_heads[-1]: (
                                                                                               min_idx + 1) * self.flow.args.hidden_dim *
                                                                                       self.flow.args.num_heads[-1]] = 0
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
            # original_idx = self.flow.model.category_idx[self.flow.test_idx[test_idx]].item()  # 测试样本在全部节点中的idx
            original_inside_idx = np.where(chosen_idx == self.flow.test_idx[test_idx])[
                0]  # 判断待预测节点本身在不在梯度前几中 如果是的话返回其索引
            if original_inside_idx.size == 1:  # 待预测的节点本身不能被遮掉吧
                chosen_idx = np.delete(chosen_idx, original_inside_idx)
            # 遮掉相应节点的特征，置为0
            h = self.flow.model.input_feature()
            # ntype_dict = {}
            # sum = 0
            # for ntype in self.flow.hg.ntypes:
            #     ntype_dict[ntype] = np.arange(sum, sum + self.flow.hg.num_nodes(ntype))
            #     sum = sum + self.flow.hg.num_nodes(ntype)
            for i, index in enumerate(chosen_idx):
                h[self.flow.category][index] = 0
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

    def node_importance_metric(self, counts, layer_name='layers.1.model.mods'):
        """
        计算各channel的节点重要性的评估指标，将最重要的前5个节点（除样本本身外）的特征全部置零，测量预测概率的改变情况
        Parameters
        ----------
        counts 利用前多少个sample计算该指标

        Returns list1 概率改变情况, list2重要性改变情况, 每个channel的评估结果 len = num_channels
        -------

        """
        metrics1 = [[] for i in range(len(self.interpreter.meta_path))]
        metrics2 = [[] for i in range(len(self.interpreter.meta_path))]
        for test_idx in tqdm(range(counts)):
            __, __, idx, grads = self.interpreter.gen_exp(test_idx)  # 获取节点总梯度tot_grad
            original_idx = self.flow.test_idx[test_idx]  # 测试样本在全部预测节点中的idx
            for i in range(len(grads)):
                max_n_idx = np.flipud(np.argsort(grads[i]))[:5]  # 梯度最大的n个梯度在grads[i]中的index
                grad = grads[i][max_n_idx]
                id = idx[i][max_n_idx]  # 梯度最大的前n个节点在整个图中的index
                original_inside_idx = np.where(id == original_idx)[0]  # 判断待预测节点本身在不在梯度前几中 如果是的话返回其索引
                if original_inside_idx.size == 1:  # 待预测的节点本身不能被遮掉吧
                    id = np.delete(id, original_inside_idx)
                # 计算指标1
                output_ori = \
                    self.interpreter.net(self.flow.hg, self.flow.model.input_feature())[self.flow.category][
                        self.flow.test_idx][
                        test_idx]
                prob_ori = torch.nn.functional.softmax(output_ori, dim=0)
                output = self.interpreter.net(self.flow.hg, self.flow.model.input_feature(), mode="eva",
                                              mp=self.interpreter.meta_path[i],
                                              idx=id)[self.flow.category][self.flow.test_idx][test_idx]
                prob = torch.nn.functional.softmax(output, dim=0)
                metrics1[i].append((torch.max(prob_ori) - prob[torch.argmax(prob_ori)]).item())
                # 计算指标2
                lm = Lime(self.flow, layer_name=layer_name)
                w_pos, w_neg = lm.gen_exp(test_idx)
                lm = Lime(self.flow, layer_name=layer_name, mode="eva", mp=self.interpreter.meta_path[i], idx=id)
                new_w_pos, new_w_neg = lm.gen_exp(test_idx)
                metrics2[i].append(w_pos[0][i] - new_w_pos[0][i])
        return [np.mean(m) for m in metrics1], [np.mean(m) for m in metrics2]
