import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from ..utils import transform_relation_graph_list
from . import BaseModel, register_model


@register_model('GTN')
class GTN(BaseModel):
    r"""
        GTN from paper `Graph Transformer Networks <https://arxiv.org/abs/1911.06455>`__
        in NeurIPS_2019. You can also see the extension paper `Graph Transformer
        Networks: Learning Meta-path Graphs to Improve GNNs <https://arxiv.org/abs/2106.06218.pdf>`__.

        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        # add self-loop edge
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels,
                   in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim,
                   num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm,
                 identity):
        super(GTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels  # 这个是什么？哦哦，好像是自动生成元路径的个数
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity  # 好像是，生成元路径的时候，要不要乘同一矩阵   按照GTN的概念，那多了一个类型的同质图，确实是多了一种类型的边，所以前面边数量要加一
                                    # 总之就是GTN模型需要用到的各种参数
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)  # 以列表的形式保持多个子模块（GTN layer模块）
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu)  # 同质图GCN图神经网络  这个GraphConv应该是只有一个GCNlayer
        self.norm = EdgeWeightNorm(norm='right')  # 给邻接矩阵做正则化用的
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)  # embedding拼接起来，输出是一个embedding的大小
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)   # 再来一个线性层进行分类  （linear层应该是没有激活函数的）
        self.category_idx = None
        self.A = None
        self.h = None
        self.X_ = None  # 用于存储GCN生成的基于元路径的embedding

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):  # 对于每个channel
            g = H[i]  # gtlayer生成的图
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def predict_lime(self, data):
        """

        Parameters
        ----------
        data: numpy array，测试数据。GCN的输出embedding

        Returns 预测结果y
        -------

        """
        X_tensor = torch.from_numpy(data).float()
        X_1 = self.linear1(X_tensor)
        X_2 = F.relu(X_1)
        logits = self.linear2(X_2)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().numpy()

    def forward(self, hg, h):
        with hg.local_scope():  # 这个意思是，对hg的数据的操作，都限制在这个局部内，当离开这个局部，这些改变没有发生。
            hg.ndata['h'] = h
            # * =============== Extract edges in original graph  从原始图中提取边的过程 ================
            if self.category_idx is None:  # 这个一开始就是空的
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category,
                                                                             identity=self.identity)
                # A是所有同质图的列表，h是所有节点的特征，category_idx是预测要用的节点的相应idx
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            # X_ = self.gcn(g, self.h)
            A = self.A
            # * =============== Get new graph structure  GTN的部分 GTLayers ================
            for i in range(self.num_layers):  # num_layers是GTlayer的层数
                if i == 0:
                    H, W = self.layers[i](A)  # W是conv层的相应权重，H是新生成图的list，一共有channel个
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm == True:
                    H = self.normalization(H)  # 做正则化，其实对w_sum做正则化就是对邻接矩阵的正则化，加权矩阵的权重就是写在邻接矩阵中的嘛
                # Ws.append(W)
            # * =============== GCN Encoder  然后就该往GCN里面输入了 ================
            for i in range(self.num_channels):  # 对于每个channel分别操作
                g = dgl.remove_self_loop(H[i])  # 去除图中指向节点自己的边
                edge_weight = g.edata['w_sum']  # 获取每个边的权重
                g = dgl.add_self_loop(g)  # 又重新把指向自己的边加上了？？？GCN应该是要用到？
                edge_weight = th.cat((edge_weight, th.full((g.number_of_nodes(),), 1, device=g.device)))  # 把新加上的这些边的权重添加到edge_weight中，这些权重均为1
                edge_weight = self.norm(g, edge_weight)  # 又把图正则化了一下
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)  # 带入GCN 开始学习
                else:
                    X_ = th.cat((X_, self.gcn(g, h, edge_weight=edge_weight)), dim=1)  # 在这里直接把所有的embedding拼接起来了 不是最后才拼接的
            self.X_ = X_[self.category_idx]  # 加一句，获取待预测节点的embedding输入 1x1024(8x128)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}  # 挑出待预测类型节点的预测结果  作为一个字典返回


class GTLayer(nn.Module):
    r"""
        CTLayer multiply each combination adjacency matrix :math:`l` times to a :math:`l-length`
        meta-paths adjacency matrix.

        The method to generate :math:`l-length` meta-path adjacency matrix can be described as:

        .. math::
            A_{(l)}=\Pi_{i=1}^{l} A_{i}

        where :math:`A_{i}` is the combination adjacency matrix generated by GT conv.

        Parameters
        ----------
            in_channels: int
                The input dimension of GTConv which is numerically equal to the number of relations.
            out_channels: int
                The input dimension of GTConv which is numerically equal to the number of channel in GTN.
            first: bool
                If true, the first combination adjacency matrix multiply the combination adjacency matrix.

    """
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)  # 计算一个用各种类型边加权出来的图
            result_B = self.conv2(A)  # 计算一个用各种类型边加权出来的图
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]  # 把卷积时候用到的权重都保存下来
            # detach()函数用于剥离梯度，也是tensor，数值一样，但是没有梯度了
        else:
            result_A = H_
            result_B = self.conv1(A)  # 只卷一个 另一个是已经卷出来的图作为input
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):  # 对每个channel
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class GTConv(nn.Module):
    r"""
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\left(A ; W_{c}\right)=\sum_{R_{i} \in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \subseteq \mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        # 卷积层的前向过程 卷出一个全部边类型的加权图
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)  # 每一次，生成一个权重，每个边的重要性，加起来为1
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):  # 对于每个通道
            for j, g in enumerate(A):  # j是index，g是对应的边矩阵
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]  # 对每个类型边矩阵，赋予边权重
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results
