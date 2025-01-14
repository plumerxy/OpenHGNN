import torch.nn as nn
import torch.nn.functional as F
import torch

import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv import MetapathConv
from ..utils.utils import extract_metapaths


@register_model('HAN')
class HAN(BaseModel):
    r"""
    This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph HAN from paper `Heterogeneous Graph Attention Network <https://arxiv.org/pdf/1903.07293.pdf>`__..
    Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
    could not reproduce the result in HAN as they did not provide the preprocessing code, and we
    constructed another dataset from ACM with a different set of papers, connections, features and
    labels.


    .. math::
        \mathbf{h}_{i}^{\prime}=\mathbf{M}_{\phi_{i}} \cdot \mathbf{h}_{i}

    where :math:`h_i` and :math:`h'_i` are the original and projected feature of node :math:`i`

    .. math::
        e_{i j}^{\Phi}=a t t_{\text {node }}\left(\mathbf{h}_{i}^{\prime}, \mathbf{h}_{j}^{\prime} ; \Phi\right)

    where :math:`{att}_{node}` denotes the deep neural network.

    .. math::
        \alpha_{i j}^{\Phi}=\operatorname{softmax}_{j}\left(e_{i j}^{\Phi}\right)=\frac{\exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{j}^{\prime}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}^{\Phi}} \exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{k}^{\prime}\right]\right)\right)}

    where :math:`\sigma` denotes the activation function, || denotes the concatenate
    operation and :math:`a_{\Phi}` is the node-level attention vector for meta-path :math:`\Phi`.

    .. math::
        \mathbf{z}_{i}^{\Phi}=\prod_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}^{\Phi}} \alpha_{i j}^{\Phi} \cdot \mathbf{h}_{j}^{\prime}\right)

    where :math:`z^{\Phi}_i` is the learned embedding of node i for the meta-path :math:`\Phi`.
    Given the meta-path set {:math:`\Phi_0 ,\Phi_1,...,\Phi_P`},after feeding node features into node-level attentionwe can obtain P groups of
    semantic-specific node embeddings, denotes as {:math:`Z_0 ,Z_1,...,Z_P`}.
    We use MetapathConv to finish Node-level Attention and Semantic-level Attention.


    Parameters
    ------------
    meta_paths : list
        contain multiple meta-paths.
    category : str
        The category means the head and tail node of metapaths.
    in_size : int
        input feature dimension.
    hidden_size : int
        hidden layer dimension.
    out_size : int
        output feature dimension.
    num_heads : int
        number of attention heads.
    dropout : float
        Dropout probability.

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths_dict is None:
            meta_paths = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths = args.meta_paths_dict
    
        return cls(meta_paths=meta_paths, category=args.out_node_type,
                    in_size=args.hidden_dim, hidden_size=args.hidden_dim,
                    out_size=args.out_dim,
                    num_heads=args.num_heads,
                    dropout=args.dropout)

    def __init__(self, meta_paths, category, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.category = category
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.linear = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.hidden_dim = hidden_size * num_heads[-1]
        self.meta_paths_num = len(meta_paths)

    def forward(self, g, h_dict, **kwargs):
        for i, gnn in enumerate(self.layers):
            if i == 0 and len(kwargs) != 0 and kwargs['mode'] == 'eva':  # 测试节点重要性，对h_dict进行节点置零操作
                h_dict = gnn(g, h_dict, **kwargs)
            else:
                h_dict = gnn(g, h_dict)
        out_dict = {ntype: self.linear(h_dict[ntype]) for ntype in self.category}
        
        return out_dict

    def get_emb(self, g, h_dict):
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: h.detach().cpu().numpy()}

    def predict_lime(self, data):
        """

        Parameters
        ----------
        data: numpy array，测试数据。 han的各同质图拼接起来输入进来，然后再拆开

        Returns 预测结果y
        -------

        """
        data = data.reshape(-1, self.meta_paths_num, self.hidden_dim)
        X_tensor = torch.from_numpy(data).float()
        X_tensor = [X_tensor[:, i] for i in range(len(X_tensor[0]))]
        embedding = self.layers[-1].model.SemanticConv(X_tensor)
        logits = self.linear(embedding)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().numpy()

class HANLayer(nn.Module):
    """
    HAN layer.

    Parameters
    -----------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        a cached graph
    _cached_coalesced_graph : list
        _cached_coalesced_graph list generated by *dgl.metapath_reachable_graph()*
    """
    def __init__(self, meta_paths_dict, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods,semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, **kwargs):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(  # 用于生成基于元路径的同质图
                        g, mp_value)
        h = self.model(self._cached_coalesced_graph, h, **kwargs)
        return h
