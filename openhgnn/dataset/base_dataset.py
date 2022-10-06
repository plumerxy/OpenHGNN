from abc import ABC, ABCMeta, abstractmethod
from dgl.data.utils import load_graphs


class BaseDataset(ABC):  # ABC是抽象基类 表明basedataset是一个抽象类
    def __init__(self, *args, **kwargs):
        super(BaseDataset, self).__init__()
        self.logger = kwargs['logger']
        self.g = None
        self.meta_paths = None
        self.meta_paths_dict = None

    def load_graph_from_disk(self, file_path):
        """
        load graph from disk and the file path of graph is generally stored in ``./openhgnn/dataset/``.

        从graph.bin中读取异质图，获得一个dgl.hetrograph对象。
        从这里看，load dataset的时候，是已经处理好的二进制格式的图数据了。

        Parameters
        ----------
        file_path: the file path storing the graph.bin

        Returns
        -------
        g: dgl.DGLHetrograph
        """
        g, _ = load_graphs(file_path)
        return g[0]


