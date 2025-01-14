import importlib
from dgl.data import DGLDataset
from .base_dataset import BaseDataset
from .utils import load_acm, load_acm_raw, generate_random_hg
from .academic_graph import AcademicDataset
from .hgb_dataset import HGBDataset
from .ohgb_dataset import OHGBDataset
from .gtn_dataset import *
from .adapter import AsLinkPredictionDataset, AsNodeClassificationDataset

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New dataset types can be added to cogdl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, BaseDataset):
            raise ValueError("Dataset ({}: {}) must extend cogdl.data.Dataset".format(name, cls.__name__))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_task_dataset(task):
    if task not in DATASET_REGISTRY:
        if task in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[task])
        else:
            print(f"Failed to import {task} dataset.")
            return False
    return True


hgbl_datasets = ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']
hgbn_datasets = ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB']
kg_lp_datasets = ['wn18', 'FB15k', 'FB15k-237']
ohgbl_datasets = ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2', 'ohgbl-Freebase']
ohgbn_datasets = ['ohgbn-Freebase', 'ohgbn-yelp2', 'ohgbn-acm', 'ohgbn-imdb']


def build_dataset(dataset, task, *args, **kwargs):
    if isinstance(dataset, DGLDataset):
        return dataset
    if dataset in CLASS_DATASETS:  # 如果使用dplp4gtn，acm4gtn和imdb4gtn，用这个函数读取数据集。
        return build_dataset_v2(dataset, task)
    if not try_import_task_dataset(task):
        exit(1)
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        _dataset = 'rdf_' + task
    elif dataset in ['acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm_han', 'acm_han_raw', 'acm4HeCo', 'dblp',
                     'dblp4MAGNN', 'imdb4MAGNN', 'imdb4GTN', 'acm4NARS', 'demo_graph', 'yelp4HeGAN', 'DoubanMovie',
                     'Book-Crossing', 'amazon4SLICE', 'MTWM', 'HNE-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']:
        _dataset = 'hin_' + task
    elif dataset in ohgbn_datasets + ohgbn_datasets:
        _dataset = 'ohgb_' + task
    elif dataset in ['ogbn-mag']:
        _dataset = 'ogbn_' + task
    elif dataset in hgbn_datasets:
        _dataset = 'HGBn_node_classification'
    elif dataset in hgbl_datasets:
        _dataset = 'HGBl_link_prediction'
    elif dataset in kg_lp_datasets:
        assert task == 'link_prediction'
        _dataset = 'kg_link_prediction'
    elif dataset in ['LastFM4KGCN']:
        _dataset = 'kgcn_recommendation'
    elif dataset in ['yelp4rec']:
        _dataset = 'hin_' + task
    elif dataset == 'demo':
        _dataset = 'demo_' + task
    return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'])


SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
    "link_prediction": "openhgnn.dataset.LinkPredictionDataset",
    "recommendation": "openhgnn.dataset.RecommendationDataset"
}

from .NodeClassificationDataset import NodeClassificationDataset
from .LinkPredictionDataset import LinkPredictionDataset
from .RecommendationDataset import RecommendationDataset


def build_dataset_v2(dataset, task):  # GTN数据集
    if dataset in CLASS_DATASETS:
        path = ".".join(CLASS_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)  # openhgnn.dataset 算是一个module   动态调起这个module
        class_name = CLASS_DATASETS[dataset].split(".")[-1]  # 再获取要加载的数据集对应的类名
        dataset_class = getattr(module, class_name)  # 动态调起数据集对应的类
        d = dataset_class()  # 构造一个对应数据集类的对象
        if task == 'node_classification':
            target_ntype = getattr(d, 'category')
            if target_ntype is None:
                target_ntype = getattr(d, 'target_ntype')
            res = AsNodeClassificationDataset(d, target_ntype=target_ntype)  # 拿到数据集，还要把它的格式进行一些转化，以便于适应节点分类任务。 添加了mask之类的
            if d.name == 'imdb4GTN' or d.name == 'acm4GTN':
                res.meta_paths_dict = d.meta_paths_dict
        elif task == 'link_prediction':
            target_link = getattr(d, 'target_link')
            target_link_r = getattr(d, 'target_link_r')
            res = AsLinkPredictionDataset(d, target_link=target_link, target_link_r=target_link_r)
        return res


CLASS_DATASETS = {
    "dblp4GTN": "openhgnn.dataset.DBLP4GTNDataset",
    "acm4GTN": "openhgnn.dataset.ACM4GTNDataset",
    "imdb4GTN": "openhgnn.dataset.IMDB4GTNDataset",
}

__all__ = [
    'BaseDataset',
    'NodeClassificationDataset',
    'LinkPredictionDataset',
    'RecommendationDataset',
    'AcademicDataset',
    'HGBDataset',
    'OHGBDataset',
    'GTNDataset',
    'AsLinkPredictionDataset',
    'AsNodeClassificationDataset'
]

classes = __all__
