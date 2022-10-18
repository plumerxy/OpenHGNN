import os.path

import torch
from torch.nn.functional import normalize

from .config import Config
from .utils import set_random_seed, set_best_config, Logger
from .trainerflow import build_flow
from .auto import hpo_experiment
import lime
import lime.lime_tabular
import numpy as np
from openhgnn.Interpret import GradCAM
from openhgnn.Interpret import Lime
from openhgnn.Interpret import InterpretEvaluator
from openhgnn.Interpret import Saliency
import itertools

__all__ = ['Experiment']


class Experiment(object):
    r"""Experiment.

    Parameters
    ----------
    model : str or nn.Module
        Name of the model or a hetergenous gnn model provided by the user.
    dataset : str or DGLDataset
        Name of the model or a DGLDataset provided by the user.
    use_best_config: bool
        Whether to load the best config of specific models and datasets. Default: False
    load_from_pretrained : bool
        Whether to load the model from the checkpoint. Default: False
    hpo_search_space :
        Search space for hyperparameters.
    hpo_trials : int
        Number of trials for hyperparameter search.
    Examples
    --------
    >>> experiment = Experiment(model='RGCN', dataset='imdb4GTN', task='node_classification', gpu=-1)
    >>> experiment.run()
    """

    default_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    specific_trainerflow = {
        'HetGNN': 'hetgnntrainer',
        'HGNN_AC': 'node_classification_ac',
        'NSHE': 'nshetrainer',
        'HeCo': 'HeCo_trainer',
        'DMGI': 'DMGI_trainer',
        'KGCN': 'kgcntrainer',
        'Metapath2vec': 'mp2vec_trainer',
        'HERec': 'herec_trainer',
        'SLiCE': 'slicetrainer',
        'HeGAN': 'HeGAN_trainer',
        'HDE': 'hde_trainer',
        'GATNE-T': 'GATNE_trainer',
        'TransE': 'TransX_trainer',
        'TransH': 'TransX_trainer',
        'TransR': 'TransX_trainer',
        'TransD': 'TransX_trainer',
    }
    immutable_params = ['model', 'dataset', 'task']

    def __init__(self, model, dataset, task,
                 gpu: int = -1,
                 use_best_config: bool = False,
                 load_from_pretrained: bool = False,
                 hpo_search_space=None,
                 hpo_trials: int = 100,
                 output_dir: str = "./openhgnn/output",
                 conf_path: str = default_conf_path,
                 **kwargs):
        self.config = Config(file_path=conf_path, model=model, dataset=dataset, task=task, gpu=gpu)
        self.config.model = model
        self.config.dataset = dataset
        self.config.task = task
        self.config.gpu = gpu
        self.config.use_best_config = use_best_config
        # self.config.use_hpo = use_hpo
        self.config.load_from_pretrained = load_from_pretrained
        self.config.output_dir = os.path.join(output_dir, self.config.model_name)
        # self.config.seed = seed
        self.config.hpo_search_space = hpo_search_space
        self.config.hpo_trials = hpo_trials

        if not getattr(self.config, 'seed', False):
            self.config.seed = 0
        if use_best_config:
            self.config = set_best_config(self.config)
        self.set_params(**kwargs)
        print(self)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            assert key not in self.immutable_params
            self.config.__setattr__(key, value)

    def run(self):
        """ run the experiment """
        self.config.logger = Logger(self.config)
        set_random_seed(self.config.seed)
        trainerflow = self.specific_trainerflow.get(self.config.model,
                                                    self.config.task)  # 如果这个model没有特定的trainerflow，默认task就是对应flow的名字啦。
        if self.config.hpo_search_space is not None:
            # hyper-parameter search
            hpo_experiment(self.config, trainerflow)
        else:
            # ------- 训练部分 -------- #
            flow = build_flow(self.config,
                              trainerflow)  # 所有可用的trainerflow类会被注册到一个字典中，根据config所设定的flow名，构造一个相应的trainerflow对象。
            # result = flow.train()  # 训练一个分类器
            flow.model = torch.load("gtn.pth")
            """test"""
            sl = Saliency(flow, "h_list")
            idx, grad = sl.gen_exp(0)
            idx1, grad1 = sl.gen_exp(1)
            idx2, grad2 = sl.gen_exp(2)
            # ------- lime元路径解释 ------ #
            # exp = self.metapath_interpret_lime(flow)
            lm = Lime(flow)  # 初始化lime explainer
            w_pos, w_neg = lm.gen_exp(0)  # 查看第i个测试用例的正负权重
            eva = InterpretEvaluator(lm)  # 初始化可解释性评估器
            m1, m2 = eva.test_metrics(2000)  # 计算指标
            print("-------------lime for 2000 samples-----------")
            print("m1 metric: " + str(m1))
            print("m2 metric: " + str(m2))
            # ------- grad-cam元路径解释 ------
            gc = GradCAM(flow, flow.model, "linear1")
            w_pos, w_neg = gc.gen_exp(0)
            eva = InterpretEvaluator(gc)
            m1, m2 = eva.test_metrics(2000)
            print("-------------grad-cam for 2000 samples-----------")
            print("m1 metric: " + str(m1))
            print("m2 metric: " + str(m2))
            # ------- GTN的权重 ----------- #
            # metapath, metapath_weight = self.gtn_metapath_weight(flow)
            # return result
    def gtn_metapath_weight(self, flow):
        """ GTN生成的每个同质图的具体元路径权重计算"""
        # y_pred = flow.model(flow.hg, flow.model.input_feature())[flow.category][
        #     flow.train_idx].detach().numpy()  # 训练集的分类结果yc
        etypes_dict = flow.hg.canonical_etypes.copy()  # 边类型
        etypes_dict.append(('', '', ''))
        conv_weights = []  # conv 的权重
        for i in range(flow.model.num_layers):
            conv_weights.append(flow.model.layers[i].conv1.filter.detach().numpy().T)
            if i == 0:
                conv_weights.append(flow.model.layers[i].conv2.filter.detach().numpy().T)
        metapath_weights = []  # 记录有效元路径的权重
        metapath = []  # 记录有效元路径
        for i, wi in enumerate(conv_weights[0]):
            for j, wj in enumerate(conv_weights[1]):
                for k, wk in enumerate(conv_weights[2]):
                    if (etypes_dict[i][2] == etypes_dict[j][0]) and (etypes_dict[j][2] == etypes_dict[k][0]):
                        str_m = etypes_dict[i][1] + '-' + etypes_dict[j][2] + '-' + etypes_dict[k][2]
                        if str_m in metapath:
                            metapath_weights[self.index_metapath(str_m, metapath)] += wi * wj * wk
                        else:
                            metapath.append(str_m)
                            metapath_weights.append(wi * wj * wk)
                    elif (etypes_dict[i][0] == '') and (etypes_dict[j][2] == etypes_dict[k][0]):
                        str_m = etypes_dict[j][1] + '-' + etypes_dict[k][2]
                        if str_m in metapath:
                            metapath_weights[self.index_metapath(str_m, metapath)] += wi * wj * wk
                        else:
                            metapath.append(str_m)
                            metapath_weights.append(wi * wj * wk)
                    elif (etypes_dict[j][0] == '') and (etypes_dict[i][2] == etypes_dict[k][0]):
                        str_m = etypes_dict[i][1] + '-' + etypes_dict[k][2]
                        if str_m in metapath:
                            metapath_weights[self.index_metapath(str_m, metapath)] += wi * wj * wk
                        else:
                            metapath.append(str_m)
                            metapath_weights.append(wi * wj * wk)
                    elif (etypes_dict[k][0] == '') and (etypes_dict[i][2] == etypes_dict[j][0]):
                        str_m = etypes_dict[i][1] + '-' + etypes_dict[j][2]
                        if str_m in metapath:
                            metapath_weights[self.index_metapath(str_m, metapath)] += wi * wj * wk
                        else:
                            metapath.append(str_m)
                            metapath_weights.append(wi * wj * wk)
        del metapath[len(metapath)-1]
        del metapath_weights[len(metapath)-1]

        """保存到csv"""
        # import pandas as pd
        # import openpyxl
        # dt = pd.DataFrame(metapath_weights, index=metapath)
        # dt.to_excel("meta_path.xlsx")
        return metapath, metapath_weights


    def index_metapath(self, str_m, metapath):
        """查找重复的元路径，返回index"""
        for i, str in enumerate(metapath):
            if str_m == str:
                return i
        return -1



    def __repr__(self):
        basic_info = '------------------------------------------------------------------------------\n' \
                     ' Basic setup of this experiment: \n' \
                     '     model: {}    \n' \
                     '     dataset: {}   \n' \
                     '     task: {}. \n' \
                     ' This experiment has following parameters. You can use set_params to edit them.\n' \
                     ' Use print(experiment) to print this information again.\n' \
                     '------------------------------------------------------------------------------\n'. \
            format(self.config.model_name, self.config.dataset_name, self.config.task)
        params_info = ''
        for attr in dir(self.config):
            if '__' not in attr and attr not in self.immutable_params:
                params_info += '{}: {}\n'.format(attr, getattr(self.config, attr))
        return basic_info + params_info
