import os.path

import torch

from .config import Config
from .utils import set_random_seed, set_best_config, Logger
from .trainerflow import build_flow
from .auto import hpo_experiment
import lime
import lime.lime_tabular
import numpy as np
from openhgnn.GradCAM import GradCAM

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
        trainerflow = self.specific_trainerflow.get(self.config.model, self.config.task)  # 如果这个model没有特定的trainerflow，默认task就是对应flow的名字啦。
        if self.config.hpo_search_space is not None:
            # hyper-parameter search
            hpo_experiment(self.config, trainerflow)
        else:
            # ------- 训练部分 -------- #
            flow = build_flow(self.config, trainerflow)  # 所有可用的trainerflow类会被注册到一个字典中，根据config所设定的flow名，构造一个相应的trainerflow对象。
            # result = flow.train()  # 训练一个分类器
            flow.model = torch.load("gtn.pth")
            # ------- lime元路径解释 ------ #
            # self.metapath_interpret_lime(flow)
            # ------- grad-cam元路径解释 ------ #
            gc = GradCAM(flow.model, "linear2")  # grad-cam对象
            gc.grad_cam(flow)


            # return result

    def metapath_interpret_lime(self, flow):
        # ------- 解释部分  lime v1.0 -------- #
        """数据处理"""
        # y_pred = flow.model(flow.hg, flow.model.input_feature())[flow.category][flow.train_idx].detach().numpy()  # 训练集的分类结果yc
        train = flow.model.X_[flow.train_idx].detach().numpy()  # 训练集中 GCN的输出embedding
        test = flow.model.X_[flow.test_idx].detach().numpy()  # 测试集集中 GCN的输出embedding
        feature_names = ["mp" + str(channel) + '_' + str(dim) for channel in range(1, flow.args.num_channels + 1) for
                         dim in range(1, flow.args.hidden_dim + 1)]  # embedding中每个特征的名称
        class_names = ["class" + str(i) for i in range(flow.num_classes)]  # 类别名称
        """解释生成"""
        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                                           discretize_continuous=False)  # lime解释器
        # print(flow.model.predict_lime(test))
        # m1 = 原始正类概率-影响最大元路径置零后正类概率   m2 = 原始正类概率-影响最小元路径置零后正类概率
        m1_list = []
        m2_list = []
        for test_idx in range(20):
            print("----------------------")
            print("test id: " + str(test_idx))
            exp = explainer.explain_instance(test[test_idx], flow.model.predict_lime,
                                             num_features=flow.args.hidden_dim * flow.args.num_channels,
                                             top_labels=1)  # 生成单个实例的解释
            local_exp = exp.local_exp[list(exp.local_exp.keys())[0]]  # 特征重要性列表
            local_exp_sum_pos = [0 for i in range(flow.args.num_channels)]  # 为每个元路径计算正类的总重要性
            local_exp_sum_neg = [0 for i in range(flow.args.num_channels)]  # 负类的重要性
            for i, attr in local_exp:
                if attr > 0:
                    local_exp_sum_pos[int(i / flow.args.hidden_dim)] = local_exp_sum_pos[
                                                                           int(i / flow.args.hidden_dim)] + attr
                else:
                    local_exp_sum_neg[int(i / flow.args.hidden_dim)] = local_exp_sum_neg[
                                                                           int(i / flow.args.hidden_dim)] + attr
            """处理解释 计算一个简单指标"""
            prob = flow.model.predict_lime(test[test_idx].reshape(1, -1))
            print("init prob: " + str(prob))
            # 影响最大的元路径embedding直接置零
            max_pos_index = local_exp_sum_pos.index(max(local_exp_sum_pos))
            max_neg_index = local_exp_sum_neg.index(min(local_exp_sum_neg))
            test_new_pos = test[test_idx].copy()
            test_new_pos[max_pos_index * 128: (max_pos_index + 1) * 128] = 0
            test_new_neg = test[test_idx].copy()
            test_new_neg[max_neg_index * 128: (max_neg_index + 1) * 128] = 0
            prob_new_pos = flow.model.predict_lime(test_new_pos.reshape(1, -1))
            prob_new_neg = flow.model.predict_lime(test_new_neg.reshape(1, -1))
            print("delete the most important metapath for positive class: " + str(prob_new_pos))
            print("delete the most important metapath for negative class: " + str(prob_new_neg))
            m1_list.append(np.max(prob, axis=1) - prob_new_pos[0][np.argmax(prob)])  # m1指标计算
            # 影响最小的元路径embedding直接置零
            min_pos_index = local_exp_sum_pos.index(min(local_exp_sum_pos))
            min_neg_index = local_exp_sum_neg.index(max(local_exp_sum_neg))
            test_new_pos = test[test_idx].copy()
            test_new_pos[min_pos_index * 128: (min_pos_index + 1) * 128] = 0
            test_new_neg = test[test_idx].copy()
            test_new_neg[min_neg_index * 128: (min_neg_index + 1) * 128] = 0
            prob_new_pos = flow.model.predict_lime(test_new_pos.reshape(1, -1))
            prob_new_neg = flow.model.predict_lime(test_new_neg.reshape(1, -1))
            print("delete the less important metapath for positive class: " + str(prob_new_pos))
            print("delete the less important metapath for negative class: " + str(prob_new_neg))
            m2_list.append(np.max(prob, axis=1) - prob_new_pos[0][np.argmax(prob)])  # m2指标计算
        m1 = np.mean(m1_list)
        m2 = np.mean(m2_list)
        print("-------------")
        print("m1 metric: " + str(m1))
        print("m2 metric: " + str(m2))


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

