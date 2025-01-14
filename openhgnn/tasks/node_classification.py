import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator

@register_task("node_classification")
class NodeClassification(BaseTask):
    r"""
    Node classification tasks.

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric


    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """
    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'node_classification', logger=self.logger)  # task中包含数据集，先搭建数据集（如果到时候想用不同的数据集，不适配的时候可以从这里debug进去看看）
        # self.evaluator = Evaluator()
        self.logger = args.logger
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split()  # 划分数据集用的mask也存在task中
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()  # 节点分类任务所需label
        self.multi_label = self.dataset.multi_label
        
        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            if args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
                self.evaluation_metric = 'acc'
            else:
                self.evaluation_metric = 'f1'  # 节点分类任务，默认使用f1score作为评估指标。 如果需要别的，应该可以指定吧，写在evaluator中就行，

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy  # 分类任务默认loss func是cross entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        if self.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to('cpu')
            
        if self.evaluation_metric == 'acc':
            acc = self.evaluator.cal_acc(self.labels[mask], pred)
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'acc-ogbn-mag':
            from ogb.nodeproppred import Evaluator
            evaluator = Evaluator(name='ogbn-mag')
            logits = logits.unsqueeze(dim=1)
            input_dict = {"y_true": logits, "y_pred": self.labels[self.test_idx]}
            result_dict = evaluator.eval(input_dict)
            return result_dict
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], pred)  # 图节点分类的评估指标已经写在evaluator中了，到时候我的指标也可以加在evaluator中吧
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == 'f1_lr':
            micro_f1, macro_f1 = self.evaluator.nc_with_LR(logits, self.labels, self.train_idx, self.test_idx)
            return dict(Macro_f1=macro_f1, Mirco_f1=micro_f1)
    
    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels
