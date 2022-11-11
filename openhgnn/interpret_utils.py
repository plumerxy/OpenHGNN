import numpy as np
from openhgnn.InterpretGTN import GradCAM
from openhgnn.InterpretGTN import Lime
from openhgnn.InterpretGTN import InterpretEvaluator
from openhgnn.InterpretGTN import Saliency

from openhgnn import InterpretHAN


def node_distribution_plot(grad_idx, grad, flag='0'):
    """
    为基于不同同质图生成的节点重要性绘制散点图
    Parameters
    ----------
    grad_idx: list, grad_idx[i]是第i个同质图具有节点重要性的节点索引
    grad: list,  grad[i]是第i个同质图中的节点重要性

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    plt.title("Distribution of node importance")
    for i in range(len(grad)):
        if flag == 'linear':
            plt.plot(grad_idx[i], grad[i], label="channel " + str(i))
        else:
            plt.plot(grad_idx[i], grad[i], 'o', label="channel " + str(i))

    plt.legend()
    plt.xlabel("node")
    plt.ylabel("attribution")
    plt.show()


def gtn_metapath_weight(flow):
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
                        metapath_weights[index_metapath(str_m, metapath)] += wi * wj * wk
                    else:
                        metapath.append(str_m)
                        metapath_weights.append(wi * wj * wk)
                elif (etypes_dict[i][0] == '') and (etypes_dict[j][2] == etypes_dict[k][0]):
                    str_m = etypes_dict[j][1] + '-' + etypes_dict[k][2]
                    if str_m in metapath:
                        metapath_weights[index_metapath(str_m, metapath)] += wi * wj * wk
                    else:
                        metapath.append(str_m)
                        metapath_weights.append(wi * wj * wk)
                elif (etypes_dict[j][0] == '') and (etypes_dict[i][2] == etypes_dict[k][0]):
                    str_m = etypes_dict[i][1] + '-' + etypes_dict[k][2]
                    if str_m in metapath:
                        metapath_weights[index_metapath(str_m, metapath)] += wi * wj * wk
                    else:
                        metapath.append(str_m)
                        metapath_weights.append(wi * wj * wk)
                elif (etypes_dict[k][0] == '') and (etypes_dict[i][2] == etypes_dict[j][0]):
                    str_m = etypes_dict[i][1] + '-' + etypes_dict[j][2]
                    if str_m in metapath:
                        metapath_weights[index_metapath(str_m, metapath)] += wi * wj * wk
                    else:
                        metapath.append(str_m)
                        metapath_weights.append(wi * wj * wk)
    del metapath[len(metapath) - 1]
    del metapath_weights[len(metapath) - 1]

    """保存到csv"""
    # import pandas as pd
    # import openpyxl
    # dt = pd.DataFrame(metapath_weights, index=metapath)
    # dt.to_excel("meta_path.xlsx")
    return metapath, metapath_weights


def index_metapath(str_m, metapath):
    """查找重复的元路径，返回index"""
    for i, str in enumerate(metapath):
        if str_m == str:
            return i
    return -1


def han_inter(flow):
    """
    不同的数据集，对应的han结构可能不一样，layer_name可能需要更改

    acm_raw： 只有一个GNN的卷积层，所以用layers.0
    imdb4GTN: 两个GNN卷积层，元路径用layers.1，节点用layers.0
    Parameters
    ----------
    flow

    Returns
    -------

    """
    flow.model(flow.hg, flow.model.input_feature())
    # ------- 节点重要性 ------ #
    sl = InterpretHAN.Saliency(flow, "layers.0.model.mods")  # 需要input layers.0
    tot_idx, tot_grad, idx, grad = sl.gen_exp(2)
    # node_distribution_plot(idx[0].reshape(1, -1), grad[0].reshape(1, -1))
    eva = InterpretHAN.InterpretEvaluator(sl)
    print("----------------node importance evaluation----------------")
    m_tot_node_importance = eva.tot_node_importance(2)
    print("total prob diff: %.5f" % m_tot_node_importance)
    nim1, nim2 = eva.node_importance_metric(2, layer_name="layers.0.model.mods")  # 需要output，layers.-1，根据实际情况设定-1
    print("prob diff: %s" % str(nim1))
    print("lime importance diff: %s" % str(nim2))

    # ------- grad-cam元路径解释 ------
    gc = InterpretHAN.GradCAM(flow, flow.model, "layers.0.model.mods")  # 需要output，layers.-1，根据实际情况设定-1
    w_pos, w_neg = gc.gen_exp(0)
    eva = InterpretHAN.InterpretEvaluator(gc)
    m1, m2 = eva.test_metrics(2)
    print("-------------grad-cam for 2000 samples-----------")
    print("m1 metric: " + str(m1))
    print("m2 metric: " + str(m2))

    # ------- lime元路径解释 ------ #
    # exp = self.metapath_interpret_lime(flow)
    lm = InterpretHAN.Lime(flow, layer_name="layers.0.model.mods")  # 初始化lime explainer # 需要output，layers.-1，根据实际情况设定-1
    w_pos, w_neg = lm.gen_exp(0)  # 查看第i个测试用例的正负权重
    eva = InterpretHAN.InterpretEvaluator(lm)  # 初始化可解释性评估器
    lmm1, lmm2 = eva.test_metrics(2)  # 计算指标
    print("-------------lime for 2000 samples-----------")
    print("m1 metric: " + str(lmm1))
    print("m2 metric: " + str(lmm2))


def gtn_inter(flow):
    """ 可解释性 """
    # ------- 节点重要性 ------ #
    sl = Saliency(flow, "h_list")
    # tot_idx, tot_grad, idx, grad = sl.gen_exp(2)
    eva = InterpretEvaluator(sl)
    print("----------------node importance evaluation----------------")
    m_tot_node_importance = eva.tot_node_importance(1)
    print("total prob diff: %.5f" % m_tot_node_importance)
    nim1, nim2 = eva.node_importance_metric(1)
    print("prob diff: %s" % str(nim1))
    print("lime importance diff: %s" % str(nim2))

    # ------- lime元路径解释 ------ #
    # exp = self.metapath_interpret_lime(flow)
    lm = Lime(flow)  # 初始化lime explainer
    w_pos, w_neg = lm.gen_exp(10)  # 查看第i个测试用例的正负权重
    eva = InterpretEvaluator(lm)  # 初始化可解释性评估器
    lmm1, lmm2 = eva.test_metrics(1)  # 计算指标
    print("-------------lime for 2000 samples-----------")
    print("m1 metric: " + str(lmm1))
    print("m2 metric: " + str(lmm2))

    # ------- grad-cam元路径解释 ------
    gc = GradCAM(flow, flow.model, "linear1")
    w_pos, w_neg = gc.gen_exp(10)
    eva = InterpretEvaluator(gc)
    m1, m2 = eva.test_metrics(1)
    print("-------------grad-cam for 2000 samples-----------")
    print("m1 metric: " + str(m1))
    print("m2 metric: " + str(m2))

    # ------- GTN的权重 ----------- #
    metapath, metapath_weight = gtn_metapath_weight(flow)

    return {"total node importance": m_tot_node_importance, "node importance metric": (nim1, nim2), "lime metapath "
                                                                                                    "importance": (
        lmm1, lmm2), "grad-cam metapath importance": (m1, m2)}
