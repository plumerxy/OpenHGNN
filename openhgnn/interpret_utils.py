import numpy as np


def node_distribution_plot(grad_idx, grad):
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
        plt.plot(grad_idx[i], grad[i], 'o', label="channel " + str(i))
        # plt.plot(x, grad[i], label="channel " + str(i))
    plt.legend()
    plt.xlabel("node")
    plt.ylabel("attribution")
    plt.show()

