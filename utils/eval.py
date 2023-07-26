import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from utils.utils import get_img_num_per_cls

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def cluster_acc(y_true, y_pred, num_labeled, num_unlabeled, dataset_name, ratio):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    dataset_num = {
        "CIFAR10":[50000],
        "CIFAR100":[50000],
        "tiny-imagenet": [100000],
        "ImageNet": [129200],
    }
    mapping, w = compute_best_mapping(y_true, y_pred)

    total_num = dataset_num[dataset_name][0]
    class_nub = get_img_num_per_cls(num_unlabeled, 'exp', ratio, total_num * (num_unlabeled / (num_unlabeled + num_labeled)))
    class_len = int(num_unlabeled / 3)
    many, medium = class_len, 2 * class_len
    class_id = np.arange(num_labeled, num_labeled + num_unlabeled)
    class_id -= class_id.min()
    pred_many_nub, pred_medium_nub, pred_few_nub = 0, 0, 0
    pred_many_id, pred_medium_id, pred_few_id = 0, 0, 0
    for nub, id in zip(class_nub, class_id):
        for i, j in mapping:
            if j == id:
                if id <= many:
                    pred_many_nub += w[i, j]
                    pred_many_id += np.sum(y_true == j)
                elif id <= medium and id > many:
                    pred_medium_nub += w[i, j]
                    pred_medium_id += np.sum(y_true == j)
                else:
                    pred_few_nub += w[i, j]
                    pred_few_id += np.sum(y_true == j)

    pred_many = pred_many_nub / pred_many_id if not pred_many_id == 0 else 0
    pred_medium = pred_medium_nub / pred_medium_id if not pred_medium_id == 0 else 0
    pred_few = pred_few_nub / pred_few_id if not pred_few_id == 0 else 0
    #return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size
    return [sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size, pred_many, pred_medium, pred_few]


def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class ClusterMetrics(Metric):
    def __init__(self, num_heads,num_labeled,num_unlabeled,imb_ratio,dataset):
        super().__init__()
        self.num_heads = num_heads
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.imb_ratio = imb_ratio
        self.dataset = dataset


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)
        targets -= targets.min()
        acc, nmi, ari = [], [], []
        for head in range(self.num_heads):
            t = targets.cpu().numpy()
            p = preds[head].cpu().numpy()
            acc.append(torch.tensor(cluster_acc(t, p, self.num_labeled, self.num_unlabeled, self.dataset, self.imb_ratio), device=preds.device))
            nmi.append(torch.tensor(nmi_score(t, p), device=preds.device))
            ari.append(torch.tensor(ari_score(t, p), device=preds.device))
        return {"acc": acc, "nmi": nmi, "ari": ari}


