# this code is adapted from pytorch_lightning.metrics

from typing import Any, Callable, Optional
import torch
# import numpy as np
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.utils import _input_format_classification
from utils.utils import get_img_num_per_cls

class Accuracy(Metric):
    r"""
    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math:: \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y_i})

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.  Works with binary, multiclass, and multilabel
    data.  Accepts logits from a model output or integer class values in
    prediction.  Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:

        >>> from pytorch_lightning.metrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)

    """
    def __init__(
        self,
        num_labeled,
        num_unlabeled,
        imb_ratio,
        dataset,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.imb_ratio = imb_ratio
        self.dataset = dataset
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_many", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_many", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_medium", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_medium", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_few", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_few", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        # dataset_name = "cub"
        self.dataset_num = {
            "CIFAR10":[50000],
            "CIFAR100":[50000],
            "tiny-imagenet": [100000],
            "ImageNet": [129200],
        }
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape
        class_nub = get_img_num_per_cls(self.num_labeled, 'exp', self.imb_ratio,
                                        self.dataset_num[self.dataset][0] * (self.num_labeled / (self.num_unlabeled + self.num_labeled)))
        class_id = range(self.num_labeled)
        class_len = int(self.num_labeled / 3)
        many, medium = class_len, 2 * class_len
        for nub, id in zip(class_nub, class_id):
            if id <= many:
                self.correct_many += torch.sum(preds[target == id] == target[target == id])
                self.total_many += torch.sum(target == id)
            elif id > many and id <= medium:
                self.correct_medium += torch.sum(preds[target == id] == target[target == id])
                self.total_medium += torch.sum(target == id)
            else:
                self.correct_few += torch.sum(preds[target == id] == target[target == id])
                self.total_few += torch.sum(target == id)

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

        # print('flag4:', self.correct, self.total)
    def compute(self):
        """
        Computes accuracy over state.
        """
        self.acc_many = self.correct_many / self.total_many if not self.total_many == 0 else 0
        self.acc_medium = self.correct_medium / self.total_medium if not self.total_medium == 0 else 0
        self.acc_few = self.correct_few / self.total_few if not self.total_few == 0 else 0

        return [self.correct.float() / self.total, self.acc_many, self.acc_medium, self.acc_few]
