# this code is for the second stage: novel class discovery, 
# using the pretrained checkpoint from the first stage in main_pretrain.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pytorch_lightning.metrics import Accuracy
from utils.Accuracy import Accuracy
from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp_prior_new
from utils.utils import calculate_prior, calculate_prior_mixed
import numpy as np
from argparse import ArgumentParser
# from datetime import datetime


parser = ArgumentParser()

parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="whether to download")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=12, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")

parser.add_argument("--base_lr", default=0.2, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")

parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=4, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--num_views", default=2, type=int, help="number of views")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
# parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--comment", default='', type=str)
parser.add_argument("--project", default="NCD", type=str, help="wandb project")
parser.add_argument("--entity", default=None, type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")

parser.add_argument("--init_prior", type=int, default=1, help="initialization prior")
parser.add_argument("--true_prior", type=int, default=100, help="imbalance ratio")
parser.add_argument("--est_prior", type=str, default="hard_hard", help="oracle|soft|hard: how to estimate the prior")
parser.add_argument("--est_epoch", type=int, default=5, help="starting epoch to estimate prior")
parser.add_argument("--ema", type=float, default=0.99, help="0.9|0.99|0.999, for moving average running prior")
parser.add_argument("--q_size", type=int, default=12000, help="queue size; set to 0 to disable the queue")
parser.add_argument("--scale", type=float, default=1, help="scale * max(pred, dim=-1), aka dynamic temperature")


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res=("CIFAR" in self.hparams.dataset) or ("tiny" in self.hparams.dataset),
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )

        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)

        # Sinkorn-Knopp
        self.sk_prior_new = SinkhornKnopp_prior_new(
            num_iters=self.hparams.num_iters_sk,
            epsilon=self.hparams.epsilon_sk,
        )

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads, self.hparams.num_labeled_classes,
                               self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
                ClusterMetrics(self.hparams.num_heads, self.hparams.num_labeled_classes,
                               self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
                Accuracy(self.hparams.num_labeled_classes, self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads, self.hparams.num_labeled_classes,
                               self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
                ClusterMetrics(self.hparams.num_heads, self.hparams.num_labeled_classes,
                               self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
                Accuracy(self.hparams.num_labeled_classes, self.hparams.num_unlabeled_classes, 1.0/self.hparams.true_prior, self.hparams.dataset),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

        # initialize running prior with the given ``init_prior``
        self.register_buffer('running_prior', calculate_prior(self.hparams.init_prior, self.hparams.num_unlabeled_classes))
        self.register_buffer('running_prior_mixed', calculate_prior_mixed(self.hparams.init_prior, self.hparams.num_labeled_classes, self.hparams.num_unlabeled_classes))

        self.register_buffer('est_prior_mixed', torch.ones(self.hparams.num_labeled_classes + self.hparams.num_unlabeled_classes).mul_(-1))

        # ground truth prior
        self.true_prior = calculate_prior(self.hparams.true_prior, self.hparams.num_unlabeled_classes).to(self.device)
        self.true_prior_mixed = calculate_prior_mixed(self.hparams.true_prior, self.hparams.num_labeled_classes, self.hparams.num_unlabeled_classes).to(self.device)

        # the queue
        if self.hparams.q_size:
            self.register_buffer('q_logits_base', torch.ones(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.q_size, self.hparams.num_labeled_classes).mul_(-1))
            self.register_buffer('q_logits_novel', torch.ones(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.q_size, self.hparams.num_unlabeled_classes).mul_(-1))
            self.register_buffer('q_logits_mixed', torch.ones(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.q_size, self.hparams.num_labeled_classes+self.hparams.num_unlabeled_classes).mul_(-1))
            self.register_buffer('q_mask_mixed', torch.ones(self.hparams.q_size).mul_(-1))
            self.q_pointer_dict = {
                'base': torch.zeros(1, dtype=torch.long),
                'novel': torch.zeros(1, dtype=torch.long),
                'mixed': torch.zeros(1, dtype=torch.long),
                'mask': torch.zeros(1, dtype=torch.long),
            }

        self.est_prior_mixed = self.running_prior_mixed


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_views):
            for other_view in np.delete(range(self.hparams.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_views * (self.hparams.num_views - 1))

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        views, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return views, labels, mask_lab

    @torch.no_grad()
    def queuing(self, logits, queue, in_size, flag='base'):
        pointer = int(self.q_pointer_dict[flag])
        if flag == 'mask':  # now logits is actually a mask
            if (pointer + in_size) // self.hparams.q_size == 0:
                queue[pointer:pointer + in_size] = logits
            else:
                new_point = (pointer + in_size) % self.hparams.q_size
                queue[pointer:] = logits[new_point:]
                queue[:new_point] = logits[:new_point]
            self.q_pointer_dict[flag][0] = (pointer + in_size) % self.hparams.q_size
        else:
            if (pointer+in_size) // self.hparams.q_size == 0:
                queue[:,:,pointer:pointer+in_size,:] = logits
            else:
                new_point = (pointer+in_size) % self.hparams.q_size
                queue[:,:,pointer:,:] = logits[:,:,new_point:,:]
                queue[:,:,:new_point,:] = logits[:,:,:new_point,:]
            self.q_pointer_dict[flag][0] = (pointer+in_size) % self.hparams.q_size

    def convert_logits(self, logits, est_type='hard'):
        if est_type == 'soft':
            preds = F.softmax(logits / self.hparams.temperature, dim=-1)
        elif est_type == 'hard':
            preds = F.one_hot(logits.max(dim=-1)[1], logits.shape[-1])
        elif est_type == 'uniform':
            preds = torch.ones_like(logits) / logits.shape[-1]
        else:
            raise NotImplementedError('Invalid prior estimation method!')
        return preds

    def estimate_queue_prior(self, queue, est_type='hard'):
        if est_type == 'oracle':
            return self.true_prior.type_as(self.running_prior)
        elif -1 in queue:
            return self.running_prior
        else:
            logits = queue.mean(0).mean(0)
            preds = self.convert_logits(logits, est_type)
            prior = preds.sum(0)
            return prior / prior.sum()

    def estimate_queue_prior_mixed(self, queue, est_type='soft_hard'):
        est_type_split = est_type.split('_')
        mask_lab = self.q_mask_mixed.bool()
        logits = queue.mean(0).mean(0)
        logits_base = logits[mask_lab, :self.hparams.num_labeled_classes]
        logits_novel = logits[~mask_lab, self.hparams.num_labeled_classes:]
        if est_type == 'oracle_oracle':
            return self.true_prior_mixed.type_as(self.running_prior_mixed)
        elif -1 in queue:
            return self.running_prior_mixed
        elif 'oracle_' in est_type:
            preds_base = self.true_prior_mixed.type_as(self.running_prior_mixed)[:self.hparams.num_labeled_classes]
            preds_base = preds_base / preds_base.max()
            preds_novel = self.convert_logits(logits_novel, est_type_split[1]).sum(0)
            preds_novel = preds_novel / preds_novel.max()
            prior_mixed = torch.cat([preds_base, preds_novel], dim=-1)
        elif '_oracle' in est_type:
            preds_base = self.convert_logits(logits_base, est_type_split[0]).sum(0)
            preds_base = preds_base / preds_base.max()
            preds_novel = self.true_prior_mixed.type_as(self.running_prior_mixed)[self.hparams.num_labeled_classes:]
            preds_novel = preds_novel / preds_novel.max()
            prior_mixed = torch.cat([preds_base, preds_novel], dim=-1)
        else:
            preds_base = self.convert_logits(logits_base, est_type_split[0])
            preds_novel = self.convert_logits(logits_novel, est_type_split[1])
            prior_mixed = torch.cat([preds_base.sum(0), preds_novel.sum(0)], dim=-1)
        return prior_mixed / prior_mixed.sum()

    def training_step(self, batch, idx):
        views, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes
        self.train()

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)

        # gather outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)

        # now put them into the queue
        if self.hparams.q_size:
            self.queuing(logits[:,:,mask_lab,:self.hparams.num_labeled_classes], self.q_logits_base, int((mask_lab).sum()), 'base')
            self.queuing(logits[:,:,~mask_lab,self.hparams.num_labeled_classes:], self.q_logits_novel, int((~mask_lab).sum()), 'novel')
            self.queuing(logits[:,:,:,:], self.q_logits_mixed, logits.shape[2], 'mixed')
            self.queuing(mask_lab, self.q_mask_mixed, logits.shape[2], 'mask')

        # calculate the new estimated prior based on the queue info
        estimated_prior_mixed = None
        if self.current_epoch >= self.hparams.est_epoch:
            if '_' in self.hparams.est_prior:  # estimate both base and novel classes
                estimated_prior_mixed = self.estimate_queue_prior_mixed(self.q_logits_mixed, est_type=self.hparams.est_prior)
            else:  # only estimate novel classes
                estimated_prior = self.estimate_queue_prior(self.q_logits_novel, est_type=self.hparams.est_prior)
                estimated_prior_novel = estimated_prior
        else:  # do nothing but impose uniform prior
            estimated_prior = self.running_prior
            estimated_prior_novel = estimated_prior

        # ema updating the estimated prior
        if estimated_prior_mixed is not None:
            if -1 in self.est_prior_mixed:
                self.est_prior_mixed = estimated_prior_mixed
            else:
                self.est_prior_mixed = self.hparams.ema * self.est_prior_mixed + (1-self.hparams.ema) * estimated_prior_mixed
            estimated_prior_novel = self.est_prior_mixed[self.hparams.num_labeled_classes:]

        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        targets = torch.zeros_like(logits)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(self.hparams.num_views):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                sk_label_prior_new = self.sk_prior_new(
                    outputs["logits_unlab"][v, h, ~mask_lab], estimated_prior_novel
                ).type_as(targets)

                targets[v, h, ~mask_lab, nlc:] = sk_label_prior_new

        # now scale the logits according to the confidence, aka dynamic temperature
        if self.hparams.scale:
            confidence = self.hparams.scale * F.softmax(logits/self.hparams.temperature, dim=-1).max(dim=-1, keepdim=True)[0]
            logits *= confidence

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # total loss
        loss_cluster = loss_cluster.mean()
        loss = loss_cluster

        # log
        results = {
            "loss": loss.detach(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch
        # forward
        self.eval()
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]
        outputs = self(images)

        if "unlab" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )

        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)

    def validation_epoch_end(self, _):
        self.metrics[0].epoch = self.current_epoch
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    if "acc" in metric:
                        # AVG = str(torch.stack(values).mean(dim=0).tolist())
                        AVG = torch.stack(values).mean(dim=0)
                        avg, avg_many, avg_medium, avg_few = AVG[0], AVG[1], AVG[2], AVG[3]
                        # AVG_inc = str(torch.stack(values_inc).mean(dim=0).tolist())
                        AVG_inc = torch.stack(values_inc).mean(dim=0)
                        avg_inc, avg_many_inc, avg_medium_inc, avg_few_inc = AVG_inc[0], AVG_inc[1], AVG_inc[2], AVG_inc[3]
                        # BEST = values[torch.argmin(self.loss_per_head)]
                        # best, best_many, best_medium, best_few = BEST[0], BEST[1], BEST[2], BEST[3]
                        # BEST_inc = values_inc[torch.argmin(self.loss_per_head)]
                        # best_inc, best_many_inc, best_medium_inc, best_few_inc = BEST_inc[0], BEST_inc[1], BEST_inc[2], BEST_inc[3]
                        self.log(name + "/avg", avg, sync_dist=True)
                        self.log(name + "/avg_many", avg_many, sync_dist=True)
                        self.log(name + "/avg_medium", avg_medium, sync_dist=True)
                        self.log(name + "/avg_few", avg_few, sync_dist=True)
                        # self.log(name + "/best", best, sync_dist=True)
                        # self.log(name + "/best_many", best_many, sync_dist=True)
                        # self.log(name + "/best_medium", best_medium, sync_dist=True)
                        # self.log(name + "/best_few", best_few, sync_dist=True)
                        self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                        self.log(name_inc + "/avg_many", avg_many_inc, sync_dist=True)
                        self.log(name_inc + "/avg_medium", avg_medium_inc, sync_dist=True)
                        self.log(name_inc + "/avg_few", avg_few_inc, sync_dist=True)
                        # self.log(name_inc + "/best", best_inc, sync_dist=True)
                        # self.log(name_inc + "/best_many", best_many_inc, sync_dist=True)
                        # self.log(name_inc + "/best_medium", best_medium_inc, sync_dist=True)
                        # self.log(name_inc + "/best_few", best_few_inc, sync_dist=True)
                    else:
                        name = "/".join([prefix, metric])
                        name_inc = "/".join([prefix_inc, metric])
                        avg = torch.stack(values).mean()
                        avg_inc = torch.stack(values_inc).mean()
                        # best = values[torch.argmin(self.loss_per_head)]
                        # best_inc = values_inc[torch.argmin(self.loss_per_head)]
                        self.log(name + "/avg", avg, sync_dist=True)
                        # self.log(name + "/best", best, sync_dist=True)
                        self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                        # self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result[0])
                self.log(prefix + "/acc_many", result[1])
                self.log(prefix + "/acc_medium", result[2])
                self.log(prefix + "/acc_few", result[3])
                self.log(prefix_inc + "/acc", result_inc[0])
                self.log(prefix_inc + "/acc_many", result_inc[1])
                self.log(prefix_inc + "/acc_medium", result_inc[2])
                self.log(prefix_inc + "/acc_few", result_inc[3])


def main(args):
    # build datamodule
    dm = get_datamodule(args, "discover")

    split = str(args.num_labeled_classes) + "_" + str(args.num_unlabeled_classes)
    prior = 'LT' + str(args.true_prior)
    est = str(args.est_prior)
    q = 'q' + str(args.q_size)
    e = 'e' + str(args.est_epoch)
    ema = 'ema' + str(args.ema)
    s = 's' + str(args.scale) if args.scale is not None else ''
    bs = 'bs' + str(args.batch_size)
    lr = 'lr' + str(args.base_lr)

    # run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    run_name = "-".join(["discover", args.arch, args.dataset, split, prior, est, q, e, s, ema, bs, lr, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    main(args)
