import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoveryTargetTransform
from utils.utils import get_img_num_per_cls
# from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
# from scipy import io as mat_io
import numpy as np
import os
# import pandas as pd

def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "ImageNet":
            return PretrainImageNetDataModule(args)
        elif args.dataset == "tiny-imagenet":
            return PretrainTinyDataModule(args)
        else:
            return PretrainCIFARDataModule(args)
    elif mode == "discover":
        if args.dataset == "ImageNet":
            return DiscoverImageNetDataModule(args)
        elif args.dataset == "tiny-imagenet":
            return DiscoverTinyDataModule(args)
        else:
            return DiscoverCIFARDataModule(args)

class PretrainCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.prior = 1.0 / args.true_prior

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )

        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 50000 * (self.num_labeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(labeled_classes, class_nub):
            train_indices_lab = np.where(
                np.isin(np.array(self.train_dataset.targets), i)
            )[0]
            train_indices_lab.sort()

            train_indices_lab = train_indices_lab[:j]
            train_indices_lab_lt.append(train_indices_lab)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab_lt)

        # val datasets
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        val_indices_lab = np.where(np.isin(np.array(self.val_dataset.targets), labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class DiscoverCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.prior = 1.0 / args.true_prior

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        # train dataset
        self.train_dataset = self.dataset_class(self.data_dir, train=True, transform=self.transform_train)
        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        # label classes, train set

        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 50000 * (self.num_labeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(labeled_classes, class_nub):
            train_indices_lab = np.where(
                np.isin(np.array(self.train_dataset.targets), i)
            )[0]
            train_indices_lab.sort()

            train_indices_lab = train_indices_lab[:j]
            train_indices_lab_lt.append(train_indices_lab)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)

        # unlabeled classes , train set

        val_indices_unlab_train_lt = []
        class_nub = get_img_num_per_cls(self.num_unlabeled_classes, 'exp', self.prior, 50000 * (self.num_unlabeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(unlabeled_classes, class_nub):
            val_indices_unlab_train = np.where(
                np.isin(np.array(val_dataset_train.targets), i)
            )[0]
            val_indices_unlab_train.sort()
            val_indices_unlab_train = val_indices_unlab_train[:j]
            val_indices_unlab_train_lt.append(val_indices_unlab_train)
        val_indices_unlab_train_lt = np.hstack(val_indices_unlab_train_lt)

        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train_lt)
        # val_subset_lab_train = torch.utils.data.Subset(val_dataset_train, train_indices_lab_lt)

        train_indices_lt = np.append(train_indices_lab_lt, val_indices_unlab_train_lt)
        self.train_dataset_lt = torch.utils.data.Subset(self.train_dataset, train_indices_lt)


        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)


        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]
        # self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test, val_subset_lab_train]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset_lt,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


IMAGENET_CLASSES_118 = [
    "n01498041",
    "n01537544",
    "n01580077",
    "n01592084",
    "n01632777",
    "n01644373",
    "n01665541",
    "n01675722",
    "n01688243",
    "n01729977",
    "n01775062",
    "n01818515",
    "n01843383",
    "n01883070",
    "n01950731",
    "n02002724",
    "n02013706",
    "n02092339",
    "n02093256",
    "n02095314",
    "n02097130",
    "n02097298",
    "n02098413",
    "n02101388",
    "n02106382",
    "n02108089",
    "n02110063",
    "n02111129",
    "n02111500",
    "n02112350",
    "n02115913",
    "n02117135",
    "n02120505",
    "n02123045",
    "n02125311",
    "n02134084",
    "n02167151",
    "n02190166",
    "n02206856",
    "n02231487",
    "n02256656",
    "n02398521",
    "n02480855",
    "n02481823",
    "n02490219",
    "n02607072",
    "n02666196",
    "n02672831",
    "n02704792",
    "n02708093",
    "n02814533",
    "n02817516",
    "n02840245",
    "n02843684",
    "n02870880",
    "n02877765",
    "n02966193",
    "n03016953",
    "n03017168",
    "n03026506",
    "n03047690",
    "n03095699",
    "n03134739",
    "n03179701",
    "n03255030",
    "n03388183",
    "n03394916",
    "n03424325",
    "n03467068",
    "n03476684",
    "n03483316",
    "n03627232",
    "n03658185",
    "n03710193",
    "n03721384",
    "n03733131",
    "n03785016",
    "n03786901",
    "n03792972",
    "n03794056",
    "n03832673",
    "n03843555",
    "n03877472",
    "n03899768",
    "n03930313",
    "n03935335",
    "n03954731",
    "n03995372",
    "n04004767",
    "n04037443",
    "n04065272",
    "n04069434",
    "n04090263",
    "n04118538",
    "n04120489",
    "n04141975",
    "n04152593",
    "n04154565",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04258138",
    "n04311004",
    "n04326547",
    "n04367480",
    "n04447861",
    "n04483307",
    "n04522168",
    "n04548280",
    "n04554684",
    "n04597913",
    "n04612504",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07716906",
    "n12998815",
    "n13133613",
]

IMAGENET_CLASSES_30 = {
    "A": [
        "n01580077",
        "n01688243",
        "n01883070",
        "n02092339",
        "n02095314",
        "n02098413",
        "n02108089",
        "n02120505",
        "n02123045",
        "n02256656",
        "n02607072",
        "n02814533",
        "n02840245",
        "n02843684",
        "n02877765",
        "n03179701",
        "n03424325",
        "n03483316",
        "n03627232",
        "n03658185",
        "n03785016",
        "n03794056",
        "n03899768",
        "n04037443",
        "n04069434",
        "n04118538",
        "n04154565",
        "n04311004",
        "n04522168",
        "n07695742",
    ],
    "B": [
        "n01883070",
        "n02013706",
        "n02093256",
        "n02097130",
        "n02101388",
        "n02106382",
        "n02112350",
        "n02167151",
        "n02490219",
        "n02814533",
        "n02843684",
        "n02870880",
        "n03017168",
        "n03047690",
        "n03134739",
        "n03394916",
        "n03424325",
        "n03483316",
        "n03658185",
        "n03721384",
        "n03733131",
        "n03786901",
        "n03843555",
        "n04120489",
        "n04152593",
        "n04208210",
        "n04258138",
        "n04522168",
        "n04554684",
        "n12998815",
    ],
    "C": [
        "n01580077",
        "n01592084",
        "n01632777",
        "n01775062",
        "n01818515",
        "n02097130",
        "n02097298",
        "n02098413",
        "n02111500",
        "n02115913",
        "n02117135",
        "n02398521",
        "n02480855",
        "n02817516",
        "n02843684",
        "n02877765",
        "n02966193",
        "n03095699",
        "n03394916",
        "n03424325",
        "n03710193",
        "n03733131",
        "n03785016",
        "n03995372",
        "n04090263",
        "n04120489",
        "n04326547",
        "n04522168",
        "n07697537",
        "n07716906",
    ],
}

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)


        return img, label

def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset
class DiscoverImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        # self.imagenet_split = args.imagenet_split
        # self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.prior = 1.0 / args.true_prior

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        classes_100 = [8, 11, 13, 23, 26, 29, 44, 66, 68, 72, 74, 76, 77, 80, 98, 100, 109, 119, 143, 153, 163, 170,
                       180, 198, 200, 203, 226, 227, 230, 246, 262, 276, 285, 292, 297, 300, 333, 341, 343, 355, 356, 362,
                       364, 367, 378, 397, 422, 442, 459, 466, 471, 484, 492, 509, 523, 526, 540, 544, 551, 608, 612, 652,
                       655, 661, 669, 673, 682, 690, 696, 698, 717, 738, 755, 761, 763, 767, 780, 783, 789, 795, 818, 826,
                       837, 843, 851, 859, 890, 892, 895, 909, 930, 932, 951, 957, 960, 973, 979, 986, 987, 993]
        cls_map = {i: j for i, j in zip(classes_100, range(100))}

        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(self.num_labeled_classes, self.num_unlabeled_classes + self.num_labeled_classes)

        imagenet_train_dataset = ImageNetBase(root=os.path.join(self.data_dir, 'train'), transform=self.transform_train)
        imagenet_val_dataset = ImageNetBase(root=os.path.join(self.data_dir, 'train'), transform=self.transform_val)
        imagenet_test_dataset = ImageNetBase(root=os.path.join(self.data_dir, 'val'), transform=self.transform_val)

        train_dataset = subsample_classes(imagenet_train_dataset, include_classes=classes_100)
        train_dataset.samples = [(s[0], cls_map[s[1]]) for s in train_dataset.samples]
        train_dataset.targets = [s[1] for s in train_dataset.samples]
        train_dataset.uq_idxs = np.array(range(len(train_dataset)))
        train_dataset.target_transform = None

        val_dataset = subsample_classes(imagenet_val_dataset, include_classes=classes_100)
        val_dataset.samples = [(s[0], cls_map[s[1]]) for s in val_dataset.samples]
        val_dataset.targets = [s[1] for s in val_dataset.samples]
        val_dataset.uq_idxs = np.array(range(len(val_dataset)))
        val_dataset.target_transform = None

        test_dataset = subsample_classes(imagenet_test_dataset, include_classes=classes_100)
        test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
        test_dataset.targets = [s[1] for s in test_dataset.samples]
        test_dataset.uq_idxs = np.array(range(len(test_dataset)))
        test_dataset.target_transform = None



        # label indices lt
        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 64600)

        for i, j in zip(labeled_classes, class_nub):
            train_indices_lab = [x for x, t in enumerate(train_dataset.targets) if t == i]
            train_indices_lab.sort()

            train_indices_lab = train_indices_lab[:j]
            train_indices_lab_lt.append(train_indices_lab)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)

        train_indices_unlab_lt = []
        class_nub = get_img_num_per_cls(self.num_unlabeled_classes, 'exp', self.prior, 64600)
        for i, j in zip(unlabeled_classes, class_nub):
            train_indices_unlab = [x for x, t in enumerate(train_dataset.targets) if t == i]
            train_indices_unlab.sort()

            train_indices_unlab = train_indices_unlab[:j]
            train_indices_unlab_lt.append(train_indices_unlab)
        train_indices_unlab_lt = np.hstack(train_indices_unlab_lt)


        # train dataset
        train_indices = np.append(train_indices_lab_lt, train_indices_unlab_lt)
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        # val datasets
        unlabeled_subset_train = torch.utils.data.Subset(val_dataset, train_indices_unlab_lt)
        # unlabeled classes, test set
        unlabeled_idxs = [x for x, t in enumerate(test_dataset.targets) if t in unlabeled_classes]
        unlabeled_subset_test = torch.utils.data.Subset(test_dataset, unlabeled_idxs)
        # labeled classes, test set
        labeled_idxs = [x for x, t in enumerate(test_dataset.targets) if t in labeled_classes]
        labeled_subset_test = torch.utils.data.Subset(test_dataset, labeled_idxs)

        self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


class PretrainImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.prior = 1.0 / args.true_prior
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        classes_100 = [8, 11, 13, 23, 26, 29, 44, 66, 68, 72, 74, 76, 77, 80, 98, 100, 109, 119, 143, 153, 163, 170, 180,
                       198, 200, 203, 226, 227, 230, 246, 262, 276, 285, 292, 297, 300, 333, 341, 343, 355, 356, 362, 364,
                       367, 378, 397, 422, 442, 459, 466, 471, 484, 492, 509, 523, 526, 540, 544, 551, 608, 612, 652, 655,
                       661, 669, 673, 682, 690, 696, 698, 717, 738, 755, 761, 763, 767, 780, 783, 789, 795, 818, 826, 837,
                       843, 851, 859, 890, 892, 895, 909, 930, 932, 951, 957, 960, 973, 979, 986, 987, 993]
        cls_map = {i: j for i, j in zip(classes_100, range(100))}
        labeled_classes = range(self.num_labeled_classes)

        imagenet_train_dataset = ImageNetBase(root=os.path.join(self.data_dir, 'train'), transform=self.transform_train)
        imagenet_val_dataset = ImageNetBase(root=os.path.join(self.data_dir, 'val'), transform=self.transform_val)

        train_dataset = subsample_classes(imagenet_train_dataset, include_classes=classes_100)
        train_dataset.samples = [(s[0], cls_map[s[1]]) for s in train_dataset.samples]
        train_dataset.targets = [s[1] for s in train_dataset.samples]
        train_dataset.uq_idxs = np.array(range(len(train_dataset)))
        train_dataset.target_transform = None

        val_dataset = subsample_classes(imagenet_val_dataset, include_classes=classes_100)
        val_dataset.samples = [(s[0], cls_map[s[1]]) for s in val_dataset.samples]
        val_dataset.targets = [s[1] for s in val_dataset.samples]
        val_dataset.uq_idxs = np.array(range(len(val_dataset)))
        val_dataset.target_transform = None

        # label indices lt
        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 64600)

        for i, j in zip(labeled_classes, class_nub):
            train_indices_lab = [x for x, t in enumerate(imagenet_train_dataset.targets) if t == i]
            train_indices_lab.sort()

            train_indices_lab = train_indices_lab[:j]
            train_indices_lab_lt.append(train_indices_lab)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)
        # train dataset
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab_lt)

        # val datasets

        val_dataset_indices = [x for x, t in enumerate(val_dataset.targets) if t in labeled_classes]
        self.val_dataset = torch.utils.data.Subset(imagenet_val_dataset, val_dataset_indices)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

TINY_IMAGENET_CLASSES_100 = [
    "n07871810",
    "n03637318",
    "n02226429",
    "n02504458",
    "n02165456",
    "n03770439",
    "n02125311",
    "n07875152",
    "n03977966",
    "n02268443",
    "n02190166",
    "n02132136",
    "n03706229",
    "n01641577",
    "n02085620",
    "n03599486",
    "n02963159",
    "n04398044",
    "n04067472",
    "n02814860",
    "n01944390",
    "n04008634",
    "n07720875",
    "n09246464",
    "n04532670",
    "n02231487",
    "n03992509",
    "n07734744",
    "n04465501",
    "n01698640",
    "n03026506",
    "n04118538",
    "n02094433",
    "n02795169",
    "n02950826",
    "n02814533",
    "n03733131",
    "n04259630",
    "n03085013",
    "n04597913",
    "n01910747",
    "n02481823",
    "n02002724",
    "n02279972",
    "n03584254",
    "n07579787",
    "n03617480",
    "n03201208",
    "n07753592",
    "n02074367",
    "n04099969",
    "n03126707",
    "n02099601",
    "n03796401",
    "n02906734",
    "n02058221",
    "n02056570",
    "n03970156",
    "n03400231",
    "n01983481",
    "n02480495",
    "n03983396",
    "n02815834",
    "n02808440",
    "n04376876",
    "n04275548",
    "n04532106",
    "n02509815",
    "n03089624",
    "n04487081",
    "n03763968",
    "n02321529",
    "n04254777",
    "n07715103",
    "n03042490",
    "n01443537",
    "n02837789",
    "n03976657",
    "n02236044",
    "n01629819",
    "n02793495",
    "n01950731",
    "n02988304",
    "n02410509",
    "n02791270",
    "n02977058",
    "n06596364",
    "n02843684",
    "n02123394",
    "n02823428",
    "n02802426",
    "n09428293",
    "n03100240",
    "n02415577",
    "n04074963",
    "n03160309",
    "n03937543",
    "n03980874",
    "n02099712",
    "n03393912",
]

class DiscoveryDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)

class DiscoverTinyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.imagenet_split = args.imagenet_split
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.prior = 1.0 / args.true_prior
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # split classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(TINY_IMAGENET_CLASSES_100))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]
        unlabeled_classes = TINY_IMAGENET_CLASSES_100
        unlabeled_classes.sort()
        unlabeled_class_idxs = [mapping[c] for c in unlabeled_classes]

        # target transform
        all_classes = labeled_classes + unlabeled_classes
        target_transform = DiscoveryTargetTransform(
            {mapping[c]: i for i, c in enumerate(all_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])

        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 100000 * (
                    self.num_labeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(labeled_class_idxs, class_nub):
            labeled_idxs = np.where(np.isin(targets, i))[0]
            labeled_idxs.sort()
            labeled_idxs = labeled_idxs[:j]
            train_indices_lab_lt.append(labeled_idxs)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)

        train_indices_unlab_lt = []
        class_nub = get_img_num_per_cls(self.num_unlabeled_classes, 'exp', self.prior, 100000 * (
                self.num_unlabeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(unlabeled_class_idxs, class_nub):
            unlabeled_idxs = np.where(np.isin(targets, i))[0]
            unlabeled_idxs.sort()
            unlabeled_idxs = unlabeled_idxs[:j]
            train_indices_unlab_lt.append(unlabeled_idxs)
        train_indices_unlab_lt = np.hstack(train_indices_unlab_lt)

        self.train_dataset = torch.utils.data.Subset(train_dataset, np.append(train_indices_lab_lt, train_indices_unlab_lt))
        # val datasets
        val_dataset_train = self.dataset_class(
            train_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets_train = np.array([img[1] for img in val_dataset_train.imgs])
        targets_test = np.array([img[1] for img in val_dataset_test.imgs])
        # unlabeled classes, train set
        unlabeled_subset_train = torch.utils.data.Subset(val_dataset_train, train_indices_unlab_lt)
        # unlabeled classes, test set
        unlabeled_idxs = np.where(np.isin(targets_test, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_test = torch.utils.data.Subset(val_dataset_test, unlabeled_idxs)
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets_test, np.array(labeled_class_idxs)))[0]
        labeled_subset_test = torch.utils.data.Subset(val_dataset_test, labeled_idxs)

        self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]


class PretrainTinyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.prior = 1.0 / args.true_prior
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # find labeled classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(TINY_IMAGENET_CLASSES_100))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]

        # target transform
        target_transform = DiscoveryTargetTransform(
            {mapping[c]: i for i, c in enumerate(labeled_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        #
        train_indices_lab_lt = []
        class_nub = get_img_num_per_cls(self.num_labeled_classes, 'exp', self.prior, 100000 * (
                self.num_labeled_classes / (self.num_labeled_classes + self.num_unlabeled_classes)))
        for i, j in zip(labeled_class_idxs, class_nub):
            labeled_idxs = np.where(np.isin(targets, i))[0]
            labeled_idxs.sort()
            labeled_idxs = labeled_idxs[:j]
            train_indices_lab_lt.append(labeled_idxs)
        train_indices_lab_lt = np.hstack(train_indices_lab_lt)
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab_lt)

        # val datasets
        val_dataset = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets = np.array([img[1] for img in val_dataset.imgs])
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.val_dataset = torch.utils.data.Subset(val_dataset, labeled_idxs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
