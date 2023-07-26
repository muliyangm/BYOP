# some codes are adapted from the following sources:
# LDAM-DRW (https://github.com/kaidic/LDAM-DRW)

import torch


def calculate_prior(imb_factor, num_class):
    prior_list = []
    for i in range(num_class):
        prior_list.append((1 / imb_factor) ** (i / (num_class - 1.0)))
    prior = torch.tensor(prior_list)
    return prior / prior.sum()

def calculate_prior_mixed(imb_factor, num_base_class, num_novel_class):
    prior_list = []
    for i in range(num_base_class):
        prior_list.append((1 / imb_factor) ** (i / (num_base_class - 1.0)))
    for i in range(num_novel_class):
        prior_list.append((1 / imb_factor) ** (i / (num_novel_class - 1.0)))
    prior = torch.tensor(prior_list)
    return prior / prior.sum()

def get_img_num_per_cls(cls_num, imb_type, imb_factor, data_num):
    img_max = data_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

