# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch as pt
from torchvision.datasets import CelebA
import torchvision.datasets as dset

class MyCelebA(CelebA):
    def __init__(self, root, split="train", transform=None,
                 target_transform=None, download=False):
        print(f'CelebA root {root}')
        super(MyCelebA, self).__init__(root, split, "attr", transform, target_transform, download)

    def __getitem__(self, item):
        img, label = super(MyCelebA, self).__getitem__(item)
        code = label[20]
        return img, code


# lambda path='', split='', transform='', download='': dset.ImageFolder(root=path, transform=transform)
class CelebAWrapper(dset.VisionDataset):
    def __init__(self, root, transform=None,
                 target_transform=None, n_rand_labels=2):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.folder_dset = dset.ImageFolder(root=root, transform=transform)
        self.n_rand_labels = n_rand_labels

    def __getitem__(self, item):
        x, _ = self.folder_dset.__getitem__(item)
        rnd_y = pt.randint(0, self.n_rand_labels, (1,))
        return x, rnd_y

    def __len__(self):
        return len(self.folder_dset)
