# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from src.trainer import SynthDataset


def get_fid(args):
    data = np.load(args.data)
    gen_img = torch.tensor(data['x'])

    gen_img = gen_img.to(args.device)

    xmin = torch.min(gen_img)
    xmax = torch.max(gen_img)
    xrange = xmax - xmin

    print(f'old range: {xmin} to {xmax}')
    gen_img -= xmin
    gen_img /= xrange

    xmin = torch.min(gen_img)
    xmax = torch.max(gen_img)

    print(f'new range: {xmin} to {xmax}')

    embedding_files = {'imagenet_32_2': 'imagenet32.npz',
                       'imagenet_0_1': 'imagenet32_0_1.npz',
                       'celeb_32_2': 'celeba_32_normed05.npz',
                       'celeb_32_0_1': 'celeba_32_0_1.npz',
                       'cifar10': 'cifar10_32_normed05.npz'
                       }
    real_data_stats_file = os.path.join('../dp-gfmn/data/fid_stats',
                                        embedding_files[args.emb_type])
    dims = 2048

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(args.device)

    stats = np.load(real_data_stats_file)
    mu_real, sig_real = stats['mu'], stats['sig']
    print('real stats loaded')
    # load synth dataset
    data = gen_img[np.random.permutation(gen_img.shape[0])[:5000]].cpu()
    print(data.shape, type(data))
    synth_data = SynthDataset(data=data, targets=None, to_tensor=False)
    synth_data_loader = torch.utils.data.DataLoader(synth_data, batch_size=100, shuffle=False,
                                                    drop_last=False, num_workers=1)

    print('synth data loaded, getting stats')
    # stats from dataloader
    model.eval()

    pred_list = []
    start_idx = 0

    for batch in tqdm(synth_data_loader):
        x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
        x = x.to(args.device)

        with torch.no_grad():
            pred = model(x)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        # pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        pred_list.append(pred)

        start_idx = start_idx + pred.shape[0]

    pred_arr = np.concatenate(pred_list, axis=0)
    # return pred_arr
    mu_syn = np.mean(pred_arr, axis=0)
    sig_syn = np.cov(pred_arr, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sig_real, mu_syn, sig_syn)
    print(fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--emb", type=str, default='fid_checkpoints')
    parser.add_argument("--emb_type", type=str, default='fid_checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()


    get_fid(args)
    # set up directories
