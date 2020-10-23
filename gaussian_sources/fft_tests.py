# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from dl_framework.data import open_bundle, get_bundles, save_fft_pair
from simulations.uv_simulations import sample_freqs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from dl_framework.data import get_bundles
from dl_framework.data import DataBunch, get_dls, h5_dataset, split_real_imag
import h5py
import torch
from dl_framework.hooks import model_summary
import dl_framework.architectures as architecture
from dl_framework.learner import get_learner
from dl_framework.callbacks import Recorder, AvgStatsCallback,\
                                   BatchTransformXCallback, CudaCallback,\
                                   SaveCallback, view_tfm, ParamScheduler,\
                                   normalize_tfm
import torch.nn as nn
from functools import partial

data_path = 'data2/'
bundle_paths = get_bundles(data_path)
train = [path for path in bundle_paths if re.findall('fft_samp_train', path.name)]
valid = [path for path in bundle_paths if re.findall('fft_samp_valid', path.name)]
train[:1] ,valid[:1]

train_ds = h5_dataset(train[:2])
valid_ds = h5_dataset(valid[:2])

bs = 64
data = DataBunch(*get_dls(train_ds, valid_ds, bs))

# +
# Define architecture
arch = getattr(architecture, 'convs')()

# Define resize for mnist data
mnist_view = view_tfm(2, 64, 64)

# make normalisation
norm = normalize_tfm('./data2/normalization_factors.csv')


# Define callback functions
cbfs = [
    Recorder,
    # test for use of multiple Metrics or Loss functions
    partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
    CudaCallback,
    partial(BatchTransformXCallback, norm),
    partial(BatchTransformXCallback, mnist_view),
    SaveCallback,
]

# Define optimiser function
adam_opt = torch.optim.Adam

# Combine model and data in learner
learn = get_learner(data, arch, 1e-3, opt_func=adam_opt, loss_func=nn.MSELoss(), cb_funcs=cbfs)
# -
model_summary(learn, find_all=False)

learn.fit(25)

learn.recorder.plot_loss()

import pandas as pd
from dl_framework.data import do_normalisation
evaluate_model(data.valid_ds, learn.model, 'data2/normalization_factors.csv', nrows=10)
# plt.savefig('gauss_test.pdf')

# +
from matplotlib.colors import LogNorm
def get_eval_img(valid_ds, model, norm_path):
    rand = np.random.randint(0, len(valid_ds))
    img = valid_ds[rand][0].cuda()
    norm = pd.read_csv(norm_path)
    img = do_normalisation(img, norm)
    h = int(np.sqrt(img.shape[1]))
    img = img.view(-1, h, h).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand


def evaluate_model(valid_ds, model, norm_path, nrows=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 6*nrows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model, norm_path)
        axes[i][0].set_title('x')
        axes[i][0].imshow(img[:, 0].view(h, h).cpu(), cmap='RdGy_r',
                          vmax=img.max(), vmin=-img.max())
        axes[i][1].set_title('y_pred')
        im = axes[i][1].imshow(pred.view(h, h),
                               #norm=LogNorm(vmin=1e-8),
                               #vmin=valid_ds[rand][1].min(),
                               #vmax=valid_ds[rand][1].max()
                              )
        axes[i][2].set_title('y_true')
        axes[i][2].imshow(valid_ds[rand][1].view(h, h),
                          #norm=LogNorm(
                          #    vmin=1e-8,
                          #    vmax=valid_ds[rand][1].max()
                          #),
                          #vmin=valid_ds[rand][1].min(),
                          #vmax=valid_ds[rand][1].max()
                         )
        fig.colorbar(im, cax=axes[i][3])
    plt.tight_layout()


# -
from dl_framework.utils import get_batch
from dl_framework.model import fft, flatten

batch = get_batch(learn.data.valid_dl, learn)
batch[0].shape

plt.imshow(batch[0][0][0].cpu())

test = flatten(batch[0]).cpu()
test.shape

test_fft = fft(test)

real = test_fft[0][0]
imag = test_fft[0][1]

img = torch.sqrt(real**2 + imag**2)

img.shape

plt.imshow(img)
plt.colorbar()


plt.imshow(batch[1][0].cpu().reshape(64, 64))

tr = np.array([1, 1, 1, 1, 1, 1])
ti = np.array([2, 2, 2, 2, 2, 2])
t = np.expand_dims(np.stack((tr, ti)), 0)

t.shape

flatten(t)

np.complex(batch[0][0][0].flatten(), batch[0][0][1].flatten())

nump = batch[0][0][0].cpu().numpy() + batch[0][0][1].cpu().numpy() * 1j
nump.shape

plt.imshow(np.abs(np.fft.ifft2(nump)), norm=LogNorm())
plt.colorbar()

plt.imshow(img, norm=LogNorm())
plt.colorbar()

np.isclose(np.abs(np.fft.ifft2(nump)), img.numpy())


