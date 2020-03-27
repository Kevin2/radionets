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

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import matplotlib.pyplot as plt

from mnist_cnn.preprocessing import prepare_dataset, get_dls, DataBunch
from mnist_cnn.utils import get_h5_data
from torch import nn
from dl_framework.learner import Learner
from dl_framework.optimizer import sgd_opt
from dl_framework.model import conv, Lambda, flatten, init_cnn
from dl_framework.callbacks import (Recorder, AvgStatsCallback, ParamScheduler,
                                    CudaCallback, BatchTransformXCallback, view_tfm,
                                    SaveCallback, normalize_tfm)
from functools import partial

# +
# Load train and valid data
path_train = 'data/mnist_samp_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = 'data/mnist_samp_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
train_ds, valid_ds = prepare_dataset(x_train[0:2048], y_train[0:2048], x_valid[0:2048], y_valid[0:2048], log=False)

# Create databunch with definde batchsize
bs = 64
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)
# -
loader = data.train_dl

next(iter(loader))







# +
# import numpy as np
# a = data.train_ds.x.reshape(10, 4096)
# print(a.shape)
# a.mean()

# +
# from preprocessing import noramlize_data
# a = data.train_ds.x.reshape(10, 4096)
# b = data.valid_ds.x.reshape(10, 4096)
# x_train, x_valid = noramlize_data(a, b)

# +
# img = data.train_ds.x[4]
# plt.imshow(img.reshape(64, 64), cmap='RdGy_r', vmax=img.max(), vmin=-img.max())
# plt.xlabel('u')
# plt.ylabel('v')
# plt.colorbar(label='Amplitude')

# +
from torch import optim
# Define model
def get_model(data):
    model = nn.Sequential(
        *conv(1, 4, (3,3), 2, 1),
        *conv(4, 8, (3,3), 2, 1),
        *conv(8, 16, (3,3), 2, 1),
        nn.MaxPool2d((3,3)),
        *conv(16, 32, (2,2), 2, 1),
        *conv(32, 64, (2,2), 2, 1),
        nn.MaxPool2d((2,2)),
        Lambda(flatten),
        nn.Linear(64, data.c)
    )
    return model

from dl_framework.learner import get_learner

# +
# from dl_framework.callbacks import LR_Find
# from dl_framework.optimizer import StatefulOptimizer, momentum_step, weight_decay, AverageGrad

# sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step, weight_decay],
#                      stats=AverageGrad(), wd=0.01)

# def find_lr(data):
#     mnist_view = view_tfm(1, 64, 64)
#     cbs = [
#         CudaCallback,
#         partial(BatchTransformXCallback, mnist_view),
#         LR_Find,
#         Recorder
#     ]
#     lr_find = get_learner(data, 1e-1, opt_func=adam_opt, cb_funcs=cbs)
#     print(lr_find.opt_func)
#     lr_find.fit(2)
#     lr_find.recorder.plot()

# # Find learning rate
# find_lr(data)

# +
# Combine model and data in learner
import dl_framework.architectures as architecture

# Define model
arch = getattr(architecture, 'cnn')()

# Define resize for mnist data
mnist_view = view_tfm(2, 64, 64)

# make normalisation
norm = normalize_tfm('data/normalization_factors.csv')

from dl_framework.param_scheduling import sched_cos, combine_scheds, sched_lin
from dl_framework.callbacks import MixUp

# sched = combine_scheds([0.3,0.7], [sched_cos(1e-3, 5e-2), sched_cos(5e-2, 8e-4)])
# sched = combine_scheds([0.4,0.6], [sched_cos(5e-2, 2e-1), sched_cos(2e-1, 4e-2)])
# sched = combine_scheds([0.7, 0.3], [sched, sched_lin(4e-2, 4e-2)])
sched = sched_lin(9e-2, 9e-2)

mnist_view = view_tfm(2, 64, 64)
normalize = normalize_tfm('./data/normalization_factors.csv')

cbfs = [
    Recorder,
    partial(AvgStatsCallback, nn.MSELoss()),
#     partial(ParamScheduler, 'lr', sched),
    CudaCallback,
    partial(BatchTransformXCallback, normalize),
    partial(BatchTransformXCallback, mnist_view),
#     MixUp,
    SaveCallback,
]

adam = torch.optim.Adam

learn = get_learner(data, arch, 1e-2, opt_func=adam, cb_funcs=cbfs)
# -


learn.fit(5)

learn.recorder.plot_loss()

learn.recorder.train_losses

adam.state_dict()['param_groups'][0]['lr']

# +
# Evaluate model
from inspection import evaluate_model

evaluate_model(valid_ds, learn.model, norm_path='./data/normalization_factors.csv', nrows=2)
# plt.savefig('mixup_results.png', dpi=100, bbox_inches='tight', pad_inches=0.01)
# -
learn.recorder.plot_loss()
plt.yscale('log')


learn.recorder.plot_lr()
plt.yscale('log')

from dl_framework.utils import get_batch

test = learn.xb


def get_batch(dl, run):
    run.xb,run.yb = next(iter(dl))
    run.in_train=True
    for cb in run.cbs: cb.set_runner(run), print(cb)
    run('begin_batch')
#     run.in_train=False
    return run.xb,run.yb


x, y = get_batch(learn.data.train_dl, learn)
x.shape

x.shape

# img = x[1].squeeze(0).cpu() 
img = y[2].reshape(64, 64).cpu()
plt.imshow(img, cmap='RdBu', vmin=-img.max(), vmax=img.max())
plt.colorbar()
# plt.savefig('show_mixup_real_y_b.pdf', dpi=100, bbox_inches='tight', pad_inches=0.01)

# +
# Show model summary
# model_summary(run, learn, data)
# -
import numpy as np



plt.imshow(img)

img_trans = np.abs(np.fft.fftshift(np.fft.fft2(img)))
plt.imshow(img_trans)

img_true = y[3].reshape(64, 64).cpu()
plt.imshow(img_true)

# Train model
run.fit(10, learn)

# %debug

# +
# Evaluate model
from inspection import evaluate_model

evaluate_model(valid_ds, learn.model, nrows=2)
plt.savefig('mnist_samp_results2.pdf', dpi=100, bbox_inches='tight', pad_inches=0.01)

# +
# Save model
# state = learn.model.state_dict()
# torch.save(state, './mnist_cnn_big_1.model')
# -

# Load model
import torch
m = learn.model
m.load_state_dict(torch.load('./models/mnist_mixup_adam_leaky.model'))
learn.model.cuda()

import imp
module = imp.load_source('get_model', 'models/simple_cnn.py')

module.get_model()

import numpy as np
a = np.arange(4096)
a = a.reshape(1, 1, 64, 64)
a.shape

a.resize(1, 1, 2, 2)
a.shape

a

import numpy as np
import matplotlib.pyplot as plt

a = np.ones((512, 28, 28))

np.fft.fft2(a).shape

from skimage.transform import resize

plt.imshow(resize(a, (a.shape[-1], 63, 63), anti_aliasing=True, mode="constant")[0])
plt.colorbar()

60000 / 500

12 * 100 / 60 




