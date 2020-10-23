import torch.nn as nn
from dl_framework.model import init_cnn
import sys
from dl_framework.loss_functions import (
    init_feature_loss,
    splitted_mse,
    loss_amp,
    loss_phase,
    loss_msssim,
    loss_mse_msssim,
    loss_mse_msssim_phase,
    loss_mse_msssim_amp,
    loss_msssim_amp,
    list_loss,
    loss_pos,
    loss_pos_,
    pos_loss,
    my_loss,
    likelihood,
    likelihood_phase,
    spe,
)
from dl_framework.callbacks import (
    normalize_tfm,
    BatchTransformXCallback,
    SaveTempCallback,
    TelegramLoggerCallback,
    DataAug,
)
from fastai.optimizer import Adam
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.data import CudaCallback
from fastai.callback.schedule import ParamScheduler, combined_cos


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds,
        data.valid_ds,
    )
    return Learner(dls, arch, loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(
    data,
    arch,
    train_conf,
    cbfs=[],
    test=False,
):
    model_path = train_conf["model_path"]
    model_name = model_path.split("models/")[-1].split("/")[0]
    lr = train_conf["lr"]
    opt_func = Adam
    if train_conf["norm_path"] != "none":
        cbfs.extend(
            [
                BatchTransformXCallback(normalize_tfm(train_conf["norm_path"])),
            ]
        )
    if train_conf["param_scheduling"]:
        sched = {
            "lr": combined_cos(
                0.25,
                train_conf["lr_start"],
                train_conf["lr_max"],
                train_conf["lr_stop"],
            )
        }
        cbfs.extend([ParamScheduler(sched)])
    if not test:
        cbfs.extend(
            [
                CudaCallback,
            ]
        )
    if not test:
        cbfs.extend(
            [
                SaveTempCallback(model_path=model_path),
                # DataAug,
            ]
        )
    if train_conf["telegram_logger"]:
        cbfs.extend(
            [
                TelegramLoggerCallback(model_name=model_name),
            ]
        )

    loss_func = train_conf["loss_func"]
    if loss_func == "feature_loss":
        loss_func = init_feature_loss()
    elif loss_func == "l1":
        loss_func = nn.L1Loss()
    elif loss_func == "mse":
        loss_func = nn.MSELoss()
    elif loss_func == "splitted_mse":
        loss_func = splitted_mse
    elif loss_func == "my_loss":
        loss_func = my_loss
    elif loss_func == "likelihood":
        loss_func = likelihood
    elif loss_func == "likelihood_phase":
        loss_func = likelihood_phase
    elif loss_func == "loss_amp":
        loss_func = loss_amp
    elif loss_func == "loss_phase":
        loss_func = loss_phase
    elif loss_func == "msssim":
        loss_func = loss_msssim
    elif loss_func == "mse_msssim":
        loss_func = loss_mse_msssim
    elif loss_func == "mse_msssim_phase":
        loss_func = loss_mse_msssim_phase
    elif loss_func == "mse_msssim_amp":
        loss_func = loss_mse_msssim_amp
    elif loss_func == "msssim_amp":
        loss_func = loss_msssim_amp
    elif loss_func == "list_loss":
        loss_func = list_loss
    elif loss_func == "loss_pos":
        loss_func = loss_pos
    elif loss_func == "loss_pos_":
        loss_func = loss_pos_
    elif loss_func == "pos_loss":
        loss_func = pos_loss
    elif loss_func == "spe":
        loss_func = spe
    else:
        print("\n No matching loss function or architecture! Exiting. \n")
        sys.exit(1)

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=opt_func, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
