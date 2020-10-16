from dl_framework.utils import listify, param_getter
from dl_framework.callbacks import TrainEvalCallback, normalize_tfm
import torch
from dl_framework.optimizer import sgd_opt
import torch.nn as nn
from dl_framework.model import init_cnn
from tqdm import tqdm
import sys
from functools import partial
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
)
from dl_framework.callbacks import (
    AvgStatsCallback,
    BatchTransformXCallback,
    CudaCallback,
    Recorder,
    SaveCallback,
    LoggerCallback,
    data_aug,
    LR_Find,
    Recorder_lr_find,
)


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Learner:
    def __init__(
        self,
        model,
        data,
        loss_func,
        opt_func=torch.optim.SGD,
        lr=1e-2,
        splitter=param_getter,
        cbs=None,
        cb_funcs=None,
    ):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.lr = lr
        self.splitter = splitter
        self.in_train = False
        self.log = print
        self.opt = None

        # NB: Things marked "NEW" are covered in lesson 12
        # NEW: avoid need for set_runner
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs):
            self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        self.epochs, self.loss = epochs, torch.tensor(0.0)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self("begin_epoch")

    def fit(self, epochs, cbs=None, reset_opt=False):
        # NEW: pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        # NEW: create optimizer on fit(), optionally replacing existing
        if reset_opt or not self.opt:
            self.opt = self.opt_func(
                self.splitter(self.model), lr=self.lr
            )  # weight_decay=0.1)

        try:
            self.do_begin_fit(epochs)
            for epoch in tqdm(range(epochs)):
                self.do_begin_epoch(epoch)
                if not self("begin_epoch"):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self("begin_validate"):
                        self.all_batches()
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {
        "begin_batch",
        "after_pred",
        "after_loss",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_epoch",
        "begin_fit",
        "begin_epoch",
        "begin_epoch",
        "begin_validate",
        "after_epoch",
        "after_cancel_train",
        "after_fit",
    }

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=sgd_opt, **kwargs
):
    init_cnn(arch)
    return Learner(arch, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)


def define_learner(
    data,
    arch,
    train_conf,
    # max_iter=400,
    # max_lr=1e-1,
    # min_lr=1e-6,
    cbfs=[],
    test=False,
    lr_find=False,
    # opt_func=torch.optim.Adam,
):
    model_path = train_conf["model_path"]
    model_name = model_path.split("models/")[-1].split("/")[0]
    lr = train_conf["lr"]
    if train_conf["norm_path"] != "none":
        cbfs.extend(
            [
                partial(
                    BatchTransformXCallback, normalize_tfm(train_conf["norm_path"])
                ),
            ]
        )
    if not test:
        cbfs.extend(
            [
                CudaCallback,
            ]
        )
    if not lr_find:
        cbfs.extend(
            [
                Recorder,
                partial(AvgStatsCallback, metrics=[]),
                partial(SaveCallback, model_path=model_path, model_name=model_name),
            ]
        )
    if not test and not lr_find:
        cbfs.extend(
            [
                # data_aug,
            ]
        )
    if train_conf["telegram_logger"]:
        cbfs.extend(
            [
                partial(LoggerCallback, model_name=model_name),
            ]
        )
    if lr_find:
        cbfs.extend(
            [
                partial(AvgStatsCallback, metrics=[]),
                partial(
                    LR_Find,
                    max_iter=len(data.train_ds) * 2 // train_conf["bs"],
                    max_lr=train_conf["max_lr"],
                    min_lr=train_conf["min_lr"],
                ),
                Recorder_lr_find,
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
    else:
        print("\n No matching loss function or architecture! Exiting. \n")
        sys.exit(1)

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=torch.optim.Adam, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
