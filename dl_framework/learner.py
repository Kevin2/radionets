from dl_framework.utils import listify, param_getter
from dl_framework.callbacks import TrainEvalCallback
import torch
from dl_framework.optimizer import sgd_opt
import torch.nn as nn
from dl_framework.model import init_cnn
from tqdm import tqdm
from functools import partial
import dl_framework.loss_functions as loss_functions
from dl_framework.callbacks import (
    AvgStatsCallback,
    BatchTransformXCallback,
    CudaCallback,
    Recorder,
    SaveCallback,
    LoggerCallback,
    data_aug,
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
    norm,
    loss_func,
    cbfs=[],
    lr=1e-3,
    model_name='',
    model_path='',
    max_iter=400,
    max_lr=1e-1,
    min_lr=1e-6,
    test=False,
    lr_find=False,
    opt_func=torch.optim.Adam,
):
    cbfs.extend([
        # commented out because of normed and limited input values
        # partial(BatchTransformXCallback, norm),
    ])
    if not test:
        cbfs.extend([
            CudaCallback,
        ])
    if not lr_find:
        cbfs.extend([
            Recorder,
            partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
            partial(SaveCallback, model_path=model_path),
        ])
    if not test and not lr_find:
        cbfs.extend([
            partial(LoggerCallback, model_name=model_name),
            data_aug,
        ])

    # get loss func
    if loss_func == "feature_loss":
        loss_func = loss_functions.init_feature_loss()
    else:
        loss_func = getattr(loss_functions, loss_func)

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=torch.optim.Adam, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
