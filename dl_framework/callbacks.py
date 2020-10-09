import torch
import numpy as np
from dl_framework.utils import camel2snake, AvgStats, listify
from re import sub
import matplotlib.pyplot as plt
import pandas as pd
from dl_framework.data import do_normalisation
from dl_framework.logger import make_notifier
from dl_framework.model import save_model


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Callback:
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    _order = 0

    def begin_fit(self):
        self.run.n_epochs = 0.0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        # We use the logger function of the `Learner` here, it can be
        # customized to write in a file or in a progress bar
        self.log(self.train_stats.avg_print())
        self.log(self.valid_stats.avg_print())


class ParamScheduler(Callback):
    _order = 3

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs = fs * len(self.opt.param_groups)
        pos = self.n_epochs / self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)


class Recorder(Callback):
    _order = 5

    def begin_fit(self):
        if not hasattr(self, "lrs"):
            self.lrs = []
        if not hasattr(self, "train_losses"):
            self.train_losses = []
        if not hasattr(self, "valid_losses"):
            self.valid_losses = []
        if not hasattr(self, "losses"):
            self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.state_dict()["param_groups"][0]["lr"])

    def after_epoch(self):
        self.train_losses.append(self.avg_stats.train_stats.avg_stats[1])
        self.valid_losses.append(self.avg_stats.valid_stats.avg_stats[1])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self):
        print("Used learning rate: ", list(set(self.lrs)))
        plt.plot(self.lrs)
        plt.xlabel(r"Number of Batches")
        plt.ylabel(r"Learning rate")
        plt.tight_layout()

    def plot_loss(self, log=True):
        # import matplotlib as mpl

        # # make nice Latex friendly plots
        # mpl.use("pgf")
        # mpl.rcParams.update(
        #     {
        #         "font.size": 12,
        #         "font.family": "sans-serif",
        #         "text.usetex": True,
        #         "pgf.rcfonts": False,
        #         "pgf.texsystem": "lualatex",
        #     }
        # )

        plt.plot(self.train_losses, label="training loss")
        plt.plot(self.valid_losses, label="validation loss")
        if log:
            plt.yscale("log")
        plt.xlabel(r"Number of Epochs")
        plt.ylabel(r"Loss")
        plt.legend()
        plt.tight_layout()

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.xlabel(r"learning rate")
        plt.ylabel(r"loss")
        plt.plot(self.lrs[:n], losses[:n])
        plt.show()


class Recorder_lr_find(Callback):
    """
    Recorder class for the lr_find. Main difference between the recorder
    and this class is that the loss is appended after each batch and not
    after each epoch.
    """

    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.state_dict()["param_groups"][0]["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot(self, skip_last=0, save=False):
        losses = [o.item() for o in self.losses]
        n = len(losses) - skip_last
        plt.plot(self.lrs[:n], losses[:n])
        plt.xscale("log")
        plt.xlabel(r"learning rate")
        plt.ylabel(r"loss")
        if save is False:
            plt.show()


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        """
        max_iter should be slightly bigger than the number of batches.
        Only this way maximum and minimum learning rate are set.
        """
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.run.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            return CancelTrainException
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class LoggerCallback(Callback):
    def __init__(self, model_name):
        self.model_name = model_name

    def begin_fit(self):
        logger = make_notifier()
        logger.info("Start des Trainings von Modell {}".format(self.model_name))

    def after_epoch(self):
        if (self.epoch + 1) % 10 == 0:
            logger = make_notifier()
            logger.info(
                "{}: Epoche {}/{} mit Loss {}".format(
                    self.model_name,
                    self.epoch + 1,
                    self.epochs,
                    self.avg_stats.valid_stats.avg_stats[1],
                )
            )

    def after_fit(self):
        logger = make_notifier()
        logger.info(
            "{}: Ende des Trainings nach {} Epochen mit Loss {}".format(
                self.model_name, self.epoch + 1, self.avg_stats.valid_stats.avg_stats[1]
            )
        )


class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.run.xb = self.run.xb.cuda()
        self.run.yb = self.run.yb.cuda()


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.run.xb)


def view_tfm(*size):
    def _inner(x):
        """
        add correct shape (bs, #channels, shape of array)
        """
        a = x.view(*((-1,) + size))
        return a

    return _inner


def normalize_tfm(norm_path, pointsources=False):
    def _inner(x):
        norm = pd.read_csv(norm_path)
        a = do_normalisation(x.clone(), norm, pointsources)
        assert x[:, 0].mean() != a[:, 0].mean()
        # mean for imag and phase is approx 0
        # assert x[:, 1].mean() != a[:, 1].mean()
        return a

    return _inner


def zero_imag():
    def _inner(x):
        a = x
        imag = a[:, 1, :]
        num = 0
        for i in range(imag.shape[0]):
            if imag[i].max() < 1e-9:
                # print(imag[i].mean().item())
                num += 1
                imag[i] = torch.zeros(imag.shape[1])
        a[:, 1, :] = imag
        # print(num)
        return a

    return _inner


class data_aug(Callback):
    _order = 3

    def begin_batch(self):
        x = self.run.xb.clone()
        y = self.run.yb.clone()
        randint = np.random.randint(0, 4, x.shape[0])
        for i in range(x.shape[0]):
            x[i, 0] = torch.rot90(x[i, 0], int(randint[i]))
            x[i, 1] = torch.rot90(x[i, 1], int(randint[i]))
            y[i, 0] = torch.rot90(y[i, 0], int(randint[i]))
            y[i, 1] = torch.rot90(y[i, 1], int(randint[i]))
        self.run.xb = x
        self.run.yb = y


class SaveCallback(Callback):
    _order = 95

    def __init__(self, model_path, model_name):
        self.model_path = "/".join(model_path.split("/", 2)[:2])
        self.model_name = model_name
    def after_epoch(self):
        if round(self.n_epochs) % 10 == 0:
            save_model(
                self, self.model_path + f"/temp_{round(self.n_epochs)}_{self.model_name}"
            )
            print("\nFinished Epoch {}, model saved.\n".format(round(self.n_epochs)))
