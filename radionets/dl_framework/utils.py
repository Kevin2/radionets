import re
from typing import Iterable
import numpy as np
from torch import nn


def lin_comb(v1, v2, beta):
    return beta * v1 + (1 - beta) * v2


_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def camel2snake(name):
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


def is_listy(x):
    return isinstance(x, (tuple, list))


def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


class AvgStats:
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.0, 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        """
        Creates a list with the total loss and appends the total metric
        of every used metric to that list.
        """
        all_stats_list = [self.tot_loss.item()]
        for i in range(len(self.tot_mets)):
            all_stats_list += [self.tot_mets[i].item()]
        return all_stats_list

    @property
    def avg_stats(self):
        """
        Creates a list which contains the value of the averaged loss and
        the value of every given metric. For a clear output, the name of
        every metric is printed in front of the corresponding value
        """
        stats_list = []
        i = 0
        for o in self.all_stats:
            if i == 0:
                stats_list = ["Loss: ", np.round(o / self.count, 7)]
            else:
                name = str(self.metrics[i - 1]).split("()")[0]
                stats_list += [name, np.round(o / self.count, 7)]
            i += 1
        return stats_list

    def avg_print(self):
        if not self.count:
            return ""
        return f'{"train" if self.in_train else "valid"}: {self.avg_stats}'

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


def param_getter(m):
    return m.parameters()


def get_batch(dl, learn):
    learn.xb, learn.yb = next(iter(dl))
    for cb in learn.cbs:
        cb.set_runner(learn)
    learn("begin_batch")
    return learn.xb, learn.yb


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)


def compose(x, funcs, *args, order_key="_order", **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)  # bool mask
                return [o for m, o in zip(idx, self.items) if m]
            return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} \
                ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


def children(m):
    "returns the children of m as a list"
    return list(m.children())
