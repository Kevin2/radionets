from functools import partial
from dl_framework.utils import ListContainer, get_batch, find_modules,\
                                is_lin_layer


class Hook():
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


# aus utils hier rüberkopiert, weil hooks benötigt werden
def model_summary(learn, find_all=False):
    xb, yb = get_batch(learn.data.valid_dl, learn)
    # Model may not be on the GPU yet
    device = next(learn.model.parameters()).device
    xb, yb = xb.to(device), yb.to(device)
    if find_all:
        mods = find_modules(learn.model, is_lin_layer)
    else:
        mods = learn.model.children()
    # mods = find_modules(learn.model, is_lin_layer)
    # if find_all else learn.model.children()
    f = lambda hook, mod, inp, out: print(f"{mod}\n{out.shape}\n"), print(learn.loss_func)    
    with Hooks(mods, f) as hooks:
        learn.model(xb)
