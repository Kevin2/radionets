from torch import nn, Tensor
from radionets.dl_framework.utils import is_listy

# ab hier aus dem source code von fastai kopiert
# https://github.com/fastai/fastai/blob/master/fastai/callbacks/hooks.py#L58


class Hook:
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m: nn.Module, hook_func, is_forward=True, detach: bool = True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module: nn.Module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (
                (o.detach() for o in output) if is_listy(output) else output.detach()
            )
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks:
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward: bool = True, detach: bool = True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self, i: int) -> Hook:
        return self.hooks[i]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(m, i, o):
    return o if isinstance(o, Tensor) else o if is_listy(o) else list(o)


def hook_outputs(modules, detach=True, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)
