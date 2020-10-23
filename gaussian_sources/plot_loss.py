import click

import dl_framework.architectures as architecture
import torch
from dl_framework.callbacks import Recorder
from dl_framework.learner import get_learner
from dl_framework.model import load_pre_model
from dl_framework.inspection import plot_loss, plot_lr
from dl_framework.data import load_data


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.option(
    "-fourier",
    type=bool,
    required=False,
    help="true, if target variables get fourier transformed",
)
def main(data_path, model_path, arch, fourier=True):
    # Load data, just for getting the image size
    data = []
    train_ds = load_data(data_path, "train", fourier=fourier)

    img_size = train_ds[0][0][0].shape[1]
    # Define model
    if arch == "filter_deep" or arch == "filter_deep_amp" or arch == "filter_deep_phase":
        arch = getattr(architecture, arch)(img_size)
    else:
        arch = getattr(architecture, arch)()
    cbfs = [
        Recorder,
    ]
    learn = get_learner(data, arch, 1e-3, opt_func=torch.optim.Adam, cb_funcs=cbfs)

    load_pre_model(learn, model_path)

    # Plot loss
    plot_lr(learn, model_path)
    plot_loss(learn, model_path)


if __name__ == "__main__":
    main()
