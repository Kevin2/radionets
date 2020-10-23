from functools import partial

import click

import dl_framework.architectures as architecture
from dl_framework.callbacks import LR_Find, Recorder_lr_find, normalize_tfm
from dl_framework.data import DataBunch, get_dls, load_data
from dl_framework.inspection import plot_lr_loss
from dl_framework.learner import define_learner
from dl_framework.model import load_pre_model


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("loss_func", type=str)
@click.argument("batch_size", type=int)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option(
    "-max_iter", type=float, required=True, help="maximal iterations for lr_find"
)
@click.option(
    "-min_lr", type=float, required=True, help="minimal learning rate for lr_find"
)
@click.option(
    "-max_lr", type=float, required=True, help="maximal learning rate for lr_find"
)
@click.option(
    "-fourier",
    type=bool,
    required=True,
    help="true, if target variables are fourier transformed",
)
@click.option(
    "-amp_phase",
    type=bool,
    required=True,
    help="true, if amplitude and phase splitting instead of real and imaginary",
)
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-save", type=bool, required=False, help="save the lr vs loss plot")
@click.option(
    "-test", type=bool, default=False, required=True, help="Disable logger in tests"
)
def main(
    data_path,
    arch,
    model_path,
    norm_path,
    batch_size,
    loss_func,
    max_iter,
    min_lr,
    max_lr,
    fourier,
    amp_phase,
    pretrained=False,
    pretrained_model=None,
    save=False,
    test=False,
):
    """
    Train the neural network with existing training and validation data.
    TRAIN_PATH is the path to the training data\n
    VALID_PATH ist the path to the validation data\n
    ARCH is the name of the architecture which is used\n
    NORM_PATH is the path to the normalisation factors\n
    PRETRAINED_MODEL is the path to a pretrained model, which is
                     loaded at the beginning of the training\n
    """
    # Load data
    train_ds = load_data(data_path, "train", fourier=fourier)
    valid_ds = load_data(data_path, "valid", fourier=fourier)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    # First guess for max_iter
    print("\nTotal number of batches ~ ", len(data.train_ds) * 2 // bs)

    img_size = train_ds[0][0][0].shape[1]
    # Define model
    arch_name = arch
    if arch == "filter_deep" or arch == "filter_deep_amp" or arch == "filter_deep_phase":
        arch = getattr(architecture, arch)(img_size)
    else:
        arch = getattr(architecture, arch)()

    # make normalisation
    norm = normalize_tfm(norm_path)

    cbfs = [
        partial(LR_Find, max_iter=max_iter, max_lr=max_lr, min_lr=min_lr),
        Recorder_lr_find,
    ]

    learn = define_learner(
        data, arch, norm, loss_func, test=test, cbfs=cbfs, lr_find=True,
    )

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn, pretrained_model, lr_find=True)

    learn.fit(2)
    if save:
        plot_lr_loss(learn, arch_name, model_path, skip_last=5)
    else:
        learn.recorder_lr_find.plot(skip_last=5)


if __name__ == "__main__":
    main()
