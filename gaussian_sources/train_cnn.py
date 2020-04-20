import re
import sys
from functools import partial

import click
import matplotlib.pyplot as plt

import dl_framework.architectures as architecture
from dl_framework.callbacks import (
    AvgStatsCallback,
    LoggerCallback,
    Recorder,
    SaveCallback,
    normalize_tfm,
    zero_imag,
)
from dl_framework.data import DataBunch, get_bundles, get_dls, h5_dataset
from dl_framework.hooks import model_summary
from dl_framework.learner import define_learner
from dl_framework.loss_functions import init_feature_loss, splitted_mse
from dl_framework.model import load_pre_model, save_model
from inspection import evaluate_model, plot_loss
from torch import nn


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("num_epochs", type=int)
@click.argument("lr", type=float)
@click.argument("loss_func", type=str)
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option(
    "-fourier",
    type=bool,
    required=False,
    help="true, if target variables get fourier transformed",
)
@click.option(
    "-amp_phase",
    type=bool,
    required=False,
    help="true, if if amplitude and phase splitting instead of real and imaginary",
)
@click.option("-log", type=bool, required=True, help="use of logarithm")
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-inspection", type=bool, required=False, help="make an inspection plot")
def main(
    data_path,
    model_path,
    arch,
    norm_path,
    num_epochs,
    lr,
    loss_func,
    fourier,
    amp_phase,
    log=True,
    pretrained=False,
    pretrained_model=None,
    inspection=False,
):
    """
    Train the neural network with existing training and validation data.

    TRAIN_PATH is the path to the training data\n
    VALID_PATH ist the path to the validation data\n
    MODEL_PATH is the Path to which the model is saved\n
    ARCH is the name of the architecture which is used\n
    NORM_PATH is the path to the normalisation factors\n
    NUM_EPOCHS is the number of epochs\n
    LR is the learning rate\n
    PRETRAINED_MODEL is the path to a pretrained model, which is
                     loaded at the beginning of the training\n
    """
    # Load data
    bundle_paths = get_bundles(data_path)
    train = [path for path in bundle_paths if re.findall("fft_samp_train", path.name)]
    valid = [path for path in bundle_paths if re.findall("fft_samp_valid", path.name)]

    # Create train and valid datasets
    train_ds = h5_dataset(train, tar_fourier=fourier, amp_phase=amp_phase)
    valid_ds = h5_dataset(valid, tar_fourier=fourier, amp_phase=amp_phase)

    # Create databunch with defined batchsize
    bs = 16
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    # Define model
    arch = getattr(architecture, arch)()

    # make normalisation
    norm = normalize_tfm(norm_path)

    zero = zero_imag()

    # get model name for recording in LoggerCallback
    model_name = model_path.split("models/")[-1].split("/")[0]

    # Define callback functions
    cbfs = [
        Recorder,
        partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
        # partial(BatchTransformXCallback, zero),
        partial(SaveCallback, model_path=model_path),
        partial(LoggerCallback, model_name=model_name),
    ]

    learn = define_learner(
        data,
        arch,
        norm,
        loss_func,
        cbfs=cbfs,
        lr=lr,
        model_name=model_name,
        model_path=model_path,
    )

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn, pretrained_model)

    # Print model architecture
    # print(learn.model, "\n")
    # model_summary(learn)

    # Train the model, make it possible to stop at any given time
    try:
        learn.fit(num_epochs)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)")
        save = str(input())
        if save == "y":
            # saving the model if asked
            print("Saving the model after epoch {}".format(learn.epoch))
            save_model(learn, model_path)

            # Plot loss
            plot_loss(learn, model_path)

            # Plot input, prediction and true image if asked
            if inspection is True:
                evaluate_model(valid_ds, learn.model, norm_path)
                plt.savefig(
                    "inspection_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01
                )
        else:
            print("Stopping after epoch {}".format(learn.epoch))
        sys.exit(1)

    # Save model
    save_model(learn, model_path)

    # Plot loss
    plot_loss(learn, model_path)

    # Plot input, prediction and true image if asked
    if inspection is True:
        evaluate_model(valid_ds, learn.model, norm_path, nrows=10)
        plt.savefig(
            "inspection_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01
        )


if __name__ == "__main__":
    main()
