import sys
from pathlib import Path

import click

import dl_framework.architectures as architecture
from dl_framework.callbacks import normalize_tfm, zero_imag
from dl_framework.data import DataBunch, get_dls, load_data
from dl_framework.hooks import model_summary
from dl_framework.inspection import eval_model, get_images, plot_loss, reshape_2d
from dl_framework.learner import define_learner
from dl_framework.model import load_pre_model, save_model
from mnist_cnn.scripts.visualize import plot_results


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("num_epochs", type=int)
@click.argument("lr", type=float)
@click.argument("loss_func", type=str)
@click.argument("batch_size", type=int)
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
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-inspection", type=bool, required=False, help="make an inspection plot")
@click.option(
    "-test", type=bool, default=False, required=False, help="Disable logger in tests"
)
def main(
    data_path,
    model_path,
    arch,
    norm_path,
    num_epochs,
    lr,
    loss_func,
    batch_size,
    fourier,
    amp_phase,
    pretrained=False,
    pretrained_model=None,
    inspection=False,
    test=False,
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
    train_ds = load_data(data_path, "train", fourier=fourier)
    valid_ds = load_data(data_path, "valid", fourier=fourier)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    img_size = train_ds[0][0][0].shape[1]
    # Define model
    if arch == "filter_deep" or arch == "filter_deep_amp" or arch == "filter_deep_phase":
        arch = getattr(architecture, arch)(img_size)
    else:
        arch = getattr(architecture, arch)()

    # make normalisation
    norm = normalize_tfm(norm_path)

    zero = zero_imag()

    # get model name for recording in LoggerCallback
    model_name = model_path.split("models/")[-1].split("/")[0]

    learn = define_learner(
        data,
        arch,
        norm,
        loss_func,
        lr=lr,
        model_name=model_name,
        model_path=model_path,
        test=test,
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
                test_ds = load_data(data_path, "test", fourier=False)
                img_test, img_true = get_images(test_ds, 5, norm_path)
                pred = eval_model(img_test, learn.model)
                out_path = Path(model_path).parent
                plot_results(
                    img_test,
                    reshape_2d(pred),
                    reshape_2d(img_true),
                    out_path,
                    save=True,
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
        test_ds = load_data(data_path, "test", fourier=False)
        img_test, img_true = get_images(test_ds, 5, norm_path)
        pred = eval_model(img_test, learn.model.cpu())
        out_path = Path(model_path).parent
        plot_results(
            img_test, reshape_2d(pred), reshape_2d(img_true), out_path, save=True
        )


if __name__ == "__main__":
    main()
