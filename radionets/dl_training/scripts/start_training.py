import click
import sys
import toml
from radionets.dl_training.utils import (
    read_config,
    check_outpath,
    create_databunch,
    define_arch,
    pop_interrupt,
    end_training,
)
from radionets.dl_framework.learner import define_learner
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.inspection import (
    plot_lr_loss,
    plot_loss,
    plot_lr,
    create_inspection_plots,
    create_inspection_lists,
)
from radionets.evaluation.train_inspection import create_inspection_plots
from pathlib import Path


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(
        [
            "train",
            "lr_find",
            "plot_loss",
            "evaluate",
        ],
        case_sensitive=False,
    ),
    default="train",
)
def main(configuration_path, mode):
    """
    Start DNN training with options specified in configuration file.
    Parameters
    ----------
    configuration_path: str
        Path to the configuration toml file
    Modes
    -----
    train: start training of deep learning model (default option)
    lr_find: execute learning rate finder
    plot_loss: plot losscurve of existing model
    """
    config = toml.load(configuration_path)
    train_conf = read_config(config)

    click.echo("\n Train config:")
    print(train_conf, "\n")

    # create databunch
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["bs"],
        transformed_imgs=train_conf["transformed_imgs"],
    )

    # get image size

    transformed_imgs = train_conf["transformed_imgs"]
    if transformed_imgs:
        train_conf["image_size"] = data.train_ds[0][0].shape[1]
    else:
        train_conf["image_size"] = data.train_ds[0][0][0].shape[1]

    # define architecture
    arch = define_arch(
        arch_name=train_conf["arch_name"], img_size=train_conf["image_size"]
    )

    if mode == "train":
        # check out path and look for existing model files
        check_outpath(train_conf["model_path"], train_conf)

        click.echo("Start training of the model.\n")

        # define_learner
        learn = define_learner(
            data,
            arch,
            train_conf,
        )

        # load pretrained model
        if train_conf["pre_model"] != "none":
            load_pre_model(learn, train_conf["pre_model"])

        # Train the model, except interrupt
        try:
            learn.fit(train_conf["num_epochs"])
        except KeyboardInterrupt:
            pop_interrupt(learn, train_conf)

        end_training(learn, train_conf)

        if train_conf["inspection"]:
            num_tests = train_conf["num_tests"]
            create_inspection_lists(learn, train_conf, mode, num_tests)

    if mode == "lr_find":
        click.echo("Start lr_find.\n")

        # define_learner
        learn = define_learner(
            data,
            arch,
            train_conf,
            lr_find=True,
        )

        # load pretrained model
        if train_conf["pre_model"] != "none":
            load_pre_model(learn, train_conf["pre_model"], lr_find=True)

        learn.fit(2)

        # save loss plot
        plot_lr_loss(
            learn,
            train_conf["arch_name"],
            Path(train_conf["model_path"]).parent,
            skip_last=5,
        )

    if mode == "plot_loss":
        click.echo("Start plotting loss.\n")

        # define_learner
        learn = define_learner(
            data,
            arch,
            train_conf,
        )
        # load pretrained model
        if train_conf["pre_model"] != "none":
            load_pre_model(learn, train_conf["pre_model"])
        else:
            click.echo("No pretrained model was selected.")
            click.echo("Exiting.\n")
            sys.exit()

        plot_lr(learn, Path(train_conf["model_path"]))
        plot_loss(learn, Path(train_conf["model_path"]), log=True)

    if mode == "evaluate":
        click.echo("Start evaluation of a pretrained model.\n")

        learn = define_learner(
            data,
            arch,
            train_conf,
        )
        if train_conf["pre_model"] != "none":
            load_pre_model(learn, train_conf["pre_model"])
        else:
            click.echo("No pretrained model was selected.")
            click.echo("Exiting.\n")
            sys.exit()

        num_tests = train_conf["num_tests"]
        create_inspection_lists(learn, train_conf, mode, num_tests=num_tests)

    print(learn.model)


if __name__ == "__main__":
    main()
