import click
import toml
from dl_training.utils import (
    read_config,
    check_outpath,
    create_databunch,
    define_arch,
    pop_interrupt,
    end_training,
)
from dl_framework.learner import define_learner
from dl_framework.model import load_pre_model
from dl_framework.inspection import create_inspection_plots


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Start DNN training with options specified in configuration file.

    configuration_path: Path to the config toml file
    """
    config = toml.load(configuration_path)
    train_conf = read_config(config)

    click.echo("\n Train config:")
    print(train_conf, "\n")

    # check out path and look for existing files
    check_outpath(train_conf["model_path"], train_conf)

    # create databunch
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["bs"],
        source_list=train_conf["source_list"],
    )

    # get image size
    train_conf["image_size"] = data.train_ds[0][0][0].shape[1]

    # define architecture
    arch = define_arch(
        arch_name=train_conf["arch_name"], img_size=train_conf["image_size"]
    )

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
        create_inspection_plots(learn, train_conf)


if __name__ == "__main__":
    main()
