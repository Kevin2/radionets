import click
import toml
from radionets.simulations.simulate import create_fft_images, sample_fft_images
from radionets.simulations.utils import check_outpath, read_config, calc_norm


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Generate monte carlo simulation data sets to train and test neural networks for
    reconstruction of radio interferometric data.

    configuration_path: Path to the config toml file
    """
    config = toml.load(configuration_path)

    # check out path and look for existing files
    data_path = config["paths"]["data_path"]
    sim_fft, sim_sampled = check_outpath(
        data_path,
        data_format=config["paths"]["data_format"],
        batch_mode=config["mode"]["batch_mode"],
    )

    # declare source options
    sim_conf = read_config(config)

    click.echo("\n Simulation config:")
    print(sim_conf, "\n")

    # start simulations
    if sim_fft is True:
        click.echo("Starting simulation of fft_files!")
        create_fft_images(sim_conf)

    if sim_sampled is True:
        click.echo("Start sampling fft_files!")
        sample_fft_images(sim_conf)
        click.echo("\n Calculating normalization factors.")
        calc_norm(sim_conf)


if __name__ == "__main__":
    main()
