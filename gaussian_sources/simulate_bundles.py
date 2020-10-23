import click
from pathlib import Path
from simulations.gaussian_simulations import create_n_bundles


@click.command()
@click.argument("num_bundles_train", type=int)
@click.argument("num_bundles_valid", type=int)
@click.argument("num_bundles_test", type=int)
@click.argument("bundle_size", type=int)
@click.argument("img_size", type=int)
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
def main(
    num_bundles_train,
    num_bundles_valid,
    num_bundles_test,
    bundle_size,
    img_size,
    out_path,
):
    create_n_bundles(
        num_bundles_train,
        bundle_size,
        img_size,
        Path(out_path) / "gaussian_sources_train",
    )

    create_n_bundles(
        num_bundles_valid,
        bundle_size,
        img_size,
        Path(out_path) / "gaussian_sources_valid",
    )

    create_n_bundles(
        num_bundles_test,
        bundle_size,
        img_size,
        Path(out_path) / "gaussian_sources_test",
    )


if __name__ == "__main__":
    main()
