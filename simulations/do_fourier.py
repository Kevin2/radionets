import sys
from tqdm import tqdm
import click
from radionets.dl_framework.data import fourier_trafo, save_fft_pair
from radionets.simulations.utils import check_samp, get_samp_files

@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
def main(data_path):
    """
    Do an inverse Fourier Transformation to retrieve 'dirty' images from
    sampled data and save them to new bundles 'tra_' with unchanged truth
    files.

    """

    samp_files = check_samp(data_path)

    if samp_files is not None:

        bundles = get_samp_files(samp_files, data_path)

        click.echo("\n Starting Fourier Transformation of samp_files!")

        for path in tqdm(bundles):
            
            tra, slist = fourier_trafo(path)
            out = data_path + f"/tra_" + path.name.split("_")[-1]

            save_fft_pair(out, tra, slist)

    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
