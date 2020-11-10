import click
from tqdm import tqdm
import torch
from radionets.simulations.utils import get_fft_bundle_paths
from radionets.dl_framework.data import open_fft_pair

@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
def main(data_path):
    """
    Inspect dirty Images (tra_) for null amplitudes. Output: Number of images
    with null amplitude.
    """
    err = []

    for mode in ['train','valid', 'test']:

        paths = get_fft_bundle_paths(data_path, 'tra', mode)
        click.echo('\n Inspecting '+mode+' data set')

        for path in tqdm(paths):

            x, y = open_fft_pair(path)
            x = torch.from_numpy(x)

            for i in range(500):
                if torch.max(x[i]).item() < 1e-3:
                    err.append(i)

        err.append(-1)

    print(err)

if __name__ == "__main__":
    main()
