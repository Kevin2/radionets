import click
from dl_framework.data import (
    open_fft_pair_npz,
    open_fft_pair,
    get_bundles,
    mean_and_std,
)
import pandas as pd
import numpy as np
import re
from tqdm import tqdm


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
def main(data_path, out_path):
    bundle_paths = get_bundles(data_path)
    bundle_paths = [
        path for path in bundle_paths if re.findall("fft_samp_train", path.name)
    ]

    means_amp = np.array([])
    stds_amp = np.array([])
    means_imag = np.array([])
    stds_imag = np.array([])

    for path in tqdm(bundle_paths):
        # distinguish between compressed (.npz) and not compressed (.h5) files
        if re.search(".npz", str(path)):
            x, _ = open_fft_pair_npz(path)
        else:
            x, _ = open_fft_pair(path)
        x_amp, x_imag = np.double(x[:, 0]), np.double(x[:, 1])
        mean_amp, std_amp = mean_and_std(x_amp)
        mean_imag, std_imag = mean_and_std(x_imag)
        means_amp = np.append(mean_amp, means_amp)
        means_imag = np.append(mean_imag, means_imag)
        stds_amp = np.append(std_amp, stds_amp)
        stds_imag = np.append(std_imag, stds_imag)

    mean_amp = means_amp.mean()
    std_amp = stds_amp.mean()
    mean_imag = means_imag.mean()
    std_imag = stds_imag.mean()

    d = {
        "train_mean_c0": [mean_amp],
        "train_std_c0": [std_amp],
        "train_mean_c1": [mean_imag],
        "train_std_c1": [std_imag],
    }

    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
