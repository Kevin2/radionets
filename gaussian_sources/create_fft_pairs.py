import click
import os
import numpy as np
from tqdm import tqdm
from dl_framework.data import (
    open_bundle,
    save_fft_pair,
    get_bundles,
    split_amp_phase,
    split_real_imag,
)
from simulations.uv_simulations import sample_freqs
from simulations.gaussian_simulations import add_noise
import re
from numpy import savez_compressed


@click.command()
@click.argument("data_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("antenna_config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-amp_phase", type=bool, required=True)
@click.option("-fourier", type=bool, required=True)
@click.option("-specific_mask", type=bool)
@click.option("-compressed", type=bool)
@click.option("-lon", type=float, required=False)
@click.option("-lat", type=float, required=False)
@click.option("-steps", type=float, required=False)
@click.option("-noise", type=bool)
@click.option("-preview", type=bool, required=False, default=False)
def main(
    data_path,
    out_path,
    antenna_config_path,
    amp_phase,
    fourier,
    compressed=False,
    specific_mask=False,
    lon=None,
    lat=None,
    steps=None,
    noise=False,
    preview=False,
):
    """
    get list of bundles
    get len of all all bundles
    split bundles into train and valid (factor 0.2?)
    for every bundle
        calculate fft -> create fft pairs
        save to new h5 file
    tagg train and valid in filename
    """
    modes = ["train", "valid", "test"]

    for mode in modes:
        print(f"Sampling {mode} data set.")
        bundles = get_bundles(data_path)
        bundles = [
            path
            for path in bundles
            if re.findall("gaussian_sources_{}".format(mode), path.name)
        ]

        for path in tqdm(bundles):
            bundle = open_bundle(path)
            images = bundle.copy()
            size = bundle.shape[-1]
            if noise is True:
                images = add_noise(images, preview=preview)
            bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in images])

            if amp_phase:
                amp, phase = split_amp_phase(bundle_fft)
                amp = (np.log10(amp + 1e-10) / 10) + 1

                # Test new masking for 511 Pixel pictures
                if amp.shape[1] == 511:
                    mask = amp > 0.1
                    phase[~mask] = 0
                bundle_fft = np.stack((amp, phase), axis=1)
            else:
                real, imag = split_real_imag(bundle_fft)
                bundle_fft = np.stack((real, imag), axis=1)

            copy = bundle_fft.copy()
            if specific_mask is True:
                bundle_samp = sample_freqs(
                    copy,
                    antenna_config_path,
                    size,
                    lon,
                    lat,
                    steps,
                    plot=False,
                    test=False,
                )
            else:
                bundle_samp = sample_freqs(
                    copy, antenna_config_path, size=size, specific_mask=False
                )
            out = out_path + path.name.split("_")[-1]
            if fourier:
                if compressed:
                    savez_compressed(out, x=bundle_samp, y=bundle_fft)
                    os.remove(path)
                else:
                    save_fft_pair(out, bundle_samp, bundle_fft)
            else:
                save_fft_pair(out, bundle_samp, images)


if __name__ == "__main__":
    main()
