import os
from tqdm import tqdm
from numpy import savez_compressed
from radionets.simulations.utils import get_fft_bundle_paths, prepare_fft_images
from radionets.dl_framework.data import (
    open_fft_bundle,
    open_fft_pair,
    save_fft_pair,
    save_fft_pair_list,
)
from radionets.simulations.uv_simulations import sample_freqs
import h5py
import numpy as np


def sample_frequencies(
    data_path,
    amp_phase,
    real_imag,
    fourier,
    compressed,
    specific_mask,
    antenna_config,
    lon=None,
    lat=None,
    steps=None,
    source_list=False,
):
    for mode in ["train", "valid", "test"]:
        print(f"\n Sampling {mode} data set.\n")

        bundle_paths = get_fft_bundle_paths(data_path, "fft", mode)

        for path in tqdm(bundle_paths):
            if source_list:
                fft, truth = open_fft_bundle(path)
            else:
                fft, truth = open_fft_pair(path)

            size = fft.shape[-1]

            fft_scaled = prepare_fft_images(fft.copy(), amp_phase, real_imag)

            if specific_mask is True:
                fft_samp = sample_freqs(
                    fft_scaled.copy(),
                    antenna_config,
                    size,
                    lon,
                    lat,
                    steps,
                    plot=False,
                    test=False,
                )
            else:
                fft_samp = sample_freqs(
                    fft_scaled.copy(),
                    antenna_config,
                    size=size,
                    specific_mask=False,
                )
            out = data_path + "/samp_" + path.name.split("_")[-1]

            if fourier:
                if compressed:
                    savez_compressed(out, x=fft_samp, y=fft_scaled)
                    os.remove(path)
                else:
                    save_fft_pair(out, fft_samp, fft_scaled)
            else:
                save_fft_pair(out, fft_samp, truth)
