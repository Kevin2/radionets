import os
from tqdm import tqdm
from numpy import savez_compressed
from simulations.utils import get_fft_bundle_paths, prepare_fft_images
from dl_framework.data import open_fft_bundle, save_fft_pair
from simulations.uv_simulations import sample_freqs


def sample_frequencies(
    data_path,
    amp_phase,
    real_imag,
    fourier,
    compressed,
    specific_mask,
    antenna_config_path,
    lon=None,
    lat=None,
    steps=None,
):
    for mode in ["train", "valid", "test"]:
        print(f"\n Sampling {mode} data set.\n")

        bundle_paths = get_fft_bundle_paths(data_path, mode)

        for path in tqdm(bundle_paths):
            fft, truth = open_fft_bundle(path)
            size = fft.shape[-1]

            fft_scaled = prepare_fft_images(fft.copy(), amp_phase, real_imag)

            if specific_mask is True:
                fft_samp = sample_freqs(
                    fft_scaled.copy(),
                    antenna_config_path,
                    size,
                    lon,
                    lat,
                    steps,
                    plot=False,
                    test=False,
                )
            else:
                fft_samp = sample_freqs(
                    fft_scaled.copy(), antenna_config_path, size=size, specific_mask=False
                )
            out = data_path + f"/fft_samp_" + path.name.split("_")[-1]

            if fourier:
                if compressed:
                    savez_compressed(out, x=fft_samp, y=fft_scaled)
                    os.remove(path)
                else:
                    save_fft_pair(out, fft_samp, fft_scaled)
            else:
                save_fft_pair(out, fft_samp, truth)
