from pathlib import Path
import click
import gzip
import pickle
import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from dl_framework.data import (
    save_fft_pair,
    open_fft_pair,
    open_fft_pair_npz,
    get_bundles,
    split_amp_phase,
    split_real_imag,
    mean_and_std,
)


def check_outpath(outpath, data_format, batch_mode=False):
    """
    Check if outpath exists. Check for existing fft_files and sampled-files.
    Ask to overwrite or reuse existing files.

    Parameters
    ----------
    path : str
        path to out directory

    Returns
    -------
    sim_fft : bool
        flag to enable/disable fft routine
    sim_sampled : bool
        flag to enable/disable sampling routine
    """
    path = Path(outpath)
    exists = path.exists()
    if exists is True:
        fft = {p for p in path.rglob("*fft*." + str(data_format)) if p.is_file()}
        samp = {p for p in path.rglob("*samp*." + str(data_format)) if p.is_file()}
        if fft:
            click.echo("Found existing fft_files!")
            if batch_mode:
                click.echo("Overwriting old fft_files!")
                [p.unlink() for p in fft]
                [p.unlink() for p in samp]
                sim_fft = True
                sim_sampled = True
                return sim_fft, sim_sampled
            elif click.confirm(
                "Do you really want to overwrite the files?", abort=False
            ):
                click.echo("Overwriting old fft_files!")
                [p.unlink() for p in fft]
                [p.unlink() for p in samp]
                sim_fft = True
                sim_sampled = True
                return sim_fft, sim_sampled
            else:
                click.echo("Using old fft_files!")
                sim_fft = False
        else:
            sim_fft = True
        if samp:
            click.echo("Found existing samp_files!")
            if batch_mode:
                click.echo("Overwriting old samp_files!")
                [p.unlink() for p in samp]
                sim_sampled = True
            elif click.confirm(
                "Do you really want to overwrite the files?", abort=False
            ):
                click.echo("Overwriting old samp_files!")
                [p.unlink() for p in samp]
                sim_sampled = True
            else:
                click.echo("No new images sampled!")
                sim_sampled = False
                sys.exit()
        else:
            sim_sampled = True
    else:
        Path(path).mkdir(parents=True, exist_ok=False)
        sim_fft = True
        sim_sampled = True
    return sim_fft, sim_sampled


def read_config(config):
    sim_conf = {}
    sim_conf["data_path"] = config["paths"]["data_path"]
    sim_conf["data_format"] = config["paths"]["data_format"]
    if config["mnist"]["simulate"]:
        click.echo("Create fft_images from mnist data set!")

        sim_conf["type"] = "mnist"
        sim_conf["resource"] = config["mnist"]["resource"]
    if config["gaussians"]["simulate"]:
        click.echo("Create fft_images from gaussian data set! \n")

        sim_conf["type"] = "gaussians"
        if config["gaussians"]["pointsources"]:
            sim_conf["num_pointsources"] = config["gaussians"]["num_pointsources"]
            click.echo("Adding pointsources.")
        else:
            sim_conf["num_pointsources"] = None

        if config["gaussians"]["pointlike_gaussians"]:
            sim_conf["num_pointlike_gaussians"] = config["gaussians"][
                "num_pointlike_gaussians"
            ]
            sim_conf["source_list"] = config["gaussians"]["source_list"]
            click.echo("Adding pointlike gaussians.")
        else:
            sim_conf["num_pointlike_gaussians"] = None

        if config["gaussians"]["extended_gaussians"]:
            sim_conf["num_components"] = config["gaussians"]["num_components"]
            click.echo("Adding extended gaussian sources.")
        else:
            sim_conf["num_components"] = None

    sim_conf["bundles_train"] = config["image_options"]["bundles_train"]
    sim_conf["bundles_valid"] = config["image_options"]["bundles_valid"]
    sim_conf["bundles_test"] = config["image_options"]["bundles_test"]
    sim_conf["bundle_size"] = config["image_options"]["bundle_size"]
    sim_conf["img_size"] = config["image_options"]["img_size"]
    sim_conf["noise"] = config["image_options"]["noise"]

    sim_conf["amp_phase"] = config["sampling_options"]["amp_phase"]
    sim_conf["real_imag"] = config["sampling_options"]["real_imag"]
    sim_conf["antenna_config_path"] = config["sampling_options"]["antenna_config_path"]
    sim_conf["specific_mask"] = config["sampling_options"]["specific_mask"]
    sim_conf["lon"] = config["sampling_options"]["lon"]
    sim_conf["lat"] = config["sampling_options"]["lat"]
    sim_conf["steps"] = config["sampling_options"]["steps"]
    sim_conf["fourier"] = config["sampling_options"]["fourier"]
    sim_conf["compressed"] = config["sampling_options"]["compressed"]
    sim_conf["keep_fft_files"] = config["sampling_options"]["keep_fft_files"]
    return sim_conf


def open_mnist(path):
    """
    Open MNIST data set pickle file.

    Parameters
    ----------
    path: str
        path to MNIST pickle file

    Returns
    -------
    train_x: 2d array
        50000 x 784 images
    valid_x: 2d array
        10000 x 784 images
    """
    with gzip.open(path, "rb") as f:
        ((train_x, _), (valid_x, _), _) = pickle.load(f, encoding="latin-1")
    return train_x, valid_x


def adjust_outpath(path, option, form="h5"):
    """
    Add number to out path when filename already exists.

    Parameters
    ----------
    path: str
        path to save directory
    option: str
        additional keyword to add to path

    Returns
    -------
    out: str
        adjusted path
    """
    counter = 0
    filename = str(path) + (option + "{}." + form)
    while os.path.isfile(filename.format(counter)):
        counter += 1
    out = filename.format(counter)
    return out


def prepare_mnist_bundles(bundle, path, option, noise=False, pixel=63):
    """
    Resize images to specific squared pixel size and calculate fft

    Parameters
    ----------
    image: 2d array
        input image
    pixel: int
        number of pixel in x and y

    Returns
    -------
    x: 2d array
        fft of rescaled input image
    y: 2d array
        rescaled input image
    """
    y = resize(
        bundle.swapaxes(0, 2),
        (pixel, pixel),
        anti_aliasing=True,
        mode="constant",
    ).swapaxes(2, 0)
    y_prep = y.copy()
    if noise:
        y_prep = add_noise(y_prep)
    x = np.fft.fftshift(np.fft.fft2(y_prep))
    path = adjust_outpath(path, "/fft_" + option)
    save_fft_pair(path, x, y)


def get_fft_bundle_paths(data_path, ftype, mode):
    bundles = get_bundles(data_path)
    bundle_paths = [
        path for path in bundles if re.findall(f"{ftype}_{mode}", path.name)
    ]
    return bundle_paths


def prepare_fft_images(fft_images, amp_phase, real_imag):
    if amp_phase:
        amp, phase = split_amp_phase(fft_images)
        amp = (np.log10(amp + 1e-10) / 10) + 1

        # Test new masking for 511 Pixel pictures
        if amp.shape[1] == 511:
            mask = amp > 0.1
            phase[~mask] = 0
        fft_scaled = np.stack((amp, phase), axis=1)
    else:
        real, imag = split_real_imag(fft_images)
        fft_scaled = np.stack((real, imag), axis=1)
    return fft_scaled


def get_noise(image, scale, mean=0, std=1):
    """
    Calculate random noise values for all image pixels.

    Parameters
    ----------
    image: 2darray
        2d image
    scale: float
        scaling factor to increase noise
    mean: float
        mean of noise values
    std: float
        standard deviation of noise values

    Returns
    -------
    out: ndarray
        array with noise values in image shape
    """
    return np.random.normal(mean, std, size=image.shape) * scale


def add_noise(bundle):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked. Using 0.05 * max(image) as scaling factor.

    Parameters
    ----------
    bundle: path
        path to hdf5 bundle file
    preview: bool
        enable/disable showing 10 preview images

    Returns
    -------
    bundle_noised hdf5_file
        bundle with noised images
    """
    bundle_noised = np.array(
        [img + get_noise(img, (img.max() * 0.05)) for img in bundle]
    )
    return bundle_noised


def calc_norm(sim_conf):
    bundle_paths = get_fft_bundle_paths(sim_conf["data_path"], "samp", "train")

    # create empty arrays
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
    df.to_csv(sim_conf["data_path"] + "/norm_factors.csv", index=False)
