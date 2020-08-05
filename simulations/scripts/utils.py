from pathlib import Path
import click
import gzip
import pickle
import os
import numpy as np
from skimage.transform import resize
from simulations.gaussian_simulations import add_noise
from dl_framework.data import save_fft_pair


def check_outpath(outpath, data_format):
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
    print(data_format)
    if exists is True:
        fft = {p for p in path.rglob("*fft*." + str(data_format)) if p.is_file()}
        samp = {p for p in path.rglob("*sampled*." + str(data_format)) if p.is_file()}
        if fft:
            click.echo("Found existing fft_files!")
            if click.confirm("Do you really want to overwrite the files?", abort=False):
                click.echo("Overwriting old fft_files!")
                [p.unlink() for p in fft]
                sim_fft = True
            else:
                click.echo("Using old fft_files!")
                sim_fft = False
        else:
            sim_fft = True
        if samp:
            click.echo("Found existing sampled_files!")
            if click.confirm("Do you really want to overwrite the files?", abort=False):
                click.echo("Overwriting old sampled_files!")
                [p.unlink() for p in samp]
                sim_sampled = True
            else:
                click.echo("No new images sampled!")
                sim_sampled = False
        else:
            sim_sampled = True
    else:
        Path(path).mkdir(parents=True, exist_ok=False)
        sim_fft = True
        sim_sampled = True
    return sim_fft, sim_sampled


def read_config(config):
    sim_conf = {}
    sim_conf['out_path'] = config["paths"]["out_path"]
    if config["mnist"]["simulate"]:
        click.echo("Create fft_images from mnist data set!")

        sim_conf["type"] = "mnist"
        sim_conf["resource"] = config["mnist"]["resource"]
    # else:
    #     click.echo("Create fft_images from simulated data set!")

    #     points = config["pointsources"]["simulate"]
    #     if points:
    #         num_sources_points = config["pointsources"]["num_sources"]

    #     pointg = config["pointlike_gaussians"]["simulate"]
    #     if pointg:
    #         num_sources_pointg = config["pointlike_gaussians"]["num_sources"]

    #     extendedg = config["extended_gaussians"]["simulate"]
    #     if extendedg:
    #         num_components = config["pointlike_gaussians"]["num_components"]

    sim_conf["num_bundles"] = config["image_options"]["num_bundles"]
    sim_conf["bundle_size"] = config["image_options"]["bundle_size"]
    sim_conf["img_size"] = config["image_options"]["img_size"]
    sim_conf["noise"] = config["image_options"]["noise"]
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
        bundle.swapaxes(0, 2), (pixel, pixel), anti_aliasing=True, mode="constant",
    ).swapaxes(2, 0)
    y_prep = y.copy()
    if noise:
        y_prep = add_noise(y_prep)
    x = np.fft.fftshift(np.fft.fft2(y_prep))
    path = adjust_outpath(path, "/fft_bundle_" + option)
    save_fft_pair(path, x, y)
