import numpy as np
import pandas as pd
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.data import do_normalisation
import radionets.dl_framework.architecture as architecture
import torch
import torch.nn.functional as F


def read_config(config):
    eval_conf = {}
    eval_conf["data_path"] = config["paths"]["data_path"]
    eval_conf["model_path"] = config["paths"]["model_path"]
    eval_conf["model_path_2"] = config["paths"]["model_path_2"]
    eval_conf["norm_path"] = config["paths"]["norm_path"]

    eval_conf["quiet"] = config["mode"]["quiet"]
    eval_conf["gpu"] = config["mode"]["gpu"]

    eval_conf["format"] = config["general"]["output_format"]
    eval_conf["fourier"] = config["general"]["fourier"]
    eval_conf["amp_phase"] = config["general"]["amp_phase"]
    eval_conf["arch_name"] = config["general"]["arch_name"]
    eval_conf["source_list"] = config["general"]["source_list"]
    eval_conf["separate"] = config["general"]["separate"]
    eval_conf["arch_name_2"] = config["general"]["arch_name_2"]

    eval_conf["vis_pred"] = config["inspection"]["visualize_prediction"]
    eval_conf["vis_source"] = config["inspection"]["visualize_source_reconstruction"]
    eval_conf["vis_dr"] = config["inspection"]["visualize_dynamic_range"]
    eval_conf["vis_blobs"] = config["inspection"]["visualize_blobs"]
    eval_conf["vis_ms_ssim"] = config["inspection"]["visualize_ms_ssim"]
    eval_conf["num_images"] = config["inspection"]["num_images"]
    eval_conf["random"] = config["inspection"]["random"]

    eval_conf["viewing_angle"] = config["eval"]["evaluate_viewing_angle"]
    eval_conf["dynamic_range"] = config["eval"]["evaluate_dynamic_range"]
    eval_conf["ms_ssim"] = config["eval"]["evaluate_ms_ssim"]
    return eval_conf


def reshape_2d(array):
    """
    Reshape 1d arrays into 2d ones.

    Parameters
    ----------
    array: 1d array
        input array

    Returns
    -------
    array: 2d array
        reshaped array
    """
    shape = [int(np.sqrt(array.shape[-1]))] * 2
    return array.reshape(-1, *shape)


def make_axes_nice(fig, ax, im, title, phase=False):
    """Create nice colorbars with bigger label size for every axis in a subplot.
    Also use ticks for the phase.
    Parameters
    ----------
    fig : figure object
        current figure
    ax : axis object
        current axis
    im : ndarray
        plotted image
    title : str
        title of subplot
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(title, fontsize=16)

    if phase:
        cbar = fig.colorbar(
            im, cax=cax, orientation="vertical", ticks=[-np.pi, 0, np.pi]
        )
    else:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")

    cbar.set_label("Intensity / a.u.", size=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    if phase:
        # set ticks for colorbar
        cbar.ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def reshape_split(img):
    """
    reshapes and splits the the given image based on the image shape.
    If the image is based on two channels, it reshapes with shape
    (1, 2, img_size, img_size), otherwise with shape (img_size, img_size).
    Afterwards, the array is splitted in real and imaginary part if given.
    Parameters
    ----------
    img : ndarray
        image
    Returns
    ----------
    img_reshaped : ndarry
        contains the reshaped image in a numpy array
    img_real, img_imag: ndarrays
        contain the real and the imaginary part
    -------
    """
    if img.shape[0] == 1:
        img_size = int(np.sqrt(img.shape[0]))
        img_reshaped = img.reshape(img_size, img_size)

        return img_reshaped

    else:
        img_size = int(np.sqrt(img.shape[0] / 2))
        img_reshaped = img.reshape(1, 2, img_size, img_size)
        img_real = img_reshaped[0, 0, :]
        img_imag = img_reshaped[0, 1, :]

        return img_real, img_imag


def check_vmin_vmax(inp):
    """
    Check wether the absolute of the maxmimum or the minimum is bigger.
    If the minimum is bigger, return value with minus. Otherwise return
    maximum.
    Parameters
    ----------
    inp : float
        input image
    Returns
    -------
    float
        negative minimal or maximal value
    """
    if np.abs(inp.min()) > np.abs(inp.max()):
        a = -inp.min()
    else:
        a = inp.max()
    return a


def load_pretrained_model(arch_name, model_path, img_size=63):
    """
    Load model architecture and pretrained weigths.

    Parameters
    ----------
    arch_name: str
        name of the architecture (architectures are in dl_framework.architectures)
    model_path: str
        path to pretrained model

    Returns
    -------
    arch: architecture object
        architecture with pretrained weigths
    """
    if (
        arch_name == "filter_deep"
        or arch_name == "filter_deep_amp"
        or arch_name == "filter_deep_phase"
        or arch_name == "filter_deep_list"
        or arch_name == "filter_deep_xy"
    ):
        arch = getattr(architecture, arch_name)(img_size)
    else:
        arch = getattr(architecture, arch_name)()
    load_pre_model(arch, model_path, visualize=True)
    return arch


def get_images(test_ds, num_images, norm_path="none", rand=False):
    """
    Get n random test and truth images.

    Parameters
    ----------
    test_ds: h5_dataset
        data set with test images
    num_images: int
        number of test images
    norm_path: str
        path to normalization factors, if None: no normalization is applied

    Returns
    -------
    img_test: n 2d arrays
        test images
    img_true: n 2d arrays
        truth images
    """
    indices = torch.arange(num_images)
    if rand:
        indices = torch.randint(0, len(test_ds), size=(num_images,))
    img_test = test_ds[indices][0]
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
    img_test = do_normalisation(img_test, norm)
    img_true = test_ds[indices][1]
    return img_test, img_true


def eval_model(img, model):
    """
    Put model into eval mode and evaluate test images.

    Parameters
    ----------
    img: str
        test image
    model: architecture object
        architecture with pretrained weigths

    Returns
    -------
    pred: n 1d arrays
        predicted images
    """
    if len(img.shape) == (3):
        img = img.unsqueeze(0)
    model.eval()
    model.cuda()
    with torch.no_grad():
        pred = model(img.float().cuda())
    return pred.cpu()


def get_ifft(array, amp_phase=False):
    if amp_phase:
        amp = 10 ** (10 * array[:, 0] - 10) - 1e-10

        a = amp * np.cos(array[:, 1])
        b = amp * np.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    return np.abs(np.fft.ifft2(compl))


def pad_unsqueeze(tensor):
    while tensor.shape[-1] < 160:
        tensor = F.pad(input=tensor, pad=(1, 1, 1, 1), mode="constant", value=0)
    tensor = tensor.unsqueeze(1)
    return tensor


def round_n_digits(tensor, n_digits=3):
    return (tensor * 10 ** n_digits).round() / (10 ** n_digits)
