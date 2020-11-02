import numpy as np
import pandas as pd
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.data import do_normalisation
import radionets.dl_framework.architecture as architecture
import torch
from math import pi


def read_config(config):
    eval_conf = {}
    eval_conf["data_path"] = config["paths"]["data_path"]
    eval_conf["model_path"] = config["paths"]["model_path"]
    eval_conf["norm_path"] = config["paths"]["norm_path"]

    eval_conf["batch_mode"] = config["mode"]["batch_mode"]
    eval_conf["gpu"] = config["mode"]["gpu"]

    eval_conf["fourier"] = config["general"]["fourier"]
    eval_conf["amp_phase"] = config["general"]["amp_phase"]
    eval_conf["arch_name"] = config["general"]["arch_name"]
    eval_conf["source_list"] = config["general"]["source_list"]

    eval_conf["vis_pred"] = config["inspection"]["visualize_prediction"]
    eval_conf["vis_source"] = config["inspection"]["visualize_source_reconstruction"]
    eval_conf["num_images"] = config["inspection"]["num_images"]

    eval_conf["viewing_angle"] = config["eval"]["evaluate_viewing_angle"]
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


def bmul(vec, mat, axis=0):
    """Expand vector for batchwise matrix multiplication.
    Parameters
    ----------
    vec : 2dtensor
        vector for multiplication
    mat : 3dtensor
        matrix for multiplication
    axis : int, optional
        batch axis, by default 0
    Returns
    -------
    3dtensor
        Product of matrix multiplication. (bs, n, m)
    """
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


def pca(image):
    """
    Compute the major components of an image. The Image is treated as a
    distribution.
    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be used as distribution
    Returns
    -------
    cog_x: Skalar
            X-position of the distributions center of gravity
    cog_y: Skalar
            Y-position of the distributions center of gravity
    psi: Skalar
            Angle between first mjor component and x-axis
    """
    torch.set_printoptions(precision=16)

    pix_x, pix_y, image = im_to_array_value(image)

    print(pix_x.shape)

    cog_x = (torch.sum(pix_x * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)
    cog_y = (torch.sum(pix_y * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    inp = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1)], dim=1)

    cov_w = bmul(
        (cog_x - 1 * torch.sum(image * image, axis=1).unsqueeze(-1) / cog_x).squeeze(1),
        (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
    )

    eig_vals_torch, eig_vecs_torch = torch.symeig(cov_w, eigenvectors=True)

    psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

    return cog_x, cog_y, psi_torch


def calc_jet_angle(image):
    """Caluclate the jet angle from an image created with gaussian sources. This is achieved by a PCA.
    Parameters
    ----------
    image : ndarray
        input image
    Returns
    -------
    float
        slope of the line
    float
        intercept of the line
    float
        angle between the horizontal axis and the jet axis
    """
    image = image.clone()
    # ignore negagive pixels, which can appear in predictions
    # image[image < 0] = 0
    image = torch.where(image > 0, image, torch.tensor(0).double())
    import matplotlib.pyplot as plt

    print(image[0])
    print(image.shape)
    plt.imshow(image[0])
    plt.show()

    # only use brightest pixel
    max_val = torch.tensor([i.max() * 0.4 for i in image]).unsqueeze(1)
    print(max_val.shape)
    image[image < max_val] = 0
    # image = torch.where(image > 0.4 * image.max(), image, torch.tensor(0).double())

    plt.imshow(image[0])
    plt.show()
    print(image[0])

    print(image.shape)
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    print(image.shape)

    _, _, alpha_pca = pca(image)
    print("alpha_pca", alpha_pca)

    # Use pixel with highest pixel value for the computation of the intercept
    x_mid = torch.ones(63, 63).shape[0] // 2
    y_mid = torch.ones(63, 63).shape[1] // 2

    m = torch.tan(pi / 2 - alpha_pca)
    n = torch.tensor(y_mid) - m * torch.tensor(x_mid)
    alpha = (alpha_pca) * 180 / pi
    return m, n, alpha


def im_to_array_value(image):
    """
    Transforms the image to an array of pixel coordinates and the containt
    intensity

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be transformed

    Returns
    -------
    x_coords: Numpy 1Darray (N*M, 1)
            Contains the x-pixel-position of every pixel in the image
    y_coords: Numpy 1Darray (N*M, 1)
            Contains the y-pixel-position of every pixel in the image
    value: Numpy 1Darray (N*M, 1)
            Contains the image-value corresponding to every x-y-pair

    """
    num = image.shape[0]
    pix = image.shape[-1]

    a = torch.arange(0, pix, 1)
    grid_x, grid_y = torch.meshgrid(a, a)
    x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
    y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
    value = image.reshape(-1, pix ** 2)
    return x_coords, y_coords, value


def get_ifft(array, amp_phase=False):
    if amp_phase:
        amp = 10 ** (10 * array[:, 0] - 10) - 1e-10

        a = amp * np.cos(array[:, 1])
        b = amp * np.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    return np.abs(np.fft.ifft2(compl))