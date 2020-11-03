import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from radionets.simulations.utils import adjust_outpath
from tqdm import tqdm
from radionets.evaluation.utils import (
    reshape_2d,
    make_axes_nice,
    check_vmin_vmax,
    calc_jet_angle,
)


plot_format = "pdf"


def plot_target(h5_dataset, log=False):
    index = np.random.randint(len(h5_dataset) - 1)
    plt.figure(figsize=(5.78, 3.57))
    target = reshape_2d(h5_dataset[index][1]).squeeze(0)
    if log:
        plt.imshow(target, norm=LogNorm())
    else:
        plt.imshow(target)
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    plt.colorbar(label="Intensity / a.u.")


def plot_inp_tar(h5_dataset, fourier=False, amp_phase=False):
    index = np.random.randint(len(h5_dataset) - 1)

    if fourier is False:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.45, 3.57))

        inp1 = h5_dataset[index][0][0]
        lim1 = check_vmin_vmax(inp1)
        im1 = ax1.imshow(inp1, cmap="RdBu", vmin=-lim1, vmax=lim1)
        make_axes_nice(fig, ax1, im1, "Input: real part")

        inp2 = h5_dataset[index][0][1]
        lim2 = check_vmin_vmax(inp2)
        im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-lim2, vmax=lim2)
        make_axes_nice(fig, ax2, im2, "Input: imaginary part")

        tar = reshape_2d(h5_dataset[index][1]).squeeze(0)
        im3 = ax3.imshow(tar, cmap="inferno")
        make_axes_nice(fig, ax3, im3, "Target: source image")

    if fourier is True:
        if amp_phase is False:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.45, 8.92))

            inp1 = h5_dataset[index][0][0]
            lim1 = check_vmin_vmax(inp1)
            im1 = ax1.imshow(inp1, cmap="RdBu", vmin=-lim1, vmax=lim1)
            make_axes_nice(fig, ax1, im1, "Input: real part")

            inp2 = h5_dataset[index][0][1]
            lim2 = check_vmin_vmax(inp2)
            im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-lim2, vmax=lim2)
            make_axes_nice(fig, ax2, im2, "Input: imaginary part")

            tar1 = h5_dataset[index][1][0]
            lim_t1 = check_vmin_vmax(tar1)
            im3 = ax3.imshow(tar1, cmap="RdBu", vmin=-lim_t1, vmax=lim_t1)
            make_axes_nice(fig, ax3, im3, "Target: real part")

            tar2 = h5_dataset[index][1][1]
            lim_t2 = check_vmin_vmax(tar2)
            im4 = ax4.imshow(tar2, cmap="RdBu", vmin=-lim_t2, vmax=lim_t2)
            make_axes_nice(fig, ax4, im4, "Target: imaginary part")

        if amp_phase is True:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.45, 8.92))

            inp1 = h5_dataset[index][0][0]
            im1 = ax1.imshow(inp1, cmap="inferno")
            make_axes_nice(fig, ax1, im1, "Input: amplitude")

            inp2 = h5_dataset[index][0][1]
            lim2 = check_vmin_vmax(inp2)
            im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-pi, vmax=pi)
            make_axes_nice(fig, ax2, im2, "Input: phase")

            tar1 = h5_dataset[index][1][0]
            im3 = ax3.imshow(tar1, cmap="inferno")
            make_axes_nice(fig, ax3, im3, "Target: amplitude")

            tar2 = h5_dataset[index][1][1]
            im4 = ax4.imshow(tar2, cmap="RdBu", vmin=-pi, vmax=pi)
            make_axes_nice(fig, ax4, im4, "Target: phase")

    fig.tight_layout()


def plot_results(inp, pred, truth, model_path, save=False):
    """
    Plot input images, prediction and true image.
    Parameters
    ----------
    inp: n 2d arrays with 2 channel
        input images
    pred: n 2d arrays
        predicted images
    truth:n 2d arrays
        true images
    """
    for i in tqdm(range(len(inp))):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        real = inp[i][0]
        im1 = ax1.imshow(real, cmap="RdBu", vmin=-real.max(), vmax=real.max())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax1.set_title(r"Real Input")
        fig.colorbar(im1, cax=cax, orientation="vertical")

        imag = inp[i][1]
        im2 = ax2.imshow(imag, cmap="RdBu", vmin=-imag.max(), vmax=imag.max())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax2.set_title(r"Imag Input")
        fig.colorbar(im2, cax=cax, orientation="vertical")

        pre = pred[i]
        im3 = ax3.imshow(pre, cmap="RdBu", vmin=-pre.max(), vmax=pre.max())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax3.set_title(r"Prediction")
        fig.colorbar(im3, cax=cax, orientation="vertical")

        true = truth[i]
        im4 = ax4.imshow(true, cmap="RdBu", vmin=-pre.max(), vmax=pre.max())
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax4.set_title(r"Truth")
        fig.colorbar(im4, cax=cax, orientation="vertical")

        plt.tight_layout()

        if save:
            out = model_path / "predictions/"
            out.mkdir(parents=True, exist_ok=True)

            out_path = adjust_outpath(out, "/prediction", form="pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def visualize_with_fourier(i, img_input, img_pred, img_truth, amp_phase, out_path):
    """
    Visualizing, if the target variables are displayed in fourier space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (2*img_size^2)
    img_truth: current true image as a numpy array with shape (2*img_size^2)
    out_path: str which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = img_input[0], img_input[1]
    real_pred, imag_pred = img_pred[0], img_pred[1]
    real_truth, imag_truth = img_truth[0], img_truth[1]

    if amp_phase:
        inp_real = 10 ** (10 * inp_real - 10) - 1e-10
        real_pred = 10 ** (10 * real_pred - 10) - 1e-10
        real_truth = 10 ** (10 * real_truth - 10) - 1e-10

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, figsize=(16, 10), sharex=True, sharey=True
    )

    if amp_phase:
        im1 = ax1.imshow(inp_real, cmap="inferno")
        make_axes_nice(fig, ax1, im1, r"Amplitude Input")

        im2 = ax2.imshow(real_pred, cmap="inferno")
        make_axes_nice(fig, ax2, im2, r"Amplitude Prediction")

        im3 = ax3.imshow(real_truth, cmap="inferno")
        make_axes_nice(fig, ax3, im3, r"Amplitude Truth")

        a = check_vmin_vmax(inp_imag)
        im4 = ax4.imshow(inp_imag, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax4, im4, r"Phase Input", phase=True)

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_pred, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax5, im5, r"Phase Prediction", phase=True)

        a = check_vmin_vmax(imag_truth)
        im6 = ax6.imshow(imag_truth, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax6, im6, r"Phase Truth", phase=True)
    else:
        a = check_vmin_vmax(inp_real)
        im1 = ax1.imshow(inp_real, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax1, im1, r"Real Input")

        a = check_vmin_vmax(real_truth)
        im2 = ax2.imshow(real_pred, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax2, im2, r"Real Prediction")

        a = check_vmin_vmax(real_truth)
        im3 = ax3.imshow(real_truth, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax3, im3, r"Real Truth")

        a = check_vmin_vmax(inp_imag)
        im4 = ax4.imshow(inp_imag, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax4, im4, r"Imaginary Input")

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_pred, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax5, im5, r"Imaginary Prediction")

        a = check_vmin_vmax(imag_truth)
        im6 = ax6.imshow(imag_truth, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax6, im6, r"Imaginary Truth")

    ax1.set_ylabel(r"Pixels", fontsize=20)
    ax4.set_ylabel(r"Pixels", fontsize=20)
    ax4.set_xlabel(r"Pixels", fontsize=20)
    ax5.set_xlabel(r"Pixels", fontsize=20)
    ax6.set_xlabel(r"Pixels", fontsize=20)
    ax1.tick_params(axis="both", labelsize=20)
    ax2.tick_params(axis="both", labelsize=20)
    ax3.tick_params(axis="both", labelsize=20)
    ax4.tick_params(axis="both", labelsize=20)
    ax5.tick_params(axis="both", labelsize=20)
    ax6.tick_params(axis="both", labelsize=20)
    plt.tight_layout(pad=1.5)

    outpath = str(out_path) + f"/prediction_{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    return real_pred, imag_pred, real_truth, imag_truth


def visualize_source_reconstruction(ifft_pred, ifft_truth, out_path, i):
    m_truth, n_truth, alpha_truth = calc_jet_angle(ifft_truth)
    m_pred, n_pred, alpha_pred = calc_jet_angle(ifft_pred)
    x_space = torch.arange(0, 511, 1)

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    ax1.plot(
        x_space,
        m_pred * x_space + n_pred,
        "r-",
        alpha=0.5,
        label=fr"$\alpha = {np.round(alpha_pred, 3)}$",
    )
    im1 = ax1.imshow(np.abs(ifft_pred), vmax=np.abs(ifft_truth).max())
    ax2.plot(
        x_space,
        m_truth * x_space + n_truth,
        "r-",
        alpha=0.5,
        label=fr"$\alpha = {np.round(alpha_truth, 3)}$",
    )
    im2 = ax2.imshow(np.abs(ifft_truth))

    make_axes_nice(fig, ax1, im1, r"FFT Prediction")
    make_axes_nice(fig, ax2, im2, r"FFT Truth")

    ax1.set_ylabel(r"Pixels")
    ax1.set_xlabel(r"Pixels")
    ax2.set_xlabel(r"Pixels")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    plt.tight_layout(pad=1.5)

    outpath = str(out_path) + f"/fft_pred_{i}.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    return np.abs(ifft_pred), np.abs(ifft_truth)
