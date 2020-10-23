import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from dl_framework.data import do_normalisation, load_data
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from simulations.utils import adjust_outpath
from pathlib import Path

# ==from gaussian_sources.inspection import visualize_with_fourier


# make nice Latex friendly plots
# mpl.use("pgf")
# mpl.rcParams.update(
#     {
#         "font.size": 12,
#         "font.family": "sans-serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#         "pgf.texsystem": "lualatex",
#     }
# )


def load_pretrained_model(arch_name, model_path):
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

    arch = getattr(architecture, arch_name)()
    load_pre_model(arch, model_path, visualize=True)
    return arch


def get_images(test_ds, num_images, norm_path=None):
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
    rand = torch.randint(0, len(test_ds), size=(num_images,))
    img_test = test_ds[rand][0]
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
        img_test = do_normalisation(img_test, norm)
    img_true = test_ds[rand][1]
    # print(img_true.shape)
    if num_images == 1:
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
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
    # if len(img.shape) == (3):
    # img = img.unsqueeze(0)
    model.eval()
    model.cuda()
    with torch.no_grad():
        pred = model(img.float().cuda())
    return pred.cpu()


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


def plot_loss(learn, model_path):
    """
    Plot train and valid loss of model.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Loss for: {model_path.stem}\n")
    learn.recorder.plot_loss()
    plt.title(r"{}".format(str(model_path.stem).replace("_", " ")))
    plt.yscale("log")
    plt.savefig(f"{save_path}_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr(learn, model_path):
    """
    Plot learning rate of model.
    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Learning rate for: {model_path.stem}\n")
    plt.plot(learn.recorder.lrs)
    plt.savefig(f"{save_path}_lr.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr_loss(learn, arch_name, out_path, skip_last):
    """
    Plot loss of learning rate finder.
    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    arch_path: str
        name of the architecture
    out_path: str
        path to save loss plot
    skip_last: int
        skip n last points
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    print(f"\nPlotting Lr vs Loss for architecture: {arch_name}\n")
    learn.recorder.plot_lr_find()
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    mpl.rcParams.update(mpl.rcParamsDefault)


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


def create_inspection_plots(learn, train_conf):
    test_ds = load_data(
        train_conf["data_path"],
        "test",
        fourier=train_conf["fourier"],
        transformed_imgs=train_conf["transformed_imgs"],
    )
    img_test, img_true = get_images(test_ds, 5, train_conf["norm_path"])  # 5

    pred = eval_model(img_test.cuda(), learn.model)

    model_path = train_conf["model_path"]
    out_path = Path(model_path).parent

    print(img_true.reshape(5, -1, 5))  # 5
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print(pred.cpu().reshape(5, 5, -1))  # .reshape(5,-1,5)

    if train_conf["fourier"]:
        for i in range(len(img_test)):
            m = 0
            # visualize_with_fourier(
            #     i, img_test[i], pred[i], img_true[i], amp_phase=True, out_path=out_path
            # )
    # else:
    #    plot_results(
    #        img_test.cpu(),
    #        pred.cpu().reshape(5,-1,5),
    #        img_true.reshape(5,-1,5),
    #        out_path,
    #        save=True,
    #    )


def make_axes_nice(fig, ax, im, title):
    """Create nice colorbars for every axis in a subplot

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
    ax.set_title(title)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Intensity / a.u.")
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.update_ticks()
