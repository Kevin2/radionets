import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from dl_framework.data import do_normalisation


def get_eval_img(valid_ds, model, norm_path):
    x_t = valid_ds.x.float()
    rand = np.random.randint(0, len(x_t))
    img = x_t[rand].cuda()
    norm = pd.read_csv(norm_path)
    img = do_normalisation(img, norm)
    h = int(np.sqrt(img.shape[1]))
    img = img.view(-1, h, h).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand


def evaluate_model(valid_ds, model, norm_path, nrows=3):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=4,
        figsize=(18, 6 * nrows),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
    )

    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model, norm_path)
        axes[i][0].set_title("x")
        axes[i][0].imshow(
            img[:, 0].view(h, h).cpu(), cmap="RdGy_r", vmax=img.max(), vmin=-img.max()
        )
        axes[i][1].set_title("y_pred")
        im = axes[i][1].imshow(pred.view(h, h), vmin=pred.min(), vmax=pred.max())
        axes[i][2].set_title("y_true")
        axes[i][2].imshow(
            valid_ds.y[rand].view(h, h),
            vmin=valid_ds.y[rand].min(),
            vmax=valid_ds.y[rand].max(),
        )
        fig.colorbar(im, cax=axes[i][3])
    plt.tight_layout()


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
    matplotlib.use("Agg")
    plt.ioff()
    name_model = model_path.split("/")[-1].split(".")[0]
    save_path = model_path.split(".model")[0]
    print("\nPlotting Loss for: {}\n".format(name_model))
    learn.recorder.plot_loss()
    plt.savefig("{}_loss.pdf".format(save_path), bbox_inches="tight", pad_inches=0.01)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


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
    matplotlib.use("Agg")
    plt.ioff()
    print("\nPlotting Lr vs Loss for architecture: {}\n".format(arch_name))
    learn.recorder_lr_find.plot(skip_last, save=True)
    plt.savefig(out_path + "/lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
