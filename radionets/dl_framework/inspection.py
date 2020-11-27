import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from radionets.dl_framework.model import (
    load_pre_model, build_matcher, sort, extract_amp
)
from radionets.dl_framework.data import do_normalisation, load_data
import radionets.dl_framework.architectures as architecture
from radionets.simulations.utils import adjust_outpath
from pathlib import Path
from radionets.evaluation.plotting import visualize_with_fourier


# make nice Latex friendly plots
#mpl.use("pgf")
#mpl.rcParams.update(
#    {
#        "font.size": 12,
#        "font.family": "sans-serif",
#        "text.usetex": True,
#        "pgf.rcfonts": False,
#        "pgf.texsystem": "lualatex",
#    }
#)


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


def get_images(test_ds, num_images, norm_path, ind=0):
    """
    Get n random test and truth images.

    Parameters
    ----------
    test_ds: h5_dataset
        data set with test images
    num_images: int
        number of test images
    norm_path: str
        path to normalization factors

    Returns
    -------
    img_test: n 2d arrays
        test images
    img_true: n 2d arrays
        truth images
    """
    if ind == 0:
        if len(test_ds) == num_images:
            ind = torch.arange(num_images)
        else:
            ind = torch.randint(0, len(test_ds), size=(1,))
            while ind.shape[0] != num_images:
                ind = torch.randint(0, len(test_ds), size=(num_images,))
                ind = torch.unique(ind)
    else:
        ind = torch.unique(torch.tensor(ind))
    img_test = test_ds[ind][0]
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
        img_test = do_normalisation(img_test, norm)
    img_true = test_ds[ind][1]
    if num_images == 1:
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
    return img_test, img_true, ind


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
    with torch.no_grad():
        pred = model(img.float())
    return pred


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


def plot_loss(learn, model_path, log=True):
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
    if log:
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
    plt.plot(learn.recorder.lrs)#learn.recorder.plot_sched()#_lr
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
    learn.recorder.plot_lr_find()#.recorder_lr_find.plot(skip_last, save=True)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_results(inp, pred, truth, transformed_imgs, model_path, save=False):
    """
    Plot input images, prediction and true image.
    Parameters
    ----------
    inp: n 2d arrays with 2 channel if transformed_imgs==False, 1 else
        input images
    pred: n 2d arrays
        predicted images
    truth:n 2d arrays
        true images
    transformed_imgs: bool
    """
    for i in tqdm(range(len(inp))):
        if not transformed_imgs:
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
        else:
            fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize=(10, 8))
            dirt = inp[i]
            im1 = ax1.imshow(dirt, cmap='RdBu', vmin=-dirt.max(), vmax=dirt.max())
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax3.set_title(r"Input")
            fig.colorbar(im1, cax=cax, orientation="vertical")

        pre = pred[i]
        im3 = ax3.imshow(pre, cmap="RdBu", vmin=-dirt.max(), vmax=dirt.max())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax3.set_title(r"Prediction")
        fig.colorbar(im3, cax=cax, orientation="vertical")

        true = truth[i]
        im4 = ax4.imshow(true, cmap="RdBu", vmin=-dirt.max(), vmax=dirt.max())
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


def create_inspection_lists(learn, train_conf, mode):
    path = Path(train_conf["data_path"])
    if path.match('data_amp'):
        inspect_amp(learn, train_conf, mode)
    else:
        inspect_pos(learn, train_conf, mode)


def inspect_amp(learn, train_conf, mode):
    test_ds = load_data(
        train_conf["data_path"],
        "test",
        fourier=train_conf["fourier"],
        transformed_imgs=train_conf["transformed_imgs"],
        source_list=True
    )
    num_tests = train_conf["num_tests"]
    img_test, list_true, ind = get_images(test_ds, num_tests, train_conf["norm_path"])

    print("\n")
    print("Calculating Predictions: ")

    truth = list_true[:, 0, -1]

    if mode == "train":
        pred = eval_model(img_test.cuda(), learn.model)
    elif mode == "evaluate":
        pred = eval_model(img_test, learn.model)

    pred = pred.reshape(truth.shape)

    rel_var = abs(pred-truth)/truth
    var = rel_var.mean().item()
    print('The mean relative error between truth and predictions is {:.2f}%'.format(1e2*var))
    c = torch.stack((pred, truth), dim=1)
    i = np.random.randint(0, num_tests, size=5)
    print(c[i])


def inspect_pos(learn, train_conf, mode):
    test_ds = load_data(
        train_conf["data_path"],
        "test",
        fourier=train_conf["fourier"],
        transformed_imgs=train_conf["transformed_imgs"],
        source_list=True
    )
    num_tests = train_conf["num_tests"]
    img_test, list_true, ind = get_images(test_ds, num_tests, train_conf["norm_path"])

    print("\n")
    print("Calculating Predictions: ")

    truth = list_true.reshape(num_tests, -1, 5)
    truth = truth[:, :, :2]

    if mode == "train":
        pred = eval_model(img_test.cuda(), learn.model)
        pred = pred.cpu()
    elif mode == "evaluate":
        pred = eval_model(img_test, learn.model)

    pred = pred.reshape(num_tests, -1, 2)
    num_s = pred.shape[1]

    matcher = build_matcher()
    matches = matcher(pred, truth)
    pred_ord, _ = zip(*matches)

    pred = torch.stack([sort(pred[v], pred_ord[v]) for v in range(num_tests)])

    dist = [torch.cdist(pred[v], truth[v]).trace() / num_s for v in range(num_tests)]
    acc = [(1 - err / img_test.shape[-1]) for err in dist]

    dist = torch.stack(dist)
    acc = torch.stack(acc)

    if num_tests <= 10:
        compare = torch.stack((pred, truth), dim=1)
        print(compare)
        print(ind)
        print(dist)
    else:
        print("Showing only the first 10 tests")
        print("\n")
        compare = torch.stack((pred[:10], truth[:10]), dim=1)
        print(compare)
        print(ind[:10])
        print(dist[:10])

    print("Mean accuracy: ", "{:.2f}".format(1e2 * acc.mean().item()), "%")
    print("For ", num_tests, "tests.")


def create_inspection_plots(learn, train_conf, mode):
    test_ds = load_data(
        train_conf["data_path"],
        "test",
        fourier=train_conf["fourier"],
        transformed_imgs=train_conf["transformed_imgs"],
        source_list=False,
    )

    test_size = len(test_ds)
    img_test, img_true, _ = get_images(test_ds, test_size, train_conf["norm_path"])

    seg_true = img_true/img_true
    seg_true[seg_true!=seg_true] = 0
    seg_true = seg_true.cpu()

    if mode == "train":
        seg_pred = eval_model(img_test.cuda(), learn.model)
        seg_pred = seg_pred.cpu()
    elif mode == "evaluate":
        seg_pred = eval_model(img_test, learn.model)

    if type(seg_pred)==tuple:
        amp_pred = seg_pred[1]
        seg_pred = seg_pred[0]

        amp_true = extract_amp(img_true)
        amp_true = amp_true.reshape(amp_pred.shape)

        amp_loss = torch.nn.L1Loss()
        amp_loss = amp_loss(amp_pred, amp_true)
        print("Mean L1 Loss for the amplitude prediction: ", amp_loss.item())

    model_path = train_conf["model_path"]
    out_path = Path(model_path).parent

    ind = np.random.randint(0, test_size, size=1)
    while ind.shape[0] != train_conf["num_tests"]:
        ind = np.random.randint(0, test_size, size=train_conf["num_tests"])
        ind = np.unique(ind)

    if train_conf["fourier"]:
        for i in range(test_size):
            visualize_with_fourier(
                i, img_test[i], seg_pred[i], seg_true[i], amp_phase=True, out_path=out_path
            )
    else:
        plot_results(
            img_test[ind].cpu(),
            reshape_2d(seg_pred[ind].cpu()),
            seg_true[ind].cpu(),
            train_conf["transformed_imgs"],
            out_path,
            save=True,
        )

    for n in 1 - np.linspace(0.5, 0.9, 6):
        print("\n")
        print('Inspecting with threshold value: {:.2f}'.format(n))
        dist, pred_ns, indices = segmap_insp(seg_true, reshape_2d(seg_pred), n)
        acc = 1 - len(pred_ns[0])/test_size
        print('Distance Loss: {}'.format(dist))
        print('Accuracy: {:.2f}% for threshold value: {:.2f}'.format(1e2*acc, n))

    if acc >= 0.98:
        print(pred_ns)
        print(indices)

def segmap_insp(truth, pred, threshold):
    size = len(truth)

    i, x, y = torch.where(truth==1)
    I, X, Y = torch.where(pred >= threshold)
    tru_pos = torch.stack((x, y), dim=1)
    pre_pos = torch.stack((X, Y), dim=1)

    arr = {}
    z = [[], []]
    matcher = build_matcher()

    for j in tqdm(range(size)):
        index_p = torch.where(I==j)
        p = len(index_p[0])
        index_t = torch.where(i==j)
        t = len(index_t[0])

        tru = tru_pos[index_t]
        pre = pre_pos[index_p]

        if t >= p:
            tru_ord, _ = matcher(tru, pre)[0]
            tru = tru[tru_ord, :]
            d = abs(tru-pre).sum()/(2*p)
        else:
            pre_ord, _ = matcher(pre, tru)[0]
            pre = pre[pre_ord, :]
            d = abs(tru-pre).sum()/(2*t)

        if d.item() != 0:
            arr[j] = d.item()

        z[0].append(p)
        z[1].append(t)

    z = np.array(z)
    err = np.where(z[0]!=z[1])[0]

    return arr, z[:, err], err
