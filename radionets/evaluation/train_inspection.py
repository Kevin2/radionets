import click
import numpy as np
from pathlib import Path
from radionets.dl_framework.data import load_data
from radionets.evaluation.plotting import (
    visualize_with_fourier,
    plot_results,
    visualize_source_reconstruction,
)
from radionets.evaluation.utils import (
    reshape_2d,
    load_pretrained_model,
    get_images,
    eval_model,
    calc_jet_angle,
    get_ifft,
)


def get_prediction(conf, num_images=None, rand=False):
    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )
    model = load_pretrained_model(conf["arch_name"], conf["model_path"])
    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(
        test_ds, num_images, norm_path=conf["norm_path"], rand=rand
    )
    pred = eval_model(img_test, model)
    return pred, img_test, img_true


def create_inspection_plots(conf, num_images=3, rand=False):
    pred, img_test, img_true = get_prediction(conf, num_images, rand=rand)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation/"
    out_path.mkdir(parents=True, exist_ok=True)
    if conf["fourier"]:
        for i in range(len(img_test)):
            visualize_with_fourier(
                i,
                img_test[i],
                pred[i],
                img_true[i],
                amp_phase=conf["amp_phase"],
                out_path=out_path,
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
        )


def create_source_plots(conf, num_images=3, rand=False):
    # (i, real_pred, imag_pred, real_truth, imag_truth, amp_phase, out_path):
    """
    function for visualizing the output of a inverse fourier transform. For now, it is
    necessary to take the absolute of the result of the inverse fourier transform,
    because the output is complex.
    i: current index of the loop, just used for saving
    real_pred: real part of the prediction computed in visualize with fourier
    imag_pred: imaginary part of the prediction computed in visualize with fourier
    real_truth: real part of the truth computed in visualize with fourier
    imag_truth: imaginary part of the truth computed in visualize with fourier
    """
    pred, img_test, img_true = get_prediction(conf, num_images, rand=rand)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    pred = pred.numpy()

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        visualize_source_reconstruction(pred, truth, out_path, i)

    return np.abs(ifft_pred), np.abs(ifft_truth)


def evaluate_viewing_angle(conf):
    pred, img_test, img_true = get_prediction(conf)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    print(pred.shape)

    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    print(ifft_pred.shape)

    m_truth, n_truth, alpha_truth = calc_jet_angle(ifft_truth)
    m_pred, n_pred, alpha_pred = calc_jet_angle(ifft_pred)

    print(alpha_pred)
