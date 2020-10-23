import click
import numpy as np

import dl_framework.architectures as architecture
from dl_framework.data import load_data, do_normalisation
from dl_framework.inspection import eval_model
from dl_framework.model import load_pre_model
from gaussian_sources.inspection import save_indices_and_data


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("norm_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("pretrained_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-num", type=int, required=False)
@click.option("-fourier", type=bool, required=True)
@click.option("-separate", type=bool, default=False, required=False)
def main(
    data_path, norm_path, arch, pretrained_path, out_path, fourier, separate, num=20
):
    """
    Create input, predictions and truth csv files for further investigation,
    such as visualize_predictions.

    Parameters
    ----------
    data_path : click path object
        path to the data files. Just the folder is necessary
    data_path : click path object
        path to the normalization factors. Just the name of the file
    arch : string
        contains the name of the used architecture
    pretrained_path : click path object
        path to the pretrained model
    out_path : click path object
        path for the saving folder
    fourier : bool
        true, if the target images are fourier transformed
    separate : bool
        true, if there are separate architectures for amplitude and phase
    num : int, optional
        number of images taken from the test dataset
    """
    # Load data and create random indices
    test_ds = load_data(data_path, "test", fourier=fourier)
    indices = np.random.randint(0, len(test_ds), size=num)

    # get input and target images according to random indices
    images = test_ds[indices][0]
    images_y = test_ds[indices][1].numpy().reshape(num, -1)

    # normalization
    # norm = pd.read_csv(norm_path)
    # images = do_normalisation(images, norm)

    # save input images after normalization
    images_x = images.numpy().reshape(num, -1)

    # load pretrained model
    img_size = test_ds[0][0][0].shape[1]
    if arch == "filter_deep" or "filter_deep_amp":
        arch = getattr(architecture, arch)(img_size)
    else:
        arch = getattr(architecture, arch)()
    load_pre_model(arch, pretrained_path, visualize=True)

    # create predictions
    prediction = eval_model(images, arch).numpy().reshape(num, -1)

    if separate:
        pre_path = "../models/fd_diff_mask_msssim_phase/fd_diff_mask_msssim_phase.model"
        arch = "filter_deep_phase"

        # load pretrained model
        arch = getattr(architecture, arch)(img_size)
        load_pre_model(arch, pre_path, visualize=True)

        # create predictions
        prediction2 = eval_model(images, arch).numpy().reshape(num, -1)

        prediction = np.stack([prediction, prediction2], 1).reshape(num, -1)

    # save input images, predictions and target images
    outpath = str(out_path) + "input.csv"
    save_indices_and_data(indices, images_x, outpath)

    outpath = str(out_path) + "predictions.csv"
    save_indices_and_data(indices, prediction, outpath)

    outpath = str(out_path) + "truth.csv"
    save_indices_and_data(indices, images_y, outpath)


if __name__ == "__main__":
    main()
