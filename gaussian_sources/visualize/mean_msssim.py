import click
import numpy as np
from tqdm import tqdm

import dl_framework.architectures as architecture
import pytorch_msssim
import torch
from dl_framework.data import load_data
from dl_framework.inspection import eval_model
from dl_framework.model import load_pre_model
from gaussian_sources.inspection import save_indices_and_data
from torch.utils.data import DataLoader


def fft(amp, phase):
    amp = 10 ** (10 * amp - 10) - 1e-10
    a = amp * np.cos(phase)
    b = amp * np.sin(phase)
    comp = a + b * 1j

    fft = np.fft.ifft2(comp)
    return np.abs(fft)


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("norm_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("pretrained_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-fourier", type=bool, required=True)
def main(
    data_path, norm_path, arch, pretrained_path, out_path, fourier,
):
    # Load data and create DataLoader
    test_ds = load_data(data_path, "test", fourier=fourier)
    loader = DataLoader(test_ds, batch_size=128, shuffle=True)
    indices = np.arange(0, len(test_ds))

    # load pretrained model
    img_size = test_ds[0][0][0].shape[1]
    if arch == "filter_deep" or "filter_deep_amp":
        arch = getattr(architecture, arch)(img_size)
    else:
        arch = getattr(architecture, arch)()
    load_pre_model(arch, pretrained_path, visualize=True)

    values = []

    # iterate trough DataLoader
    for i, (xb, yb) in enumerate(tqdm(loader)):
        images = xb
        images_y = yb.numpy()

        # create predictions
        prediction = eval_model(images, arch).numpy()

        for i in range(len(xb)):
            img_truth = fft(images_y[i][0], images_y[i][1])
            img_pred = fft(prediction[i][0], prediction[i][1])

            # scale all images to the same magnitude
            if img_truth.max() < 0.099999:
                magnitude = int(np.round(np.abs(np.log10(img_truth.max()))))
                img_pred = img_pred * 10 ** magnitude
                img_truth = img_truth * 10 ** magnitude

            # convert to 32-bit torch tensors
            tensor_pred = torch.tensor(np.float32(img_pred)).unsqueeze(0).unsqueeze(1)
            tensor_truth = torch.tensor(np.float32(img_truth)).unsqueeze(0).unsqueeze(1)
            # calculate msssim
            msssim = pytorch_msssim.msssim(tensor_pred, tensor_truth, normalize="None")
            # append to list
            values.append(msssim)

    # Save msssim values as csv and print the mean value
    outpath = str(out_path) + "msssim.csv"
    save_indices_and_data(indices, values, outpath)
    print(np.mean(values))


if __name__ == "__main__":
    main()
