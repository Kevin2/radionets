import matplotlib.pyplot as plt
import torch


def plot_mnist(img):
    plt.imshow(img, cmap="RdGy", vmin=-img.max(), vmax=img.max())
    plt.colorbar(label='Amplitude')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.tight_layout()


def eval_model(img, model):
    model.eval()
    with torch.no_grad():
        pred = model(img.float())
    return pred
