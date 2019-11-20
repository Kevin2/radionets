import click
from mnist_cnn.visualize.utils import (load_architecture, load_pre_model,
                                       eval_model)
from mnist_cnn.preprocessing import get_h5_data
from mnist_cnn.inspection import get_normalization
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


@click.command()
@click.argument('arch_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('in_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-index', type=int, required=False)
def main(arch_path, pretrained_path, in_path,
         norm_path, out_path, index=None):
    i = index
    x_valid, y_valid = get_h5_data(in_path, columns=['x_valid', 'y_valid'])
    img = torch.tensor(x_valid[i])
    img_log = torch.log(img)
    img_reshaped = img_log.view(1, 1, 64, 64)
    img_normed = get_normalization(img_reshaped, norm_path)

    model = load_architecture(arch_path)
    model_pre = load_pre_model(model, pretrained_path)
    prediction = eval_model(img_normed, model_pre)

    print(prediction)
    print(prediction.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    inp = img_normed.reshape(64, 64).numpy()
    im1 = ax1.imshow(inp, cmap='RdBu', vmin=-inp.max(), vmax=inp.max())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    pred_img = prediction.reshape(64, 64).numpy()
    im2 = ax2.imshow(pred_img)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im3 = ax3.imshow(y_valid[i].reshape(64, 64))
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    main()
