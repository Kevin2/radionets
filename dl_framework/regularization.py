import torch
import torch.nn.functional as F
import kornia
from math import pi
from dl_framework.model import fft, euler, flatten


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


def PCA(image):
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

    a = torch.arange(0, pix, 1).cuda()
    grid_x, grid_y = torch.meshgrid(a, a)
    x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
    y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
    value = image.reshape(-1, pix ** 2)
    return x_coords, y_coords, value


def torch_abs(data):
    """Compute absolute value of complex number for h5 datasets.

    Parameters
    ----------
    data : h5_dataset_like
        data needs shape (bs, c, h, w), c[0]: real values, c[1]: imaginary values

    Returns
    -------
    tensor
        tensor in local space (bs, h, w)
    """
    return torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)


def pad(inp_tensor, n):
    """Pads n 0 pixel to tensor. Batch size is preserved.

    Parameters
    ----------
    inp_tensor : tensor
        input tensor
    n : int
        number of pad pixel

    Returns
    -------
    tensor
        padded tensor
    """
    return F.pad(input=inp_tensor, pad=(n, n, n, n), mode="constant", value=0)


def rot(img, angle):
    """Rotates batch of images (bs, h, w) by angle (bs, 1) using kornia.

    Parameters
    ----------
    img : 3dtesnor
        images to be rotated
    angle : 2dtensor
        tensor holding rotation angles, in degrees

    Returns
    -------
    tensor
        tensor of rotated images
    """
    img_pad = pad(img, n=15)
    if len(img_pad.shape) == 3:
        img_pad = img_pad.unsqueeze(1)

    bs = img_pad.shape[0]
    center = torch.ones(bs, 2).cuda()
    center[..., 0] = img_pad.shape[3] / 2
    center[..., 1] = img_pad.shape[2] / 2
    scale = torch.ones(bs).cuda()

    M = kornia.get_rotation_matrix2d(center, angle, scale)

    _, _, h, w = img_pad.shape
    img_warped = kornia.warp_affine(img_pad, M, dsize=(h, w))
    return img_warped


def calc_jet_angle(img):
    """Calculates rotation angle and linear params of central jet.

    Parameters
    ----------
    img : 3dtensor
        tensor holding images

    Returns
    -------
    tensors
        m, n, rotation angle
    """
    img = img.clone()

    # delete outer parts
    img[:, 0:10] = 0
    img[:, 53:63] = 0
    img[:, :, 0:10] = 0
    img[:, :, 53:63] = 0

    for i in img:
        # only use brightest pixel
        i[i < i.max() * 0.4] = 0

    # pca
    y, x, alpha = PCA(img)

    # Get line of major component
    m = torch.tan(pi / 2 - alpha)
    n = y - m.unsqueeze(-1) * x
    alpha = (2 * pi - alpha) * 180 / pi
    return m, n, alpha


def calc_spec(img_rot):
    """Sums along axis perpendicular to jet axis to generate 1d spec of image.

    Parameters
    ----------
    img_rot : 4dtensor (bs, c, h, w)
        tensor with rotated images

    Returns
    -------
    tensor
        tensor with 1d spectra (bs, s)
    """
    s = img_rot.sum(axis=2).squeeze(1)
    return s


def inv_fft(amp, phase):
    """Combine amp and phase tensor, do inverse FFT and calculate abs image.

    Parameters
    ----------
    amp : tensor
        tensor with amp images
    phase : tensor
        tensor with phase images

    Returns
    -------
    tensor
        tensor with images in local space
    """
    x = torch.cat([amp, phase], dim=1)
    comp = flatten(euler(x))
    fft_comp = fft(comp)
    img = torch_abs(fft_comp)
    return img
