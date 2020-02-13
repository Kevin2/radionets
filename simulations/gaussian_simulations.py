import numpy as np
from scipy.ndimage import gaussian_filter
from dl_framework.data import save_bundle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_rot_mat(alpha):
    '''
    Create 2d rotation matrix for given alpha
    alpha: rotation angle in rad
    '''
    rot_mat = np.array([
        [np.cos(alpha), np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot=0, center=None):
    ''' Create a gaussian component on a 2d grid

    x: x coordinates of 2d grid
    y: y coordinates of 2d grid
    flux: peak amplitude of component
    x_fwhm: full-width-half-maximum in x direction
    y_fwhm: full-width-half-maximum in y direction
    rot: rotation of component
    center: center of component
    '''
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(np.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2

    gauss = flux * np.exp(-((x_0 - x)**2/(2*(x_fwhm)**2) +
                          (y_0 - y)**2 / (2*(y_fwhm)**2)))
    return gauss


def create_grid(pixel):
    ''' Create a square 2d grid

    pixel: number of pixel in x and y
    '''
    x = np.linspace(0, pixel-1, num=pixel)
    y = np.linspace(0, pixel-1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.zeros(X.shape) + 1e-10, X, Y])
    return grid


def add_gaussian(grid, amp, x, y, sig_x, sig_y, rot):
    '''
    Takes a grid and adds a Gaussian component relative to the center

    grid: 2d grid
    amp: amplitude
    x: x position, will be calculated rel. to center
    y: y position, will be calculated rel. to center
    sig_x: standard deviation in x
    sig_y: standard deviation in y
    '''
    cent = np.array([len(grid[0])//2 + x, len(grid[0])//2 + y])
    X = grid[1]
    Y = grid[2]
    gaussian = grid[0]
    gaussian += gaussian_component(
                                X,
                                Y,
                                amp,
                                sig_x,
                                sig_y,
                                rot,
                                center=cent,
    )

    return gaussian


def create_gaussian_source(comps, amp, x, y, sig_x, sig_y,
                           rot, grid, sides=0, blur=True):
    '''
    takes grid
    side: one-sided or two-sided
    core dominated or lobe dominated
    number of components
    angle of the jet

    components should not have too big gaps between each other
    '''
    if sides == 1:
        comps += comps-1
        amp = np.append(amp, amp[1:])
        x = np.append(x, -x[1:])
        y = np.append(y, -y[1:])
        sig_x = np.append(sig_x, sig_x[1:])
        sig_y = np.append(sig_y, sig_y[1:])

    for i in range(comps):
        source = add_gaussian(
            grid=grid,
            amp=amp[i],
            x=x[i],
            y=y[i],
            sig_x=sig_x[i],
            sig_y=sig_y[i],
            rot=rot,
        )
    if blur is True:
        source = gaussian_filter(source, sigma=1.5)
    return source


def gauss_paramters():
    '''
    get random set of Gaussian parameters
    '''
    # random number of components between 4 and 9
    comps = np.random.randint(4, 7)  # decrease for smaller images

    # start amplitude between 10 and 1e-3
    amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # if start amp is 0, draw a new number
    while amp_start == 0:
        amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # logarithmic decrease to outer components
    amp = np.array([amp_start/np.exp(i) for i in range(comps)])

    # linear distance bestween the components
    x = np.arange(0, comps) * 5
    y = np.zeros(comps)

    # extension of components
    # random start value between 1 - 0.375 and 1 - 0
    # linear distance between components
    # distances scaled by factor between 0.25 and 0.5
    # randomnized for each sigma
    off1 = (np.random.random() + 0.5) / 4
    off2 = (np.random.random() + 0.5) / 4
    fac1 = (np.random.random() + 1) / 4
    fac2 = (np.random.random() + 1) / 4
    sig_x = (np.arange(1, comps+1) - off1) * fac1
    sig_y = (np.arange(1, comps+1) - off2) * fac2

    # jet rotation
    rot = np.random.randint(0, 360)
    # jet one- or two-sided
    sides = np.random.randint(0, 2)

    return comps, amp, x, y, sig_x, sig_y, rot, sides


def gaussian_source(img_size):
    grid = create_grid(img_size)
    comps, amp, x, y, sig_x, sig_y, rot, sides = gauss_paramters()
    s = create_gaussian_source(comps, amp, x, y, sig_x, sig_y,
                               rot, grid, sides, blur=True)
    return s


def create_bundle(img_size, bundle_size):
    bundle = np.array([gaussian_source(img_size) for i in range(bundle_size)])
    return bundle


def create_n_bundles(num_bundles, bundle_size, img_size, out_path):
    for j in tqdm(range(num_bundles)):
        bundle = create_bundle(img_size, bundle_size)
        save_bundle(out_path, bundle, j)


def get_noise(image, scale, mean=0, std=1):
    return np.random.normal(mean, std, size=image.shape) * scale


def add_noise(bundle, mean=0, std=1, index=0, preview=False):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked.
    """
    bundle_noised = np.array([img + get_noise(img, (img.max()*0.05)) for img in bundle])

    if preview:
        for i in range(10):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

            ax1.set_title(r'Original')
            im1 = ax1.imshow(bundle[i])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')

            ax2.set_title(r"Noised")
            im2 = ax2.imshow(bundle_noised[i])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.show()
            # fig.savefig('data/plots/input_plot_{}.pdf'.format(index), pad_inches=0,
            #             bbox_inches='tight')

    return bundle_noised
