import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from dl_framework.data import save_fft_pair, save_fft_pair_list
from simulations.utils import adjust_outpath, add_noise


def simulate_gaussian_sources(
    data_path,
    option,
    num_bundles,
    bundle_size,
    img_size,
    num_comp_ext,
    num_pointlike,
    num_pointsources,
    noise,
    source_list
):
    for i in tqdm(range(num_bundles)):
        grid = create_grid(img_size, bundle_size)
        ext_gaussian = 0
        pointlike = 0
        pointsource = 0
        list_sources = 0

        if num_comp_ext is not None:
            ext_gaussian = create_ext_gauss_bundle(grid)
        if num_pointlike is not None:
            pointlike = create_gauss(grid[:, 0], bundle_size, num_pointlike, True, source_list)
        if num_pointsources is not None:
            pointsource = gauss_pointsources(grid[:, 0], bundle_size, num_pointsources)
        if source_list:
            list_sources = pointlike[1]
            pointlike = pointlike[0]

            ##Sort the source list along x##

#            a = list_sources[:,:,0]
#            indices = np.argsort(a)
#            for j in range(bundle_size):
#                list_sources[j] = list_sources[j, indices[j], :]

        bundle = ext_gaussian + pointlike + pointsource
        images = bundle.copy()

        if noise:
            images = add_noise(images)

        bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in images])
        path = adjust_outpath(data_path, "/fft_" + option)
        save_fft_pair_list(path, bundle_fft, bundle, list_sources)



def create_grid(pixel, bundle_size):
    """
    Creates a square 2d grid.

    Parameters
    ----------
    pixel: int
        number of pixel in x and y

    Returns
    -------
    grid: ndarray
        2d grid with 1e-10 pixels, X meshgrid, Y meshgrid
    """
    x = np.linspace(0, pixel - 1, num=pixel)
    y = np.linspace(0, pixel - 1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.zeros(X.shape) + 1e-10, X, Y])
    grid = np.repeat(
        grid[None, :, :, :],
        bundle_size,
        axis=0,
    )
    return grid


# draw random parameters for extended gaussian sources


def gauss_paramters():
    """
    Generate a random set of Gaussian parameters.

    Parameters
    ----------
    None

    Returns
    -------
    comps: int
        Number of components
    amp: float
        Amplitude of the core component
    x: array
        x positions of components
    y: array
        y positions of components
    sig_x:
        standard deviation in x
    sig_y:
        standard deviation in y
    rot: int
        rotation in degree
    sides: int
        0 for one-sided and 1 for two-sided jets
    """
    # random number of components between 4 and 9
    comps = np.random.randint(4, 7)  # decrease for smaller images

    # start amplitude between 10 and 1e-3
    amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # if start amp is 0, draw a new number
    while amp_start == 0:
        amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # logarithmic decrease to outer components
    amp = np.array([amp_start / np.exp(i) for i in range(comps)])

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
    sig_x = (np.arange(1, comps + 1) - off1) * fac1
    sig_y = (np.arange(1, comps + 1) - off2) * fac2

    # jet rotation
    rot = np.random.randint(0, 360)
    # jet one- or two-sided
    sides = np.random.randint(0, 2)

    return comps, amp, x, y, sig_x, sig_y, rot, sides


def create_rot_mat(alpha):
    """
    Create 2d rotation matrix for given alpha

    Parameters
    ----------
    alpha: float
        rotation angle in rad

    Returns
    -------
    rot_mat: 2darray
        2d rotation matrix
    """
    rot_mat = np.array([[np.cos(alpha), np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot=0, center=None):
    """
    Adds a gaussian component to a 2d grid.

    Parameters
    ----------
    x: 2darray
        x coordinates of 2d meshgrid
    y: 2darray
        y coordinates of 2d meshgrid
    flux: float
        peak amplitude of component
    x_fwhm: float
        full-width-half-maximum in x direction
    y_fwhm: float
        full-width-half-maximum in y direction
    rot: int
        rotation of component in degree
    center: 2darray
        enter of component

    Returns
    -------
    gauss: 2darray
        2d grid with gaussian component
    """
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(np.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2

    gauss = flux * np.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    return gauss


def add_gaussian(grid, amp, x, y, sig_x, sig_y, rot):
    """
    Takes a grid and adds n Gaussian component relative to the center.

    Parameters
    ----------
    grid: 2darray
        2d grid
    amp: float
        amplitude of gaussian component
    x: float
        x position, will be calculated rel. to center
    y: float
        y position, will be calculated rel. to center
    sig_x: float
        standard deviation in x
    sig_y: float
        standard deviation in y
    rot: int
        rotation in degree

    Returns
    -------
    gaussian: 2darray
        grid with gaussian component
    """
    cent = np.array([len(grid[0]) // 2 + x, len(grid[0]) // 2 + y])
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


def create_gaussian_source(
    grid, comps, amp, x, y, sig_x, sig_y, rot, sides=0, blur=True
):
    """
    Combines Gaussian components on a 2d grid to create a Gaussian source

    takes grid
    side: one-sided or two-sided
    core dominated or lobe dominated
    number of components
    angle of the jet

    Parameters
    ----------
    grid: ndarray
        2dgrid + X and Y meshgrid
    comps: int
        number of components
    amp: 1darray
        amplitudes of components
    x: 1darray
        x positions of components
    y: 1darray
        y positions of components
    sig_x: 1darray
        standard deviations of components in x
    sig_y: 1darray
        standard deviations of components in y
    rot: int
        rotation of the jet in degree
    sides: int
        0 one-sided, 1 two-sided jet
    blur: bool
        use Gaussian filter to blur image

    Returns
    -------
    source: 2darray
        2d grid containing Gaussian source

    Comments
    --------
    components should not have too big gaps between each other
    """
    if sides == 1:
        comps += comps - 1
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


def gaussian_source(grid):
    """
    Creates random Gaussian source parameters and returns an image
    of a Gaussian source.

    Parameters
    ----------
    grid: nd array
        array holding 2d grid and axis for one image

    Returns
    -------
    s: 2darray
       Image containing a simulated Gaussian source.
    """
    # grid = create_grid(img_size)
    comps, amp, x, y, sig_x, sig_y, rot, sides = gauss_paramters()
    s = create_gaussian_source(
        grid, comps, amp, x, y, sig_x, sig_y, rot, sides, blur=True
    )
    return s


def create_ext_gauss_bundle(grid):
    """
    Creates a bundle of Gaussian sources.

    Parameters
    ----------
    grid: nd array
        array holding 2d grid and axis for whole bundle

    Returns
    -------
    bundle ndarray
        bundle of Gaussian sources
    """
    bundle = np.array([gaussian_source(g) for g in grid])
    return bundle


# pointlike gaussians


def create_gauss(img, N, sources, spherical, source_list):
    # img = [img]
    mx = np.random.randint(1, 63, size=(N, sources))
    my = np.random.randint(1, 63, size=(N, sources))
    amp = (np.random.randint(1, 100, size=(N)) *1/10* np.random.randint(5, 10)) / 1e2

    if spherical:
        sx = np.random.randint(1, 15, size=(N, sources)) / 10
        sy = sx
    else:
        sx = np.random.randint(1, 15, size=(N, sources))
        sy = np.random.randint(1, 15, size=(N, sources))
        theta = np.random.randint(0, 360, size=(N, sources))
    
    s = np.zeros((N,sources,5))
    for i in range(N):
        for j in range(sources):
            g = gauss(mx[i, j], my[i, j], sx[i, j], sy[i, j], amp[i])
            s[i,j] = np.array([mx[i,j],my[i,j],sx[i,j],sy[i,j],amp[i]])
            if spherical:
                img[i] += g
            else:
                # rotation around center of the source
                padX = [g.shape[0] - mx[i, j], mx[i, j]]
                padY = [g.shape[1] - my[i, j], my[i, j]]
                imgP = np.pad(g, [padY, padX], "constant")
                imgR = ndimage.rotate(imgP, theta[i, j], reshape=False)
                imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
                img[i] += imgC
    if source_list:
        return img, s
    else:
        return img


# pointsources


def gauss_pointsources(img, num_img, sources):
    mx = np.random.randint(0, 63, size=(num_img, sources))
    my = np.random.randint(0, 63, size=(num_img, sources))
    amp = (np.random.randint(0, 100, size=(num_img)) * np.random.random()) / 1e2
    sigma = 0.05
    for i in range(num_img):
        targets = np.random.randint(2, sources + 1)
        for j in range(targets):
            g = gauss(mx[i, j], my[i, j], sigma, sigma, amp[i])
            img[i] += g
    return np.array(img)


def gauss(mx, my, sx, sy, amp=0.01):
    x = np.arange(63)[None].astype(np.float)
    y = x.T
    return amp * np.exp(-((y - my) ** 2) / sy).dot(np.exp(-((x - mx) ** 2) / sx))
