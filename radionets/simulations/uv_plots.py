import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from radionets.simulations.uv_simulations import get_uv_coverage
from matplotlib.colors import LogNorm


def plot_uv_coverage(u, v):
    """
    Visualize (uv)-coverage

    Parameters
    ----------
    u: 1darray
        array of u coordinates
    v: 1darray
        array of v coordinates

    Returns
    -------
    None
    """
    plt.plot(u, v, marker="o", linestyle="none", markersize=2, color="#1f77b4")
    plt.xlabel(r"u / $\lambda$", fontsize=16)
    plt.ylabel(r"v / $\lambda$", fontsize=16)
    plt.tight_layout()


def plot_baselines(antenna):
    """
    Visualize baselines of an antenna layout

    Parameters
    ----------
    antenna: antenna class object
        class object with antenna positions and baselines between telescopes

    Returns
    -------
    None
    """
    x_base, y_base = antenna.get_baselines()
    plt.plot(
        x_base,
        y_base,
        linestyle="--",
        color="#2ca02c",
        zorder=0,
        label="Baselines",
        alpha=0.35,
    )
    plt.tight_layout()


def plot_antenna_distribution(source_lon, source_lat, source, antenna, baselines=False):
    """
    Visualize antenna distribution seen from a specific source position

    Parameters
    ----------
    source_lon: float
        longitude of the source
    source_lat: float
        latitude of the source
    source: source class object
        class object containing source position
    antenna: antenna class object
        class object with antenna positions and baselines between telescopes
    baselines: bool
        enable baseline plotting

    Returns
    -------
    None
    """
    x, y, z = source.to_ecef(val=[source_lon, source_lat])  # only use source ?
    x_enu_ant, y_enu_ant = antenna.to_enu(x, y, z)

    ax = plt.axes(projection=ccrs.Orthographic(source_lon, source_lat))
    ax.set_global()
    ax.coastlines()

    plt.plot(
        x_enu_ant,
        y_enu_ant,
        marker="o",
        markersize=6,
        color="#1f77b4",
        linestyle="none",
        label="Antenna positions",
    )
    plt.plot(
        x,
        y,
        marker="*",
        linestyle="none",
        color="#ff7f0e",
        markersize=15,
        transform=ccrs.Geodetic(),
        zorder=10,
        label="Source",
    )

    if baselines is True:
        plot_baselines(antenna)

    plt.legend(fontsize=16, markerscale=1.5)
    plt.tight_layout()


def animate_baselines(source, antenna, filename, fps=5):
    """
    Create gif to animate change of baselines during an observation

    Parameters
    ----------
    source: source class object
        class object containing source position
    antenna: antenna class object
        class object with antenna positions and baselines between telescopes
    filename: str
        name of the created gif
    fps: int
        frames per seconds of the gif

    Returns
    -------
    None
    """
    s_lon = source.lon_prop
    s_lat = source.lat_prop

    fig = plt.figure(figsize=(6, 6), dpi=100)

    def init():
        pass

    def update(frame):
        lon = s_lon[frame]
        lat = s_lat[frame]
        plot_antenna_distribution(lon, lat, source, antenna, baselines=True)

    ani = FuncAnimation(
        fig, update, frames=len(s_lon), init_func=init, interval=1000 / fps
    )

    ani.save(str(filename) + ".gif", writer=PillowWriter(fps=fps))


def animate_uv_coverage(source, antenna, filename, fps=5):
    """
    Create gif to animate improvement of (uv)-coverage during an observation

    Parameters
    ----------
    source: source class object
        class object containing source position
    antenna: antenna class object
        class object with antenna positions and baselines between telescopes
    filename: str
        name of the created gif
    fps: int
        frames per seconds of the gif

    Returns
    -------
    None
    """
    u, v, steps = get_uv_coverage(source, antenna, iterate=True)

    fig = plt.figure(figsize=(6, 6), dpi=100)

    def init():
        pass

    def update(frame):
        plot_uv_coverage(u[frame], v[frame])
        plt.ylim(-5e8, 5e8)
        plt.xlim(-5e8, 5e8)

    ani = FuncAnimation(
        fig, update, frames=steps, init_func=init, interval=0.001, repeat=False
    )

    ani.save(str(filename) + ".gif", dpi=80, writer=PillowWriter(fps=fps))


def plot_source(img, ft=False, log=False):
    """
    Visualize a radio source

    Parameters
    ----------
    img: 2darray
        values of Gaussian source
    ft: bool
        if True, the Fourier transformation (frequency space) of the image is plotted

    Returns
    -------
    None
    """
    plt.rcParams.update({"font.size": 18})
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    if ft is False:
        img = img
        ax.set_xlabel("l")
        ax.set_ylabel("m")
        if log is True:
            s = ax.imshow(img, cmap="inferno", norm=LogNorm(vmin=1e-8, vmax=img.max()))
        else:
            s = ax.imshow(img, cmap="inferno")
        fig.colorbar(s, label="Intensity / a.u.")
    else:
        img = np.abs(FT(img))
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        if log is True:
            s = ax.imshow(img, cmap="inferno", norm=LogNorm())
        else:
            s = ax.imshow(img, cmap="inferno")
        fig.colorbar(s, label="Amplitude / a.u.")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    plt.tight_layout()


def FT(img):
    """
    Computes the 2d Fourier trafo of an image

    Parameters
    ----------
    img: 2darray
        values of Gaussian source

    Returns
    -------
    out: 2darray
        Fourier transform of input array
    """
    return np.fft.fftshift(np.fft.fft2(img))


def apply_mask(img, mask):
    """
    Applies a boolean mask to a 2d image

    Parameters
    ----------
    img: 2darray
        values of Gaussian source
    mask: bool
        mask for sampling frequencies

    Returns
    -------
    out: 2darray
        array with sampled frequencies
    """
    img = img.copy()
    img[~mask.astype(bool)] = 0
    return img
