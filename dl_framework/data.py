from torch.utils.data import DataLoader
import torch
import h5py
import re
import numpy as np
from pathlib import Path


def normalize(x, m, s):
    return (x - m) / s


def do_normalisation(x, norm, pointsource=False):
    """
    :param x        Object to be normalized
    :param norm     Pandas Dataframe which includes the normalisation factors
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if norm == "none":
        return x
    # if pointsource is True:
    #     train_mean = torch.tensor(norm['train_mean'].values[0]).float()
    #     train_std = torch.tensor(norm['train_std'].values[0]).float()
    #     x = normalize(x, train_mean, train_std)
    else:
        train_mean_c0 = torch.tensor(norm["train_mean_c0"].values[0]).double()
        train_std_c0 = torch.tensor(norm["train_std_c0"].values[0]).double()
        train_mean_c1 = torch.tensor(norm["train_mean_c1"].values[0]).double()
        train_std_c1 = torch.tensor(norm["train_std_c1"].values[0]).double()
        x[:, 0] = normalize(x[:, 0], train_mean_c0, train_std_c0)
        x[:, 1] = normalize(x[:, 1], train_mean_c1, train_std_c1)

    assert not torch.isinf(x).any()
    return x


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class h5_dataset:
    def __init__(self, bundle_paths, tar_fourier, source_list, amp_phase=None):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier
        self.source_list = source_list
        self.amp_phase = amp_phase

    def __call__(self):
        return print("This is the h5_dataset class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        if self.source_list:
            x = self.open_image("x", i)
            y = self.open_image("z", i)
        else:
            x = self.open_image("x", i)
            y = self.open_image("y", i)
        return x, y

    def open_bundle(self, bundle_path, var):
        # distinguish between compressed (npz) or not compressed (h5)
        if re.search(".npz", str(bundle_path)):
            bundle = np.load(bundle_path, mmap_mode="r")
        else:
            bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = indices // self.num_img
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        # distinguish between compressed (npz) or not compressed (h5)
        if re.search(".npz", str(self.bundles[bundle[0]])):
            bundle_paths = [
                np.load(self.bundles[bundle], mmap_mode='r') for bundle in bundle_unique
            ]
        else:
            bundle_paths = [
                h5py.File(self.bundles[bundle], 'r') for bundle in bundle_unique
            ]
        bundle_paths_str = list(map(str, bundle_paths))
        data = torch.tensor(
            [
                bund[var][img]
                for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                for img in image[bundle == bundle_unique[bundle_paths_str.index(bund_str)]]
            ]
        )
        if var == "x" or self.tar_fourier is True:
            if len(i) == 1:
                data_amp, data_phase = data[:, 0], data[:, 1]

                data_channel = torch.cat([data_amp, data_phase], dim=0)
            else:
                data_amp, data_phase = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)

                data_channel = torch.cat([data_amp, data_phase], dim=1)
        else:
            if data.shape[1] == 2:
                raise ValueError("Two channeled data is used despite Fourier being False. Set Fourier to True!")
            if len(i) == 1:
                data_channel = data.reshape(data.shape[-1] ** 2)
            else:
                data_channel = data.reshape(-1, data.shape[-1] ** 2)
        return data_channel.float()


def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 2, 0)


def split_real_imag(array):
    """
    takes a complex array and returns the real and the imaginary part
    """
    return array.real, array.imag


def split_amp_phase(array):
    """
    takes a complex array and returns the amplitude and the phase
    """
    amp = np.abs(array)
    phase = np.angle(array)
    return amp, phase


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs, shuffle=True, **kwargs),
    )


class DataBunch:
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def save_bundle(path, bundle, counter, name="gs_bundle"):
    with h5py.File(str(path) + str(counter) + ".h5", "w") as hf:
        hf.create_dataset(name, data=bundle)
        hf.close()


# open and save functions should be generalized in future versions


def open_bundle(path):
    """
    open radio galaxy bundles created in first analysis step
    """
    f = h5py.File(path, "r")
    bundle = np.array(f["gs_bundle"])
    return bundle


def open_fft_bundle(path):
    """
    open radio galaxy bundles created in first analysis step
    """
    f = h5py.File(path, "r")
    x = np.array(f["x"])
    y = np.array(f["y"])
    return x, y


def get_bundles(path):
    """
    returns list of bundle paths located in a directory
    """
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


def save_fft_pair(path, x, y, name_x="x", name_y="y"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


def open_fft_pair(path):
    """
    open fft_pairs which were created in second analysis step
    """
    f = h5py.File(path, "r")
    bundle_x = np.array(f["x"])
    bundle_y = np.array(f["y"])
    return bundle_x, bundle_y


def open_fft_pair_npz(path):
    """
    open fft_pairs for files saved in .npz format
    """
    f = np.load(path)
    bundle_x = np.array(f["x"])
    bundle_y = np.array(f["y"])
    return bundle_x, bundle_y


def mean_and_std(array):
    return array.mean(), array.std()


def load_data(data_path, mode, fourier=False, source_list=False):
    """
    Load data set from a directory and return it as h5_dataset.

    Parameters
    ----------
    data_path: str
        path to data directory
    mode: str
        specify data set type, e.g. test
    fourier: bool
        use Fourier images as target if True, default is False

    Returns
    -------
    test_ds: h5_dataset
        dataset containing x and y images
    """
    bundle_paths = get_bundles(data_path)
    data = [path for path in bundle_paths if re.findall("samp_" + mode, path.name)]
    ds = h5_dataset(data, tar_fourier=fourier, source_list=source_list)
    return ds
