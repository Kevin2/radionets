from tqdm import tqdm
from radionets.simulations.utils import open_mnist, prepare_mnist_bundles


def mnist_fft(resource_path, out_path, size, bundle_size, noise=False):
    """
    Load MNIST dataset, split it into bundles, resize and compute fft.
    Bundles are saved to hdf5 files.

    Parameters
    ----------
    data_path: str
        path to MNIST pickle file
    out_path: str
        directory where bundles are saved

    Options
    -------
    size: int
        image size
    bundle_size: int
        number of images in one bundle
    noise: bool
        if true: images are noised before fft
    """
    train_x, valid_x = open_mnist(resource_path)

    print("\nCreating train set.\n")
    bundles_train = train_x.reshape(
        int(len(train_x) / bundle_size), bundle_size, 28, 28
    )
    [
        prepare_mnist_bundles(bund, out_path, "train", pixel=size, noise=noise)
        for bund in tqdm(bundles_train)
    ]

    print("\nCreating valid set.\n")
    bundles_evaluation = valid_x.reshape(
        int(len(valid_x) / bundle_size), bundle_size, 28, 28
    )
    bundles_valid = bundles_evaluation[0:-2]
    [
        prepare_mnist_bundles(bund, out_path, "valid", pixel=size, noise=noise)
        for bund in tqdm(bundles_valid)
    ]

    print("\nCreating test set.\n")
    bundles_test = bundles_evaluation[-2:]
    [
        prepare_mnist_bundles(bund, out_path, "test", pixel=size, noise=noise)
        for bund in tqdm(bundles_test)
    ]
