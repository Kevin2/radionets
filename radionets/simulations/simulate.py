from radionets.simulations.mnist import mnist_fft
from radionets.simulations.gaussians import simulate_gaussian_sources
from radionets.simulations.sampling import sample_frequencies
import click
from pathlib import Path


def create_fft_images(sim_conf):
    """
    Create fft source images and save them to h5 files.

    Parameters
    ----------
    sim_conf : dict
        dict holding simulation parameters
    """
    if sim_conf["type"] == "mnist":
        mnist_fft(
            resource_path=sim_conf["resource"],
            out_path=sim_conf["data_path"],
            size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            noise=sim_conf["noise"],
        )
    if sim_conf["type"] == "gaussians":
        for opt in ["train", "valid", "test"]:
            simulate_gaussian_sources(
                data_path=sim_conf["data_path"],
                option=opt,
                num_bundles=sim_conf["bundles_" + str(opt)],
                bundle_size=sim_conf["bundle_size"],
                img_size=sim_conf["img_size"],
                num_comp_ext=sim_conf["num_components"],
                num_pointlike=sim_conf["num_pointlike_gaussians"],
                num_pointsources=sim_conf["num_pointsources"],
                noise=sim_conf["noise"],
                source_list=sim_conf["source_list"],
                seg_map=sim_conf["seg_map"],
            )


def sample_fft_images(sim_conf):
    """
    check for fft files
    keep fft_files?
    """
    sample_frequencies(
        data_path=sim_conf["data_path"],
        amp_phase=sim_conf["amp_phase"],
        real_imag=sim_conf["real_imag"],
        specific_mask=sim_conf["specific_mask"],
        antenna_config=sim_conf["antenna_config_path"],
        lon=sim_conf["lon"],
        lat=sim_conf["lat"],
        steps=sim_conf["steps"],
        fourier=sim_conf["fourier"],
        compressed=sim_conf["compressed"],
        source_list=sim_conf["source_list"],
        seg_map=sim_conf["seg_map"],
    )
    if sim_conf["keep_fft_files"] is not True:
        if click.confirm("Do you really want to delete the fft_files?", abort=False):
            fft = {
                p
                for p in Path(sim_conf["data_path"]).rglob(
                    "*fft*." + str(sim_conf["data_format"])
                )
                if p.is_file()
            }
            [p.unlink() for p in fft]
