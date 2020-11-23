import numpy as np
import matplotlib.pyplot as plt

from radionets.simulations.gaussian_simulations import gaussian_source
from radionets.simulations.uv_plots import (
    FT,
    plot_source,
    # animate_baselines,
    # animate_uv_coverage,
    plot_uv_coverage,
    apply_mask,
    plot_antenna_distribution,
    plot_mask,
)
from radionets.simulations.uv_simulations import (
    antenna,
    source,
    get_antenna_config,
    create_mask,
    get_uv_coverage,
)


sim_source = gaussian_source(63)

plot_source(sim_source, ft=False, log=True)
plt.savefig(
    "examples/gaussian_source.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

plot_source(sim_source, ft=True, log=True)
plt.savefig(
    "examples/fft_gaussian_source.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

fft = FT(sim_source)

plot_source(fft.real, ft=False, log=False)
plt.xlabel("u")
plt.ylabel("v")
plt.savefig(
    "examples/fft_gaussian_source_real.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

plot_source(fft.imag, ft=False, log=False)
plt.xlabel("u")
plt.ylabel("v")
plt.savefig(
    "examples/fft_gaussian_source_imag.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

plot_source(np.abs(fft), ft=False, log=True)
plt.xlabel("u")
plt.ylabel("v")
plt.savefig(
    "examples/fft_gaussian_source_amp.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

plot_source(np.angle(fft), ft=False, log=False)
plt.xlabel("u")
plt.ylabel("v")
plt.savefig(
    "examples/fft_gaussian_source_phase.pdf",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.01,
)

ant = antenna(*get_antenna_config("./layouts/vlba.txt"))
s = source(-80, 40)

# plot beginning of uv_coverage
s.propagate(num_steps=1)
u, v, steps = get_uv_coverage(s, ant, iterate=False)
fig = plt.figure(figsize=(6, 6), dpi=100)
plot_uv_coverage(u, v)
plt.ylim(-5e8, 5e8)
plt.xlim(-5e8, 5e8)
plt.tick_params(axis="both", labelsize=20)
plt.savefig(
    "examples/uv_coverage_begin.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05
)

s.propagate()
# animate_baselines(s, ant, "examples/baselines", 5)
# animate_uv_coverage(s, ant, "examples/uv_coverage", 5)
s_lon = s.lon_prop
s_lat = s.lat_prop
plot_antenna_distribution(s_lon[0], s_lat[0], s, ant, baselines=True)
plt.savefig("examples/baselines.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)
# plot also end of baseline simulation
plot_antenna_distribution(s_lon[-1], s_lat[-1], s, ant, baselines=True)
plt.savefig("examples/baselines_end.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

u, v, steps = get_uv_coverage(s, ant, iterate=False)

fig = plt.figure(figsize=(6, 6), dpi=100)
plot_uv_coverage(u, v)
plt.ylim(-5e8, 5e8)
plt.xlim(-5e8, 5e8)
plt.tick_params(axis="both", labelsize=20)
plt.savefig("examples/uv_coverage.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)

fig = plt.figure(figsize=(8, 6), dpi=100)
mask = create_mask(u, v)
plot_mask(fig, mask)
plt.savefig("examples/mask.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

sampled_freqs = apply_mask(FT(sim_source), mask)

plot_source(sampled_freqs, ft=False, log=True)
plt.xlabel("u", fontsize=20)
plt.ylabel("v", fontsize=20)
plt.savefig("examples/sampled_freqs.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)

plot_source(sampled_freqs, ft=True, log=True, ft2=True)
plt.xlabel("l", fontsize=20)
plt.ylabel("m", fontsize=20)
plt.savefig("examples/recons_source.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)
