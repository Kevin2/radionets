import matplotlib.pyplot as plt

from radionets.simulations.gaussian_simulations import gaussian_source
from radionets.simulations.uv_plots import (
    FT,
    plot_source,
    animate_baselines,
    animate_uv_coverage,
    plot_uv_coverage,
    apply_mask,
)
from radionets.simulations.uv_simulations import (
    antenna,
    source,
    get_antenna_config,
    create_mask,
    get_uv_coverage,
)


sim_source = gaussian_source(64)

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
s.propagate()
animate_baselines(s, ant, "examples/baselines", 5)
animate_uv_coverage(s, ant, "examples/uv_coverage", 5)

u, v, steps = get_uv_coverage(s, ant, iterate=False)

fig = plt.figure(figsize=(6, 6), dpi=100)
plot_uv_coverage(u, v)
plt.ylim(-5e8, 5e8)
plt.xlim(-5e8, 5e8)
plt.savefig("examples/uv_coverage.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111)
mask = create_mask(u, v)
plt.imshow(mask, cmap="inferno")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")
plt.savefig("examples/mask.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

sampled_freqs = apply_mask(FT(sim_source), mask)

plot_source(sampled_freqs, ft=False, log=True)
plt.xlabel("u")
plt.ylabel("v")
plt.savefig("examples/sampled_freqs.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

plot_source(sampled_freqs, ft=True, log=True)
plt.xlabel("l")
plt.ylabel("m")
plt.savefig("examples/recons_source.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)
