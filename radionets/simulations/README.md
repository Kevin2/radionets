## Tools to simulate radio interferometric observations and Gaussian sources


### gaussian_simulations

Functions to simulate radio galaxies consisting of Gaussian components, by adding 2d Gaussians to a two dimensional grid. Simulations are implemented for all pixel sizes of squared images.

Varied parameters:
* Number of components
* Jet rotation
* Flux: peak, logarithmic decrease for jet components
* Extension of components
* One- and two-sided jets 

ToDo:
* Different power law indices for jet components flux decrease
* Lorentz factor for counter jet
* Flexible (more random) distance between components
* More varied extension of jet components (simulate FRI and FRII)
* Scale sources to image size

### uv_simulations

A antenna and a source class are used to simulate radio interferometric observations. Both classes hold information about the coordinates of the antennas/sources. It is possible to create masks to simulate (u, v)-sampling and apply these mask to 
simulated (u, v)-spaces. In this way, toy monte carlo datasets are created.


### uv_plots

Functions to visualize simulated sources and (u, v)-coverages. It is possible to create gifs of array layouts and 
(u, v)-space sampling during an simulated observation. These functions are mainly used to create images for 
presentations.

### Examples: visualize_sampling 
Creates visualization plots, which can be found in the examples directory. Run `make examples`.

Explanations:
* ***gaussian_source.pdf***: A source consisting of Gaussian components. Can be one or two sided.
  The flux is decresing logarithmically towards the outer components.
* ***fft_gaussian_source.pdf***: The Fourier transformation of the Gaussian source. Low frequencies 
  are located at the center, higher frequencies at the outer parts.
* ***uv_coverage.pdf***: Simulated (uv)-coverage for a radio interferometric observation. The used antenna
  positions correspond to the layout of the [VLBA](https://science.nrao.edu/facilities/vlba/introduction-to-the-VLBA).
* ***baselines.gif/uv_coverage.gif***: These gifs visualize the sampling during a radio interferometric observation.
* ***mask.pdf***: [2d histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html)
  of the (uv)-coverage. Masks are used to sample the Fourier space of Gaussian sources. By
  doing so, incomplete measurements of the sources are simulated.
* ***sampled_frequs.pdf***: Visualizes the sampled frequencies of the source's Fourier space.
* ***recons_source.pdf***: The inverse Fourier transformation of the incomplete sample.