import numpy as np
import astropy.coordinates as ac
import radionets.simulations.layouts.layouts as layouts


class source:
    """
    Source class that holds longitude and latitude information.
    Can be converted to geocentric coordinates. Position of source
    can be propagated to simulate an ongoing observation.
    """

    def __init__(self, lon, lat):
        """
        Paramters
        ---------
        lon: float
            longitude of source
        lat: float
            latitude of source
        """
        self.lon = lon
        self.lat = lat

    def to_ecef(self, val=None, prop=False):
        """
        Converts from geodetic to geocentric coordinates

        Parameters
        ----------
        val: list with [lon, lat]
            A specific geodetic position
        prop: bool
            use True on lists of propagated source positions, default is False

        Returns
        -------
        x, y, z: 1darrays
            Positions in geocentric coordinates
        """
        if prop is True:
            quant = ac.EarthLocation(self.lon_prop, self.lat_prop).to_geocentric()
        else:
            quant = ac.EarthLocation([self.lon], [self.lat]).to_geocentric()
        if val is not None:
            quant = ac.EarthLocation([val[0]], [val[1]]).to_geocentric()
        x = quant[0].value
        y = quant[1].value
        z = quant[2].value
        return x, y, z

    def propagate(self, num_steps=None, multi_pointing=False):
        """
        Propagates a source position with random parameters

        Parameters
        ----------
        num_steps: int
            number of propagation steps
        multi_pointing: bool
            when True observation blocks are simulated, default is False

        Returns
        -------
        lon: 1darray
            array with propagated lons
        lat: 1darray
            array with propagated lats
        """
        if num_steps is None:
            num_steps = np.random.randint(30, 60)
        lon_start = self.lon
        lon_stop = lon_start - num_steps
        lon_step = 0.5
        lon = np.arange(lon_stop, lon_start, lon_step)
        lon = lon[::-1]

        lat_start = self.lat
        direction = np.sign(np.random.randint(0, 1) - 0.5)
        lat_stop = round((lat_start + direction * num_steps / 50) + 0.005, 3)
        lat_step = 0.01
        if lat_start > lat_stop:
            lat = np.arange(lat_stop, lat_start, lat_step)
            lat = lat[::-1]
        else:
            lat = np.arange(lat_start, lat_stop, lat_step)

        if len(lon) != len(lat):
            raise ValueError("Length of lon and lat are different!")

        if multi_pointing is True:
            lon = self.mod_delete(lon, 5, 10)
            lat = self.mod_delete(lat, 5, 10)

        self.lon_prop = lon
        self.lat_prop = lat
        return lon, lat

    def mod_delete(self, a, n, m):
        """
        Deletes all m steps n values in a

        Parameters
        ----------
        a: 1darray
            array with coordinates
        n: int
            number of deleted points
        m: int
            range between two deletions

        Returns
        -------
        a: 1darray
            array with reduced coordinate points
        """
        return a[np.mod(np.arange(a.size), n + m) < n]


class antenna:
    """
    Antenna class that holds information about the geocentric coordinates of the
    radio telescopes. Can be converted to geodetic. All baselines between the
    the telescopes can be computed. Antenna positions can be shifted into a ENU frame
    of a specific observation, for which the (u, v)-coverage can be computed.
    """

    def __init__(self, X, Y, Z):
        """
        Parameters
        ----------
        X, Y, Z: array
            X, Y, Z coordinates of antennas
        """
        self.all = np.array(list(zip(X, Y, Z)))
        self.len = len(self.all)
        self.baselines = self.len * (self.len - 1)
        self.X = X
        self.Y = Y
        self.Z = Z

    def to_geodetic(self, x_ref, y_ref, z_ref, enu=False):
        """
        Converts geocentric coordinates to geodetic.

        Parameters
        ----------
        x_ref, y_ref, z_ref: float
            x, y, z reference positon
        enu: bool
            when True:
        """
        import astropy.units as u

        quant = ac.EarthLocation(x_ref, y_ref, z_ref, u.meter).to_geodetic()
        if enu is True:
            return quant.lon.deg, quant.lat.deg
        else:
            self.lon = quant.lon.deg
            self.lat = quant.lat.deg

    def get_baselines(self):
        """
        Calculates baselines between antenna pairs

        Returns
        -------
        x_base, y_base: 1darrays
            x, y values of the baselines
        """
        x_base = []
        y_base = []
        for i in range(self.len):
            ref = np.ones((self.len, 2)) * ([self.x_enu[i], self.y_enu[i]])
            pairs = np.array([self.x_enu, self.y_enu])
            baselines = np.array(list(zip(ref, pairs.T))).ravel()
            x = baselines[0::2]
            y = baselines[1::2]
            x_base = np.append(x_base, x)
            y_base = np.append(y_base, y)

        drops = np.asarray(
            [
                ((i * 2 + np.array([1, 2])) - 1) + (i * self.len * 2)
                for i in range(self.len)
            ]
        )
        coords = np.delete(np.stack([x_base, y_base], axis=1), drops.ravel(), axis=0).T
        x_base = coords[0]
        y_base = coords[1]
        return x_base, y_base

    def to_enu(self, x_ref, y_ref, z_ref):
        """
        Converts from geodetic to geocentric coordinates projected onto 2d plane

        Parameters
        ----------
        x_ref, y_ref, z_ref: 1darrays
            x, y, z reference coordinates
        """
        lon_ref, lat_ref = self.to_geodetic(x_ref, y_ref, z_ref, enu=True)
        if isinstance(x_ref, int):
            x_ref, y_ref, z_ref = [x_ref], [y_ref], [z_ref]
            lon_ref, lat_ref = [lon_ref], [lat_ref]
        ref = np.array(list(zip(x_ref, y_ref, z_ref)))

        def rot(lon, lat):
            """
            Calculates roytation matrix
            """
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)
            return np.array(
                [
                    [-np.sin(lon), np.cos(lon), 0],
                    [
                        -np.sin(lat) * np.cos(lon),
                        -np.sin(lat) * np.sin(lon),
                        np.cos(lat),
                    ],
                    [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
                ]
            )

        enu = np.array(
            [
                rot(lon_ref[j], lat_ref[j]) @ (self.all[i] - ref[j])
                for i in range(self.len)
                for j in range(len(lon_ref))
            ]
        )
        self.ant_enu = enu
        self.x_enu = enu.ravel()[0::3]
        self.y_enu = enu.ravel()[1::3]
        self.z_enu = enu.ravel()[2::3]
        return self.x_enu, self.y_enu

    def get_uv(self):
        """
        Calculates (u, v)-coordinates

        Returns
        -------
        u, v: 1d arrays
            u, v coordinates
        steps: int
            number of observation steps
        """
        u = []
        v = []
        steps = int(len(self.x_enu) / self.len)
        for j in range(steps):
            for i in range(self.len):
                x = self.x_enu[j::steps]
                y = self.y_enu[j::steps]
                x_ref = x[i] * np.ones(self.len)
                y_ref = y[i] * np.ones(self.len)
                x_base = x - x_ref
                y_base = y - y_ref
                x_base = x_base[x_base != 0] / 0.02
                y_base = -y_base[y_base != 0] / 0.02
                u = np.append(u, x_base)
                v = np.append(v, y_base)

        if len(u) != len(v):
            raise ValueError("Length of u and v are different!")
        return u, v, steps


def get_uv_coverage(source, antenna, iterate=False):
    """
    Converts source position and antenna positions into an (u, v)-coverage.

    Parameters
    ----------
    source: source class object
        source class containing source positions
    antenna: antenna clas object
        antenna class containing antenna positions
    iterate: bool
        use True while creating (u, v)-coverage gif

    Returns
    -------
    u: 1darray
        u coordinates
    v: 1darray
        v coordinates
    steps: 1darray
        number of observation steps
    """
    antenna.to_enu(*source.to_ecef(prop=True))
    u, v, steps = antenna.get_uv()

    if iterate is True:
        num_base = antenna.baselines
        u.resize((steps, num_base))
        v.resize((steps, num_base))

    return u, v, steps


def create_mask(u, v, size=63):
    """Create 2d mask from a given (uv)-coverage

    u: array of u coordinates
    v: array of v coordinates
    size: number of bins
    """
    uv_hist, _, _ = np.histogram2d(u, v, bins=size)
    # exclude center
    ex_l = size // 2 - 2
    ex_h = size // 2 + 3
    uv_hist[ex_l:ex_h, ex_l:ex_h] = 0
    mask = uv_hist > 0
    return np.rot90(mask)


def test_mask():
    """
    Test mask for filter tests
    """
    mask = np.ones((2, 63, 63))
    mask[:, 19, 30] = 0
    mask[:, 23, 23] = 0
    mask[:, 30, 19] = 0
    mask[:, 43, 32] = 0
    mask[:, 39, 39] = 0
    mask[:, 32, 43] = 0
    mask[:, 33:35, 33:35] = 0
    mask[:, 28:30, 28:30] = 0
    return mask


def sample_freqs(
    img,
    ant_config,
    size=63,
    lon=None,
    lat=None,
    num_steps=None,
    plot=False,
    test=False,
    specific_mask=True,
):
    """
    Sample specific frequencies in 2d Fourier space. Using antenna and source class to
    simulate a radio interferometric observation.

    Parameters
    ----------
    img: 2darray
        2d Fourier space
    ant_config: str
        name of antenna config
    size: int
        pixel size of input image, default 64x64 pixel
    lon: float
        start lon of source, if None: random start value between -90 and -70 is used
    lat: float
        start lat of source, if None a random start value between 30 and 80 is used
    num_steps: int
        number of observation steps
    plot: bool
        if True: returns sampled Fourier spectrum and sampling mask
    test_mask: bool
        if True: use same test mask for every image

    Returns
    -------
    img: 2darray
        sampled Fourier Spectrum
    """

    def get_mask(lon, lat, num_steps, ant, size):
        s = source(lon, lat)
        s.propagate(num_steps=num_steps, multi_pointing=False)
        u, v, _ = get_uv_coverage(s, ant, iterate=False)
        single_mask = create_mask(u, v, size)
        return single_mask

    bundle_size = img.shape[0]
    num_channel = img.shape[1]
    if test:
        mask = test_mask()
    else:
        layout = getattr(layouts, ant_config)
        ant = antenna(*layout())
        if specific_mask is True:
            s = source(lon, lat)
            s.propagate(num_steps=num_steps, multi_pointing=False)
            u, v, _ = get_uv_coverage(s, ant, iterate=False)
            single_mask = create_mask(u, v, size)
            mask = np.repeat(
                np.repeat(single_mask[None, None, :, :], num_channel, axis=1),
                bundle_size,
                axis=0,
            )
        else:
            mask = np.array([None, None, None])
            if lon is None:
                lon = np.random.randint(-90, -70, size=(bundle_size,))
            if lat is None:
                lat = np.random.randint(30, 80, size=(bundle_size,))
            mask_woc = np.asarray(
                [
                    get_mask(lon[i], lat[i], num_steps, ant, size)
                    for i in range(bundle_size)
                ]
            )
            mask = np.repeat(mask_woc[:, None, :, :], num_channel, axis=1)
    img = img.copy()
    img[~mask.astype(bool)] = 0
    if plot is True:
        return img, mask
    else:
        return img
