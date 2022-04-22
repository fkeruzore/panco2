import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import astropy.units as u
import astropy.constants as const
from scipy.stats import norm, gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
import pdb
import _pointsources, _utils, _data
from _utils import sz_fact
from shell_pl import shell_pl


class Dataset:
    """
    A class to perform per-dataset operations.

    Parameters
    ----------
    in_fits_file : str
        path to a fits file;
    hdu_data : int
        extension containing the map to be fitted;
    hdu_rms : int
        extension containing the noise RMS map;
    reso_fwhm : Quantity
        the FWHM of the beam of the instrument, in angle units;
    crop : Quantity or None
        the size of the map to be fitted, in angle units;
    coords_center : SkyCoord or None
        the center of the map to be fitted in equatorial coordinates;
    inv_covmat : array or None
        inverse of the noise covariance matrix, in the same units as
        the input map to the power of -2;
    file_noise_simus : str or None
        path to a fits file where each extension is a correlated
        noise realization normalized to the noise RMS, to be used
        to compute the covariace matrix if it is not provided;
    tf_k : Quantity
        Angular modes covered by the transfer function, in units of
        1/angle (e.g. arcmin-1). Must be in Nyquist convention
        (1/map size = Nyquist frequency);
    tf_filtering : array
        Values of filtering for each mode in `tf_k`;
    conversion : tuple, float or None
        Conversion coefficient from map units to Compton y.
        If `None`, is fixed to 1 (map already in y).
        If `float`, fixed to the given value.
        If `tuple`, used as nuisance parameter with Gaussian prior
        where the tuples gives the mean and standard deviation.
    d_a : Quantity
        Angular diameter distance to cluster in distance units
    no_point_sources : bool
        Set to True to not treat point sources in the model.
    """

    # ---------------------------------------------------------------------- #
    # ====  INITIALIZATIONS  =============================================== #
    # ---------------------------------------------------------------------- #

    def __init__(
        self,
        in_fits_file,
        hdu_data,
        hdu_rms,
        reso_fwhm,
        crop=None,
        coords_center=None,
        inv_covmat=None,
        file_noise_simus=None,
        tf_k=None,
        tf_filtering=None,
        conversion=None,
        name=None,
        d_a=1.0 * u.Mpc,
        no_point_sources=True,
    ):
        # ===== Read input data
        (
            self.map,
            self.rms,
            _,
            self.inv_covmat,
            self.wcs,
            self.pix_size,
        ) = _data.read_data(
            in_fits_file,
            hdu_data,
            hdu_rms,
            crop=crop,
            coords_center=coords_center,
            inv_covmat=inv_covmat,
            file_noise_simus=file_noise_simus,
        )
        self.npix = self.map.shape[0]
        self.map_size = self.npix * self.pix_size
        self.reso_fwhm = reso_fwhm
        self.reso_sigma_pix = _utils.adim(reso_fwhm / self.pix_size)

        # ===== Radii profiles and maps

        # 1D radius in the sky plane, only half the map
        theta_x = np.arange(0, int(self.npix / 2) + 1) * self.pix_size.to("rad").value
        r_x = d_a.to("kpc").value * np.tan(theta_x)  # distance, in kpc

        # 2D (x, y) radius in the sky plane to compute compton map
        center = (0, 0)  # For future developpers to change
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a.to("kpc").value
                * np.tan(center[0] * self.pix_size.to("rad").value),
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a.to("kpc").value
                * np.tan(center[1] * self.pix_size.to("rad").value),
            )
        )
        self.init_prof = {
            "r_x": r_x,
            "r_xy": r_xy,
            "theta_x": theta_x * u.rad.to("arcmin"),
        }

        # ===== Transfer function
        if (tf_k is not None) and (tf_filtering is not None):
            tf_k = tf_k.to("arcsec-1")

            tf_side = 1.0 / tf_k.min() / np.sqrt(2)
            pad = int(0.5 * ((tf_side / self.pix_size.to("arcsec")).value - self.npix))
            k_vec = np.fft.fftfreq(
                self.npix + 2 * pad, self.pix_size.to("arcsec").value
            )
            karr = np.hypot(*np.meshgrid(k_vec, k_vec))

            interp = interp1d(
                tf_k, tf_filtering, bounds_error=False, fill_value=tf_filtering[-1]
            )
            tf_arr = interp(karr)

            self.init_tf = {"tf_arr": tf_arr, "pad": pad}

            def convolve_tf(in_map):
                in_map_pad = np.pad(in_map, pad, mode="constant", constant_values=0.0)
                in_map_fourier = np.fft.fft2(in_map_pad)
                conv_in_map = np.real(np.fft.ifft2(in_map_fourier * tf_arr))
                return conv_in_map[pad:-pad, pad:-pad]

        else:

            def convolve_tf(in_map):
                return in_map

        self._convolve_tf = convolve_tf

        # ===== Misc
        self.name = name if (name is not None) else "Dataset"
        self.radius_tab = None  # will be overwritten later
        self.n_bins = None  # will be overwritten later

        # ===== Conversion from y to map units
        if conversion is None:
            self.conversion = (1.0, 0.0)
        elif isinstance(conversion, float) or isinstance(conversion, int):
            self.conversion = (float(conversion), 0.0)
        elif isinstance(conversion, tuple):
            self.conversion = conversion
        else:
            raise TypeError(f"Invalid type for `conversion`: {conversion.__class__}")

        # ===== Bypass pointsource computations if needed
        if no_point_sources:
            def _compute_model_ps(ps_fluxes):
                return 0.0
            self._compute_model_ps = _compute_model_ps

    # ---------------------------------------------------------------------- #

    def init_point_sources(
        self,
        path="",
        ps_prior_type="pdf",
        do_subtract=True,
    ):
        """
        Initialize everything to treat point sources.
        A catalog is created with the pixel positions and fluxes of the sources
        to fit.
        The ones to subtract are subtracted from the input map.

        Parameters
        ----------
        path : str
            path to your point sources results;
        ps_prior_type : str
            For fitted source, wether to use the real flux PDF as
            prior ("pdf") or rather a gaussian distribution wth its
            mean and standard deviation ("gaussian");
        do_subtract : bool
            wether you want to remove the sources flagged as `subtract`
            in your catalog

        Returns
        -------
        nps : int
            The number of point sources
        """

        full_ps_cat = Table.read(path + "Catalog.fits")

        # ===== Initialize the catalog ===== #
        ps_pos = []
        ps_fluxes = []
        ps_fluxes_err = []
        ps_pdfs = []
        ps_logpdfs = []
        ps_tosubtract_pos = []
        ps_tosubtract_fluxes = []
        interp_range = np.linspace(0.0, 1e-2, 100)

        nps = 0
        for i, ps in enumerate(full_ps_cat):
            coords = SkyCoord(ps["COORDS"])
            if ps["SUBTRACT"] == 0:
                nps += 1
                ps_pos.append(skycoord_to_pixel(coords, self.wcs))
                posterior_fluxes = np.load(path + ps["NAME"] + "_fluxes_dist.npy")
                flux_avg = np.average(posterior_fluxes)
                flux_std = np.std(posterior_fluxes)
                ps_fluxes.append(flux_avg)
                ps_fluxes_err.append(flux_std)
                if ps_prior_type == "pdf":
                    kde = gaussian_kde(posterior_fluxes)
                    ps_logpdfs.append(kde.logpdf(interp_range))
                    ps_pdfs.append(kde.pdf(interp_range))
                elif ps_prior_type == "gaussian":
                    gauss = norm(flux_avg, flux_std)
                    ps_logpdfs.append(gauss.logpdf(interp_range))
                    ps_pdfs.append(gauss.pdf(interp_range))

                print(f"    {i+1}) {coords.to_string('hmsdms')}: to fit")

            else:
                ps_tosubtract_pos.append(skycoord_to_pixel(coords, self.wcs))
                ps_tosubtract_fluxes.append(ps["FLUX"])
                print(f"    {i+1}) {coords.to_string('hmsdms')}: to subtract")

        # ===== Subtract the point sources flagged for subtraction ===== #
        if do_subtract and ps_tosubtract_fluxes != []:
            data_cleaned = _pointsources.subtract_ps(
                self.map,
                np.array(ps_tosubtract_fluxes),
                np.array(ps_tosubtract_pos),
                reso_sigma_pix,
            )
            self.map = data_cleaned

        # ===== Initialize the interpolation of PDFs for the fluxes ===== #
        interp_prior_ps = [
            interp1d(
                interp_range,
                ps_logpdfs[i],
                bounds_error=False,
                fill_value=-np.inf,
                assume_sorted=True,
            )
            for i in range(nps)
        ]
        # interp2d_prior_ps = interp2d(interp_range, np.arange(nps), np.array(ps_logpdfs))
        # NB: 2d interpolation is slightly faster, but error prone and a bit overkill

        # ===== Positions ===== #
        npix = self.wcs.array_shape[0]
        xmap, ymap = np.meshgrid(np.arange(npix), np.arange(npix))
        xmap_ps = np.stack([xmap for i in range(nps)])
        ymap_ps = np.stack([ymap for i in range(nps)])

        self.init_ps = {
            "nps": nps,
            "interp_prior_ps": interp_prior_ps,
            "ps_pos": np.array(ps_pos),
            "xmap_ps": xmap_ps,
            "ymap_ps": ymap_ps,
        }
        self.do_ps = True

        def _compute_model_ps(ps_fluxes):
            ps_map = _pointsources.ps_map(
                ps_fluxes,
                self.init_ps["ps_pos"],
                self.init_ps["xmap_ps"],
                self.init_ps["ymap_ps"],
                self.pix_size,
            )

            return ps_map

        self._compute_model_ps = _compute_model_ps

        return nps

    # ---------------------------------------------------------------------- #
    # ====  MODEL COMPUTATION RELATED FUNCTIONS  =========================== #
    # ---------------------------------------------------------------------- #

    def convolve_tf(self, in_map):
        """
        Convolves a map with the transfer function, as initialized
        in ``self.init_transfer_fnction``.

        Parameters
        ----------
        in_map : array
            The map to be convolved

        Returns
        -------
        array
            The convolved map
        """

        return self._convolve_tf(in_map)

    # ---------------------------------------------------------------------- #

    def compute_model_ps(self, par):
        """
        Computes the point source contamination model map.

        Parameters
        ----------
        par : dict
            the parameters of the model,

        Returns
        -------
        ps_map : array
            the point source model map
        """
        return self._compute_model_ps(par)

    # ---------------------------------------------------------------------- #

    def compute_compton_map(self, pressure_tab):
        """
        TODO: write docstring

        Parameters
        ----------
        pressure_tab : array
            Values of pressure at each radus in `self.radius_tab`

        Returns
        -------
        y_map_filt : array
            Model map in compton parameter units, filtered by the
                beam and transfer function
        """
        # ===== Compute slopes
        r_bins = self.radius_tab
        lr_bins = np.log(r_bins)
        lp_bins = np.log(pressure_tab)

        alphas = -np.ediff1d(lp_bins) / np.ediff1d(lr_bins)
        alphas = np.concatenate(
            ([alphas[0]], alphas, [alphas[-1]])
        )  # Fill the first and last values for extrapolation

        # ===== Compton profile
        r_x = self.init_prof["r_x"][1:]
        integrals = np.zeros((self.n_bins, r_x.size))
        r_bins_integ = np.concatenate(([0.0], r_bins, [-1.0]))
        for i in range(self.n_bins):
            alpha_i = alphas[i]
            integrals[i] = shell_pl(
                pressure_tab[i], alpha_i, r_bins_integ[i], r_bins_integ[i + 1], r_x
            )
        y_prof = np.sum(integrals * sz_fact, axis=0)

        # ===== Compton parameter map
        y_map = _utils.prof2map(
            y_prof, self.init_prof["r_x"][1:], self.init_prof["r_xy"]
        )
        y_map_filt = gaussian_filter(y_map, self.reso_sigma_pix)

        return y_map_filt

    # ---------------------------------------------------------------------- #

    def compute_SZ_map(self, pressure_tab, conv):
        sz_model = conv * self.compute_compton_map(pressure_tab)
        return sz_model

    # ---------------------------------------------------------------------- #

    def compute_model(self, pressure_tab, conv, zero, ps_fluxes):
        return (
            self.compute_SZ_map(pressure_tab, conv)
            + zero
            + self.compute_model_ps(ps_fluxes)
        )
