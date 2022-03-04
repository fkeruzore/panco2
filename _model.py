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
import _pointsources, _utils


class Model:
    """
    Class encapsulating the tools for model computation.

    Args:
        cluster (_cluster.Cluster): a Cluster object.

    Notes:
        This is a parent class from which derive
        ``_model_gnfw.ModelGNFW``
        and
        ``_model_nonparam.ModelNonParam``
    """

    # ---------------------------------------------------------------------- #
    # ====  INITIALIZATIONS  =============================================== #
    # ---------------------------------------------------------------------- #

    def __init__(self, cluster, fit_zl=True, do_ps=False):
        self.cluster = cluster
        self.sz_fact = (
            (const.sigma_T / (const.m_e * const.c ** 2))
            .to(u.cm ** 3 / u.keV / u.kpc)
            .value
        )
        self.fit_zl = fit_zl
        self.do_ps = do_ps

        if (self.fit_zl) and (self.do_ps):

            def compute_model(par):
                SZ_map, Y_500 = self.compute_model_SZ(par)
                ps_map = self.compute_model_ps(par)
                model_map = self.convolve_tf(SZ_map + ps_map)
                return model_map + par["zero"], Y_500

        elif (not self.fit_zl) and (self.do_ps):

            def compute_model(par):
                SZ_map, Y_500 = self.compute_model_SZ(par)
                ps_map = self.compute_model_ps(par)
                model_map = self.convolve_tf(SZ_map + ps_map)
                return model_map, Y_500

        elif (self.fit_zl) and (not self.do_ps):

            def compute_model(par):
                SZ_map, Y_500 = self.compute_model_SZ(par)
                model_map = self.convolve_tf(SZ_map)
                return model_map + par["zero"], Y_500

        elif (not self.fit_zl) and (not self.do_ps):

            def compute_model(par):
                SZ_map, Y_500 = self.compute_model_SZ(par)
                model_map = self.convolve_tf(SZ_map)
                return model_map, Y_500

        self.__compute_model = compute_model

    # ---------------------------------------------------------------------- #

    def init_point_sources(
        self,
        path="",
        data=None,
        wcs=None,
        reso=3,
        beam=17.6,
        ps_prior_type="pdf",
        fixed_error=None,
        do_subtract=True,
        which_ps=None,
    ):
        """
        Initialize everything to treat point sources.
        A catalog is created with the pixel positions and fluxes of the sources
        to fit.
        The ones to subtract are subtracted from the input map.

        Args:
            path (str) : path to your point sources results,

            data (array): NIKA2 150 GHz map,

            wcs (astropy.coordinates.WCS): WCS associated to the NIKA2 map,

            reso (float) : pixel size in arcsecs,

            beam (float) : instrumental beam FWHM in arcsec,

            ps_prior_type (str) : "pdf" or "gaussian",

            fixed_error (float, None):
                If ps_prior_type = "gaussian", you can
                force an error bar on your prior,

            do_subtract (bool) :
                wether you want to remove the sources
                flagged as `subtract` in your catalog

            which_ps (list) :
                which point sources to take into account.
                Example: [0, 1] takes the first two sources of the catalog.
                If None, takes them all (super misleading)
        """

        full_ps_cat = Table.read(path + "Catalog.fits")
        if which_ps is not None:
            full_ps_cat = full_ps_cat[which_ps]
        # Beam sigma in pixels
        beam_pix = 0.4247 * _utils.adim(beam / reso)

        # ===== Initialize the catalog ===== #
        ps_pos = []
        ps_fluxes = []
        ps_fluxes_err = []
        ps_pdfs = []
        ps_logpdfs = []
        ps_tosubtract_pos = []
        ps_tosubtract_fluxes = []
        interp_range = np.linspace(
            0.0, 2e-3, 100
        )  # up to 2 mJy. TODO: is that too low?

        nps = 0
        for i, ps in enumerate(full_ps_cat):
            coords = SkyCoord(ps["COORDS"])
            if ps["SUBTRACT"] == 0:
                nps += 1
                ps_pos.append(skycoord_to_pixel(coords, wcs))
                posterior_fluxes = np.load(path + ps["NAME"] + "_fluxes_dist.npy")
                flux_avg = np.average(posterior_fluxes)
                if fixed_error is None:
                    flux_std = np.std(posterior_fluxes)
                else:
                    flux_std = fixed_error
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

                print(
                    "    "
                    + str(i + 1)
                    + ") "
                    + coords.to_string("hmsdms")
                    + "\t"
                    + str(round(np.average(1e3 * posterior_fluxes), 3))
                    + "\t"
                    + " mJy/beam : to fit"
                )

            else:
                ps_tosubtract_pos.append(skycoord_to_pixel(coords, wcs))
                ps_tosubtract_fluxes.append(ps["FLUX"])
                print(
                    "    "
                    + str(i + 1)
                    + ") "
                    + coords.to_string("hmsdms")
                    + "\t"
                    + str(round(1e3 * ps["FLUX"], 3))
                    + " mJy/beam : to subtract"
                )

        # ===== Subtract the point sources flagged for subtraction ===== #
        if do_subtract and ps_tosubtract_fluxes != []:
            data_cleaned = _pointsources.subtract_ps(
                data,
                np.array(ps_tosubtract_fluxes),
                np.array(ps_tosubtract_pos),
                beam_pix,
            )
        else:
            data_cleaned = data

        ps_cat = full_ps_cat[full_ps_cat["SUBTRACT"] == 0]

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
        """
        Note: I stopped using the 2d interpolation because it was slightly faster,
              but SO CONFUSING... Seemed overkill just to avoid a tiny for loop.
        """

        # ===== Positions ===== #
        npix = wcs.array_shape[0]
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
        return data_cleaned, np.array(ps_fluxes)

    # ---------------------------------------------------------------------- #

    def init_transfer_function(self, tf_file, npix, pad, reso):
        """
        Initialize the transfer function convolutions.

        Args:
            tf_file (str) :
                path to the ``.fits`` file with the transfer function.
                This recognizes files from ``SZ_RDA`` and ``SZ_IMCM``.
            npix (int) : size of the NIKA2 map.
        """

        # ===== Load the TF file ===== #
        if tf_file is not None:
            with fits.open(tf_file) as file_test:
                do_old_tf = len(file_test) == 3  # SZ_RDA convention for k
                do_new_tf = len(file_test) == 2  # Nyquist convention for k

            if do_old_tf:
                print("    TF convention: SZ_RDA")
                t = Table.read(tf_file, 2)
                tf_k = t["WAVE_NUMBER_ARCSEC"].data.reshape(-1)
                tf_tf = t["TF"].data.reshape(-1)

                k_vec = np.fft.fftfreq(npix + 2 * pad, reso)
                karr = np.hypot(*np.meshgrid(k_vec, k_vec))
                karr /= karr.max() * reso

                interp = interp1d(tf_k, tf_tf, bounds_error=False, fill_value=tf_tf[-1])
                tf_arr = interp(karr)

            elif do_new_tf:
                print("    TF convention: NYQUIST")
                t = Table.read(tf_file, 1)
                tf_k = t["k"].to("arcsec-1").value
                tf_tf = t["tf_2mm"]

                k_vec = np.fft.fftfreq(npix + 2 * pad, reso)
                karr = np.hypot(*np.meshgrid(k_vec, k_vec))

                interp = interp1d(tf_k, tf_tf, bounds_error=False, fill_value=tf_tf[-1])
                tf_arr = interp(karr)

            self.init_tf = {"tf_arr": tf_arr, "pad": pad}

        # ===== Init convolution function ===== #
        if tf_file is not None:

            def convolve_tf(in_map):
                in_map_pad = np.pad(in_map, pad, mode="constant", constant_values=0.0)
                in_map_fourier = np.fft.fft2(in_map_pad)
                conv_in_map = np.real(np.fft.ifft2(in_map_fourier * tf_arr))
                return conv_in_map[pad:-pad, pad:-pad]

        else:

            def convolve_tf(in_map):
                return in_map

        self.__convolve_tf = convolve_tf

    # ---------------------------------------------------------------------- #

    def params_to_dict(self, params):
        """
        Given an input parameter vector (e.g. from your MCMC), returns a
        dict with all the parameters called by names.
        """
        return {key: params[self.indices[key]] for key in self.indices.keys()}

    # ---------------------------------------------------------------------- #

    def angle_distances_conversion(self, to_convert):
        """
        Converts an angular size to physical distance, or reciprocally.

        Args:
            to_convert (astropy.units.Quantity): an angle or distance
                value.

        Returns:
            (astropy.units.Quantity) the input converted to distance 
                or angle.
        """
        input_unit = to_convert.unit
        if input_unit.is_equivalent(u.rad):  # input is angle
            return np.tan(to_convert.to("rad").value) * self.cluster.d_a
        elif input_unit.is_equivalent(u.kpc):  # input is angle
            return np.arctan((to_convert.to("kpc") / self.cluster.d_a).value) * u.rad

    # ---------------------------------------------------------------------- #
    # ====  MODEL COMPUTATION RELATED FUNCTIONS  =========================== #
    # ---------------------------------------------------------------------- #

    def __call__(self, par):
        """
        Computes the total model map and integrated SZ signal.

        Args:
            par (dict): the parameters of the model,

        Returns:
            (tuple): tuple containing:

                * model_map (array) : the resulting map
                * Y_500 (float) : the integrated SZ signal
        """

        return self.__compute_model(par)

    # ---------------------------------------------------------------------- #

    def compute_model_ps(self, par):
        """
        Computes the point source contamination model map.

        Args:
            par (dict) : the parameters of the model,

        Returns:
            (array) : the point source model map
        """
        ps_map = _pointsources.ps_map(
            par["ps_fluxes"],
            self.init_ps["ps_pos"],
            self.init_ps["xmap_ps"],
            self.init_ps["ymap_ps"],
            self.init_prof["reso"],
        )

        return ps_map

    # ---------------------------------------------------------------------- #

    def convolve_tf(self, in_map):
        """
        Convolves a map with the transfer function, as initialized
        in ``self.init_transfer_fnction``.

        Args:
            in_map (array): The map to be convolved

        Returns:
            (array): The convolved map
        """

        return self.__convolve_tf(in_map)
