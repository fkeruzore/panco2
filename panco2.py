import numpy as np
import astropy.units as u
from _dataset import Dataset
from _model_nonparam import ModelNonParam


class PressureProfile:

    """
    The main class of panco2.

    Parameters
    ----------
    cluster : _cluster.Cluster instance
        a Cluster class instance containing information on the source,
        such as its redshift, mass, etc.
    path_to_results : str
        path to where the results are to be saved.
    """

    def __init__(self, cluster, path_to_results):
        self.path_to_results = path_to_results
        self.datasets = []
        self.cluster = cluster
        self.do_ps = False
        self.n_ps = 0

    # ====================================================================== #

    def params2dict(self, params):
        """
        Creates a comprehensive dictionnary from a vector in the parameter
        space.

        Parameters
        ----------
        params : list or array
            A vector in the parameter space.

        Returns
        -------
        par : dict
            A key-value dictionnary containing named parameter values.

        See also
        --------
        dict2params : reverse operation
        """
        return {key: params[self.indices[key]] for key in self.indices.keys()}

    # ====================================================================== #

    def dict2params(self, params):
        """
        Creates a vector in the parameter space from a dict with parameter
        values.

        Parameters
        ----------
        par : dict
            A key-value dictionnary containing named parameter values.

        Returns
        -------
        params : array
            A vector in the parameter space.

        See also
        --------
        params2dict : reverse operation
        """
        return None  # TODO

    # ====================================================================== #

    def add_dataset(self, ds):
        """
        Add a dataset to be fitted.

        Parameters
        ----------
        ds : _dataset.Dataset instance
        """

        self.datasets.append(ds)
        print(f"==> Dataset `{ds.name}` loaded:")
        print(f"    beam FWHM = {ds.reso_fwhm:.2f},")
        print(f"    map size = {ds.map_size.to('arcmin'):.2f},")
        print(f"    {'No' if (ds.inv_covmat is None) else ''} noise covariance matrix")

    # ================================================================================ #

    def add_integrated_compton(self, Y_integ, err_Y_integ, r_integ):
        """
        Add a constraint on integrated Compton parameter Y.

        Parameters
        ----------
        Y_integ : Quantity
            value of integrated y.
            Must be in units of area (e.g. Mpc2);
        err_Y_integ : Quantity
            1sigma uncertainty on Y_integ.
            Must be in units of area (e.g. Mpc2);
        r_integ : Quantity
            radius inside which Y_integ was computed.
            Must be in units of length (e.g. Mpc).

        Notes
        -----
        Each input must be in units of physical length (area),
        not angular size. To convert accordinlgy, see
        ``_cluster.Cluster.angle_distances_conversion``.
        """
        self.Yinteg = Y_integ.to("kpc2")
        self.err_Yinteg = err_Y_integ.to("kpc2")
        self.rinteg = r_integ.to("kpc")
        print("==> Integrated SZ signal added:")
        print(
            f"    Da2*Y = ({Y_integ.value:.2f} +- {err_Y_integ.value:.2f}) kpc2",
            f"within {r_integ.value:.2f} kpc",
        )

    # ================================================================================ #

    def init_radial_bins(self, n_bins=8, radius_tab=None):
        """
        Initialize the radial binning to be used.
        It can either be forced through `radius_tab`, or computed as
            `n_bins` log-spaced points between the smallest beam HWHM
            and largest half map size in your datasets.

        Parameters
        ----------
        n_bins : int
            the number of radial bins;
        radius_tab : Quantity or None
            radial binning to be forced,
            in units of physical distances (e.g. kpc).

        Notes
        -----
        If `radius_tab` is `None`, the binning is initialized
        depending on the resolutions and map sizes to be
        fitted; therefore, this function must be called
        after initializing the datasets.
        """
        if radius_tab is not None:
            self.radius_tab = radius_tab
        else:
            min_rad_angle = 0.5 * np.min(
                [d.reso_fwhm.to("arcsec").value for d in self.datasets]
            )
            max_rad_angle = 0.5 * np.max(
                [d.map_size.to("arcsec").value for d in self.datasets]
            )
            radius_tab_angle = np.logspace(
                np.log10(min_rad_angle), np.log10(max_rad_angle), n_bins
            )
            self.radius_tab = self.cluster.angle_distances_conversion(
                radius_tab_angle * u.arcsec
            )
        self.n_bins = self.radius_tab.size
        for ds in self.datasets:
            ds.radius_tab = self.radius_tab.value
            ds.n_bins = self.n_bins

    # ================================================================================ #

    def init_point_sources(
        self, path="", ps_prior_type="pdf", do_subtract=True,
    ):
        """
        Initialize everything to treat point sources.
        A catalog is created with the pixel positions and fluxes of the sources
        to fit.
        The ones to subtract are subtracted from the input map.

        Parameters
        ----------
        path: str
            path to your point sources results;
        ps_prior_type: str
            For fitted source, wether to use the real flux PDF as
            prior ("pdf") or rather a gaussian distribution wth its
            mean and standard deviation ("gaussian");
        do_subtract: bool
            wether you want to remove the sources flagged as `subtract`
            in your catalog
        """
        self.do_ps = True
        for ds in self.datasets:
            self.n_ps = ds.init_point_sources(
                path=path, ps_prior_type=ps_prior_type, do_subtract=do_subtract
            )

    # ================================================================================ #

    def init_param_indices(self, verbose=True):
        """
        Generate the names of model parameters and their positions in the
        parameter space.
        """

        self.param_names = [f"P_{i}" for i in range(self.n_bins)]
        self.indices_press = slice(0, self.n_bins)
        self.param_names_fancy = ["$P_{" + str(i) + "}$" for i in range(self.n_bins)]

        for i, ds in enumerate(self.datasets):
            self.param_names.append(f"conv_{i}")
            self.param_names.append(f"zero_{i}")
            self.param_names_fancy.append(f"Conversion {ds.name}")
            self.param_names_fancy.append(f"Zero level {ds.name}")
            ds.i = i
            # Last line: so that each dataset knows where
            # its conv / zero are in the parameter space

        if self.do_ps:
            self.indices_ps = slice(len(self.param_names), None)
            for i in range(self.init_ps["nps"]):
                self.param_names.append(f"F_{i}")
                self.param_names_fancy.append("$F_{$" + str(i) + "}$")

        self.indices = {param: i for i, param in enumerate(self.param_names)}
        self.nparams = len(self.param_names)
        if verbose:
            print(f"==> {len(self.param_names)} parameters:", self.param_names)

    # ================================================================================ #

    def log_likelihood(self, par_vec):
        """
        Computes the log-likelihood associated to a parameter vector.

        Parameters
        ----------
        par_vec : list or array
            The vector of model parameters
        """
        press = par_vec[self.indices_press]
        par_dict = self.params2dict(par_vec)
        sz_maps = [
            ds.compute_compton_map(press, par_dict[f"conv_{i}"], par_dict[f"zero_{i}"])
            for i, ds in enumerate(self.datasets)
        ]

    # ================================================================================ #

    def log_posterior(self, par_vec):
        """
        Computes the log-posterior associated to a parameter vector.

        Parameters
        ----------
        par_vec : list or array
            The vector of model parameters
        """

        return 0.0
