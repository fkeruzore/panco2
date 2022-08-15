import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d


class Filter:
    def __init__(self, beam_sigma_pix=0.0, tf=None, pad_pix=0):

        if beam_sigma_pix != 0.0:
            self.has_beam = True
            self.beam_sigma_pix = beam_sigma_pix
        else:
            self.has_beam = False

        if tf is not None:
            self.has_tf = True
            self.transfer_function = tf
            self.pad_pix = int(pad_pix)
        else:
            self.has_tf = False

    def __call__(self, in_map):
        if self.has_tf:
            pad = self.pad_pix
            in_map_pad = np.pad(
                in_map, pad, mode="constant", constant_values=0.0
            )
            in_map_fourier = np.fft.fft2(in_map_pad)
            in_map = np.real(
                np.fft.ifft2(in_map_fourier * self.transfer_function)
            )
            in_map = in_map[pad:-pad, pad:-pad]
        if self.has_beam:
            in_map = gaussian_filter(in_map, self.beam_sigma_pix)
        return in_map


class Filter1d(Filter):
    def __init__(
        self, npix, pix_size, k_1d, tf_1d, beam_sigma_pix=0.0, pad_pix=0
    ):

        pad = int(pad_pix)

        # Compute the modes covered in the map
        k_map_1d = np.fft.fftfreq(npix + 2 * pad, pix_size)
        k_map_2d = np.hypot(*np.meshgrid(k_map_1d, k_map_1d))

        # Extrapolation with extreme values
        i_sort_k_in = np.argsort(k_1d)
        sorted_tf_1d = tf_1d[i_sort_k_in]
        tf_smallk, tf_highk = sorted_tf_1d[0], sorted_tf_1d[-1]

        interp = interp1d(
            k_1d, tf_1d, bounds_error=False, fill_value=(tf_smallk, tf_highk)
        )
        tf_map_2d = interp(k_map_2d)

        super().__init__(
            beam_sigma_pix=beam_sigma_pix, tf=tf_map_2d, pad_pix=pad
        )


class Filter2d(Filter):
    def __init__(
        self, npix, pix_size, kx, ky, tf, beam_sigma_pix=0.0, pad_pix=0
    ):

        pad = int(pad_pix)

        # Compute the modes covered in the map
        kx_map = np.fft.fftfreq(npix + 2 * pad, pix_size)
        ky_map = np.copy(kx_map)

        interp = interp2d(kx, ky, tf, bounds_error=False, fill_value=0.0)
        tf_map_2d = interp(kx_map, ky_map)

        super().__init__(
            beam_sigma_pix=beam_sigma_pix, tf=tf_map_2d, pad_pix=pad
        )
