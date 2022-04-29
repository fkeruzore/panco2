import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


class Filter:
    def __init__(
        self, npix, pix_size, beam_sigma_pix=0.0, tf_k=None, k=None, pad=60.0
    ):

        if beam_sigma_pix != 0.0:
            self.has_beam = True
            self.beam_sigma_pix = beam_sigma_pix
        else:
            self.has_beam = False

        if (k is not None) and (tf_k is not None):
            if pad is None:
                pad = 0.0
            pad = int(pad / pix_size)
            self.has_tf = True
            # Compute the modes covered in the map
            k_vec = np.fft.fftfreq(npix + 2 * pad, pix_size)
            karr = np.hypot(*np.meshgrid(k_vec, k_vec))

            interp = interp1d(k, tf_k, bounds_error=False, fill_value=tf_k[-1])
            tf_arr = interp(karr)

            self.transfer_function = {"k": karr, "tf_k": tf_arr}
            self.pad_pix = pad
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
                np.fft.ifft2(in_map_fourier * self.transfer_function["tf_k"])
            )
            in_map = in_map[pad:-pad, pad:-pad]
        if self.has_beam:
            in_map = gaussian_filter(in_map, self.beam_sigma_pix)
        return in_map
