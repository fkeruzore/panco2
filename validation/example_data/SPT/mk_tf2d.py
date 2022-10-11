import numpy as np

n_pix = int(5.0 * 3600.0 / 15.0)
if n_pix % 2 == 0:
    n_pix += 1

kx, ky = [np.fft.fftshift(np.fft.fftfreq(n_pix, 15.0)) for _ in range(2)]
kxx, kyy = np.meshgrid(kx, ky, indexing="ij")

ellxx, ellyy = [k * 180.0 * 3600.0 for k in (kxx, kyy)]
ell = np.hypot(ellxx, ellyy)
filt = np.logical_and(
    (np.abs(ell) > 2000) & (np.abs(ell) < 20000),
    np.abs(ellxx) > 1000,
)

np.savez_compressed("./tf2d.npz", ell_x=ellxx, ell_y=ellyy, tf=filt)
