import numpy as np
import matplotlib.pyplot as plt

n_pix = int(5.0 * 3600.0 / 15.0)
if n_pix % 2 == 0:
    n_pix += 1

kx, ky = [np.fft.fftshift(np.fft.fftfreq(n_pix, 15.0)) for _ in range(2)]
kxx, kyy = np.meshgrid(kx, ky, indexing="xy")

ellxx, ellyy = [k * 180.0 * 3600.0 for k in (kxx, kyy)]
ell = np.hypot(ellxx, ellyy)
filt = np.logical_and(np.abs(ell) > 500, np.abs(ellxx) > 300).astype(float)
filt *= 1.0 - (0.1 * np.abs(ellxx)) / np.max(ellxx)

np.savez_compressed("tf2d.npz", ell_x=ellxx, ell_y=ellyy, tf=filt)

fig, ax = plt.subplots()
im = ax.imshow(
    filt,
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
    extent=(ellxx.min(), ellxx.max(), ellyy.min(), ellyy.max()),
    interpolation="none",
)
cb = fig.colorbar(im, ax=ax)
cb.set_label("SPT transfer function")
ax.set_xlabel("$\ell_x$")
ax.set_ylabel("$\ell_y$")
fig.savefig("tf2d.pdf")
