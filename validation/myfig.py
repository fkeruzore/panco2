import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{txfonts}",
        "font.family": "STIXGeneral",
    }
)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], "o-")
ax.set_xlabel("Aaaaaaa")
ax.set_ylabel("$F_1(r, y, z)$")
fig.savefig("myfig.pdf")
