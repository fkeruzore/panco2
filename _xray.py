import numpy as np
from astropy.table import Table
import json


def recover_x_profiles(x_file):
    """
    Open the X-ray parprod and returns the profiles.

    Args:
        x_file(str): the parprod file.
            Can be in .fits or in .json; if you only have a .save, use
            `/data/Workspace/SZ_program_data/XMM_files/parprod2json.pro`

    Returns:
        (dict): 
            a dict with the thermodynamical profiles of the parprod. Keys are:

                * ``rd``, ``d``, ``errd``: radius [kpc], density and error [cm-3],
                * ``rt``, ``t``, ``errt``: radius [kpc], temperature and error [keV],
                * ``rp``, ``p``, ``errp``: radius [kpc], pressure and error [keV cm-3],
                * ``rk``, ``k``, ``errk``: radius [kpc], entropy and error [keV cm2],
                * ``rm``, ``m``, ``errm``: radius [kpc], HS mass and error [Msun]
    """

    if x_file.endswith(".fits"):
        table = Table.read(x_file)
        x_profs = {q: table[q].data[0] for q in table.colnames}
    elif x_file.endswith(".json"):
        with open(x_file, "r") as f:
            full_x_profs = json.load(f)  # there is a lot of data here
        x_profs = {
            "rd": [],
            "d": [],
            "errd": [],
            "rt": [],
            "t": [],
            "errt": [],
            "rp": [],
            "p": [],
            "errp": [],
            "rk": [],
            "k": [],
            "errk": [],
            "rm": [],
            "m": [],
            "errm": [],
        }
        for point in full_x_profs["DENSPROF"]:
            if point["R"] <= 0.0:
                break
            x_profs["rd"].append(point["R"])
            x_profs["d"].append(point["DENS"])
            x_profs["errd"].append(point["ERRDENS"])
        for point in full_x_profs["PPROF"]:
            if point["R"] <= 0.0:
                break
            x_profs["rp"].append(point["R"])
            x_profs["p"].append(point["P"])
            x_profs["errp"].append(point["ERRP"])
        for point in full_x_profs["KTPROF"]:
            if point["R"] <= 0.0:
                break
            x_profs["rt"].append(point["R"])
            x_profs["t"].append(point["T"])
            x_profs["errt"].append(point["ERRT"])
        for point in full_x_profs["KPROF"]:
            if point["R"] <= 0.0:
                break
            x_profs["rk"].append(point["R"])
            x_profs["k"].append(point["K"])
            x_profs["errk"].append(point["ERRK"])
        for point in full_x_profs["MASSPROF"]:
            if point["R"] <= 0.0:
                break
            x_profs["rm"].append(point["R"])
            x_profs["m"].append(point["M"])
            x_profs["errm"].append(point["ERRM"])
        for k in x_profs.keys():
            x_profs[k] = np.array(x_profs[k])

    return x_profs
