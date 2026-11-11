"""Data of the SO(2) proof figure: the interval cover of rx_F_so2cert.py for TRPO seed 2, and V and its one-step change at points."""
import os
import runpy
import sys

import numpy as np

from paths import OUT, REPO

L = REPO / "lie_rl"
d, s = "trpo_SO2_ls-1.5", 2
os.chdir(L)
sys.argv = ["x", d, str(s), f"results/final/cert/SO2__{d}__s{s}__base.pkl", f"results/final/eval/SO2__{d}__s{s}__base.json",
            str(OUT / f"so2cert_{d}_{s}.json")]
g = runpy.run_path("scripts/rx_F_so2cert.py")
z = np.linspace(-g["gamma"], g["gamma"], 40001)
Vz = g["V_int"]((z, z))[0]
Vp = 0.5 * (Vz[0] + Vz[1])
Fv, _ = g["F_int"]((z, z))
Fp = 0.5 * (Fv[0] + Fv[1])
VF = g["V_int"]((Fp, Fp))[0]
S = g["S"]
np.savez(OUT / f"{d}_{s}.npz", z=z, V=Vp, dV=0.5 * (VF[0] + VF[1]) - Vp, LO=g["LO"], HI=g["HI"], gU=S["gU"], Vlo=S["Vlo"],
         Vhi=S["Vhi"], l2hi=S["l2hi"], rho=g["rho"], lvl=g["lvl"], gamma=g["gamma"], r_b=g["r_b"], delta=g["delta"])
print(d, s, g["rho"], g["lvl"], len(g["LO"]))
