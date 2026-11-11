"""Decide whether a perturbed closed loop needs a recomputed candidate.

Usage: python scripts/rx_decide.py CARRIED_EVAL_JSON ORACLE_JSON NOMINAL_JSON PATH_JSON
"""
import json, os, sys
from pathlib import Path
ev, orc, nom, out = sys.argv[1:5]
tol = float(os.environ.get("TOL", "2")) / 100
e = json.load(open(ev)); o = json.load(open(orc))
n = json.load(open(nom)) if Path(nom).exists() else None
carried = e["fns"]["V"]["region_mu0"]; oracle = o["region_mu0"]
path = "carry" if carried >= oracle - tol else "warm"
rec = dict(path=path, carried_region_mu0=carried, oracle_region_mu0=oracle, tol=tol,
           nominal_still_holds=None if n is None else bool(n["decrease_violations"] == 0 and n["containment_violations"] == 0
                                                             and n["nonconverged_in_nominal_region_mu0"] == 0))
Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out, "w"), indent=1)
print(path)
