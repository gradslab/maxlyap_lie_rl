"""Faster D/E: after a recomputation, keep the larger of the carried and the recomputed region (as the refinement keeps
the candidate with the largest estimated region over its rounds). Copies the kept evaluation to FINAL_EVAL_JSON,
records the choice in PATH_JSON and prints the path of the kept candidate.

Usage: python scripts/rx_pick.py CARRIED_EVAL WARM_EVAL BASE_CERT WARM_CERT FINAL_EVAL_JSON PATH_JSON
"""
import json, shutil, sys
from pathlib import Path
car, wev, base, wcert, out, pth = sys.argv[1:7]
rc = json.load(open(car))["fns"]["V"]["region_mu0"]; rw = json.load(open(wev))["fns"]["V"]["region_mu0"]
src, kept, which = (wev, wcert, "warm") if rw >= rc else (car, base, "carry_after_warm")
Path(out).parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, out)
rec = json.load(open(pth)); rec.update(kept=which, warm_region_mu0=rw); json.dump(rec, open(pth, "w"), indent=1)
print(kept)
