"""Turn the paper's actuator-error table (tab_pert_all.tex) into docs/assets/data/actuator_loops.json for the page.

Usage: python web/table_data.py PATH/TO/tab_pert_all.tex
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def num(s):
    s = re.sub(r"\$\^\{?[a-z*]+\}?\$|\$\^\\star\$|\^\{?[a-z*]+\}?", "", s).replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main(path):
    rows, group = [], None
    for line in Path(path).read_text().splitlines():
        if "&" not in line or line.lstrip().startswith("group"):
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if cells[0]:
            group = cells[0].replace("$", "").replace("\\", "")
        algo, seed = cells[1].split()
        err = cells[2].replace("$^\\circ$", "°")
        kept = "^{c}" in cells[5]
        r = dict(group=group, algo=algo, seed=int(seed), error=err, spr=num(cells[3]), oracle=num(cells[4]),
                 V=num(cells[5]), Vd=num(cells[6]), VP=num(cells[7]), Vh=num(cells[8]), rV=num(cells[9]), rVd=num(cells[10]), rVP=num(cells[11]), rVh=num(cells[12]),
                 noVh=('^{*}' in cells[8]), kept=kept)
        rows.append(r)
    out = ROOT / "docs/assets/data/actuator_loops.json"
    out.write_text(json.dumps(rows, indent=1))
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
