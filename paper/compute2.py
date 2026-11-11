"""Numbers of the text (Sec 5, conclusion, abstract) from rows_<MODE>.json of build2.py: V against V_d and V_P separately."""
import json, os, statistics as st, sys
from decimal import Decimal, ROUND_HALF_UP
MODE = os.environ.get("MODE", "final"); from paths import OUT
D = json.load(open(OUT / f"rows_{MODE}.json"))
q = lambda x, d="0.1": str(Decimal(f"{x:.9f}").quantize(Decimal(d), rounding=ROUND_HALF_UP))
nom, pert = D["nom"], D["pert"]
print("== 39 policies")
whole = [r for r in nom if all(abs(r[n] - r["orc"]) <= 0.01 for n in ("V", "Vd", "VP"))]; print(" all four whole ball (within 0.01):", len(whole))
oth = [r for r in nom if r not in whole]; print(" others:", len(oth), [r["id"] for r in oth])
gt = [r for r in oth if r["V"] > max(r["Vd"], r["VP"])]; print(" V > both baselines on", len(gt), "of the others")
g1 = [r for r in oth if r["V"] - max(r["Vd"], r["VP"]) > 1]; print(" by >1 point on", len(g1), [r["id"] for r in g1])
five = [r for r in g1 if r["G"] in ("SE2", "SE3")]; print(" SE(n) among them:", len(five))
for b in ("Vd", "VP"): print(f"   V-{b} on these: {q(min(r['V']-r[b] for r in five))} .. {q(max(r['V']-r[b] for r in five))}")
for r in sorted(five, key=lambda r: r["id"]): print(f"     {r['id']}: V {r['V']:.3f} Vd {r['Vd']:.3f} VP {r['VP']:.3f} oracle {r['orc']:.3f}")
worst = min(min(r["V"] - r["Vd"], r["V"] - r["VP"]) for r in nom); print(" most negative V - baseline over 39: %.4f points = %.1f sample points" % (worst, -worst/100*2e5))
vo = [100*r["V"]/r["orc"] for r in nom]; print(" V/oracle >= 99 on", sum(x >= 99 for x in vo), "; below:", [(r["id"], q(100*r["V"]/r["orc"]), q(r["V"]), q(r["orc"])) for r in nom if 100*r["V"]/r["orc"] < 99])
print("== 55 closed loops with an actuator error")
for b in ("Vd", "VP"):
    m = [r["V"] - r[b] for r in pert]
    print(f"  V-{b}: min {q(min(m))} median {q(st.median(m))} max {q(max(m))}; V>{b} on {sum(x>0 for x in m)}; raw min {min(m):.4f} median {st.median(m):.4f} max {max(m):.4f}")
vo = [100*r["V"]/r["orc"] for r in pert]
print("  V/oracle median", q(st.median(vo)), "raw %.4f" % st.median(vo), " >=99 on", sum(x >= 99 for x in vo), " min", q(min(vo)))
for b in ("Vd", "VP"): x = [100*r[b]/r["orc"] for r in pert]; print(f"  {b}/oracle median {q(st.median(x))} raw {st.median(x):.4f}")
low = [r for r in pert if 100*r["V"]/r["orc"] < 99]
print("  below 99:", len(low), "PPO SE3 among them:", sum(r["G"] == "SE3" and r["alg"] == "PPO" for r in low))
for r in low: print("    ", r["id"], q(100*r["V"]/r["orc"]))
