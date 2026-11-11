"""The anchored candidate: the maximal Lyapunov function's own second-order term at the goal, blended with the
learned candidate.
"""
import jax.numpy as jp
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import certificate as C
from . import roa as R

_V0 = C.V_chart


def local_quadratic(T_chart, G, m, h=1e-4):
    """A = DF(0) (central differences), H = Hessian of l^2 at 0 (central differences), P_m from the Stein equation."""
    A = R.jacobian_at_zero(T_chart, m)
    ref = G.obs(G.identity(1))
    f = lambda x: float(jp.sum((G.obs(G.Exp(jp.asarray(x)[None])) - ref) ** 2))
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            ei = np.zeros(m); ei[i] = h; ej = np.zeros(m); ej[j] = h
            H[i, j] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * h * h) / 2.0
    H = 0.5 * (H + H.T)
    P = solve_discrete_lyapunov(A.T, H)
    return 0.5 * (P + P.T), A, H


def anchored(P, r_a):
    Pj = jp.asarray(P); ra2 = float(r_a) ** 2

    def V_chart(p, cfg, ell, z):
        q = jp.einsum("ni,ij,nj->n", z, Pj, z)
        r2 = jp.sum(z * z, axis=-1)
        s = r2 / (r2 + ra2)
        return (1.0 - s) * q + s * _V0(p, cfg, ell, z)
    return V_chart


def enable(P, r_a):
    from . import rx_certificate as RC
    V = anchored(P, r_a)
    C.V_chart = V
    RC.V_chart = V
    return V


def enable_from_cert(cert):
    """if the certificate pickle carries an anchor, switch the process to the anchored candidate."""
    a = cert.get("anchor")
    if a is None:
        return False
    enable(np.asarray(a["P"]), a["r_a"])
    return True
