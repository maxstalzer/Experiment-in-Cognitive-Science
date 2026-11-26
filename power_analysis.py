"""
Sensitivity / power analysis for the tempo experiment.

Design: within-subjects repeated measures, 1 factor, 3 levels (90/120/150 BPM).
Unit of analysis: participant (n = 16). The 120 pairwise synchrony values are
NOT independent observations, so effective N is bounded by participant count.

Replicates G*Power "ANOVA: Repeated measures, within factors".
"""

import numpy as np
from scipy.stats import f as fdist, ncf

N = 16        # participants
M = 3         # tempo levels
RHO = 0.5     # assumed correlation among repeated measures
ALPHA = 0.05
TARGET = 0.80


def rm_power(n, f_eff, m=M, rho=RHO, eps=1.0, alpha=ALPHA):
    """Power of the within-subjects F test for Cohen's f = f_eff."""
    df1 = (m - 1) * eps
    df2 = (n - 1) * (m - 1) * eps
    if df2 <= 0:
        return 0.0
    lam = f_eff**2 * n * m * eps / (1 - rho)        # noncentrality
    fcrit = fdist.ppf(1 - alpha, df1, df2)
    return 1 - ncf.cdf(fcrit, df1, df2, lam)


def n_required(f_eff, target=TARGET, **kw):
    """Smallest n reaching the target power."""
    for n in range(3, 2000):
        if rm_power(n, f_eff, **kw) >= target:
            return n
    return None


def min_detectable_f(n, target=TARGET, **kw):
    """Smallest Cohen's f detectable with n participants."""
    for f_eff in np.arange(0.01, 2.0, 0.005):
        if rm_power(n, f_eff, **kw) >= target:
            return round(float(f_eff), 3)
    return None


def f_to_eta2p(f_eff):
    return f_eff**2 / (1 + f_eff**2)


if __name__ == "__main__":
    print(f"n = {N} participants, {M} levels, alpha = {ALPHA}, rho = {RHO}\n")

    f_min = min_detectable_f(N)
    print("SENSITIVITY — smallest detectable effect at 80% power")
    print(f"  f >= {f_min:.2f}  (partial eta^2 = {f_to_eta2p(f_min):.3f})\n")

    print("ACHIEVED POWER at literature effect sizes (Bernardi et al. 2017)")
    for f_eff in [0.14, 0.20]:
        print(f"  f = {f_eff:.2f} (eta^2p = {f_to_eta2p(f_eff):.3f})"
              f"  ->  power = {rm_power(N, f_eff):.2f}")
    print()

    print("N REQUIRED for 80% power")
    for f_eff in [0.14, 0.20, 0.25, 0.40]:
        print(f"  f = {f_eff:.2f}  ->  {n_required(f_eff)} participants")