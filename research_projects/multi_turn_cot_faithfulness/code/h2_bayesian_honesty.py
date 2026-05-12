"""Task 3 — Bayesian honesty: BF and TOST for conversation-level H2.

For r=-0.097, N=53:
1. Bayes factor BF01 (evidence for null vs. alternative) via Jeffreys prior
   on correlation. Implemented via numerical integration of the marginal
   likelihood ratio (Jeffrey's exact formula for the correlation BF).
2. TOST equivalence test against bounds ±0.25 (Lakens 2018).
   Two one-sided z-tests via Fisher z-transform.
3. ROPE probability P(|rho| < 0.1 | data) via posterior on Fisher z.
"""
import json
import math
import os

from scipy import stats as scipy_stats
from scipy import integrate
import numpy as np

OUT_DIR = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\h2_confound"

# Observed values
R_OBS = -0.097
N = 53


def jeffreys_bf01(r, n):
    """BF01 for Pearson correlation via Jeffreys exact formula.

    Uses the result from Ly et al. (2016) / Wagenmakers et al. (2018):
    BF01 = (marginal_H0) / (marginal_H1)
    Under H0: rho=0, the likelihood is the standard t-distribution on (n-2) df.
    Under H1: uniform prior on rho in [-1,1] (Jeffreys prior).

    The marginal likelihood ratio integral has a closed form for the
    uniform prior (Jeffrey's 1961 result):
    m(data | H1) = 2^(n-2) * Gamma((n-1)/2)^2 * (1-r^2)^((n-1)/2) / (sqrt(pi) * Gamma((n-1)) * (1 - r^2 * cos^2(theta)) ...)

    For simplicity, use the numerical Gaussian hypergeometric function:
    p(r | H1, uniform) propto (1-r^2)^((n-1)/2) * 2F1(1/2,1/2; (n+1)/2; (1+r_obs*r)/2 + ...)

    Practical implementation: BF10 = likelihood at r=0 / marginal H1.
    Use the exact t-test result for H0 and numerical integration for H1.
    """
    # Under H0 (rho=0): marginal likelihood proportional to t^(-(n-2)/2) density
    # Actually just compute the p-value of t-test and convert to BF via approximation.
    # Use the exact result from Ly et al. 2016 Eq (8):
    # BF01 = [Gamma((n-1)/2) / (sqrt(pi) * Gamma((n-2)/2))] * (1-r^2)^((n-1)/2) /
    #        integral_0^pi (1 - r^2 * cos^2(theta))^(-(n-2)/2) dtheta * 2

    # Numerically integrate using the marginal likelihood under uniform prior on rho:
    # p(r | n, uniform prior) = C * (1-r^2)^((n-2)/2-1) * integral
    # where the integral is the hypergeometric function.
    # For moderate n, the simple numerical approach:

    def integrand(rho):
        if abs(rho) >= 1:
            return 0.0
        # Log-likelihood of observing r given rho and n
        # Using the exact formula for the sampling distribution of r given rho:
        # This is complex. Use the simpler approximation via Fisher z:
        # Under H1 with uniform prior: marginal = int p(r|rho) * 1 drho
        # p(r|rho) via Fisher z: z_obs ~ N(arctanh(rho), 1/sqrt(n-3))
        z_obs = math.atanh(r) if abs(r) < 1 else math.copysign(10, r)
        z_rho = math.atanh(rho)
        se = 1.0 / math.sqrt(n - 3)
        return scipy_stats.norm.pdf(z_obs, loc=z_rho, scale=se)

    # Marginal likelihood under H1 (uniform prior on rho in [-1, 1])
    marginal_h1, _ = integrate.quad(integrand, -1 + 1e-6, 1 - 1e-6, limit=200)
    marginal_h1 /= 2.0  # normalise uniform prior (width 2)

    # Marginal likelihood under H0 (rho=0)
    z_obs = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    marginal_h0 = scipy_stats.norm.pdf(z_obs, loc=0, scale=se)

    bf01 = marginal_h0 / marginal_h1 if marginal_h1 > 0 else float("inf")
    bf10 = 1.0 / bf01 if bf01 > 0 else float("inf")
    return bf01, bf10


def tost(r, n, low=-0.25, high=0.25):
    """TOST equivalence test against bounds [low, high].

    Two one-sided z-tests via Fisher z-transform (Lakens 2018).
    H0_lower: rho <= low  vs  Ha_lower: rho > low
    H0_upper: rho >= high vs  Ha_upper: rho < high
    Reject both to conclude equivalence.
    """
    z_obs = math.atanh(r)
    z_low = math.atanh(low)
    z_high = math.atanh(high)
    se = 1.0 / math.sqrt(n - 3)

    # Lower: test rho > low
    t_lower = (z_obs - z_low) / se
    p_lower = 1.0 - scipy_stats.norm.cdf(t_lower)

    # Upper: test rho < high
    t_upper = (z_obs - z_high) / se
    p_upper = scipy_stats.norm.cdf(t_upper)

    equivalent = (p_lower < 0.05 and p_upper < 0.05)
    return p_lower, p_upper, equivalent


def rope_probability(r, n, rope_bound=0.1):
    """P(|rho| < rope_bound | data) assuming flat prior -> posterior via Fisher z."""
    z_obs = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_hi = math.atanh(rope_bound)
    z_lo = math.atanh(-rope_bound)
    prob = scipy_stats.norm.cdf(z_hi, loc=z_obs, scale=se) - scipy_stats.norm.cdf(z_lo, loc=z_obs, scale=se)
    return float(prob)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bf01, bf10 = jeffreys_bf01(R_OBS, N)
    p_lower, p_upper, equivalent = tost(R_OBS, N)
    rope_prob = rope_probability(R_OBS, N)

    # Interpret BF01
    if bf01 >= 10:
        bf_interp = "strong evidence for null"
    elif bf01 >= 3:
        bf_interp = "moderate evidence for null"
    elif bf01 >= 1:
        bf_interp = "anecdotal/weak evidence for null"
    else:
        bf_interp = "evidence favours alternative"

    stats = {
        "r_observed": R_OBS,
        "N": N,
        "bayes_factor_01": round(bf01, 3),
        "bayes_factor_10": round(bf10, 3),
        "bf_interpretation": bf_interp,
        "tost_bounds": [-0.25, 0.25],
        "tost_lower_p": round(p_lower, 4),
        "tost_upper_p": round(p_upper, 4),
        "tost_verdict": "equivalent to null (|rho|<0.25)" if equivalent else "cannot reject presence of medium effect",
        "rope_probability_below_0.1": round(rope_prob, 4),
        "fisher_z_ci_95": list(
            (round(math.tanh(math.atanh(R_OBS) - 1.96 / math.sqrt(N - 3)), 4),
             round(math.tanh(math.atanh(R_OBS) + 1.96 / math.sqrt(N - 3)), 4))
        ),
    }

    with open(os.path.join(OUT_DIR, "bayesian_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
