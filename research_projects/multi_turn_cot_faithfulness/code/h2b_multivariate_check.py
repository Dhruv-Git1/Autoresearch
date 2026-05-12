"""Task 2 — Multivariate sanity check on H2b.

H2b (Session 6) found Spearman rho=-0.426, p=0.001 for
(anchor_rate x answer_update_rate) on N=59 conversations.
Long conversations have both higher anchoring AND potentially lower
update rates — so the bivariate result may be length-mediated.

This script runs:
1. OLS regression: answer_update_rate ~ anchor_rate + n_turns
2. Partial Spearman correlation of (anchor_rate, update_rate) after
   controlling for n_turns (via residuals approach)
3. Bivariate correlations of length with both variables

Determines Case A (H2b survives) or Case B (H2b collapses).
"""
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy import stats as scipy_stats

H2B_PATH = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\h2b\integration_stats.json"
OUT_DIR = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\h2b"


def partial_spearman(x, y, z):
    """Partial Spearman rho of x~y controlling for z, via residuals."""
    def rank(v):
        return scipy_stats.rankdata(v).tolist()
    rx = np.array(rank(x), dtype=float)
    ry = np.array(rank(y), dtype=float)
    rz = np.array(rank(z), dtype=float)

    def residuals(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)

    res_x = residuals(rx, rz)
    res_y = residuals(ry, rz)
    rho, p = pearsonr(res_x, res_y)
    return float(rho), float(p)


def main():
    d = json.load(open(H2B_PATH))
    rows = d["per_conv_data"]

    anchor = np.array([r["anchoring_rate"] for r in rows])
    update = np.array([r["answer_update_rate"] for r in rows])
    n_turns = np.array([r["n_turns"] for r in rows], dtype=float)
    N = len(rows)

    # Bivariate (reproduced from Session 6)
    rho_bi, p_bi = spearmanr(anchor, update)

    # Bivariate: length vs each
    rho_len_anchor, p_len_anchor = spearmanr(n_turns, anchor)
    rho_len_update, p_len_update = spearmanr(n_turns, update)

    # OLS: update ~ anchor + n_turns
    X = np.column_stack([np.ones(N), anchor, n_turns])
    betas, residuals, _, _ = np.linalg.lstsq(X, update, rcond=None)
    # Standard errors via OLS formula
    y_hat = X @ betas
    sigma2 = float(np.sum((update - y_hat) ** 2) / (N - 3))
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = betas / se
    p_values = [2 * (1 - scipy_stats.t.cdf(abs(t), df=N - 3)) for t in t_stats]

    # Partial Spearman
    rho_partial, p_partial = partial_spearman(anchor, update, n_turns)

    # Verdict
    anchor_p_ols = p_values[1]
    case = "A" if (anchor_p_ols < 0.05 or abs(p_partial) < 0.05) else "B"
    if p_partial < 0.05 and rho_partial < 0:
        case = "A"
    elif p_partial >= 0.05:
        case = "B"

    stats = {
        "N": N,
        "bivariate_spearman": {
            "rho": round(float(rho_bi), 4),
            "p": round(float(p_bi), 4),
        },
        "length_vs_anchoring": {
            "spearman_rho": round(float(rho_len_anchor), 4),
            "p": round(float(p_len_anchor), 4),
        },
        "length_vs_update_rate": {
            "spearman_rho": round(float(rho_len_update), 4),
            "p": round(float(p_len_update), 4),
        },
        "ols_regression": {
            "formula": "answer_update_rate ~ 1 + anchor_rate + n_turns",
            "intercept": {"beta": round(float(betas[0]), 4), "se": round(float(se[0]), 4), "p": round(float(p_values[0]), 4)},
            "anchor_rate": {"beta": round(float(betas[1]), 4), "se": round(float(se[1]), 4), "p": round(float(p_values[1]), 4)},
            "n_turns": {"beta": round(float(betas[2]), 4), "se": round(float(se[2]), 4), "p": round(float(p_values[2]), 4)},
        },
        "partial_spearman_given_length": {
            "rho_partial": round(rho_partial, 4),
            "p": round(p_partial, 4),
        },
        "verdict": case,
        "interpretation": (
            f"Case {case}: H2b {'SURVIVES' if case == 'A' else 'COLLAPSES'} length control. "
            f"Bivariate rho={rho_bi:.3f} (p={p_bi:.3f}); "
            f"Partial rho={rho_partial:.3f} (p={p_partial:.3f}); "
            f"OLS anchor_rate beta={betas[1]:.3f} (p={p_values[1]:.3f}), "
            f"n_turns beta={betas[2]:.4f} (p={p_values[2]:.3f})"
        ),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "multivariate_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\n{'='*60}")
    print(f"VERDICT: Case {case}")
    print(f"  Bivariate:     rho={rho_bi:.3f}, p={p_bi:.4f}")
    print(f"  Partial:       rho={rho_partial:.3f}, p={p_partial:.4f}")
    print(f"  OLS anchor:    beta={betas[1]:.3f}, p={p_values[1]:.4f}")
    print(f"  OLS n_turns:   beta={betas[2]:.4f}, p={p_values[2]:.4f}")
    print(f"  len~anchor:    rho={rho_len_anchor:.3f}, p={p_len_anchor:.4f}")
    print(f"  len~update:    rho={rho_len_update:.3f}, p={p_len_update:.4f}")


if __name__ == "__main__":
    main()
