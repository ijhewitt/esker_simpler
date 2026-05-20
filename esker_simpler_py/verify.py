"""Compare MATLAB and Python outputs and print error metrics."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat


def rel_err(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan"), float("nan")
    diff = a[mask] - b[mask]
    abs_err = np.max(np.abs(diff))
    denom = np.max(np.abs(a[mask])) if np.max(np.abs(a[mask])) > 0 else 1.0
    return abs_err, abs_err / denom


def main():
    import sys as _sys
    suffix = _sys.argv[1] if len(_sys.argv) > 1 else ""  # "" or "_dim"
    base = Path(__file__).resolve().parent
    m = loadmat(base.parent / "output" / f"matlab_output{suffix}.mat", squeeze_me=True)
    p = loadmat(base / "output" / f"python_output{suffix}.mat", squeeze_me=True)

    print(f"Comparing matlab_output{suffix}.mat vs python_output{suffix}.mat")
    print(f"{'field':<10} {'max_abs_err':>14} {'max_rel_err':>14}")
    print("-" * 42)
    fields = ["x", "Z_b", "Z_s", "phi_s", "psi_s", "td",
              "S", "phi", "Q", "A", "N", "Qs", "Qeq", "m", "m_s"]
    worst = 0.0
    for f in fields:
        if f not in m or f not in p:
            print(f"{f:<10}  (missing)")
            continue
        ae, re = rel_err(m[f], p[f])
        print(f"{f:<10} {ae:>14.3e} {re:>14.3e}")
        if np.isfinite(re):
            worst = max(worst, re)

    print(f"\nWorst relative error across fields: {worst:.3e}")
    threshold = 5e-2
    if worst < threshold:
        print(f"PASS: results agree within {threshold:.0%} relative tolerance.")
    else:
        print(f"WARN: relative error exceeds {threshold:.0%}.")


if __name__ == "__main__":
    main()
