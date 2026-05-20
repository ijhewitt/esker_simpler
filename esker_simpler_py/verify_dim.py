"""Detailed comparison of the dimensional MATLAB vs Python outputs.

Reports per-field absolute and relative error metrics (max, RMS, time-final),
and writes side-by-side overlay plots of all primary variables at the final
time so the agreement can be inspected visually.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def stats(m, p):
    """Return dict of error metrics, ignoring NaNs in either array."""
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(m) & np.isfinite(p)
    if not np.any(mask):
        return None
    d = m[mask] - p[mask]
    denom = np.max(np.abs(m[mask]))
    denom = denom if denom > 0 else 1.0
    return {
        "max_abs": float(np.max(np.abs(d))),
        "rms_abs": float(np.sqrt(np.mean(d ** 2))),
        "max_rel": float(np.max(np.abs(d)) / denom),
        "rms_rel": float(np.sqrt(np.mean(d ** 2)) / denom),
        "m_range": (float(np.min(m[mask])), float(np.max(m[mask]))),
        "p_range": (float(np.min(p[mask])), float(np.max(p[mask]))),
    }


def main():
    base = Path(__file__).resolve().parent
    m = loadmat(base.parent / "output" / "matlab_output_dim.mat", squeeze_me=True)
    p = loadmat(base / "output" / "python_output_dim.mat", squeeze_me=True)

    geom = ["x", "Z_b", "Z_s", "phi_s", "psi_s", "td"]
    state = ["S", "phi", "Q", "A", "N", "Qs", "Qeq", "m", "m_s"]

    print("=" * 90)
    print("DIMENSIONAL MODEL: MATLAB vs Python  (matlab_output_dim.mat vs python_output_dim.mat)")
    print("=" * 90)

    print("\nGeometry / time arrays (should be ~machine precision):")
    print(f"{'field':<8} {'max_abs':>14} {'max_rel':>12}")
    for f in geom:
        s = stats(m[f], p[f])
        print(f"{f:<8} {s['max_abs']:>14.3e} {s['max_rel']:>12.3e}")

    print("\nState variables — full space-time arrays:")
    hdr = f"{'field':<6} {'max_abs':>12} {'rms_abs':>12} {'max_rel':>10} {'rms_rel':>10} {'MATLAB range':>22} {'Python range':>22}"
    print(hdr)
    print("-" * len(hdr))
    worst_rel = 0.0
    worst_field = ""
    for f in state:
        s = stats(m[f], p[f])
        mr = f"[{s['m_range'][0]:.3g}, {s['m_range'][1]:.3g}]"
        pr = f"[{s['p_range'][0]:.3g}, {s['p_range'][1]:.3g}]"
        print(f"{f:<6} {s['max_abs']:>12.3e} {s['rms_abs']:>12.3e} {s['max_rel']:>10.2%} {s['rms_rel']:>10.2%} {mr:>22} {pr:>22}")
        if s["max_rel"] > worst_rel:
            worst_rel = s["max_rel"]
            worst_field = f

    print("\nState variables — final time slice only (t = td[-1]):")
    print(hdr)
    print("-" * len(hdr))
    for f in state:
        s = stats(m[f][:, -1], p[f][:, -1])
        mr = f"[{s['m_range'][0]:.3g}, {s['m_range'][1]:.3g}]"
        pr = f"[{s['p_range'][0]:.3g}, {s['p_range'][1]:.3g}]"
        print(f"{f:<6} {s['max_abs']:>12.3e} {s['rms_abs']:>12.3e} {s['max_rel']:>10.2%} {s['rms_rel']:>10.2%} {mr:>22} {pr:>22}")

    # Per-timestep RMS error for the dominant state variables
    print("\nPer-timestep RMS error (sampled):")
    times = m["td"]
    sample_idx = np.linspace(0, len(times) - 1, 6).astype(int)
    print(f"{'t [s]':>12} " + " ".join(f"{f:>12}" for f in ["S", "phi", "Q", "Qs"]))
    for k in sample_idx:
        rms_per_field = []
        for f in ["S", "phi", "Q", "Qs"]:
            a = m[f][:, k]; b = p[f][:, k]
            mask = np.isfinite(a) & np.isfinite(b)
            denom = max(np.max(np.abs(a[mask])), 1e-30)
            rms_per_field.append(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)) / denom)
        print(f"{times[k]:>12.3e} " + " ".join(f"{v:>12.3%}" for v in rms_per_field))

    print(f"\nWorst max-relative-error field across all space-time: {worst_field} = {worst_rel:.3%}")
    if worst_field == "m_s":
        print("(m_s is a piecewise/threshold-switched field; sensitive to solver path.)")

    # ---- side-by-side overlay plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_km = m["x"] / 1e3
    rho_w = 1000.0  # both runs use the same constant
    g = 9.8

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    plot_fields = [
        ("S",   r"$S$ [m$^2$]"),
        ("phi", r"$\phi/(\rho_w g)$ [m]"),
        ("Q",   r"$Q$ [m$^3$/s]"),
        ("N",   r"$N$ [Pa]"),
        ("Qs",  r"$Q_s$ [m$^3$/s]"),
        ("A",   r"$A$ [m$^2$]"),
    ]
    for ax, (f, label) in zip(axes.ravel(), plot_fields):
        mv = m[f][:, -1].copy()
        pv = p[f][:, -1].copy()
        if f == "phi":
            mv = mv / (rho_w * g); pv = pv / (rho_w * g)
        ax.plot(x_km, mv, "k-", lw=2, label="MATLAB")
        ax.plot(x_km, pv, "r--", lw=1.5, label="Python")
        ax.set_xlabel(r"$x$ [km]")
        ax.set_ylabel(label)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Dimensional run — MATLAB vs Python at final time")
    fig.tight_layout()
    out_path = base / "output" / "compare_dim_final.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nOverlay plot saved to {out_path}")


if __name__ == "__main__":
    main()
