"""Python port of esker_simpler_run_dim.m

Same problem as esker_simpler_run.py, but feeds the solver dimensional
inputs (the solver performs the non-dim/re-dim internally).
Saves outputs and figures to ./output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esker_simpler import esker_simpler


def main():
    oo = {"nondiminput": False}

    # PARAMETERS (dimensional)
    pd = {}
    pd["rho_i"] = 916.0
    pd["rho_w"] = 1000.0
    pd["g"] = 9.8
    pd["f"] = 0.2
    pd["L"] = 3.3e5
    pd["ty"] = 365 * 24 * 60 * 60.0
    pd["l_eq"] = 100.0
    pd["rho_s"] = 2600.0
    pd["D_50"] = 1e-3
    pd["n_s"] = 0.4
    pd["tau_s"] = 0.047
    pd["n"] = 3.0
    pd["A_i"] = 6.8e-24 * (pd["n"] ** pd["n"] / 2.0)
    pd["n_sed"] = 1.0
    pd["A_s"] = 0.0
    pd["Z_0"] = 0.0
    pd["h_r"] = 0.1
    pd["l_r"] = 1.0
    pd["u_b"] = 30.0 / pd["ty"]
    pd["l_c"] = 10e3
    pd["sigma"] = 0.0
    pd["alpha"] = 5.0 / 4.0
    pd["k_c"] = (2 * 2 ** (1 / 4) * np.pi ** (1 / 4)
                 / (np.pi + 2) ** 0.5 / pd["rho_w"] ** 0.5 / pd["f"] ** 0.5)
    pd["gamma"] = 0.5
    pd["Atilde_i"] = 2 * pd["A_i"] / pd["n"] ** pd["n"]
    pd["Atilde_s"] = 2 * pd["A_s"] / pd["n"] ** pd["n"]
    pd["k_s"] = (8 * ((pd["rho_s"] - pd["rho_w"]) / pd["rho_w"]) ** 0.5
                 * pd["g"] ** 0.5 * pd["D_50"] ** 1.5 * (8 / np.pi) ** 0.5)
    pd["A_min"] = 0.0

    # DIMENSIONAL GEOMETRY AND INPUTS
    ad = {}
    ad["x"] = np.linspace(-50e3, 0, 400)
    ad["Z_b"] = 0.0 - 400.0 / (pd["rho_w"] / pd["rho_i"]) + 0 * ad["x"]
    ad["Z_s"] = ad["Z_b"] + (400.0 + 0.01 * (-ad["x"]))

    M_in_const = 10.0 / (ad["x"][-1] - ad["x"][0])
    ad["M_in"] = lambda t, _M=M_in_const, _x=ad["x"]: _M * np.ones_like(_x)
    ad["Ms_in"] = lambda t, _x=ad["x"]: np.zeros_like(_x)
    ad["xi_m"] = np.array([0], dtype=int)  # MATLAB 1 -> Python 0
    ad["S_m"] = np.array([0.0])
    ad["Q_m"] = lambda t: np.array([0.0])
    ad["Qs_m"] = lambda t: np.array([0.0])

    # STAGE 1: steady state with sediment transport off
    rem = pd["tau_s"]
    pd["tau_s"] = np.inf
    td1 = np.arange(0, 20 + 1e-9, 0.1) * (pd["ty"] / 365.0)
    td1_out, ad1, ud1 = esker_simpler(td1, ad, pd, None, oo)

    # STAGE 2: turn on sediment transport
    pd["tau_s"] = rem
    ad["Qs_m"] = lambda t: np.array([0.0])
    ad["Ms_in"] = lambda t, _x=ad["x"], _M=M_in_const: 1e-3 * _M * np.ones_like(_x)
    td2 = np.arange(0, 50 + 1e-9, 0.1) * (pd["ty"] / 365.0)
    td2_out, ad_out, ud = esker_simpler(td2, ad, pd, ud1[-1], oo)

    # Assemble output arrays
    nI = len(ad_out["x"])
    nt = len(ud)
    fields = ["S", "phi", "Q", "A", "N", "Qs", "Qeq", "m", "m_s"]
    arrs = {f: np.zeros((nI, nt)) for f in fields}
    for k, uk in enumerate(ud):
        for f in fields:
            arrs[f][:, k] = uk[f]

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_output_dim.mat"
    savemat(str(out_path), {
        "x": ad_out["x"], "Z_b": ad_out["Z_b"], "Z_s": ad_out["Z_s"],
        "phi_s": ad_out["phi_s"], "psi_s": ad_out["psi_s"], "td": td2_out,
        **arrs, "pd": pd,
    })

    print(f"Python (dim) output saved to {out_path}. nt={nt}, nI={nI}")
    print(f"S(end) range: [{np.nanmin(arrs['S'][:,-1]):g}, {np.nanmax(arrs['S'][:,-1]):g}]")
    print(f"phi(end) range: [{np.nanmin(arrs['phi'][:,-1]):g}, {np.nanmax(arrs['phi'][:,-1]):g}]")

    # FIGURES (dimensional units throughout)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ti = nt - 1
    x_km = ad_out["x"] / 1e3
    t_days = td2_out / pd["ty"] * 365.0

    fig1, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes
    ax0.plot(x_km, arrs["Q"][:, ti], "b", lw=2); ax0.set_ylabel(r"$Q$ [m$^3$/s]")
    ax1.plot(x_km, ad_out["Z_b"], "k", lw=2)
    ax1.plot(x_km, ad_out["Z_s"], "k", lw=2)
    ax1.plot(x_km, ad_out["phi_s"] / pd["rho_w"] / pd["g"], "k--", lw=2)
    ax1.plot(x_km, arrs["phi"][:, ti] / pd["rho_w"] / pd["g"], "b", lw=2)
    ax1.set_ylabel(r"Elevation $z$ [m]")
    ax2.plot(x_km, (ad_out["phi_s"] - arrs["phi"][:, ti]) / (pd["rho_w"] * pd["g"]), "b", lw=2)
    ax2.set_ylabel(r"$N$ [m w.e.]")
    ax3.plot(x_km, arrs["S"][:, ti], "b", lw=2); ax3.set_ylabel(r"$S$ [m$^2$]")
    ax4.plot(x_km, arrs["Qeq"][:, ti], "b--", lw=2)
    ax4.plot(x_km, arrs["Qs"][:, ti], "b-", lw=2)
    ax4.set_ylabel(r"$Q_s$ [m$^3$/s]")
    ax5.plot(x_km, arrs["A"][:, ti], "b", lw=2); ax5.set_ylabel(r"$A$ [m$^2$]")
    ax5.set_xlabel(r"Distance $x-x_m$ [km]")
    fig1.tight_layout()
    fig1.savefig(out_dir / "esker_simpler_run_dim_fig1.pdf")
    plt.close(fig1)

    fig2, axs = plt.subplots(2, 2, figsize=(11, 8))
    extent = [x_km.min(), x_km.max(), t_days.min(), t_days.max()]
    im = axs[0, 0].imshow(arrs["N"].T / 1e6, origin="lower", aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[0, 0]).set_label(r"$N$ [MPa]")
    axs[0, 0].set_xlabel(r"$x$ [km]"); axs[0, 0].set_ylabel(r"$t$ [d]")
    im = axs[0, 1].imshow(arrs["S"].T, origin="lower", aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[0, 1]).set_label(r"$S$ [m$^2$]")
    axs[0, 1].set_xlabel(r"$x$ [km]"); axs[0, 1].set_ylabel(r"$t$ [d]")
    im = axs[1, 0].imshow(arrs["Qs"].T, origin="lower", aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[1, 0]).set_label(r"$Q_s$ [m$^3$/s]")
    axs[1, 0].set_xlabel(r"$x$ [km]"); axs[1, 0].set_ylabel(r"$t$ [d]")
    E = (pd["ty"] / 365.0) * arrs["m_s"].T / (1 - pd["n_s"])
    im = axs[1, 1].imshow(E, origin="lower", aspect="auto", extent=extent,
                          cmap="bwr", vmin=-1, vmax=1)
    fig2.colorbar(im, ax=axs[1, 1]).set_label(r"$E$ [m$^2$/d]")
    axs[1, 1].set_xlabel(r"$x$ [km]"); axs[1, 1].set_ylabel(r"$t$ [d]")
    fig2.tight_layout()
    fig2.savefig(out_dir / "esker_simpler_run_dim_fig2.pdf")
    plt.close(fig2)

    print(f"Figures saved to {out_dir}/esker_simpler_run_dim_fig{{1,2}}.pdf")


if __name__ == "__main__":
    main()
