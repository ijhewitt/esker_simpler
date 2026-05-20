"""Python port of esker_simpler_run.m

Runs the same two-stage simulation (steady state, then with sediment) and
saves a .mat file with all outputs to ../output/python_output.mat for
comparison against the MATLAB reference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esker_simpler import esker_simpler


def main():
    oo = {"nondiminput": True}

    # PARAMETERS
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

    # SCALING
    pd["Q"] = 10.0
    pd["x"] = 5000.0
    pd["z"] = 500.0
    pd["phi"] = pd["rho_i"] * pd["g"] * pd["z"]
    pd["psi"] = pd["phi"] / pd["x"]
    pd["S"] = (pd["Q"] / pd["k_c"] / pd["psi"] ** 0.5) ** (1.0 / pd["alpha"])
    pd["m"] = pd["Q"] * pd["psi"] / pd["rho_w"] / pd["L"]
    pd["U"] = pd["Q"] / pd["S"]
    pd["Qs"] = (pd["k_s"] * pd["S"] ** pd["gamma"]
                * (1.0 / 8.0 * pd["f"] * pd["rho_w"] * pd["U"] ** 2
                   / ((pd["rho_s"] - pd["rho_w"]) * pd["g"] * pd["D_50"])) ** 1.5)
    pd["m_s"] = pd["Qs"] / pd["x"]
    pd["t"] = pd["ty"] / 365.0
    pd["A"] = pd["S"]

    # DIMENSIONLESS PARAMETERS
    pp = {}
    pp["alpha"] = pd["alpha"]
    pp["n"] = pd["n"]
    pp["n_sed"] = pd["n_sed"]
    pp["gamma"] = pd["gamma"]
    pp["r"] = pd["rho_w"] / pd["rho_i"]
    pp["epsilon"] = pd["S"] * pd["x"] / pd["Q"] / pd["t"]
    pp["nu"] = pd["l_eq"] / pd["x"]
    pp["mu_i"] = pd["Q"] * pd["psi"] / pd["rho_i"] / pd["L"] / pd["S"] * pd["t"]
    pp["mu_s"] = pd["m_s"] / (1 - pd["n_s"]) / pd["S"] * pd["t"]
    pp["omega_i"] = pd["Atilde_i"] * pd["S"] ** (2 * pd["gamma"]) * pd["phi"] ** pd["n"] / pd["S"] * pd["t"]
    pp["omega_s"] = pd["Atilde_s"] * pd["S"] ** (2 * pd["gamma"]) * pd["phi"] ** pd["n_sed"] / pd["S"] * pd["t"]
    pp["upsilon"] = pd["h_r"] * pd["u_b"] / pd["S"] * pd["t"]
    pp["upsilon2"] = pd["h_r"] * pd["l_r"] / pd["S"]
    pp["beta"] = pd["rho_i"] / pd["rho_w"] * pd["S"] * pd["z"] / pd["Q"] / pd["t"]
    pp["sigma"] = pd["rho_i"] / pd["rho_w"] * pd["sigma"] * pd["l_c"] * pd["z"] * pd["x"] / pd["Q"] / pd["t"]
    pp["tau_s"] = (pd["tau_s"] * ((pd["rho_s"] - pd["rho_w"]) * pd["g"] * pd["D_50"])
                   / (1.0 / 8.0 * pd["f"] * pd["rho_w"] * pd["U"] ** 2))
    pp["Z_0"] = pd["Z_0"] / pd["z"]
    pp["A_min"] = pd["A_min"] / pd["A"]
    pp["eps38"] = 1e-4
    pp["eps53"] = 1e-4
    pp["epsilon"] = 0.0  # remove melt contribution to mass conservation

    # GEOMETRY (dimensionless)
    aa = {}
    aa["x"] = np.linspace(-50e3, 0, 400) / pd["x"]
    aa["Z_b"] = 0.0 / pd["z"] - 400.0 / pp["r"] / pd["z"] + 0 * aa["x"]
    aa["Z_s"] = aa["Z_b"] + (400.0 + 0.01 * (-pd["x"] * aa["x"])) / pd["z"]

    M_in_const = (10.0 / pd["Q"]) / (aa["x"][-1] - aa["x"][0])
    aa["M_in"] = lambda t, _M=M_in_const, _x=aa["x"]: _M * np.ones_like(_x)
    aa["Ms_in"] = lambda t, _x=aa["x"]: np.zeros_like(_x)
    # MATLAB index 1 -> Python index 0
    aa["xi_m"] = np.array([0], dtype=int)
    aa["S_m"] = np.array([0.0])
    aa["Q_m"] = lambda t: np.array([0.0])
    aa["Qs_m"] = lambda t: np.array([0.0])

    # STAGE 1: steady state with sediment transport off (tau_s = inf)
    rem = pp["tau_s"]
    pp["tau_s"] = np.inf
    td1 = np.arange(0, 10 + 1e-9, 0.1) * (pd["ty"] / 365.0) / pd["t"]
    td1_out, ad1, ud1 = esker_simpler(td1, aa, pp, None, oo)

    # STAGE 2: turn on sediment
    pp["tau_s"] = rem
    aa["Qs_m"] = lambda t: np.array([0.0])
    aa["Ms_in"] = lambda t, _x=aa["x"], _M=M_in_const: 1e-1 * _M * np.ones_like(_x)
    td2 = np.arange(0, 50 + 1e-9, 0.1) * (pd["ty"] / 365.0) / pd["t"]
    td2_out, ad, ud = esker_simpler(td2, aa, pp, ud1[-1], oo)

    # Assemble output arrays
    nI = len(ad["x"])
    nt = len(ud)
    fields = ["S", "phi", "Q", "A", "N", "Qs", "Qeq", "m", "m_s"]
    arrs = {f: np.zeros((nI, nt)) for f in fields}
    for k, uk in enumerate(ud):
        for f in fields:
            arrs[f][:, k] = uk[f]

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_output.mat"
    savemat(str(out_path), {
        "x": ad["x"], "Z_b": ad["Z_b"], "Z_s": ad["Z_s"],
        "phi_s": ad["phi_s"], "psi_s": ad["psi_s"], "td": td2_out,
        **arrs, "pp": pp, "pd": pd,
    })

    print(f"Python output saved to {out_path}. nt={nt}, nI={nI}")
    print(f"S(end) range: [{np.nanmin(arrs['S'][:,-1]):g}, {np.nanmax(arrs['S'][:,-1]):g}]")
    print(f"phi(end) range: [{np.nanmin(arrs['phi'][:,-1]):g}, {np.nanmax(arrs['phi'][:,-1]):g}]")

    # PLOTS  (ports of figure 1 and figure 2 from esker_simpler_run.m)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ti = nt - 1
    x_km = pd["x"] / 1e3 * ad["x"]
    t_days = pd["t"] / pd["ty"] * 365.0 * td2_out

    # ---- Figure 1: profiles at final time ----
    fig1, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes

    ax0.plot(x_km, pd["Q"] * arrs["Q"][:, ti], "b", lw=2)
    ax0.set_ylabel(r"$Q$ [m$^3$/s]")

    ax1.plot(x_km, pd["z"] * ad["Z_b"], "k", lw=2)
    ax1.plot(x_km, pd["z"] * ad["Z_s"], "k", lw=2)
    ax1.plot(x_km, pd["phi"] * ad["phi_s"] / pd["rho_w"] / pd["g"], "k--", lw=2)
    ax1.plot(x_km, pd["phi"] * arrs["phi"][:, ti] / pd["rho_w"] / pd["g"], "b", lw=2)
    ax1.set_ylabel(r"Elevation $z$ [m]")

    ax2.plot(x_km, pd["phi"] * (ad["phi_s"] - arrs["phi"][:, ti]) / (pd["rho_w"] * pd["g"]), "b", lw=2)
    ax2.set_ylabel(r"$N$ [m w.e.]")

    ax3.plot(x_km, pd["S"] * arrs["S"][:, ti], "b", lw=2)
    ax3.set_ylabel(r"$S$ [m$^2$]")

    ax4.plot(x_km, pd["Qs"] * arrs["Qeq"][:, ti], "b--", lw=2)
    ax4.plot(x_km, pd["Qs"] * arrs["Qs"][:, ti], "b-", lw=2)
    ax4.set_ylabel(r"$Q_s$ [m$^3$/s]")

    ax5.plot(x_km, pd["A"] * arrs["A"][:, ti], "b", lw=2)
    ax5.set_ylabel(r"$A$ [m$^2$]")
    ax5.set_xlabel(r"Distance $x-x_m$ [km]")

    fig1.tight_layout()
    fig1.savefig(out_dir / "esker_simpler_run_fig1.pdf")
    plt.close(fig1)

    # ---- Figure 2: space-time fields ----
    fig2, axs = plt.subplots(2, 2, figsize=(11, 8))
    extent = [x_km.min(), x_km.max(), t_days.min(), t_days.max()]

    im = axs[0, 0].imshow(pd["phi"] / 1e6 * arrs["N"].T, origin="lower",
                          aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[0, 0]).set_label(r"$N$ [MPa]")
    axs[0, 0].set_xlabel(r"$x$ [km]"); axs[0, 0].set_ylabel(r"$t$ [d]")

    im = axs[0, 1].imshow(pd["S"] * arrs["S"].T, origin="lower",
                          aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[0, 1]).set_label(r"$S$ [m$^2$]")
    axs[0, 1].set_xlabel(r"$x$ [km]"); axs[0, 1].set_ylabel(r"$t$ [d]")

    im = axs[1, 0].imshow(pd["Qs"] * arrs["Qs"].T, origin="lower",
                          aspect="auto", extent=extent)
    fig2.colorbar(im, ax=axs[1, 0]).set_label(r"$Q_s$ [m$^3$/s]")
    axs[1, 0].set_xlabel(r"$x$ [km]"); axs[1, 0].set_ylabel(r"$t$ [d]")

    E = (pd["ty"] / 365.0) * (pd["S"] / pd["t"]) * pp["mu_s"] * arrs["m_s"].T
    im = axs[1, 1].imshow(E, origin="lower", aspect="auto",
                          extent=extent, cmap="bwr", vmin=-1, vmax=1)
    fig2.colorbar(im, ax=axs[1, 1]).set_label(r"$E$ [m$^2$/d]")
    axs[1, 1].set_xlabel(r"$x$ [km]"); axs[1, 1].set_ylabel(r"$t$ [d]")

    fig2.tight_layout()
    fig2.savefig(out_dir / "esker_simpler_run_fig2.pdf")
    plt.close(fig2)

    print(f"Figures saved to {out_dir}/esker_simpler_run_fig1.pdf and _fig2.pdf")


if __name__ == "__main__":
    main()
