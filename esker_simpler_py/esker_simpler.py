"""
Python port of esker_simpler.m

Solves time-dependent discretised channel equations for [S, phi, Qs, A].

The MATLAB original solves M(Y) * dY/dt = F(Y) with ode15s and a mass matrix.
Here we exploit the block-lower-triangular structure of M to solve the system
analytically for dY/dt and integrate with scipy's BDF method.

IJH 10 May 2026 (MATLAB original), Python port May 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, lil_matrix


# ---------- helpers ----------

def _as_col(x):
    return np.asarray(x, dtype=float).reshape(-1)


@dataclass
class Disc:
    I: int
    x: np.ndarray
    xx: np.ndarray
    dx: np.ndarray
    dxx: np.ndarray
    ddx: csc_matrix
    ddxx: csc_matrix
    avx: csc_matrix
    avxx: csc_matrix
    xin: np.ndarray
    xxin: np.ndarray
    xext: np.ndarray
    xxext: np.ndarray
    avxin: csc_matrix
    avxxin: csc_matrix


def discretize(x: np.ndarray) -> Disc:
    """Replicate MATLAB discretize() function (1-based indexing translated to 0-based)."""
    x = _as_col(x)
    I = len(x)

    xx = np.empty(I)
    xx[0] = x[0]
    xx[1:] = 0.5 * (x[:-1] + x[1:])

    dx = np.empty(I)
    dx[:-1] = xx[1:] - xx[:-1]
    dx[-1] = x[-1] - xx[-1]

    dxx = np.empty(I)
    dxx[0] = 0.0
    dxx[1:] = x[1:] - x[:-1]

    # ddx: divergence operator (derivative on nodes of quantity on edges).
    # MATLAB: sparse([1:I 1:I-1],[1:I 1+(1:I-1)], [-1/dx; 1/dx(1:I-1)], I, I)
    ddx = lil_matrix((I, I))
    for i in range(I):
        ddx[i, i] = -1.0 / dx[i]
    for i in range(I - 1):
        ddx[i, i + 1] = 1.0 / dx[i]
    ddx = ddx.tocsc()

    # ddxx: gradient operator (derivative on edges of quantity on nodes), with
    # first edge duplicated from second.
    # MATLAB final assignment uses dxx([2 2:I]) for diagonal/off-diagonal.
    ddxx = lil_matrix((I, I))
    # Row 0 (first edge) = row 1 (duplicated)
    ddxx[0, 0] = -1.0 / dxx[1]
    ddxx[0, 1] = 1.0 / dxx[1]
    for i in range(1, I):
        ddxx[i, i - 1] = -1.0 / dxx[i]
        ddxx[i, i] = 1.0 / dxx[i]
    ddxx = ddxx.tocsc()

    # avx: averaging on nodes (from edges)
    # MATLAB: sparse([1:I 1:I-1],[1:I 1+(1:I-1)], [0.5*ones(I-1,1); 1; 0.5*ones(I-1,1)], I, I)
    # Diagonal: 0.5 for i=1..I-1, 1 for i=I (last). Off-diagonal: 0.5.
    avx = lil_matrix((I, I))
    for i in range(I):
        avx[i, i] = 0.5 if i < I - 1 else 1.0
    for i in range(I - 1):
        avx[i, i + 1] = 0.5
    avx = avx.tocsc()

    # avxx: averaging on edges (from nodes)
    # MATLAB: sparse([2:I 1:I],[(2:I)-1 (1:I)], [0.5*ones(I-1,1); 1; 0.5*ones(I-1,1)], I, I)
    # Diagonal: 1 for first (i=1), 0.5 for i=2..I. Lower: 0.5.
    avxx = lil_matrix((I, I))
    for i in range(I):
        avxx[i, i] = 1.0 if i == 0 else 0.5
    for i in range(1, I):
        avxx[i, i - 1] = 0.5
    avxx = avxx.tocsc()

    # Indices (0-based): xext = last node, xxext = first edge.
    xext = np.array([I - 1], dtype=int)
    xxext = np.array([0], dtype=int)
    xin = np.array([i for i in range(I) if i not in xext], dtype=int)
    xxin = np.array([i for i in range(I) if i not in xxext], dtype=int)

    # avxin: avx restricted so columns at xxext are zeroed, then re-normalized
    ones_xxin = np.zeros(I)
    ones_xxin[xxin] = 1.0
    tmp = avx.dot(ones_xxin)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(tmp != 0, 1.0 / tmp, 0.0)
    D = csc_matrix((inv, (np.arange(I), np.arange(I))), shape=(I, I))
    avxin = (D @ avx).tolil()
    for j in xxext:
        avxin[:, j] = 0
    avxin = avxin.tocsc()

    ones_xin = np.zeros(I)
    ones_xin[xin] = 1.0
    tmp = avxx.dot(ones_xin)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(tmp != 0, 1.0 / tmp, 0.0)
    D = csc_matrix((inv, (np.arange(I), np.arange(I))), shape=(I, I))
    avxxin = (D @ avxx).tolil()
    for j in xext:
        avxxin[:, j] = 0
    avxxin = avxxin.tocsc()

    return Disc(I=I, x=x, xx=xx, dx=dx, dxx=dxx, ddx=ddx, ddxx=ddxx,
                avx=avx, avxx=avxx, xin=xin, xxin=xxin, xext=xext, xxext=xxext,
                avxin=avxin, avxxin=avxxin)


# ---------- unpack ----------

def unpack_full(Y, aa, dd: Disc, pp):
    """Replicate MATLAB unpack_full. Y has shape (4*Iin,) or (4*Iin, K)."""
    Y = np.atleast_2d(Y)
    if Y.shape[0] != 4 * len(dd.xin):
        Y = Y.T
    K = Y.shape[1]
    I = dd.I
    xin = dd.xin
    xxin = dd.xxin
    xext = dd.xext
    xxext = dd.xxext
    Iin = len(xin)

    S = np.full((I, K), np.nan)
    phi = np.full((I, K), np.nan)
    Qs = np.full((I, K), np.nan)
    A = np.full((I, K), np.nan)
    phi[xext, :] = pp["phi_m"]
    Qs[xxext, :] = pp["Qs_0"]

    S[xin, :] = Y[0:Iin, :]
    phi[xin, :] = Y[Iin:2 * Iin, :]
    Qs[xxin, :] = Y[2 * Iin:3 * Iin, :]
    A[xin, :] = Y[3 * Iin:4 * Iin, :]

    Q = np.full((I, K), np.nan)
    Q[xxext, :] = pp["Q_0"]
    Sav = dd.avxxin @ S
    dphi = dd.ddxx @ phi
    eps = np.finfo(float).eps
    absdphi = np.maximum(np.abs(dphi), eps)
    tmp = -(Sav ** pp["alpha"]) * absdphi ** (-0.5) * dphi
    Q[xxin, :] = tmp[xxin, :]

    m = dd.avxin @ (-Q * dphi)

    N = aa["phi_s"][:, None] - phi

    U = Q / Sav
    Qeq = Sav ** pp["gamma"] * np.maximum(U ** 2 - pp["tau_s"], 0.0) ** 1.5

    m_s = np.zeros((I, K))
    tmp = -(dd.avxin @ (Qs - Qeq)) / pp["nu"]
    A_min = pp.get("A_min", 0.0)
    ind = (tmp < 0) | (A > A_min)
    m_s[ind] = tmp[ind]

    return S, phi, Q, A, N, Qeq, Qs, m, m_s


# ---------- right-hand side and mass ----------

def fun_F(t, Y, aa, dd: Disc, pp):
    """Right-hand side of equations, returns block-stacked F restricted to interior."""
    S, phi, Q, A, N, Qeq, Qs, m, m_s = unpack_full(Y, aa, dd, pp)
    K = S.shape[1]
    ext = np.ones((1, K))

    Q_in = np.zeros(dd.I)
    Q_in[aa["xi_m"]] = aa["Q_m"](t)
    M_in = aa["M_in"](t)
    if np.ndim(M_in) == 0:
        M_in = M_in * np.ones(dd.I)
    Qs_in = np.zeros(dd.I)
    Qs_in[aa["xi_m"]] = aa["Qs_m"](t)
    Ms_in = aa["Ms_in"](t)
    if np.ndim(Ms_in) == 0:
        Ms_in = Ms_in * np.ones(dd.I)

    F1 = (pp["upsilon"] * np.maximum(0.0, 1.0 - S / pp["upsilon2"])
          + pp["mu_i"] * m + pp["mu_s"] * m_s
          - pp["omega_i"] * S ** (2 * pp["gamma"]) * np.abs(N) ** (pp["n"] - 1) * N
          - pp["omega_s"] * S ** (2 * pp["gamma"]) * np.abs(N) ** (pp["n_sed"] - 1) * N)

    F2 = -(dd.ddx @ Q) + pp["epsilon"] / pp["r"] * pp["mu_i"] * m \
         + (Q_in / dd.dx)[:, None] @ ext + M_in[:, None] @ ext

    F3 = -(dd.ddx @ Qs) + m_s + (Qs_in / dd.dx)[:, None] @ ext + Ms_in[:, None] @ ext

    F4 = -pp["mu_s"] * m_s

    xin = dd.xin
    F = np.vstack([F1[xin, :], F2[xin, :], F3[xin, :], F4[xin, :]])
    return F


def rhs(t, y, aa, dd: Disc, pp):
    """Compute dy/dt by exploiting block-lower-triangular structure of M.

    M structure (Iin x Iin blocks):
        [ I                                  0    0    0 ]
        [ epsilon*I   sigma+eps38*S+beta*S_in/dx  0    0 ]
        [ 0           0                      eps53*I 0 ]
        [ 0           0                      0    I ]
    """
    y_col = y.reshape(-1, 1)
    F = fun_F(t, y_col, aa, dd, pp)  # shape (4*Iin, 1)
    F = F.ravel()
    Iin = len(dd.xin)
    F1 = F[0:Iin]; F2 = F[Iin:2*Iin]; F3 = F[2*Iin:3*Iin]; F4 = F[3*Iin:4*Iin]

    # dS/dt = F1
    dS = F1

    # dphi/dt: diagonal coefficient depends on S
    S_in_vec = np.zeros(dd.I)
    S_in_vec[aa["xi_m"]] = aa["S_m"]
    S = y[0:Iin]  # interior S
    diag_phi = pp["sigma"] + pp["eps38"] * S + pp["beta"] * S_in_vec[dd.xin] / dd.dx[dd.xin]
    dphi = (F2 - pp["epsilon"] * dS) / diag_phi

    # dQs/dt
    dQs = F3 / pp["eps53"]

    # dA/dt
    dA = F4

    return np.concatenate([dS, dphi, dQs, dA])


# ---------- initial condition ----------

def initial(t, aa, dd: Disc, pp):
    Smin = 1e-3
    Q_in = np.zeros(dd.I)
    Q_in[aa["xi_m"]] = aa["Q_m"](t)
    M_in = aa["M_in"](t)
    if np.ndim(M_in) == 0:
        M_in = M_in * np.ones(dd.I)
    Qs_in = np.zeros(dd.I)
    Qs_in[aa["xi_m"]] = aa["Qs_m"](t)
    Ms_in = aa["Ms_in"](t)
    if np.ndim(Ms_in) == 0:
        Ms_in = Ms_in * np.ones(dd.I)
    Q = np.cumsum(Q_in + M_in * dd.dx)
    phi = aa["phi_s"].copy()
    dphi = dd.ddxx @ phi
    eps_ = np.finfo(float).eps
    S = np.abs(-Q / (np.maximum(np.abs(dphi), eps_) ** -0.5 * dphi)) ** (1.0 / pp["alpha"])
    S = np.maximum(S, Smin)
    Qs = np.cumsum(Qs_in + Ms_in * dd.dx)
    A = np.zeros(dd.I)
    return S, phi, Qs, A


def pack(S, phi, Qs, A, dd: Disc):
    return np.concatenate([S[dd.xin], phi[dd.xin], Qs[dd.xxin], A[dd.xin]])


# ---------- top-level solver ----------

def non_dimensionalise(ad, pd):
    """Port of MATLAB non_dimensionalise().

    Fills in default scales in `pd` (mutated/returned) and returns
    (aa, pp, pd) — dimensionless geometry/inputs, dimensionless params, and
    the updated scale dict.
    """
    pd = dict(pd)
    pd.setdefault("Q", 10.0)
    pd.setdefault("x", 5000.0)
    pd.setdefault("z", 500.0)
    pd.setdefault("phi", pd["rho_i"] * pd["g"] * pd["z"])
    pd.setdefault("psi", pd["phi"] / pd["x"])
    pd.setdefault("S", (pd["Q"] / pd["k_c"] / pd["psi"] ** 0.5) ** (1.0 / pd["alpha"]))
    pd.setdefault("m", pd["Q"] * pd["psi"] / pd["rho_w"] / pd["L"])
    pd.setdefault("U", pd["Q"] / pd["S"])
    pd.setdefault("Qs", pd["k_s"] * pd["S"] ** pd["gamma"] *
                  (1.0 / 8.0 * pd["f"] * pd["rho_w"] * pd["U"] ** 2 /
                   ((pd["rho_s"] - pd["rho_w"]) * pd["g"] * pd["D_50"])) ** 1.5)
    pd.setdefault("m_s", pd["Qs"] / pd["x"])
    pd.setdefault("t", 24 * 60 * 60.0)
    pd.setdefault("A", pd["S"])

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
    pp["tau_s"] = (pd["tau_s"] * ((pd["rho_s"] - pd["rho_w"]) * pd["g"] * pd["D_50"]) /
                   (1.0 / 8.0 * pd["f"] * pd["rho_w"] * pd["U"] ** 2))
    pp["Z_0"] = pd["Z_0"] / pd["z"]
    if "A_min" in pd:
        pp["A_min"] = pd["A_min"] / pd["A"]

    aa = {}
    aa["x"] = ad["x"] / pd["x"]
    aa["Z_b"] = ad["Z_b"] / pd["z"]
    aa["Z_s"] = ad["Z_s"] / pd["z"]
    if "xi_m" in ad:
        aa["xi_m"] = ad["xi_m"]
    if "S_m" in ad:
        aa["S_m"] = np.atleast_1d(np.asarray(ad["S_m"], dtype=float)) / pd["S"]
    if "Q_m" in ad:
        Q_m = ad["Q_m"]
        aa["Q_m"] = lambda t, _f=Q_m, _t=pd["t"], _Q=pd["Q"]: np.asarray(_f(_t * t)) / _Q
    if "M_in" in ad:
        M_in = ad["M_in"]
        aa["M_in"] = lambda t, _f=M_in, _t=pd["t"], _Q=pd["Q"], _x=pd["x"]: np.asarray(_f(_t * t)) / (_Q / _x)
    if "Qs_m" in ad:
        Qs_m = ad["Qs_m"]
        aa["Qs_m"] = lambda t, _f=Qs_m, _t=pd["t"], _Qs=pd["Qs"]: np.asarray(_f(_t * t)) / _Qs
    if "Ms_in" in ad:
        Ms_in = ad["Ms_in"]
        aa["Ms_in"] = lambda t, _f=Ms_in, _t=pd["t"], _Qs=pd["Qs"], _x=pd["x"]: np.asarray(_f(_t * t)) / (_Qs / _x)

    return aa, pp, pd


def _unscale_aa(aa, pd):
    """Port of MATLAB unscale_aa."""
    ad = {}
    ad["x"] = pd["x"] * aa["x"]
    ad["Z_b"] = pd["z"] * aa["Z_b"]
    ad["Z_s"] = pd["z"] * aa["Z_s"]
    if "xi_m" in aa:
        ad["xi_m"] = aa["xi_m"]
    if "Q_m" in aa:
        Q_m = aa["Q_m"]
        ad["Q_m"] = lambda t, _f=Q_m, _t=pd["t"], _Q=pd["Q"]: _Q * np.asarray(_f(t / _t))
    if "M_in" in aa:
        M_in = aa["M_in"]
        ad["M_in"] = lambda t, _f=M_in, _t=pd["t"], _Q=pd["Q"], _x=pd["x"]: (_Q / _x) * np.asarray(_f(t / _t))
    if "Qs_m" in aa:
        Qs_m = aa["Qs_m"]
        ad["Qs_m"] = lambda t, _f=Qs_m, _t=pd["t"], _Qs=pd["Qs"]: _Qs * np.asarray(_f(t / _t))
    if "Ms_in" in aa:
        Ms_in = aa["Ms_in"]
        ad["Ms_in"] = lambda t, _f=Ms_in, _t=pd["t"], _Qs=pd["Qs"], _x=pd["x"]: (_Qs / _x) * np.asarray(_f(t / _t))
    ad["phi_s"] = pd["phi"] * aa["phi_s"]
    ad["phi_b"] = pd["phi"] * aa["phi_b"]
    ad["psi_s"] = pd["phi"] / pd["x"] * aa["psi_s"]
    return ad


def _unscale_uu(uu, pd):
    out = []
    for u in uu:
        out.append({
            "S": pd["S"] * u["S"],
            "phi": pd["phi"] * u["phi"],
            "Q": pd["Q"] * u["Q"],
            "A": pd["A"] * u["A"],
            "N": pd["phi"] * u["N"],
            "Qeq": pd["Qs"] * u["Qeq"],
            "Qs": pd["Qs"] * u["Qs"],
            "m": pd["m"] * u["m"],
            "m_s": pd["m_s"] * u["m_s"],
        })
    return out


def esker_simpler(td, aa, pp, ud_init=None, oo=None):
    """Solve the system over timespan td.

    If oo['nondiminput'] is True (default), `aa` and `pp` are taken as
    already-dimensionless inputs (matching the MATLAB no-arg path).
    Otherwise `aa` is `ad` (dimensional geometry/inputs) and `pp` is `pd`
    (dimensional parameters / scales); the routine non-dimensionalises,
    solves, and re-dimensionalises outputs internally.
    """
    if oo is None:
        oo = {"nondiminput": True}

    pd_scales = None
    if not oo.get("nondiminput", True):
        ad_in = aa
        pd_scales = pp
        aa, pp, pd_scales = non_dimensionalise(ad_in, pd_scales)
        td = np.asarray(td, dtype=float) / pd_scales["t"]
        if ud_init is not None:
            ud_init = {
                "S": np.asarray(ud_init["S"]) / pd_scales["S"],
                "phi": np.asarray(ud_init["phi"]) / pd_scales["phi"],
                "Qs": np.asarray(ud_init["Qs"]) / pd_scales["Qs"],
                "A": np.asarray(ud_init["A"]) / pd_scales["A"],
            }

    td = np.asarray(td, dtype=float)
    aa = dict(aa)  # shallow copy
    pp = dict(pp)

    dd = discretize(aa["x"])

    # Fill missing fields with defaults
    aa.setdefault("xi_m", np.array([], dtype=int))
    aa["xi_m"] = np.atleast_1d(np.asarray(aa["xi_m"], dtype=int))
    aa.setdefault("S_m", np.zeros_like(aa["xi_m"], dtype=float))
    aa["S_m"] = np.atleast_1d(np.asarray(aa["S_m"], dtype=float))
    aa.setdefault("Q_m", lambda t: np.zeros_like(aa["xi_m"], dtype=float))
    aa.setdefault("M_in", lambda t: np.zeros(dd.I))
    aa.setdefault("Qs_m", lambda t: np.zeros_like(aa["xi_m"], dtype=float))
    aa.setdefault("Ms_in", lambda t: np.zeros(dd.I))

    pp.setdefault("phi_m", max(pp["r"] * aa["Z_b"][-1], pp.get("Z_0", 0.0)))
    pp.setdefault("Q_0", 0.0)
    pp.setdefault("Qs_0", 0.0)
    pp.setdefault("eps38", 1e-4)
    pp.setdefault("eps53", 1e-4)
    pp.setdefault("A_min", 0.0)

    # Derived geometry
    aa["phi_s"] = aa["Z_s"] + (pp["r"] - 1.0) * aa["Z_b"]
    aa["phi_b"] = pp["r"] * aa["Z_b"]
    aa["psi_s"] = -(dd.ddxx @ aa["phi_s"])

    if ud_init is None:
        S, phi, Qs, A = initial(td[0], aa, dd, pp)
    else:
        S = ud_init["S"].copy()
        phi = ud_init["phi"].copy()
        Qs = ud_init["Qs"].copy()
        A = ud_init["A"].copy()

    Y0 = pack(S, phi, Qs, A, dd)

    print("Solving ...")
    rhs_fn = lambda t, y: rhs(t, y, aa, dd, pp)
    sol = solve_ivp(rhs_fn, (td[0], td[-1]), Y0, method="BDF",
                    t_eval=td, rtol=1e-6, atol=1e-4)
    if not sol.success:
        # Fall back to Radau for genuinely stiffer regimes (e.g. nonzero
        # epsilon in dimensional runs) where BDF's step controller stalls.
        print(f"  BDF failed ({sol.message}); retrying with Radau")
        sol = solve_ivp(rhs_fn, (td[0], td[-1]), Y0, method="Radau",
                        t_eval=td, rtol=1e-6, atol=1e-6,
                        max_step=(td[-1] - td[0]) / 200.0)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # Extract variables at each output time
    uu: List[dict] = []
    for ti in range(sol.t.size):
        Yi = sol.y[:, ti:ti + 1]
        S, phi, Q, A, N, Qeq, Qs, m, m_s = unpack_full(Yi, aa, dd, pp)
        uu.append({
            "S": S.ravel(), "phi": phi.ravel(), "Q": Q.ravel(),
            "A": A.ravel(), "N": N.ravel(), "Qeq": Qeq.ravel(),
            "Qs": Qs.ravel(), "m": m.ravel(), "m_s": m_s.ravel(),
        })

    print("Done")
    if pd_scales is not None:
        td_out = pd_scales["t"] * sol.t
        ad_out = _unscale_aa(aa, pd_scales)
        ud_out = _unscale_uu(uu, pd_scales)
        return td_out, ad_out, ud_out
    return sol.t, aa, uu
