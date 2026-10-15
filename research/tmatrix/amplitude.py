"""Fixed-orientation amplitudes from the T-matrix, and the brute-force orientation average of the
phase matrix: the reference for the analytic random-orientation formulas.

Conventions (those of ebcm.py, without Mishchenko's (−1)^m, which cancels since T is diagonal in
m): M_mn = γₙ zₙ(kr) C_mn e^{imφ}, N_mn = γₙ [n(n+1) zₙ/(kr) d_mn r̂ + ([kr zₙ]'/(kr)) B_mn] e^{imφ},
C_mn = i π_mn θ̂ − τ_mn φ̂, B_mn = τ_mn θ̂ + i π_mn φ̂, γₙ = √((2n+1)/(4π n(n+1))).
  incident E₀ e^{ik n̂·r} = Σ a_mn RgM_mn + b_mn RgN_mn, a_mn = 4π iⁿ γₙ C*_mn(θᵢ)·E₀ e^{−imφᵢ},
  b_mn = 4π i^{n−1} γₙ B*_mn(θᵢ)·E₀ e^{−imφᵢ};
  scattered Σ p_mn M_mn + q_mn N_mn, (p, q) = T (a, b) (T¹¹ M←M, T¹² M←N, T²¹ N←M, T²² N←N);
  far field E_sca → e^{ikr}/(kr) Σ γₙ e^{imφ} [(−i)^{n+1} p_mn C_mn + (−i)ⁿ q_mn B_mn], so the
  amplitude matrix (k = 1) is S E₀ = that sum, in the (θ̂, φ̂) basis of each direction.
"""

import math

import mpmath as mp
import numpy as np
from ebcm import qmatrices


def angular_signed(m, nmax, theta):
    """d_mn, π_mn, τ_mn (normalized, Mishchenko's sign) for any integer m, n = 0..nmax, float."""
    am = abs(m)
    ct, st = math.cos(theta), math.sin(theta)
    d = np.zeros(nmax + 1)
    if am <= nmax:
        a = 1.0
        for k in range(1, am + 1):
            a *= math.sqrt((2 * k - 1) / (2 * k))
        d[am] = a * st**am
        if am + 1 <= nmax:
            d[am + 1] = math.sqrt(2 * am + 1) * ct * d[am]
        for n in range(am + 1, nmax):
            d[n + 1] = (
                (2 * n + 1) * ct * d[n] - math.sqrt(n * n - am * am) * d[n - 1]
            ) / math.sqrt((n + 1) ** 2 - am * am)
    tau = np.zeros(nmax + 1)
    for n in range(max(am, 1), nmax + 1):
        prev = d[n - 1] if n - 1 >= am else 0.0
        tau[n] = (n * ct * d[n] - math.sqrt(n * n - am * am) * prev) / st
    if m < 0:
        sign = (-1) ** am
        d, tau = sign * d, sign * tau
    pi_ = np.array([m * d[n] / st for n in range(nmax + 1)])
    return d, pi_, tau


def gamma(n):
    return math.sqrt((2 * n + 1) / (4 * math.pi * n * (n + 1)))


def t_modes(x, eps, s, nmax, ntheta, dps=30):
    """T of every mode m = −N..N (dict m → (orders, 2K×2K numpy array), rows/columns M then N)."""
    out = {}
    with mp.workdps(dps):
        for m in range(-nmax, nmax + 1):
            q, p = qmatrices(x, eps, mp.mpc(s), m, nmax, ntheta)
            t = -(p * mp.inverse(q))
            arr = np.array([[complex(t[i, j]) for j in range(t.cols)] for i in range(t.rows)])
            out[m] = (list(range(max(abs(m), 1), nmax + 1)), arr)
    return out


def vsh(m, n, theta):
    """C_mn and B_mn at θ as (θ̂, φ̂) component pairs (without e^{imφ})."""
    nmax = n
    _, pi_, tau = angular_signed(m, nmax, theta)
    c = np.array([1j * pi_[n], -tau[n]])
    b = np.array([tau[n], 1j * pi_[n]])
    return c, b


def amplitude(tm, th_i, ph_i, th_s, ph_s, e0):
    """S E₀ in the (θ̂, φ̂) basis at (θ_s, φ_s), for E₀ given in the (θ̂, φ̂) basis at (θ_i, φ_i)."""
    out = np.zeros(2, dtype=complex)
    for m, (orders, t) in tm.items():
        k = len(orders)
        a = np.zeros(k, dtype=complex)
        b = np.zeros(k, dtype=complex)
        for i, n in enumerate(orders):
            c_i, b_i = vsh(m, n, th_i)
            ph = np.exp(-1j * m * ph_i)
            a[i] = 4 * math.pi * 1j**n * gamma(n) * np.dot(np.conj(c_i), e0) * ph
            b[i] = 4 * math.pi * 1j ** (n - 1) * gamma(n) * np.dot(np.conj(b_i), e0) * ph
        pq = t @ np.concatenate([a, b])
        p, q = pq[:k], pq[k:]
        for i, n in enumerate(orders):
            c_s, b_s = vsh(m, n, th_s)
            ph = np.exp(1j * m * ph_s)
            out += gamma(n) * ph * ((-1j) ** (n + 1) * p[i] * c_s + (-1j) ** n * q[i] * b_s)
    return out


def bases(theta, phi):
    """θ̂, φ̂ unit vectors (Cartesian) at (θ, φ)."""
    th = np.array(
        [math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)]
    )
    ph = np.array([-math.sin(phi), math.cos(phi), 0.0])
    return th, ph


def direction(theta, phi):
    return np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )


def to_angles(v):
    v = v / np.linalg.norm(v)
    theta = math.acos(max(-1.0, min(1.0, v[2])))
    phi = math.atan2(v[1], v[0])
    return theta, phi


def amplitude_2x2(tm, n_i, e_par_i, e_perp_i, n_s, e_par_s, e_perp_s):
    """The 2×2 amplitude matrix in given polarization bases: rows (par, perp) scattered, columns
    (par, perp) incident; all vectors Cartesian in the particle's frame."""
    ti, pi_ = to_angles(n_i)
    ts, ps = to_angles(n_s)
    thi, phi_i = bases(ti, pi_)
    ths, phs = bases(ts, ps)
    s = np.zeros((2, 2), dtype=complex)
    for col, e in enumerate((e_par_i, e_perp_i)):
        e0 = np.array([np.dot(e, thi), np.dot(e, phi_i)])
        f = amplitude(tm, ti, pi_, ts, ps, e0)
        field = f[0] * ths + f[1] * phs
        s[0, col] = np.dot(field, e_par_s)
        s[1, col] = np.dot(field, e_perp_s)
    return s


def mueller(s):
    """Mueller matrix from the amplitude matrix [[S2, S3], [S4, S1]] (Bohren & Huffman's
    ordering: par/par = S2, perp/perp = S1)."""
    s2, s3, s4, s1 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    a = lambda v: abs(v) ** 2  # noqa: E731
    z = np.zeros((4, 4))
    z[0, 0] = 0.5 * (a(s1) + a(s2) + a(s3) + a(s4))
    z[0, 1] = 0.5 * (a(s2) - a(s1) + a(s4) - a(s3))
    z[1, 0] = 0.5 * (a(s2) - a(s1) - a(s4) + a(s3))
    z[1, 1] = 0.5 * (a(s2) + a(s1) - a(s4) - a(s3))
    z[2, 2] = (s1 * np.conj(s2) + s3 * np.conj(s4)).real
    z[2, 3] = (s2 * np.conj(s1) + s4 * np.conj(s3)).imag
    z[3, 2] = (s1 * np.conj(s2) - s3 * np.conj(s4)).imag
    z[3, 3] = (s1 * np.conj(s2) - s3 * np.conj(s4)).real
    return z


def scattering_geometry(Theta, rot):
    """Incident along R ẑ, scattering at Θ in the R-rotated xz plane; parallel and perpendicular
    unit vectors of each wave (perpendicular = the plane's normal R ŷ)."""
    z = rot @ np.array([0.0, 0.0, 1.0])
    yv = rot @ np.array([0.0, 1.0, 0.0])
    xv = rot @ np.array([1.0, 0.0, 0.0])
    n_s = math.cos(Theta) * z + math.sin(Theta) * xv
    par_i = xv
    par_s = math.cos(Theta) * xv - math.sin(Theta) * z
    perp = yv
    return z, par_i, perp, n_s, par_s, perp


def rotation(beta, gamma_):
    """R = Rz(α=0) Ry(β) Rz(γ)."""
    cb, sb, cg, sg = math.cos(beta), math.sin(beta), math.cos(gamma_), math.sin(gamma_)
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return ry @ rz


def averaged_mueller(tm, thetas, nbeta, ngamma):
    """⟨Z(Θ)⟩ over orientations: β by Gauss–Legendre in cos β, γ uniform (the average about the
    symmetry axis is trivial for an axisymmetric particle)."""
    xs, ws = np.polynomial.legendre.leggauss(nbeta)
    out = np.zeros((len(thetas), 4, 4))
    for cb, wb in zip(xs, ws, strict=True):
        beta = math.acos(cb)
        for ig in range(ngamma):
            g = 2 * math.pi * ig / ngamma
            rot = rotation(beta, g)
            for it, Theta in enumerate(thetas):
                n_i, par_i, perp_i, n_s, par_s, perp_s = scattering_geometry(Theta, rot)
                s = amplitude_2x2(tm, n_i, par_i, perp_i, n_s, par_s, perp_s)
                out[it] += wb / 2 / ngamma * mueller(s)
    return out
