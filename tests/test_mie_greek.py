"""scatrs.mie.phase_matrix and scatrs.mie.greek: values, shapes, derivatives and errors (the
checks shared with the other Mie functions are in test_mie.py)."""

import numpy as np
import pytest

from scatrs import InvalidInputError, mie, special


def test_phase_matrix_is_built_from_the_amplitudes():
    x, m, mu = 7.0, 1.5 + 0.01j, np.array([1.0, 0.3, -0.8, -1.0])
    s = mie.amplitudes(x, m, mu)
    q = mie.efficiencies(x, m)
    f = mie.phase_matrix(x, m, mu)
    n1, n2 = np.abs(s.s1) ** 2, np.abs(s.s2) ** 2
    p = s.s2 * np.conj(s.s1)
    c = 4 / (x**2 * q.q_sca)
    for got, expected in [
        (f.f11, (n1 + n2) / 2), (f.f12, (n2 - n1) / 2), (f.f33, p.real), (f.f34, p.imag)
    ]:  # fmt: skip
        np.testing.assert_allclose(got, c * expected, rtol=1e-14, atol=1e-14 * f.f11.max())
    u = mie.phase_matrix(x, m, mu, normalized=False)
    np.testing.assert_allclose(u.f11, f.f11 * q.q_sca, rtol=1e-14)


def test_normalized_phase_function_averages_to_one():
    nodes, weights = special.gauss_legendre(200)
    f = mie.phase_matrix([0.5, 30.0], 1.33, nodes)
    np.testing.assert_allclose(f.f11 @ weights / 2, 1.0, rtol=1e-13)


def test_greek_identities():
    x, m = np.array([0.3, 5.0, 60.0]), 1.5 + 0.01j
    g = mie.greek(x, m)
    q = mie.efficiencies(x, m)
    np.testing.assert_array_equal(g.alpha1[:, 0], 1.0)
    np.testing.assert_allclose(g.alpha1[:, 1], 3 * q.g, rtol=1e-13)
    # F11(mu = 1) = sum of alpha1, since d00_l(1) = 1.
    f = mie.phase_matrix(x, m, 1.0)
    np.testing.assert_allclose(g.alpha1.sum(axis=-1), f.f11, rtol=1e-12)
    u = mie.greek(x, m, normalized=False)
    np.testing.assert_allclose(u.alpha1[:, 0], q.q_sca, rtol=1e-15)
    np.testing.assert_allclose(u.beta1, g.beta1 * q.q_sca[:, None], atol=1e-14 * q.q_sca.max())


def test_greek_expansion_reproduces_the_phase_matrix():
    x, m, mu = 4.0, 1.33 + 1e-3j, np.array([0.9, 0.1, -0.6])
    g, f = mie.greek(x, m), mie.phase_matrix(x, m, mu)
    lmax = g.alpha1.shape[-1] - 1
    d00, d22, d2m2, d02 = (
        special.wigner_d(a, b, lmax, mu) for a, b in [(0, 0), (2, 2), (2, -2), (0, 2)]
    )
    np.testing.assert_allclose(d00 @ g.alpha1, f.f11, rtol=1e-12)
    np.testing.assert_allclose(d22 @ (g.alpha2 + g.alpha3), f.f11 + f.f33, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(d2m2 @ (g.alpha2 - g.alpha3), f.f11 - f.f33, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(d00 @ g.alpha4, f.f33, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(d02 @ g.beta1, f.f12, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(d02 @ g.beta2, f.f34, rtol=1e-12, atol=1e-13)


def test_shapes_broadcast_and_greek_is_zero_padded():
    x = np.array([[0.5], [20.0]])
    m = np.array([1.33, 1.5 + 0.01j, 2.0])
    mu = np.linspace(-1, 1, 6).reshape(2, 3)
    f = mie.phase_matrix(x, m, mu)
    assert f.f11.shape == (2, 3, 2, 3) and f.f34.dtype == np.float64
    g = mie.greek(x, m)
    n = mie.default_n_max(x)[:, 0]
    assert g.n_terms.shape == (2, 3)
    assert g.alpha1.shape == (2, 3, 2 * n.max() + 1)
    assert np.all(g.alpha1[0, :, 2 * n[0] + 1 :] == 0)
    assert mie.greek(1.0, 1.5, 3).alpha1.shape == (4,)
    # An order above 2 n_terms pads with exact zeros, as for the smaller spheres of a batch.
    small = mie.greek(1.0, 1.5)
    big = mie.greek(1.0, 1.5, 50)
    np.testing.assert_array_equal(big.alpha1[: small.alpha1.size], small.alpha1)
    assert np.all(big.alpha1[small.alpha1.size :] == 0)


def test_elements_equal_scalar_calls():
    x = np.array([0.1, 3.0, 40.0])
    m = np.array([1.33 + 1e-8j, 2.5 + 1j, 0.75])
    g = mie.greek(x, m, 30)
    f = mie.phase_matrix(x, m, [0.5, -1.0])
    for i in range(3):
        np.testing.assert_array_equal(g.beta2[i], mie.greek(x[i], m[i], 30).beta2)
        np.testing.assert_array_equal(f.f12[i], mie.phase_matrix(x[i], m[i], [0.5, -1.0]).f12)


def _fd(f, h):
    return (f(+h) - f(-h)) / (2 * h)


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize(("x", "m"), [(0.7, 1.33 + 1e-3j), (6.0, 1.5 + 0.05j), (25.0, 2.5 + 1j)])
def test_derivatives_match_finite_differences(normalized, x, m):
    mu = [0.95, 0.2, -1.0]
    n = int(mie.default_n_max(x)) + 1  # x ± h keeps the number of terms
    kw = {"n_max": n, "normalized": normalized}
    f = mie.phase_matrix(x, m, mu, derivatives=True, **kw)
    g = mie.greek(x, m, 2 * n, derivatives=True, **kw)
    steps = {
        "x": (lambda h: (x + h, m), 1e-6 * x),
        "m_real": (lambda h: (x, m + h), 1e-7),
        "m_imag": (lambda h: (x, m + 1j * h), 1e-7),
    }
    for p, (at, h) in steps.items():
        # Measured like the tolerances: the phase matrix against F11, the Greek coefficients
        # against alpha1.
        fds = {
            name: _fd(
                lambda e, at=at, name=name: getattr(mie.phase_matrix(*at(e), mu, **kw), name), h
            )
            for name in ("f11", "f12", "f33", "f34")
        }
        scale = np.abs(fds["f11"]).max()
        for name, fd in fds.items():
            np.testing.assert_allclose(getattr(f.d(p), name), fd, rtol=0, atol=1e-6 * scale)
        names = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
        fdg = {
            name: _fd(lambda e, at=at, name=name: getattr(mie.greek(*at(e), 2 * n, **kw), name), h)
            for name in names
        }
        scale = max(np.abs(fdg["alpha1"]).max(), 1e-3)
        for name, fd in fdg.items():
            np.testing.assert_allclose(getattr(g.d(p), name), fd, rtol=0, atol=1e-6 * scale)
    with pytest.raises(ValueError, match="are real"):
        f.d("m")
    with pytest.raises(ValueError, match="are real"):
        g.d("m")
    assert g.d("x").n_terms is g.n_terms


def test_batched_derivatives_equal_scalar_calls():
    x = np.array([[0.5], [12.0]])
    m = np.array([1.33, 1.5 + 0.01j, 0.75])
    mu = np.array([[0.9, 0.0], [-0.3, -1.0]])
    for normalized in (True, False):
        g = mie.greek(x, m, normalized=normalized, derivatives=True)
        f = mie.phase_matrix(x, m, mu, normalized=normalized, derivatives=True)
        width = g.alpha1.shape[-1]
        for i in range(2):
            for j in range(3):
                # The smaller spheres are zero-padded to the batch's order.
                gs = mie.greek(x[i, 0], m[j], width - 1, normalized=normalized, derivatives=True)
                fs = mie.phase_matrix(x[i, 0], m[j], mu, normalized=normalized, derivatives=True)
                for p in mie.PARAMETERS:
                    np.testing.assert_array_equal(g.d(p).beta1[i, j], gs.d(p).beta1)
                    np.testing.assert_array_equal(g.d(p).alpha3[i, j], gs.d(p).alpha3)
                    np.testing.assert_array_equal(f.d(p).f34[i, j], fs.d(p).f34)


def test_greek_errors():
    with pytest.raises(InvalidInputError, match="order must not be negative"):
        mie.greek(1.0, 1.5, -1)
    with pytest.raises(InvalidInputError, match="Greek order must be at most 2097152"):
        mie.greek(1.0, 1.5, mie.MAX_GREEK_ORDER + 1)
    with pytest.raises(TypeError):
        mie.greek(1.0, 1.5, 3.0)
    # Empty inputs do not hide a bad order.
    with pytest.raises(InvalidInputError, match="Greek order must be at most"):
        mie.greek([], 1.5, mie.MAX_GREEK_ORDER + 1)
    # A sphere whose quadrature rule would exceed the cap, named by its element.
    with pytest.raises(InvalidInputError, match=r"element 0: .* quadrature nodes, more than"):
        mie.greek([1.0, 1.0], 1.5, n_max=70_000, allow_unvalidated=True)
    with pytest.raises(TypeError, match="normalized must be a bool"):
        mie.greek(1.0, 1.5, normalized=1)
    with pytest.raises(TypeError, match="normalized must be a bool"):
        mie.phase_matrix(1.0, 1.5, 0.5, normalized=None)
    with pytest.raises(InvalidInputError, match="element 0 of mu"):
        mie.phase_matrix(1.0, 1.5, 1.5)
    # The output bounds count the orders and the derivatives.
    with pytest.raises(InvalidInputError, match=r"2 \* MAX_MIE_VALUES"):
        mie.greek(np.ones(2**12), 1.5, 2**16)
    with pytest.raises(InvalidInputError, match=r"2 \* MAX_MIE_VALUES"):
        mie.phase_matrix(np.ones(2**12), 1.5, np.zeros(2**14), derivatives=True)


def test_empty_inputs():
    g = mie.greek(np.zeros(0), 1.5)
    assert g.alpha1.shape == (0, 1) and g.n_terms.shape == (0,)
    assert mie.phase_matrix([1.0], 1.5, np.zeros(0)).f11.shape == (1, 0)
    with pytest.raises(InvalidInputError, match="element 1 of mu"):
        mie.phase_matrix([], 1.5, [0.5, np.nan])


def test_results_are_frozen_and_read_only():
    g = mie.greek(1.0, 1.5, derivatives=True)
    f = mie.phase_matrix(1.0, 1.5, [0.5], derivatives=True)
    for a in (g.alpha1, g.n_terms, g.d("x").beta1, f.f11, f.d("m_imag").f34):
        with pytest.raises(ValueError, match="read-only"):
            a[...] = 0
