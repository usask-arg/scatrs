"""scatrs.layered: shapes, names, the identities with scatrs.mie, derivatives and errors."""

import numpy as np
import pytest

from scatrs import InvalidInputError, OutsideValidatedDomainError, layered, mie

X = np.array([[0.7, 1.3], [3.0, 8.0], [10.0, 12.0]])
M = np.array([1.95 + 0.79j, 1.43 + 1e-8j])
MU = np.array([-1.0, -0.3, 0.4, 1.0])


def test_shapes_broadcast_over_the_leading_axes():
    x = X.reshape(3, 1, 2)
    m = np.stack([M, [1.5 + 0.01j, 1.33]])  # (2, 2)
    q = layered.efficiencies(x, m)
    assert q.q_ext.shape == (3, 2)
    s = layered.amplitudes(x, m, MU.reshape(2, 2))
    assert s.s1.shape == (3, 2, 2, 2)
    c = layered.coefficients(x, m)
    assert c.a.shape == (3, 2, c.n_terms.max()) and c.n_terms.shape == (3, 2)
    g = layered.greek(x, m, 6)
    assert g.alpha1.shape == (3, 2, 7)
    f = layered.phase_matrix(x, m, MU)
    assert f.f11.shape == (3, 2, 4)


@pytest.mark.parametrize(("x", "m"), [(0.3, 1.33), (5.0, 1.5 + 0.01j), (40.0, 2.5 + 1j)])
def test_one_layer_is_the_sphere_bitwise(x, m):
    def same(u, v):
        return np.array_equal(u, v)

    c, c0 = (
        layered.coefficients([x], [m], derivatives=True),
        mie.coefficients(x, m, derivatives=True),
    )
    assert same(c.a, c0.a) and same(c.b, c0.b)
    assert same(c.d("layer0.x").a, c0.d("x").a) and same(c.d("layer0.m").b, c0.d("m").b)
    q, q0 = (
        layered.efficiencies([x], [m], derivatives=True),
        mie.efficiencies(x, m, derivatives=True),
    )
    assert same(q.q_abs, q0.q_abs) and same(q.d("layer0.m_imag").q_abs, q0.d("m_imag").q_abs)
    s, s0 = layered.amplitudes([x], [m], MU), mie.amplitudes(x, m, MU)
    assert same(s.s1, s0.s1) and same(s.s2, s0.s2)
    g, g0 = layered.greek([x], [m]), mie.greek(x, m)
    assert same(g.alpha1, g0.alpha1) and same(g.beta2, g0.beta2)


def test_identical_layers_are_the_sphere_bitwise():
    q = layered.efficiencies([2.0, 5.0], [1.5 + 0.01j] * 2)
    q0 = mie.efficiencies(5.0, 1.5 + 0.01j)
    assert np.array_equal(q.q_ext, q0.q_ext) and np.array_equal(q.g, q0.g)


def test_derivatives_match_central_differences():
    x, m = np.array([1.0, 2.5]), np.array([1.95 + 0.79j, 1.43 + 0.01j])
    q = layered.efficiencies(x, m, derivatives=True)
    h = 1e-6
    for layer in range(2):
        for part in ("x", "m_real", "m_imag"):
            xp, xm, mp, mm = x.copy(), x.copy(), m.copy(), m.copy()
            if part == "x":
                xp[layer] += h
                xm[layer] -= h
            else:
                step = h if part == "m_real" else 1j * h
                mp[layer] += step
                mm[layer] -= step
            fd = (layered.efficiencies(xp, mp).q_ext - layered.efficiencies(xm, mm).q_ext) / (2 * h)
            got = q.d(f"layer{layer}.{part}").q_ext
            assert got == pytest.approx(fd, rel=1e-7, abs=1e-9), (layer, part)


def test_complex_derivatives_and_selected_names():
    c = layered.coefficients(X, M, derivatives=["layer1.m_real"])
    assert c.params == ("layer1.m_real",)
    np.testing.assert_array_equal(c.d("layer1.m").a, c.d("layer1.m_real").a)
    with pytest.raises(ValueError, match="no derivative"):
        c.d("layer0.x")
    s = layered.amplitudes(X, M, MU, derivatives=True)
    np.testing.assert_array_equal(s.d("layer0.m_imag").s1, 1j * s.d("layer0.m").s1)
    q = layered.efficiencies(X, M, derivatives=True)
    with pytest.raises(ValueError, match="real"):
        q.d("layer0.m")
    assert layered.parameters(2) == (
        "layer0.x",
        "layer0.m_real",
        "layer0.m_imag",
        "layer1.x",
        "layer1.m_real",
        "layer1.m_imag",
    )


def test_results_are_read_only():
    q = layered.efficiencies(X, M, derivatives=True)
    with pytest.raises(ValueError):
        q.q_ext[0] = 1.0
    with pytest.raises(ValueError):
        q.d("layer0.x").q_ext[0] = 1.0


def test_errors():
    with pytest.raises(ValueError, match="trailing axis"):
        layered.efficiencies(1.0, 1.5)
    with pytest.raises(ValueError, match="one value per layer"):
        layered.efficiencies([1.0, 2.0], [1.5, 1.33, 1.2])
    with pytest.raises(ValueError, match="at least one layer"):
        layered.efficiencies(np.zeros((2, 0)), np.zeros((2, 0)))
    with pytest.raises(ValueError, match="unknown parameter"):
        layered.efficiencies(X, M, derivatives=["x"])
    with pytest.raises(TypeError):
        layered.efficiencies(X, M, derivatives="layer0.x")
    with pytest.raises(InvalidInputError, match="nondecreasing"):
        layered.efficiencies([2.0, 1.0], [1.5, 1.33])
    with pytest.raises(InvalidInputError, match="element 1"):
        layered.efficiencies([[1.0, 2.0], [1.0, 2.0]], [[1.5, 1.33], [1.5, -1.0]])
    with pytest.raises(OutsideValidatedDomainError):
        layered.efficiencies([1.0, 2e4], [1.5, 1.33])
    assert layered.efficiencies([1.0, 2e4], [1.5, 1.33], allow_unvalidated=True).q_ext > 0
    # Derivatives are validated up to an outer size parameter of 100.
    with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
        layered.efficiencies([50.0, 200.0], [1.5, 1.33], derivatives=True)
    assert layered.efficiencies([50.0, 200.0], [1.5, 1.33]).q_ext > 0


def test_default_n_max_is_the_spheres_at_the_outer_size():
    np.testing.assert_array_equal(layered.default_n_max(X), mie.default_n_max(X[:, -1]))
