"""scatrs.tmatrix: shapes, the sphere limit, derivative names, the TMatrix object and errors."""

import numpy as np
import pytest

from scatrs import (
    InvalidInputError,
    OutsideValidatedDomainError,
    mie,
    tmatrix,
)

MU = np.array([-1.0, -0.3, 0.4, 1.0])


def test_shapes_broadcast():
    x = np.array([1.0, 2.5]).reshape(2, 1)
    eps = np.array([0.5, 1.0, 2.0])
    q = tmatrix.efficiencies(x, 1.5 + 0.01j, eps)
    assert q.q_ext.shape == (2, 3) and q.n_terms.shape == (2, 3) and q.error.shape == (2, 3)
    g = tmatrix.greek(x, 1.5 + 0.01j, eps, 6)
    assert g.alpha1.shape == (2, 3, 7) and g.n_terms.shape == (2, 3)
    f = tmatrix.phase_matrix(x, 1.5 + 0.01j, eps, MU.reshape(2, 2))
    assert f.f22.shape == (2, 3, 2, 2) and f.f44.shape == (2, 3, 2, 2)


def test_an_aspect_ratio_of_one_is_the_sphere():
    x, m = 3.0, 1.5 + 0.01j
    q, q0 = tmatrix.efficiencies(x, m, 1.0, rtol=1e-10), mie.efficiencies(x, m)
    for name in ("q_ext", "q_sca", "q_abs", "q_back", "g"):
        assert getattr(q, name) == pytest.approx(getattr(q0, name), rel=1e-9), name
    f, f0 = tmatrix.phase_matrix(x, m, 1.0, MU, rtol=1e-10), mie.phase_matrix(x, m, MU)
    scale = f0.f11.max()
    for a, b in [(f.f11, f0.f11), (f.f22, f0.f11), (f.f33, f0.f33), (f.f44, f0.f33)]:
        np.testing.assert_allclose(a, b, atol=1e-9 * scale)


def test_oblate_and_prolate_differ():
    q = tmatrix.efficiencies(3.0, 1.5, [0.5, 2.0])
    assert abs(q.q_ext[0] - q.q_ext[1]) > 1e-3


def test_derivative_names():
    q = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0, derivatives=["aspect_ratio", "m_imag"])
    assert q.params == ("aspect_ratio", "m_imag")
    assert np.isfinite(q.d("aspect_ratio").q_ext)
    with pytest.raises(ValueError, match="no derivative"):
        q.d("x")
    with pytest.raises(ValueError, match="real"):
        q.d("m")
    with pytest.raises(ValueError, match="unknown parameter"):
        tmatrix.efficiencies(2.0, 1.5, 2.0, derivatives=["radius"])
    g = tmatrix.greek(2.0, 1.5, 2.0, 4, derivatives=True)
    assert g.d("x").alpha1.shape == (5,)
    f = tmatrix.phase_matrix(2.0, 1.5, 2.0, MU, derivatives=True)
    assert f.d("m_real").f22.shape == (4,)


def test_derivative_matches_a_difference():
    h = 1e-5
    kw = {"n_terms": 12, "n_theta": 30}
    q = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0, derivatives=True, **kw)
    qp = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0 + h, **kw)
    qm = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0 - h, **kw)
    fd = (qp.q_sca - qm.q_sca) / (2 * h)
    assert q.d("aspect_ratio").q_sca == pytest.approx(fd, rel=1e-6)


def test_the_modes_above_the_cutoff_are_not_computed():
    for derivatives in (False, True):
        t = tmatrix.compute(3.0, 1.33, 2.0, derivatives=derivatives)
        assert t.n_modes < t.n_terms + 1 and len(t.precision_paths) == t.n_modes
    assert t.block(t.n_modes - 1).orders[0] == t.n_modes - 1
    with pytest.raises(IndexError, match="computed modes"):
        t.block(t.n_modes)
    assert f"n_modes={t.n_modes}" in repr(t)


def test_the_tmatrix_object():
    t = tmatrix.compute(3.0, 1.33, 2.0, derivatives=True)
    assert t.converged and t.error_estimate < 1e-8 and len(t.precision_paths) == t.n_modes
    b = t.block(2)
    assert b.t11.shape == (t.n_terms - 1, t.n_terms - 1) and b.orders[0] == 2
    with pytest.raises(IndexError):
        t.block(t.n_terms + 1)
    q, g, f = t.random_orientation(MU, 6)
    q1 = tmatrix.efficiencies(3.0, 1.33, 2.0)
    assert q.q_ext == pytest.approx(q1.q_ext, rel=1e-12)
    assert g.alpha1.shape == (7,) and f is not None and f.f11.shape == (4,)
    assert np.isfinite(q.d("x").q_ext)


def test_errors():
    with pytest.raises(InvalidInputError):
        tmatrix.efficiencies(1.0, 1.5, 0.0)
    with pytest.raises(InvalidInputError, match="element 1"):
        tmatrix.efficiencies([1.0, -1.0], 1.5, 2.0)
    with pytest.raises(OutsideValidatedDomainError):
        tmatrix.efficiencies(1.0, 1.5, 9.0)
    with pytest.raises(ValueError, match="precision"):
        tmatrix.efficiencies(1.0, 1.5, 2.0, precision="quad")
