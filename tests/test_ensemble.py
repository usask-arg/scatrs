"""scatrs.ensemble: values, shapes, derivatives, mixtures and errors."""

import threading
import time

import numpy as np
import pytest

from scatrs import InvalidInputError, OutsideValidatedDomainError, ensemble, mie

LOGNORMAL = ensemble.LogNormal(0.08, 1.6)


def test_monodisperse_is_the_single_sphere():
    r, wl, m = 0.7, np.array([0.45, 0.55]), 1.5 + 0.01j
    e = ensemble.compute(ensemble.Monodisperse(r), wl, m, greek="all", mu=[1.0, -1.0])
    q = mie.efficiencies(2 * np.pi * r / wl, m)
    np.testing.assert_allclose(e.c_ext, np.pi * r**2 * q.q_ext, rtol=1e-15)
    np.testing.assert_allclose(e.g, q.g, rtol=1e-14)
    g = mie.greek(2 * np.pi * r / wl, m)
    np.testing.assert_allclose(e.greek.beta1, g.beta1, atol=1e-12)
    assert np.all(e.converged) and np.all(e.panels == 0)


def test_shapes_and_frozen_results():
    wl = np.array([[0.35, 0.5], [0.75, 1.0]])
    r = ensemble.compute(LOGNORMAL, wl, 1.43, greek=8, mu=np.linspace(-1, 1, 5), derivatives=True)
    assert r.c_ext.shape == (2, 2) and r.converged.dtype == bool
    assert r.greek.alpha1.shape == (2, 2, 9) and r.phase.f34.shape == (2, 2, 5)
    assert r.error.greek.beta2.shape == (2, 2, 9)
    assert r.params == ("median_radius", "width", "m_real", "m_imag")
    np.testing.assert_array_equal(r.greek.alpha1[..., 0], 1.0)
    for a in (r.c_ext, r.greek.alpha1, r.d("width").g, r.error.ssa, r.converged):
        with pytest.raises(ValueError, match="read-only"):
            a[...] = 0
    s = ensemble.compute(LOGNORMAL, 0.5, 1.43)
    assert s.c_ext.shape == () and s.greek is None and s.phase is None


def test_refractive_index_broadcasts_to_the_wavelengths():
    wl = [0.4, 0.6]
    both = ensemble.compute(LOGNORMAL, wl, [1.45, 1.43 + 1e-3j])
    one = ensemble.compute(LOGNORMAL, wl[1], 1.43 + 1e-3j)
    assert both.c_ext[1] == one.c_ext


def _fd(f, h):
    return (f(+h) - f(-h)) / (2 * h)


def test_derivatives_match_finite_differences():
    wl, m = 0.55, 1.45 + 0.02j
    kw = {"rtol": 1e-11, "max_panels": 50_000, "greek": 4}
    r = ensemble.compute(ensemble.LogNormal(0.2, 1.6), wl, m, derivatives=True, **kw)
    steps = {
        "median_radius": lambda e: (ensemble.LogNormal(0.2 + e, 1.6), m),
        "width": lambda e: (ensemble.LogNormal(0.2, 1.6 + e), m),
        "m_real": lambda e: (ensemble.LogNormal(0.2, 1.6), m + e),
        "m_imag": lambda e: (ensemble.LogNormal(0.2, 1.6), m + 1j * e),
    }
    for p, at in steps.items():
        h = 1e-4
        for name in ("c_ext", "ssa", "g", "c_back"):
            fd = _fd(
                lambda e, at=at, name=name: getattr(
                    ensemble.compute(at(e)[0], wl, at(e)[1], **kw), name
                ),
                h,
            )
            assert abs(getattr(r.d(p), name) - fd) <= 1e-5 * abs(fd) + 1e-12, (p, name)
        fd = _fd(lambda e, at=at: ensemble.compute(at(e)[0], wl, at(e)[1], **kw).greek.alpha1[2], h)
        assert abs(r.d(p).greek.alpha1[2] - fd) <= 1e-5 * abs(fd) + 1e-10, p


def test_mixtures_and_their_weights():
    a = ensemble.Component(ensemble.LogNormal(0.1, 1.5), 1.45, 2.0)
    b = ensemble.Component(ensemble.Gamma(1.0, 0.1), 1.53 + 0.01j, 0.5)
    m = ensemble.compute_mixture([a, b], 0.6, derivatives=["c0.weight"])
    ra = ensemble.compute(a.distribution, 0.6, a.refractive_index)
    rb = ensemble.compute(b.distribution, 0.6, b.refractive_index)
    np.testing.assert_allclose(m.c_ext, (2.0 * ra.c_ext + 0.5 * rb.c_ext) / 2.5, rtol=1e-13)
    # ∂/∂w₀ of (w₀ A + w₁ B)/(w₀ + w₁) = (A − mix)/(w₀ + w₁).
    np.testing.assert_allclose(m.d("c0.weight").c_ext, (ra.c_ext - m.c_ext) / 2.5, rtol=1e-12)
    names = ensemble.compute_mixture([a, b], 0.6, derivatives=True).params
    assert names[:3] == ("c0.median_radius", "c0.width", "c0.m_real") and names[-1] == "c1.weight"
    bi = ensemble.MultiModal(
        [(2.0, ensemble.LogNormal(0.1, 1.5)), (0.5, ensemble.Monodisperse(1.0))]
    )
    assert ensemble.compute(bi, 0.5, 1.5, derivatives=True).params[:3] == (
        "mode0.median_radius",
        "mode0.width",
        "mode0.weight",
    )


def test_tabulated_and_power_law_agree():
    t = ensemble.compute(ensemble.Tabulated([0.1, 0.5], [1.0, 1.0]), 0.5, 1.5, rtol=1e-9)
    p = ensemble.compute(ensemble.PowerLaw(0.0, 0.1, 0.5), 0.5, 1.5, rtol=1e-9)
    np.testing.assert_allclose(t.c_ext, p.c_ext, rtol=1e-8)


def test_errors():
    with pytest.raises(InvalidInputError, match="width"):
        ensemble.compute(ensemble.LogNormal(0.1, 0.5), 0.5, 1.5)
    with pytest.raises(InvalidInputError, match="rtol"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, rtol=1.0)
    with pytest.raises(InvalidInputError, match="wavelength"):
        ensemble.compute(LOGNORMAL, -0.5, 1.5)
    with pytest.raises(OutsideValidatedDomainError, match="radius"):
        ensemble.compute(ensemble.Monodisperse(2000.0), 0.5, 1.5)
    with pytest.raises(ValueError, match="unknown parameter"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives=["radius"])
    with pytest.raises(ValueError, match="no derivative"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5).d("width")
    with pytest.raises(ValueError, match="greek must be"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, greek="some")
    with pytest.raises(TypeError, match="Component"):
        ensemble.compute_mixture([LOGNORMAL], 0.5)
    with pytest.raises(TypeError, match="derivatives must be"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives="width")
    with pytest.raises(InvalidInputError, match="weights"):
        ensemble.compute_mixture([ensemble.Component(LOGNORMAL, 1.5, 0.0)], 0.5)
    with pytest.raises(ValueError):
        ensemble.compute(LOGNORMAL, [0.4, 0.5], [1.5, 1.5, 1.5])


def test_gil_is_released_while_computing():
    span = {}

    def worker():
        span["start"] = time.perf_counter()
        ensemble.compute(
            ensemble.LogNormal(0.3, 1.6), np.linspace(0.3, 1.0, 8), 1.45 + 1e-3j, greek=32
        )
        span["end"] = time.perf_counter()

    stamps = []
    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        time.sleep(0.0005)
        stamps.append(time.perf_counter())
    thread.join()
    quarter = (span["end"] - span["start"]) / 4
    inside = sum(span["start"] + quarter < t < span["end"] - quarter for t in stamps)
    assert span["end"] - span["start"] > 0.02
    assert inside >= 5
