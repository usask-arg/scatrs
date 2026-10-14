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
            ensemble.LogNormal(0.3, 1.6), np.linspace(0.3, 1.0, 64), 1.45 + 1e-3j, greek=32
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


def test_results_cannot_be_made_writeable():
    r = ensemble.compute(LOGNORMAL, [0.4, 0.5], 1.43, greek=4, mu=[0.0], derivatives=["width"])
    arrays = [
        r.c_ext,
        r.converged,
        r.panels,
        r.error.g,
        r.greek.alpha1,
        r.error.greek.beta1,
        r.d("width").greek.alpha2,
        r.d_error("width").phase.f11,
    ]
    for a in arrays:
        with pytest.raises(ValueError):
            a.flags.writeable = True


def test_buffer_error_when_an_input_is_borrowed_for_writing():
    from scatrs._scatrs import _testing

    wl = np.array([0.4, 0.5])
    with pytest.raises(BufferError):
        _testing.with_write_borrow(wl, lambda: ensemble.compute(LOGNORMAL, wl, 1.5))


def test_refractive_indices_broadcast_to_the_wavelengths_shape():
    wl = np.array([[0.4, 0.5], [0.6, 0.7]])
    m = np.array([1.45, 1.40 + 1e-3j])  # along the last axis
    r = ensemble.compute(LOGNORMAL, wl, m)
    for i in range(2):
        for j in range(2):
            assert r.c_ext[i, j] == ensemble.compute(LOGNORMAL, wl[i, j], m[j]).c_ext
    with pytest.raises(ValueError, match="refractive_index of shape"):
        ensemble.compute(LOGNORMAL, wl, [1.4, 1.5, 1.6])
    with pytest.raises(ValueError, match="refractive_index of shape"):
        ensemble.compute(LOGNORMAL, [0.4, 0.5, 0.6, 0.7], m.reshape(2, 1) * np.ones((2, 2)))


def test_distributions_are_values():
    t1 = ensemble.Tabulated(np.array([0.1, 0.5]), [1.0, 2.0])
    t2 = ensemble.Tabulated([0.1, 0.5], (1.0, 2.0))
    assert t1 == t2 and hash(t1) == hash(t2)
    b = ensemble.MultiModal([(1.0, LOGNORMAL), (2.0, t1)])
    assert hash(b) == hash(ensemble.MultiModal([(1, LOGNORMAL), (2, t2)]))
    c = ensemble.Component(LOGNORMAL, np.array([1.5]))
    assert c == c and c != ensemble.Component(LOGNORMAL, np.array([1.5]))
    with pytest.raises(TypeError, match="must be one of"):
        ensemble.compute("lognormal", 0.5, 1.5)
    with pytest.raises(TypeError, match="must be one of"):
        ensemble.MultiModal([(1.0, "x")])


def test_greek_orders_and_parameter_names():
    empty = ensemble.compute(LOGNORMAL, np.zeros(0), 1.5, greek=7)
    assert empty.greek.alpha1.shape == (0, 8) and empty.c_ext.shape == (0,)
    with pytest.raises(InvalidInputError, match="Greek order"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, greek=2**63)
    a = ensemble.compute(ensemble.Monodisperse(0.3), [0.3, 3.0], 1.5, greek="all")
    n = mie.default_n_max(2 * np.pi * 0.3 / np.array([0.3, 3.0]))
    assert a.greek.alpha1.shape == (2, 2 * n[0] + 1)
    assert np.all(a.greek.alpha1[1, 2 * n[1] + 1 :] == 0)
    for d, names in [
        (ensemble.Gamma(1.0, 0.1), ("effective_radius", "effective_variance")),
        (ensemble.ModifiedGamma(2.0, 15.0, 0.5), ("alpha", "b", "gamma")),
        (ensemble.PowerLaw(3.0, 0.1, 1.0), ("exponent",)),
        (ensemble.Tabulated([0.1, 1.0], [1.0, 1.0]), ()),
    ]:
        assert ensemble.compute(d, 0.5, 1.5, derivatives=True).params == (
            *names,
            "m_real",
            "m_imag",
        )
    with pytest.raises(ValueError, match="twice"):
        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives=["width", "width"])


def test_error_estimates_of_derivatives_and_convergence_flags():
    r = ensemble.compute(LOGNORMAL, 0.5, 1.43, derivatives=["median_radius"])
    e = r.d_error("median_radius")
    assert e.c_ext >= 0 and e.c_ext < 1e-4 * abs(r.d("median_radius").c_ext)
    capped = ensemble.compute(ensemble.LogNormal(1.0, 2.0), 0.5, 1.33, max_panels=1)
    assert not capped.converged and capped.panels > 0
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        ensemble.compute(LOGNORMAL, np.full(2**14, 0.5), 1.5, greek=2**15)


def test_a_single_distribution_has_one_component():
    from scatrs import _scatrs

    spec = (LOGNORMAL._spec(), np.array([1.5 + 0j]), 1.0)
    with pytest.raises(InvalidInputError, match="exactly one component"):
        _scatrs.ensemble_parameters([spec, spec], False)
