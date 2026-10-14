"""Ensemble and Mixture: values, shapes, derivatives, grids, mixtures and errors."""

import copy
import pickle
import threading
import time
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from scatrs import (
    Ensemble,
    Gamma,
    InvalidInputError,
    LogNormal,
    Mixture,
    ModifiedGamma,
    Monodisperse,
    MultiModal,
    OutsideValidatedDomainError,
    PowerLaw,
    Sphere,
    Tabulated,
    mie,
    refractive,
)
from scatrs.ensemble import SCALARS

LOGNORMAL = LogNormal(0.08, 1.6)


def compute(distribution, wavelength, m, **kw):
    """Spheres of the constant refractive index ``m`` over ``distribution``."""
    return Ensemble(Sphere(m), distribution).compute(wavelength, **kw)


def test_monodisperse_is_the_single_sphere():
    r, wl, m = 0.7, np.array([0.45, 0.55]), 1.5 + 0.01j
    e = compute(Monodisperse(r), wl, m, greek="all", mu=[1.0, -1.0])
    q = mie.efficiencies(2 * np.pi * r / wl, m)
    np.testing.assert_allclose(e.c_ext, np.pi * r**2 * q.q_ext, rtol=1e-15)
    np.testing.assert_allclose(e.g, q.g, rtol=1e-14)
    g = mie.greek(2 * np.pi * r / wl, m)
    np.testing.assert_allclose(e.greek.beta1, g.beta1, atol=1e-12)
    assert np.all(e.converged) and np.all(e.panels == 0)


def test_shapes_and_frozen_results():
    wl = np.array([[0.35, 0.5], [0.75, 1.0]])
    r = compute(LOGNORMAL, wl, 1.43, greek=8, mu=np.linspace(-1, 1, 5), derivatives=True)
    assert r.c_ext.shape == (2, 2) and r.converged.dtype == bool
    assert r.greek.alpha1.shape == (2, 2, 9) and r.phase.f34.shape == (2, 2, 5)
    assert r.error.greek.beta2.shape == (2, 2, 9)
    assert r.params == ("median_radius", "width", "m_real", "m_imag")
    np.testing.assert_array_equal(r.greek.alpha1[..., 0], 1.0)
    for a in (r.c_ext, r.greek.alpha1, r.d("width").g, r.error.ssa, r.converged):
        with pytest.raises(ValueError, match="read-only"):
            a[...] = 0
    s = compute(LOGNORMAL, 0.5, 1.43)
    assert s.c_ext.shape == () and s.greek is None and s.phase is None


def test_the_model_is_evaluated_at_each_wavelength():
    wl = np.array([[0.4, 0.5], [0.6, 0.7]])
    table = refractive.Table([0.4, 0.7], [1.45, 1.40 + 3e-3j])
    r = Ensemble(Sphere(table), LOGNORMAL).compute(wl)
    np.testing.assert_array_equal(r.refractive_index, table.evaluate(wl))
    np.testing.assert_array_equal(r.wavelength, wl)
    for i in range(2):
        for j in range(2):
            m = complex(table.evaluate(wl[i, j]))
            assert r.c_ext[i, j] == compute(LOGNORMAL, wl[i, j], m).c_ext
    with pytest.raises(InvalidInputError, match="outside the Table's range"):
        Ensemble(Sphere(table), LOGNORMAL).compute([0.5, 0.8])


def _fd(f, h):
    return (f(+h) - f(-h)) / (2 * h)


def test_derivatives_match_finite_differences():
    wl, m = 0.55, 1.45 + 0.02j
    kw = {"rtol": 1e-11, "max_panels": 50_000, "greek": 4}
    r = compute(LogNormal(0.2, 1.6), wl, m, derivatives=True, **kw)
    steps = {
        "median_radius": lambda e: (LogNormal(0.2 + e, 1.6), m),
        "width": lambda e: (LogNormal(0.2, 1.6 + e), m),
        "m_real": lambda e: (LogNormal(0.2, 1.6), m + e),
        "m_imag": lambda e: (LogNormal(0.2, 1.6), m + 1j * e),
    }
    for p, at in steps.items():
        h = 1e-4
        for name in ("c_ext", "ssa", "g", "c_back"):
            fd = _fd(
                lambda e, at=at, name=name: getattr(compute(at(e)[0], wl, at(e)[1], **kw), name),
                h,
            )
            assert abs(getattr(r.d(p), name) - fd) <= 1e-5 * abs(fd) + 1e-12, (p, name)
        fd = _fd(lambda e, at=at: compute(at(e)[0], wl, at(e)[1], **kw).greek.alpha1[2], h)
        assert abs(r.d(p).greek.alpha1[2] - fd) <= 1e-5 * abs(fd) + 1e-10, p


def test_mixtures_and_their_weights():
    a = Ensemble(Sphere(1.45), LogNormal(0.1, 1.5))
    b = Ensemble(Sphere(1.53 + 0.01j), Gamma(1.0, 0.1))
    m = Mixture([(2.0, a), (0.5, b)]).compute(0.6, derivatives=["c0.weight"])
    ra, rb = a.compute(0.6), b.compute(0.6)
    np.testing.assert_allclose(m.c_ext, (2.0 * ra.c_ext + 0.5 * rb.c_ext) / 2.5, rtol=1e-13)
    # ∂/∂w₀ of (w₀ A + w₁ B)/(w₀ + w₁) = (A − mix)/(w₀ + w₁).
    np.testing.assert_allclose(m.d("c0.weight").c_ext, (ra.c_ext - m.c_ext) / 2.5, rtol=1e-12)
    names = Mixture([(2.0, a), (0.5, b)]).compute(0.6, derivatives=True).params
    assert names[:3] == ("c0.median_radius", "c0.width", "c0.m_real") and names[-1] == "c1.weight"
    bi = MultiModal([(2.0, LogNormal(0.1, 1.5)), (0.5, Monodisperse(1.0))])
    assert compute(bi, 0.5, 1.5, derivatives=True).params[:3] == (
        "mode0.median_radius",
        "mode0.width",
        "mode0.weight",
    )


def test_mixture_results_carry_each_components_refractive_index():
    wl = [[0.4, 0.5, 0.6]]
    a = Ensemble(Sphere(1.45), LOGNORMAL)
    b = Ensemble(Sphere(refractive.Cauchy(1.5, 0.004)), LOGNORMAL)
    r = Mixture([(1.0, a), (1.0, b)]).compute(wl)
    assert r.refractive_index.shape == (2, 1, 3) and r.c_ext.shape == (1, 3)
    np.testing.assert_array_equal(r.refractive_index[1], b.particle.refractive_index.evaluate(wl))
    assert dict(r.grid) == {}


def test_tabulated_and_power_law_agree():
    t = compute(Tabulated([0.1, 0.5], [1.0, 1.0]), 0.5, 1.5, rtol=1e-9)
    p = compute(PowerLaw(0.0, 0.1, 0.5), 0.5, 1.5, rtol=1e-9)
    np.testing.assert_allclose(t.c_ext, p.c_ext, rtol=1e-8)


def test_errors():
    with pytest.raises(InvalidInputError, match="width"):
        compute(LogNormal(0.1, 0.5), 0.5, 1.5)
    with pytest.raises(InvalidInputError, match="rtol"):
        compute(LOGNORMAL, 0.5, 1.5, rtol=1.0)
    with pytest.raises(InvalidInputError, match="wavelength"):
        compute(LOGNORMAL, -0.5, 1.5)
    with pytest.raises(OutsideValidatedDomainError, match="radius"):
        compute(Monodisperse(2000.0), 0.5, 1.5)
    with pytest.raises(ValueError, match="unknown parameter"):
        compute(LOGNORMAL, 0.5, 1.5, derivatives=["radius"])
    with pytest.raises(ValueError, match="no derivative"):
        compute(LOGNORMAL, 0.5, 1.5).d("width")
    with pytest.raises(ValueError, match="greek must be"):
        compute(LOGNORMAL, 0.5, 1.5, greek="some")
    with pytest.raises(TypeError, match="derivatives must be"):
        compute(LOGNORMAL, 0.5, 1.5, derivatives="width")
    with pytest.raises(InvalidInputError, match="weights"):
        Mixture([(0.0, Ensemble(Sphere(1.5), LOGNORMAL))]).compute(0.5)
    with pytest.raises(InvalidInputError, match="needs a component"):
        Mixture([]).compute(0.5)
    with pytest.raises(TypeError, match="pair"):
        Mixture([LOGNORMAL])
    with pytest.raises(TypeError, match=r"pair, not \(float, LogNormal\)$"):
        Mixture([(1.0, LOGNORMAL)])
    with pytest.raises(TypeError, match="must be a Sphere"):
        Ensemble(1.5, LOGNORMAL)
    with pytest.raises(TypeError, match="must be one of"):
        Ensemble(Sphere(1.5), "lognormal")


def test_gil_is_released_while_computing():
    span = {}
    ens = Ensemble(Sphere(1.45 + 1e-3j), LogNormal(0.3, 1.6))

    def worker():
        span["start"] = time.perf_counter()
        ens.compute(np.linspace(0.3, 1.0, 64), greek=32)
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
    r = compute(LOGNORMAL, [0.4, 0.5], 1.43, greek=4, mu=[0.0], derivatives=["width"])
    g = compute(LogNormal([0.1, 0.2], 1.5), 0.5, 1.43)
    arrays = [
        r.c_ext,
        r.converged,
        r.panels,
        r.error.g,
        r.greek.alpha1,
        r.error.greek.beta1,
        r.d("width").greek.alpha2,
        r.d_error("width").phase.f11,
        r.wavelength,
        r.refractive_index,
        g.grid["median_radius"],
    ]
    for a in arrays:
        with pytest.raises(ValueError):
            a.flags.writeable = True


def test_ensembles_and_mixtures_are_values():
    e1 = Ensemble(Sphere(1.5), LogNormal(np.array([0.1, 0.2]), 1.5))
    e2 = Ensemble(Sphere(1.5), LogNormal(np.array([0.1, 0.2]), 1.5))
    assert e1 == e2 and hash(e1) == hash(e2)
    assert e1 != Ensemble(Sphere(1.5), LogNormal([0.1, 0.3], 1.5))
    assert e1 != Ensemble(Sphere(1.6), LogNormal([0.1, 0.2], 1.5))
    with pytest.raises(FrozenInstanceError):
        e1.size = LOGNORMAL
    a1, b1 = Ensemble(Sphere(1.5), LogNormal(0.1, 1.5)), Ensemble(Sphere(1.6), Gamma(1.0, 0.1))
    a2, b2 = Ensemble(Sphere(1.5), LogNormal(0.1, 1.5)), Ensemble(Sphere(1.6), Gamma(1.0, 0.1))
    x1, x2 = Mixture([(1, a1), (2.0, b1)]), Mixture([(1.0, a2), (2.0, b2)])
    assert x1 == x2 and hash(x1) == hash(x2)
    assert x1 != Mixture([(1.0, a2), (3.0, b2)])
    assert x1 != Mixture([(1.0, a2), (2.0, Ensemble(Sphere(1.6), Gamma(1.0, 0.2)))])
    with pytest.raises(FrozenInstanceError):
        x1.components = ()

    def f(wavelength):
        return np.full(wavelength.shape, 1.5 + 0j)

    e = Ensemble(Sphere(refractive.Function(f)), LOGNORMAL)
    assert e == e and hash(e) == hash(e)
    assert e != Ensemble(Sphere(refractive.Function(f)), LOGNORMAL)


def test_buffer_error_when_an_input_is_borrowed_for_writing():
    from scatrs._scatrs import _testing

    wl = np.array([0.4, 0.5])
    with pytest.raises(BufferError):
        _testing.with_write_borrow(wl, lambda: compute(LOGNORMAL, wl, 1.5))


def test_greek_orders_and_parameter_names():
    empty = compute(LOGNORMAL, np.zeros(0), 1.5, greek=7)
    assert empty.greek.alpha1.shape == (0, 8) and empty.c_ext.shape == (0,)
    with pytest.raises(InvalidInputError, match="Greek order"):
        compute(LOGNORMAL, 0.5, 1.5, greek=2**63)
    a = compute(Monodisperse(0.3), [0.3, 3.0], 1.5, greek="all")
    n = mie.default_n_max(2 * np.pi * 0.3 / np.array([0.3, 3.0]))
    assert a.greek.alpha1.shape == (2, 2 * n[0] + 1)
    assert np.all(a.greek.alpha1[1, 2 * n[1] + 1 :] == 0)
    for d, names in [
        (Gamma(1.0, 0.1), ("effective_radius", "effective_variance")),
        (ModifiedGamma(2.0, 15.0, 0.5), ("alpha", "b", "gamma")),
        (PowerLaw(3.0, 0.1, 1.0), ("exponent",)),
        (Tabulated([0.1, 1.0], [1.0, 1.0]), ()),
    ]:
        assert compute(d, 0.5, 1.5, derivatives=True).params == (*names, "m_real", "m_imag")
    with pytest.raises(ValueError, match="twice"):
        compute(LOGNORMAL, 0.5, 1.5, derivatives=["width", "width"])


def test_error_estimates_of_derivatives_and_convergence_flags():
    r = compute(LOGNORMAL, 0.5, 1.43, derivatives=["median_radius"])
    e = r.d_error("median_radius")
    assert e.c_ext >= 0 and e.c_ext < 1e-4 * abs(r.d("median_radius").c_ext)
    capped = compute(LogNormal(1.0, 2.0), 0.5, 1.33, max_panels=1)
    assert not capped.converged and capped.panels > 0
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        compute(LOGNORMAL, np.full(2**14, 0.5), 1.5, greek=2**15)


def test_a_single_distribution_has_one_component():
    from scatrs import _scatrs

    spec = (("lognormal", 0.08, 1.6), np.array([1.5 + 0j]), 1.0)
    with pytest.raises(InvalidInputError, match="exactly one component"):
        _scatrs.ensemble_parameters([spec, spec], False)


def test_a_grid_shares_nodes_and_matches_each_distribution():
    size = LogNormal(np.array([0.1, 0.2, 0.3])[:, None], [1.5, 1.7])
    wl = np.array([[0.4, 0.6, 0.8]])
    ens = Ensemble(Sphere(1.45 + 1e-3j), size)
    assert ens.grid_shape == (3, 2)
    r = ens.compute(wl, greek=8, mu=[0.5, -0.5])
    assert r.c_ext.shape == (3, 2, 1, 3) and r.greek.alpha1.shape == (3, 2, 1, 3, 9)
    assert r.phase.f11.shape == (3, 2, 1, 3, 2) and r.converged.all()
    assert r.params == () and not r.c_ext.flags.writeable
    assert set(r.grid) == {"median_radius", "width"} and r.grid["width"].shape == (3, 2)
    np.testing.assert_array_equal(r.grid["median_radius"][:, 1], [0.1, 0.2, 0.3])
    assert np.all(r.panels == r.panels[0, 0])  # one set of panels per wavelength
    for i, rg in enumerate([0.1, 0.2, 0.3]):
        for j, sg in enumerate([1.5, 1.7]):
            a = compute(LogNormal(rg, sg), wl, 1.45 + 1e-3j, greek=8)
            np.testing.assert_array_less(
                np.abs(r.c_ext[i, j] - a.c_ext), 2 * (r.error.c_ext[i, j] + a.error.c_ext) + 1e-15
            )
            np.testing.assert_array_less(
                np.abs(r.greek.alpha1[i, j] - a.greek.alpha1),
                2 * (r.error.greek.alpha1[i, j] + a.error.greek.alpha1) + 1e-13,
            )


def test_grids_of_multimodal_distributions_and_power_law_limits():
    bi = MultiModal(
        [([1.0, 2.0, 3.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, [1.4, 1.6, 1.8]))]
    )
    r = Ensemble(Sphere(1.5), bi).compute(0.5)
    assert r.c_ext.shape == (3,)
    assert set(r.grid) == {"mode0.weight", "mode1.width"}
    one = compute(MultiModal([(2.0, LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, 1.6))]), 0.5, 1.5)
    assert abs(r.c_ext[1] - one.c_ext) <= 2 * (r.error.c_ext[1] + one.error.c_ext)
    p = Ensemble(Sphere(1.5), PowerLaw(3.0, 0.1, [0.5, 1.0])).compute([0.4, 0.5])
    assert p.c_ext.shape == (2, 2) and np.all(p.c_ext[1] > p.c_ext[0])
    # A table is data: Tabulated never makes a grid.
    assert Ensemble(Sphere(1.5), Tabulated([0.1, 0.2, 0.3], [1.0, 2.0, 1.0])).grid_shape == ()


def test_grid_errors():
    with pytest.raises(ValueError, match="do not broadcast together"):
        Ensemble(Sphere(1.5), LogNormal([0.1, 0.2], [1.5, 1.6, 1.7]))
    grid = Ensemble(Sphere(1.5), LogNormal([0.1, 0.2], 1.5))
    with pytest.raises(NotImplementedError, match="grid"):
        grid.compute(0.5, derivatives=True)
    with pytest.raises(NotImplementedError, match="grid"):
        grid.compute(0.5, derivatives=["width"])
    with pytest.raises(ValueError, match="a mixture needs one distribution"):
        Mixture([(1.0, grid)])
    with pytest.raises(InvalidInputError, match="needs a distribution"):
        Ensemble(Sphere(1.5), LogNormal(np.zeros(0), 1.5)).compute(0.5)


def test_repr_prints_no_arrays():
    r = compute(LOGNORMAL, np.linspace(0.4, 0.8, 50), 1.5, greek=8, derivatives=["width"])
    text = repr(r)
    assert text == (
        "EnsembleResult(shape=(50,), greek=(50, 9), params=('width',), converged=50/50)"
    )
    g = compute(LogNormal([0.1, 0.2], 1.5), 0.5, 1.5)
    assert (
        repr(g) == "EnsembleResult(shape=(2,), params=(), grid=('median_radius',), converged=2/2)"
    )


def _copies(x):
    """``x`` pickled and unpickled, and deep-copied."""
    return [pickle.loads(pickle.dumps(x)), copy.deepcopy(x)]


def _assert_outputs_equal(a, b):
    """The outputs ``a`` and ``b`` are equal, and those of ``a`` read-only."""
    arrays = [(getattr(a, n), getattr(b, n)) for n in SCALARS]
    for part in ("greek", "phase"):
        pa, pb = getattr(a, part), getattr(b, part)
        assert (pa is None) == (pb is None)
        if pa is not None:
            arrays += [(getattr(pa, f.name), getattr(pb, f.name)) for f in fields(pa)]
    for x, y in arrays:
        np.testing.assert_array_equal(x, y)
        with pytest.raises(ValueError):
            x.flags.writeable = True


def test_results_pickle_and_deepcopy():
    r = compute(LOGNORMAL, [0.4, 0.5], 1.43, greek=4, mu=[0.0, 0.5], derivatives=["width"])
    g = compute(LogNormal([0.1, 0.2], 1.5), [0.4, 0.5], 1.43)
    for x in (r, g):
        for c in _copies(x):
            assert c.params == x.params and repr(c) == repr(x)
            _assert_outputs_equal(c, x)
            _assert_outputs_equal(c.error, x.error)
            for p in x.params:
                _assert_outputs_equal(c.d(p), x.d(p))
                _assert_outputs_equal(c.d_error(p), x.d_error(p))
            for n in ("converged", "panels", "wavelength", "refractive_index"):
                np.testing.assert_array_equal(getattr(c, n), getattr(x, n))
                with pytest.raises(ValueError):
                    getattr(c, n).flags.writeable = True
            assert list(c.grid) == list(x.grid) and len(c.grid) == len(x.grid)
            for n in x.grid:
                np.testing.assert_array_equal(c.grid[n], x.grid[n])
                with pytest.raises(ValueError):
                    c.grid[n].flags.writeable = True
            assert not hasattr(c.grid, "__setitem__")
    assert list(g.grid) == ["median_radius"] and "median_radius" in g.grid
    assert dict(r.grid) == {} and repr(g.grid) == "Grid(median_radius=(2,))"


def test_ensembles_and_mixtures_pickle():
    bi = MultiModal([([1.0, 2.0], LogNormal(0.1, 1.5)), (1.0, Tabulated([0.1, 0.5], [1.0, 1.0]))])
    e = Ensemble(Sphere(1.5), bi)
    x = Mixture(
        [
            (1.0, Ensemble(Sphere(1.5), Tabulated([0.1, 0.5], [1.0, 2.0]))),
            (2.0, Ensemble(Sphere(1.6), MultiModal([(1.0, LOGNORMAL), (2.0, Gamma(1.0, 0.1))]))),
        ]
    )
    for v in (e, x):
        for c in _copies(v):
            assert c == v and hash(c) == hash(v)
    for c in _copies(e):
        assert c.grid_shape == (2,)
        with pytest.raises(ValueError):
            c.size.modes[0][0].flags.writeable = True


def test_ensembles_of_multimodal_tables_compare_the_tables():
    a = Ensemble(Sphere(1.5), MultiModal([(1, Tabulated([0.1, 0.2, 0.3], [0.0, 1.0, 0.0]))]))
    b = Ensemble(Sphere(1.5), MultiModal([(1, Tabulated([1.0, 2.0, 3.0], [0.0, 5.0, 0.0]))]))
    assert a != b and hash(a) != hash(b)


def test_mixture_weights_are_real_scalars():
    e = Ensemble(Sphere(1.5), LOGNORMAL)
    for w in (True, np.bool_(True), "1.0", b"1", None):
        with pytest.raises(TypeError, match="the weight of component 1"):
            Mixture([(1.0, e), (w, e)])
    with pytest.raises(TypeError, match=r"weight of component 0 must be a real number, not an ar"):
        Mixture([([1.0, 2.0], e)])
    with pytest.raises(TypeError, match=r"weight of component 0 must be a real number, not bool$"):
        Mixture([(np.array(True), e)])
    with pytest.raises(TypeError, match=r"weight of component 0 must be a real number, not str$"):
        Mixture([("1.0", e)])
    w = Mixture([(np.int64(2), e)]).components[0][0]
    assert type(w) is float and w == 2.0


def test_a_list_of_distributions_is_a_batch_of_any_kinds():
    sizes = [
        LogNormal(0.1, 1.5),
        Gamma(0.4, 0.1),
        Tabulated([0.1, 0.2, 0.3], [0.0, 1.0, 0.0]),
        Tabulated([0.2, 0.4, 0.6], [0.0, 2.0, 0.0]),
        MultiModal([(2.0, LogNormal(0.1, 1.5)), (1.0, Monodisperse(0.3))]),
    ]
    wl = np.array([0.4, 0.6, 0.8])
    batch = Ensemble(Sphere(1.45 + 1e-3j), sizes)
    assert batch.size == tuple(sizes) and batch.grid_shape == (5,)
    assert batch == Ensemble(Sphere(1.45 + 1e-3j), tuple(sizes))
    assert hash(batch) == hash(Ensemble(Sphere(1.45 + 1e-3j), tuple(sizes)))
    r = batch.compute(wl, greek=8, mu=[0.5, -0.5])
    assert r.c_ext.shape == (5, 3) and r.greek.alpha1.shape == (5, 3, 9)
    assert r.phase.f11.shape == (5, 3, 2) and r.converged.all()
    assert r.params == () and dict(r.grid) == {}
    for i, d in enumerate(sizes):
        a = compute(d, wl, 1.45 + 1e-3j, greek=8)
        np.testing.assert_array_less(
            np.abs(r.c_ext[i] - a.c_ext), 2 * (r.error.c_ext[i] + a.error.c_ext) + 1e-15
        )
        np.testing.assert_array_less(
            np.abs(r.greek.alpha1[i] - a.greek.alpha1),
            2 * (r.error.greek.alpha1[i] + a.error.greek.alpha1) + 1e-13,
        )


def test_list_errors():
    with pytest.raises(TypeError, match=r"size\[1\] must be one of"):
        Ensemble(Sphere(1.5), [LOGNORMAL, "lognormal"])
    with pytest.raises(ValueError, match=r"size\[0\] is a grid of shape \(2,\)"):
        Ensemble(Sphere(1.5), [LogNormal([0.1, 0.2], 1.5)])
    batch = Ensemble(Sphere(1.5), [LOGNORMAL, Gamma(0.4, 0.1)])
    with pytest.raises(NotImplementedError, match="list"):
        batch.compute(0.5, derivatives=True)
    with pytest.raises(ValueError, match="a mixture needs one distribution"):
        Mixture([(1.0, batch)])
    with pytest.raises(InvalidInputError, match="needs a distribution"):
        Ensemble(Sphere(1.5), []).compute(0.5)
