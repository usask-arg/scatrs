"""LayeredSphere: construction, ensembles and mixtures of layered spheres, derivatives, errors."""

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scatrs import (
    Ensemble,
    LayeredSphere,
    LogNormal,
    Mixture,
    Monodisperse,
    OutsideValidatedDomainError,
    Sphere,
    refractive,
)
from scatrs.ensemble import SCALARS

SIZE = LogNormal(0.1, 1.5)
CORE, SHELL = 1.95 + 0.79j, 1.43 + 1e-8j


def test_construction_converts_and_stores_tuples():
    p = LayeredSphere([CORE, refractive.Cauchy(1.5, 0.004)], [0.5])
    assert p.refractive_index == (refractive.Constant(CORE), refractive.Cauchy(1.5, 0.004))
    assert p.radius_fraction == (0.5,) and p.layers == 2
    assert LayeredSphere.coated(CORE, SHELL, core_fraction=0.5) == LayeredSphere(
        (CORE, SHELL), (0.5,)
    )
    assert LayeredSphere([1.5]).radius_fraction == ()
    assert LayeredSphere(np.array([1.5, 1.33]), np.array([0.4])).radius_fraction == (0.4,)
    # Frozen and hashable.
    assert hash(p) == hash(LayeredSphere([CORE, refractive.Cauchy(1.5, 0.004)], (0.5,)))
    with pytest.raises(FrozenInstanceError):
        p.radius_fraction = (0.6,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("m", "f", "error", "match"),
    [
        ([], (), ValueError, "at least one layer"),
        (1.5, (), TypeError, "sequence"),
        ("1.5", (), TypeError, "sequence"),
        ([1.5, [1.33]], (0.5,), TypeError, "for values at given wavelengths"),
        ([1.5, 1.33], (), ValueError, "needs 1 radius fractions"),
        ([1.5, 1.33], (0.5, 0.6), ValueError, "needs 1 radius fractions"),
        ([1.5, 1.33], (0.0,), ValueError, r"\(0, 1\]"),
        ([1.5, 1.33], (1.2,), ValueError, r"\(0, 1\]"),
        ([1.5, 1.33], (float("nan"),), ValueError, r"\(0, 1\]"),
        ([1.5, 1.4, 1.33], (0.6, 0.5), ValueError, "nondecreasing"),
        ([1.5, 1.33], (True,), TypeError, "real number"),
        ([1.5, 1.33], 0.5, TypeError, "sequence"),
    ],
)
def test_bad_layered_spheres_are_rejected(m, f, error, match):
    with pytest.raises(error, match=match):
        LayeredSphere(m, f)


def test_one_layer_is_the_sphere_ensemble_bitwise():
    wl = np.array([0.45, 0.8])
    model = refractive.Cauchy(1.5, 0.004)
    kw = {"greek": 10, "mu": [0.3, -0.8], "derivatives": True}
    a = Ensemble(Sphere(model), SIZE).compute(wl, **kw)
    b = Ensemble(LayeredSphere([model]), SIZE).compute(wl, **kw)
    assert b.params == ("median_radius", "width", "layer0.m_real", "layer0.m_imag")
    for name in SCALARS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_array_equal(
            getattr(a.d("m_imag"), name), getattr(b.d("layer0.m_imag"), name)
        )
    np.testing.assert_array_equal(a.greek.beta1, b.greek.beta1)
    np.testing.assert_array_equal(a.d("m_real").phase.f12, b.d("layer0.m_real").phase.f12)
    assert b.refractive_index.shape == (1, 2)
    np.testing.assert_array_equal(b.refractive_index[0], a.refractive_index)


def test_derivative_names_and_the_refractive_index():
    wl = np.array([[0.45, 0.55, 0.8]])
    p = LayeredSphere([CORE, 1.6 + 0.01j, SHELL], [0.3, 0.8])
    r = Ensemble(p, SIZE).compute(wl, derivatives=True)
    assert r.params == (
        "median_radius",
        "width",
        "layer0.m_real",
        "layer0.m_imag",
        "layer1.m_real",
        "layer1.m_imag",
        "layer2.m_real",
        "layer2.m_imag",
        "layer0.radius_fraction",
        "layer1.radius_fraction",
    )
    assert r.c_ext.shape == (1, 3) and r.refractive_index.shape == (3, 1, 3)
    np.testing.assert_array_equal(r.refractive_index[1], np.full((1, 3), 1.6 + 0.01j))


def test_radius_fraction_derivative_matches_a_central_difference():
    wl, h = 0.55, 1e-5

    def run(f, **kw):
        p = LayeredSphere.coated(CORE, SHELL, core_fraction=f)
        return Ensemble(p, SIZE).compute(wl, rtol=1e-11, max_panels=50_000, **kw)

    r = run(0.5, derivatives=["layer0.radius_fraction"])
    plus, minus = run(0.5 + h), run(0.5 - h)
    for name in ("c_ext", "c_abs", "g"):
        fd = (getattr(plus, name) - getattr(minus, name)) / (2 * h)
        assert getattr(r.d("layer0.radius_fraction"), name) == pytest.approx(fd, rel=1e-6)
    # A larger black-carbon core absorbs more.
    assert r.d("layer0.radius_fraction").c_abs > 0


def test_mixtures_and_grids_of_layered_spheres():
    wl = np.array([0.45, 0.55])
    coated = Ensemble(LayeredSphere.coated(CORE, SHELL, core_fraction=0.5), SIZE)
    sphere = Ensemble(Sphere(1.45), LogNormal(0.2, 1.4))
    mix = Mixture([(2.0, sphere), (0.5, coated)])
    values = mix.compute(wl)
    a, b = sphere.compute(wl), coated.compute(wl)
    np.testing.assert_allclose(values.c_ext, 0.8 * a.c_ext + 0.2 * b.c_ext, rtol=1e-14)
    r = mix.compute(wl, derivatives=["c1.layer0.radius_fraction", "c0.m_real"])
    assert r.params == ("c1.layer0.radius_fraction", "c0.m_real")
    assert np.all(r.d("c1.layer0.radius_fraction").c_abs > 0)
    assert [m.shape for m in r.refractive_index] == [(2,), (2, 2)]
    # The result pickles and copies read-only.
    for c in (pickle.loads(pickle.dumps(r)), copy.deepcopy(r)):
        np.testing.assert_array_equal(c.refractive_index[1], r.refractive_index[1])
        with pytest.raises(ValueError):
            c.refractive_index[1].flags.writeable = True
    # A grid of distributions of layered spheres, against its points.
    grid = Ensemble(coated.particle, LogNormal(np.array([0.1, 0.3]), 1.5)).compute(wl)
    for k, rg in enumerate((0.1, 0.3)):
        alone = Ensemble(coated.particle, LogNormal(rg, 1.5)).compute(wl)
        np.testing.assert_allclose(grid.c_ext[k], alone.c_ext, atol=2 * alone.error.c_ext.max())


def test_the_derivatives_domain_and_wrong_particles():
    coated = LayeredSphere.coated(CORE, SHELL, core_fraction=0.5)
    big = Ensemble(coated, Monodisperse(200.0))  # x_L ≈ 2300 at λ = 0.55
    assert big.compute(0.55).c_ext > 0
    with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
        big.compute(0.55, derivatives=["layer0.radius_fraction"])
    # So does the radius derivative of a monodisperse ensemble, at fixed fractions.
    with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
        big.compute(0.55, derivatives=["radius"])
    with pytest.raises(TypeError, match="Sphere, a LayeredSphere or a Spheroid"):
        Ensemble(1.5, SIZE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown parameter"):
        Ensemble(coated, SIZE).compute(0.55, derivatives=["m_real"])
