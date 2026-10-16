"""Ensembles of randomly oriented spheroids: an aspect ratio of 1 is the sphere ensemble, a
monodisperse ensemble is the single spheroid, the derivative names and the aspect-ratio
derivative, mixtures with spheres, and the particle's validation."""

import numpy as np
import pytest

import scatrs
from scatrs import tmatrix

DUST = 1.53 + 0.008j


def test_an_aspect_ratio_of_one_is_the_sphere_ensemble():
    size = scatrs.LogNormal(0.05, 1.5)
    kw = {"greek": 8, "mu": [1.0, 0.0, -1.0]}
    s = scatrs.Ensemble(scatrs.Sphere(DUST), size).compute(1.0, **kw)
    p = scatrs.Ensemble(scatrs.Spheroid(DUST, 1.0), size).compute(1.0, **kw)
    assert p.converged
    np.testing.assert_allclose(p.c_ext, s.c_ext, rtol=1e-9)
    np.testing.assert_allclose(p.g, s.g, rtol=0, atol=1e-9)
    np.testing.assert_allclose(p.greek.alpha2, s.greek.alpha2, rtol=0, atol=1e-8)
    np.testing.assert_allclose(p.phase.f22, s.phase.f11, rtol=1e-8)
    np.testing.assert_allclose(p.phase.f44, s.phase.f33, rtol=1e-8, atol=1e-10)
    assert abs(p.depolarization) < 1e-10
    # A sphere's F22 and F44 are its F11 and F33, and it does not depolarize.
    np.testing.assert_array_equal(s.phase.f22, s.phase.f11)
    np.testing.assert_array_equal(s.phase.f44, s.phase.f33)
    assert s.depolarization == 0


def test_a_monodisperse_ensemble_is_the_single_spheroid():
    r, eps = 0.3, 2.0
    e = scatrs.Ensemble(scatrs.Spheroid(DUST, eps), scatrs.Monodisperse(r)).compute(1.0)
    q = tmatrix.efficiencies(2 * np.pi * r, DUST, eps)
    np.testing.assert_allclose(e.c_ext, np.pi * r * r * q.q_ext, rtol=1e-14)
    np.testing.assert_allclose(e.g, q.g, rtol=0, atol=1e-14)
    assert e.depolarization > 0.01


def test_the_aspect_ratio_derivative():
    size = scatrs.LogNormal(0.04, 1.4)
    mono = scatrs.Ensemble(scatrs.Spheroid(DUST, 2.0), scatrs.Monodisperse(0.1))
    assert mono.compute(1.0, derivatives=True).params == (
        "radius",
        "m_real",
        "m_imag",
        "aspect_ratio",
    )
    ensemble = scatrs.Ensemble(scatrs.Spheroid(DUST, 2.0), size)
    r = ensemble.compute(1.0, rtol=1e-8, derivatives=["aspect_ratio"])
    h = 1e-4
    c = [
        scatrs.Ensemble(scatrs.Spheroid(DUST, 2.0 + s * h), size).compute(1.0, rtol=1e-8).c_ext
        for s in (1, -1)
    ]
    assert r.d("aspect_ratio").c_ext == pytest.approx((c[0] - c[1]) / (2 * h), rel=1e-5)


def test_a_mixture_of_spheres_and_spheroids():
    sphere = scatrs.Ensemble(scatrs.Sphere(1.45), scatrs.LogNormal(0.05, 1.5))
    spheroid = scatrs.Ensemble(scatrs.Spheroid(DUST, 3.0), scatrs.LogNormal(0.08, 1.4))
    kw = {"mu": [0.5, -1.0], "normalized": False}
    mix = scatrs.Mixture([(2.0, sphere), (0.5, spheroid)]).compute(1.0, **kw)
    a, b = sphere.compute(1.0, **kw), spheroid.compute(1.0, **kw)
    np.testing.assert_allclose(mix.c_ext, 0.8 * a.c_ext + 0.2 * b.c_ext, rtol=1e-14)
    np.testing.assert_allclose(mix.phase.f22, 0.8 * a.phase.f22 + 0.2 * b.phase.f22, rtol=1e-13)
    assert 0 < mix.depolarization < b.depolarization


def test_the_particle_is_validated():
    p = scatrs.Spheroid(1.5, 2)
    assert p.aspect_ratio == 2.0 and isinstance(p.aspect_ratio, float)
    assert p == scatrs.Spheroid(1.5, 2.0) and hash(p) == hash(scatrs.Spheroid(1.5, 2.0))
    with pytest.raises(ValueError, match="finite and positive"):
        scatrs.Spheroid(1.5, 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        scatrs.Spheroid(1.5, float("inf"))
    with pytest.raises(TypeError, match="real number"):
        scatrs.Spheroid(1.5, "2")
    with pytest.raises(TypeError, match="real number"):
        scatrs.Spheroid(1.5, True)
    with pytest.raises(TypeError, match="Sphere, a LayeredSphere or a Spheroid"):
        scatrs.Ensemble("dust", scatrs.LogNormal(0.1, 1.5))
