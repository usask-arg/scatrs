"""Ensemble and Mixture call the engine exactly as the functions they replaced did, so
these results are bitwise those of 0.1.0's ``scatrs.ensemble.compute``, ``compute_batch`` and
``compute_mixture`` for the same inputs.
"""

import numpy as np

from scatrs import Ensemble, Gamma, LogNormal, Mixture, MultiModal, Sphere, Tabulated, _scatrs

WL = np.array([[0.35, 0.5], [0.75, 1.0]])
MU = np.linspace(-1, 1, 7)
OPTIONS = dict(rtol=1e-6, max_panels=4096, allow_unvalidated=False, refine_backscatter=False)


def _engine_options():
    return [
        8,  # greek order
        False,  # all orders
        np.ascontiguousarray(MU),
        True,  # normalized
    ]


def _equal(raw, result, shape, params=()):
    names = (
        "c_ext",
        "c_sca",
        "c_abs",
        "c_back",
        "ssa",
        "g",
        "lidar_ratio",
        "depolarization",
        "effective_radius",
        "mean_area",
        "mean_volume",
    )
    for key, outputs in [("values", result), ("errors", result.error)]:
        for row, name in zip(raw[key], names, strict=True):
            np.testing.assert_array_equal(getattr(outputs, name), row.reshape(shape))
    greek = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
    for row, name in zip(raw["greek"], greek, strict=True):
        np.testing.assert_array_equal(getattr(result.greek, name), row.reshape((*shape, 9)))
    for key, outputs in [("phase", result), ("phase_errors", result.error)]:
        for row, name in zip(raw[key], ("f11", "f12", "f33", "f34"), strict=True):
            np.testing.assert_array_equal(getattr(outputs.phase, name), row.reshape((*shape, 7)))
    for k, p in enumerate(params):
        for row, name in zip(raw["d"][k], names, strict=True):
            np.testing.assert_array_equal(getattr(result.d(p), name), row.reshape(shape))
        for row, name in zip(raw["greek_d"][k], greek, strict=True):
            np.testing.assert_array_equal(
                getattr(result.d(p).greek, name), row.reshape((*shape, 9))
            )
    np.testing.assert_array_equal(result.converged, raw["converged"].reshape(shape))
    np.testing.assert_array_equal(result.panels, raw["panels"].reshape(shape))


def test_a_scalar_ensemble_is_the_single_distribution_call():
    m = 1.45 + 2e-3j
    size = MultiModal([(2.0, LogNormal(0.1, 1.5)), (1.0, Gamma(0.5, 0.1))])
    r = Ensemble(Sphere(m), size).compute(
        WL, greek=8, mu=MU, derivatives=["mode0.width", "m_imag"], **OPTIONS
    )
    spec = ("multimodal", [(2.0, ("lognormal", 0.1, 1.5)), (1.0, ("gamma", 0.5, 0.1))])
    items = [(spec, ("sphere", np.full(4, m)), 1.0)]
    names = _scatrs.ensemble_parameters(items, False)
    raw = _scatrs.ensemble_compute(
        items,
        WL.reshape(-1),
        False,
        *_engine_options(),
        [names.index("mode0.width"), names.index("m_imag")],
        *OPTIONS.values(),
    )
    _equal(raw, r, WL.shape, ("mode0.width", "m_imag"))


def test_a_grid_is_the_batch_call():
    m = 1.45 + 2e-3j
    radii, widths = np.array([0.1, 0.2, 0.3]), np.array([1.5, 1.7])
    r = Ensemble(Sphere(m), LogNormal(radii[:, None], widths)).compute(
        WL, greek=8, mu=MU, **OPTIONS
    )
    specs = [("lognormal", float(a), float(b)) for a in radii for b in widths]
    raw = _scatrs.ensemble_compute_batch(
        specs, ("sphere", np.full(4, m)), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
    )
    _equal(raw, r, (3, 2, *WL.shape))


def test_a_mixture_is_the_mixture_call():
    a = Ensemble(Sphere(1.45), LogNormal(0.1, 1.5))
    b = Ensemble(Sphere(1.53 + 0.01j), Gamma(1.0, 0.1))
    r = Mixture([(2.0, a), (0.5, b)]).compute(
        WL, greek=8, mu=MU, derivatives=["c1.weight"], **OPTIONS
    )
    items = [
        (("lognormal", 0.1, 1.5), ("sphere", np.full(4, 1.45 + 0j)), 2.0),
        (("gamma", 1.0, 0.1), ("sphere", np.full(4, 1.53 + 0.01j)), 0.5),
    ]
    names = _scatrs.ensemble_parameters(items, True)
    raw = _scatrs.ensemble_compute(
        items,
        WL.reshape(-1),
        True,
        *_engine_options(),
        [names.index("c1.weight")],
        *OPTIONS.values(),
    )
    _equal(raw, r, WL.shape, ("c1.weight",))


def test_a_list_is_the_batch_call():
    m = 1.45 + 2e-3j
    sizes = [LogNormal(0.1, 1.5), Gamma(0.5, 0.1), Tabulated([0.1, 0.3], [1.0, 1.0])]
    r = Ensemble(Sphere(m), sizes).compute(WL, greek=8, mu=MU, **OPTIONS)
    specs = [("lognormal", 0.1, 1.5), ("gamma", 0.5, 0.1), ("tabulated", [0.1, 0.3], [1.0, 1.0])]
    raw = _scatrs.ensemble_compute_batch(
        specs, ("sphere", np.full(4, m)), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
    )
    _equal(raw, r, (3, *WL.shape))
