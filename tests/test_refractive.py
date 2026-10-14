"""scatrs.refractive models and scatrs.Sphere."""

import numpy as np
import pytest

from scatrs import InvalidInputError, Sphere, refractive
from scatrs.refractive import Cauchy, Constant, Function, Sellmeier, Table

WL = np.array([[0.35, 0.5, 0.65], [0.8, 1.0, 1.6]])


def test_constant():
    c = Constant(1.5 + 1e-3j)
    np.testing.assert_array_equal(c.evaluate(WL), np.full(WL.shape, 1.5 + 1e-3j))
    assert c.evaluate(0.5).shape == ()
    assert c.parameters == ("n", "k")
    d = c.derivatives(WL)
    assert d.shape == (2, *WL.shape)
    np.testing.assert_array_equal(d[0], 1.0)
    np.testing.assert_array_equal(d[1], 1j)
    assert Constant(1.5) == Constant(1.5 + 0j) and hash(Constant(1.5)) == hash(Constant(1.5 + 0j))


def test_table_interpolates_linearly_and_is_exact_at_its_nodes():
    wl, m = [0.4, 0.6, 1.0], [1.50 + 1e-3j, 1.46 + 3e-3j, 1.44 + 1e-2j]
    t = Table(wl, m)
    np.testing.assert_array_equal(t.evaluate(wl), m)
    np.testing.assert_allclose(t.evaluate(0.5), 1.48 + 2e-3j, rtol=1e-15)
    np.testing.assert_allclose(t.evaluate(0.8), 1.45 + 6.5e-3j, rtol=1e-15)
    assert t.parameters == () and t.derivatives(WL[:, 1:]).shape == (0, 2, 2)
    assert t == Table(np.array(wl), tuple(m)) and hash(t) == hash(Table(np.array(wl), tuple(m)))


def test_table_log_k_interpolates_the_logarithm_of_k():
    t = Table([0.4, 0.8], [1.5 + 1e-6j, 1.5 + 1e-2j], interpolation="log_k")
    np.testing.assert_allclose(t.evaluate(0.6), 1.5 + 1e-4j, rtol=1e-14)
    np.testing.assert_allclose(t.evaluate([0.4, 0.8]), [1.5 + 1e-6j, 1.5 + 1e-2j], rtol=1e-15)


def test_table_errors():
    t = Table([0.4, 0.8], [1.5, 1.4])
    with pytest.raises(InvalidInputError, match=r"wavelength 0\.3 is outside the Table's range"):
        t.evaluate([0.5, 0.3, 0.9])
    with pytest.raises(InvalidInputError, match="outside"):
        t.evaluate(np.nan)
    with pytest.raises(InvalidInputError, match="at least 2"):
        Table([0.4], [1.5])
    with pytest.raises(InvalidInputError, match="same length"):
        Table([0.4, 0.5], [1.5, 1.4, 1.3])
    with pytest.raises(InvalidInputError, match="strictly increasing"):
        Table([0.5, 0.4], [1.5, 1.4])
    with pytest.raises(InvalidInputError, match="finite"):
        Table([0.4, np.inf], [1.5, 1.4])
    with pytest.raises(InvalidInputError, match="≥ 0"):
        Table([0.4, 0.5], [1.5, 1.4 - 1e-3j])
    with pytest.raises(InvalidInputError, match="> 0 for log_k"):
        Table([0.4, 0.5], [1.5, 1.4 + 1e-3j], interpolation="log_k")
    with pytest.raises(ValueError, match="interpolation"):
        Table([0.4, 0.5], [1.5, 1.4], interpolation="cubic")


def test_cauchy():
    c = Cauchy(1.45, 0.004, 1e-4, k=1e-5)
    np.testing.assert_allclose(
        c.evaluate(WL), 1.45 + 0.004 / WL**2 + 1e-4 / WL**4 + 1e-5j, rtol=1e-15
    )
    assert c.parameters == ("A", "B", "C", "k")
    with pytest.raises(InvalidInputError, match="k must not be negative"):
        Cauchy(1.45, k=-1e-3)
    with pytest.raises(InvalidInputError, match="finite"):
        Cauchy(np.nan)


def test_sellmeier():
    # Fused silica (Malitson 1965), λ in µm.
    b, c = (0.6961663, 0.4079426, 0.8974794), (0.0684043**2, 0.1162414**2, 9.896161**2)
    s = Sellmeier(b, c)
    np.testing.assert_allclose(s.evaluate(0.5893), 1.4584, atol=1e-4)
    l2 = WL**2
    n = np.sqrt(1 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c, strict=True)))
    np.testing.assert_allclose(s.evaluate(WL), n, rtol=1e-15)
    assert s.parameters == ("B0", "B1", "B2", "C0", "C1", "C2", "k")
    with pytest.raises(InvalidInputError, match="n² must be positive"):
        Sellmeier([1.0], [0.25]).evaluate([0.6, 0.4])  # n² < 0 just above the pole at λ = 0.5
    with pytest.raises(InvalidInputError, match="n² must be positive"):
        Sellmeier([1.0], [0.25]).evaluate(0.5)  # at the pole
    with pytest.raises(InvalidInputError, match="same length"):
        Sellmeier([1.0, 2.0], [0.25])


@pytest.mark.parametrize(
    "model",
    [
        Constant(1.5 + 1e-3j),
        Cauchy(1.45, 0.004, 1e-4, k=1e-5),
        Sellmeier([0.6961663, 0.4079426], [0.0047, 0.0135], k=2e-4),
    ],
    ids=lambda m: type(m).__name__,
)
def test_derivatives_match_central_differences(model):
    from dataclasses import replace

    d = model.derivatives(WL)
    assert d.shape == (len(model.parameters), *WL.shape)
    for i, p in enumerate(model.parameters):
        h = 1e-6

        def at(e, p=p):
            if isinstance(model, Constant):
                return replace(model, m=model.m + (e if p == "n" else 1j * e))
            if isinstance(model, Sellmeier) and p != "k":
                seq, j = p[0], int(p[1:])
                values = list(getattr(model, seq))
                values[j] += e
                return replace(model, **{seq: values})
            return replace(model, **{p: getattr(model, p) + e})

        fd = (at(h).evaluate(WL) - at(-h).evaluate(WL)) / (2 * h)
        np.testing.assert_allclose(d[i], fd, rtol=1e-7, atol=1e-9, err_msg=p)


def test_function_wraps_a_callable_and_checks_its_values():
    f = Function(lambda wl: 1.33 + 1e-3 * wl * 1j)
    np.testing.assert_allclose(f.evaluate(WL), 1.33 + 1e-3j * WL, rtol=1e-15)
    assert f.parameters == () and f.derivatives(WL).shape == (0, *WL.shape)
    assert f == f and f != Function(f.f)
    with pytest.raises(InvalidInputError, match=r"m = \(1\.5-0\.001j\) at wavelength 0\.5"):
        Function(lambda wl: np.full(wl.shape, 1.5 - 1e-3j)).evaluate([0.5, 0.6])
    with pytest.raises(InvalidInputError, match="returned shape"):
        Function(lambda wl: 1.5).evaluate([0.5, 0.6])
    with pytest.raises(InvalidInputError, match="not numbers"):
        Function(lambda wl: np.full(wl.shape, "x")).evaluate([0.5])
    with pytest.raises(InvalidInputError, match="finite"):
        Function(lambda wl: wl * np.nan).evaluate([0.5])
    with pytest.raises(TypeError, match="callable"):
        Function(1.5)
    # The callable gets a copy: changing it changes nothing.
    wl = np.array([0.5, 0.6])
    Function(lambda w: w.__setitem__(..., 0.0) or np.full(w.shape, 1.5)).evaluate(wl)
    np.testing.assert_array_equal(wl, [0.5, 0.6])


def test_sphere_converts_its_refractive_index():
    assert Sphere(1.5).refractive_index == Constant(1.5)
    assert Sphere(np.complex128(1.5 + 1e-3j)).refractive_index == Constant(1.5 + 1e-3j)
    assert Sphere(2).refractive_index == Constant(2.0)
    t = Table([0.4, 0.5], [1.5, 1.4])
    assert Sphere(t).refractive_index is t
    s = Sphere(lambda wl: np.full(wl.shape, 1.5))
    assert isinstance(s.refractive_index, Function)
    assert Sphere(1.5) == Sphere(1.5 + 0j) and hash(Sphere(1.5)) == hash(Sphere(1.5 + 0j))
    with pytest.raises(TypeError, match=r"refractive\.Table"):
        Sphere(np.array([1.5, 1.4]))
    with pytest.raises(TypeError, match=r"refractive\.Table"):
        Sphere([1.5, 1.4])
    with pytest.raises(TypeError, match="bool"):
        Sphere(True)
    with pytest.raises(TypeError, match="str"):
        Sphere("1.5")
    with pytest.raises(InvalidInputError, match="Im m ≥ 0"):
        Sphere(1.5 - 1e-3j)


def test_models_are_immutable():
    from dataclasses import FrozenInstanceError

    for model in (
        Constant(1.5),
        Table([0.4, 0.5], [1.5, 1.4]),
        Cauchy(1.4),
        Sellmeier([1.0], [0.01]),
    ):
        with pytest.raises(FrozenInstanceError):
            model.k = 0.1
    assert isinstance(Constant(1.5), refractive.RefractiveIndex)
