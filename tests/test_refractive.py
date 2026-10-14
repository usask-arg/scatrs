"""scatrs.refractive models and scatrs.Sphere."""

import re
import warnings

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
    assert Constant(np.float32(1.5)) == Constant(2 - 0.5) == Constant(np.complex64(1.5))


def test_constant_errors():
    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Constant: m must be finite with Im m ≥ 0, got \(1\.5-0\.001j\)$",
    ):
        Constant(1.5 - 1e-3j)
    with pytest.raises(InvalidInputError, match=r"^invalid input: Constant: m must be finite"):
        Constant(complex(np.nan, 0))
    for bad in (True, np.bool_(False), "1.5", b"1.5", None, [1.5]):
        with pytest.raises(
            TypeError, match=f"Constant: m must be a number, not {type(bad).__name__}"
        ):
            Constant(bad)


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
    with pytest.raises(InvalidInputError, match="finite and positive, got nan"):
        t.evaluate(np.nan)
    with pytest.raises(InvalidInputError, match=r"^invalid input: a Table's wavelengths must be"):
        Table([0.0, 0.5], [1.5, 1.4])
    with pytest.raises(InvalidInputError, match=r"^invalid input: a Table's wavelengths must be"):
        Table([-0.5, 0.5], [1.5, 1.4])
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
    with pytest.raises(InvalidInputError, match="≥ 0"):
        Table([0.4, 0.5], [1.5, 1.4 - 5e-324j])
    Table([0.4, 0.5], [1.5 + 5e-324j, 1.4 + 1e-3j], interpolation="log_k")
    with pytest.raises(InvalidInputError, match="> 0 for log_k"):
        Table([0.4, 0.5], [1.5, 1.4 + 1e-3j], interpolation="log_k")
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: a Table's interpolation must be 'linear' or"
    ):
        Table([0.4, 0.5], [1.5, 1.4], interpolation="cubic")


def test_cauchy():
    c = Cauchy(1.45, 0.004, 1e-4, k=1e-5)
    np.testing.assert_allclose(
        c.evaluate(WL), 1.45 + 0.004 / WL**2 + 1e-4 / WL**4 + 1e-5j, rtol=1e-15
    )
    assert c.parameters == ("A", "B", "C", "k")
    assert Cauchy(np.float32(1.5), 1, np.int64(0)) == Cauchy(1.5, 1.0)


def test_cauchy_errors():
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Cauchy: k must not be negative, got -0\.001$"
    ):
        Cauchy(1.45, k=-1e-3)
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Cauchy: A must be finite, got nan$"
    ):
        Cauchy(np.nan)
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Cauchy: B must be finite, got nan$"
    ):
        Cauchy(1.4, np.nan)
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Cauchy: k must be finite, got inf$"
    ):
        Cauchy(1.4, k=np.inf)
    for bad in (True, np.bool_(True), "1.4", None, 1.4 + 0j):
        with pytest.raises(
            TypeError, match=f"Cauchy: A must be a real number, not {type(bad).__name__}"
        ):
            Cauchy(bad)
    with pytest.raises(TypeError, match="Cauchy: k must be a real number, not bool"):
        Cauchy(1.4, k=False)


def test_cauchy_at_zero_wavelength_raises_without_warnings():
    c = Cauchy(1.4, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InvalidInputError, match=r"finite and positive, got 0\.0$"):
            c.evaluate(0.0)
        with pytest.raises(InvalidInputError, match=r"finite and positive, got 0\.0$"):
            Cauchy(1.4, 0.01, 1e-4).evaluate([0.5, 0.0])
        with pytest.raises(InvalidInputError, match=r"finite and positive, got 0\.0$"):
            c.derivatives(0.0)
        # Tiny but positive: λ⁻⁴ overflows, and evaluate() reports it.
        with pytest.raises(InvalidInputError, match=r"at wavelength 1e-200"):
            Cauchy(1.4, 0.01, 1e-4).evaluate(1e-200)
        with pytest.raises(
            InvalidInputError, match=r"Cauchy.*∂m/∂B = \(inf\+0j\) at wavelength 1e-200"
        ):
            Cauchy(1.4, 0.01, 1e-4).derivatives(1e-200)


_MODELS = [
    Constant(1.5),
    Table([0.4, 0.8], [1.5, 1.4]),
    Cauchy(1.4, 0.01),
    Sellmeier([1.0], [0.01]),
    Function(lambda wl: np.full(wl.shape, 1.5)),
]


@pytest.mark.parametrize("model", _MODELS, ids=lambda m: type(m).__name__)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.5, 0.0, -0.0])
def test_wavelengths_must_be_finite_and_positive(model, bad):
    msg = rf"^invalid input: wavelengths must be finite and positive, got {re.escape(str(bad))}$"
    with pytest.raises(InvalidInputError, match=msg):
        model.evaluate(bad)
    with pytest.raises(InvalidInputError, match=msg):
        model.evaluate([[0.5, 0.6], [bad, -1.0]])  # the first bad one is named
    with pytest.raises(InvalidInputError, match=msg):
        model.derivatives([0.5, bad])


def test_derivatives_are_checked():
    class Toy(refractive.RefractiveIndex):
        """Names a parameter but does not give its derivative."""

        def _m(self, wavelength):
            return np.full(wavelength.shape, 1.5)

        @property
        def parameters(self):
            return ("x",)

        def __repr__(self):
            return "Toy()"

    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Toy\(\) returned derivatives of shape \(0, 2\) for 1 "
        r"parameters and wavelengths of shape \(2,\); expected \(1, 2\)$",
    ):
        Toy().derivatives([0.5, 0.6])

    class NanToy(Toy):
        def _dm(self, wavelength):
            out = np.ones((1, *wavelength.shape), dtype=complex)
            out[0, 1] = np.nan
            return out

    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Toy\(\) gives ∂m/∂x = \(nan\+0j\) at wavelength 0\.6; "
        "derivatives must be finite$",
    ):
        NanToy().derivatives([0.5, 0.6, 0.7])
    with pytest.raises(InvalidInputError, match="n² must be positive"):
        Sellmeier([1.0], [0.25]).derivatives([0.6, 0.5])  # at the pole


def test_sellmeier_at_huge_wavelengths_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for wl in (1e200, 1e150):
            for f in (Sellmeier([1.0], [0.25]).evaluate, Sellmeier([1.0], [0.25]).derivatives):
                try:
                    f(wl)
                except InvalidInputError:
                    pass


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
    assert Sellmeier(np.array([1.0]), (np.float32(0.25),)) == Sellmeier([1], [0.25])


def test_sellmeier_errors():
    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Sellmeier: B and C must have the same length, at least 1; "
        r"got 2 and 1$",
    ):
        Sellmeier([1.0, 2.0], [0.25])
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Sellmeier: B and C .* got 0 and 0$"
    ):
        Sellmeier([], [])
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Sellmeier: B\[1\] must be finite"
    ):
        Sellmeier([1.0, np.nan], [0.1, 0.2])
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Sellmeier: C\[0\] must be finite"
    ):
        Sellmeier([1.0], [np.inf])
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Sellmeier: k must not be negative"
    ):
        Sellmeier([1.0], [0.1], k=-1.0)
    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Sellmeier: B must be a one-dimensional sequence, got shape \(\)$",
    ):
        Sellmeier(1.0, [0.1])
    with pytest.raises(
        InvalidInputError,
        match=r"^invalid input: Sellmeier: C must be a one-dimensional sequence, got shape "
        r"\(1, 1\)$",
    ):
        Sellmeier([1.0], np.array([[0.1]]))
    with pytest.raises(
        InvalidInputError, match=r"^invalid input: Sellmeier: B .* got a ragged sequence$"
    ):
        Sellmeier([[1.0], [2.0, 3.0]], [0.1])  # ragged
    with pytest.raises(TypeError, match=r"Sellmeier: B\[0\] must be a real number, not str"):
        Sellmeier(["1.0"], [0.1])
    with pytest.raises(TypeError, match=r"Sellmeier: B\[1\] must be a real number, not bool"):
        Sellmeier([1.0, True], [0.1, 0.2])
    with pytest.raises(TypeError, match=r"Sellmeier: C\[0\] must be a real number, not bool"):
        Sellmeier([1.0], np.array([True]))
    with pytest.raises(TypeError, match="Sellmeier: B must be a sequence of real numbers, not str"):
        Sellmeier("1.0", [0.1])


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
    with pytest.raises(TypeError, match=r"^Function: f must be callable, not float$"):
        Function(1.5)
    # The callable gets a copy: changing it changes nothing.
    wl = np.array([0.5, 0.6])
    Function(lambda w: w.__setitem__(..., 0.0) or np.full(w.shape, 1.5)).evaluate(wl)
    np.testing.assert_array_equal(wl, [0.5, 0.6])


def test_sphere_converts_its_refractive_index():
    assert Sphere(1.5).refractive_index == Constant(1.5)
    assert Sphere(np.complex128(1.5 + 1e-3j)).refractive_index == Constant(1.5 + 1e-3j)
    assert Sphere(2).refractive_index == Constant(2.0)
    assert Sphere(np.array(1.5)).refractive_index == Constant(1.5)
    assert Sphere(np.array(1.5 + 1e-3j)).refractive_index == Constant(1.5 + 1e-3j)
    assert type(Sphere(np.array(2)).refractive_index.m) is complex
    t = Table([0.4, 0.5], [1.5, 1.4])
    assert Sphere(t).refractive_index is t
    s = Sphere(lambda wl: np.full(wl.shape, 1.5))
    assert isinstance(s.refractive_index, Function)
    assert Sphere(1.5) == Sphere(1.5 + 0j) and hash(Sphere(1.5)) == hash(Sphere(1.5 + 0j))
    with pytest.raises(TypeError, match=r"refractive\.Table"):
        Sphere(np.array([1.5, 1.4]))
    with pytest.raises(TypeError, match=r"refractive\.Table"):
        Sphere([1.5, 1.4])
    accepted = r"a number, a scatrs\.refractive model or a callable, not"
    for m in (True, np.bool_(True), np.array(True)):
        with pytest.raises(TypeError, match=rf"{accepted} a bool$"):
            Sphere(m)
    with pytest.raises(TypeError, match=rf"{accepted} str_$"):
        Sphere(np.array("1.5"))
    with pytest.raises(TypeError, match="str"):
        Sphere("1.5")
    with pytest.raises(InvalidInputError, match="Im m ≥ 0"):
        Sphere(1.5 - 1e-3j)
    for cls in (Constant, Function, refractive.RefractiveIndex, float):
        with pytest.raises(TypeError, match=rf"not the class {cls.__name__}; instantiate"):
            Sphere(cls)


def test_models_are_immutable():
    from dataclasses import FrozenInstanceError

    for model, field in (
        (Constant(1.5), "m"),
        (Table([0.4, 0.5], [1.5, 1.4]), "wavelength"),
        (Cauchy(1.4), "k"),
        (Sellmeier([1.0], [0.01]), "k"),
        (Function(np.sqrt), "f"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(model, field, getattr(model, field))
    assert isinstance(Constant(1.5), refractive.RefractiveIndex)
