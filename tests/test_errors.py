import datetime
import math
import pickle

import numpy as np
import pytest

import scatrs
from scatrs import _scatrs, special


def test_invalid_n_raises_invalid_input():
    with pytest.raises(scatrs.InvalidInputError) as info:
        special.gauss_legendre(0)
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, scatrs.ScatrsError)


def test_negative_l_max_raises_invalid_input():
    with pytest.raises(scatrs.InvalidInputError):
        special.wigner_d(0, 0, -1, 0.5)


def test_x_outside_unit_interval_raises_from_rust():
    with pytest.raises(scatrs.InvalidInputError, match="-1 <= x <= 1"):
        special.wigner_d(0, 0, 3, [0.0, 1.5])


def test_not_converged_error_carries_accuracy():
    err = scatrs.NotConvergedError("not converged", achieved=1e-3, requested=1e-6)
    assert (err.achieved, err.requested) == (1e-3, 1e-6)
    assert isinstance(err, ArithmeticError)


# --- The exception hierarchy -------------------------------------------------------------------


def test_hierarchy():
    assert issubclass(scatrs.ScatrsError, Exception)
    for cls, std in (
        (scatrs.InvalidInputError, ValueError),
        (scatrs.NotConvergedError, ArithmeticError),
        (scatrs.OutsideValidatedDomainError, ValueError),
    ):
        assert issubclass(cls, scatrs.ScatrsError)
        assert issubclass(cls, std)
    assert scatrs.errors.ScatrsError is scatrs.ScatrsError


def test_not_converged_error_defaults_and_pickles():
    err = scatrs.NotConvergedError("not converged")
    assert (err.achieved, err.requested) == (None, None)
    err = scatrs.NotConvergedError("not converged", achieved=math.inf, requested=1e-6)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is scatrs.NotConvergedError
    assert (str(back), back.achieved, back.requested) == (str(err), math.inf, 1e-6)


# --- Mapping of scatrs::Error (Rust) onto the Python classes --------------------------------------


@pytest.mark.parametrize(
    ("variant", "cls", "message"),
    [
        ("InvalidInput", scatrs.InvalidInputError, "invalid input: bad"),
        (
            "OutsideValidatedDomain",
            scatrs.OutsideValidatedDomainError,
            "outside validated domain: bad",
        ),
    ],
)
def test_core_errors_map_to_python_classes(variant, cls, message):
    with pytest.raises(cls) as info:
        _scatrs._raise_core_error(variant, "bad")
    assert type(info.value) is cls
    assert str(info.value) == message


@pytest.mark.parametrize("achieved", [1e-3, math.inf])
def test_core_not_converged_maps_with_its_accuracies(achieved):
    with pytest.raises(scatrs.NotConvergedError) as info:
        _scatrs._raise_core_error("NotConverged", achieved=achieved, requested=2.0**-90)
    assert (info.value.achieved, info.value.requested) == (achieved, 2.0**-90)
    assert str(info.value).startswith("not converged: achieved ")


def test_unknown_core_error_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        _scatrs._raise_core_error("Nonsense")


# --- Robustness: integer arguments -------------------------------------------------------------

CAP = special.MAX_GAUSS_LEGENDRE_POINTS
VALUES = special.MAX_WIGNER_D_VALUES  # 2²⁸


@pytest.mark.parametrize("n", [0, -1, -(2**70), CAP + 1, 2**63, 2**64, 2**200])
def test_gauss_legendre_n_out_of_range_is_invalid_input(n):
    with pytest.raises(scatrs.InvalidInputError, match="Gauss-Legendre rule needs"):
        special.gauss_legendre(n)


@pytest.mark.parametrize(
    "n", [3.0, 3.5, math.nan, True, False, np.True_, np.float64(3), "3", None, 3 + 0j]
)
def test_gauss_legendre_non_integer_n_is_type_error(n):
    with pytest.raises(TypeError, match="n must be an integer"):
        special.gauss_legendre(n)


@pytest.mark.parametrize("name", ["m", "n", "l_max"])
@pytest.mark.parametrize("bad", [1.0, True, np.bool_(False), np.float32(1), "1", None])
def test_wigner_d_non_integer_orders_are_type_errors(name, bad):
    args = {"m": 1, "n": 0, "l_max": 3} | {name: bad}
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        special.wigner_d(args["m"], args["n"], args["l_max"], 0.5)


@pytest.mark.parametrize("name", ["m", "n"])
@pytest.mark.parametrize("bad", [2**31, -(2**31) - 1, 2**64, -(2**200)])
def test_wigner_d_orders_beyond_32_bits_are_overflow_errors(name, bad):
    args = {"m": 1, "n": 0} | {name: bad}
    with pytest.raises(OverflowError, match=f"{name} = "):
        special.wigner_d(args["m"], args["n"], 3, 0.5)


@pytest.mark.parametrize(
    ("l_max", "x"),
    [
        (-1, 0.5),
        (-(2**70), 0.5),
        (VALUES, 0.5),  # one row of 2²⁸ + 1 values
        (2**31, 0.5),
        (2**70, 0.5),
        (VALUES, []),  # bounded even when x is empty
        (VALUES // 2, [0.1, 0.2]),  # 2·(2²⁷ + 1) values
        (VALUES // 2**11, np.zeros(2**11)),  # 2¹¹·(2¹⁷ + 1) values
    ],
)
def test_wigner_d_l_max_and_output_size_are_bounded(l_max, x):
    with pytest.raises(scatrs.InvalidInputError):
        special.wigner_d(0, 0, l_max, x)


def test_wigner_d_size_bound_is_inclusive():
    # max(len(x), 1)·(l_max + 1) == 2²⁸ is accepted; with x empty this costs nothing.
    assert special.wigner_d(0, 0, VALUES - 1, []).shape == (0, VALUES)
    assert _scatrs.wigner_d(0, 0, VALUES - 1, np.zeros(0)).shape == (0, VALUES)
    with pytest.raises(scatrs.InvalidInputError, match=f"<= {VALUES}, got x.size = 0"):
        special.wigner_d(0, 0, VALUES, [])


# --- Robustness: x ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x",
    [math.nan, [0.0, math.nan], math.inf, -math.inf, 1.5, -1.0000000000000002, [[0.2], [7.0]]],
)
def test_wigner_d_x_outside_the_domain_is_invalid_input(x):
    with pytest.raises(scatrs.InvalidInputError, match="-1 <= x <= 1"):
        special.wigner_d(1, 1, 3, x)


@pytest.mark.parametrize(
    "x",
    [
        0.5 + 0j,
        np.array([0.5, 0.1], dtype=np.complex128),
        np.array([0.5], dtype=np.complex64),
        True,
        np.array([True, False]),
        "0.5",
        ["0.5"],
        [0.5, None],
        np.array([0.5], dtype=object),
        datetime.datetime(2026, 1, 1),
        np.array([1], dtype="timedelta64[s]"),
        2**100,  # no numpy integer type holds it
    ],
)
def test_wigner_d_non_real_x_is_type_error(x):
    with pytest.raises(TypeError, match="x must be real numbers"):
        special.wigner_d(1, 1, 3, x)


# --- Robustness: the extension module called directly --------------------------------------------
# scatrs._scatrs is private; it relies on the wrapper for conversions, but must still never crash.


def test_extension_rejects_out_of_range_integers():
    with pytest.raises(OverflowError):
        _scatrs.gauss_legendre(-1)
    with pytest.raises(OverflowError):
        _scatrs.gauss_legendre(2**64)
    with pytest.raises(OverflowError):
        _scatrs.wigner_d(2**31, 0, 3, np.zeros(1))
    with pytest.raises(OverflowError):
        _scatrs.wigner_d(0, 0, -1, np.zeros(1))
    with pytest.raises(TypeError):
        _scatrs.gauss_legendre(3.0)


def test_extension_checks_limits_itself():
    with pytest.raises(scatrs.InvalidInputError, match=str(CAP)):
        _scatrs.gauss_legendre(CAP + 1)
    with pytest.raises(scatrs.InvalidInputError):
        _scatrs.gauss_legendre(0)
    with pytest.raises(scatrs.InvalidInputError, match=f"<= {VALUES}, got x.size = 0"):
        _scatrs.wigner_d(0, 0, VALUES, np.zeros(0))
    with pytest.raises(scatrs.InvalidInputError, match=f"<= {VALUES}, got x.size = 3"):
        _scatrs.wigner_d(0, 0, VALUES // 2, np.zeros(3))
    with pytest.raises(scatrs.InvalidInputError):
        _scatrs.wigner_d(0, 0, 2**64 - 1, np.zeros(1))


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(4)[::2],  # not contiguous
        np.frombuffer(b"\0" * 17, dtype=np.float64, count=2, offset=1),  # misaligned
        np.zeros(2, dtype=np.dtype(np.float64).newbyteorder()),  # byte-swapped
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 2)),
        np.array(0.5),
        [0.5],
        0.5,
    ],
)
def test_extension_rejects_anything_but_aligned_contiguous_1d_float64(x):
    with pytest.raises(TypeError):
        _scatrs.wigner_d(0, 0, 3, x)
