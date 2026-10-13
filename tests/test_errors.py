import datetime
import math
import pickle
import subprocess
import sys
import types

import numpy as np
import pytest

import scatrs
from scatrs import _args, _scatrs, special
from scatrs._scatrs import _testing


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
        _testing.raise_core_error(variant, "bad")
    assert type(info.value) is cls
    assert str(info.value) == message


@pytest.mark.parametrize("achieved", [1e-3, math.inf])
def test_core_not_converged_maps_with_its_accuracies(achieved):
    with pytest.raises(scatrs.NotConvergedError) as info:
        _testing.raise_core_error("NotConverged", achieved=achieved, requested=2.0**-90)
    assert (info.value.achieved, info.value.requested) == (achieved, 2.0**-90)
    assert str(info.value).startswith("not converged: achieved ")


def test_unknown_core_error_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        _testing.raise_core_error("Nonsense")


# --- Robustness: integer arguments -------------------------------------------------------------

CAP = special.MAX_GAUSS_LEGENDRE_POINTS
USIZE_MAX = _args._USIZE_MAX  # 2**64 - 1 or 2**32 - 1, as the platform's usize
VALUES = special.MAX_WIGNER_D_VALUES  # 2²⁸


@pytest.mark.parametrize(
    ("n", "message"),
    [
        # Values the Rust core sees, and words the errors for:
        (0, "Gauss-Legendre rule needs n >= 1"),
        (CAP + 1, f"Gauss-Legendre rule needs n <= {CAP}, got {CAP + 1}"),
        (USIZE_MAX, f"Gauss-Legendre rule needs n <= {CAP}, got {USIZE_MAX}"),
        # Values that do not fit a usize, rejected while converting:
        (-1, "n must not be negative, got -1"),
        (-(2**70), "n must not be negative"),
        (USIZE_MAX + 1, f"n must be at most {USIZE_MAX}, got {USIZE_MAX + 1}"),
        (2**200, "n must be at most"),
    ],
)
def test_gauss_legendre_n_out_of_range_is_invalid_input(n, message):
    with pytest.raises(scatrs.InvalidInputError, match=f"^invalid input: {message}"):
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
        (-1, 0.5),  # rejected while converting, as is 2**70 below
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


# Run in a fresh interpreter: it caps its own address space at 1 GiB above what it already uses,
# then asks for a full 2 GiB output (2²⁸ values, the most MAX_WIGNER_D_VALUES allows).
_OUTPUT_ALLOCATION_FAILS = """
import resource

import numpy as np

from scatrs import special

small = special.wigner_d(0, 0, 3, [0.25, 0.5])
with open("/proc/self/status", encoding="ascii") as status:
    in_use = next(int(ln.split()[1]) * 1024 for ln in status if ln.startswith("VmSize:"))
_, hard = resource.getrlimit(resource.RLIMIT_AS)
cap = in_use + 2**30
if hard != resource.RLIM_INFINITY:
    cap = min(cap, hard)
resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
try:
    special.wigner_d(0, 0, special.MAX_WIGNER_D_VALUES - 1, [0.5])
except MemoryError as e:
    print(f"MemoryError: {e}")
print("intact:", np.array_equal(special.wigner_d(0, 0, 3, [0.25, 0.5]), small))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux: RLIMIT_AS and /proc")
def test_failed_output_allocation_is_memory_error_not_an_abort():
    # A request within the size bound can still fail to allocate. It must raise MemoryError,
    # leaving the interpreter usable, rather than abort the process.
    run = subprocess.run(
        [sys.executable, "-c", _OUTPUT_ALLOCATION_FAILS],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert len(lines) == 2, run.stdout
    assert lines[0].startswith(
        f"MemoryError: wigner_d cannot allocate {VALUES} float64 values for its output: "
    ), run.stdout
    assert lines[1] == "intact: True"


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


# --- Robustness: arrays borrowed elsewhere, and failures while raising ---------------------------


def test_x_borrowed_for_writing_by_another_extension_is_buffer_error():
    # rust-numpy tracks borrows across all extensions built on it; with_write_borrow holds a
    # writeable one, as another such extension could, while the callback runs.
    x = np.linspace(-1, 1, 6)
    for call in (
        lambda: special.wigner_d(0, 0, 3, x),
        lambda: special.wigner_d(0, 0, 3, x.reshape(2, 3)),
        lambda: _scatrs.wigner_d(0, 0, 3, x),
    ):
        with pytest.raises(BufferError, match="x cannot be read: The given array is already"):
            _testing.with_write_borrow(x, call)
    with pytest.raises(BufferError):  # a borrow of an overlapping view conflicts too
        _testing.with_write_borrow(x[2:4], lambda: special.wigner_d(0, 0, 3, x))
    # Disjoint data does not conflict, and the borrow ends with the call.
    assert _testing.with_write_borrow(x[:2], lambda: special.wigner_d(0, 0, 3, x[3:])).shape == (
        3,
        4,
    )
    np.testing.assert_array_equal(special.wigner_d(0, 0, 3, x)[:, 0], 1.0)


def test_error_mapping_without_scatrs_errors_falls_back_to_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "scatrs.errors", None)  # importing it now fails
    with pytest.raises(RuntimeError, match=r"^invalid input: bad$") as info:
        _testing.raise_core_error("InvalidInput", "bad")
    assert isinstance(info.value.__cause__, ImportError)
    with pytest.raises(RuntimeError, match=r"^not converged: achieved inf") as info:
        _testing.raise_core_error("NotConverged", achieved=math.inf, requested=1e-6)
    assert isinstance(info.value.__cause__, ImportError)
    with pytest.raises(RuntimeError, match="-1 <= x <= 1"):
        special.wigner_d(0, 0, 3, [math.nan])


class _InterruptingModule(types.ModuleType):
    def __getattr__(self, name):
        raise KeyboardInterrupt


def test_error_mapping_passes_non_exceptions_through(monkeypatch):
    monkeypatch.setitem(sys.modules, "scatrs.errors", _InterruptingModule("scatrs.errors"))
    with pytest.raises(KeyboardInterrupt):
        _testing.raise_core_error("InvalidInput", "bad")


def test_extension_rejects_out_of_range_integers():
    with pytest.raises(OverflowError):
        _scatrs.gauss_legendre(-1)
    with pytest.raises(OverflowError):
        _scatrs.gauss_legendre(USIZE_MAX + 1)
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
        _scatrs.wigner_d(0, 0, USIZE_MAX, np.zeros(1))


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
