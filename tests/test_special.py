import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scatrs import _args, _scatrs, special


def test_gauss_legendre_matches_numpy():
    x, w = special.gauss_legendre(7)
    xr, wr = np.polynomial.legendre.leggauss(7)
    np.testing.assert_allclose(x, xr, rtol=0, atol=1e-15)
    np.testing.assert_allclose(w, wr, rtol=1e-14)


def test_gauss_legendre_integrates_polynomials_exactly():
    x, w = special.gauss_legendre(257)
    for k in range(0, 40, 2):
        assert abs(np.sum(w * x**k) - 2 / (k + 1)) < 1e-14


def test_gauss_legendre_accepts_numpy_integers():
    expected = special.gauss_legendre(7)
    for n in (np.int64(7), np.uint8(7), np.intp(7)):
        got = special.gauss_legendre(n)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


def test_wigner_d_00_is_legendre():
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    d = special.wigner_d(0, 0, 10, x)
    assert d.shape == (3, 4, 11)
    for ell in range(11):
        expected = np.polynomial.legendre.Legendre.basis(ell)(x)
        np.testing.assert_allclose(d[..., ell], expected, atol=1e-14)


def test_wigner_d_22_closed_form():
    x = np.array([-0.3, 0.2, 0.9])
    d = special.wigner_d(2, 2, 2, x)
    np.testing.assert_allclose(d[:, 2], ((1 + x) / 2) ** 2, rtol=1e-15)
    np.testing.assert_array_equal(d[:, :2], 0.0)


def test_wigner_d_accepts_scalars_and_non_contiguous_arrays():
    x = np.linspace(-1, 1, 10)[::3]
    np.testing.assert_allclose(special.wigner_d(1, -1, 4, x), special.wigner_d(1, -1, 4, x.copy()))
    assert special.wigner_d(0, 0, 3, 0.5).shape == (4,)


def test_wigner_d_scalar_forms_agree():
    expected = special.wigner_d(1, 0, 3, np.array([0.5]))[0]
    for x in (0.5, np.float64(0.5), np.array(0.5), np.float32(0.5)):
        got = special.wigner_d(1, 0, 3, x)
        assert got.shape == (4,)
        np.testing.assert_array_equal(got, expected)
    # Integer x is exact in float64.
    np.testing.assert_array_equal(special.wigner_d(2, 1, 5, 1), special.wigner_d(2, 1, 5, 1.0))


def test_wigner_d_converts_other_real_layouts_and_dtypes():
    x64 = np.array([[-1.0, -0.25, 0.0], [0.5, 0.75, 1.0]])
    expected = special.wigner_d(2, -1, 6, x64)
    byte_swapped = x64.astype(x64.dtype.newbyteorder())
    fortran = np.asfortranarray(x64)
    read_only = x64.copy()
    read_only.flags.writeable = False
    # A float64 array whose data starts one byte into a buffer is misaligned.
    misaligned = np.frombuffer(b"\0" + x64.tobytes(), dtype=np.float64, count=6, offset=1)
    misaligned = misaligned.reshape(2, 3)
    assert not misaligned.flags.aligned
    for x in (
        byte_swapped,
        fortran,
        read_only,
        misaligned,
        x64.tolist(),
        x64.astype(np.float32),  # these values are exact in float32 and float16
        x64.astype(np.float16),
        x64.astype(np.longdouble),
    ):
        np.testing.assert_array_equal(special.wigner_d(2, -1, 6, x), expected)
    integers = np.array([[-1, 0], [0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(
        special.wigner_d(2, -1, 6, integers), special.wigner_d(2, -1, 6, integers.astype(float))
    )


def test_wigner_d_empty_x():
    assert special.wigner_d(0, 0, 5, []).shape == (0, 6)
    assert special.wigner_d(0, 0, 5, np.empty((2, 0))).shape == (2, 0, 6)
    assert special.wigner_d(0, 0, 5, np.empty((0, 3), dtype=np.int32)).shape == (0, 3, 6)


def test_wigner_d_accepts_every_32_bit_order():
    # Below l = max(|m|, |n|) every value is zero, without any work.
    for m, n in ((-(2**31), 0), (2**31 - 1, 5), (3, -(2**31))):
        np.testing.assert_array_equal(special.wigner_d(m, n, 4, [0.1, -1.0]), 0.0)
    np.testing.assert_array_equal(
        special.wigner_d(np.int16(2), np.int64(-1), np.uint32(6), 0.3),
        special.wigner_d(2, -1, 6, 0.3),
    )


# --- Zero-copy outputs and inputs --------------------------------------------------------------


def _root(a):
    """The array at the root of a chain of numpy views, and the object owning its memory."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a, a.base


def _assert_owned_by_rust(a):
    root, owner = _root(a)
    # rust-numpy's from_vec/into_pyarray hand the Vec to numpy without copying; the Vec is
    # freed when the array is. PySliceContainer is rust-numpy's internal owner class, so this
    # may need updating when rust-numpy is upgraded.
    assert type(owner).__name__ == "PySliceContainer"
    assert np.shares_memory(a, root)
    assert a.dtype == np.float64
    assert a.flags.c_contiguous
    assert a.flags.writeable


def test_gauss_legendre_outputs_are_rust_vectors_handed_over_without_copying():
    for a in special.gauss_legendre(9):
        assert a.base is not None and not isinstance(a.base, np.ndarray)
        _assert_owned_by_rust(a)


def test_wigner_d_output_is_a_rust_vector_handed_over_without_copying():
    raw = _scatrs.wigner_d(1, 0, 4, np.linspace(-1, 1, 6))
    assert raw.shape == (6, 5)
    assert not isinstance(raw.base, np.ndarray)
    _assert_owned_by_rust(raw)
    out = special.wigner_d(1, 0, 4, np.linspace(-1, 1, 6).reshape(2, 3))
    assert out.shape == (2, 3, 5)
    _assert_owned_by_rust(out)  # a reshaped view of the Rust array, not a copy


class _Spy:
    """Stands in for the extension module and records the x arrays handed to it."""

    def __init__(self):
        self.seen = []

    def __getattr__(self, name):
        return getattr(_scatrs, name)

    def wigner_d(self, m, n, l_max, x):
        self.seen.append(x)
        return _scatrs.wigner_d(m, n, l_max, x)


def test_wigner_d_passes_contiguous_float64_input_without_copying(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(special, "_scatrs", spy)
    for x in (np.linspace(-1, 1, 12), np.linspace(-1, 1, 12).reshape(3, 4), np.array(0.5)):
        special.wigner_d(1, 0, 5, x)
        assert spy.seen[-1].ctypes.data == x.ctypes.data
        assert np.shares_memory(spy.seen[-1], x)
        # A fresh view, never x itself (see scatrs._args.real_array).
        assert spy.seen[-1] is not x
    # Anything else is converted to contiguous float64 once.
    for x in (np.linspace(-1, 1, 12)[::2], np.linspace(-1, 1, 12).astype(np.float32)):
        special.wigner_d(1, 0, 5, x)
        assert not np.shares_memory(spy.seen[-1], x)
        assert spy.seen[-1].dtype == np.float64 and spy.seen[-1].flags.c_contiguous


def test_real_array_returns_qualifying_arrays_as_they_are():
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    assert _args.real_array(x, "x") is x
    for other in (x.T, x.astype(np.float32), x.astype(x.dtype.newbyteorder()), x.tolist()):
        converted = _args.real_array(other, "x")
        assert converted.dtype == np.float64 and converted.dtype.isnative
        assert converted.flags.c_contiguous and converted.flags.aligned
        np.testing.assert_array_equal(converted, np.asarray(other, dtype=np.float64))


# --- Threads and the GIL -----------------------------------------------------------------------


def test_calls_from_threads_agree():
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: special.gauss_legendre(300)[1], range(8)))
    for r in results[1:]:
        np.testing.assert_array_equal(r, results[0])


def test_threaded_results_equal_serial_results():
    x = np.linspace(-1, 1, 101)
    serial_gl = special.gauss_legendre(200)
    serial_wd = special.wigner_d(3, -2, 300, x)
    with ThreadPoolExecutor(4) as pool:
        gls = list(pool.map(lambda _: special.gauss_legendre(200), range(8)))
        wds = list(pool.map(lambda _: special.wigner_d(3, -2, 300, x), range(8)))
    for nodes, weights in gls:
        np.testing.assert_array_equal(nodes, serial_gl[0])
        np.testing.assert_array_equal(weights, serial_gl[1])
    for wd in wds:
        np.testing.assert_array_equal(wd, serial_wd)


def _main_thread_stamps_during(call):
    """Runs ``call`` in a worker thread while the main thread keeps taking timestamps.

    Returns the number of timestamps that fall in the middle half of the call and the call's
    duration. Taking a timestamp needs the GIL, so if the call held it, none could fall there:
    the worker enters the call within a few bytecodes of its start time (well under the 5 ms
    switch interval), and leaves it just before its end time.
    """
    span = {}

    def worker():
        span["start"] = time.perf_counter()
        call()
        span["end"] = time.perf_counter()

    stamps = []
    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        time.sleep(0.0005)
        stamps.append(time.perf_counter())
    thread.join()
    start, end = span["start"], span["end"]
    quarter = (end - start) / 4
    return sum(start + quarter < t < end - quarter for t in stamps), end - start


@pytest.mark.parametrize(
    "call",
    [
        # The largest rule three times: about 0.12 s on an Apple M4 Pro, longer elsewhere.
        pytest.param(
            lambda: [special.gauss_legendre(special.MAX_GAUSS_LEGENDRE_POINTS) for _ in range(3)],
            id="gauss_legendre",
        ),
        # About 2·10⁷ values (160 MB) in the general difference form, the slowest per value.
        pytest.param(
            lambda: special.wigner_d(0, 2, 19_999, np.linspace(0.95, 1, 1000)), id="wigner_d"
        ),
    ],
)
def test_gil_is_released_while_computing(call):
    inside, duration = _main_thread_stamps_during(call)
    assert duration > 0.02, "the call is too short for this test to discriminate"
    assert inside >= 5, f"main thread ran {inside} times during a {duration:.3f} s call"
