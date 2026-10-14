import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scatrs import InvalidInputError, OutsideValidatedDomainError, _args, _scatrs, mie
from scatrs._scatrs import _testing

# Each Mie function with x and m only; amplitudes at one angle.
CALLS = {
    "efficiencies": lambda x, m, **kw: mie.efficiencies(x, m, **kw),
    "coefficients": lambda x, m, **kw: mie.coefficients(x, m, **kw),
    "amplitudes": lambda x, m, **kw: mie.amplitudes(x, m, 0.5, **kw),
    "phase_matrix": lambda x, m, **kw: mie.phase_matrix(x, m, 0.5, **kw),
    "greek": lambda x, m, **kw: mie.greek(x, m, 4, **kw),
}


def test_wiscombe_published_values():
    # Wiscombe (1979), x = 1000, m = 0.75: Q_sca and S(0) (conjugated: MIEV0 uses e^(+iwt)).
    q = mie.efficiencies(1000.0, 0.75)
    assert abs(q.q_sca - 1.997908) < 1e-6
    assert q.q_abs == 0.0 and q.q_ext == q.q_sca
    s = mie.amplitudes(1000.0, 0.75, 1.0)
    assert abs(s.s1 - (4.994770e5 + 1.336502e4j)) < 1e-6 * abs(s.s1)


def test_shapes_broadcast():
    x = np.array([[0.5], [5.0], [50.0]])
    m = np.array([1.33, 1.5 + 0.01j])
    q = mie.efficiencies(x, m)
    for f in (q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g):
        assert f.shape == (3, 2) and f.dtype == np.float64
    c = mie.coefficients(x, m)
    assert c.n_terms.shape == (3, 2)
    assert c.a.shape == (3, 2, c.n_terms.max()) and c.a.dtype == np.complex128
    mu = np.linspace(-1, 1, 12).reshape(3, 4)
    s = mie.amplitudes(x, m, mu)
    assert s.s1.shape == (3, 2, 3, 4)
    # Scalars give scalars (0-d arrays).
    assert mie.efficiencies(1.0, 1.5).q_ext.shape == ()


def test_elements_equal_scalar_calls():
    x = np.array([0.1, 3.0, 300.0])
    m = np.array([1.33 + 1e-8j, 2.5 + 1j, 0.2 + 3j])
    q = mie.efficiencies(x, m)
    for i in range(3):
        qi = mie.efficiencies(x[i], m[i])
        assert q.q_ext[i] == qi.q_ext and q.g[i] == qi.g


def test_coefficients_are_zero_padded():
    c = mie.coefficients([0.1, 20.0], 1.5)
    n0, n1 = c.n_terms
    assert n0 < n1 == c.a.shape[-1]
    assert np.all(c.a[0, n0:] == 0) and np.all(c.b[0, n0:] == 0)
    assert np.all(c.a[0, :n0] != 0)
    np.testing.assert_array_equal(c.n_terms, mie.default_n_max([0.1, 20.0]))


def test_default_n_max():
    np.testing.assert_array_equal(mie.default_n_max([1e-6, 0.1, 1.0, 1000.0]), [4, 6, 10, 1063])
    for bad in (np.nan, np.inf, -1.0, 0.0, 2e6):
        with pytest.raises(InvalidInputError, match="element 1: the size parameter"):
            mie.default_n_max([1.0, bad])


def test_n_max_and_its_domain():
    c = mie.coefficients(10.0, 1.5, n_max=60)
    assert c.a.shape == (60,)
    with pytest.raises(OutsideValidatedDomainError):
        mie.efficiencies(10.0, 1.5, n_max=5)
    mie.efficiencies(10.0, 1.5, n_max=5, allow_unvalidated=True)


def test_optical_theorem_and_symmetries():
    for x, m in [(0.3, 1.33), (25.0, 1.5 + 0.1j), (700.0, 0.2 + 3j)]:
        q = mie.efficiencies(x, m)
        s = mie.amplitudes(x, m, [1.0, -1.0])
        assert abs(4 / x**2 * s.s1[0].real / q.q_ext - 1) < 1e-12
        assert s.s1[0] == s.s2[0] and s.s1[1] == -s.s2[1]
        assert abs(4 * abs(s.s1[1]) ** 2 / x**2 / q.q_back - 1) < 1e-11


def _fd(f, h):
    return (f(+h) - f(-h)) / (2 * h)


@pytest.mark.parametrize(("x", "m"), [(0.7, 1.5 + 0.01j), (8.0, 1.33 + 1e-4j), (40.0, 2.5 + 1j)])
def test_derivatives_match_finite_differences(x, m):
    q = mie.efficiencies(x, m, derivatives=True)
    # One term more than the default, so that x ± h keeps the same number of terms.
    n = int(mie.default_n_max(x)) + 1
    c = mie.coefficients(x, m, n_max=n, derivatives=True)
    s = mie.amplitudes(x, m, [0.9, 0.0, -0.5], derivatives=True)
    hx, hm = 1e-6 * x, 1e-7
    steps = {
        "x": (lambda h: (x + h, m), hx),
        "m_real": (lambda h: (x, m + h), hm),
        "m_imag": (lambda h: (x, m + 1j * h), hm),
    }
    for p, (at, h) in steps.items():
        for name in ("q_ext", "q_sca", "q_abs", "q_back", "g"):
            fd = _fd(lambda e, at=at, name=name: getattr(mie.efficiencies(*at(e)), name), h)
            got = getattr(q.d(p), name)
            assert abs(got - fd) <= 1e-6 * max(abs(fd), abs(getattr(q, name))), (p, name)
        for name in ("a", "b"):
            fd_c = _fd(
                lambda e, at=at, name=name: getattr(mie.coefficients(*at(e), n_max=n), name), h
            )
            got = getattr(c.d(p), name)
            np.testing.assert_allclose(got, fd_c, rtol=0, atol=1e-6 * np.abs(fd_c).max())
        for name in ("s1", "s2"):
            fd_s = _fd(
                lambda e, at=at, name=name: getattr(mie.amplitudes(*at(e), [0.9, 0.0, -0.5]), name),
                h,
            )
            got = getattr(s.d(p), name)
            np.testing.assert_allclose(got, fd_s, rtol=0, atol=1e-6 * np.abs(fd_s).max())
    # The complex derivative: d/dm = ∂/∂(Re m), and ∂/∂(Im m) = i d/dm.
    np.testing.assert_array_equal(c.d("m").a, c.d("m_real").a)
    np.testing.assert_array_equal(c.d("m_imag").a, 1j * c.d("m").a)
    np.testing.assert_array_equal(s.d("m_imag").s2, 1j * s.d("m").s2)


def test_derivative_subsets():
    q = mie.efficiencies(5.0, 1.5 + 0.1j, derivatives=["m_imag"])
    assert q.params == ("m_imag",)
    q.d("m_imag")
    with pytest.raises(ValueError, match="no derivative"):
        q.d("x")
    with pytest.raises(ValueError, match="unknown parameter"):
        mie.efficiencies(5.0, 1.5, derivatives=["radius"])
    with pytest.raises(TypeError):
        mie.efficiencies(5.0, 1.5, derivatives="x")
    with pytest.raises(ValueError, match="no derivative"):
        mie.efficiencies(5.0, 1.5).d("x")
    c = mie.coefficients(5.0, 1.5 + 0.1j, derivatives=["m_real"])
    c.d("m")  # the complex derivative is available with either m parameter
    with pytest.raises(ValueError, match="efficiencies are real"):
        mie.efficiencies(5.0, 1.5, derivatives=True).d("m")
    assert mie.efficiencies(5.0, 1.5, derivatives=np.True_).params == mie.PARAMETERS
    assert mie.efficiencies(5.0, 1.5, derivatives=np.False_).params == ()
    for bad in (None, 1, 1.0, b"x"):
        with pytest.raises(TypeError, match="derivatives must be a bool or a sequence"):
            mie.efficiencies(5.0, 1.5, derivatives=bad)
    for bad in (None, 1, "no"):
        with pytest.raises(TypeError, match="allow_unvalidated must be a bool"):
            mie.efficiencies(5.0, 1.5, allow_unvalidated=bad)
    mie.efficiencies(3e4, 1.5, allow_unvalidated=np.True_)


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
@pytest.mark.parametrize(
    ("x", "m"),
    [(-1.0, 1.5), (0.0, 1.5), (np.nan, 1.5), (1e-31, 1.5), (1e7, 1.5), (1.0, 1.5 - 1e-9j),
     (1.0, -1.5), (1.0, np.nan), (1.0, 0.0), (1.0, 2e3)],
)  # fmt: skip
def test_invalid_inputs(call, x, m):
    with pytest.raises(InvalidInputError):
        call(x, m, allow_unvalidated=True)


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_errors_name_the_element(call):
    with pytest.raises(InvalidInputError, match="element 2: the size parameter"):
        call([1.0, 2.0, -3.0], 1.5)
    with pytest.raises(InvalidInputError, match="element 1: the size parameter"):
        call([1.0, np.inf], 1.5)
    with pytest.raises(InvalidInputError, match="element 3: the refractive index"):
        call(np.ones((2, 2)), [[1.5, 1.5], [1.5, -1.5]])
    with pytest.raises(OutsideValidatedDomainError, match="element 1"):
        call([1.0, 3e4], 1.5)
    with pytest.raises(InvalidInputError, match=r"number of terms must be in \[1, 1048576\]"):
        call(1.0, 1.5, n_max=2**40)


@pytest.mark.parametrize(
    ("x", "m"),
    [(3e4, 1.5), (1.0, 11.0), (1.0, 1.5 + 11j), (1.0, 2j), (1.0, 0.05), (1.0, 0.01 + 0.01j)],
)
def test_outside_the_domain(x, m):
    with pytest.raises(OutsideValidatedDomainError):
        mie.efficiencies(x, m)
    assert np.isfinite(mie.efficiencies(x, m, allow_unvalidated=True).q_ext)


def test_type_errors():
    with pytest.raises(TypeError):
        mie.efficiencies(1.0 + 0j, 1.5)
    with pytest.raises(TypeError):
        mie.efficiencies(1.0, "1.5")
    with pytest.raises(ValueError):
        mie.efficiencies([1.0, 2.0], [1.5, 1.6, 1.7])  # shapes do not broadcast
    with pytest.raises(InvalidInputError):
        mie.amplitudes(1.0, 1.5, [1.5])


def test_empty_inputs():
    q = mie.efficiencies(np.zeros((0, 3)), 1.5)
    assert q.q_ext.shape == (0, 3)
    c = mie.coefficients(np.zeros(0), 1.5)
    assert c.a.shape == (0, 0)
    s = mie.amplitudes([1.0, 2.0], 1.5, np.zeros(0))
    assert s.s1.shape == (2, 0)
    # Empty inputs do not hide invalid ones.
    for n_max in (0, 2**21):
        with pytest.raises(InvalidInputError, match="number of terms"):
            mie.coefficients([], 1.5, n_max=n_max)
    with pytest.raises(InvalidInputError, match="element 1 of mu"):
        mie.amplitudes([], 1.5, [0.5, np.nan])
    with pytest.raises(InvalidInputError, match="element 1: the size parameter"):
        mie.amplitudes([1.0, -1.0], 1.5, np.zeros(0))


def test_output_size_limits():
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        mie.amplitudes(np.full(2**14, 1.0), 1.5, np.zeros(2**13), allow_unvalidated=True)
    # Broadcasting makes outputs far larger than the inputs: these ask for 2**30 spheres from
    # 512 KiB of inputs, and must fail before allocating anything.
    x, m = np.ones((2**15, 1)), np.full((1, 2**15), 1.5)
    with pytest.raises(InvalidInputError, match=r"2 \* MAX_MIE_VALUES"):
        mie.efficiencies(x, m)
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        mie.coefficients(x, m)
    # Within the bound on the number of spheres, but not with their numbers of terms.
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        mie.coefficients(np.full(2**14, 1e4), 1.5)
    # No angles, but each sphere is still checked: bounded as if there were one.
    with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
        mie.amplitudes(x, m, np.zeros(0))


def test_outputs_are_rust_vectors_handed_over_without_copying():
    x, m = np.array([1.0, 2.0]), np.array([1.5 + 0j, 1.5 + 0j])
    values, _ = _scatrs.mie_efficiencies(x, m, None, False, False)
    assert not isinstance(values.base, np.ndarray)
    q = mie.efficiencies(np.array([1.0, 2.0]), 1.5)
    assert q.q_ext.flags.c_contiguous
    root = q.q_ext
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    assert not isinstance(root, np.ndarray)  # a view of the Rust vector, not a copy


def test_results_are_frozen_and_read_only():
    q = mie.efficiencies([1.0, 2.0], 1.5 + 0.1j, derivatives=True)
    c = mie.coefficients(1.0, 1.5, derivatives=True)
    s = mie.amplitudes(1.0, 1.5, [0.5], derivatives=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.q_ext = q.q_sca
    for a in (q.q_ext, q.d("x").g, c.a, c.n_terms, c.d("m").a, c.d("m_imag").b, s.s2, s.d("x").s1):
        with pytest.raises(ValueError, match="read-only"):
            a[...] = 0
        with pytest.raises(ValueError):
            a.flags.writeable = True  # they are views of read-only arrays
    # Arrays the caller passes in stay as they are.
    a = np.ones(3, dtype=complex)
    mie.MieCoefficients(a, a, np.ones(3, dtype=np.int64))
    assert a.flags.writeable
    # Identity comparison and hashing, not elementwise array comparison.
    assert q == q and q != mie.efficiencies([1.0, 2.0], 1.5 + 0.1j, derivatives=True)
    assert len({q, c, s}) == 3


def test_other_layouts_and_dtypes_give_the_same_results():
    x = np.array([[0.5, 2.0], [7.0, 30.0]])
    m = np.array([[1.33 + 0.01j, 1.5], [2.0 + 1j, 1.25]])
    expected = mie.efficiencies(x, m).q_ext
    misaligned = np.frombuffer(b"\0" + x.tobytes(), dtype=np.float64, count=4, offset=1)
    for xi, mi in [
        (np.asfortranarray(x), np.asfortranarray(m)),
        (x.astype(x.dtype.newbyteorder()), m.astype(m.dtype.newbyteorder())),
        (misaligned.reshape(2, 2), m),
        (x[::-1][::-1], m[:, ::-1][:, ::-1]),
        (x.tolist(), m.tolist()),
        (x.astype(np.float32), m),  # exact in float32
    ]:
        np.testing.assert_array_equal(mie.efficiencies(xi, mi).q_ext, expected)


def test_complex_array_returns_qualifying_arrays_as_they_are():
    m = (np.linspace(1, 2, 12) + 0.1j).reshape(3, 4)
    assert _args.complex_array(m, "m") is m
    for other in (m.T, m.astype(np.complex64), m.astype(m.dtype.newbyteorder()), m.tolist(),
                  m.real, m.real.astype(np.float32), m.real.astype(np.int64)):  # fmt: skip
        converted = _args.complex_array(other, "m")
        assert converted.dtype == np.complex128 and converted.dtype.isnative
        assert converted.flags.c_contiguous and converted.flags.aligned
        np.testing.assert_array_equal(converted, np.asarray(other, dtype=np.complex128))
    for bad in (np.array([True]), np.array(["1.5"]), np.array([1], dtype=object)):
        with pytest.raises(TypeError):
            _args.complex_array(bad, "m")


class _Spy:
    def __init__(self):
        self.seen = []

    def __getattr__(self, name):
        return getattr(_scatrs, name)

    def mie_efficiencies(self, x, m, *args):
        self.seen.append((x, m))
        return _scatrs.mie_efficiencies(x, m, *args)


def test_inputs_are_broadcast_without_copying(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(mie, "_scatrs", spy)
    x = np.linspace(1, 2, 4).reshape(4, 1)
    m = np.array([1.5 + 0j, 1.6 + 0j, 1.7 + 0j])
    mie.efficiencies(x, m)
    xb, mb = spy.seen[-1]
    assert xb.shape == mb.shape == (4, 3)
    assert np.shares_memory(xb, x) and np.shares_memory(mb, m)
    assert xb is not x and mb is not m
    assert 0 in xb.strides and 0 in mb.strides  # broadcast, not materialised
    # Inputs that already have the broadcast shape are passed as fresh views too, never as
    # the caller's arrays (see scatrs._args.real_array).
    x_full, m_full = np.linspace(1, 2, 3), np.array([1.5 + 0j, 1.6 + 0j, 1.7 + 0j])
    for xi, mi in [(x_full, 1.5), (x_full, m_full)]:
        mie.efficiencies(xi, mi)
        xb, mb = spy.seen[-1]
        assert xb is not x_full and mb is not m_full
        assert np.shares_memory(xb, x_full)


def test_buffer_error_when_an_input_is_borrowed_for_writing():
    x = np.array([1.0, 2.0])
    with pytest.raises(BufferError):
        _testing.with_write_borrow(x, lambda: mie.efficiencies(x, 1.5))


def test_threaded_results_equal_serial_results():
    x = np.geomspace(0.1, 500, 40)
    serial = mie.efficiencies(x, 1.5 + 0.01j).q_ext
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: mie.efficiencies(x, 1.5 + 0.01j).q_ext, range(8)))
    for r in results:
        np.testing.assert_array_equal(r, serial)


def test_gil_is_released_while_computing():
    span = {}

    def worker():
        span["start"] = time.perf_counter()
        mie.efficiencies(np.full(200, 1.5e4), 1.5 + 0.01j, derivatives=True)
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
