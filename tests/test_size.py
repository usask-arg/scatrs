"""scatrs.size: distributions as values, array-valued parameters, grids."""

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scatrs import LogNormal, MultiModal, PowerLaw, Tabulated, size


def test_scalar_parameters_are_floats_and_arrays_read_only_copies():
    d = LogNormal(np.float32(0.1), 2)
    assert type(d.median_radius) is float and d.width == 2.0
    radii = np.array([0.1, 0.2])
    g = LogNormal(radii, 1.5)
    radii[0] = 9.0
    assert g.median_radius[0] == 0.1 and not g.median_radius.flags.writeable
    with pytest.raises(FrozenInstanceError):
        g.width = 1.6
    with pytest.raises(TypeError, match="real numbers"):
        LogNormal(0.1 + 1j, 1.5)


def test_array_parameters_cannot_be_made_writeable():
    d = LogNormal([0.1, 0.2], 1.5)
    with pytest.raises(ValueError):
        d.median_radius.flags.writeable = True
    b = MultiModal([([1.0, 2.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.2, 1.6))])
    with pytest.raises(ValueError):
        b.modes[0][0].flags.writeable = True


def test_distributions_are_values():
    t1 = Tabulated(np.array([0.1, 0.5]), [1.0, 2.0])
    t2 = Tabulated([0.1, 0.5], (1.0, 2.0))
    assert t1 == t2 and hash(t1) == hash(t2)
    b = MultiModal([(1.0, LogNormal(0.08, 1.6)), (2.0, t1)])
    assert hash(b) == hash(MultiModal([(1, LogNormal(0.08, 1.6)), (2, t2)]))
    g1, g2 = LogNormal([0.1, 0.2], 1.5), LogNormal(np.array([0.1, 0.2]), 1.5)
    assert g1 == g2 and hash(g1) == hash(g2)
    assert g1 != LogNormal([0.1, 0.3], 1.5) and g1 != LogNormal([[0.1, 0.2]], 1.5)
    assert LogNormal(0.1, 1.5) != PowerLaw(0.1, 1.5, 2.0)
    with pytest.raises(TypeError, match="must be one of"):
        MultiModal([(1.0, "x")])


def test_grids_broadcast_and_flatten_in_c_order():
    d = PowerLaw([[3.0], [4.0]], 0.1, [0.5, 1.0, 2.0])
    shape, specs = size._specs(d)
    assert shape == (2, 3) and len(specs) == 6
    assert specs[1] == ("power_law", 3.0, 0.1, 1.0)
    assert specs[3] == ("power_law", 4.0, 0.1, 0.5)
    assert size._specs(LogNormal(0.1, 1.5)) == ((), [("lognormal", 0.1, 1.5)])
    m = MultiModal([([1.0, 2.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, 1.6))])
    shape, specs = size._specs(m)
    assert shape == (2,)
    assert specs[1] == (
        "multimodal",
        [(2.0, ("lognormal", 0.1, 1.5)), (1.0, ("lognormal", 0.5, 1.6))],
    )
    shape, arrays = size._grid(m)
    assert set(arrays) == {"mode0.weight"}


def _copies(x):
    """``x`` pickled and unpickled, and deep-copied."""
    return [pickle.loads(pickle.dumps(x)), copy.deepcopy(x)]


def test_pickled_and_copied_distributions_stay_immutable_values():
    g = LogNormal([0.1, 0.2], 1.5)
    b = MultiModal([([1.0, 2.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.2, [1.6, 1.7]))])
    t = Tabulated([0.1, 0.5], [1.0, 2.0])
    for d, arrays in [
        (g, lambda c: [c.median_radius]),
        (b, lambda c: [c.modes[0][0], c.modes[1][1].width]),
        (t, lambda c: []),
    ]:
        for c in _copies(d):
            assert type(c) is type(d) and c == d and hash(c) == hash(d)
            for a in arrays(c):
                with pytest.raises(ValueError):
                    a.flags.writeable = True
    for c in _copies(t):
        assert c.radius == (0.1, 0.5) and c.density == (1.0, 2.0)


def test_mode_weights_and_pairs_are_checked():
    ln = LogNormal(0.1, 1.5)
    for w in (True, np.bool_(False), "1.0", b"1", [True, False]):
        with pytest.raises(TypeError, match="the weight of mode 1"):
            MultiModal([(1.0, ln), (w, ln)])
    with pytest.raises(
        TypeError, match=r"mode 1 must be a \(weight, distribution\) pair, not LogN"
    ):
        MultiModal([(1.0, ln), ln])
    with pytest.raises(
        TypeError, match=r"mode 0 must be a \(weight, distribution\) pair, not tuple"
    ):
        MultiModal([(1.0, ln, 2.0)])
