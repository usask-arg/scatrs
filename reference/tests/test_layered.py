"""Self-checks of the layered-sphere generator against the homogeneous one."""

from flint import acb, arb, ctx

from scatrs_reference.generators import layered, mie


def _close(u: acb, v: acb, rel: float) -> bool:
    """Both balls enclose the exact value, so they must overlap; and they are tight."""
    tight = all(b.rad() <= rel * abs(b.mid()) for w in (u, v) for b in (w.real, w.imag))
    return bool(u.overlaps(v) and tight)


def test_identical_layers_are_the_homogeneous_sphere():
    with ctx.workprec(320):
        for x, (m_re, m_im) in [(0.5, (1.5, 0.01)), (20.0, (1.33, 1e-8)), (3.0, (10.0, 10.0))]:
            n = mie.size_terms(x, 6, 3)
            h = mie.solve(x, m_re, m_im, n)
            for fractions in [(0.3, 1.0), (0.2, 0.5, 1.0)]:
                xs = [arb(f * x) for f in fractions]
                ms = [acb(m_re, m_im)] * len(xs)
                s = layered.solve(xs, ms, [m_im == 0.0] * len(xs), n)
                for k in range(n):
                    assert _close(s.a[k], h.a[k], 1e-6), (x, fractions, k)
                    assert _close(s.b[k], h.b[k], 1e-6), (x, fractions, k)


def test_an_air_shell_gives_the_bare_core():
    with ctx.workprec(320):
        for x, (m_re, m_im) in [(0.01, (1.5, 0.01)), (2.0, (1.95, 0.79)), (15.0, (1.33, 0.0))]:
            n = mie.size_terms(x, 6, 3)
            h = mie.solve(x, m_re, m_im, n)
            xs, ms = [arb(x), arb(1.7 * x)], [acb(m_re, m_im), acb(1)]
            s = layered.solve(xs, ms, [m_im == 0.0, True], n)
            for k in range(n):
                assert _close(s.a[k], h.a[k], 1e-6), (x, k)
                assert _close(s.b[k], h.b[k], 1e-6), (x, k)
