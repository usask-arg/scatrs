"""Complex disk arithmetic on top of Arb, for long complex recurrences.

Arb's complex balls (``acb``) are rectangles: the real and imaginary parts each carry a radius.
A recurrence that multiplies by complex numbers of modulus near 1 rotates the rectangle, and
Arb must enclose each rotated rectangle in a larger axis-parallel one, up to √2 times larger
per step, although the set of possible values stays a disk of about the same size. In the Mie
recurrences the radius grew by about 1.5× per step, and the balls became useless after a few
hundred orders. A :class:`Disk` keeps an exact centre and a radius instead (circular complex
interval arithmetic, Gargantini & Henrici 1972): each operation computes the centre with Arb,
recentres on the midpoint of Arb's result, and adds a rigorous bound on the propagated radius.
"""

from __future__ import annotations

from dataclasses import dataclass

from flint import acb, arb


@dataclass(frozen=True, slots=True)
class Disk:
    """The closed disk {w : |w − centre| ≤ radius}.

    The module builds disks with an exact ``centre`` (a midpoint) and an exact ``radius`` (an
    upper bound). The operations stay enclosures when ``centre`` or ``radius`` carry radii of
    their own: they then cover every disk that the balls describe.
    """

    centre: acb
    radius: arb


def _recentre(v: acb, extra: arb | None = None) -> Disk:
    """The disk centred on the midpoint of ``v`` that contains ``v`` grown by ``extra``.

    The rectangle of ``v`` lies within ``re.rad() + im.rad()`` of its midpoint (its
    half-diagonal is no longer than that).
    """
    r = v.real.rad() + v.imag.rad()
    if extra is not None:
        r = r + extra
    return Disk(v.mid(), r.abs_upper())


def disk(v: acb) -> Disk:
    """The disk centred on the midpoint of ``v``, of radius re.rad + im.rad: it contains ``v``."""
    return _recentre(v)


def ball(d: Disk) -> acb:
    """An Arb ball (a square) containing the disk ``d``."""
    return d.centre + acb(arb(0, d.radius), arb(0, d.radius))


def sub_from(c: acb, d: Disk) -> Disk:
    """A disk containing ``c − w`` for every point ``c`` of the ball and ``w`` of ``d``."""
    return _recentre(c - d.centre, d.radius)


class DiskContainsZero(ArithmeticError):
    """Raised by `inv` for a disk that may contain zero."""


def inv(d: Disk) -> Disk:
    """A disk containing ``1/w`` for every ``w`` in ``d``.

    For ``|w − c| ≤ r < |c|``: ``|1/w − 1/c| = |c − w| / (|w| |c|) ≤ r / ((|c| − r) |c|)``.

    Raises
    ------
    DiskContainsZero
        If ``d`` may contain zero.
    """
    m = abs(d.centre)
    lower = m - d.radius
    if not lower > 0:
        raise DiskContainsZero("the disk may contain zero")
    return _recentre(1 / d.centre, d.radius / (m * lower))


def mul(d: Disk, e: Disk) -> Disk:
    """A disk containing ``v w`` for every ``v`` in ``d`` and ``w`` in ``e``.

    ``|v w − c₁ c₂| ≤ |c₁| r₂ + |c₂| r₁ + r₁ r₂``.
    """
    return _recentre(
        d.centre * e.centre,
        abs(d.centre) * e.radius + abs(e.centre) * d.radius + d.radius * e.radius,
    )
