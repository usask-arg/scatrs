"""Registry of reference generators. Each returns ``(precision_bits, records)``."""

from collections.abc import Callable

from scatrs_reference.generators import (
    constants,
    dd_arith,
    dd_complex,
    dd_elementary,
    dd_parse,
    dd_trig,
    gauss_legendre,
    mie,
    mie_greek,
    wigner_d,
)

Generator = Callable[[], tuple[int, list[dict]]]

GENERATORS: dict[str, Generator] = {
    "constants": constants.generate,
    "dd_arith": dd_arith.generate,
    "dd_complex": dd_complex.generate,
    "dd_elementary": dd_elementary.generate,
    "dd_parse": dd_parse.generate,
    "dd_trig": dd_trig.generate,
    "gauss_legendre": gauss_legendre.generate,
    "mie": mie.generate,
    "mie_derivatives": mie.generate_derivatives,
    "mie_greek": mie_greek.generate,
    "wigner_d": wigner_d.generate,
}
