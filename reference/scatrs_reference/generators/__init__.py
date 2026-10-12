"""Registry of reference generators. Each returns ``(precision_bits, records)``."""

from collections.abc import Callable

from scatrs_reference.generators import constants, dd_arith, dd_elementary

Generator = Callable[[], tuple[int, list[dict]]]

GENERATORS: dict[str, Generator] = {
    "constants": constants.generate,
    "dd_arith": dd_arith.generate,
    "dd_elementary": dd_elementary.generate,
}
