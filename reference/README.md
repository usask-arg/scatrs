# scatrs reference generator

Independent ground truth for scatrs tests, computed with Arb ball arithmetic through
[python-flint](https://python-flint.readthedocs.io). It does not import scatrs.

    uv run --directory reference python -m scatrs_reference generate            # all files
    uv run --directory reference python -m scatrs_reference generate --only constants
    uv run --directory reference python -m pytest

Each value is stored as `hi`/`lo` doubles, a 25-digit decimal `dec`, and a rigorous error
bound `err` on |true - (hi + lo)|. `label` is `certified` when every source of error
(including series truncation) is rigorously bounded, otherwise `high-precision`.
