# scatrs reference generator

Independent ground truth for scatrs tests, computed with Arb ball arithmetic through
[python-flint](https://python-flint.readthedocs.io). It does not import scatrs.

    uv run --directory reference python -m scatrs_reference generate            # all files
    uv run --directory reference python -m scatrs_reference generate --only constants
    uv run --directory reference python -m pytest

Each value is stored as three doubles `hi`/`lo`/`lo2` (`hi + lo` is a normalised
double-double and `lo2` the double nearest to the rest, about 160 bits in all), a decimal
string `dec` (up to 25 significant digits, fewer when the ball is wider), and a rigorous error
bound `err` on |true - (hi + lo + lo2)|. A double-double result can therefore be checked
almost exactly: |got - true| <= |got - (hi + lo + lo2)| + err.
`label` is `certified` when every source of error (including series truncation) is
rigorously bounded, otherwise `high-precision`.
