"""Command line: ``python -m scatrs_reference generate [--out DIR] [--only NAME ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scatrs_reference.generators import GENERATORS
from scatrs_reference.records import update_manifest, write_jsonl

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scatrs_reference")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="(re)generate reference data files")
    gen.add_argument("--out", type=Path, default=DEFAULT_OUT)
    gen.add_argument("--only", nargs="+", choices=sorted(GENERATORS))
    args = parser.parse_args(argv)
    for name in args.only or sorted(GENERATORS):
        precision_bits, records = GENERATORS[name]()
        filename = f"{name}.jsonl"
        digest = write_jsonl(args.out / filename, records)
        update_manifest(args.out, filename, name, precision_bits, len(records), digest)
        print(f"{filename}: {len(records)} records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
