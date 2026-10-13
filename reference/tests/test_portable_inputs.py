"""The generators must draw their inputs without the platform's maths library.

libm results (``math.cos``, ``10.0 ** u``, ...) are not correctly rounded everywhere, so inputs
drawn through them, and with them the committed data, would differ between platforms; the
weekly drift check regenerates the data on Linux and macOS. Arb does the rounding instead:
see ``records.nearest_pow10``. This test scans the generator sources for the patterns that
reach libm.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

GENERATORS = Path(__file__).resolve().parent.parent / "scatrs_reference" / "generators"

#: ``math`` names that are exact or correctly rounded on every IEEE 754 platform, and constants.
MATH_ALLOWED = frozenset(
    {
        "ceil", "comb", "copysign", "e", "fabs", "factorial", "floor", "fma", "fmod", "frexp",
        "fsum", "gcd", "inf", "isclose", "isfinite", "isinf", "isnan", "isqrt", "lcm", "ldexp",
        "modf", "nan", "nextafter", "perm", "pi", "prod", "remainder", "sqrt", "tau", "trunc",
        "ulp",
    }
)  # fmt: skip

#: ``random.Random`` methods that call libm (log, exp, cos, ...) internally.
RANDOM_VIA_LIBM = frozenset(
    {
        "betavariate", "binomialvariate", "expovariate", "gammavariate", "gauss",
        "lognormvariate", "normalvariate", "paretovariate", "vonmisesvariate", "weibullvariate",
    }
)  # fmt: skip

#: Calls that return integers, accepted as exponents of 2.0.
INTEGER_CALLS = frozenset({"int", "len", "randint", "randrange"})

#: Calls that return floats: never an exponent (``y ** rng.uniform(a, b)`` calls libm's pow).
FLOAT_CALLS = frozenset({"float", "random", "triangular", "uniform"})


def _is_float(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_float(node.operand)
    return isinstance(node, ast.Constant) and isinstance(node.value, float)


def _is_integer(node: ast.expr) -> bool:
    """Whether ``node`` evidently has an integer value (names are taken to be integers)."""
    match node:
        case ast.Constant(value=bool()):
            return False
        case ast.Constant(value=int()) | ast.Name():
            return True
        case ast.UnaryOp(op=ast.USub() | ast.UAdd(), operand=operand):
            return _is_integer(operand)
        case ast.BinOp(op=ast.Add() | ast.Sub() | ast.Mult() | ast.FloorDiv() | ast.LShift()):
            return _is_integer(node.left) and _is_integer(node.right)
        case ast.Call(func=ast.Name(id=f) | ast.Attribute(attr=f)):
            return f in INTEGER_CALLS
    return False


def violations(source: str) -> list[str]:
    """The libm-dependent constructs in a generator's source, as ``line: description``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        where = f"{getattr(node, 'lineno', '?')}: "
        if isinstance(node, ast.Import | ast.ImportFrom):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if module.split(".")[0] in ("cmath", "numpy"):
                    found.append(where + f"imports {module}")
            if isinstance(node, ast.Import) and any(
                a.name == "math" and a.asname not in (None, "math") for a in node.names
            ):
                found.append(where + "imports math under another name")
            if isinstance(node, ast.ImportFrom) and node.module == "math":
                found += [
                    where + f"math.{a.name}" for a in node.names if a.name not in MATH_ALLOWED
                ]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "math" and node.attr not in MATH_ALLOWED:
                found.append(where + f"math.{node.attr}")
            elif node.attr in RANDOM_VIA_LIBM:
                found.append(where + f"random {node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "pow":
                found.append(where + "builtin pow")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = node.left, node.right
            if _is_float(exponent):
                found.append(where + "** with a float exponent")
            elif _is_float(base) and not (
                isinstance(base, ast.Constant) and base.value == 2.0 and _is_integer(exponent)
            ):
                found.append(where + "** of a float other than 2.0 ** integer")
            elif isinstance(exponent, ast.Call) and _called(exponent) in FLOAT_CALLS:
                found.append(where + "** with a float exponent")
    return found


def _called(call: ast.Call) -> str:
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return ""


@pytest.mark.parametrize("path", sorted(GENERATORS.glob("*.py")), ids=lambda p: p.name)
def test_generator_draws_no_input_through_libm(path):
    assert violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source",
    [
        "x = 10.0 ** rng.uniform(-3, 3)",
        "x = 10.0 ** k",
        "x = 2.0 ** rng.uniform(0, 1)",
        "x = y ** 0.5",
        "x = y ** -0.5",
        "x = r * math.cos(t)",
        "x = math.exp(t)",
        "x = math.pow(10.0, 3)",
        "from math import sin",
        "import cmath",
        "import math as m",
        "x = pow(10.0, u)",
        "x = rng.gauss(0.0, 1.0)",
        "x = y ** rng.random()",
    ],
)
def test_libm_dependent_constructs_are_found(source):
    assert len(violations(source)) == 1, violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "x = 2.0**-52",
        "x = 2.0 ** e",
        "x = 2.0 ** -rng.randint(1, 60)",
        "x = 2 ** rng.randint(1, 40)",
        "x = (2**53) << s",
        "x = arb(10) ** arb(u)",
        "x = arb(2) ** -969",
        "x = Fraction(10) ** exponent",
        "x = math.ulp(f) * math.pi + math.sqrt(0.5)",
        "x = rng.uniform(-math.pi, math.pi)",
        "from math import nextafter",
    ],
)
def test_portable_constructs_are_accepted(source):
    assert violations(source) == []
