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

#: Calls that return integers: exponents that keep ``2.0 ** e`` exact.
INTEGER_CALLS = frozenset({"int", "len", "randint", "randrange"})

#: Calls that return floats: never an exponent (``y ** rng.uniform(a, b)`` calls libm's pow).
FLOAT_CALLS = frozenset({"float", "random", "triangular", "uniform"})


def _is_float(node: ast.expr) -> bool:
    """Whether ``node`` is a float literal, possibly signed."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_float(node.operand)
    return isinstance(node, ast.Constant) and isinstance(node.value, float)


def _is_integer(node: ast.expr) -> bool:
    """Whether ``node`` is evidently an integer: built from integer literals and integer calls
    (``rng.randint``, ``int``, ...). A bare name is not: it might hold a float."""
    match node:
        case ast.Constant(value=bool()):
            return False
        case ast.Constant(value=int()):
            return True
        case ast.UnaryOp(op=ast.USub() | ast.UAdd(), operand=operand):
            return _is_integer(operand)
        case ast.BinOp(
            op=ast.Add() | ast.Sub() | ast.Mult() | ast.FloorDiv() | ast.Mod() | ast.LShift()
        ):
            return _is_integer(node.left) and _is_integer(node.right)
        case ast.Call():
            return _called(node) in INTEGER_CALLS
    return False


def _called(call: ast.Call) -> str:
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return ""


def _numeric(node: ast.expr, names: dict[str, str]) -> str | None:
    """``"int"`` or ``"float"`` if ``node`` is a plain Python number (a literal, ``int(...)``,
    ``float(...)``, a name bound to one, or arithmetic on them), else None (an Arb or Fraction
    value, or anything unknown)."""
    match node:
        case ast.Constant(value=bool()):
            return None
        case ast.Constant(value=int()):
            return "int"
        case ast.Constant(value=float()):
            return "float"
        case ast.UnaryOp(op=ast.USub() | ast.UAdd(), operand=operand):
            return _numeric(operand, names)
        case ast.Call() if _called(node) in ("int", "float"):
            return _called(node)
        case ast.Name(id=name):
            return names.get(name)
        case ast.BinOp(op=op, left=left, right=right):
            kinds = {_numeric(left, names), _numeric(right, names)}
            if None in kinds:
                return None
            return "float" if "float" in kinds or isinstance(op, ast.Div) else "int"
    return None


def _numeric_names(tree: ast.Module) -> dict[str, str]:
    """Names bound anywhere in the module to a plain number, as in ``ten = 10.0``."""
    names: dict[str, str] = {}
    for _ in range(2):  # a second pass resolves names defined through other names
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name) and (kind := _numeric(value, names)):
                names[target.id] = "float" if names.get(target.id) == "float" else kind
    return names


def _power(base: ast.expr, exponent: ast.expr, names: dict[str, str]) -> str | None:
    """Why ``base ** exponent`` may call libm's pow, or None if it cannot."""
    if _is_float(exponent) or (isinstance(exponent, ast.Call) and _called(exponent) in FLOAT_CALLS):
        return "** with a float exponent"
    kind = _numeric(base, names)
    if kind is None:  # Arb, Fraction, ...: exact or correctly rounded
        return None
    if not _is_integer(exponent):
        return f"** of a Python {kind} with an exponent that is not evidently an integer"
    two = isinstance(base, ast.Constant) and base.value == 2
    if kind == "float" and not two:
        return "** of a float other than 2.0 ** integer"
    if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub) and not two:
        return "** of an integer with a negative exponent (a float pow)"
    return None


def violations(source: str) -> list[str]:
    """The libm-dependent constructs in a generator's source, as ``line: description``."""
    tree = ast.parse(source)
    names = _numeric_names(tree)
    found = []
    for node in ast.walk(tree):
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
            elif (
                node.func.id == "getattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in ("math", "cmath")
            ):
                found.append(where + f"getattr on {node.args[0].id}")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Pow):
            found.append(where + "**= (augmented power)")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if why := _power(node.left, node.right, names):
                found.append(where + why)
    return found


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
        "x **= 0.5",
        "x **= k",
        "x = 10 ** u",
        "x = 10 ** -3",
        "x = 2.0 ** e",
        "x = 2 ** e",
        "ten = 10.0\nx = ten ** u",
        "two = 2\nx = two ** k",
        "x = float(10) ** u",
        "x = getattr(math, 'cos')(t)",
        "x = getattr(cmath, name)",
    ],
)
def test_libm_dependent_constructs_are_found(source):
    assert len(violations(source)) == 1, violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "x = 2.0**-52",
        "x = 2.0 ** -rng.randint(1, 60)",
        "x = 2.0 ** (rng.randint(1, 9) * 2 + 1)",
        "x = 2 ** rng.randint(1, 40)",
        "x = 2**-3",
        "x = 10**289",
        "x = (2**53) << s",
        "x = a ** b",
        "x = base**e",
        "x = arb(fmpz(10)) ** (1 << i)",
        "x = math.ldexp(1.0, e)",
        "ten = arb(10)\nx = ten ** u",
        "x = getattr(flint, 'arb')",
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
