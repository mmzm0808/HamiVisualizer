"""Safe, deliberately small expression language for model parameters.

Model files and table cells are user input.  SymPy's ``sympify(str)`` uses
Python evaluation internally and must therefore not be used on those strings.
This module translates a restricted Python expression AST into SymPy objects.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping

import sympy as sp


MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 96
MAX_AST_DEPTH = 20
MAX_NAME_LENGTH = 48
MAX_ABS_NUMERIC_EXPONENT = 64


class ExpressionError(ValueError):
    """An expression is invalid or outside the supported safe language."""


_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}
_FUNCTIONS = {
    "sqrt": (sp.sqrt, 1),
    "sin": (sp.sin, 1),
    "cos": (sp.cos, 1),
    "tan": (sp.tan, 1),
    "exp": (sp.exp, 1),
    "log": (sp.log, 1),
    "abs": (sp.Abs, 1),
    "Abs": (sp.Abs, 1),
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E, "I": sp.I}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ExpressionError("只允许实数或复数字面量")
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ExpressionError("数值必须有限，不能使用 NaN 或 Inf")
        return sp.Float(value.real) + sp.I * sp.Float(value.imag)
    if not math.isfinite(float(value)):
        raise ExpressionError("数值必须有限，不能使用 NaN 或 Inf")
    return sp.Integer(value) if isinstance(value, int) else sp.Float(value)


class _SafeParser:
    def __init__(self, symbols: Mapping[str, sp.Symbol] | None = None):
        self.symbols = dict(symbols or {})
        self._count = 0

    def visit(self, node: ast.AST, depth: int = 0):
        self._count += 1
        if self._count > MAX_AST_NODES:
            raise ExpressionError(f"表达式过于复杂（最多 {MAX_AST_NODES} 个语法节点）")
        if depth > MAX_AST_DEPTH:
            raise ExpressionError(f"表达式嵌套过深（最多 {MAX_AST_DEPTH} 层）")

        if isinstance(node, ast.Expression):
            return self.visit(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.Name):
            name = node.id
            if len(name) > MAX_NAME_LENGTH or name.startswith("_"):
                raise ExpressionError(f"非法参数名：{name!r}")
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            if name in _FUNCTIONS:
                raise ExpressionError(f"函数 {name} 必须使用括号调用")
            return self.symbols.setdefault(name, sp.Symbol(name, real=True))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self.visit(node.operand, depth + 1))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = self.visit(node.left, depth + 1)
            right = self.visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and right.is_number:
                try:
                    exponent = float(right)
                except (TypeError, ValueError):
                    exponent = 0.0
                if not math.isfinite(exponent) or abs(exponent) > MAX_ABS_NUMERIC_EXPONENT:
                    raise ExpressionError(
                        f"数值指数绝对值不能超过 {MAX_ABS_NUMERIC_EXPONENT}"
                    )
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ExpressionError("只允许 sqrt/sin/cos/tan/exp/log/abs 函数")
            if node.keywords:
                raise ExpressionError("函数不支持关键字参数")
            fn, arity = _FUNCTIONS[node.func.id]
            if len(node.args) != arity:
                raise ExpressionError(f"函数 {node.func.id} 需要 {arity} 个参数")
            return fn(*[self.visit(arg, depth + 1) for arg in node.args])

        raise ExpressionError(f"不支持的表达式语法：{type(node).__name__}")


def parse_expression(
    text: str,
    *,
    symbols: Mapping[str, sp.Symbol] | None = None,
    empty_value=sp.Integer(1),
):
    """Parse a safe expression into an exact SymPy expression.

    Supported syntax: numeric literals, parameter names, ``pi/E/I``, the
    arithmetic operators ``+ - * / **`` and a small mathematical function
    allow-list.  Attribute access, indexing, comprehensions and arbitrary
    Python calls are rejected before SymPy sees the input.
    """

    source = str(text).strip().replace("^", "**")
    if not source:
        return empty_value
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"表达式过长（最多 {MAX_EXPRESSION_LENGTH} 个字符）"
        )
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"表达式语法错误：{exc.msg}") from None
    try:
        return _SafeParser(symbols).visit(tree)
    except (ExpressionError, ZeroDivisionError):
        raise
    except Exception as exc:
        raise ExpressionError(f"无法解析表达式：{exc}") from None


def collect_symbols(expressions) -> set[str]:
    """Return free parameter names, raising on the first invalid expression."""

    names: set[str] = set()
    for source in expressions:
        expr = parse_expression(source)
        names.update(str(symbol) for symbol in expr.free_symbols)
    # ``kx`` is the historical spelling; ``k_x`` is accepted as the readable
    # alias used by the LaTeX-style renderer.  Neither should become a user
    # parameter slider.
    return names - {"kx", "k_x"}


def evaluate_expression(text: str, params: Mapping[str, float]) -> complex:
    """Safely evaluate an expression using a complete numeric parameter map."""

    symbols = {name: sp.Symbol(name, real=True) for name in params}
    expr = parse_expression(text, symbols=symbols)
    unknown = sorted(str(symbol) for symbol in expr.free_symbols if str(symbol) not in params)
    if unknown:
        raise ExpressionError(
            f"未知参数 {unknown}：表达式 {str(text)!r} 中的符号没有对应数值"
        )
    substitutions = {symbols[name]: sp.Float(float(value)) for name, value in params.items()}
    value = complex(expr.evalf(subs=substitutions))
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ExpressionError("表达式结果必须是有限数值")
    return value
