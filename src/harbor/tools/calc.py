"""Arithmetic with a syntax tree. Names, calls, and attributes are refused."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from harbor.sandbox import ToolError
from harbor.tools.base import Tool
from harbor.tools.common import require_str

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_ABS = 10**12
_MAX_EXPONENT = 32
_MAX_EXPRESSION_LENGTH = 200


def make_calculate() -> Tool:
    def handler(args: dict[str, Any]) -> str:
        return evaluate(require_str(args, "expression"))

    return Tool(
        name="calculate",
        description=(
            "Evaluate a safe arithmetic expression. "
            "Supports +, -, *, /, //, %, **, parentheses, and numbers. "
            "Example: (12 + 8) / 4. Does not call functions or read files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, for example (12 + 8) / 4.",
                }
            },
            "required": ["expression"],
        },
        handler=handler,
    )


def evaluate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression is required")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc.msg}") from exc
    return _format_number(_eval(tree.body))


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if type(node.value) is int or type(node.value) is float:
            return _check(node.value)
        raise ToolError("only integers and decimals are allowed")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolError("operator is not allowed")
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError("exponent is too large")
        try:
            result = op(left, right)
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc
        if isinstance(result, complex) or not isinstance(result, int | float):
            raise ToolError("result is not a real number")
        return _check(result)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise ToolError("operator is not allowed")
        return _check(op(_eval(node.operand)))
    raise ToolError("expression contains something that is not arithmetic")


def _check(value: int | float) -> int | float:
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("result is not a finite number")
    if abs(value) > _MAX_ABS:
        raise ToolError("result is too large")
    return value


def _format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".10g")
    return str(value)
