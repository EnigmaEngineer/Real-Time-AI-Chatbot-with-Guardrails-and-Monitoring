"""Safe arithmetic calculator. Walks a Python AST instead of eval()."""

from __future__ import annotations

import ast
import operator
from typing import Any

from src.agents.tools.base import Tool, ToolResult


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 2**1000 freezes Python — cap the exponent.
_MAX_POW_EXPONENT = 50


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"operator {op_type.__name__} not allowed")
        if op_type is ast.Pow:
            right = _eval_node(node.right)
            if isinstance(right, (int, float)) and abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"exponent magnitude > {_MAX_POW_EXPONENT}")
        return _BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unary operator {op_type.__name__} not allowed")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    raise ValueError(f"node type {type(node).__name__} not allowed")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a Python arithmetic expression. Supports + - * / // % ** and parens. "
        "No variables, no function calls. Returns a number."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression like '(2 + 3) * 4' or '7 ** 2'",
            }
        },
        "required": ["expression"],
    }
    policies = ("max_arg_length", "no_pii_in_args")

    async def run(self, args: dict[str, Any]) -> ToolResult:
        expr = args.get("expression", "")
        if not isinstance(expr, str):
            return ToolResult(ok=False, output="", error="expression must be a string")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_node(tree)
        except SyntaxError as exc:
            return ToolResult(ok=False, output="", error=f"syntax error: {exc.msg}")
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(ok=False, output="", error=str(exc))
        return ToolResult(ok=True, output=str(value), metadata={"expression": expr})
