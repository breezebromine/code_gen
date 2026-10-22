# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interpret a small Python task language without importing or executing Python code."""

import ast
import copy
import math
import operator
import re
from dataclasses import dataclass


class ProgramError(ValueError):
    """Invalid task source, action, or exhausted execution budget."""


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict


SKILLS = {
    "pick": "object_id",
    "grasp_actor": "object_id",
    "place": "target_id",
    "place_actor": "target_id",
    "observe": None,
    "home": None,
    "wait": "seconds",
    "move_to_pose": "pose",
    "move_object_to_pose": "object_id",
    "align_object": "object_id",
    "insert_object": "object_id",
    "move_by_displacement": "displacement",
    "open_gripper": None,
    "close_gripper": "object_id",
}
QUERIES = {
    "snapshot": None,
    "get_object_pose": "object_id",
    "get_target_pose": "target_id",
    "get_grasp_pose": "object_id",
}
_PICK_ARGUMENTS = {
    "grasp_id",
    "grasp_offset_m",
    "close_width_m",
    "lift_height_m",
    "yaw_candidates_deg",
    "tilt_candidates_deg",
}
SKILL_OPTIONAL_ARGUMENTS = {
    "pick": _PICK_ARGUMENTS,
    "grasp_actor": _PICK_ARGUMENTS,
    "get_grasp_pose": {"grasp_id"},
    "move_to_pose": {"vertical"},
    "move_object_to_pose": {"pose", "vertical"},
    "align_object": {"pose"},
    "insert_object": {"pose", "approach_height"},
    "move_by_displacement": {"frame"},
    "close_gripper": {"close_width_m"},
}
_ALLOWED = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Yield,
    ast.If,
    ast.For,
    ast.Return,
    ast.Pass,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
)
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def extract_program(response):
    """Accept one complete source block; never silently repair candidates."""
    if not isinstance(response, str) or not response.strip():
        raise ProgramError("Model returned no final source")
    response = response.strip()
    if response.startswith("<think>"):
        _, end, response = response.partition("</think>")
        if not end:
            raise ProgramError("Unfinished model thinking")
    if "```" in response:
        blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", response, re.DOTALL)
        if len(blocks) != 1:
            raise ProgramError("Expected exactly one Python code block")
        response = blocks[0]
    source = response.strip() + "\n"
    validate_program(source)
    return source


def validate_program(source):
    if not isinstance(source, str) or len(source.encode()) > 20000:
        raise ProgramError("Program source exceeds 20 KB")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        raise ProgramError("Invalid Python source") from exc
    # RoboTwin-style wrapper is interpreted, never imported or executed as Python.
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef):
        cls = tree.body[0]
        if (
            cls.name != "GeneratedTask"
            or cls.bases
            or cls.keywords
            or cls.decorator_list
            or len(cls.body) != 1
            or not isinstance(cls.body[0], ast.FunctionDef)
        ):
            raise ProgramError("Use class GeneratedTask with only play_once(self)")
        method = cls.body[0]
        if method.name != "play_once" or len(method.args.args) != 1 or method.args.args[0].arg != "self":
            raise ProgramError("Required method: play_once(self)")
        method.name, method.args.args[0].arg = "run", "ctx"
        for item in ast.walk(method):
            if isinstance(item, ast.Name) and item.id == "self":
                item.id = "ctx"
        tree.body = [method]
    nodes = list(ast.walk(tree))
    if len(nodes) > 500:
        raise ProgramError("Program exceeds 500 AST nodes")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramError("Define only def run(ctx), without imports or top-level statements")
    function = tree.body[0]
    args = function.args
    if (
        function.name != "run"
        or function.decorator_list
        or function.returns
        or args.posonlyargs
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
        or args.vararg
        or args.kwarg
        or len(args.args) != 1
        or args.args[0].arg != "ctx"
        or args.args[0].annotation
    ):
        raise ProgramError("Required entry point: def run(ctx)")
    for node in nodes:
        _validate_node(node, function)
    for parent in nodes:
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.Yield) and not isinstance(parent, (ast.Assign, ast.Expr)):
                raise ProgramError("Yield must be a complete statement or assignment")
            if isinstance(child, ast.Attribute) and not isinstance(parent, ast.Call):
                raise ProgramError("ctx methods must be called directly")
            if isinstance(child, ast.Call) and child.func.attr not in QUERIES and not isinstance(parent, ast.Yield):
                raise ProgramError("Action requests must be yielded directly")
    return function


def _validate_node(node, function):
    if not isinstance(node, _ALLOWED):
        raise ProgramError(f"Unsupported syntax: {type(node).__name__}")
    if isinstance(node, ast.FunctionDef) and node is not function:
        raise ProgramError("Nested functions are not supported")
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ProgramError("Private names are not supported")
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ProgramError("Only JSON-compatible literals are supported")
        if isinstance(value, str) and len(value) > 1000:
            raise ProgramError("String literal too long")
        if isinstance(value, (int, float)) and (abs(value) > 1e6 or not math.isfinite(value)):
            raise ProgramError("Numeric literal out of range")
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id != "ctx":
            raise ProgramError("Only ctx methods may be accessed")
        if node.attr not in {*SKILLS, *QUERIES}:
            raise ProgramError(f"Unknown ctx method: {node.attr}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise ProgramError("Only ctx method calls are supported")
        if any(k.arg is None for k in node.keywords):
            raise ProgramError("Keyword expansion is not supported")
        if len({k.arg for k in node.keywords}) != len(node.keywords):
            raise ProgramError("Duplicate keyword argument")
        name = node.func.attr
        key = SKILLS.get(name, QUERIES.get(name))
        allowed = ({key} if key else set()) | SKILL_OPTIONAL_ARGUMENTS.get(name, set())
        if any(k.arg not in allowed for k in node.keywords):
            raise ProgramError(f"Unsupported arguments for {name}")
        if name in ("move_object_to_pose", "align_object", "insert_object") and not any(
            k.arg == "pose" for k in node.keywords
        ):
            raise ProgramError(f"{name} requires an object pose")
        if len(node.args) > (1 if key else 0):
            raise ProgramError(f"Too many positional arguments for {name}")
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id == "ctx":
            raise ProgramError("Assign only to one local variable")
    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name) or node.target.id == "ctx" or node.orelse:
            raise ProgramError("Use for NAME in LIST without else")
    if isinstance(node, ast.Dict) and any(k is None for k in node.keys):
        raise ProgramError("Dictionary expansion is not supported")


class _Returned(Exception):
    pass


class Program:
    """Resumable interpreter over JSON copies; no eval, exec or simulator object."""

    def __init__(self, source, max_actions=24, max_operations=2000):
        self.function = validate_program(source)
        self.locals = {}
        self.snapshot = {}
        self.remaining = max_operations
        self.max_actions = max_actions
        self.action_count = 0
        self.started = False
        self.iterator = self._run()

    def advance(self, snapshot, result=None):
        self.snapshot = copy.deepcopy(snapshot)
        try:
            if not self.started:
                self.started = True
                return next(self.iterator)
            return self.iterator.send(copy.deepcopy(result))
        except StopIteration:
            return None
        except ProgramError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
            raise ProgramError(f"Program evaluation failed: {type(exc).__name__}: {exc}") from None

    def _tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ProgramError("Program operation budget exhausted")

    def _run(self):
        try:
            yield from self._block(self.function.body)
        except _Returned:
            return

    def _block(self, statements):
        for node in statements:
            self._tick()
            if isinstance(node, (ast.Assign, ast.Expr)):
                value = node.value
                if isinstance(value, ast.Yield):
                    request = self._eval(value.value)
                    if not isinstance(request, Action):
                        raise ProgramError("Yield a ctx action request")
                    self.action_count += 1
                    if self.action_count > self.max_actions:
                        raise ProgramError("Program action budget exhausted")
                    value = yield request
                else:
                    value = self._eval(value)
                if isinstance(node, ast.Assign):
                    self.locals[node.targets[0].id] = value
            elif isinstance(node, ast.If):
                yield from self._block(node.body if self._eval(node.test) else node.orelse)
            elif isinstance(node, ast.For):
                values = self._eval(node.iter)
                if not isinstance(values, (list, tuple)) or len(values) > 32:
                    raise ProgramError("For loops require a list with at most 32 entries")
                for value in values:
                    self.locals[node.target.id] = value
                    yield from self._block(node.body)
            elif isinstance(node, ast.Return):
                raise _Returned()
            if not isinstance(node, (ast.Assign, ast.Expr, ast.If, ast.For, ast.Return, ast.Pass)):
                raise ProgramError("Unsupported statement")

    def _eval(self, node):
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.locals[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(x) for x in node.elts]
        if isinstance(node, ast.Dict):
            return {self._eval(k): self._eval(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            return self._eval(node.value)[self._eval(node.slice)]
        if isinstance(node, ast.Call):
            name = node.func.attr
            arguments = [self._eval(a) for a in node.args]
            keywords = {k.arg: self._eval(k.value) for k in node.keywords}
            key = SKILLS.get(name, QUERIES.get(name))
            if key:
                optional = SKILL_OPTIONAL_ARGUMENTS.get(name, set())
                if len(arguments) == 1 and key not in keywords:
                    keywords = {key: arguments[0], **keywords}
                elif arguments:
                    raise ProgramError(f"{name} requires one {key}")
                if key not in keywords or set(keywords) - {key} - optional:
                    raise ProgramError(f"Invalid arguments for {name}")
                if name in ("move_object_to_pose", "align_object", "insert_object") and "pose" not in keywords:
                    raise ProgramError(f"{name} requires an object pose")
                if name == "wait":
                    seconds = keywords[key]
                    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 10:
                        raise ProgramError("wait seconds must be between 0 and 10")
                elif key in ("object_id", "target_id") and not isinstance(keywords[key], str):
                    raise ProgramError(f"{key} must be a string")
            elif arguments or keywords:
                raise ProgramError(f"{name} takes no arguments")
            if name == "snapshot":
                return copy.deepcopy(self.snapshot)
            if name in ("get_object_pose", "get_target_pose"):
                group = "objects" if name == "get_object_pose" else "targets"
                item = self.snapshot[group][keywords[key]]
                return [*item["position"], *item.get("quaternion", [1, 0, 0, 0])]
            if name == "get_grasp_pose":
                from isaaclab_arena.grasping.geometry import grasp_frame

                obj = self.snapshot["objects"][keywords["object_id"]]
                _, value = grasp_frame(obj, keywords.get("grasp_id", obj.get("default_grasp_id")))
                return value
            return Action(name, keywords)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            value = self._number(value)
            return self._number(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BinOp):
            a, b = self._number(self._eval(node.left)), self._number(self._eval(node.right))
            return self._number(_ARITHMETIC[type(node.op)](a, b))
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self._eval(item)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            for op, item in zip(node.ops, node.comparators):
                right = self._eval(item)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        raise ProgramError(f"Unsupported expression: {type(node).__name__}")

    @staticmethod
    def _number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e6:
            raise ProgramError("Arithmetic requires finite numbers within +/- 1e6")
        return value
