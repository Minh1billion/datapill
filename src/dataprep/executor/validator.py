import ast
import textwrap
from dataclasses import dataclass, field

import polars as pl


_BANNED_IMPORTS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "http", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "imaplib", "poplib",
    "ctypes", "cffi", "multiprocessing", "threading",
    "importlib", "builtins", "pickle", "shelve",
    "pty", "tty", "termios", "signal", "resource",
    "tempfile", "glob", "fnmatch",
})


_BANNED_BUILTINS: frozenset[str] = frozenset({
    "__import__", "eval", "exec", "compile",
    "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "vars", "dir",
    "globals", "locals",
})


_BANNED_NODE_TYPES: tuple[type, ...] = (
    ast.Global,
    ast.Nonlocal,
)


@dataclass
class ASTViolation:
    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"Line {self.line}, col {self.col}: {self.message}"


@dataclass
class ASTAnalysisResult:
    ok: bool
    violations: list[ASTViolation] = field(default_factory=list)

    def add(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            ASTViolation(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=message,
            )
        )
        self.ok = False


class _ASTAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.result = ASTAnalysisResult(ok=True)


    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BANNED_IMPORTS:
                self.result.add(node, f"Import of banned module '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BANNED_IMPORTS:
                self.result.add(
                    node,
                    f"Import from banned module '{node.module}'",
                )
        self.generic_visit(node)


    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_BUILTINS:
            self.result.add(node, f"Call to banned built-in '{node.func.id}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in _BANNED_BUILTINS:
            self.result.add(node, f"Reference to banned built-in '{node.id}'")
        self.generic_visit(node)


    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.result.add(
                node,
                f"Access to dunder attribute '{node.attr}' is not allowed",
            )
        self.generic_visit(node)


    def visit_Global(self, node: ast.Global) -> None:
        self.result.add(node, "Use of 'global' statement is not allowed")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.result.add(node, "Use of 'nonlocal' statement is not allowed")
        self.generic_visit(node)


def analyze_ast(source: str) -> ASTAnalysisResult:
    source = textwrap.dedent(source)
    tree = ast.parse(source, filename="<user_code>")
    analyzer = _ASTAnalyzer()
    analyzer.visit(tree)
    return analyzer.result


@dataclass
class SchemaViolation:
    column: str | None
    message: str

    def __str__(self) -> str:
        prefix = f"Column '{self.column}': " if self.column else ""
        return prefix + self.message


@dataclass
class SchemaValidationResult:
    ok: bool
    violations: list[SchemaViolation] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    changed_dtypes: list[tuple[str, str, str]] = field(default_factory=list)  # (col, before, after)

    def add(self, col: str | None, message: str) -> None:
        self.violations.append(SchemaViolation(col, message))
        self.ok = False


def validate_schema(
    input_df: pl.DataFrame,
    output_df: pl.DataFrame,
    *,
    allow_add_columns: bool = True,
    allow_remove_columns: bool = False,
    allow_dtype_change: bool = True,
) -> SchemaValidationResult:
    result = SchemaValidationResult(ok=True)

    in_cols: dict[str, pl.DataType] = dict(zip(input_df.columns, input_df.dtypes))
    out_cols: dict[str, pl.DataType] = dict(zip(output_df.columns, output_df.dtypes))

    added = [c for c in out_cols if c not in in_cols]
    removed = [c for c in in_cols if c not in out_cols]

    result.added_columns = added
    result.removed_columns = removed

    if added and not allow_add_columns:
        for col in added:
            result.add(col, "Column was added but allow_add_columns=False")

    if removed and not allow_remove_columns:
        for col in removed:
            result.add(
                col,
                "Column was removed without being explicitly dropped "
                "(set allow_remove_columns=True or use a drop_columns step)",
            )

    for col in in_cols:
        if col not in out_cols:
            continue
        before = str(in_cols[col])
        after = str(out_cols[col])
        if before != after:
            result.changed_dtypes.append((col, before, after))
            if not allow_dtype_change:
                result.add(
                    col,
                    f"dtype changed from {before} to {after} "
                    "but allow_dtype_change=False",
                )

    if len(input_df) > 0 and len(output_df) == 0:
        result.add(None, "Output DataFrame is empty but input was not — possible logic error")

    return result