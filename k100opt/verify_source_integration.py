#!/usr/bin/env python3
"""Static and dependency-free checks for the K100_AI source integration."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
INTEGRATION = PYTHON_ROOT / "sglang/srt/hcu_optim/integration.py"


def load_integration():
    spec = importlib.util.spec_from_file_location("hcu_integration_check", INTEGRATION)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_path(module_name: str) -> Path:
    return PYTHON_ROOT / Path(*module_name.split(".")).with_suffix(".py")


def check_source_points(integration) -> None:
    for target, entries in integration.TARGETS.items():
        target_path = module_path(target)
        assert target_path.is_file(), f"missing target: {target_path}"
        tree = ast.parse(target_path.read_text(encoding="utf-8"), str(target_path))
        calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_apply_hcu_optim"
        ]
        assert len(calls) == 1, f"{target}: expected one module-level integration call"
        assert tree.body[-1] is calls[0], f"{target}: integration call must be last"
        for implementation, env_name in entries:
            assert env_name.startswith("SGLANG_")
            implementation_path = module_path(implementation)
            assert implementation_path.is_file(), f"missing implementation: {implementation_path}"


def check_default_off(integration) -> None:
    target_name = next(iter(integration.TARGETS))
    target = ModuleType(target_name)
    feature_flags = {
        flag for entries in integration.TARGETS.values() for _, flag in entries
    }
    clean_env = {key: value for key, value in os.environ.items() if key not in feature_flags}
    with patch.dict(os.environ, clean_env, clear=True), patch.object(
        integration.importlib, "import_module"
    ) as importer:
        integration.apply_for_module(target)
        importer.assert_not_called()


def check_enabled_dispatch(integration) -> None:
    target_name, entries = next(iter(integration.TARGETS.items()))
    implementation, flag = entries[0]
    target = ModuleType(target_name)
    installed = []
    extension = ModuleType(implementation)
    extension.install = installed.append
    with patch.dict(os.environ, {flag: "1"}, clear=True), patch.object(
        integration.importlib, "import_module", return_value=extension
    ):
        integration.apply_for_module(target)
    assert installed == [target], "enabled feature was not dispatched to its owner"


def check_legacy_overlay_removed() -> None:
    assert not (ROOT / "k100opt/patch/sitecustomize.py").exists()
    tracked_patch = ROOT / "k100opt/patch/kpatch"
    assert not tracked_patch.exists() or not any(tracked_patch.rglob("*.py"))
    for script in (ROOT / "k100opt").glob("start*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "k100opt/patch" not in text, f"legacy PYTHONPATH remains in {script}"


def main() -> None:
    integration = load_integration()
    check_source_points(integration)
    check_default_off(integration)
    check_enabled_dispatch(integration)
    check_legacy_overlay_removed()
    print(
        f"PASS: {len(integration.TARGETS)} target modules; "
        f"{sum(len(v) for v in integration.TARGETS.values())} feature bindings; "
        "default-off and legacy-overlay checks passed"
    )


if __name__ == "__main__":
    main()
