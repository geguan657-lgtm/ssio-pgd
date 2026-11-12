"""Tiny test runner so the suite works without pytest.

    python tests/run_tests.py            # run everything
    python tests/run_tests.py metrics    # run one module
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    pattern = argv[0] if argv else ""
    files = sorted(p for p in Path(__file__).parent.glob("test_*.py") if pattern in p.name)
    if not files:
        print(f"no test files matched {pattern!r}")
        return 2

    passed, failed = 0, []
    for path in files:
        mod = load_module(path)
        names = [n for n in dir(mod) if n.startswith("test_")]
        if not names:
            continue
        print(f"\n{path.name}")
        for name in names:
            try:
                getattr(mod, name)()
            except Exception:  # noqa: BLE001 - a failing test must not stop the run
                failed.append(f"{path.stem}::{name}")
                print(f"  FAIL  {name}")
                traceback.print_exc(limit=2)
            else:
                passed += 1
                print(f"  ok    {name}")

    print(f"\n{passed} passed, {len(failed)} failed")
    for name in failed:
        print(f"  - {name}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))