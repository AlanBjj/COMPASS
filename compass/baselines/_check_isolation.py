"""Isolation guard for the baselines track (run before any baseline run / in CI).

Baselines reproduce OTHER people's methods. They must depend ONLY on the method-AGNOSTIC shared
layer (the vLLM client, dataset loaders, and the evaluation/judge harness — the shared "referee")
and NEVER on COMPASS's own method modules. Importing COMPASS method code risks contaminating a
baseline reproduction with our own approach (e.g. mimicking gate/fusion logic or tuning baselines
like our own hyperparameters), which would invalidate the controlled comparison.

This script scans compass/baselines/*.py and FAILS (exit 1) if any file imports a forbidden
module. Allowed shared deps: compass.llm, compass.datasets, compass.eval, compass.envload.

Usage: python -m compass.baselines._check_isolation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# COMPASS METHOD modules — baselines must never import these.
FORBIDDEN = ("gate", "decompose", "solvers", "fusion", "pipeline")

BASE = Path(__file__).resolve().parent
# `from ..gate ...`, `from ..solvers.x import`, `import compass.fusion`, `from compass.pipeline import`
PATTERNS = [re.compile(rf"from\s+\.\.{m}\b") for m in FORBIDDEN]
PATTERNS += [re.compile(rf"from\s+compass\.{m}\b") for m in FORBIDDEN]
PATTERNS += [re.compile(rf"import\s+compass\.{m}\b") for m in FORBIDDEN]


def main() -> int:
    violations = []
    for f in sorted(BASE.glob("*.py")):
        if f.name == "_check_isolation.py":
            continue
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in PATTERNS):
                violations.append(f"{f.name}:{i}: {line.strip()}")
    if violations:
        print("ISOLATION VIOLATION — baselines import COMPASS method modules:")
        for v in violations:
            print("  " + v)
        print(f"\nForbidden method modules: {', '.join(FORBIDDEN)}. "
              "Vendor what you need or use only compass.{llm,datasets,eval,envload}.")
        return 1
    print("OK: baselines isolated from COMPASS method modules "
          f"(scanned {len(list(BASE.glob('*.py')))} files; forbidden: {', '.join(FORBIDDEN)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
