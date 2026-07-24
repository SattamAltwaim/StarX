import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FIXTURE = REPO / "tests" / "fixtures" / "20203_7e31e92a_0000.json"


@pytest.fixture(scope="session")
def fixture_design():
    from starx import fusion

    return fusion.load_design(FIXTURE)


def triposr_dir():
    """Locate a local TripoSR clone for tests that compare against tsr.utils."""
    candidates = [REPO / "third_party" / "TripoSR"]
    env = os.environ.get("STARX_TRIPOSR_DIR")
    if env:
        candidates.insert(0, Path(env))
    for cand in candidates:
        if (cand / "tsr" / "utils.py").exists():
            return cand
    return None
