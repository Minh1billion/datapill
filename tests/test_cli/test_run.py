import re

from .conftest import dp


def test_dp_run_pipeline(run_config, out_dir):
    result = dp("run", str(run_config), out_dir=out_dir)
    combined = result.stdout + result.stderr
    assert re.search(r"Ingest complete.*[1-9][0-9]* rows", combined, re.IGNORECASE), (
        f"dp run did not complete successfully\n{combined[:500]}"
    )