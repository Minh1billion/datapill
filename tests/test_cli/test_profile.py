import re
import subprocess

import pytest

from conftest import dp


def _assert_profile_ok(result):
    combined = result.stdout + result.stderr
    assert re.search(r"Profile ID", combined, re.IGNORECASE), (
        f"profile output missing 'Profile ID'\n{combined[:500]}"
    )


def test_profile_full(sample_parquet, out_dir):
    result = dp("profile", "--input", str(sample_parquet), "--mode", "full", out_dir=out_dir)
    _assert_profile_ok(result)


def test_profile_summary_random_sample(sample_parquet, out_dir):
    result = dp(
        "profile", "--input", str(sample_parquet),
        "--mode", "summary",
        "--sample-strategy", "random",
        "--sample-size", "50",
        out_dir=out_dir,
    )
    _assert_profile_ok(result)