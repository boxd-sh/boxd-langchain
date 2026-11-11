"""Shared settings for the live boxd tests.

The tests create real machines. They need ``BOXD_API_KEY`` and are skipped
without it. Every machine gets a name starting with ``BOXD_TEST_NAME_PREFIX``
(default ``t-langchain-``), the labels in ``BOXD_TEST_LABELS`` (default
``boxd-langchain=integration-test``) and a one-hour ``delete_after`` timer, so a
crashed run cleans up after itself.
"""

from __future__ import annotations

import os
import uuid

import pytest

requires_boxd = pytest.mark.skipif(
    not os.environ.get("BOXD_API_KEY"),
    reason="BOXD_API_KEY not set; skipping live boxd tests",
)

SAFETY_DELETE_AFTER = 3600


def machine_name(suffix: str) -> str:
    prefix = os.environ.get("BOXD_TEST_NAME_PREFIX", "t-langchain-")
    return f"{prefix}{suffix}-{uuid.uuid4().hex[:8]}"


def machine_labels() -> dict[str, str]:
    raw = os.environ.get("BOXD_TEST_LABELS", "boxd-langchain=integration-test")
    pairs = (item.split("=", 1) for item in raw.split(",") if item)
    return {key.strip(): value.strip() for key, value in pairs}
