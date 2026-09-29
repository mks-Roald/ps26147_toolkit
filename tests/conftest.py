"""Emit a machine-readable result ledger for end-to-end regression tests."""
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if "regression" not in item.keywords or report.when != "call":
        return
    record = getattr(item, "_regression_record", {})
    record.update({"test_name": item.nodeid, "pass": report.passed})
    if report.failed:
        record["error"] = report.longreprtext[-2000:]
    ledger = getattr(item.config, "_regression_ledger", [])
    ledger.append(record)
    item.config._regression_ledger = ledger


def pytest_sessionfinish(session, exitstatus):
    ledger = getattr(session.config, "_regression_ledger", [])
    if not ledger:
        return
    target = Path(os.environ.get("PS26147_REGRESSION_REPORT", "regression-report.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"suite": "PS26147 end-to-end regression", "results": ledger,
                                  "passed": sum(row["pass"] for row in ledger),
                                  "failed": sum(not row["pass"] for row in ledger)}, indent=2), encoding="utf-8")
