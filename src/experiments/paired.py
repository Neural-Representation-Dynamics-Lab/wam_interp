"""Paired inference helpers with common observation and seed."""

from dataclasses import replace
from typing import Any

from wam_interp.types import InferenceRequest


def run_paired_conditions(adapter: Any, request: InferenceRequest,
                          conditions: tuple[str, ...] = ("first_frame", "idm"),
                          recorder_factory=None) -> dict[str, Any]:
    """Run conditions on identical inputs; every request retains the same seed."""
    results = {}
    for condition in conditions:
        paired_request = replace(request, condition=condition)
        recorder = recorder_factory(condition) if recorder_factory is not None else None
        results[condition] = adapter.predict_action(paired_request, recorder=recorder)
    return results