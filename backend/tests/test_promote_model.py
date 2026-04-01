"""Unit tests for promotion hardening rules."""

from scripts.promote_model import _promotion_gate_status


def test_promotion_gate_status_accepts_strict_walk_forward_pass() -> None:
    """Strictly validated payloads should remain promotable."""
    ok, reasons = _promotion_gate_status(
        {
            "promotion_gate": {"passed": True, "failed_reasons": []},
            "walk_forward_enforced": True,
            "walk_forward_gate": {"enabled": True, "strict_pass_all_folds": True},
            "data_leakage_checks": {"pass": True},
            "martingale_enforced": False,
        }
    )

    assert ok is True
    assert reasons == []


def test_promotion_gate_status_rejects_non_enforced_walk_forward() -> None:
    """Diagnostic-only walk-forward should not count as a trusted promotion."""
    ok, reasons = _promotion_gate_status(
        {
            "promotion_gate": {"passed": True, "failed_reasons": []},
            "walk_forward_enforced": False,
            "walk_forward_gate": {"enabled": True, "strict_pass_all_folds": True},
        }
    )

    assert ok is False
    assert reasons == ["walk_forward_gate_not_enforced"]


def test_promotion_gate_status_rejects_failed_strict_walk_forward() -> None:
    """Strict walk-forward must pass every required fold."""
    ok, reasons = _promotion_gate_status(
        {
            "promotion_gate": {"passed": True, "failed_reasons": []},
            "walk_forward_enforced": True,
            "walk_forward_gate": {"enabled": True, "strict_pass_all_folds": False},
        }
    )

    assert ok is False
    assert reasons == ["walk_forward_gate_not_passed"]
