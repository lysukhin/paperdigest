"""Tests for the pipeline progress tracking module."""

import time
from unittest.mock import patch

import pytest

from paperdigest.progress import (
    NullStageContext,
    NullTracker,
    PipelineTracker,
    StageStatus,
    _compute_eta,
    _format_elapsed,
    create_tracker,
)


class TestNullStageContext:
    """NullStageContext should be a safe no-op."""

    def test_advance_is_noop(self):
        ctx = NullStageContext()
        ctx.advance(10)  # should not raise

    def test_set_detail_is_noop(self):
        ctx = NullStageContext()
        ctx.set_detail("some detail")

    def test_set_cost_is_noop(self):
        ctx = NullStageContext()
        ctx.set_cost(1.23)

    def test_cost_returns_zero(self):
        ctx = NullStageContext()
        assert ctx.cost == 0.0


class TestNullTracker:
    """NullTracker should be a safe no-op context manager."""

    def test_context_manager(self):
        tracker = NullTracker()
        with tracker:
            pass  # should not raise

    def test_stage_yields_null_context(self):
        tracker = NullTracker()
        with tracker:
            with tracker.stage("Test", total=10) as ctx:
                assert isinstance(ctx, NullStageContext)
                ctx.advance(1)
                ctx.set_detail("detail")
                ctx.set_cost(0.5)

    def test_skip_stage_is_noop(self):
        tracker = NullTracker()
        with tracker:
            tracker.skip_stage("Filter")


class TestPipelineTracker:
    """PipelineTracker renders stages with rich."""

    def test_stage_lifecycle(self):
        tracker = PipelineTracker()
        with tracker:
            with tracker.stage("Fetch") as ctx:
                ctx.set_detail("100 papers")

        # After context exit, stage should be completed
        assert len(tracker._stages) == 1
        assert tracker._stages[0].status == StageStatus.COMPLETED
        assert tracker._stages[0].detail == "100 papers"

    def test_stage_with_progress(self):
        tracker = PipelineTracker()
        with tracker:
            with tracker.stage("Filter", total=50) as ctx:
                for _ in range(50):
                    ctx.advance(1)
                ctx.set_cost(0.05)

        info = tracker._stages[0]
        assert info.completed == 50
        assert info.total == 50
        assert info.cost == pytest.approx(0.05)
        assert info.status == StageStatus.COMPLETED

    def test_skip_stage(self):
        tracker = PipelineTracker()
        with tracker:
            tracker.skip_stage("Filter")

        assert len(tracker._stages) == 1
        assert tracker._stages[0].status == StageStatus.SKIPPED
        assert tracker._stages[0].name == "Filter"

    def test_failed_stage(self):
        tracker = PipelineTracker()
        with pytest.raises(ValueError):
            with tracker:
                with tracker.stage("Fetch") as ctx:
                    raise ValueError("test error")

        assert tracker._stages[0].status == StageStatus.FAILED

    def test_elapsed_time_tracked(self):
        tracker = PipelineTracker()
        with tracker:
            with tracker.stage("Fetch") as ctx:
                time.sleep(0.05)

        assert tracker._stages[0].elapsed >= 0.04

    def test_multiple_stages(self):
        tracker = PipelineTracker()
        with tracker:
            with tracker.stage("Fetch") as ctx:
                ctx.set_detail("done")
            tracker.skip_stage("Dedup")
            with tracker.stage("Filter", total=10) as ctx:
                for _ in range(10):
                    ctx.advance(1)

        assert len(tracker._stages) == 3
        assert tracker._stages[0].name == "Fetch"
        assert tracker._stages[1].name == "Dedup"
        assert tracker._stages[1].status == StageStatus.SKIPPED
        assert tracker._stages[2].name == "Filter"
        assert tracker._stages[2].completed == 10


class TestCreateTracker:
    """create_tracker factory tests."""

    def test_explicit_interactive_true(self):
        tracker = create_tracker(interactive=True)
        assert isinstance(tracker, PipelineTracker)

    def test_explicit_interactive_false(self):
        tracker = create_tracker(interactive=False)
        assert isinstance(tracker, NullTracker)

    def test_auto_detect_non_tty(self):
        with patch("paperdigest.progress.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = False
            tracker = create_tracker()
            assert isinstance(tracker, NullTracker)

    def test_auto_detect_tty(self):
        with patch("paperdigest.progress.sys") as mock_sys, \
             patch.dict("os.environ", {}, clear=False):
            mock_sys.stdout.isatty.return_value = True
            # Ensure NO_PROGRESS is not set
            import os
            os.environ.pop("NO_PROGRESS", None)
            tracker = create_tracker()
            assert isinstance(tracker, PipelineTracker)

    def test_no_progress_env_var(self):
        with patch.dict("os.environ", {"NO_PROGRESS": "1"}):
            tracker = create_tracker()
            assert isinstance(tracker, NullTracker)


class TestHelpers:
    """Tests for helper functions."""

    def test_format_elapsed(self):
        assert _format_elapsed(0) == "0:00"
        assert _format_elapsed(59) == "0:59"
        assert _format_elapsed(60) == "1:00"
        assert _format_elapsed(125) == "2:05"
        assert _format_elapsed(3661) == "61:01"

    def test_compute_eta_zero_completed(self):
        assert _compute_eta(10.0, 0, 100) == ""

    def test_compute_eta_zero_total(self):
        assert _compute_eta(10.0, 5, 0) == ""

    def test_compute_eta_reasonable(self):
        # 10 seconds elapsed, 5 of 10 done → 10s remaining
        eta = _compute_eta(10.0, 5, 10)
        assert eta == "0:10"

    def test_compute_eta_almost_done(self):
        # 9 seconds elapsed, 9 of 10 done → 1s remaining
        eta = _compute_eta(9.0, 9, 10)
        assert eta == "0:01"
