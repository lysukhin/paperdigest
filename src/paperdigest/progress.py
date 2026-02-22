"""Rich terminal progress tracking for pipeline stages."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, ProgressColumn, Task
from rich.table import Table
from rich.text import Text


class StageStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageInfo:
    name: str
    status: StageStatus = StageStatus.PENDING
    detail: str = ""
    cost: float = 0.0
    elapsed: float = 0.0
    completed: int = 0
    total: int = 0


class StageContext:
    """Handle passed to loop modules to report progress."""

    def __init__(self, info: StageInfo, live: Live):
        self._info = info
        self._live = live

    def advance(self, n: int = 1) -> None:
        self._info.completed += n
        self._live.refresh()

    def set_detail(self, text: str) -> None:
        self._info.detail = text
        self._live.refresh()

    def set_cost(self, cost: float) -> None:
        self._info.cost = cost
        self._live.refresh()

    @property
    def cost(self) -> float:
        return self._info.cost


class NullStageContext:
    """No-op context for non-interactive mode."""

    def advance(self, n: int = 1) -> None:
        pass

    def set_detail(self, text: str) -> None:
        pass

    def set_cost(self, cost: float) -> None:
        pass

    @property
    def cost(self) -> float:
        return 0.0


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _compute_eta(elapsed: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= 0:
        return ""
    rate = elapsed / completed
    remaining = rate * (total - completed)
    return _format_elapsed(remaining)


class ETAColumn(ProgressColumn):
    """Shows ETA based on StageInfo timing."""

    def render(self, task: Task) -> Text:
        eta = task.fields.get("eta", "")
        if eta:
            return Text(f"ETA {eta}", style="cyan")
        return Text("")


class PipelineTracker:
    """Rich live display of pipeline stages with progress bars and ETA."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._stages: list[StageInfo] = []
        self._live: Live | None = None

    def __enter__(self):
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live:
            # Final render with all completed stages
            self._live.update(self._render())
            self._live.__exit__(*exc)
            self._live = None

    def _render(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(width=1)   # icon
        table.add_column(width=10)  # name
        table.add_column(min_width=40)  # detail / progress
        table.add_column(justify="right", min_width=8)  # time/cost

        for info in self._stages:
            icon, style = self._stage_icon(info.status)
            detail = self._stage_detail(info)
            right = self._stage_right(info)
            table.add_row(
                Text(icon, style=style),
                Text(info.name, style="bold" if info.status == StageStatus.ACTIVE else ""),
                detail,
                right,
            )

        return table

    @staticmethod
    def _stage_icon(status: StageStatus) -> tuple[str, str]:
        return {
            StageStatus.PENDING: ("○", "dim"),
            StageStatus.ACTIVE: ("◉", "bold cyan"),
            StageStatus.COMPLETED: ("✓", "bold green"),
            StageStatus.SKIPPED: ("─", "dim"),
            StageStatus.FAILED: ("✗", "bold red"),
        }[status]

    @staticmethod
    def _stage_detail(info: StageInfo) -> Text:
        if info.status == StageStatus.ACTIVE and info.total > 0:
            pct = info.completed / info.total * 100
            filled = int(pct / 100 * 20)
            bar = "━" * filled + "╸" + "━" * (19 - filled) if filled < 20 else "━" * 20
            eta = _compute_eta(info.elapsed, info.completed, info.total)
            parts = [
                f"{bar}  {info.completed}/{info.total}  {pct:.0f}%",
            ]
            if eta:
                parts.append(f"ETA {eta}")
            return Text("  ".join(parts), style="cyan")

        if info.detail:
            style = "dim" if info.status != StageStatus.ACTIVE else ""
            return Text(info.detail, style=style)

        return Text("")

    @staticmethod
    def _stage_right(info: StageInfo) -> Text:
        parts = []
        if info.cost > 0:
            parts.append(f"${info.cost:.3f}")
        if info.elapsed > 0 and info.status in (StageStatus.COMPLETED, StageStatus.FAILED):
            parts.append(f"({_format_elapsed(info.elapsed)})")
        return Text("  ".join(parts), style="dim")

    @contextmanager
    def stage(self, name: str, total: int = 0) -> Generator[StageContext, None, None]:
        info = StageInfo(name=name, status=StageStatus.ACTIVE, total=total)
        self._stages.append(info)
        if self._live:
            self._live.update(self._render())

        start = time.monotonic()
        ctx = StageContext(info, self._live)
        try:
            yield ctx
            info.status = StageStatus.COMPLETED
        except Exception:
            info.status = StageStatus.FAILED
            raise
        finally:
            info.elapsed = time.monotonic() - start
            if self._live:
                self._live.update(self._render())

    def skip_stage(self, name: str) -> None:
        info = StageInfo(name=name, status=StageStatus.SKIPPED)
        self._stages.append(info)
        if self._live:
            self._live.update(self._render())


class NullTracker:
    """No-op tracker for non-interactive environments."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    @contextmanager
    def stage(self, name: str, total: int = 0) -> Generator[NullStageContext, None, None]:
        yield NullStageContext()

    def skip_stage(self, name: str) -> None:
        pass


def _is_interactive() -> bool:
    if os.environ.get("NO_PROGRESS"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def create_tracker(interactive: bool | None = None) -> PipelineTracker | NullTracker:
    """Factory: returns PipelineTracker for interactive terminals, NullTracker otherwise."""
    if interactive is None:
        interactive = _is_interactive()
    if interactive:
        return PipelineTracker()
    return NullTracker()
