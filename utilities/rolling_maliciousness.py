"""
rolling_maliciousness.py

Turns a sequence of independent, single-snapshot maliciousness estimates
(one per timestamp, produced by timeline_runner.run_timeline) into a
smoothed trend a SOC analyst can actually act on.

Each snapshot's final_P_mal is a point-in-time estimate from independent
detectors; on its own it says nothing about whether the situation is
getting worse, improving, or just noisy. This module maintains an
exponentially weighted moving average (EWMA) across snapshots so that a
single anomalous (or falsely calm) reading doesn't dominate the picture,
while still reacting faster to real, sustained escalation than a plain
average over the full history would.

EWMA formula:
    rolling[t]  = alpha * value[t] + (1 - alpha) * rolling[t-1]
    rolling[t0] = value[t0]                      (seed on first snapshot)

alpha in (0, 1]:
    - higher alpha -> more weight on the newest snapshot (reacts faster,
      noisier -- closer to "just trust the latest reading")
    - lower alpha  -> more weight on accumulated history (smoother,
      slower to react to a genuine spike)
    - alpha = 1.0 disables smoothing entirely (rolling == raw value)

Why EWMA over a plain moving average: a fixed-window average weights the
oldest and newest snapshot in the window equally and drops old readings
off a cliff once they exit the window. EWMA instead decays older evidence
continuously, which better matches how a SOC analyst actually reasons
about a system's trajectory -- recent behavior matters more, but nothing
is discarded outright.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RollingMaliciousnessTracker:
    """
    Maintains EWMA state across snapshots as they arrive.

    Designed to be passed as the `on_step` callback into
    timeline_runner.run_timeline, so each result dict gets its rolling
    score attached the moment it's produced -- the same way a live
    monitor would update a running score as each new snapshot comes in,
    rather than only being computable after the fact over a finished
    history. This is also what a future container/replay loop would call
    on each tick.

    Tracks rolling scores for both base_P_mal and final_P_mal so the
    plotted trend shows the effect of CNN evidence on the trajectory, not
    just the raw detector-fusion output.
    """
    alpha: float = 0.3
    _rolling_base: Optional[float] = field(default=None, init=False, repr=False)
    _rolling_final: Optional[float] = field(default=None, init=False, repr=False)

    def step(self, result: dict) -> dict:
        """
        Updates the EWMA state with this snapshot's result and attaches
        the rolling scores onto the result dict in place -- so when used
        as run_timeline's on_step, the rolling values land directly in
        `history` with no separate post-processing pass required.
        """
        base_val = result["base_P_mal"]
        final_val = result["final_P_mal"]

        self._rolling_base = (
            base_val if self._rolling_base is None
            else self.alpha * base_val + (1 - self.alpha) * self._rolling_base
        )
        self._rolling_final = (
            final_val if self._rolling_final is None
            else self.alpha * final_val + (1 - self.alpha) * self._rolling_final
        )

        result["rolling_base_P_mal"] = self._rolling_base
        result["rolling_final_P_mal"] = self._rolling_final
        return result


def compute_rolling_maliciousness(history: list, alpha: float = 0.3) -> list:
    """
    Batch/offline equivalent of RollingMaliciousnessTracker.

    Computes rolling_base_P_mal and rolling_final_P_mal over an
    already-completed `history` list (must be in chronological order) and
    returns a NEW list with those fields added -- the input history is
    not mutated. Useful when:
      - history was produced without a tracker attached (e.g. loaded back
        from a persisted JSON file), or
      - you want to experiment with a different alpha without re-running
        inference on every snapshot.
    """
    tracker = RollingMaliciousnessTracker(alpha=alpha)
    return [tracker.step(dict(result)) for result in history]
