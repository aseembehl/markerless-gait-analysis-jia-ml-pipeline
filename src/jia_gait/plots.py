"""Gait-cycle plots: mean +- SD across cycles per joint and walking direction.

Port of the aggregate plot in ``plots_v7.ipynb`` (``plot_fun(mode="agg")``):
cycles (already interpolated and LOWESS-smoothed, see ``gait.cycles``) are
averaged; mean and SD (ddof=1) are smoothed for display with a Savitzky-Golay
filter (window 11, order 3, ``mode="interp"``) and drawn as a line with a
+-1 SD band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gait.cycles import GaitCycles

JOINT_TITLES = {
    "hip": "Flexion Hip Extension (deg)",
    "knee": "Flexion Knee Extension (deg)",
    "ankle": "Flexion Ankle Plantar Flexion (deg)",
}
DIRECTION_LABELS = {"ltr": "left to right", "rtl": "right to left"}


@dataclass
class PlotConfig:
    # y-axis limits per joint (from plots_v7); None -> automatic
    axis_range: dict = field(
        default_factory=lambda: {"hip": (-40, 60), "knee": (-20, 80), "ankle": (-70, 30)}
    )
    color: str = "red"
    show_cycles: bool = False  # draw individual cycles faintly
    savgol_window: int = 11
    savgol_order: int = 3
    figsize: tuple = (10, 5)
    dpi: int = 300
    transparent: bool = True


def mean_sd_display(Y: np.ndarray, cfg: PlotConfig) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD across cycles (rows), smoothed for display as in plots_v7."""
    from scipy.signal import savgol_filter

    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1) if len(Y) > 1 else np.zeros_like(mu)
    win = cfg.savgol_window
    if win >= len(mu):
        win = max(5, len(mu) // 2 * 2 - 1)
    mu = savgol_filter(mu, window_length=win, polyorder=cfg.savgol_order, mode="interp")
    sd = savgol_filter(sd, window_length=win, polyorder=cfg.savgol_order, mode="interp")
    return mu, sd


def plot_joint(Y: np.ndarray, grid: np.ndarray, joint: str, title: str, cfg: PlotConfig):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=cfg.figsize)
    if cfg.show_cycles:
        for y in Y:
            ax.plot(grid, y, color="gray", alpha=0.3, linewidth=0.8)
    mu, sd = mean_sd_display(Y, cfg)
    ax.fill_between(grid, mu - sd, mu + sd, alpha=0.2, color=cfg.color)
    ax.plot(grid, mu, color=cfg.color)

    ax.axhline(0, 0, 1, color="black")
    ax.set(xlabel="% Gait Cycle", ylabel=JOINT_TITLES[joint], title=title)
    ax.set_xlim(0, 1)
    ticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x * 100:.0f}" for x in ticks])
    rng = (cfg.axis_range or {}).get(joint)
    if rng is None:
        lo = np.floor((np.nanmin(Y) - 10) / 10) * 10
        hi = max(np.ceil((np.nanmax(Y) + 10) / 10) * 10, lo + 10)
        ax.set_yticks(np.arange(lo, hi + 1e-9, 10))
    else:
        ax.set_ylim(*rng)
        ax.set_yticks(np.arange(rng[0], rng[1] + 1e-9, 10))
    return fig


def save_cycle_plots(
    cycles: GaitCycles, out_dir: str | Path, name: str = "", cfg: PlotConfig | None = None
) -> list[Path]:
    """Write ``<joint>_<direction>.png`` for every direction with at least one cycle."""
    import matplotlib.pyplot as plt

    cfg = cfg or PlotConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*_[lr]t[lr].png"):  # plots from a previous run
        old.unlink()
    written = []
    for d in ("ltr", "rtl"):
        dc = cycles[d]
        if dc.n_cycles == 0:
            continue
        for joint, Y in dc.angles.items():
            title = (
                f"{name} {joint}, walking {DIRECTION_LABELS[d]} (n = {dc.n_cycles} cycles)".strip()
            )
            fig = plot_joint(Y, cycles.grid, joint, title, cfg)
            path = out_dir / f"{joint}_{d}.png"
            fig.savefig(path, dpi=cfg.dpi, transparent=cfg.transparent, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written
