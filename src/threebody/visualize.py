"""Lightweight visualization for three-body trajectories.

Requirements:
- Matplotlib + blitting (no matplotlib.animation.FuncAnimation)
- One figure, three colored traces, fading trail
- Optional energy/time panel
- Controls: Space pause/play, arrows scrub, scroll zoom, r reset
- Headless frame rendering: render_frames(states, out_dir, dpi=300)

This module is intentionally small and avoids framework-style abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def _extract_xy(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract planar (x,y) positions for each body over time.

    Supports state vectors of length 12 (planar) or 18 (spatial).

    Returns:
        x: (n,3)
        y: (n,3)
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("states must have shape (n, d)")

    d = states.shape[1]
    if d == 12:
        r = states[:, :6].reshape(-1, 3, 2)
        return r[:, :, 0], r[:, :, 1]
    if d == 18:
        r = states[:, :9].reshape(-1, 3, 3)
        return r[:, :, 0], r[:, :, 1]

    raise ValueError(f"Unsupported state dimension {d}; expected 12 or 18")


def _limits_from_xy(x: np.ndarray, y: np.ndarray, *, pad: float = 0.08) -> Tuple[float, float, float, float]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy)
    if span <= 0.0:
        span = 1.0

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad)
    return cx - half, cx + half, cy - half, cy + half


@dataclass(frozen=True)
class VisualizerConfig:
    # Trail is intentionally short-lived (visual emphasis on current motion).
    # This is a count of sampled points (>= 2 to draw at least one segment).
    trail_len: int = 30
    # Exponential fade: alpha_k = trail_alpha * trail_alpha_decay^k
    # Values closer to 1.0 make the trail more visible.
    trail_alpha: float = 1.0
    trail_alpha_decay: float = 0.9

    # Keep the trail thin; the glow layer provides perceived thickness.
    line_width: float = 1.0
    marker_size: float = 6.0
    fps: float = 120.0
    glow_width_factor: float = 5
    glow_alpha_factor: float = 0.4


class ThreeBodyVisualizer:
    """Blitted interactive viewer for three-body trajectories."""

    def __init__(
        self,
        t: Sequence[float] | None,
        states: np.ndarray,
        *,
        energy: Optional[Sequence[float]] = None,
        show_energy: bool = False,
        config: VisualizerConfig = VisualizerConfig(),
        fixed_limits: bool = True,
    ) -> None:
        self.states = np.asarray(states, dtype=float)
        if self.states.ndim != 2:
            raise ValueError("states must have shape (n, d)")
        self.n = int(self.states.shape[0])
        if self.n < 2:
            raise ValueError("states must contain at least 2 time points")

        if t is None:
            self.t = None
        else:
            self.t = np.asarray(t, dtype=float)
            if self.t.shape != (self.n,):
                raise ValueError("t must have shape (n,)")

        self.energy = None
        if energy is not None:
            e = np.asarray(energy, dtype=float)
            if e.shape != (self.n,):
                raise ValueError("energy must have shape (n,)")
            self.energy = e

        self.show_energy = bool(show_energy and self.energy is not None)
        self.config = config
        self.fixed_limits = bool(fixed_limits)

        self._paused = False
        self._i = 0

        self._x, self._y = _extract_xy(self.states)
        self._xlim0, self._xlim1, self._ylim0, self._ylim1 = _limits_from_xy(self._x, self._y)

        # Matplotlib objects are created in _build_figure
        self._fig = None
        self._ax = None
        self._axE = None
        self._background_main = None
        self._background_energy = None

        self._trail_collections = []
        self._trail_glow_collections = []
        self._markers = []
        self._glow_markers = []
        self._energy_line = None
        self._energy_marker = None
        self._time_text = None

        self._timer = None

        # Cached fade weights (oldest -> newest) for up to trail_len points.
        self._trail_alpha_lut = self._build_trail_alpha_lut()
        self._trail_core_rgb = None
        self._trail_glow_rgb = None

    def _set_dynamic_animated(self, animated: bool) -> None:
        """Toggle animated state of dynamic artists.

        Matplotlib typically does not draw `animated=True` artists during a normal
        full draw. We use animated=True for interactive blitting, but we must
        disable it for headless rendering (savefig / buffer_rgba).
        """
        for lc in self._trail_glow_collections:
            lc.set_animated(animated)
        for lc in self._trail_collections:
            lc.set_animated(animated)
        for m in self._glow_markers:
            m.set_animated(animated)
        for m in self._markers:
            m.set_animated(animated)
        if self._time_text is not None:
            self._time_text.set_animated(animated)
        if self._energy_marker is not None:
            self._energy_marker.set_animated(animated)

    def _build_trail_alpha_lut(self) -> np.ndarray:
        """Precompute alpha weights for segment fades to reduce per-frame work."""
        seg_max = max(0, int(self.config.trail_len) - 1)
        if seg_max == 0:
            return np.empty((0,), dtype=float)

        base = float(self.config.trail_alpha)
        decay = float(self.config.trail_alpha_decay)
        # Oldest -> newest (small -> large). We exponentiate in reverse so the newest
        # segment is closest to base.
        a = base * (decay ** np.arange(seg_max - 1, -1, -1))
        return np.clip(a.astype(float, copy=False), 0.0, 1.0)

    def _set_dynamic_visible(self, visible: bool) -> None:
        """Show/hide dynamic artists so background capture stays clean."""
        for lc in self._trail_glow_collections:
            lc.set_visible(visible)
        for lc in self._trail_collections:
            lc.set_visible(visible)
        for m in self._glow_markers:
            m.set_visible(visible)
        for m in self._markers:
            m.set_visible(visible)
        if self._time_text is not None:
            self._time_text.set_visible(visible)
        if self._energy_marker is not None:
            self._energy_marker.set_visible(visible)

    # ------------------------- Figure setup -------------------------

    def _build_figure(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        if self.show_energy:
            fig = plt.figure(figsize=(8.5, 7.0), constrained_layout=True)
            gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.2])
            ax = fig.add_subplot(gs[0, 0])
            axE = fig.add_subplot(gs[1, 0])
        else:
            fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
            axE = None

        self._fig = fig
        self._ax = ax
        self._axE = axE

        # Black-background style.
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Three-body trajectory")
        ax.set_xlim(self._xlim0, self._xlim1)
        ax.set_ylim(self._ylim0, self._ylim1)
        ax.grid(True, alpha=0.10, color="white")

        # Aesthetics: keep spines subtle.
        for spine in ax.spines.values():
            spine.set_alpha(0.4)
            spine.set_color("white")

        # Bright colors that read on black.
        colors = ["#00D1FF", "#FF4D9D", "#9DFF00"]
        # Cache RGB for fast per-frame coloring.
        # Trail core is always white; glow is per-body.
        self._trail_core_rgb = np.array([1.0, 1.0, 1.0], dtype=float)
        self._trail_glow_rgb = [np.array([1.0, 1.0, 1.0], dtype=float) for _ in range(3)]

        for k in range(3):
            glow_lc = LineCollection(
                [],
                linewidths=self.config.line_width * self.config.glow_width_factor,
                colors=[colors[k]],
                zorder=0,
            )
            # Important: do not set a collection-wide alpha here.
            # We provide per-segment RGBA (including alpha) in _draw_frame().
            glow_lc.set_alpha(None)
            glow_lc.set_animated(True)
            ax.add_collection(glow_lc)
            self._trail_glow_collections.append(glow_lc)

            lc = LineCollection(
                [],
                linewidths=self.config.line_width,
                # White core, colored glow (underlay).
                colors=["white"],
                zorder=1,
            )
            # Per-segment alpha is provided via RGBA in _draw_frame().
            lc.set_alpha(None)
            lc.set_animated(True)
            ax.add_collection(lc)
            self._trail_collections.append(lc)

            # Store this body's glow RGB as floats in [0,1].
            try:
                import matplotlib.colors as mcolors

                self._trail_glow_rgb[k] = np.array(mcolors.to_rgb(colors[k]), dtype=float)
            except Exception:
                self._trail_glow_rgb[k] = np.array([1.0, 1.0, 1.0], dtype=float)

            # Particle glow (underlay)
            (glow_marker,) = ax.plot(
                [],
                [],
                marker="o",
                markersize=self.config.marker_size * 2.6,
                color=colors[k],
                alpha=self.config.glow_alpha_factor,
                linestyle="None",
                markeredgewidth=0.0,
                zorder=2,
            )
            glow_marker.set_animated(True)
            self._glow_markers.append(glow_marker)

            # Particle core: white fill with colored outline.
            (marker,) = ax.plot(
                [],
                [],
                marker="o",
                markersize=self.config.marker_size,
                markerfacecolor="white",
                markeredgecolor=colors[k],
                markeredgewidth=1.4,
                linestyle="None",
                zorder=3,
            )
            marker.set_animated(True)
            self._markers.append(marker)

        self._time_text = ax.text(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
        )
        self._time_text.set_animated(True)

        if self.show_energy and axE is not None:
            axE.set_facecolor("black")
            axE.set_xlabel("t")
            axE.set_ylabel("E")
            axE.grid(True, alpha=0.10, color="white")
            axE.tick_params(colors="white")
            axE.xaxis.label.set_color("white")
            axE.yaxis.label.set_color("white")
            for spine in axE.spines.values():
                spine.set_alpha(0.4)
                spine.set_color("white")

            tE = np.arange(self.n) if self.t is None else self.t
            (line,) = axE.plot(tE, self.energy, color="0.75", linewidth=1.2)
            (marker,) = axE.plot([tE[0]], [self.energy[0]], marker="o", color="white", markersize=4)

            self._energy_line = line
            self._energy_marker = marker
            self._energy_marker.set_animated(True)

        # Hook events
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        fig.canvas.mpl_connect("resize_event", self._on_resize)

        # Timer for playback
        interval_ms = max(1, int(1000.0 / float(self.config.fps)))
        self._timer = fig.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._on_timer)

        # Capture a background that does not include dynamic artists (time text,
        # markers, trails). Otherwise blitting will "paint over" old text.
        self._capture_backgrounds()

        # Draw first frame onto the cached background.
        self._draw_frame(self._i)
        self._blit()

    def _capture_backgrounds(self) -> None:
        if self._fig is None or self._ax is None:
            return

        canvas = self._fig.canvas

        # Hide dynamic artists for a clean background.
        self._set_dynamic_visible(False)
        canvas.draw()

        self._background_main = canvas.copy_from_bbox(self._ax.bbox)
        if self._axE is not None:
            self._background_energy = canvas.copy_from_bbox(self._axE.bbox)

        # Restore visibility; dynamic artists are drawn via blitting.
        self._set_dynamic_visible(True)

    # ------------------------- Updates -------------------------

    def _segments_with_alpha(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (segments, rgba) for a fading trail."""
        # x,y are 1D arrays for a single body.
        n = x.shape[0]
        if n < 2:
            return np.empty((0, 2, 2), dtype=float), np.empty((0, 4), dtype=float)

        # Use only the last trail_len points.
        trail = int(self.config.trail_len)
        start = max(0, n - trail)
        xs = x[start:n]
        ys = y[start:n]
        m = xs.shape[0]
        if m < 2:
            return np.empty((0, 2, 2), dtype=float), np.empty((0, 4), dtype=float)

        seg = np.stack(
            [
                np.stack([xs[:-1], ys[:-1]], axis=1),
                np.stack([xs[1:], ys[1:]], axis=1),
            ],
            axis=1,
        )

        # Alpha decay from oldest to newest (cached lookup).
        seg_count = int(seg.shape[0])
        if seg_count == 0:
            return seg, np.empty((0, 4), dtype=float)
        # For early frames we need only the last seg_count alphas.
        alphas = self._trail_alpha_lut[-seg_count:] if self._trail_alpha_lut.size else np.ones((seg_count,))

        rgba = np.ones((seg.shape[0], 4), dtype=float)
        rgba[:, 3] = alphas
        return seg, rgba

    def _draw_frame(self, i: int) -> None:
        if self._ax is None:
            return

        i = int(np.clip(i, 0, self.n - 1))
        self._i = i

        # Update each body
        for k in range(3):
            seg, rgba = self._segments_with_alpha(self._x[: i + 1, k], self._y[: i + 1, k])
            glow_lc = self._trail_glow_collections[k]
            lc = self._trail_collections[k]

            glow_lc.set_segments(seg)
            lc.set_segments(seg)

            # White core (thin) + colored glow (thick), both with the same fade.
            core_rgb = self._trail_core_rgb if self._trail_core_rgb is not None else np.array([1.0, 1.0, 1.0])
            glow_rgb = (
                self._trail_glow_rgb[k]
                if self._trail_glow_rgb is not None
                else np.array([1.0, 1.0, 1.0])
            )

            rgba_core = rgba.copy()
            rgba_core[:, :3] = core_rgb
            lc.set_color(rgba_core)

            rgba_glow = rgba.copy()
            rgba_glow[:, :3] = glow_rgb
            rgba_glow[:, 3] *= float(self.config.glow_alpha_factor)
            glow_lc.set_color(rgba_glow)

            self._glow_markers[k].set_data([self._x[i, k]], [self._y[i, k]])
            self._markers[k].set_data([self._x[i, k]], [self._y[i, k]])

        if self._time_text is not None:
            if self.t is None:
                self._time_text.set_text(f"frame {i+1}/{self.n}")
            else:
                self._time_text.set_text(f"t = {self.t[i]:.6g}   ({i+1}/{self.n})")

        if self.show_energy and self._axE is not None and self._energy_marker is not None:
            tE = np.arange(self.n) if self.t is None else self.t
            self._energy_marker.set_data([tE[i]], [self.energy[i]])

    def _blit(self) -> None:
        if self._fig is None or self._ax is None:
            return

        canvas = self._fig.canvas

        if self._background_main is not None:
            canvas.restore_region(self._background_main)

        # Draw updated artists onto restored background.
        for lc in self._trail_glow_collections:
            self._ax.draw_artist(lc)
        for lc in self._trail_collections:
            self._ax.draw_artist(lc)
        for m in self._glow_markers:
            self._ax.draw_artist(m)
        for m in self._markers:
            self._ax.draw_artist(m)
        if self._time_text is not None:
            self._ax.draw_artist(self._time_text)

        canvas.blit(self._ax.bbox)

        if self._axE is not None and self._background_energy is not None:
            canvas.restore_region(self._background_energy)
            if self._energy_line is not None:
                self._axE.draw_artist(self._energy_line)
            if self._energy_marker is not None:
                self._axE.draw_artist(self._energy_marker)
            canvas.blit(self._axE.bbox)

        # Avoid flush_events() on every frame; it can introduce jitter on some backends.

    # ------------------------- Interaction -------------------------

    def _on_timer(self) -> None:
        if not self._paused:
            self._i = (self._i + 1) % self.n
            self._draw_frame(self._i)
            self._blit()

    def _on_resize(self, _event) -> None:
        # Resize invalidates backgrounds.
        if self._fig is None:
            return
        self._fig.canvas.draw()
        self._capture_backgrounds()
        self._blit()

    def _on_key(self, event) -> None:
        if event.key == " ":
            self._paused = not self._paused
            return

        if event.key == "right":
            self._paused = True
            self._draw_frame(self._i + 1)
            self._blit()
            return

        if event.key == "left":
            self._paused = True
            self._draw_frame(self._i - 1)
            self._blit()
            return

        if event.key == "up":
            self._paused = True
            self._draw_frame(self._i + 10)
            self._blit()
            return

        if event.key == "down":
            self._paused = True
            self._draw_frame(self._i - 10)
            self._blit()
            return

        if event.key == "r":
            self._paused = True
            self.reset_view()
            self._draw_frame(self._i)
            if self._fig is not None:
                self._fig.canvas.draw()
                self._capture_backgrounds()
            self._blit()
            return

    def _on_scroll(self, event) -> None:
        if self._ax is None:
            return
        if event.inaxes != self._ax:
            return

        # Zoom towards the mouse position.
        x = float(event.xdata) if event.xdata is not None else 0.0
        y = float(event.ydata) if event.ydata is not None else 0.0

        zoom = 0.9 if event.button == "up" else 1.1

        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()

        nx0 = x + (x0 - x) * zoom
        nx1 = x + (x1 - x) * zoom
        ny0 = y + (y0 - y) * zoom
        ny1 = y + (y1 - y) * zoom

        self._ax.set_xlim(nx0, nx1)
        self._ax.set_ylim(ny0, ny1)

        # Re-capture background after changing limits.
        if self._fig is not None:
            self._fig.canvas.draw()
            self._capture_backgrounds()
        self._blit()

    # ------------------------- Public API -------------------------

    def reset_view(self) -> None:
        """Reset axis limits to the orbit's fixed bounding box."""
        if self._ax is None:
            return
        self._ax.set_xlim(self._xlim0, self._xlim1)
        self._ax.set_ylim(self._ylim0, self._ylim1)

    def show(self) -> None:
        """Start the interactive viewer."""
        self._build_figure()
        if self._timer is not None:
            self._timer.start()
        import matplotlib.pyplot as plt

        plt.show()

    def save_frame(self, i: int, path: Path | str, *, dpi: int = 300) -> None:
        """Render and save a single frame as an image."""
        # Create figure lazily if needed.
        if self._fig is None:
            self._build_figure()
            if self.fixed_limits:
                self.reset_view()

        self._paused = True
        self._draw_frame(i)

        # Disable blit-only rendering so saved frames include trails/markers.
        self._set_dynamic_animated(False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi)
        self._set_dynamic_animated(True)

    def render_frame_rgb(self, i: int, *, dpi: int = 300) -> np.ndarray:
        """Render a frame and return an RGB uint8 array.

        Notes:
            This is intended for headless/ffmpeg streaming.
        """
        if self._fig is None:
            self._build_figure()
            if self.fixed_limits:
                self.reset_view()

        self._paused = True
        self._draw_frame(i)

        # Disable blit-only rendering so the buffer includes trails/markers.
        self._set_dynamic_animated(False)
        self._fig.canvas.draw()

        # Agg provides a fast RGBA buffer.
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        rgb = np.ascontiguousarray(buf[..., :3])
        self._set_dynamic_animated(True)
        return rgb


def visualize(
    states: np.ndarray,
    *,
    t: Sequence[float] | None = None,
    energy: Sequence[float] | None = None,
    show_energy: bool = False,
    trail_len: int = 30,
    fps: float = 120.0,
) -> None:
    """Convenience wrapper to launch the interactive visualizer."""
    cfg = VisualizerConfig(trail_len=int(trail_len), fps=float(fps))
    ThreeBodyVisualizer(t, states, energy=energy, show_energy=show_energy, config=cfg).show()


def render_frames(states: np.ndarray, out_dir: Path | str, dpi: int = 300) -> None:
    """Render PNG frames for a trajectory in headless/batch mode.

    Args:
        states: Array of shape (n, d).
        out_dir: Directory to write frames to.
        dpi: Figure DPI.

    Notes:
        - Uses fixed axis limits computed from the full orbit.
        - Does not show any UI.
        - Produces frame_000000.png, frame_000001.png, ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure a non-interactive backend for batch rendering.
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            # If pyplot is already imported in the process, backend switching may fail.
            # In that case, saving still usually works; we just proceed.
            pass

    n = int(np.asarray(states).shape[0])
    # IMPORTANT: use a nontrivial trail length for headless rendering.
    cfg = VisualizerConfig(trail_len=30)
    viz = ThreeBodyVisualizer(t=None, states=states, show_energy=False, fixed_limits=True, config=cfg)
    for i in range(n):
        viz.save_frame(i, out_dir / f"frame_{i:06d}.png", dpi=dpi)


def render_gif_ffmpeg(
    states: np.ndarray,
    out_gif: Path | str,
    *,
    fps: float = 60.0,
    dpi: int = 300,
    trail_len: int = 30,
) -> None:
    """Render a GIF by streaming raw frames to ffmpeg (no intermediate PNGs).

    Uses `imageio_ffmpeg` to locate/provide the ffmpeg executable (no PATH requirement).
    """
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "imageio-ffmpeg is required to render GIFs via ffmpeg. Install with: pip install imageio-ffmpeg"
        ) from e

    # Ensure a non-interactive backend for batch rendering.
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass

    states = np.asarray(states)
    n = int(states.shape[0])
    if n < 2:
        raise ValueError("states must contain at least 2 time points")

    cfg = VisualizerConfig(trail_len=int(trail_len), fps=float(fps))
    viz = ThreeBodyVisualizer(t=None, states=states, show_energy=False, fixed_limits=True, config=cfg)

    # Prime figure and get dimensions.
    rgb0 = viz.render_frame_rgb(0, dpi=dpi)
    h, w, _ = rgb0.shape

    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # High-quality GIF palette generation in one pass.
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-filter_complex",
        "[0:v]split[s0][s1];[s0]palettegen=stats_mode=diff:max_colors=256[p];[s1][p]paletteuse=dither=none",
        "-loop",
        "0",
        str(out_gif),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        proc.stdin.write(rgb0.tobytes())
        for i in range(1, n):
            rgb = viz.render_frame_rgb(i, dpi=dpi)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                raise RuntimeError("Frame size changed during rendering; cannot stream to ffmpeg")
            proc.stdin.write(rgb.tobytes())
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")
