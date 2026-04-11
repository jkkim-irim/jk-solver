"""StatusWindow — dark-themed tkinter debug monitor for contact simulations.

Layout: left=tabbed notebook (Solver/Objects/Contact), right=solver conv log.
Contact tab includes matplotlib impedance plot with per-geom and blended curves.
All public methods are thread-safe (called from main sim thread).
"""

import threading
import tkinter as tk
from tkinter import ttk


class StatusWindow:
    """Dark-themed tkinter debug monitor.

    Parameters
    ----------
    title : str
        Window title.
    width, height : int
        Initial window size.
    geom_labels : list[str]
        Labels for per-geom impedance plot buttons (e.g. ["ground", "box", "sph_L", "sph_R"]).
    """

    _FONT = ("nimbus mono l", 15, "bold")
    _FONT_TITLE = ("nimbus mono l", 15, "bold")
    _FONT_LOG = ("nimbus mono l", 12, "bold")
    _BG = "#1e1e1e"
    _FG = "#d4d4d4"
    _FG_DIM = "#6a6a6a"
    _FG_HEADER = "#569cd6"
    _FG_VALUE = "#ce9178"
    _FG_HIT = "#f44747"
    _FG_OK = "#6a9955"

    def __init__(self, title="Contact Debug Monitor", width=2400, height=1200,
                 geom_labels=None):
        from collections import deque
        self._data = {}
        self._data_dirty = False
        self._conv_lines: list[tuple[str, bool, bool]] = []
        self._conv_dirty = False
        self._conv_summary = ""
        self._conv_summary_dirty = False
        self._closed = False
        self._lock = threading.Lock()
        self._solref_solimp_cache = {}
        self._active_pairs = []
        self._pair_buttons = {}
        self._geom_labels = geom_labels or []
        # Joint constraint data (for time-series plot)
        self._joint_buf_maxlen = 2000
        self._joint_t = deque(maxlen=self._joint_buf_maxlen)
        self._joint_efc_force = deque(maxlen=self._joint_buf_maxlen)
        self._joint_efc_pos = deque(maxlen=self._joint_buf_maxlen)
        self._joint_pip_torque = deque(maxlen=self._joint_buf_maxlen)
        self._joint_plot_dirty = False
        # Joint constraint conv log
        self._joint_conv_lines: list[str] = []
        self._joint_conv_dirty = False
        self._thread = threading.Thread(target=self._run, args=(title, width, height), daemon=True)
        self._thread.start()

    # ── Public API (called from main thread) ──

    def update(self, data: dict):
        """Update tab contents. Keys: 'objects', 'contact', 'solver' -> multiline strings."""
        if self._closed:
            return
        self._data = data
        self._data_dirty = True

    def push_conv(self, line: str, hit_limit: bool = False, no_contact: bool = False):
        """Append one line to solver conv log."""
        if self._closed:
            return
        with self._lock:
            self._conv_lines.append((line, hit_limit, no_contact))
            self._conv_dirty = True

    def update_conv_summary(self, text: str):
        """Update the [current] summary bar above the conv log."""
        if self._closed:
            return
        self._conv_summary = text
        self._conv_summary_dirty = True

    def update_solref_solimp(self, cache: dict):
        """Update per-geom cache. {label: ([tc,dr], [dmin,dmax,w,mid,pow], solmix)}."""
        if self._closed:
            return
        self._solref_solimp_cache = cache

    def update_active_pairs(self, pairs: list):
        """Dynamically create/remove blended sigmoid buttons for active contact pairs."""
        if self._closed:
            return
        current = set(self._pair_buttons.keys())
        new = set(pairs)
        if current == new:
            return
        for w in self._pair_btn_frame.winfo_children():
            w.destroy()
        self._pair_buttons.clear()
        for la, lb in sorted(pairs):
            key = (la, lb)
            b = tk.Button(self._pair_btn_frame, text=f"{la}<->{lb}", font=("Courier", 8),
                          bg="#2d2d2d", fg=self._FG_VALUE, bd=0, padx=6, pady=2,
                          command=lambda a=la, b=lb: self._plot_blended_impedance(a, b))
            b.pack(side=tk.LEFT, padx=3)
            self._pair_buttons[key] = b

    def push_joint_data(self, sim_time, efc_force, efc_pos, pip_torque):
        """Append one sample of Index DIP constraint data for time-series plot."""
        if self._closed:
            return
        with self._lock:
            self._joint_t.append(sim_time)
            self._joint_efc_force.append(efc_force)
            self._joint_efc_pos.append(efc_pos)
            self._joint_pip_torque.append(pip_torque)
            self._joint_plot_dirty = True

    def push_joint_conv(self, line: str):
        """Append one line to the Joint Constraint log tab."""
        if self._closed:
            return
        with self._lock:
            self._joint_conv_lines.append(line)
            self._joint_conv_dirty = True

    def clear_joint_data(self):
        """Clear joint time-series buffers and redraw empty plot."""
        if self._closed:
            return
        with self._lock:
            self._joint_t.clear()
            self._joint_efc_force.clear()
            self._joint_efc_pos.clear()
            self._joint_pip_torque.clear()
            self._joint_plot_dirty = True

    # ── Internal: tkinter setup (runs in daemon thread) ──

    def _make_text_widget(self, parent):
        t = tk.Text(parent, font=self._FONT, bg=self._BG, fg=self._FG,
                    wrap=tk.NONE, state=tk.DISABLED, bd=0, padx=10, pady=8)
        t.tag_configure("header", foreground=self._FG_HEADER, font=self._FONT_TITLE)
        t.tag_configure("value", foreground=self._FG_VALUE)
        t.tag_configure("dim", foreground=self._FG_DIM)
        return t

    @staticmethod
    def _get_primary_monitor():
        """Return (width, height, x_offset, y_offset) of the primary monitor via xrandr."""
        import subprocess
        result = subprocess.run(["xrandr", "--query"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "primary" in line and " connected" in line:
                for part in line.split():
                    if "x" in part and "+" in part:
                        res, ox, oy = part.split("+")[0], part.split("+")[1], part.split("+")[2]
                        mw, mh = res.split("x")
                        return int(mw), int(mh), int(ox), int(oy)
        return None, None, None, None

    def _run(self, title, width, height):
        self._root = tk.Tk()
        self._root.title(title)
        mw, mh, mx, my = self._get_primary_monitor()
        if mw and mh:
            w = max(width, int(mw * 0.8))
            h = max(height, int(mh * 0.6))
            x = mx + (mw - w) // 2
            y = my + (mh - h) // 2
        else:
            w, h, x, y = width, height, 0, 0
        self._root.geometry(f"{w}x{h}+{x}+{y}")
        self._root.configure(bg=self._BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_styles()
        pane = tk.PanedWindow(self._root, orient=tk.HORIZONTAL, sashwidth=6, bg="#3c3c3c", bd=0)
        pane.pack(fill=tk.BOTH, expand=True)
        self._setup_left_panel(pane, width)
        self._setup_right_panel(pane)
        self._poll()
        self._root.mainloop()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=self._BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background="#2d2d2d", foreground=self._FG,
                        font=("Courier", 11, "bold"), padding=[8, 3])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#3c3c3c")],
                  foreground=[("selected", "#ffffff")])
        style.configure("Dark.TFrame", background=self._BG)

    def _setup_left_panel(self, pane, width):
        left = tk.Frame(pane, bg=self._BG)
        nb = ttk.Notebook(left, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        # Solver tab
        f_sol = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_solver = self._make_text_widget(f_sol)
        self._txt_solver.pack(fill=tk.BOTH, expand=True)
        nb.add(f_sol, text=" Solver ")

        # Objects tab
        f_obj = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_objects = self._make_text_widget(f_obj)
        self._txt_objects.pack(fill=tk.BOTH, expand=True)
        nb.add(f_obj, text=" Objects ")

        # Contact tab (text + plot)
        f_con = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_contact = self._make_text_widget(f_con)
        self._txt_contact.config(height=15)
        self._txt_contact.pack(fill=tk.X)
        self._setup_contact_plot(f_con)
        nb.add(f_con, text=" Contact ")

        # Joint tab (time-series plot: efc_force, efc_pos, pip_torque)
        f_jnt = ttk.Frame(nb, style="Dark.TFrame")
        self._setup_joint_plot(f_jnt)
        nb.add(f_jnt, text=" Joint ")

        pane.add(left, stretch="always", minsize=100)

    def _setup_contact_plot(self, parent):
        """Create geom buttons + matplotlib canvas for impedance plots."""
        plot_frame = tk.Frame(parent, bg=self._BG)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(plot_frame, bg=self._BG)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        for label in self._geom_labels:
            tk.Button(btn_frame, text=label, font=("Courier", 9),
                      bg="#3c3c3c", fg=self._FG, bd=0, padx=10, pady=2,
                      command=lambda l=label: self._plot_impedance(l)).pack(side=tk.LEFT, padx=4)
        tk.Label(btn_frame, text=" | ", font=("Courier", 9), bg=self._BG, fg=self._FG_DIM).pack(side=tk.LEFT)
        self._pair_btn_frame = tk.Frame(btn_frame, bg=self._BG)
        self._pair_btn_frame.pack(side=tk.LEFT)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(8, 3), dpi=80, facecolor=self._BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(self._BG)
        self._ax.tick_params(colors=self._FG, labelsize=12)
        for spine in self._ax.spines.values():
            spine.set_color(self._FG_DIM)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_joint_plot(self, parent):
        """Create matplotlib figure with 3 subplots for Index DIP constraint data."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._jfig = Figure(figsize=(8, 6), dpi=120, facecolor=self._BG)
        self._jax_force = self._jfig.add_subplot(311)
        self._jax_pos = self._jfig.add_subplot(312)
        self._jax_torque = self._jfig.add_subplot(313)
        for ax in (self._jax_force, self._jax_pos, self._jax_torque):
            ax.set_facecolor(self._BG)
            ax.tick_params(colors=self._FG, labelsize=18)
            for spine in ax.spines.values():
                spine.set_color(self._FG_DIM)
        self._jax_force.set_ylabel("efc_force", color=self._FG, fontsize=18)
        self._jax_pos.set_ylabel("efc_pos (rad)", color=self._FG, fontsize=18)
        self._jax_torque.set_ylabel("PIP torque", color=self._FG, fontsize=18)
        self._jax_torque.set_xlabel("sim time (s)", color=self._FG, fontsize=18)
        self._jfig.tight_layout()
        self._jcanvas = FigureCanvasTkAgg(self._jfig, master=parent)
        self._jcanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_right_panel(self, pane):
        right = tk.Frame(pane, bg=self._BG)
        self._conv_summary_label = tk.Label(
            right, text="", font=("Courier", 12, "bold"),
            bg="#2d2d2d", fg=self._FG_OK, anchor="w", padx=10, pady=4)
        self._conv_summary_label.pack(fill=tk.X)

        # Right side: tabbed notebook (Solver log + Joint Constraint log)
        right_nb = ttk.Notebook(right, style="Dark.TNotebook")
        right_nb.pack(fill=tk.BOTH, expand=True)

        # Solver tab (existing conv log)
        f_solver_log = ttk.Frame(right_nb, style="Dark.TFrame")
        log_frame = tk.Frame(f_solver_log, bg=self._BG)
        log_frame.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(log_frame, width=18)
        self._conv_text = tk.Text(
            log_frame, font=self._FONT_LOG, bg=self._BG, fg=self._FG,
            wrap=tk.NONE, state=tk.DISABLED, bd=0, padx=8, pady=4,
            yscrollcommand=scroll.set)
        self._conv_text.tag_configure("hit", foreground=self._FG_HIT)
        self._conv_text.tag_configure("dim", foreground=self._FG_DIM)
        scroll.config(command=self._conv_text.yview)
        self._conv_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(f_solver_log, text="Clear", font=("Courier", 10),
                  bg="#3c3c3c", fg=self._FG, bd=0, padx=12, pady=4,
                  command=self._clear_conv).pack(side=tk.BOTTOM, anchor="e", padx=8, pady=4)
        right_nb.add(f_solver_log, text=" Solver ")

        # Joint Constraint tab
        f_jnt_log = ttk.Frame(right_nb, style="Dark.TFrame")
        jnt_frame = tk.Frame(f_jnt_log, bg=self._BG)
        jnt_frame.pack(fill=tk.BOTH, expand=True)
        jnt_scroll = tk.Scrollbar(jnt_frame, width=18)
        self._joint_conv_text = tk.Text(
            jnt_frame, font=self._FONT_LOG, bg=self._BG, fg=self._FG,
            wrap=tk.NONE, state=tk.DISABLED, bd=0, padx=8, pady=4,
            yscrollcommand=jnt_scroll.set)
        self._joint_conv_text.tag_configure("value", foreground=self._FG_VALUE)
        jnt_scroll.config(command=self._joint_conv_text.yview)
        self._joint_conv_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        jnt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(f_jnt_log, text="Clear", font=("Courier", 10),
                  bg="#3c3c3c", fg=self._FG, bd=0, padx=12, pady=4,
                  command=self._clear_joint_conv).pack(side=tk.BOTTOM, anchor="e", padx=8, pady=4)
        right_nb.add(f_jnt_log, text=" Joint Constraint ")

        pane.add(right, stretch="always")

    # ── Internal: polling & rendering ──

    def _set_text(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        for line in content.split("\n"):
            tag = "header" if line.startswith("──") or line.startswith("==") else None
            widget.insert(tk.END, line + "\n", tag)
        widget.config(state=tk.DISABLED)

    def _poll(self):
        if self._data_dirty:
            for key, widget in [("solver", self._txt_solver), ("objects", self._txt_objects), ("contact", self._txt_contact)]:
                if key in self._data:
                    self._set_text(widget, self._data[key])
            self._data_dirty = False
        if self._conv_summary_dirty:
            self._conv_summary_label.config(text=self._conv_summary)
            self._conv_summary_dirty = False
        if self._conv_dirty:
            with self._lock:
                pending = self._conv_lines[:]
                self._conv_lines.clear()
                self._conv_dirty = False
            self._conv_text.config(state=tk.NORMAL)
            for ln, hit, no_contact in pending:
                tag = "hit" if hit else ("dim" if no_contact else None)
                self._conv_text.insert(tk.END, ln + "\n", tag)
            self._conv_text.see(tk.END)
            self._conv_text.config(state=tk.DISABLED)
        # Joint constraint conv log
        if self._joint_conv_dirty:
            with self._lock:
                jnt_pending = self._joint_conv_lines[:]
                self._joint_conv_lines.clear()
                self._joint_conv_dirty = False
            self._joint_conv_text.config(state=tk.NORMAL)
            for ln in jnt_pending:
                self._joint_conv_text.insert(tk.END, ln + "\n")
            self._joint_conv_text.see(tk.END)
            self._joint_conv_text.config(state=tk.DISABLED)
        # Joint time-series plot (update every poll)
        if self._joint_plot_dirty:
            self._joint_plot_dirty = False
            self._redraw_joint_plot()
        self._root.after(100, self._poll)

    def _redraw_joint_plot(self):
        """Redraw the Joint tab time-series plots."""
        if not self._joint_t:
            return
        with self._lock:
            t = list(self._joint_t)
            force = list(self._joint_efc_force)
            pos = list(self._joint_efc_pos)
            torque = list(self._joint_pip_torque)
        for ax in (self._jax_force, self._jax_pos, self._jax_torque):
            ax.clear()
            ax.set_facecolor(self._BG)
            ax.tick_params(colors=self._FG, labelsize=18)
            for spine in ax.spines.values():
                spine.set_color(self._FG_DIM)
        import math
        _leg_kw = dict(fontsize=12, facecolor="#2d2d2d", edgecolor=self._FG_DIM, labelcolor=self._FG)
        self._jax_force.plot(t, force, color="#4fc3f7", linewidth=1.5, label="Index DIP efc_force")
        self._jax_force.set_ylabel("efc_force (N·m)", color=self._FG, fontsize=18)
        self._jax_force.legend(**_leg_kw)
        self._jax_pos.plot(t, pos, color="#f48771", linewidth=1.5, label="Index DIP efc_pos")
        self._jax_pos.set_ylabel("efc_pos (rad)", color=self._FG, fontsize=18)
        self._jax_pos.legend(**_leg_kw)
        self._jax_torque.plot(t, torque, color=self._FG_OK, linewidth=1.5, label="Index PIP torque")
        self._jax_torque.set_ylabel("torque (N·m)", color=self._FG, fontsize=18)
        self._jax_torque.legend(**_leg_kw)
        self._jax_torque.set_xlabel("sim time (s)", color=self._FG, fontsize=18)

        # Round y-axis limits to significant digits
        for ax, data in [(self._jax_force, force), (self._jax_pos, pos), (self._jax_torque, torque)]:
            lo, hi = float(min(data)), float(max(data))
            span = hi - lo
            if span < 1e-12:
                span = max(abs(lo), 1e-6)
            mag = 10.0 ** math.floor(math.log10(max(span, 1e-15)))
            step = mag if span / mag >= 5 else mag / 2.0
            y_lo = float(math.floor(lo / step) * step)
            y_hi = float(math.ceil(hi / step) * step)
            if y_lo == y_hi:
                y_lo -= step
                y_hi += step
            ax.set_ylim(y_lo, y_hi)

        self._jfig.tight_layout()
        self._jcanvas.draw()

    def _clear_conv(self):
        self._conv_text.config(state=tk.NORMAL)
        self._conv_text.delete("1.0", tk.END)
        self._conv_text.config(state=tk.DISABLED)

    def _clear_joint_conv(self):
        self._joint_conv_text.config(state=tk.NORMAL)
        self._joint_conv_text.delete("1.0", tk.END)
        self._joint_conv_text.config(state=tk.DISABLED)

    def _on_close(self):
        self._closed = True
        self._root.destroy()

    # ── Internal: impedance plotting ──

    @staticmethod
    def _compute_D(r, dmin, dmax, width, mid, power):
        """Compute MuJoCo solimp impedance D(r) for penetration depth r.

        D(r) = dmin + (dmax - dmin) * clamp((r - mid*width) / ((1-mid)*width), 0, 1)^power
        """
        import numpy as _np
        denom = max((1.0 - mid) * width, 1e-12)
        s = _np.clip((r - mid * width) / denom, 0.0, 1.0)
        return dmin + (dmax - dmin) * (s ** power)

    def _setup_ax(self):
        """Reset axes with dark theme styling."""
        ax = self._ax
        ax.clear()
        ax.set_facecolor(self._BG)
        ax.tick_params(colors=self._FG, labelsize=12)
        for spine in ax.spines.values():
            spine.set_color(self._FG_DIM)
        return ax

    def _plot_impedance(self, label):
        """Plot single-geom impedance curve D(r)."""
        import numpy as _np
        if label not in self._solref_solimp_cache:
            return
        solref, solimp, solmix = self._solref_solimp_cache[label]
        tc, dr = solref
        dmin, dmax, width, mid, power = solimp

        ax = self._setup_ax()
        x_max = max(width * 1.5, 1e-6)
        r = _np.linspace(0, x_max, 500)
        D = self._compute_D(r, dmin, dmax, width, mid, power)

        ax.plot(r * 1000, D, color="#4fc3f7", linewidth=2.5)
        ax.axhline(y=dmin, color=self._FG_DIM, linestyle="--", linewidth=2, label=f"dmin={dmin:.3f}")
        ax.axhline(y=dmax, color=self._FG_DIM, linestyle="--", linewidth=2, label=f"dmax={dmax:.3f}")
        ax.axvline(x=mid * width * 1000, color=self._FG_OK, linestyle=":", linewidth=2, label=f"mid={mid*width*1000:.3f}mm")
        ax.axvline(x=width * 1000, color=self._FG_HIT, linestyle=":", linewidth=2, label=f"width={width*1000:.3f}mm")
        ax.set_xlim(0, x_max * 1000)
        ax.set_xlabel("penetration (mm)", color=self._FG, fontsize=12)
        ax.set_ylabel("D(r)", color=self._FG, fontsize=12)
        ax.set_title(f"{label}  tc={tc:.4f} dr={dr:.4f} solmix={solmix:.2f}", color=self._FG_HEADER, fontsize=10)
        ax.legend(fontsize=10, facecolor="#2d2d2d", edgecolor=self._FG_DIM, labelcolor=self._FG)
        y_margin = max((dmax - dmin) * 0.1, 0.001)
        ax.set_ylim(dmin - y_margin, dmax + y_margin)
        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_blended_impedance(self, label_a, label_b):
        """Plot blended impedance for a contact pair, with individual curves overlaid."""
        import numpy as _np
        if label_a not in self._solref_solimp_cache or label_b not in self._solref_solimp_cache:
            return
        solref_a, solimp_a, solmix_a = self._solref_solimp_cache[label_a]
        solref_b, solimp_b, solmix_b = self._solref_solimp_cache[label_b]

        # Compute blending ratio (MuJoCo solmix convention)
        mix = solmix_a / max(solmix_a + solmix_b, 1e-12)
        if solmix_a < 1e-6 and solmix_b < 1e-6:
            mix = 0.5
        elif solmix_a < 1e-6:
            mix = 0.0
        elif solmix_b < 1e-6:
            mix = 1.0

        solimp_blend = [mix * a + (1 - mix) * b for a, b in zip(solimp_a, solimp_b)]
        solref_blend = [mix * a + (1 - mix) * b for a, b in zip(solref_a, solref_b)]
        dmin, dmax, width, mid, power = solimp_blend
        tc, dr = solref_blend

        ax = self._setup_ax()
        x_max = max(width * 1.5, 1e-6)
        r = _np.linspace(0, x_max, 500)

        # Individual curves (dashed)
        ax.plot(r * 1000, self._compute_D(r, *solimp_a), color="#f48771", linewidth=2.5, linestyle="--", label=label_a)
        ax.plot(r * 1000, self._compute_D(r, *solimp_b), color=self._FG_OK, linewidth=2.5, linestyle="--", label=label_b)
        # Blended curve (solid, thick)
        ax.plot(r * 1000, self._compute_D(r, dmin, dmax, width, mid, power), color="#4fc3f7", linewidth=3.5, label="blended")
        # Blended midpoint/width markers
        ax.axvline(x=mid * width * 1000, color=self._FG_OK, linestyle=":", linewidth=2, label=f"mid={mid*width*1000:.3f}mm")
        ax.axvline(x=width * 1000, color=self._FG_HIT, linestyle=":", linewidth=2, label=f"width={width*1000:.3f}mm")

        ax.set_xlim(0, x_max * 1000)
        ax.set_xlabel("penetration (mm)", color=self._FG, fontsize=12)
        ax.set_ylabel("D(r)", color=self._FG, fontsize=12)
        ax.set_title(f"{label_a}<->{label_b}  mix={mix:.2f}  tc={tc:.4f} dr={dr:.4f}", color=self._FG_HEADER, fontsize=9)
        ax.legend(fontsize=9, facecolor="#2d2d2d", edgecolor=self._FG_DIM, labelcolor=self._FG)
        y_lo = min(solimp_a[0], solimp_b[0], dmin)
        y_hi = max(solimp_a[1], solimp_b[1], dmax)
        y_margin = max((y_hi - y_lo) * 0.1, 0.001)
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
        self._fig.tight_layout()
        self._canvas.draw()
