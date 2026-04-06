"""Contact force test — measure normal/friction forces, penetration depth, solver convergence.

Scenario: 1 box (550g) + 2 spheres (5g each) on ground plane.
Spheres push into box with constant y-force.
Verifies: static normal force ≈ mg, geometric mean friction override, solimp/solref tuning.

Run:
    python jk_solver_examples/contacts/test_contact_force.py
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import warp as wp

_JK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_JK_ROOT))

import newton
import newton.examples
from jk_solver_examples import init as jk_init
from jk_solver_examples.jk_kernels import update_max_penetration_kernel
from jk_solver_examples.jk_solver import SolverJK
from newton import Contacts

# ──── Camera ────
CAM_POS = (8.5, 0.0, 0.1)
CAM_PITCH = 0.0
CAM_YAW = -180.0

# ──── Timing ────
FPS = 60
FRAME_DT = 1.0 / FPS
SIM_SUBSTEPS = 2
SIM_DT = FRAME_DT / SIM_SUBSTEPS

# ──── Physics ────
GRAVITY = 9.81
RIGID_GAP = 0.005  # broadphase AABB expansion [m] (upstream default 0.1)
BOX_HALF = 0.5
SPHERE_R = 0.3
BOX_DENSITY = 0.55       # kg/m^3 → 1m^3 box = 0.55kg (550g)
SPHERE_DENSITY = 0.04421  # kg/m^3 → r=0.3 sphere ≈ 0.005kg (5g)
BOX_MU = 0.5
SPHERE_MU = 0.5
GROUND_MU = 0.5
SPHERE_GAP = BOX_HALF + SPHERE_R + 2.0  # box side to sphere center distance
SPHERE_FORCE_Y = 0.05  # y-force applied to spheres [N]

# ──── Solver (frozen at creation — not runtime-tunable) ────
SOLVER_ITERATIONS = 5
SOLVER_LS_ITERATIONS = 5
SOLVER_CONE = "elliptic"
SOLVER_IMPRATIO = 1.0

# ──── Contact: solref/solimp defaults (per-geom, directly tunable) ────
_DEFAULT_SOLREF = (0.02, 1.0)                      # (timeconst, dampratio)
_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)    # (dmin, dmax, width, midpoint, power)


# =============================================================================
# StatusWindow — tkinter debug UI (runs in daemon thread)
# =============================================================================

class StatusWindow:
    """Dark-themed tkinter debug monitor.

    Layout: left=tabbed notebook (Solver/Objects/Contact), right=solver conv log.
    Contact tab includes matplotlib impedance plot with per-geom and blended curves.
    All public methods are thread-safe (called from main sim thread).
    """

    _FONT = ("nimbus mono l", 30, "bold")
    _FONT_TITLE = ("nimbus mono l", 30, "bold")
    _FONT_LOG = ("nimbus mono l", 30, "bold")
    _BG = "#1e1e1e"
    _FG = "#d4d4d4"
    _FG_DIM = "#6a6a6a"
    _FG_HEADER = "#569cd6"
    _FG_VALUE = "#ce9178"
    _FG_HIT = "#f44747"
    _FG_OK = "#6a9955"

    def __init__(self, title="Contact Debug Monitor", width=2400, height=1200):
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
        self._thread = threading.Thread(target=self._run, args=(title, width, height), daemon=True)
        self._thread.start()

    # ── Public API (called from main thread) ──

    def update(self, data: dict):
        """Update tab contents. Keys: 'objects', 'contact', 'solver' → multiline strings."""
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
            b = tk.Button(self._pair_btn_frame, text=f"{la}<->{lb}", font=("Courier", 16),
                          bg="#2d2d2d", fg=self._FG_VALUE, bd=0, padx=6, pady=2,
                          command=lambda a=la, b=lb: self._plot_blended_impedance(a, b))
            b.pack(side=tk.LEFT, padx=3)
            self._pair_buttons[key] = b

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
                        font=("Courier", 22, "bold"), padding=[16, 6])
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

        pane.add(left, stretch="always", minsize=width)

    def _setup_contact_plot(self, parent):
        """Create geom buttons + matplotlib canvas for impedance plots."""
        plot_frame = tk.Frame(parent, bg=self._BG)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(plot_frame, bg=self._BG)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        for label in ["ground", "box", "sph_L", "sph_R"]:
            tk.Button(btn_frame, text=label, font=("Courier", 18),
                      bg="#3c3c3c", fg=self._FG, bd=0, padx=10, pady=2,
                      command=lambda l=label: self._plot_impedance(l)).pack(side=tk.LEFT, padx=4)
        tk.Label(btn_frame, text=" | ", font=("Courier", 18), bg=self._BG, fg=self._FG_DIM).pack(side=tk.LEFT)
        self._pair_btn_frame = tk.Frame(btn_frame, bg=self._BG)
        self._pair_btn_frame.pack(side=tk.LEFT)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(8, 3), dpi=80, facecolor=self._BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(self._BG)
        self._ax.tick_params(colors=self._FG, labelsize=24)
        for spine in self._ax.spines.values():
            spine.set_color(self._FG_DIM)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_right_panel(self, pane):
        right = tk.Frame(pane, bg=self._BG)
        self._conv_summary_label = tk.Label(
            right, text="", font=("Courier", 24, "bold"),
            bg="#2d2d2d", fg=self._FG_OK, anchor="w", padx=10, pady=4)
        self._conv_summary_label.pack(fill=tk.X)

        log_frame = tk.Frame(right, bg=self._BG)
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

        tk.Button(right, text="Clear", font=("Courier", 20),
                  bg="#3c3c3c", fg=self._FG, bd=0, padx=12, pady=4,
                  command=self._clear_conv).pack(side=tk.BOTTOM, anchor="e", padx=8, pady=4)
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
        self._root.after(100, self._poll)

    def _clear_conv(self):
        self._conv_text.config(state=tk.NORMAL)
        self._conv_text.delete("1.0", tk.END)
        self._conv_text.config(state=tk.DISABLED)

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
        ax.tick_params(colors=self._FG, labelsize=24)
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
        ax.set_xlabel("penetration (mm)", color=self._FG, fontsize=24)
        ax.set_ylabel("D(r)", color=self._FG, fontsize=24)
        ax.set_title(f"{label}  tc={tc:.4f} dr={dr:.4f} solmix={solmix:.2f}", color=self._FG_HEADER, fontsize=20)
        ax.legend(fontsize=20, facecolor="#2d2d2d", edgecolor=self._FG_DIM, labelcolor=self._FG)
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
        ax.set_xlabel("penetration (mm)", color=self._FG, fontsize=24)
        ax.set_ylabel("D(r)", color=self._FG, fontsize=24)
        ax.set_title(f"{label_a}<->{label_b}  mix={mix:.2f}  tc={tc:.4f} dr={dr:.4f}", color=self._FG_HEADER, fontsize=18)
        ax.legend(fontsize=18, facecolor="#2d2d2d", edgecolor=self._FG_DIM, labelcolor=self._FG)
        y_lo = min(solimp_a[0], solimp_b[0], dmin)
        y_hi = max(solimp_a[1], solimp_b[1], dmax)
        y_margin = max((y_hi - y_lo) * 0.1, 0.001)
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
        self._fig.tight_layout()
        self._canvas.draw()


# =============================================================================
# Scene builder & solver parameter application
# =============================================================================

def _build_scene(p):
    """Build Newton model + SolverJK + state from parameter dict."""
    builder = newton.ModelBuilder(gravity=-p["gravity"])
    builder.rigid_gap = p["rigid_gap"]
    # ke/kd=0 disables auto solref conversion; we override geom_solref directly.
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=p["ground_mu"], ke=0, kd=0, density=0.0)
    builder.add_ground_plane(cfg=ground_cfg)
    box_cfg = newton.ModelBuilder.ShapeConfig(mu=p["box_mu"], ke=0, kd=0, density=p["box_density"])
    sph_cfg = newton.ModelBuilder.ShapeConfig(mu=p["sphere_mu"], ke=0, kd=0, density=p["sphere_density"])

    body_box = builder.add_link(
        xform=wp.transform(p=(0.0, 0.0, p["box_half"]), q=wp.quat_identity()), label="box_yz")
    builder.add_shape_box(body_box, hx=p["box_half"], hy=p["box_half"], hz=p["box_half"], cfg=box_cfg)
    j_box = builder.add_joint_d6(
        parent=-1, child=body_box,
        linear_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
                     newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, p["box_half"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_articulation([j_box])

    body_sphere_l = builder.add_link(
        xform=wp.transform(p=(0.0, -p["sphere_gap"], p["sphere_r"]), q=wp.quat_identity()), label="sphere_left")
    builder.add_shape_sphere(body_sphere_l, radius=p["sphere_r"], cfg=sph_cfg)
    j_sl = builder.add_joint_d6(
        parent=-1, child=body_sphere_l,
        linear_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
                     newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, -p["sphere_gap"], p["sphere_r"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_articulation([j_sl])

    body_sphere_r = builder.add_link(
        xform=wp.transform(p=(0.0, p["sphere_gap"], p["sphere_r"]), q=wp.quat_identity()), label="sphere_right")
    builder.add_shape_sphere(body_sphere_r, radius=p["sphere_r"], cfg=sph_cfg)
    j_sr = builder.add_joint_d6(
        parent=-1, child=body_sphere_r,
        linear_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
                     newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, p["sphere_gap"], p["sphere_r"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_articulation([j_sr])

    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    model = builder.finalize()
    solver = SolverJK(
        model,
        iterations=p["solver_iterations"], ls_iterations=p["solver_ls_iterations"],
        cone=p["solver_cone"], impratio=p["solver_impratio"],
        njmax=200, nconmax=200, use_mujoco_contacts=False)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision = newton.CollisionPipeline(model)
    contacts = Contacts(solver.get_max_contact_count(), 0,
                        requested_attributes=model.get_requested_contact_attributes())
    _apply_solref_solimp(solver, model, p, body_box, body_sphere_l, body_sphere_r)
    return model, solver, state_0, state_1, control, contacts, collision, body_box, body_sphere_l, body_sphere_r


def _apply_solref_solimp(solver, model, p, body_box, body_sphere_l, body_sphere_r):
    """Override per-geom solref/solimp/solmix on mjw_model after solver creation."""
    geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()
    shape_body = model.shape_body.numpy()
    solref_np = solver.mjw_model.geom_solref.numpy()
    solimp_np = solver.mjw_model.geom_solimp.numpy()
    solmix_np = solver.mjw_model.geom_solmix.numpy()

    body_param_map = {
        -1: ("ground_solref", "ground_solimp", "ground_solmix"),
        body_box: ("box_solref", "box_solimp", "box_solmix"),
        body_sphere_l: ("sphere_solref", "sphere_solimp", "sphere_solmix"),
        body_sphere_r: ("sphere_solref", "sphere_solimp", "sphere_solmix"),
    }
    for g in range(solref_np.shape[1]):
        s = geom_to_shape[0, g]
        b = shape_body[s] if s >= 0 else -1
        if b not in body_param_map:
            continue
        sr_key, si_key, sm_key = body_param_map[b]
        solref_np[0, g] = p[sr_key][:2]
        solimp_np[0, g] = p[si_key][:5]
        solmix_np[0, g] = p[sm_key]

    solver.mjw_model.geom_solref.assign(solref_np)
    solver.mjw_model.geom_solimp.assign(solimp_np)
    solver.mjw_model.geom_solmix.assign(solmix_np)


def _make_params():
    """Create parameter dict from current module-level constants."""
    return dict(
        fps=FPS, sim_substeps=SIM_SUBSTEPS, sphere_force_y=SPHERE_FORCE_Y,
        solver_iterations=SOLVER_ITERATIONS, solver_ls_iterations=SOLVER_LS_ITERATIONS,
        solver_cone=SOLVER_CONE, solver_impratio=SOLVER_IMPRATIO,
        gravity=GRAVITY, rigid_gap=RIGID_GAP, box_half=BOX_HALF, sphere_r=SPHERE_R,
        box_mass=BOX_DENSITY * (2.0 * BOX_HALF) ** 3,
        sphere_mass=SPHERE_DENSITY * (4.0 / 3.0) * np.pi * SPHERE_R ** 3,
        box_density=BOX_DENSITY, sphere_density=SPHERE_DENSITY,
        box_mu=BOX_MU, sphere_mu=SPHERE_MU, ground_mu=GROUND_MU,
        ground_solref=list(_DEFAULT_SOLREF), ground_solimp=list(_DEFAULT_SOLIMP), ground_solmix=1.0,
        box_solref=list(_DEFAULT_SOLREF), box_solimp=list(_DEFAULT_SOLIMP), box_solmix=1.0,
        sphere_solref=list(_DEFAULT_SOLREF), sphere_solimp=list(_DEFAULT_SOLIMP), sphere_solmix=1.0,
        sphere_gap=SPHERE_GAP,
    )


# =============================================================================
# Example — main simulation class
# =============================================================================

class Example:
    """Contact force test with real-time debug UI and parameter tuning.

    Creates a box + 2 spheres scene. Measures normal/friction forces,
    penetration depth, and solver convergence per substep.
    StatusWindow provides tabbed monitoring; imgui provides parameter tuning.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self._active = _make_params()
        self._p = _make_params()

        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts, self.collision,
         self.body_box, self.body_sphere_l, self.body_sphere_r) = _build_scene(self._active)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)

        self.print_interval = 0.5
        self.next_print_time = 0.0
        # Cached GPU→CPU mappings (immutable during simulation)
        self._geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()
        self._shape_body = self.model.shape_body.numpy()
        self._init_max_pen_gpu()
        self._prev_vel = {}
        self._max_vel = {}
        self._max_acc = {}
        self._status_window = StatusWindow()

    def reset(self):
        """Apply pending GUI params and rebuild entire scene."""
        self._active = dict(self._p)
        _apply_module_constants(self._active)

        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts, self.collision,
         self.body_box, self.body_sphere_l, self.body_sphere_r) = _build_scene(self._active)

        self.viewer.reset_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)
        self.sim_time = 0.0
        self.next_print_time = 0.0
        self._geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()
        self._shape_body = self.model.shape_body.numpy()
        self._init_max_pen_gpu()
        self._prev_vel = {}
        self._max_vel = {}
        self._max_acc = {}
        print("[Reset] all params applied")

    def gui(self, imgui):
        """imgui Param Tuner — edits pending dict, applied on Reset."""
        if not getattr(self.viewer, "_irim_param_tune_open", False):
            return
        p = self._p

        imgui.text("=== Solver ===")
        _, v = imgui.input_int("iter", p["solver_iterations"])
        p["solver_iterations"] = max(1, v)
        _, v = imgui.input_int("ls_iter", p["solver_ls_iterations"])
        p["solver_ls_iterations"] = max(1, v)
        _, p["solver_impratio"] = imgui.input_float("impratio", p["solver_impratio"], format="%.2f")
        cone_options = ["elliptic", "pyramidal"]
        cone_idx = cone_options.index(p["solver_cone"])
        ch, new_cone_idx = imgui.combo("cone", cone_idx, cone_options)
        if ch:
            p["solver_cone"] = cone_options[new_cone_idx]

        imgui.separator()
        imgui.text("=== Sim ===")
        _, v = imgui.input_int("FPS", p["fps"])
        p["fps"] = max(1, v)
        _, v = imgui.input_int("SUBSTEPS", p["sim_substeps"])
        p["sim_substeps"] = max(1, v)
        _, p["sphere_force_y"] = imgui.input_float("FORCE", p["sphere_force_y"], format="%.5f")
        _, p["sphere_gap"] = imgui.input_float("SPHERE_GAP", p["sphere_gap"], format="%.3f")

        imgui.separator()
        imgui.text("=== Geometry ===")
        _, p["gravity"] = imgui.input_float("GRAVITY", p["gravity"], format="%.3f")
        _, p["rigid_gap"] = imgui.input_float("RIGID_GAP", p["rigid_gap"], format="%.4f")
        _, p["box_half"] = imgui.input_float("BOX_HALF", p["box_half"], format="%.3f")
        _, p["sphere_r"] = imgui.input_float("SPHERE_R", p["sphere_r"], format="%.3f")
        _, p["box_mass"] = imgui.input_float("BOX_MASS", p["box_mass"], format="%.4f")
        box_vol = (2.0 * p["box_half"]) ** 3
        p["box_density"] = p["box_mass"] / box_vol if box_vol > 0 else 0.0
        _, p["box_mu"] = imgui.input_float("BOX_MU", p["box_mu"], format="%.3f")
        _, p["sphere_mass"] = imgui.input_float("SPH_MASS", p["sphere_mass"], format="%.4f")
        sph_vol = (4.0 / 3.0) * np.pi * p["sphere_r"] ** 3
        p["sphere_density"] = p["sphere_mass"] / sph_vol if sph_vol > 0 else 0.0
        _, p["sphere_mu"] = imgui.input_float("SPHERE_MU", p["sphere_mu"], format="%.3f")
        _, p["ground_mu"] = imgui.input_float("GROUND_MU", p["ground_mu"], format="%.3f")

        imgui.text("=== Contact (solref/solimp) ===")
        for label, key_sr, key_si, key_sm in [
            ("GND", "ground_solref", "ground_solimp", "ground_solmix"),
            ("BOX", "box_solref", "box_solimp", "box_solmix"),
            ("SPH", "sphere_solref", "sphere_solimp", "sphere_solmix"),
        ]:
            imgui.text(label)
            sr, si = p[key_sr], p[key_si]
            _, sr[0] = imgui.input_float(f"{label}_tc", sr[0], format="%.4f")
            _, sr[1] = imgui.input_float(f"{label}_dr", sr[1], format="%.4f")
            _, si[0] = imgui.input_float(f"{label}_dmin", si[0], format="%.3f")
            _, si[1] = imgui.input_float(f"{label}_dmax", si[1], format="%.3f")
            _, si[2] = imgui.input_float(f"{label}_w", si[2], format="%.4f")
            _, si[3] = imgui.input_float(f"{label}_mid", si[3], format="%.2f")
            _, si[4] = imgui.input_float(f"{label}_pow", si[4], format="%.1f")
            _, p[key_sm] = imgui.input_float(f"{label}_solmix", p[key_sm], format="%.2f")
            imgui.separator()

        imgui.separator()
        imgui.text("Reset to apply all changes")

    # ── Simulation loop ──

    def simulate(self):
        """Run one frame: substeps × (forces → collision → solver step → post)."""
        frame_dt = 1.0 / self._active["fps"]
        sim_dt = frame_dt / self._active["sim_substeps"]
        for sub in range(self._active["sim_substeps"]):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._apply_sphere_forces()
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, sim_dt)
            self._post_substep(sub)
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        """Called each frame: simulate + periodic status update."""
        self.simulate()
        self.sim_time += 1.0 / self._active["fps"]
        if self.sim_time >= self.next_print_time:
            self._update_status()
            self.next_print_time += self.print_interval

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ── Internal helpers ──

    def _apply_sphere_forces(self):
        body_f = self.state_0.body_f.numpy()
        body_f[self.body_sphere_l][1] = +self._active["sphere_force_y"]
        body_f[self.body_sphere_r][1] = -self._active["sphere_force_y"]
        self.state_0.body_f.assign(body_f)

    def _init_max_pen_gpu(self):
        """Allocate GPU array for atomic_max penetration tracking."""
        self._n_bodies = int(self.model.body_count)
        stride = self._n_bodies + 1
        self._max_pen_gpu = wp.zeros(stride * stride, dtype=wp.float32, device=self.model.device)

    def _post_substep(self, substep):
        """Per-substep: log solver conv (3 scalar .numpy()) + GPU max penetration kernel."""
        d = self.solver.mjw_data
        niter = int(d.solver_niter.numpy()[0])
        nacon = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])

        max_iter = self._active["solver_iterations"]
        hit = niter >= max_iter
        self._status_window.push_conv(
            f"t={self.sim_time:.3f} sub={substep}  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}",
            hit_limit=hit, no_contact=(nacon == 0))
        self._status_window.update_conv_summary(
            f"[current]  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}")

        if nacon > 0:
            wp.launch(
                update_max_penetration_kernel, dim=d.naconmax,
                inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                        self.solver.mjc_geom_to_newton_shape, self.model.shape_body, self._n_bodies],
                outputs=[self._max_pen_gpu], device=self.model.device)

    def _get_contact_forces(self):
        """Read efc_force from mjw_data. Returns per-body and per-pair normal/friction/penetration."""
        d = self.solver.mjw_data
        nacon = int(d.nacon.numpy()[0])
        if nacon == 0:
            return {}, {}, {}, {}, {}

        contact_geom = d.contact.geom.numpy()[:nacon]
        contact_efc = d.contact.efc_address.numpy()[:nacon]
        contact_dim = d.contact.dim.numpy()[:nacon]
        contact_worldid = d.contact.worldid.numpy()[:nacon]
        contact_dist = d.contact.dist.numpy()[:nacon]
        efc_force = d.efc.force.numpy()
        is_pyramidal = self.solver.mjw_model.opt.cone == int(self.solver._mujoco.mjtCone.mjCONE_PYRAMIDAL)

        body_nf, body_ff, pair_nf, pair_ff, pair_pen = {}, {}, {}, {}, {}

        for c in range(nacon):
            efc_addr = contact_efc[c, 0]
            if efc_addr < 0:  # invalid constraint — skip
                continue
            world = contact_worldid[c]
            dim = int(contact_dim[c])
            nf = abs(float(efc_force[world, efc_addr]))
            ff = sum(abs(float(efc_force[world, contact_efc[c, i]]))
                     for i in range(1, 2 * (dim - 1) if is_pyramidal else dim))
            pen = -float(contact_dist[c])

            g0, g1 = contact_geom[c]
            b0 = self._shape_body[self._geom_to_shape[world, g0]] if self._geom_to_shape[world, g0] >= 0 else -1
            b1 = self._shape_body[self._geom_to_shape[world, g1]] if self._geom_to_shape[world, g1] >= 0 else -1

            for b in (b0, b1):
                body_nf[b] = body_nf.get(b, 0.0) + nf
                body_ff[b] = body_ff.get(b, 0.0) + ff

            key = (min(b0, b1), max(b0, b1))
            pair_nf[key] = pair_nf.get(key, 0.0) + nf
            pair_ff[key] = pair_ff.get(key, 0.0) + ff
            pair_pen[key] = max(pair_pen.get(key, 0.0), pen)

        return body_nf, body_ff, pair_nf, pair_ff, pair_pen

    # ── Status update (split into 3 tab builders) ──

    def _update_status(self):
        """Build and send tab data to StatusWindow (called every print_interval)."""
        body_nf, body_ff, pair_nf, pair_ff, pair_pen = self._get_contact_forces()
        p = self._active
        body_labels = {-1: "ground", self.body_box: "box", self.body_sphere_l: "sph_L", self.body_sphere_r: "sph_R"}

        obj_text = self._build_objects_tab(body_nf, body_ff, p)
        con_text = self._build_contact_tab(pair_nf, pair_ff, pair_pen, body_labels)
        sol_text = self._build_solver_tab(p)

        self._status_window.update({"objects": obj_text, "contact": con_text, "solver": sol_text})

    def _build_objects_tab(self, body_nf, body_ff, p):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_f = self.state_0.body_f.numpy()
        body_mass = self.model.body_mass.numpy()
        frame_dt = 1.0 / p["fps"]

        lines = [f"t = {self.sim_time:.2f}s", ""]
        for label, idx in [("box", self.body_box), ("sphere_L", self.body_sphere_l), ("sphere_R", self.body_sphere_r)]:
            m = body_mass[idx]
            pos = body_q[idx][:3]
            vel = body_qd[idx][:3]
            ext_f = body_f[idx][:3]
            nf = body_nf.get(idx, 0.0)
            ff = body_ff.get(idx, 0.0)

            vel_mag = float(np.linalg.norm(vel))
            prev_vel = self._prev_vel.get(idx)
            if prev_vel is not None and frame_dt > 0:
                acc = (vel - prev_vel) / frame_dt
                acc_mag = float(np.linalg.norm(acc))
            else:
                acc = np.zeros(3)
                acc_mag = 0.0
            self._prev_vel[idx] = vel.copy()
            self._max_vel[idx] = max(self._max_vel.get(idx, 0.0), vel_mag)
            self._max_acc[idx] = max(self._max_acc.get(idx, 0.0), acc_mag)

            lines.append(f"── {label} ({m:.3f}kg) ──")
            lines.append(f"  pos  ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
            lines.append(f"  vel  ({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f})  |v|={vel_mag:.4f} (max {self._max_vel[idx]:.4f})")
            lines.append(f"  acc  ({acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f})  |a|={acc_mag:.4f} (max {self._max_acc[idx]:.4f})")
            lines.append(f"  ext  ({ext_f[0]:+.3f}, {ext_f[1]:+.3f}, {ext_f[2]:+.3f})")
            lines.append(f"  N={nf:.5f}  F={ff:.5f}")
            lines.append("")
        return "\n".join(lines)

    def _build_contact_tab(self, pair_nf, pair_ff, pair_pen, body_labels):
        solref_np = self.solver.mjw_model.geom_solref.numpy()
        solimp_np = self.solver.mjw_model.geom_solimp.numpy()
        solmix_np = self.solver.mjw_model.geom_solmix.numpy()

        lines = ["── Solref / Solimp (per geom) ──"]
        lines.append(f"  {'geom':<8s} {'tc':>8s} {'dr':>8s} {'dmin':>6s} {'dmax':>6s} {'width':>8s} {'mid':>6s} {'pow':>5s} {'mix':>5s}")
        plot_cache = {}
        for g in range(solref_np.shape[1]):
            s = self._geom_to_shape[0, g]
            b = self._shape_body[s] if s >= 0 else -1
            label = body_labels.get(b)
            if label is None:
                continue
            sr, si = solref_np[0, g], solimp_np[0, g]
            sm = float(solmix_np[0, g])
            lines.append(f"  {label:<8s} {sr[0]:>8.5f} {sr[1]:>8.5f} {si[0]:>6.3f} {si[1]:>6.3f} {si[2]:>8.4f} {si[3]:>6.2f} {si[4]:>5.1f} {sm:>5.2f}")
            if label not in plot_cache:
                plot_cache[label] = (list(sr), list(si), sm)
        self._status_window.update_solref_solimp(plot_cache)
        lines.append("")

        active_pairs = []
        if pair_nf:
            lines.append("── Pair Forces ──")
            lines.append(f"  {'Pair':<20s} {'Normal(N)':>10s} {'Fric(N)':>10s} {'Pen(mm)':>10s} {'Max_Pen(mm)':>10s}")
            max_pen_cpu = self._max_pen_gpu.numpy()
            stride = self._n_bodies + 1
            for (a, b) in sorted(pair_nf.keys()):
                nf = pair_nf.get((a, b), 0.0)
                ff = pair_ff.get((a, b), 0.0)
                pen = pair_pen.get((a, b), 0.0)
                max_p = float(max_pen_cpu[(a + 1) * stride + (b + 1)])
                la = body_labels.get(a, f"b{a}")
                lb = body_labels.get(b, f"b{b}")
                lines.append(f"  {la}<->{lb:<14s} {nf:>10.5f} {ff:>10.5f} {pen*1000:>10.3f} {max_p*1000:>10.3f}")
                active_pairs.append((la, lb))
        self._status_window.update_active_pairs(active_pairs)
        return "\n".join(lines)

    def _build_solver_tab(self, p):
        frame_dt = 1.0 / p["fps"]
        sim_dt = frame_dt / p["sim_substeps"]
        lines = [
            "── Solver Config ──",
            f"  iterations      {p['solver_iterations']}",
            f"  ls_iterations   {p['solver_ls_iterations']}",
            f"  cone            {p['solver_cone']}",
            f"  impratio        {p['solver_impratio']:.2f}",
            "",
            "── Sim Config ──",
            f"  FPS             {p['fps']}",
            f"  substeps        {p['sim_substeps']}",
            f"  frame_dt        {frame_dt:.6f}s",
            f"  sim_dt          {sim_dt:.6f}s",
            f"  gravity         {p['gravity']:.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Module constant sync (for Reset)
# =============================================================================

def _apply_module_constants(p):
    """Sync parameter dict back to module-level constants."""
    import jk_solver_examples.contacts.test_contact_force as m
    m.GRAVITY = p["gravity"]
    m.RIGID_GAP = p["rigid_gap"]
    m.BOX_HALF = p["box_half"]
    m.SPHERE_R = p["sphere_r"]
    m.BOX_DENSITY = p["box_density"]
    m.SPHERE_DENSITY = p["sphere_density"]
    m.BOX_MU = p["box_mu"]
    m.SPHERE_MU = p["sphere_mu"]
    m.GROUND_MU = p["ground_mu"]
    m._DEFAULT_SOLREF = tuple(p["ground_solref"])
    m._DEFAULT_SOLIMP = tuple(p["ground_solimp"])
    m.SPHERE_GAP = p["sphere_gap"]
    m.SPHERE_FORCE_Y = p["sphere_force_y"]
    m.FPS = p["fps"]
    m.FRAME_DT = 1.0 / p["fps"]
    m.SIM_SUBSTEPS = p["sim_substeps"]
    m.SIM_DT = m.FRAME_DT / p["sim_substeps"]
    m.SOLVER_ITERATIONS = p["solver_iterations"]
    m.SOLVER_LS_ITERATIONS = p["solver_ls_iterations"]
    m.SOLVER_CONE = p["solver_cone"]
    m.SOLVER_IMPRATIO = p["solver_impratio"]


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = jk_init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
