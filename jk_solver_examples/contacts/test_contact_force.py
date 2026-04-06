"""접촉력 테스트 - 법선력/접선력 측정, 접촉점 분포 확인

정적 접촉 (box가 지면에 놓인 상태) vs 동적 접촉 (box 낙하 후 충격)을 비교.
정적 접촉 시 법선력 ≈ mg 인지 검증한다.

실행: python jk_solver_examples/contacts/test_contact_force.py
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

CAM_POS = (8.5, 0.0, 0.1)
CAM_PITCH = 0.0
CAM_YAW = -180.0

FPS = 60
FRAME_DT = 1.0 / FPS
SIM_SUBSTEPS = 2
SIM_DT = FRAME_DT / SIM_SUBSTEPS

GRAVITY = 9.81
BOX_HALF = 0.5
SPHERE_R = 0.3
BOX_DENSITY = 0.55  # kg/m^3  (1m^3 box → 0.55kg = 550g)
SPHERE_DENSITY = 0.04421  # kg/m^3  (r=0.3 sphere → ~0.005kg = 5g)
BOX_MU = 0.5  # 박스 마찰계수
SPHERE_MU = 0.5  # 구 마찰계수
GROUND_MU = 0.5  # 바닥 마찰계수
SPHERE_GAP = BOX_HALF + SPHERE_R + 2.0  # 박스 옆면~구 중심 간격
SPHERE_FORCE_Y = 0.05  # 구에 가할 y방향 힘 [N]

# solver 기본값
SOLVER_ITERATIONS = 5
SOLVER_LS_ITERATIONS = 5
SOLVER_CONE = "elliptic"
SOLVER_IMPRATIO = 1.0
GROUND_KE = 0.0  # 바닥 접촉 강성
GROUND_KD = 0.0 # 바닥 접촉 감쇠
BOX_KE = 0.0  # 박스 접촉 강성
BOX_KD = 0.0 # 박스 접촉 감쇠
SPHERE_KE = 0.0  # 구 접촉 강성
SPHERE_KD = 0.0 # 구 접촉 감쇠

# solref/solimp 직접 제어 (None이면 ke/kd에서 자동 변환)
SOLREF_TIMECONST = 0.02
SOLREF_DAMPRATIO = 1.0
SOLIMP_DMIN = 0.9
SOLIMP_DMAX = 0.95
SOLIMP_WIDTH = 0.001
SOLIMP_MIDPOINT = 0.5
SOLIMP_POWER = 2.0


class StatusWindow:
    """별도 스레드 tkinter 디버깅 UI — 좌: 탭(Objects/Contact/Solver), 우: Solver Conv Log."""

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
        self._data = {}  # {"objects": str, "contact": str, "solver": str}
        self._data_dirty = False
        self._conv_lines: list[tuple[str, bool, bool]] = []  # (line, hit_limit, no_contact)
        self._conv_dirty = False
        self._conv_summary = ""
        self._conv_summary_dirty = False
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(title, width, height), daemon=True)
        self._thread.start()

    def _make_text_widget(self, parent):
        t = tk.Text(parent, font=self._FONT, bg=self._BG, fg=self._FG,
                    wrap=tk.NONE, state=tk.DISABLED, bd=0, padx=10, pady=8)
        t.tag_configure("header", foreground=self._FG_HEADER, font=self._FONT_TITLE)
        t.tag_configure("value", foreground=self._FG_VALUE)
        t.tag_configure("dim", foreground=self._FG_DIM)
        return t

    @staticmethod
    def _get_primary_monitor():
        """xrandr로 primary 모니터의 해상도와 offset을 반환."""
        import subprocess
        result = subprocess.run(["xrandr", "--query"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "primary" in line and " connected" in line:
                # e.g. "DP-0 connected primary 4096x2304+3840+562"
                for part in line.split():
                    if "x" in part and "+" in part:
                        res, ox, oy = part.split("+")[0], part.split("+")[1], part.split("+")[2]
                        mw, mh = res.split("x")
                        return int(mw), int(mh), int(ox), int(oy)
        return None, None, None, None

    def _run(self, title, width, height):
        self._root = tk.Tk()
        self._root.title(title)
        # primary 모니터 해상도 기반 동적 크기 (80%)
        mw, mh, mx, my = self._get_primary_monitor()
        if mw and mh:
            w = max(width, int(mw * 0.8))
            h = max(height, int(mh * 0.6))
            x = mx + (mw - w) // 2
            y = my + (mh - h) // 2
        else:
            w, h = width, height
            x = y = 0
        self._root.geometry(f"{w}x{h}+{x}+{y}")
        self._root.configure(bg=self._BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=self._BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background="#2d2d2d", foreground=self._FG,
                        font=("Courier", 22, "bold"), padding=[16, 6])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#3c3c3c")],
                  foreground=[("selected", "#ffffff")])
        style.configure("Dark.TFrame", background=self._BG)

        pane = tk.PanedWindow(self._root, orient=tk.HORIZONTAL, sashwidth=6,
                              bg="#3c3c3c", bd=0)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── 좌측: 탭 Notebook ──
        left = tk.Frame(pane, bg=self._BG)
        nb = ttk.Notebook(left, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        # Solver 탭
        f_sol = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_solver = self._make_text_widget(f_sol)
        self._txt_solver.pack(fill=tk.BOTH, expand=True)
        nb.add(f_sol, text=" Solver ")

        # Objects 탭
        f_obj = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_objects = self._make_text_widget(f_obj)
        self._txt_objects.pack(fill=tk.BOTH, expand=True)
        nb.add(f_obj, text=" Objects ")

        # Contact 탭
        f_con = ttk.Frame(nb, style="Dark.TFrame")
        self._txt_contact = self._make_text_widget(f_con)
        self._txt_contact.pack(fill=tk.BOTH, expand=True)
        nb.add(f_con, text=" Contact ")

        pane.add(left, stretch="always", minsize=width // 1.2)

        # ── 우측: Solver Conv Log ──
        right = tk.Frame(pane, bg=self._BG)

        # 상단 요약 바
        self._conv_summary_label = tk.Label(
            right, text="", font=("Courier", 24, "bold"),
            bg="#2d2d2d", fg=self._FG_OK, anchor="w", padx=10, pady=4,
        )
        self._conv_summary_label.pack(fill=tk.X)

        # 로그 영역
        log_frame = tk.Frame(right, bg=self._BG)
        log_frame.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(log_frame, width=18)
        self._conv_text = tk.Text(
            log_frame, font=self._FONT_LOG, bg=self._BG, fg=self._FG,
            wrap=tk.NONE, state=tk.DISABLED, bd=0, padx=8, pady=4,
            yscrollcommand=scroll.set,
        )
        self._conv_text.tag_configure("hit", foreground=self._FG_HIT)
        self._conv_text.tag_configure("dim", foreground=self._FG_DIM)
        self._conv_text.tag_configure("sep", foreground="#3c3c3c")
        scroll.config(command=self._conv_text.yview)
        self._conv_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Clear 버튼
        btn = tk.Button(right, text="Clear", font=("Courier", 20),
                        bg="#3c3c3c", fg=self._FG, bd=0, padx=12, pady=4,
                        command=self._clear_conv)
        btn.pack(side=tk.BOTTOM, anchor="e", padx=8, pady=4)

        pane.add(right, stretch="always")

        self._poll()
        self._root.mainloop()

    def _set_text(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        for line in content.split("\n"):
            if line.startswith("──") or line.startswith("=="):
                widget.insert(tk.END, line + "\n", "header")
            elif "=" in line and not line.startswith(" "):
                widget.insert(tk.END, line + "\n", "header")
            else:
                widget.insert(tk.END, line + "\n")
        widget.config(state=tk.DISABLED)

    def _poll(self):
        if self._data_dirty:
            d = self._data
            if "objects" in d:
                self._set_text(self._txt_objects, d["objects"])
            if "contact" in d:
                self._set_text(self._txt_contact, d["contact"])
            if "solver" in d:
                self._set_text(self._txt_solver, d["solver"])
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
                if hit:
                    self._conv_text.insert(tk.END, ln + "\n", "hit")
                elif no_contact:
                    self._conv_text.insert(tk.END, ln + "\n", "dim")
                else:
                    self._conv_text.insert(tk.END, ln + "\n")
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

    def update(self, data: dict):
        if self._closed:
            return
        self._data = data
        self._data_dirty = True

    def push_conv(self, line: str, hit_limit: bool = False, no_contact: bool = False):
        if self._closed:
            return
        with self._lock:
            self._conv_lines.append((line, hit_limit, no_contact))
            self._conv_dirty = True

    def update_conv_summary(self, text: str):
        if self._closed:
            return
        self._conv_summary = text
        self._conv_summary_dirty = True


def _build_scene(p):
    """파라미터 dict로 model + solver + state 생성."""
    builder = newton.ModelBuilder(gravity=-p["gravity"])
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=p["ground_mu"], ke=p["ground_ke"], kd=p["ground_kd"], density=0.0)
    builder.add_ground_plane(cfg=ground_cfg)
    box_cfg = newton.ModelBuilder.ShapeConfig(
        mu=p["box_mu"], ke=p["box_ke"], kd=p["box_kd"], density=p["box_density"],
    )
    sph_cfg = newton.ModelBuilder.ShapeConfig(
        mu=p["sphere_mu"], ke=p["sphere_ke"], kd=p["sphere_kd"], density=p["sphere_density"],
    )

    body_box = builder.add_link(
        xform=wp.transform(p=(0.0, 0.0, p["box_half"]), q=wp.quat_identity()), label="box_yz",
    )
    builder.add_shape_box(body_box, hx=p["box_half"], hy=p["box_half"], hz=p["box_half"], cfg=box_cfg)
    j_box = builder.add_joint_d6(
        parent=-1, child=body_box,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, p["box_half"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_box])

    body_sphere_l = builder.add_link(
        xform=wp.transform(p=(0.0, -p["sphere_gap"], p["sphere_r"]), q=wp.quat_identity()), label="sphere_left",
    )
    builder.add_shape_sphere(body_sphere_l, radius=p["sphere_r"], cfg=sph_cfg)
    j_sl = builder.add_joint_d6(
        parent=-1, child=body_sphere_l,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, -p["sphere_gap"], p["sphere_r"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_sl])

    body_sphere_r = builder.add_link(
        xform=wp.transform(p=(0.0, p["sphere_gap"], p["sphere_r"]), q=wp.quat_identity()), label="sphere_right",
    )
    builder.add_shape_sphere(body_sphere_r, radius=p["sphere_r"], cfg=sph_cfg)
    j_sr = builder.add_joint_d6(
        parent=-1, child=body_sphere_r,
        linear_axes=[
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
        angular_axes=[],
        parent_xform=wp.transform(wp.vec3(0.0, p["sphere_gap"], p["sphere_r"]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_sr])

    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    model = builder.finalize()
    solver = SolverJK(
        model,
        iterations=p["solver_iterations"], ls_iterations=p["solver_ls_iterations"],
        cone=p["solver_cone"], impratio=p["solver_impratio"],
        njmax=200, nconmax=200,
        use_mujoco_contacts=False,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision = newton.CollisionPipeline(model)
    contacts = Contacts(
        solver.get_max_contact_count(), 0,
        requested_attributes=model.get_requested_contact_attributes(),
    )
    return model, solver, state_0, state_1, control, contacts, collision, body_box, body_sphere_l, body_sphere_r


def _make_params():
    """현재 module 상수에서 파라미터 dict 생성."""
    return dict(
        fps=FPS, sim_substeps=SIM_SUBSTEPS, sphere_force_y=SPHERE_FORCE_Y,
        solver_iterations=SOLVER_ITERATIONS, solver_ls_iterations=SOLVER_LS_ITERATIONS,
        solver_cone=SOLVER_CONE, solver_impratio=SOLVER_IMPRATIO,
        gravity=GRAVITY, box_half=BOX_HALF, sphere_r=SPHERE_R,
        box_mass=BOX_DENSITY * (2.0 * BOX_HALF) ** 3,
        sphere_mass=SPHERE_DENSITY * (4.0 / 3.0) * np.pi * SPHERE_R ** 3,
        box_density=BOX_DENSITY, sphere_density=SPHERE_DENSITY,
        box_mu=BOX_MU, sphere_mu=SPHERE_MU, ground_mu=GROUND_MU,
        ground_ke=GROUND_KE, ground_kd=GROUND_KD,
        box_ke=BOX_KE, box_kd=BOX_KD, sphere_ke=SPHERE_KE, sphere_kd=SPHERE_KD,
        sphere_gap=SPHERE_GAP,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0

        # 활성 파라미터 (시뮬레이션에서 사용)
        self._active = _make_params()
        # GUI pending 파라미터 (Reset 전까지 시뮬레이션에 반영 안 됨)
        self._p = _make_params()

        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts,
         self.collision, self.body_box, self.body_sphere_l, self.body_sphere_r) = _build_scene(self._active)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)

        self.print_interval = 0.5
        self.next_print_time = 0.0
        self._geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()
        self._shape_body = self.model.shape_body.numpy()
        self._init_max_pen_gpu()
        self._status_window = StatusWindow()

        box_mass = BOX_DENSITY * (2.0 * BOX_HALF) ** 3
        sph_mass = SPHERE_DENSITY * (4.0 / 3.0) * np.pi * SPHERE_R ** 3
        print(f"\n  Box: half={BOX_HALF}m, density={BOX_DENSITY:.0f} ({box_mass:.2f}kg)")
        print(f"  Sphere: r={SPHERE_R}m, density={SPHERE_DENSITY:.0f} ({sph_mass:.2f}kg)\n")

    def reset(self):
        """GUI pending 값을 활성화하고 model/solver/state 전부 재생성."""
        self._active = dict(self._p)
        _apply_module_constants(self._active)

        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts,
         self.collision, self.body_box, self.body_sphere_l, self.body_sphere_r) = _build_scene(self._active)

        self.viewer.reset_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)
        self.sim_time = 0.0
        self.next_print_time = 0.0
        self._geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()
        self._shape_body = self.model.shape_body.numpy()
        self._init_max_pen_gpu()
        print("[Reset] all params applied")

    def gui(self, imgui):
        """Param Tune — pending dict에만 저장, Reset 시 일괄 적용."""
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
        imgui.text("=== Contact ===")
        _, p["ground_ke"] = imgui.input_float("GND_KE", p["ground_ke"], format="%.0f")
        _, p["ground_kd"] = imgui.input_float("GND_KD", p["ground_kd"], format="%.0f")
        _, p["box_ke"] = imgui.input_float("BOX_KE", p["box_ke"], format="%.0f")
        _, p["box_kd"] = imgui.input_float("BOX_KD", p["box_kd"], format="%.0f")
        _, p["sphere_ke"] = imgui.input_float("SPHERE_KE", p["sphere_ke"], format="%.0f")
        _, p["sphere_kd"] = imgui.input_float("SPHERE_KD", p["sphere_kd"], format="%.0f")

        imgui.separator()
        imgui.text("Reset to apply all changes")


    def apply_sphere_forces(self):
        body_f = self.state_0.body_f.numpy()
        body_f[self.body_sphere_l][1] = +self._active["sphere_force_y"]
        body_f[self.body_sphere_r][1] = -self._active["sphere_force_y"]
        self.state_0.body_f.assign(body_f)

    def simulate(self):
        frame_dt = 1.0 / self._active["fps"]
        sim_dt = frame_dt / self._active["sim_substeps"]
        for sub in range(self._active["sim_substeps"]):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.apply_sphere_forces()
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, sim_dt)
            self._post_substep(sub)
        self.solver.update_contacts(self.contacts, self.state_0)

    def _init_max_pen_gpu(self):
        """GPU에서 max penetration 추적용 배열 초기화."""
        self._n_bodies = int(self.model.body_count)
        stride = self._n_bodies + 1  # -1(ground) → index 0
        self._max_pen_gpu = wp.zeros(stride * stride, dtype=wp.float32, device=self.model.device)

    def _post_substep(self, substep):
        d = self.solver.mjw_data
        # solver conv: 스칼라 3개만 GPU→CPU
        niter = int(d.solver_niter.numpy()[0])
        nacon = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])

        max_iter = self._active["solver_iterations"]
        hit = niter >= max_iter
        line = (
            f"t={self.sim_time:.3f} sub={substep}"
            f"  niter={niter}/{max_iter}"
            f"  nacon={nacon}  nefc={nefc}"
        )
        self._status_window.push_conv(line, hit_limit=hit, no_contact=(nacon == 0))
        self._status_window.update_conv_summary(
            f"[current]  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}"
        )

        # max penetration: GPU 커널로 atomic_max → CPU 전송 없음
        if nacon > 0:
            wp.launch(
                update_max_penetration_kernel,
                dim=d.naconmax,
                inputs=[
                    d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                    self.solver.mjc_geom_to_newton_shape, self.model.shape_body,
                    self._n_bodies,
                ],
                outputs=[self._max_pen_gpu],
                device=self.model.device,
            )

    def step(self):
        self.simulate()
        frame_dt = 1.0 / self._active["fps"]
        self.sim_time += frame_dt

        if self.sim_time >= self.next_print_time:
            self._update_status()
            self.next_print_time += self.print_interval

    def _get_contact_forces(self):
        """mjw_data에서 직접 efc_force를 읽어 body별/body쌍별 법선력+마찰력+침투깊이 반환.

        Returns:
            body_nf: {body_idx: 법선력 절대값 합 (N)}
            body_ff: {body_idx: 마찰력 절대값 합 (N)}
            pair_nf: {(body_a, body_b): 법선력 절대값 합 (N)}  (a < b)
            pair_ff: {(body_a, body_b): 마찰력 절대값 합 (N)}  (a < b)
            pair_pen: {(body_a, body_b): 최대 침투깊이 (m, 양수=침투)}  (a < b)
        """
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

        body_nf = {}
        body_ff = {}
        pair_nf = {}
        pair_ff = {}
        pair_pen = {}

        for c in range(nacon):
            efc_addr = contact_efc[c, 0]
            if efc_addr < 0:
                continue
            world = contact_worldid[c]
            dim = int(contact_dim[c])

            # 법선력: efc_address[0]
            nf = abs(float(efc_force[world, efc_addr]))

            # 마찰력: 나머지 efc rows
            ff = 0.0
            if is_pyramidal:
                for i in range(1, 2 * (dim - 1)):
                    ff += abs(float(efc_force[world, contact_efc[c, i]]))
            else:
                for i in range(1, dim):
                    ff += abs(float(efc_force[world, contact_efc[c, i]]))

            # 침투깊이: -dist (dist<0이면 침투, 양수로 변환)
            pen = -float(contact_dist[c])

            g0, g1 = contact_geom[c]
            s0 = self._geom_to_shape[world, g0]
            s1 = self._geom_to_shape[world, g1]
            b0 = self._shape_body[s0] if s0 >= 0 else -1
            b1 = self._shape_body[s1] if s1 >= 0 else -1

            for b in (b0, b1):
                body_nf[b] = body_nf.get(b, 0.0) + nf
                body_ff[b] = body_ff.get(b, 0.0) + ff

            key = (min(b0, b1), max(b0, b1))
            pair_nf[key] = pair_nf.get(key, 0.0) + nf
            pair_ff[key] = pair_ff.get(key, 0.0) + ff
            pair_pen[key] = max(pair_pen.get(key, 0.0), pen)

        return body_nf, body_ff, pair_nf, pair_ff, pair_pen

    def _update_status(self):
        """탭별 데이터를 생성하여 StatusWindow에 전달."""
        import math

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_f = self.state_0.body_f.numpy()
        body_mass = self.model.body_mass.numpy()
        body_nf, body_ff, pair_nf, pair_ff, pair_pen = self._get_contact_forces()
        p = self._active
        body_labels = {-1: "ground", self.body_box: "box", self.body_sphere_l: "sph_L", self.body_sphere_r: "sph_R"}

        # ── Objects 탭 ──
        obj = [f"t = {self.sim_time:.2f}s", ""]
        for label, idx in [("box", self.body_box), ("sphere_L", self.body_sphere_l), ("sphere_R", self.body_sphere_r)]:
            m = body_mass[idx]
            pos = body_q[idx][:3]
            vel = body_qd[idx][:3]
            ext_f = body_f[idx][:3]
            nf = body_nf.get(idx, 0.0)
            ff = body_ff.get(idx, 0.0)
            obj.append(f"── {label} ({m:.3f}kg) ──")
            obj.append(f"  pos  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            obj.append(f"  vel  ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
            obj.append(f"  ext  ({ext_f[0]:+.3f}, {ext_f[1]:+.3f}, {ext_f[2]:+.3f})")
            obj.append(f"  N={nf:.5f}  F={ff:.5f}")
            obj.append("")

        # ── Contact 탭 ──
        con = []
        con.append("── Solref (from ke/kd) ──")
        for label, ke, kd in [("ground", p["ground_ke"], p["ground_kd"]), ("box", p["box_ke"], p["box_kd"]), ("sphere", p["sphere_ke"], p["sphere_kd"])]:
            if ke > 0 and kd > 0:
                tc = 2.0 / kd
                dr = kd / (2.0 * math.sqrt(ke))
            else:
                tc, dr = 0.02, 1.0
            con.append(f"  {label:<8s} tc={tc:.5f}  dr={dr:.5f}")
        con.append("")

        con.append("── Solimp (per geom) ──")
        solimp = self.solver.mjw_model.geom_solimp.numpy()
        n_geom = solimp.shape[1]
        for g in range(n_geom):
            s = self._geom_to_shape[0, g]
            b = self._shape_body[s] if s >= 0 else -1
            label = body_labels.get(b, None)
            if label is None:
                continue
            v = solimp[0, g]
            con.append(f"  {label:<8s} dmin={v[0]:.3f} dmax={v[1]:.3f} w={v[2]:.4f}")
        con.append("")

        if pair_nf:
            con.append("── Pair Forces ──")
            con.append(f"  {'Pair':<20s} {'Normal(N)':>10s} {'Fric(N)':>10s} {'Pen(mm)':>10s} {'Max_Pen(mm)':>10s}")
            max_pen_cpu = self._max_pen_gpu.numpy()
            stride = self._n_bodies + 1
            for (a, b) in sorted(pair_nf.keys()):
                nf = pair_nf.get((a, b), 0.0)
                ff = pair_ff.get((a, b), 0.0)
                pen = pair_pen.get((a, b), 0.0)
                max_p = float(max_pen_cpu[(a + 1) * stride + (b + 1)])
                la = body_labels.get(a, f"b{a}")
                lb = body_labels.get(b, f"b{b}")
                pair_str = f"{la}<->{lb}"
                con.append(f"  {pair_str:<20s} {nf:>10.5f} {ff:>10.5f} {pen*1000:>10.3f} {max_p*1000:>10.3f}")

        # ── Solver 탭 ──
        frame_dt = 1.0 / p["fps"]
        sim_dt = frame_dt / p["sim_substeps"]
        sol = []
        sol.append("── Solver Config ──")
        sol.append(f"  iterations      {p['solver_iterations']}")
        sol.append(f"  ls_iterations   {p['solver_ls_iterations']}")
        sol.append(f"  cone            {p['solver_cone']}")
        sol.append(f"  impratio        {p['solver_impratio']:.2f}")
        sol.append("")
        sol.append("── Sim Config ──")
        sol.append(f"  FPS             {p['fps']}")
        sol.append(f"  substeps        {p['sim_substeps']}")
        sol.append(f"  frame_dt        {frame_dt:.6f}s")
        sol.append(f"  sim_dt          {sim_dt:.6f}s")
        sol.append(f"  gravity         {p['gravity']:.3f}")

        self._status_window.update({
            "objects": "\n".join(obj),
            "contact": "\n".join(con),
            "solver": "\n".join(sol),
        })

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def _apply_module_constants(p):
    """파라미터 dict를 module 상수에 반영."""
    import jk_solver_examples.contacts.test_contact_force as m
    m.GRAVITY = p["gravity"]
    m.BOX_HALF = p["box_half"]
    m.SPHERE_R = p["sphere_r"]
    m.BOX_DENSITY = p["box_density"]
    m.SPHERE_DENSITY = p["sphere_density"]
    m.BOX_MU = p["box_mu"]
    m.SPHERE_MU = p["sphere_mu"]
    m.GROUND_MU = p["ground_mu"]
    m.GROUND_KE = p["ground_ke"]
    m.GROUND_KD = p["ground_kd"]
    m.BOX_KE = p["box_ke"]
    m.BOX_KD = p["box_kd"]
    m.SPHERE_KE = p["sphere_ke"]
    m.SPHERE_KD = p["sphere_kd"]
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
