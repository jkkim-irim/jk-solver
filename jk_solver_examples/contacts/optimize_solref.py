"""Optimize solref via Adam + Forward FD — with live viewer.

Each rollout renders in the viewer so you can watch the optimization.
Status window shows current iteration, parameters, cost, gradient.

Run:
    uv run jk_solver_examples/contacts/optimize_solref.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import warp as wp

_JK_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _JK_ROOT / "assets"
sys.path.insert(0, str(_JK_ROOT))

import warnings
warnings.filterwarnings("ignore")

import mujoco
import newton
import newton.examples
from newton import JointTargetMode
from jk_solver_examples import init as jk_init
from jk_solver_examples.debug_monitor import StatusWindow
from jk_solver_examples.jk_solver import SolverJK

from jk_solver_examples.contacts.test_finger_pinch_env import (
    _build_scene, _make_params, _find_joint_index, _euler_deg_to_quat,
    _apply_mjc_joint_overrides, _apply_contact_solref_solimp,
    PINCH_POSE_DEG, PINCH_DURATION, INIT_POSE_DEG, MIMIC_CONSTRAINTS,
    CAM_POS, CAM_PITCH, CAM_YAW,
)

# =============================================================================
# Optimization config
# =============================================================================

PARAM_NAMES = ["eq_tc", "box_tc", "robot_tc"]
PARAM_BOUNDS = np.array([
    [1e-10, 0.05],     # eq_tc
    [1e-10, 0.05],     # box_tc
    [1e-10, 0.05],     # robot_tc
])
FIXED_DR = 1.0        # dampratio fixed at 1.0

FD_EPS = np.array([0.005, 0.005, 0.005])

W_EQ_MEAN = 0.8      # mean eq_viol (rad) — stable gradient
W_EQ_MAX = 0.2       # max eq_viol (rad) — peak suppression
W_PEN_MM = 0.02      # mean_pen (mm)
W_DIVERGE = 10.0

HORIZON_SEC = 0.5   # rollout duration

# Adam hyperparameters
ADAM_LR = 0.01       # 5x larger for tc scale ~0.02
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Early stopping
PATIENCE = 10        # stop if best_cost doesn't improve for N iterations
COST_ATOL = 0.05     # minimum improvement to reset patience


# =============================================================================
# Adam optimizer
# =============================================================================

class Adam:
    def __init__(self, n_params, lr=ADAM_LR, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, x, grad, bounds):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        x_new = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return np.clip(x_new, bounds[:, 0], bounds[:, 1])


# =============================================================================
# Optimizer state machine (runs inside viewer loop)
# =============================================================================

class SolrefOptimizer:
    """Runs inside Example.step(), one rollout per viewer cycle."""

    # States
    IDLE = 0
    ROLLOUT = 1       # running current rollout
    NEXT_ROLLOUT = 2  # reset and start next FD rollout

    def __init__(self, example):
        self.ex = example
        self.state = self.IDLE
        self.adam = Adam(3)
        self.iteration = 0
        self.best_cost = float("inf")
        self.best_x = None

        # Current params: 3 tc values only (dr fixed at 1.0)
        p = example._active
        self.x = np.array([
            p["eq_solref"][0],
            p["box_solref"][0],
            p["robot_solref"][0],
        ], dtype=np.float64)
        self.best_x = self.x.copy()
        self._patience_counter = 0

        # FD state: 0=baseline, 1..3=forward perturbation
        self.fd_idx = 0
        self.fd_costs = np.zeros(4)  # baseline + 3 params
        self.rollout_time = 0.0
        self.max_eq_viol = 0.0
        self.max_pen = 0.0
        self.horizon_steps = 0
        self.horizon_max = int(HORIZON_SEC / (1.0 / (p["fps"] * p["sim_substeps"])))

    def start(self):
        """Begin optimization from current params."""
        self.state = self.NEXT_ROLLOUT
        self.fd_idx = 0
        self.iteration = 0
        print("=" * 60)
        print("Adam + FD Optimization STARTED")
        print("x0: %s" % dict(zip(PARAM_NAMES, ["%.10f" % v for v in self.x])))
        print("box_geoms: %s  index_geoms: %s  thumb_geoms: %s" % (
            self.ex._box_geoms, self.ex._index_geoms, self.ex._thumb_geoms))
        print("horizon_max: %d steps (%.3fs)" % (self.horizon_max, HORIZON_SEC))
        print("=" * 60)

    def stop(self):
        self.state = self.IDLE
        print()
        print("=" * 60)
        print("Optimization STOPPED (iter=%d, best_cost=%.4f)" % (self.iteration, self.best_cost))
        print("EQ_SOLREF = (%.10f, %.1f)" % (self.best_x[0], FIXED_DR))
        print("BOX_SOLREF = (%.10f, %.1f)" % (self.best_x[1], FIXED_DR))
        print("ROBOT_SOLREF = (%.10f, %.1f)" % (self.best_x[2], FIXED_DR))
        print("=" * 60)

    def _get_fd_params(self):
        """Return params for current fd_idx."""
        if self.fd_idx == 0:
            return self.x.copy()
        else:
            x_pert = self.x.copy()
            i = self.fd_idx - 1
            x_pert[i] += FD_EPS[i]
            return np.clip(x_pert, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])

    def _apply_solref_to_scene(self, x_params):
        """Override solref on mjw_model. x_params = [eq_tc, box_tc, robot_tc]."""
        mjw = self.ex.solver.mjw_model
        mj = self.ex.solver.mj_model

        eq_solref_np = mjw.eq_solref.numpy()
        for i in range(eq_solref_np.shape[1]):
            eq_solref_np[0, i] = [float(x_params[0]), FIXED_DR]
        mjw.eq_solref.assign(eq_solref_np)

        geom_solref_np = mjw.geom_solref.numpy()
        for i in range(mj.ngeom):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
            if name.startswith("shape_1"):
                geom_solref_np[0, i] = [float(x_params[1]), FIXED_DR]
            elif "/ALLEX/" in name:
                geom_solref_np[0, i] = [float(x_params[2]), FIXED_DR]
        mjw.geom_solref.assign(geom_solref_np)

    def _reset_and_start_rollout(self):
        """Reset sim state and start pinch trajectory for current FD rollout."""
        ex = self.ex

        # Apply solref BEFORE reset (so solver uses new params)
        params = self._get_fd_params()
        self._apply_solref_to_scene(params)

        # Full state reset — rebuild state/control/contacts from model
        ex.model.joint_q.assign(ex._init_joint_q)
        ex.model.joint_qd.assign(ex._init_joint_qd)
        newton.eval_fk(ex.model, ex.model.joint_q, ex.model.joint_qd, ex.model)
        ex.state_0 = ex.model.state()
        ex.state_1 = ex.model.state()
        ex.control = ex.model.control()
        ex.contacts = ex.model.contacts()
        ex.control.joint_target_pos.assign(ex._init_target)

        # Re-apply joint drive targets for init pose
        for name, deg in INIT_POSE_DEG.items():
            for i, lbl in enumerate(ex._joint_labels):
                if lbl.endswith(name):
                    qd = ex._joint_qd_starts[i]
                    ex.control.joint_target_pos.numpy()[qd] = np.radians(deg)
                    break
        ex.control.joint_target_pos.assign(ex.control.joint_target_pos.numpy())

        # Force solver to re-sync with new state
        ex.solver._step = 0

        # Start pinch trajectory
        ex._pinch_active = True
        ex._traj = {
            "start": ex._init_target.copy(),
            "goal": ex._goal_target.copy(),
            "t0": 0.0,
            "duration": PINCH_DURATION,
        }
        ex.sim_time = 0.0
        ex.next_print_time = 0.0
        ex._max_pen = {"box<->index": 0.0, "box<->thumb": 0.0}
        self._dbg_printed = False

        # Reset metrics
        self.rollout_time = 0.0
        self.max_eq_viol = 0.0
        self.eq_sum = 0.0
        self.eq_count = 0
        self.pen_sum = 0.0
        self.pen_count = 0
        self.horizon_steps = 0
        self.state = self.ROLLOUT

    def on_substep(self):
        """Called every substep during ROLLOUT state. Collect metrics."""
        if self.state != self.ROLLOUT:
            return

        self.horizon_steps += 1
        d = self.ex.solver.mjw_data
        nefc = int(d.nefc.numpy()[0])
        nacon = int(d.nacon.numpy()[0])

        # Equality violation — only Index DIP (eq[1]) and Thumb IP (eq[0])
        if nefc > 0:
            efc_pos = d.efc.pos.numpy()[0]
            efc_type = d.efc.type.numpy()[0]
            eq_rows = [i for i in range(nefc) if efc_type[i] == 0]
            for idx in [0, 1]:
                if idx < len(eq_rows):
                    r = eq_rows[idx]
                    v = abs(float(efc_pos[r]))
                    self.max_eq_viol = max(self.max_eq_viol, v)
                    self.eq_sum += v
                    self.eq_count += 1

        # Penetration — time-averaged (sum + count)
        if nacon > 0:
            cg = d.contact.geom.numpy()[:nacon]
            cd = d.contact.dist.numpy()[:nacon]
            for c in range(nacon):
                g0, g1 = int(cg[c, 0]), int(cg[c, 1])
                is_box = g0 in self.ex._box_geoms or g1 in self.ex._box_geoms
                if not is_box:
                    continue
                other = g1 if g0 in self.ex._box_geoms else g0
                if other in self.ex._index_geoms or other in self.ex._thumb_geoms:
                    pen = -float(cd[c])
                    if pen > 0:
                        self.pen_sum += pen
                        self.pen_count += 1

        # NaN check
        qpos = d.qpos.numpy()[0]
        if np.any(np.isnan(qpos)):
            self.fd_costs[self.fd_idx] = W_DIVERGE
            self._advance_fd()
            return

        # Check if rollout complete
        if self.horizon_steps >= self.horizon_max:
            mean_eq = (self.eq_sum / self.eq_count) if self.eq_count > 0 else 0.0
            mean_pen_mm = (self.pen_sum / self.pen_count * 1000.0) if self.pen_count > 0 else 0.0
            cost = W_EQ_MEAN * mean_eq + W_EQ_MAX * self.max_eq_viol + W_PEN_MM * mean_pen_mm
            self.fd_costs[self.fd_idx] = cost

            params = self._get_fd_params()
            tag = "baseline" if self.fd_idx == 0 else PARAM_NAMES[self.fd_idx - 1]
            print("[iter %d fd %d/%d (%s)] cost=%.4f mean_eq=%.4f max_eq=%.4f rad pen=%.3f mm" % (
                self.iteration, self.fd_idx, 3, tag,
                cost, mean_eq, self.max_eq_viol, mean_pen_mm))

            self._advance_fd()

    def _advance_fd(self):
        """Move to next FD rollout or finish iteration."""
        self.fd_idx += 1
        if self.fd_idx <= 3:
            # More FD rollouts needed
            self.state = self.NEXT_ROLLOUT
        else:
            # All 7 rollouts done → compute gradient and update
            self._finish_iteration()

    def _finish_iteration(self):
        """Compute FD gradient, Adam update, log results."""
        baseline_cost = self.fd_costs[0]
        grad = np.zeros(3)
        for i in range(3):
            actual_eps = FD_EPS[i]
            x_pert = self.x.copy()
            x_pert[i] += actual_eps
            x_pert[i] = np.clip(x_pert[i], PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])
            actual_eps = x_pert[i] - self.x[i]
            if abs(actual_eps) > 1e-12:
                grad[i] = (self.fd_costs[i + 1] - baseline_cost) / actual_eps

        x_new = self.adam.step(self.x, grad, PARAM_BOUNDS)

        if baseline_cost < self.best_cost - COST_ATOL:
            self.best_cost = baseline_cost
            self.best_x = self.x.copy()
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        grad_norm = np.linalg.norm(grad)
        print()
        print("[ITER %d DONE] baseline_cost=%.4f  best=%.4f  |grad|=%.4f  patience=%d/%d" % (
            self.iteration, baseline_cost, self.best_cost, grad_norm,
            self._patience_counter, PATIENCE))
        for i, name in enumerate(PARAM_NAMES):
            fmt = "%.10f" if "tc" in name else "%.6f"
            print(("  %-10s " + fmt + " -> " + fmt + "  (grad=%+.4f)") % (name, self.x[i], x_new[i], grad[i]))
        print()

        self.x = x_new
        self.iteration += 1

        # Early stopping
        if self._patience_counter >= PATIENCE:
            print("[CONVERGED] No improvement for %d iterations" % PATIENCE)
            self.stop()
            self.ex._opt_active = False
            return

        self.fd_idx = 0
        self.state = self.NEXT_ROLLOUT

    def get_status_text(self):
        """Text for status window."""
        if self.state == self.IDLE:
            return "Optimizer: IDLE"
        tag = "baseline" if self.fd_idx == 0 else PARAM_NAMES[self.fd_idx - 1]
        params = self._get_fd_params()
        lines = [
            "── Optimizer ──",
            "iter=%d  fd=%d/3 (%s)  step=%d/%d" % (
                self.iteration, self.fd_idx, tag, self.horizon_steps, self.horizon_max),
            "best_cost=%.4f" % self.best_cost,
            "",
            "── Current params ──",
        ]
        for i, name in enumerate(PARAM_NAMES):
            fmt = "%.10f" if "tc" in name else "%.6f"
            lines.append(("  %-10s " + fmt) % (name, params[i]))
        lines.append("")
        lines.append("── Current rollout ──")
        mean_eq = (self.eq_sum / self.eq_count) if self.eq_count > 0 else 0.0
        lines.append("  mean_eq=%.5f  max_eq=%.5f rad (%.2f deg)" % (mean_eq, self.max_eq_viol, np.degrees(self.max_eq_viol)))
        mean_pen = (self.pen_sum / self.pen_count * 1000) if self.pen_count > 0 else 0.0
        lines.append("  mean_pen=%.4f mm" % mean_pen)
        return "\n".join(lines)


# =============================================================================
# Example (extends pinch env with optimizer)
# =============================================================================

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self._active = _make_params()
        self._p = _make_params()

        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts) = _build_scene(self._active)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)
        self.print_interval = 0.5
        self.next_print_time = 0.0
        self._cache_joint_info()
        self._pinch_active = False
        self._lift_active = False
        self._traj = None
        self._max_pen = {"box<->index": 0.0, "box<->thumb": 0.0}
        self._cache_geom_groups()
        self._cache_init_state()
        self._status_window = StatusWindow(geom_labels=[])

        # Optimizer
        self._optimizer = SolrefOptimizer(self)
        self._opt_active = False

    def _cache_init_state(self):
        """Cache initial state for optimizer resets."""
        self._init_joint_q = self.model.joint_q.numpy().copy()
        self._init_joint_qd = self.model.joint_qd.numpy().copy()
        # Cache pinch targets
        target_pos = self.control.joint_target_pos.numpy().copy()
        self._init_target = target_pos.copy()
        goal_pos = target_pos.copy()
        for name, deg in PINCH_POSE_DEG.items():
            for i, lbl in enumerate(self._joint_labels):
                if lbl.endswith(name):
                    goal_pos[self._joint_qd_starts[i]] = np.radians(deg)
                    break
        self._goal_target = goal_pos

    def _cache_geom_groups(self):
        import mujoco as mj_mod
        mj = self.solver.mj_model
        self._box_geoms = set()
        self._index_geoms = set()
        self._thumb_geoms = set()
        for i in range(mj.ngeom):
            bname = mj_mod.mj_id2name(mj, mj_mod.mjtObj.mjOBJ_BODY, mj.geom_bodyid[i]) or ""
            if "box" in bname: self._box_geoms.add(i)
            elif "Index" in bname: self._index_geoms.add(i)
            elif "Thumb" in bname: self._thumb_geoms.add(i)

    def _get_contact_penetration(self):
        d = self.solver.mjw_data
        nacon = int(d.nacon.numpy()[0])
        pen_index, pen_thumb = 0.0, 0.0
        if nacon > 0:
            cg = d.contact.geom.numpy()[:nacon]
            cd = d.contact.dist.numpy()[:nacon]
            for c in range(nacon):
                g0, g1 = int(cg[c, 0]), int(cg[c, 1])
                pen = -float(cd[c])
                is_box = g0 in self._box_geoms or g1 in self._box_geoms
                if not is_box: continue
                other = g1 if g0 in self._box_geoms else g0
                if other in self._index_geoms: pen_index = max(pen_index, pen)
                elif other in self._thumb_geoms: pen_thumb = max(pen_thumb, pen)
        return pen_index, pen_thumb

    def _cache_joint_info(self):
        self._joint_labels = [self.model.joint_label[i] for i in range(self.model.joint_count)]
        self._joint_q_starts = self.model.joint_q_start.numpy()
        self._joint_qd_starts = self.model.joint_qd_start.numpy()
        self._idx_pip_ji = next(
            (i for i, l in enumerate(self._joint_labels) if l.endswith("R_Index_PIP_Joint")), -1)

    def reset(self):
        self._active = dict(self._p)
        (self.model, self.solver, self.state_0, self.state_1,
         self.control, self.contacts) = _build_scene(self._active)
        self.viewer.reset_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(*CAM_POS), pitch=CAM_PITCH, yaw=CAM_YAW)
        self._cache_joint_info()
        self._cache_geom_groups()
        self._cache_init_state()
        self._pinch_active = False
        self._lift_active = False
        self._traj = None
        self._max_pen = {"box<->index": 0.0, "box<->thumb": 0.0}
        self.sim_time = 0.0
        self.next_print_time = 0.0
        self._status_window.clear_joint_data()
        self._opt_active = False
        self._optimizer = SolrefOptimizer(self)
        print("[Reset] scene rebuilt")

    def gui(self, imgui):
        if not getattr(self.viewer, "_irim_param_tune_open", False):
            return
        p = self._p

        # Optimizer button
        opt_label = "Optimize [ON]" if self._opt_active else "Optimize"
        if imgui.button(opt_label):
            self._opt_active = not self._opt_active
            if self._opt_active:
                self._optimizer.start()
            else:
                self._optimizer.stop()
        imgui.same_line()

        label = "Pinch [ON]" if self._pinch_active else "Pinch"
        if imgui.button(label):
            self._pinch_active = not self._pinch_active
            self._apply_pinch_target()
        imgui.same_line()
        lift_label = "Lift [ON]" if self._lift_active else "Lift"
        if imgui.button(lift_label):
            self._lift_active = not self._lift_active
            self._apply_lift_target()
        imgui.separator()

        imgui.text("=== Solver ===")
        _, v = imgui.input_int("iter", p["solver_iterations"])
        p["solver_iterations"] = max(1, v)
        _, v = imgui.input_int("ls_iter", p["solver_ls_iterations"])
        p["solver_ls_iterations"] = max(1, v)
        _, p["solver_impratio"] = imgui.input_float("impratio", p["solver_impratio"], format="%.2f")

        imgui.separator()
        imgui.text("=== Joint Drive ===")
        _, p["wrist_ke"] = imgui.input_float("wrist_ke", p["wrist_ke"], format="%.1f")
        _, p["wrist_kd"] = imgui.input_float("wrist_kd", p["wrist_kd"], format="%.1f")

        imgui.separator()
        imgui.text("Reset to apply all changes")

    def _apply_pinch_target(self):
        current_pos = self.control.joint_target_pos.numpy().copy()
        goal_pos = current_pos.copy()
        for name in PINCH_POSE_DEG:
            deg = PINCH_POSE_DEG[name] if self._pinch_active else INIT_POSE_DEG.get(name, 0.0)
            for i, lbl in enumerate(self._joint_labels):
                if lbl.endswith(name):
                    goal_pos[self._joint_qd_starts[i]] = np.radians(deg)
                    break
        self._traj = {"start": current_pos, "goal": goal_pos, "t0": self.sim_time, "duration": PINCH_DURATION}

    def _apply_lift_target(self):
        current_pos = self.control.joint_target_pos.numpy().copy()
        goal_pos = current_pos.copy()
        deg = 0.0 if self._lift_active else INIT_POSE_DEG.get("R_Wrist_Pitch_Joint", 0.0)
        for i, lbl in enumerate(self._joint_labels):
            if lbl.endswith("R_Wrist_Pitch_Joint"):
                goal_pos[self._joint_qd_starts[i]] = np.radians(deg)
                break
        self._traj = {"start": current_pos, "goal": goal_pos, "t0": self.sim_time, "duration": PINCH_DURATION}

    def _update_traj(self):
        if self._traj is None:
            return
        elapsed = self.sim_time - self._traj["t0"]
        alpha = min(elapsed / self._traj["duration"], 1.0)
        interp = (1.0 - alpha) * self._traj["start"] + alpha * self._traj["goal"]
        self.control.joint_target_pos.assign(interp)
        if alpha >= 1.0:
            self._traj = None

    def simulate(self):
        frame_dt = 1.0 / self._active["fps"]
        sim_dt = frame_dt / self._active["sim_substeps"]

        # If optimizer needs to start next rollout, do it before stepping
        if self._opt_active and self._optimizer.state == SolrefOptimizer.NEXT_ROLLOUT:
            self._optimizer._reset_and_start_rollout()

        for sub in range(self._active["sim_substeps"]):
            self.sim_time += sim_dt
            self._update_traj()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, sim_dt)
            self._post_substep(sub)

            # Optimizer collects metrics
            if self._opt_active:
                self._optimizer.on_substep()

    def _post_substep(self, substep):
        d = self.solver.mjw_data
        niter = int(d.solver_niter.numpy()[0])
        nacon = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])
        max_iter = self._active["solver_iterations"]
        hit = niter >= max_iter
        self._status_window.push_conv(
            f"({substep})t={self.sim_time:.5f}  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}",
            hit_limit=hit, no_contact=(nacon == 0))
        self._status_window.update_conv_summary(
            f"[current]  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}")

        pen_idx, pen_thm = self._get_contact_penetration()
        self._max_pen["box<->index"] = max(self._max_pen["box<->index"], pen_idx)
        self._max_pen["box<->thumb"] = max(self._max_pen["box<->thumb"], pen_thm)
        self._status_window.update({"contact": (
            f"── Contact Penetration (t={self.sim_time:.3f}s) ──\n"
            f"  {'Pair':<20s} {'Pen(mm)':>10s} {'Max_Pen(mm)':>12s}\n"
            f"  {'box<->index':<20s} {pen_idx*1000:>10.4f} {self._max_pen['box<->index']*1000:>12.4f}\n"
            f"  {'box<->thumb':<20s} {pen_thm*1000:>10.4f} {self._max_pen['box<->thumb']*1000:>12.4f}"
        )})

        if nefc > 0:
            efc_force = d.efc.force.numpy()[0]
            efc_pos = d.efc.pos.numpy()[0]
            efc_type = d.efc.type.numpy()[0]
            eq_rows = [i for i in range(nefc) if efc_type[i] == 0]
            idx_dip_row = eq_rows[1] if len(eq_rows) > 1 else -1
            if idx_dip_row >= 0:
                dip_force = float(efc_force[idx_dip_row])
                dip_pos = float(efc_pos[idx_dip_row])
                pip_qd = self._joint_qd_starts[self._idx_pip_ji]
                pip_torque = float(d.qfrc_actuator.numpy()[0, pip_qd])
                self._status_window.push_joint_data(self.sim_time, dip_force, dip_pos, pip_torque)
                self._status_window.push_joint_conv(
                    f"({substep})t={self.sim_time:.5f}  efc_f={dip_force:>9.5f}  "
                    f"efc_p={dip_pos:>9.5f}  pip_t={pip_torque:>9.5f}")

    def _update_status(self):
        d = self.solver.mjw_data
        p = self._active
        niter = int(d.solver_niter.numpy()[0])
        nacon = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])

        sol_lines = [
            "── Solver ──",
            f"  iter: {p['solver_iterations']}  ls: {p['solver_ls_iterations']}",
            f"  niter={niter}  nacon={nacon}  nefc={nefc}",
        ]

        # Add optimizer status
        if self._opt_active:
            sol_lines.append("")
            sol_lines.append(self._optimizer.get_status_text())

        qpos = d.qpos.numpy()[0]
        obj_lines = [f"── Joints (t={self.sim_time:.2f}s) ──"]
        for name in INIT_POSE_DEG:
            for i, label in enumerate(self._joint_labels):
                if label.endswith(name):
                    qi = self._joint_q_starts[i]
                    obj_lines.append(f"  {name:<28s} {np.degrees(qpos[qi]):>8.2f} deg")
                    break

        self._status_window.update({
            "solver": "\n".join(sol_lines),
            "objects": "\n".join(obj_lines),
        })

    def step(self):
        self.simulate()
        # Always update status when optimizer is active (sim_time resets each rollout)
        if self._opt_active or self.sim_time >= self.next_print_time:
            self._update_status()
            if not self._opt_active:
                self.next_print_time += self.print_interval

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def _run_headless(max_iters=50):
    """Run optimization headless (no viewer)."""
    print("=" * 60)
    print("Adam + FD Optimization (headless)")
    print("=" * 60)

    p = _make_params()
    p["solver_iterations"] = 100
    p["solver_ls_iterations"] = 50
    p["solver_tolerance"] = 1e-6
    p["solver_ls_tolerance"] = 1e-6

    model, solver, s0, s1, ctrl, contacts = _build_scene(p)
    d = solver.mjw_data
    mj = solver.mj_model
    mjw = solver.mjw_model
    dt = 1.0 / (p["fps"] * p["sim_substeps"])
    horizon_steps = int(HORIZON_SEC / dt)

    joint_labels = [model.joint_label[i] for i in range(model.joint_count)]
    joint_qd_starts = model.joint_qd_start.numpy()
    init_joint_q = model.joint_q.numpy().copy()
    init_joint_qd = model.joint_qd.numpy().copy()
    init_target = ctrl.joint_target_pos.numpy().copy()
    goal_target = init_target.copy()
    for name, deg in PINCH_POSE_DEG.items():
        for i, lbl in enumerate(joint_labels):
            if lbl.endswith(name):
                goal_target[joint_qd_starts[i]] = np.radians(deg)
                break

    box_geoms, index_geoms, thumb_geoms = set(), set(), set()
    for i in range(mj.ngeom):
        bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, mj.geom_bodyid[i]) or ""
        if "box" in bname: box_geoms.add(i)
        elif "Index" in bname: index_geoms.add(i)
        elif "Thumb" in bname: thumb_geoms.add(i)

    x = np.array([p["eq_solref"][0], p["box_solref"][0], p["robot_solref"][0]], dtype=np.float64)
    adam = Adam(3)
    best_cost, best_x = float("inf"), x.copy()

    def apply_solref(x_params):
        eq_sr = mjw.eq_solref.numpy()
        for i in range(eq_sr.shape[1]):
            eq_sr[0, i] = [float(x_params[0]), FIXED_DR]
        mjw.eq_solref.assign(eq_sr)
        g_sr = mjw.geom_solref.numpy()
        for i in range(mj.ngeom):
            nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i)
            if nm is None: continue
            if nm.startswith("shape_1"): g_sr[0, i] = [float(x_params[1]), FIXED_DR]
            elif "/ALLEX/" in nm: g_sr[0, i] = [float(x_params[2]), FIXED_DR]
        mjw.geom_solref.assign(g_sr)

    def evaluate(x_eval):
        x_c = np.clip(x_eval, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        apply_solref(x_c)
        model.joint_q.assign(init_joint_q)
        model.joint_qd.assign(init_joint_qd)
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        s = model.state()
        c = model.control()
        c.joint_target_pos.assign(init_target)

        traj_start, traj_goal = init_target.copy(), goal_target.copy()
        max_eq, eq_sum, eq_count, pen_sum, pen_count = 0.0, 0.0, 0, 0.0, 0
        for step in range(horizon_steps):
            t = (step + 1) * dt
            alpha = min(t / PINCH_DURATION, 1.0)
            c.joint_target_pos.assign((1 - alpha) * traj_start + alpha * traj_goal)
            s.clear_forces()
            model.collide(s, contacts)
            solver.step(s, s, c, contacts, dt)
            qpos = d.qpos.numpy()[0]
            if np.any(np.isnan(qpos)): return W_DIVERGE, 0.0, 0.0, 0.0
            nefc = int(d.nefc.numpy()[0])
            nacon = int(d.nacon.numpy()[0])
            if nefc > 0:
                ep, et = d.efc.pos.numpy()[0], d.efc.type.numpy()[0]
                eq_rows = [i for i in range(nefc) if et[i] == 0]
                for idx in [0, 1]:
                    if idx < len(eq_rows):
                        v = abs(float(ep[eq_rows[idx]]))
                        max_eq = max(max_eq, v)
                        eq_sum += v
                        eq_count += 1
            if nacon > 0:
                cg, cd = d.contact.geom.numpy()[:nacon], d.contact.dist.numpy()[:nacon]
                for ci in range(nacon):
                    g0, g1 = int(cg[ci, 0]), int(cg[ci, 1])
                    ib = g0 in box_geoms or g1 in box_geoms
                    if not ib: continue
                    o = g1 if g0 in box_geoms else g0
                    if o in index_geoms or o in thumb_geoms:
                        pen = -float(cd[ci])
                        if pen > 0:
                            pen_sum += pen
                            pen_count += 1
        mean_eq = (eq_sum / eq_count) if eq_count > 0 else 0.0
        mean_pen_mm = (pen_sum / pen_count * 1000.0) if pen_count > 0 else 0.0
        cost = W_EQ_MEAN * mean_eq + W_EQ_MAX * max_eq + W_PEN_MM * mean_pen_mm
        return cost, max_eq, mean_pen_mm

    print("Initial: %s" % dict(zip(PARAM_NAMES, ["%.10f" % v for v in x])))
    print()
    patience_count = 0

    for it in range(max_iters):
        costs = np.zeros(4)  # baseline + 3
        c0, max_eq0, pen0 = evaluate(x)
        costs[0] = c0
        for i in range(3):
            x_p = x.copy()
            x_p[i] = min(x_p[i] + FD_EPS[i], PARAM_BOUNDS[i, 1])
            costs[i + 1], _, _ = evaluate(x_p)

        grad = np.zeros(3)
        for i in range(3):
            eps_actual = min(x[i] + FD_EPS[i], PARAM_BOUNDS[i, 1]) - x[i]
            if abs(eps_actual) > 1e-12:
                grad[i] = (costs[i + 1] - costs[0]) / eps_actual

        x_new = adam.step(x, grad, PARAM_BOUNDS)
        if costs[0] < best_cost - COST_ATOL:
            best_cost = costs[0]
            best_x = x.copy()
            patience_count = 0
        else:
            patience_count += 1

        mean_eq_val = costs[0] / W_EQ if W_EQ > 0 else 0  # approximate
        print("[iter %2d] cost=%.4f  best=%.4f  max_eq=%.4f rad (%.1f deg)  pen=%.3f mm  |grad|=%.2f  patience=%d/%d" % (
            it, costs[0], best_cost, max_eq0, np.degrees(max_eq0), pen0, np.linalg.norm(grad), patience_count, PATIENCE))
        for i, nm in enumerate(PARAM_NAMES):
            print("  %-10s %.10f -> %.10f  (g=%+.2f)" % (nm, x[i], x_new[i], grad[i]))
        print()
        x = x_new

        if patience_count >= PATIENCE:
            print("[CONVERGED] No improvement for %d iterations" % PATIENCE)
            break

    print("=" * 60)
    print("RESULT (best_cost=%.4f)" % best_cost)
    print("EQ_SOLREF = (%.10f, %.1f)" % (best_x[0], FIXED_DR))
    print("BOX_SOLREF = (%.10f, %.1f)" % (best_x[1], FIXED_DR))
    print("ROBOT_SOLREF = (%.10f, %.1f)" % (best_x[2], FIXED_DR))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="Run without viewer")
    ap.add_argument("--max-iters", type=int, default=50, help="Max iterations (headless)")
    known, remaining = ap.parse_known_args()

    if known.headless:
        _run_headless(max_iters=known.max_iters)
    else:
        sys.argv = [sys.argv[0]] + remaining  # pass remaining to newton parser
        parser = newton.examples.create_parser()
        viewer, args = jk_init(parser)
        example = Example(viewer, args)
        newton.examples.run(example, args)
