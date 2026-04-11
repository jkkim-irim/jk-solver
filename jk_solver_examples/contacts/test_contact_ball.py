"""Contact force test — measure normal/friction forces, penetration depth, solver convergence.

Scenario: 1 box (550g) + 2 spheres (5g each) on ground plane.
Spheres push into box with constant y-force.
Verifies: static normal force ≈ mg, geometric mean friction override, solimp/solref tuning.

Run:
    python jk_solver_examples/contacts/test_contact_ball.py
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp

_JK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_JK_ROOT))

import newton
import newton.examples
from jk_solver_examples import init as jk_init
from jk_solver_examples.debug_monitor import StatusWindow
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
        njmax=200, nconmax=200, use_mujoco_contacts=True)
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
        self._status_window = StatusWindow(geom_labels=["ground", "box", "sph_L", "sph_R"])

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
        """Read efc_force from mjw_data. Returns per-body and per-pair vector-summed normal/friction/penetration."""
        d = self.solver.mjw_data
        nacon = int(d.nacon.numpy()[0])
        if nacon == 0:
            return {}, {}, {}, {}, {}

        contact_geom = d.contact.geom.numpy()[:nacon]
        contact_efc = d.contact.efc_address.numpy()[:nacon]
        contact_dim = d.contact.dim.numpy()[:nacon]
        contact_worldid = d.contact.worldid.numpy()[:nacon]
        contact_dist = d.contact.dist.numpy()[:nacon]
        contact_frame = d.contact.frame.numpy()[:nacon]  # (nacon, 3, 3) — row0=normal
        efc_force = d.efc.force.numpy()
        is_pyramidal = self.solver.mjw_model.opt.cone == int(self.solver._mujoco.mjtCone.mjCONE_PYRAMIDAL)

        _zero3 = np.zeros(3)
        body_nf_vec, body_ff_vec = {}, {}
        pair_nf_vec, pair_ff_vec, pair_pen = {}, {}, {}

        for c in range(nacon):
            efc_addr = contact_efc[c, 0]
            if efc_addr < 0:
                continue
            world = contact_worldid[c]
            dim = int(contact_dim[c])
            normal = contact_frame[c][0]  # contact normal direction (3,)

            # normal force vector
            nf_scalar = float(efc_force[world, efc_addr])
            nf_vec = nf_scalar * normal

            # friction force vector (tangent directions = frame rows 1,2)
            ff_vec = np.zeros(3)
            if is_pyramidal:
                for i in range(1, 2 * (dim - 1)):
                    fi = float(efc_force[world, contact_efc[c, i]])
                    t_idx = (i - 1) % (dim - 1)
                    ff_vec += fi * contact_frame[c][1 + t_idx]
            else:
                for i in range(1, dim):
                    fi = float(efc_force[world, contact_efc[c, i]])
                    ff_vec += fi * contact_frame[c][i]

            pen = -float(contact_dist[c])

            g0, g1 = contact_geom[c]
            b0 = self._shape_body[self._geom_to_shape[world, g0]] if self._geom_to_shape[world, g0] >= 0 else -1
            b1 = self._shape_body[self._geom_to_shape[world, g1]] if self._geom_to_shape[world, g1] >= 0 else -1

            for b in (b0, b1):
                body_nf_vec[b] = body_nf_vec.get(b, _zero3) + nf_vec
                body_ff_vec[b] = body_ff_vec.get(b, _zero3) + ff_vec

            key = (min(b0, b1), max(b0, b1))
            pair_nf_vec[key] = pair_nf_vec.get(key, _zero3) + nf_vec
            pair_ff_vec[key] = pair_ff_vec.get(key, _zero3) + ff_vec
            pair_pen[key] = max(pair_pen.get(key, 0.0), pen)

        # convert to magnitudes
        body_nf = {b: float(np.linalg.norm(v)) for b, v in body_nf_vec.items()}
        body_ff = {b: float(np.linalg.norm(v)) for b, v in body_ff_vec.items()}
        pair_nf = {k: float(np.linalg.norm(v)) for k, v in pair_nf_vec.items()}
        pair_ff = {k: float(np.linalg.norm(v)) for k, v in pair_ff_vec.items()}

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
    import jk_solver_examples.contacts.test_contact_ball as m
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
