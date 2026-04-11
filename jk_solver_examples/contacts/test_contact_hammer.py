"""Contact force test (mesh version) — Hammer + fingertip pads on ground plane.

Scenario: 1 Hammer (USD, coacd decomposition) + 2 fingertip pads (STL, convex hull).
Right pad = Finger_Distal_Pad.stl, Left pad = R_Thumb_Distal_Pad.stl.
Pads push into hammer with constant y-force.

Run:
    python jk_solver_examples/contacts/test_contact_hammer.py
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

_JK_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _JK_ROOT / "assets"
sys.path.insert(0, str(_JK_ROOT))

import newton
import newton.examples
from jk_solver_examples import init as jk_init
from jk_solver_examples.debug_monitor import StatusWindow
from jk_solver_examples.jk_kernels import update_max_penetration_kernel
from jk_solver_examples.jk_solver import SolverJK
from newton import Contacts

# ──── Camera ────
CAM_POS = (0.2, 0.0, 0.001)
CAM_PITCH = 0.0
CAM_YAW = -180.0

# ──── Timing ────
FPS = 60
FRAME_DT = 1.0 / FPS
SIM_SUBSTEPS = 2
SIM_DT = FRAME_DT / SIM_SUBSTEPS

# ──── Physics ────
GRAVITY = 9.81
HAMMER_DENSITY = 4554.2     # kg/m^3 → vol≈0.000121m^3 → 0.55kg (550g)
FINGER_DENSITY = 3291.3     # kg/m^3 → vol≈0.0000015m^3 → 0.005kg (5g)
BOX_MU = 0.3
SPHERE_MU = 0.5
GROUND_MU = 0.1
SPHERE_GAP = 0.018  # pad center to hammer center distance [m]
SPHERE_FORCE_Y = 0.01  # y-force applied to spheres [N]

# ──── Object orientation (Euler XYZ degrees) ────
HAMMER_ROT_DEG = (0.0, 0.0, -90.0)  # (roll_x, pitch_y, yaw_z)
FINGER_ROT_DEG = (270.0, 0.0, -90.0)   # (roll_x, pitch_y, yaw_z) green pad
THUMB_ROT_DEG = (0.0, 180.0, 0.0)    # (roll_x, pitch_y, yaw_z) yellow pad

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

def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Euler XYZ (degrees) → warp quaternion. Roll=X, Pitch=Y, Yaw=Z."""
    import math
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    # warp quat = (x, y, z, w)
    return wp.quat(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _load_stl_mesh(stl_path: str) -> newton.Mesh:
    """Load STL file via trimesh, return Newton Mesh. Assumes meters, centered at origin."""
    tm = trimesh.load(stl_path, force="mesh")
    verts = np.array(tm.vertices, dtype=np.float32)
    indices = np.array(tm.faces.flatten(), dtype=np.int32)
    return newton.Mesh(verts, indices)


def _rotated_half_height(mesh: newton.Mesh, rot_deg: tuple) -> float:
    """Return half the z-extent of a mesh after applying Euler XYZ rotation (degrees)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix()
    verts = (R @ np.array(mesh.vertices).T).T
    return float(verts[:, 2].max() - verts[:, 2].min()) / 2


def _load_usd_mesh(usd_path: str, prim_path: str = "/World/Hammer_Cylinder") -> newton.Mesh:
    """Load USD mesh, triangulate quads, return Newton Mesh."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    verts = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
    face_indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    face_counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    # Center at origin
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    verts -= center
    # Triangulate (quad → 2 triangles)
    tris = []
    offset = 0
    for c in face_counts:
        for i in range(1, c - 1):
            tris.extend([face_indices[offset], face_indices[offset + i], face_indices[offset + i + 1]])
        offset += c
    return newton.Mesh(verts, np.array(tris, dtype=np.int32))


def _add_free_mesh_body(builder, mesh, pos, rot, label, cfg,
                        free_linear_axes=None, free_angular_axes=None):
    """Add a free-floating mesh body. Returns (body_idx, shape_idx).

    Joint axes are always world-aligned (parent_xform has no rotation).
    Body orientation is applied via child_xform so X and rotation stay constrained in world frame.
    free_linear_axes: list of newton.Axis to unlock translation (default: [Y, Z]).
    free_angular_axes: list of newton.Axis to unlock rotation (e.g. [newton.Axis.Y] for pitch).
    """
    if free_linear_axes is None:
        free_linear_axes = [newton.Axis.Y, newton.Axis.Z]
    xform = wp.transform(p=pos, q=rot)
    body = builder.add_link(xform=xform, label=label)
    shape = builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
    lin_axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(a) for a in free_linear_axes]
    ang_axes = []
    if free_angular_axes:
        ang_axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(a) for a in free_angular_axes]
    j = builder.add_joint_d6(
        parent=-1, child=body,
        linear_axes=lin_axes,
        angular_axes=ang_axes,
        parent_xform=wp.transform(p=pos, q=wp.quat_identity()),
        child_xform=wp.transform(p=(0.0, 0.0, 0.0), q=rot))
    builder.add_articulation([j])
    return body, shape


def _build_scene(p):
    """Build Newton model with mesh objects + SolverJK + state."""
    builder = newton.ModelBuilder(gravity=-p["gravity"])
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=p["ground_mu"], ke=0, kd=0, density=0.0)
    builder.add_ground_plane(cfg=ground_cfg)
    hammer_cfg = newton.ModelBuilder.ShapeConfig(mu=p["box_mu"], ke=0, kd=0, density=p["box_density"])
    pad_cfg = newton.ModelBuilder.ShapeConfig(mu=p["sphere_mu"], ke=0, kd=0, density=p["sphere_density"])

    # Hammer (USD → coacd decomposition)
    hammer_mesh = _load_usd_mesh(str(_ASSETS / "Hammer.usd"))
    hammer_extent = np.array(hammer_mesh.vertices).max(axis=0) - np.array(hammer_mesh.vertices).min(axis=0)
    hammer_h = float(hammer_extent[2]) / 2 + 0.01
    body_box, hammer_shape = _add_free_mesh_body(
        builder, hammer_mesh, pos=(-0.05, 0.0, hammer_h),
        rot=_euler_deg_to_quat(*HAMMER_ROT_DEG), label="hammer", cfg=hammer_cfg,
        free_angular_axes=[newton.Axis.X, newton.Axis.Y, newton.Axis.Z])

    # Right pad: Finger_Distal_Pad.stl (convex hull)
    finger_mesh = _load_stl_mesh(str(_ASSETS / "Finger_Distal_Pad.stl"))
    finger_z = _rotated_half_height(finger_mesh, FINGER_ROT_DEG)
    body_sphere_l, finger_shape = _add_free_mesh_body(
        builder, finger_mesh, pos=(0.0, -p["sphere_gap"], finger_z - 0.0005),
        rot=_euler_deg_to_quat(*FINGER_ROT_DEG), label="finger_pad", cfg=pad_cfg,
        free_linear_axes=[newton.Axis.Y])

    # Left pad: R_Thumb_Distal_Pad.stl (convex hull)
    thumb_mesh = _load_stl_mesh(str(_ASSETS / "R_Thumb_Distal_Pad.stl"))
    thumb_z = _rotated_half_height(thumb_mesh, THUMB_ROT_DEG)
    body_sphere_r, thumb_shape = _add_free_mesh_body(
        builder, thumb_mesh, pos=(0.0, p["sphere_gap"], thumb_z),
        rot=_euler_deg_to_quat(*THUMB_ROT_DEG), label="thumb_pad", cfg=pad_cfg)

    # Hammer: coacd decomposition, Pads: convex hull
    builder.approximate_meshes(method="coacd", shape_indices=[hammer_shape], threshold=0.1)
    builder.approximate_meshes(method="convex_hull", shape_indices=[finger_shape, thumb_shape])

    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    model = builder.finalize()
    solver = SolverJK(
        model,
        iterations=p["solver_iterations"], ls_iterations=p["solver_ls_iterations"],
        cone=p["solver_cone"], impratio=p["solver_impratio"],
        njmax=500, nconmax=500)
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
        gravity=GRAVITY,
        box_mass=HAMMER_DENSITY * 0.000121,   # density × hammer volume
        sphere_mass=FINGER_DENSITY * 0.0000015,  # density × finger volume
        box_density=HAMMER_DENSITY, sphere_density=FINGER_DENSITY,
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
        self._status_window = StatusWindow(geom_labels=["ground", "hammer", "finger", "thumb"])

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
        _, p["box_mass"] = imgui.input_float("HAMMER_MASS", p["box_mass"], format="%.4f")
        p["box_density"] = p["box_mass"] / 0.000121 if p["box_mass"] > 0 else 0.0
        _, p["box_mu"] = imgui.input_float("HAMMER_MU", p["box_mu"], format="%.3f")
        _, p["sphere_mass"] = imgui.input_float("FINGER_MASS", p["sphere_mass"], format="%.4f")
        p["sphere_density"] = p["sphere_mass"] / 0.0000015 if p["sphere_mass"] > 0 else 0.0
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
        imgui.text("=== Viewer Controls ===")
        picking = self.viewer.picking
        ps_np = picking.pick_state.numpy()
        changed = False
        _, v = imgui.input_float("pick_stiffness", ps_np[0]["pick_stiffness"], format="%.1f")
        if v != ps_np[0]["pick_stiffness"]:
            ps_np[0]["pick_stiffness"] = v
            picking.pick_stiffness = v
            changed = True
        _, v = imgui.input_float("pick_damping", ps_np[0]["pick_damping"], format="%.1f")
        if v != ps_np[0]["pick_damping"]:
            ps_np[0]["pick_damping"] = v
            picking.pick_damping = v
            changed = True
        if changed:
            picking.pick_state.assign(ps_np)

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
        body_labels = {-1: "ground", self.body_box: "hammer", self.body_sphere_l: "finger", self.body_sphere_r: "thumb"}

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
        for label, idx in [("hammer", self.body_box), ("finger", self.body_sphere_l), ("thumb", self.body_sphere_r)]:
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
            if label not in plot_cache:
                plot_cache[label] = (list(sr), list(si), sm)
                lines.append(f"  {label:<8s} {sr[0]:>8.5f} {sr[1]:>8.5f} {si[0]:>6.3f} {si[1]:>6.3f} {si[2]:>8.4f} {si[3]:>6.2f} {si[4]:>5.1f} {sm:>5.2f}")
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
    import jk_solver_examples.contacts.test_contact_hammer as m
    m.GRAVITY = p["gravity"]
    m.HAMMER_DENSITY = p["box_density"]
    m.FINGER_DENSITY = p["sphere_density"]
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
