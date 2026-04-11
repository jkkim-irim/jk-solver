"""Finger pinch environment — ALLEX Hand on ground plane.

Step 1: Load ALLEX_Hand.usd articulation and render in viewer.

Run:
    python jk_solver_examples/contacts/test_finger_pinch_env.py
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp

_JK_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _JK_ROOT / "assets"
sys.path.insert(0, str(_JK_ROOT))

import newton
import newton.examples
from newton import JointTargetMode
from jk_solver_examples import init as jk_init
from jk_solver_examples.debug_monitor import StatusWindow
from jk_solver_examples.jk_solver import SolverJK


# =============================================================================
# Constraint solref / solimp
# =============================================================================

# Equality constraint (DIP-PIP mimic coupling)
EQ_SOLREF = (0.01, 1.0)
EQ_SOLIMP = (0.99, 1.0, 0.001, 0.5, 2.0)

# Contact — box geom
BOX_MU = 0.5
BOX_SOLREF = (0.01, 1.0)
BOX_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)

# Contact — robot geoms (all ALLEX links)
ROBOT_MU = 0.8
ROBOT_SOLREF = (0.01, 1.0)
ROBOT_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)

# =============================================================================
# Viewer
# =============================================================================
CAM_POS = (0.4, 0.5, 0.15)
CAM_PITCH = -10.0
CAM_YAW = -90.0

# =============================================================================
# Simulation
# =============================================================================
FPS = 200
FRAME_DT = 1.0 / FPS
SIM_SUBSTEPS = 10
SIM_DT = FRAME_DT / SIM_SUBSTEPS
GRAVITY = 9.81

# =============================================================================
# Solver
# =============================================================================
SOLVER_ITERATIONS = 100
SOLVER_LS_ITERATIONS = 50
SOLVER_CONE = "elliptic"
SOLVER_IMPRATIO = 1000.0

# =============================================================================
# Hand transform
# =============================================================================
HAND_POS = (0.0, 0.0, 0.2)             # (x, y, z) meters
HAND_ROT_DEG = (180.0, 0.0, 0.0)       # (roll_x, pitch_y, yaw_z) degrees

# =============================================================================
# Joint PD gains  (kp, kd)
#   - real robot: 20 kHz, sim: ~2 kHz → /10 scaling
#   - wrist: separate constants
# =============================================================================
WRIST_KE = 100.0
WRIST_KD = 10.0
_WRIST_JOINTS = {"R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint"}

HAND_JOINT_GAINS: dict[str, tuple[float, float]] = {
    # Thumb
    "R_Thumb_Yaw_Joint":    (40.0/1.0, 4.0/1.0),
    "R_Thumb_CMC_Joint":    (40.0/20.0, 4.0/20.0),
    "R_Thumb_MCP_Joint":    (20.0/20.0, 2.0/20.0),
    # Index
    "R_Index_ABAD_Joint":   (20.0/1.0, 2.0/1.0),
    "R_Index_MCP_Joint":    (40.0/20.0, 4.0/20.0),
    "R_Index_PIP_Joint":    (20.0/20.0, 2.0/20.0),
    # Middle
    "R_Middle_ABAD_Joint":  (20.0/1.0, 2.0/1.0),
    "R_Middle_MCP_Joint":   (40.0/1.0, 4.0/1.0),
    "R_Middle_PIP_Joint":   (20.0/1.0, 2.0/1.0),
    # Ring
    "R_Ring_ABAD_Joint":    (20.0/1.0, 2.0/1.0),
    "R_Ring_MCP_Joint":     (40.0/1.0, 4.0/1.0),
    "R_Ring_PIP_Joint":     (20.0/1.0, 2.0/1.0),
    # Little
    "R_Little_ABAD_Joint":  (20.0/1.0, 2.0/1.0),
    "R_Little_MCP_Joint":   (40.0/1.0, 4.0/1.0),
    "R_Little_PIP_Joint":   (20.0/1.0, 2.0/1.0),
}

# =============================================================================
# Joint effort limits (N·m) — from MJCF actuator spec
# =============================================================================
HAND_EFFORT_LIMIT: dict[str, float] = {
    "R_Thumb_Yaw_Joint": 2.9, "R_Thumb_CMC_Joint": 4.8, "R_Thumb_MCP_Joint": 4.8,
    "R_Index_ABAD_Joint": 2.1, "R_Index_MCP_Joint": 4.8, "R_Index_PIP_Joint": 3.0,
    "R_Middle_ABAD_Joint": 2.1, "R_Middle_MCP_Joint": 4.8, "R_Middle_PIP_Joint": 3.0,
    "R_Ring_ABAD_Joint": 2.1, "R_Ring_MCP_Joint": 4.8, "R_Ring_PIP_Joint": 3.0,
    "R_Little_ABAD_Joint": 2.1, "R_Little_MCP_Joint": 4.8, "R_Little_PIP_Joint": 3.0,
}

# =============================================================================
# Joint position limits (rad) — from MJCF ctrlrange
# =============================================================================
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    # Wrist
    "R_Wrist_Yaw_Joint":   (-0.610865, 4.799655),      # -35° ~ 275°
    "R_Wrist_Roll_Joint":  (-0.872665, 0.872665),       # -50° ~ 50°
    "R_Wrist_Pitch_Joint": (-1.396263, 1.396263),       # -80° ~ 80°
    # Thumb
    "R_Thumb_Yaw_Joint":   (-2.617994, 0.0),            # -150° ~ 0°
    "R_Thumb_CMC_Joint":   (0.0, 1.570796),             # 0° ~ 90°
    "R_Thumb_MCP_Joint":   (0.0, 1.570796),             # 0° ~ 90°
    # Index
    "R_Index_ABAD_Joint":  (-0.523599, 0.523599),       # -30° ~ 30°
    "R_Index_MCP_Joint":   (-0.174532, 1.570796),       # -10° ~ 90°
    "R_Index_PIP_Joint":   (0.0, 1.745329),             # 0° ~ 100°
    # Middle
    "R_Middle_ABAD_Joint": (-0.523599, 0.523599),       # -30° ~ 30°
    "R_Middle_MCP_Joint":  (-0.174532, 1.570796),       # -10° ~ 90°
    "R_Middle_PIP_Joint":  (0.0, 1.745329),             # 0° ~ 100°
    # Ring
    "R_Ring_ABAD_Joint":   (-0.523599, 0.523599),       # -30° ~ 30°
    "R_Ring_MCP_Joint":    (-0.174532, 1.570796),       # -10° ~ 90°
    "R_Ring_PIP_Joint":    (0.0, 1.745329),             # 0° ~ 100°
    # Little
    "R_Little_ABAD_Joint": (-0.523599, 0.523599),       # -30° ~ 30°
    "R_Little_MCP_Joint":  (-0.174532, 1.570796),       # -10° ~ 90°
    "R_Little_PIP_Joint":  (0.0, 1.745329),             # 0° ~ 100°
    # DIP/IP — mimic followers, lower=0 (no hyperextension)
    "R_Thumb_IP_Joint":    (0.0, 1.369),                # 0° ~ 78.4°
    "R_Index_DIP_Joint":   (0.0, 1.483),                # 0° ~ 85°
    "R_Middle_DIP_Joint":  (0.0, 1.483),                # 0° ~ 85°
    "R_Ring_DIP_Joint":    (0.0, 1.483),                # 0° ~ 85°
    "R_Little_DIP_Joint":  (0.0, 1.483),                # 0° ~ 85°
}

# =============================================================================
# DIP-PIP mimic constraints (quartic polynomial: a0 + a1*q + a2*q² + a3*q³ + a4*q⁴)
# =============================================================================
_DIP_COEF = (-0.003849, 0.4269, 0.06589, 0.136, -0.04621)
_THUMB_COEF = (-0.0015, 0.6651, 0.0186, 0.1224, -0.0696)

MIMIC_CONSTRAINTS: list[tuple[str, str, tuple[float, ...]]] = [
    ("R_Thumb_IP_Joint",    "R_Thumb_MCP_Joint",    _THUMB_COEF),
    ("R_Index_DIP_Joint",   "R_Index_PIP_Joint",    _DIP_COEF),
    ("R_Middle_DIP_Joint",  "R_Middle_PIP_Joint",   _DIP_COEF),
    ("R_Ring_DIP_Joint",    "R_Ring_PIP_Joint",     _DIP_COEF),
    ("R_Little_DIP_Joint",  "R_Little_PIP_Joint",   _DIP_COEF),
]

# =============================================================================
# Target poses (degrees)
# =============================================================================

# Initial pose — 18 active DOFs
INIT_POSE_DEG: dict[str, float] = {
    # Wrist
    "R_Wrist_Yaw_Joint":   0.0,
    "R_Wrist_Roll_Joint":  0.0,
    "R_Wrist_Pitch_Joint": -45.0,
    # Thumb (IP follows MCP via mimic)
    "R_Thumb_Yaw_Joint":   -80.0,
    "R_Thumb_CMC_Joint":   0.0,
    "R_Thumb_MCP_Joint":   0.0,
    # Index (DIP follows PIP via mimic)
    "R_Index_ABAD_Joint":  0.0,
    "R_Index_MCP_Joint":   0.0,
    "R_Index_PIP_Joint":   0.0,
    # Middle
    "R_Middle_ABAD_Joint": 0.0,
    "R_Middle_MCP_Joint":  90.0,
    "R_Middle_PIP_Joint":  90.0,
    # Ring
    "R_Ring_ABAD_Joint":   0.0,
    "R_Ring_MCP_Joint":    90.0,
    "R_Ring_PIP_Joint":    90.0,
    # Little
    "R_Little_ABAD_Joint": 0.0,
    "R_Little_MCP_Joint":  90.0,
    "R_Little_PIP_Joint":  90.0,
}

# Pinch pose — only joints that change (others keep INIT_POSE_DEG)
PINCH_POSE_DEG: dict[str, float] = {
    "R_Thumb_CMC_Joint":   50.0,
    "R_Thumb_MCP_Joint":   20.0,
    "R_Index_MCP_Joint":   30.0,
    "R_Index_PIP_Joint":   40.0,
}
PINCH_DURATION = 0.5  # seconds to reach pinch/open target


# =============================================================================
# Scene builder
# =============================================================================

def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Euler XYZ (degrees) -> warp quaternion (x, y, z, w)."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return wp.quat(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _find_joint_index(builder, name):
    """Find joint index by label suffix (e.g. 'R_Thumb_IP_Joint')."""
    for i, label in enumerate(builder.joint_label):
        if label.endswith(name) or label == name:
            return i
    return -1


def _apply_mjc_joint_overrides(solver):
    """Override joint position limits and torque limits on both mj_model and mjw_model by joint name."""
    import mujoco
    mj = solver.mj_model
    mjw = solver.mjw_model

    # Build MuJoCo joint name → index map
    for i in range(mj.njnt):
        mj_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i)
        for short_name, (lo, hi) in JOINT_LIMITS_RAD.items():
            if mj_name.endswith(short_name):
                mj.jnt_range[i] = [lo, hi]
                mj.jnt_limited[i] = 1
                break
        for short_name, limit in HAND_EFFORT_LIMIT.items():
            if mj_name.endswith(short_name):
                mj.jnt_actfrcrange[i] = [-limit, limit]
                mj.jnt_actfrclimited[i] = 1
                break

    # Sync to mjw_model (GPU arrays)
    jnt_range_np = mjw.jnt_range.numpy()
    jnt_limited_np = mjw.jnt_limited.numpy()
    jnt_actfrcrange_np = mjw.jnt_actfrcrange.numpy()
    jnt_actfrclimited_np = mjw.jnt_actfrclimited.numpy()

    for i in range(mj.njnt):
        jnt_range_np[0, i] = mj.jnt_range[i]
        jnt_limited_np[i] = mj.jnt_limited[i]
        jnt_actfrcrange_np[0, i] = mj.jnt_actfrcrange[i]
        jnt_actfrclimited_np[i] = mj.jnt_actfrclimited[i]

    mjw.jnt_range.assign(jnt_range_np)
    mjw.jnt_limited.assign(jnt_limited_np)
    mjw.jnt_actfrcrange.assign(jnt_actfrcrange_np)
    mjw.jnt_actfrclimited.assign(jnt_actfrclimited_np)


def _apply_contact_solref_solimp(solver, p):
    """Override geom contact solref/solimp and friction (geometric mean)."""
    import mujoco
    mj = solver.mj_model
    mjw = solver.mjw_model

    solref_np = mjw.geom_solref.numpy()
    solimp_np = mjw.geom_solimp.numpy()
    friction_np = mjw.geom_friction.numpy()

    # Geometric mean friction: both box and robot geoms set to sqrt(mu_box * mu_robot)
    # so MuJoCo's max(mu_a, mu_b) = geomean for box<->robot contacts
    mu_geomean = float(np.sqrt(p["box_mu"] * p["robot_mu"]))

    for i in range(mj.ngeom):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is None:
            name = f"geom_{i}"
        if name.startswith("shape_1"):  # box geom
            solref_np[0, i] = p["box_solref"][:2]
            solimp_np[0, i] = p["box_solimp"][:5]
            friction_np[0, i, 0] = mu_geomean
        elif "/ALLEX/" in name:  # robot geoms
            solref_np[0, i] = p["robot_solref"][:2]
            solimp_np[0, i] = p["robot_solimp"][:5]
            friction_np[0, i, 0] = mu_geomean

    mjw.geom_solref.assign(solref_np)
    mjw.geom_solimp.assign(solimp_np)
    mjw.geom_friction.assign(friction_np)


def _build_scene(p):
    """Build Newton model with ALLEX Hand articulation."""
    builder = newton.ModelBuilder(gravity=-p["gravity"])
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane()

    # White box
    box_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=0, kd=0, density=400.0) 
    box_half = 0.02
    box_pos = (0.38, 0.05, box_half)
    body_box = builder.add_link(
        xform=wp.transform(p=box_pos, q=wp.quat_identity()), label="box")
    builder.add_shape_box(body_box, hx=box_half, hy=box_half, hz=box_half, cfg=box_cfg)
    j_box = builder.add_joint_free(body_box)
    builder.add_articulation([j_box])

    asset_path = str(_ASSETS / "ALLEX_Hand.usd")
    builder.add_usd(
        asset_path,
        xform=wp.transform(wp.vec3(*p["hand_pos"]), _euler_deg_to_quat(*p["hand_rot_deg"])),
        enable_self_collisions=True,
        hide_collision_shapes=True,
    )

    # Apply target pose from INIT_POSE_DEG (deg → rad)
    for name, deg in INIT_POSE_DEG.items():
        ji = _find_joint_index(builder, name)
        if ji >= 0:
            qi = builder.joint_q_start[ji]
            builder.joint_q[qi] = np.radians(deg)

    # Mimic follower joints — skip joint drive for these
    _mimic_followers = set()
    for follower_name, _, _ in MIMIC_CONSTRAINTS:
        ji = _find_joint_index(builder, follower_name)
        if ji >= 0:
            _mimic_followers.add(ji)

    # Joint drive: hold initial pose (DOF-indexed), skip mimic followers
    for ji in range(builder.joint_count):
        if ji in _mimic_followers:
            continue
        qd_start = builder.joint_qd_start[ji]
        q_start = builder.joint_q_start[ji]
        if ji + 1 < builder.joint_count:
            n_dof = builder.joint_qd_start[ji + 1] - qd_start
        else:
            n_dof = builder.joint_dof_count - qd_start
        # Per-joint gains: wrist → WRIST_KE/KD, finger → HAND_JOINT_GAINS, else default
        label = builder.joint_label[ji]
        is_wrist = any(label.endswith(w) for w in _WRIST_JOINTS)
        hand_match = next((v for k, v in HAND_JOINT_GAINS.items() if label.endswith(k)), None)
        if is_wrist:
            ke, kd = p["wrist_ke"], p["wrist_kd"]
        elif hand_match:
            ke, kd = hand_match
        else:
            ke, kd = 10.0, 1.0  # default for box free joint etc.
        for d in range(n_dof):
            builder.joint_target_ke[qd_start + d] = ke
            builder.joint_target_kd[qd_start + d] = kd
            builder.joint_target_pos[qd_start + d] = builder.joint_q[q_start + d]
            builder.joint_target_mode[qd_start + d] = int(JointTargetMode.POSITION)

    # Disable USD built-in linear mimic constraints (replaced by quartic equality below)
    for i in range(len(builder.constraint_mimic_enabled)):
        builder.constraint_mimic_enabled[i] = False

    # DIP-PIP equality constraints (quartic polynomial coupling)
    for follower_name, leader_name, coef in MIMIC_CONSTRAINTS:
        j_follower = _find_joint_index(builder, follower_name)
        j_leader = _find_joint_index(builder, leader_name)
        if j_follower < 0 or j_leader < 0:
            print(f"[WARN] mimic constraint skipped: {follower_name} -> {leader_name} "
                  f"(follower={j_follower}, leader={j_leader})")
            continue
        builder.add_equality_constraint_joint(
            joint1=j_follower,
            joint2=j_leader,
            polycoef=list(coef),
            label=f"mimic_{follower_name}",
        )

    model = builder.finalize()
    solver = SolverJK(
        model,
        iterations=p["solver_iterations"],
        ls_iterations=p["solver_ls_iterations"],
        cone=p["solver_cone"],
        impratio=p["solver_impratio"],
        njmax=500, nconmax=500,
    )

    # Override equality constraint solref/solimp for rigid DIP-PIP coupling
    eq_solref = solver.mjw_model.eq_solref.numpy()
    eq_solimp = solver.mjw_model.eq_solimp.numpy()
    for i in range(eq_solref.shape[1]):
        eq_solref[0, i] = p["eq_solref"][:2]
        eq_solimp[0, i] = p["eq_solimp"][:5]
    solver.mjw_model.eq_solref.assign(eq_solref)
    solver.mjw_model.eq_solimp.assign(eq_solimp)

    # Override joint position limits & torque limits on MuJoCo model (by joint name)
    _apply_mjc_joint_overrides(solver)

    # Override contact solref/solimp for box and robot geoms
    _apply_contact_solref_solimp(solver, p)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    return model, solver, state_0, state_1, control, contacts


def _make_params():
    return dict(
        fps=FPS, sim_substeps=SIM_SUBSTEPS, gravity=GRAVITY,
        hand_pos=HAND_POS, hand_rot_deg=HAND_ROT_DEG,
        solver_iterations=SOLVER_ITERATIONS, solver_ls_iterations=SOLVER_LS_ITERATIONS,
        solver_cone=SOLVER_CONE, solver_impratio=SOLVER_IMPRATIO,
        wrist_ke=WRIST_KE, wrist_kd=WRIST_KD,
        eq_solref=list(EQ_SOLREF), eq_solimp=list(EQ_SOLIMP),
        box_mu=BOX_MU, box_solref=list(BOX_SOLREF), box_solimp=list(BOX_SOLIMP),
        robot_mu=ROBOT_MU, robot_solref=list(ROBOT_SOLREF), robot_solimp=list(ROBOT_SOLIMP),
    )


# =============================================================================
# Example
# =============================================================================

class Example:
    """ALLEX Hand pinch environment — step 1: load and render."""

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
        self._traj = None  # trajectory interpolation state
        self._max_pen = {"box<->index": 0.0, "box<->thumb": 0.0}
        self._cache_geom_groups()
        self._status_window = StatusWindow(geom_labels=[])

    def _cache_geom_groups(self):
        """Cache geom index sets for box, index finger, and thumb."""
        import mujoco
        mj = self.solver.mj_model
        self._box_geoms = set()
        self._index_geoms = set()
        self._thumb_geoms = set()
        for i in range(mj.ngeom):
            bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, mj.geom_bodyid[i]) or ""
            if "box" in bname:
                self._box_geoms.add(i)
            elif "Index" in bname:
                self._index_geoms.add(i)
            elif "Thumb" in bname:
                self._thumb_geoms.add(i)

    def _get_contact_penetration(self):
        """Return current penetration depth for box<->index and box<->thumb pairs."""
        d = self.solver.mjw_data
        nacon = int(d.nacon.numpy()[0])
        pen_index, pen_thumb = 0.0, 0.0
        if nacon > 0:
            contact_geom = d.contact.geom.numpy()[:nacon]
            contact_dist = d.contact.dist.numpy()[:nacon]
            for c in range(nacon):
                g0, g1 = int(contact_geom[c, 0]), int(contact_geom[c, 1])
                pen = -float(contact_dist[c])  # positive = penetration
                is_box = g0 in self._box_geoms or g1 in self._box_geoms
                if not is_box:
                    continue
                other = g1 if g0 in self._box_geoms else g0
                if other in self._index_geoms:
                    pen_index = max(pen_index, pen)
                elif other in self._thumb_geoms:
                    pen_thumb = max(pen_thumb, pen)
        return pen_index, pen_thumb

    def _cache_joint_info(self):
        """Cache joint labels, q_start (for qpos), and qd_start (for joint_target_pos)."""
        self._joint_labels = [self.model.joint_label[i] for i in range(self.model.joint_count)]
        self._joint_q_starts = self.model.joint_q_start.numpy()    # for qpos indexing
        self._joint_qd_starts = self.model.joint_qd_start.numpy()  # for joint_target_pos indexing
        # Cache Index PIP joint index for torque readout
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
        self._pinch_active = False
        self._lift_active = False
        self._traj = None
        self._max_pen = {"box<->index": 0.0, "box<->thumb": 0.0}
        self.sim_time = 0.0
        self.next_print_time = 0.0
        self._status_window.clear_joint_data()
        print("[Reset] scene rebuilt")

    def gui(self, imgui):
        if not getattr(self.viewer, "_irim_param_tune_open", False):
            return
        p = self._p

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
        imgui.text("(finger gains: per-joint from MJCF)")

        imgui.separator()
        imgui.text("=== Physics ===")
        _, p["gravity"] = imgui.input_float("GRAVITY", p["gravity"], format="%.3f")

        imgui.separator()
        imgui.text("Reset to apply all changes")

    def _apply_pinch_target(self):
        """Start trajectory interpolation toward pinch or open pose."""
        current_pos = self.control.joint_target_pos.numpy().copy()
        goal_pos = current_pos.copy()
        for name in PINCH_POSE_DEG:
            deg = PINCH_POSE_DEG[name] if self._pinch_active else INIT_POSE_DEG.get(name, 0.0)
            for i, lbl in enumerate(self._joint_labels):
                if lbl.endswith(name):
                    qd = self._joint_qd_starts[i]
                    goal_pos[qd] = np.radians(deg)
                    break
        self._traj = {
            "start": current_pos,
            "goal": goal_pos,
            "t0": self.sim_time,
            "duration": PINCH_DURATION,
        }

    def _apply_lift_target(self):
        """Start trajectory interpolation: wrist pitch to 0° (lift) or back to INIT."""
        current_pos = self.control.joint_target_pos.numpy().copy()
        goal_pos = current_pos.copy()
        deg = 0.0 if self._lift_active else INIT_POSE_DEG.get("R_Wrist_Pitch_Joint", 0.0)
        for i, lbl in enumerate(self._joint_labels):
            if lbl.endswith("R_Wrist_Pitch_Joint"):
                qd = self._joint_qd_starts[i]
                goal_pos[qd] = np.radians(deg)
                break
        self._traj = {
            "start": current_pos,
            "goal": goal_pos,
            "t0": self.sim_time,
            "duration": PINCH_DURATION,
        }

    def _update_traj(self):
        """Linearly interpolate joint targets if trajectory is active."""
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
        for sub in range(self._active["sim_substeps"]):
            self.sim_time += sim_dt
            self._update_traj()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, sim_dt)
            self._post_substep(sub)

    def _post_substep(self, substep):
        d = self.solver.mjw_data
        niter = int(d.solver_niter.numpy()[0])
        nacon = int(d.nacon.numpy()[0])
        nefc = int(d.nefc.numpy()[0])
        max_iter = self._active["solver_iterations"]
        hit = niter >= max_iter
        # (sub)time: sim_time + substep fraction
        sub_dt = 1.0 / (self._active["fps"] * self._active["sim_substeps"])
        subtime = self.sim_time + substep * sub_dt
        self._status_window.push_conv(
            f"({substep})t={subtime:.5f}  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}",
            hit_limit=hit, no_contact=(nacon == 0))
        self._status_window.update_conv_summary(
            f"[current]  niter={niter}/{max_iter}  nacon={nacon}  nefc={nefc}")

        # Contact penetration (update every substep)
        pen_idx, pen_thm = self._get_contact_penetration()
        self._max_pen["box<->index"] = max(self._max_pen["box<->index"], pen_idx)
        self._max_pen["box<->thumb"] = max(self._max_pen["box<->thumb"], pen_thm)
        self._status_window.update({"contact": (
            f"── Contact Penetration (t={self.sim_time:.3f}s) ──\n"
            f"  {'Pair':<20s} {'Pen(mm)':>10s} {'Max_Pen(mm)':>12s}\n"
            f"  {'box<->index':<20s} {pen_idx*1000:>10.4f} {self._max_pen['box<->index']*1000:>12.4f}\n"
            f"  {'box<->thumb':<20s} {pen_thm*1000:>10.4f} {self._max_pen['box<->thumb']*1000:>12.4f}"
        )})

        # Index DIP constraint data (efc type=0 are equality constraints, Index DIP is 2nd)
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
                self._status_window.push_joint_data(subtime, dip_force, dip_pos, pip_torque)
                self._status_window.push_joint_conv(
                    f"({substep})t={subtime:.5f}  efc_f={dip_force:>9.5f}  "
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
        if self.sim_time >= self.next_print_time:
            self._update_status()
            self.next_print_time += self.print_interval

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = jk_init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)