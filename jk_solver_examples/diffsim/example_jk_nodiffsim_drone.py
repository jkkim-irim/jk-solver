# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JK Example: Nodiffsim Drone (contrast to diffsim_drone)
#
# Same scene, targets, and MPC horizon as ``example_diffsim_drone``, but no
# Tape, no autodiff, no SGD. Optimizes control waypoints with derivative-free
# random search (Gaussian around best-so-far, σ decay), with the same CLI
# coefficients as ``example_jk_nodiffsim_cloth`` (``--sigma0``, etc.).
#
# Waypoints use a modest trim + floor. Each frame draws ``--num-samples`` random
# candidates (argmin cost); on each **target change**, waypoints reset to hover and
# σ resets so the previous target's plan does not fight the new goal.
#
# Default ``--gravity 0``: no gravity for this nodiff demo (easier to tune search).
# Pass ``--gravity -9.81`` for Earth-like gravity.
#
# JK: cd jk-solver && uv run python jk_solver_examples/diffsim/example_jk_nodiffsim_drone.py
#
###########################################################################
import os
import secrets
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
_jk_solver_root = Path(__file__).resolve().parents[2]
for _p in (_repo_root, _jk_solver_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jk_solver_examples.jk_init import init as jk_examples_init

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder, sdf_mesh, sdf_plane, sdf_sphere

DEFAULT_DRONE_PATH = newton.examples.get_asset("crazyflie.usd")


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def replicate_states(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    bodies_per_world: int,
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    world_offset = tid * bodies_per_world
    for i in range(bodies_per_world):
        body_q_out[world_offset + i] = body_q_in[i]
        body_qd_out[world_offset + i] = body_qd_in[i]


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    targets: wp.array(dtype=wp.vec3),
    prop_control: wp.array(dtype=float),
    step: int,
    horizon_length: int,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    world_id = wp.tid()
    tf = body_q[world_id]
    target = targets[0]

    pos_drone = wp.transform_get_translation(tf)
    pos_cost = wp.length_sq(pos_drone - target)
    altitude_cost = wp.max(pos_drone[2] - 0.75, 0.0) + wp.max(0.25 - pos_drone[2], 0.0)
    upvector = wp.vec3(0.0, 0.0, 1.0)
    drone_up = wp.transform_vector(tf, upvector)
    upright_cost = 1.0 - wp.dot(drone_up, upvector)

    vel_drone = body_qd[world_id]

    vel_cost = wp.length_sq(vel_drone)

    control = wp.vec4(
        prop_control[world_id * 4 + 0],
        prop_control[world_id * 4 + 1],
        prop_control[world_id * 4 + 2],
        prop_control[world_id * 4 + 3],
    )
    control_cost = wp.dot(control, control)

    discount = 0.8 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0

    pos_weight = 1000.0
    altitude_weight = 100.0
    control_weight = 0.05
    vel_weight = 0.1
    upright_weight = 10.0
    total_weight = pos_weight + altitude_weight + control_weight + vel_weight + upright_weight

    wp.atomic_add(
        cost,
        world_id,
        (
            pos_cost * pos_weight
            + altitude_cost * altitude_weight
            + control_cost * control_weight
            + vel_cost * vel_weight
            + upright_cost * upright_weight
        )
        * (weighting / total_weight)
        * discount,
    )


@wp.kernel
def collision_cost(
    body_q: wp.array(dtype=wp.transform),
    obstacle_ids: wp.array(dtype=int, ndim=2),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    margin: float,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    world_id, obs_id = wp.tid()
    shape_index = obstacle_ids[world_id, obs_id]

    px = wp.transform_get_translation(body_q[world_id])

    X_bs = shape_X_bs[shape_index]

    x_local = wp.transform_point(wp.transform_inverse(X_bs), px)

    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]

    d = 1e6

    if geo_type == newton.GeoType.SPHERE:
        d = sdf_sphere(x_local, geo_scale[0])
    elif geo_type == newton.GeoType.BOX:
        d = sdf_box(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
    elif geo_type == newton.GeoType.CAPSULE:
        d = sdf_capsule(x_local, geo_scale[0], geo_scale[1], int(newton.Axis.Z))
    elif geo_type == newton.GeoType.CYLINDER:
        d = sdf_cylinder(x_local, geo_scale[0], geo_scale[1], int(newton.Axis.Z))
    elif geo_type == newton.GeoType.CONE:
        d = sdf_cone(x_local, geo_scale[0], geo_scale[1], int(newton.Axis.Z))
    elif geo_type == newton.GeoType.MESH:
        mesh = shape_source_ptr[shape_index]
        min_scale = wp.min(geo_scale)
        max_dist = margin / min_scale
        d = sdf_mesh(mesh, wp.cw_div(x_local, geo_scale), max_dist)
        d *= min_scale
    elif geo_type == newton.GeoType.PLANE:
        d = sdf_plane(x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5)

    d = wp.max(d, 0.0)
    if d < margin:
        c = margin - d
        wp.atomic_add(cost, world_id, weighting * c)


@wp.kernel
def enforce_control_limits(
    control_limits: wp.array(dtype=float, ndim=2),
    control_points: wp.array(dtype=float, ndim=3),
):
    world_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[world_id, t_id, control_id] = wp.clamp(control_points[world_id, t_id, control_id], lo, hi)


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dofs: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    torques: wp.array(dtype=float),
):
    world_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    control_left = control_points[world_id, t_id, control_id]
    control_right = control_points[world_id, t_id + 1, control_id]
    torque_id = world_id * torque_dim + control_dofs[control_id]
    action = control_left * (1.0 - frac) + control_right * frac
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    thrust_scale: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control * thrust_scale
    torque = dir * prop.max_torque * control * prop.turning_direction * thrust_scale
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(force, torque))


def define_propeller(
    drone: int,
    pos: wp.vec3,
    fps: float,
    thrust: float = 0.109919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
):
    air_density = 1.225

    rps = max_rpm / fps
    max_speed = rps * wp.TAU
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 0.0, 1.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2

    return prop


class Drone:
    def __init__(
        self,
        name: str,
        fps: float,
        trajectory_shape: tuple[int, int],
        variation_count: int = 1,
        size: float = 1.0,
        requires_grad: bool = False,
        state_count: int | None = None,
        gravity: float = -9.81,
    ) -> None:
        self.variation_count = variation_count
        self.requires_grad = requires_grad

        self.sim_tick = 0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
        builder.rigid_gap = 0.05

        props = []
        colliders = []
        crossbar_length = size
        crossbar_height = size * 0.05
        crossbar_width = size * 0.05
        carbon_fiber_density = 1750.0
        for i in range(variation_count):
            body = builder.add_body(label=f"{name}_{i}")

            builder.add_shape_box(
                body,
                hx=crossbar_width,
                hy=crossbar_length,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )
            builder.add_shape_box(
                body,
                hx=crossbar_length,
                hy=crossbar_width,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )

            props.extend(
                (
                    define_propeller(
                        body,
                        wp.vec3(0.0, crossbar_length, 0.0),
                        fps,
                        turning_direction=-1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, -crossbar_length, 0.0),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(-crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=-1.0,
                    ),
                ),
            )

            colliders.append(
                (
                    builder.add_shape_capsule(
                        -1,
                        xform=wp.transform(wp.vec3(0.5, 0.5, 2.0), wp.quat_identity()),
                        radius=0.15,
                        half_height=2.0,
                        cfg=newton.ModelBuilder.ShapeConfig(collision_group=i),
                    ),
                ),
            )
        self.props = wp.array(props, dtype=Propeller)
        self.colliders = wp.array(colliders, dtype=int)

        self.model = builder.finalize(requires_grad=requires_grad)

        if requires_grad:
            self.states = tuple(self.model.state() for _ in range(state_count + 1))
            self.controls = tuple(self.model.control() for _ in range(state_count))
        else:
            self.states = [self.model.state(), self.model.state()]
            self.controls = (self.model.control(),)

        for control in self.controls:
            control.prop_controls = wp.zeros(len(self.props), dtype=float, requires_grad=requires_grad)

        self.trajectories = wp.zeros(
            (variation_count, trajectory_shape[0], trajectory_shape[1]),
            dtype=float,
            requires_grad=requires_grad,
        )

        self.body_count = len(builder.body_q)
        self.collider_count = self.colliders.shape[1]
        self.collision_radius = crossbar_length

    @property
    def state(self) -> newton.State:
        return self.states[self.sim_tick if self.requires_grad else 0]

    @property
    def next_state(self) -> newton.State:
        return self.states[self.sim_tick + 1 if self.requires_grad else 1]

    @property
    def control(self) -> newton.Control:
        return self.controls[min(len(self.controls) - 1, self.sim_tick) if self.requires_grad else 0]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 360
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = args.verbose
        self.quiet = bool(getattr(args, "quiet", False))
        self.drone_path = args.drone_path

        _seed = args.seed if args.seed is not None else secrets.randbits(32)
        self.rng = np.random.default_rng(_seed)
        self._sigma0 = float(args.sigma0)
        self.sigma = self._sigma0
        self.sigma_decay = float(args.sigma_decay)
        self.sigma_min = float(args.sigma_min)
        self.qd_max = float(args.qd_max)
        self.num_samples = int(args.num_samples)
        self._hover_trim_val = float(args.hover_trim)
        self._control_floor_arg = float(args.control_floor)
        self._thrust_scale = float(args.thrust_scale)
        self._gravity = float(args.gravity)
        self._control_gain = float(args.control_gain)

        self.viewer = viewer

        self.targets = (
            wp.vec3(1.0, 0.0, 0.5),
            wp.vec3(0.0, 1.0, 0.5),
        )

        self.target_idx = -1
        self.current_target = wp.array([self.targets[self.target_idx + 1]], dtype=wp.vec3)

        self.control_point_step = 10
        self.control_point_count = 3
        self.control_point_data_count = self.control_point_count + 1
        self.control_dofs = wp.array((0, 1, 2, 3), dtype=int)
        self.control_dim = len(self.control_dofs)
        self.control_gains = wp.array((self._control_gain,) * self.control_dim, dtype=float)
        self.control_limits = wp.array(((0.1, 1.0),) * self.control_dim, dtype=float)

        drone_size = 0.2

        self.drone = Drone(
            "drone",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            size=drone_size,
            gravity=self._gravity,
        )

        self.rollout_step_count = self.control_point_step * self.control_point_count
        state_count = self.rollout_step_count * self.sim_substeps

        self.eval_drone = Drone(
            "eval",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            variation_count=1,
            size=drone_size,
            requires_grad=True,
            state_count=state_count,
            gravity=self._gravity,
        )

        self.rollout_costs = wp.zeros(1, dtype=wp.float32, requires_grad=False)
        self.cost_history = []

        self.solver_eval = newton.solvers.SolverSemiImplicit(self.eval_drone.model)
        self.solver_drone = newton.solvers.SolverSemiImplicit(self.drone.model)

        self.viewer.set_model(self.drone.model)

        self._lims = self.control_limits.numpy()
        lo0, hi0 = float(self._lims[0, 0]), float(self._lims[0, 1])
        trim = min(hi0, max(lo0, self._hover_trim_val))
        floor = min(trim, max(lo0, self._control_floor_arg))
        self._control_floor = floor
        self._hover_trim = np.full(
            (1, self.control_point_data_count, self.control_dim),
            trim,
            dtype=np.float32,
        )
        self._best_traj = self._hover_trim.copy()
        self.drone.trajectories.assign(self._best_traj)
        self.eval_drone.trajectories.assign(self._best_traj)

        self._cost0 = self.eval_trajectory_cost(self._best_traj)
        self._best_loss = float(self._cost0)

    def update_drone(self, drone: Drone, solver) -> None:
        drone.state.clear_forces()

        wp.launch(
            interpolate_control_linear,
            dim=(
                drone.variation_count,
                self.control_dim,
            ),
            inputs=(
                drone.trajectories,
                self.control_dofs,
                self.control_gains,
                drone.sim_tick / (self.sim_substeps * self.control_point_step),
                self.control_dim,
            ),
            outputs=(drone.control.prop_controls,),
        )

        wp.launch(
            compute_prop_wrenches,
            dim=len(drone.props),
            inputs=(
                drone.props,
                drone.control.prop_controls,
                drone.state.body_q,
                drone.model.body_com,
                self._thrust_scale,
            ),
            outputs=(drone.state.body_f,),
        )

        solver.step(
            drone.state,
            drone.next_state,
            None,
            None,
            self.sim_dt,
        )

        drone.sim_tick += 1

    def eval_trajectory_cost(self, traj_np: np.ndarray) -> float:
        self.eval_drone.sim_tick = 0
        self.rollout_costs.zero_()

        self.eval_drone.trajectories.assign(traj_np)

        wp.launch(
            replicate_states,
            dim=1,
            inputs=(
                self.drone.state.body_q,
                self.drone.state.body_qd,
                self.drone.body_count,
            ),
            outputs=(
                self.eval_drone.state.body_q,
                self.eval_drone.state.body_qd,
            ),
        )

        for i in range(self.rollout_step_count):
            for _ in range(self.sim_substeps):
                self.update_drone(self.eval_drone, self.solver_eval)

            wp.launch(
                drone_cost,
                dim=1,
                inputs=(
                    self.eval_drone.state.body_q,
                    self.eval_drone.state.body_qd,
                    self.current_target,
                    self.eval_drone.control.prop_controls,
                    i,
                    self.rollout_step_count,
                    1e3,
                ),
                outputs=(self.rollout_costs,),
            )
            wp.launch(
                collision_cost,
                dim=(
                    1,
                    self.eval_drone.collider_count,
                ),
                inputs=(
                    self.eval_drone.state.body_q,
                    self.eval_drone.colliders,
                    self.eval_drone.model.shape_transform,
                    self.eval_drone.model.shape_type,
                    self.eval_drone.model.shape_scale,
                    self.eval_drone.model.shape_source_ptr,
                    self.eval_drone.collision_radius,
                    1e4,
                ),
                outputs=(self.rollout_costs,),
            )

        return float(self.rollout_costs.numpy()[0])

    def _clip_traj_to_limits(self, traj: np.ndarray) -> np.ndarray:
        lo = self._lims[:, 0]
        hi = self._lims[:, 1]
        eff_lo = np.maximum(lo, self._control_floor)
        out = np.clip(traj, eff_lo[np.newaxis, :], hi[np.newaxis, :])
        return out

    def step(self):
        if self.frame % int(self.sim_steps / len(self.targets)) == 0:
            if self.verbose:
                print(f"Choosing new flight target: {self.target_idx + 1}")

            self.target_idx += 1
            self.target_idx %= len(self.targets)
            self.current_target.assign([self.targets[self.target_idx]])
            # Old waypoints were for the previous target; restarting from hover avoids
            # a huge cost / unstable rollout right after a switch.
            self._best_traj = self._hover_trim.copy()
            self._best_loss = float(self.eval_trajectory_cost(self._best_traj))
            self.sigma = self._sigma0

        best_loss = self._best_loss
        best_traj = self._best_traj.copy()
        for _ in range(self.num_samples):
            flat = self._best_traj.reshape(-1)
            delta = self.rng.normal(0.0, self.sigma, size=flat.shape)
            delta = np.clip(delta, -self.qd_max, self.qd_max)
            cand = (flat + delta).reshape(self._best_traj.shape)
            cand = self._clip_traj_to_limits(cand)
            trial_loss = self.eval_trajectory_cost(cand)
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_traj = cand.copy()

        self._best_loss = best_loss
        self._best_traj = best_traj

        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        self.drone.trajectories.assign(self._best_traj)

        self.drone.sim_tick = 0
        for _ in range(self.sim_substeps):
            self.update_drone(self.drone, self.solver_drone)
            (self.drone.states[0], self.drone.states[1]) = (self.drone.states[1], self.drone.states[0])

        if not self.quiet:
            print(f"[{(self.frame + 1):3d}/{self.sim_steps}] loss={self._best_loss:.8f}")

        self.viewer.log_scalar("/loss", self._best_loss)
        self.cost_history.append(self._best_loss)

        if self.verbose:
            print(f"  sigma={self.sigma:.4g}")

    def test_final(self):
        assert len(self.cost_history) > 0
        assert all(np.isfinite(self.cost_history))
        assert float(np.min(self.cost_history)) <= float(self._cost0) + 1e-3

    def render(self):
        self.viewer.begin_frame(self.frame * self.frame_dt)
        self.viewer.log_state(self.drone.state)

        self.viewer.log_shapes(
            "/target",
            newton.GeoType.SPHERE,
            (0.05,),
            wp.array([wp.transform(self.targets[self.target_idx], wp.quat_identity())], dtype=wp.transform),
            wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
        )

        self.viewer.end_frame()

        self.frame += 1

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--verbose", action="store_true", help="Print out additional status messages during execution."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="RNG seed for derivative-free search. Omit for a random seed each run.",
        )
        parser.add_argument(
            "--sigma0",
            type=float,
            default=0.06,
            help="Initial/resampled σ after each target change; exploration noise scale.",
        )
        parser.add_argument(
            "--sigma-decay",
            type=float,
            default=0.995,
            help="Multiply σ after each iteration.",
        )
        parser.add_argument("--sigma-min", type=float, default=0.006, help="Lower bound for σ.")
        parser.add_argument(
            "--qd-max",
            type=float,
            default=0.12,
            help="Max waypoint delta per coefficient per sample (before limits).",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=12,
            help="Random trajectory candidates per frame; more = stabler argmin, slower.",
        )
        parser.add_argument(
            "--drone_path",
            type=str,
            default=os.path.join(newton.examples.get_asset_directory(), "crazyflie.usd"),
            help="Path to the USD file (reserved; same as diffsim_drone).",
        )
        parser.add_argument(
            "--hover-trim",
            type=float,
            default=0.58,
            help="Initial waypoint (0.1–1.0). Lower if overpowered; raise if it falls.",
        )
        parser.add_argument(
            "--control-floor",
            type=float,
            default=0.38,
            help="Search floor per waypoint; widen gap to trim when thrust-scale is low.",
        )
        parser.add_argument(
            "--thrust-scale",
            type=float,
            default=0.42,
            help="Multiplies prop force/torque; too low cannot steer to the target.",
        )
        parser.add_argument(
            "--control-gain",
            type=float,
            default=0.45,
            help="Scales waypoint→prop command (diffsim uses 0.8); balance vs thrust-scale.",
        )
        parser.add_argument(
            "--gravity",
            type=float,
            default=0.0,
            help="Gravity along +Z up axis [m/s²]. 0 disables (default). Use -9.81 for Earth-like.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = jk_examples_init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
