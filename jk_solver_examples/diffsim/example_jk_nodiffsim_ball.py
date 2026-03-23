# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JK Example: Nodiffsim Ball (contrast to diffsim_ball)
#
# Same scene as ``example_diffsim_ball``, ``requires_grad=False``, no Tape.
# Optimizes initial particle velocity with **derivative-free** random search
# (Gaussian perturbation around the best-so-far, σ decay) so you can compare
# against autodiff + GD in ``diffsim_ball``.
#
# Command: python -m newton.examples jk_nodiffsim_ball
# JK: cd jk-solver && uv run python jk_solver_examples/diffsim/example_jk_nodiffsim_ball.py
#
###########################################################################
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
from newton.utils import bourke_color_map


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = args.verbose

        self.train_iter = 0
        self.target = (0.0, -2.0, 1.5)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=False)
        self.loss_history = []

        _seed = args.seed if args.seed is not None else secrets.randbits(32)
        self.rng = np.random.default_rng(_seed)
        self.sigma = float(args.sigma0)
        self.sigma_decay = float(args.sigma_decay)
        self.sigma_min = float(args.sigma_min)
        self.qd_max = float(args.qd_max)

        self.viewer = viewer
        self.viewer.show_particles = True

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)

        scene.add_particle(pos=wp.vec3(0.0, -0.5, 1.0), vel=wp.vec3(0.0, 5.0, -5.0), mass=1.0)

        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        scene.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0,
            hy=0.25,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        self.model = scene.finalize(requires_grad=False)

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=10.0,
            requires_grad=False,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.states[0], self.contacts)

        self._particle_q0 = self.states[0].particle_q.numpy().copy()
        self._particle_qd0 = self.states[0].particle_qd.numpy().copy()
        self._best_qd = self._particle_qd0.copy()
        self.forward()
        self._best_loss = float(self.loss.numpy()[0])
        self._loss0 = self._best_loss

        self.viewer.set_model(self.model)

    def forward(self):
        self.states[0].particle_q.assign(self._particle_q0)
        self.states[0].particle_qd.assign(self._particle_qd0)

        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss])

        return self.loss

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        qd_shape = self._best_qd.shape
        flat = self._best_qd.reshape(-1)
        cand = flat + self.rng.normal(0.0, self.sigma, size=flat.shape)
        cand = np.clip(cand, -self.qd_max, self.qd_max)
        self._particle_qd0 = cand.reshape(qd_shape)
        self.forward()
        trial_loss = float(self.loss.numpy()[0])
        if trial_loss < self._best_loss:
            self._best_loss = trial_loss
            self._best_qd = self._particle_qd0.copy()

        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        self._particle_qd0 = self._best_qd.copy()
        self.forward()

        if self.verbose:
            print(
                f"Iter: {self.train_iter} best_loss={self._best_loss:.6g} sigma={self.sigma:.4g} "
                f"qd={self._best_qd.reshape(-1)}"
            )

        self.train_iter += 1
        self.loss_history.append(self._best_loss)

    def test_final(self):
        assert len(self.loss_history) > 0
        assert all(np.isfinite(self.loss_history))
        assert float(np.min(self.loss_history)) <= float(self._loss0) + 1e-6

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        if self.frame > 0 and self.train_iter % 16 != 0:
            return

        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(state.particle_q.numpy()[0].tolist())

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_scalar("/loss", self._best_loss)
            self.viewer.log_state(state)
            self.viewer.log_contacts(self.contacts, state)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(traj_verts[0:-1], dtype=wp.vec3),
                wp.array(traj_verts[1:], dtype=wp.vec3),
                bourke_color_map(0.0, 7.0, self._best_loss),
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
            default=0.25,
            help="Initial Gaussian σ for velocity perturbation [m/s] (per component).",
        )
        parser.add_argument(
            "--sigma-decay",
            type=float,
            default=0.99,
            help="Multiply σ after each iteration.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="Lower bound for σ.",
        )
        parser.add_argument(
            "--qd-max",
            type=float,
            default=25.0,
            help="Clamp each initial velocity component to [-qd-max, qd-max].",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = jk_examples_init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
