# SPDX-License-Identifier: Apache-2.0
"""JK Solver — SolverMuJoCo 확장, geometric mean 마찰 오버라이드."""
from __future__ import annotations

import warp as wp

import newton
from jk_solver_examples.jk_kernels import override_contact_friction_geomean


class SolverJK(newton.solvers.SolverMuJoCo):
    """SolverMuJoCo 상속 — 접촉 마찰을 geometric mean으로 오버라이드."""

    def step(self, state_in, state_out, control, contacts, dt):
        self._enable_rne_postconstraint(state_out)
        self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
        if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
            self._update_mjc_data(self.mjw_data, self.model, state_in)
        self.mjw_model.opt.timestep.fill_(dt)
        with wp.ScopedDevice(self.model.device):
            if self.mjw_model.opt.run_collision_detection:
                self._mujoco_warp_step()
            else:
                self._convert_contacts_to_mjwarp(self.model, state_in, contacts)
                # friction override: max → geometric mean
                wp.launch(
                    override_contact_friction_geomean,
                    dim=self.mjw_data.naconmax,
                    inputs=[
                        self.mjw_data.nacon,
                        self.mjw_data.contact.geom,
                        self.mjw_data.contact.worldid,
                        self.mjw_model.geom_friction,
                    ],
                    outputs=[self.mjw_data.contact.friction],
                    device=self.model.device,
                )
                self._mujoco_warp_step()
        self._update_newton_state(self.model, state_out, self.mjw_data, state_prev=state_in)
        self._step += 1
