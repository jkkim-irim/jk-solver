# SPDX-License-Identifier: Apache-2.0
"""Trajectory generator: via point CSV → cubic Hermite spline → target trajectory NPZ."""

from trajectory_generator.hermite_spline import generate_trajectory, generate_trajectory_from_csv

__all__ = ["generate_trajectory", "generate_trajectory_from_csv"]
