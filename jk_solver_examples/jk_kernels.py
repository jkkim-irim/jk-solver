# SPDX-License-Identifier: Apache-2.0
"""JK Solver Warp kernels — friction override."""
from __future__ import annotations

import warp as wp

from newton._src.core.types import vec5


@wp.kernel(enable_backward=False)
def override_contact_friction_geomean(
    nacon: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=wp.int32),
    geom_friction: wp.array2d(dtype=wp.vec3),
    contact_friction: wp.array(dtype=vec5),
):
    """접촉 마찰계수를 max(mu1,mu2) → sqrt(mu1*mu2) geometric mean으로 교체."""
    tid = wp.tid()
    if tid >= nacon[0]:
        return

    g1 = contact_geom[tid][0]
    g2 = contact_geom[tid][1]
    world = contact_worldid[tid]

    f1 = geom_friction[world, g1]
    f2 = geom_friction[world, g2]

    _EPS = 1.0e-6
    gm_slide = wp.max(wp.sqrt(f1[0] * f2[0]), _EPS)
    gm_torsional = wp.max(wp.sqrt(f1[1] * f2[1]), _EPS)
    gm_rolling = wp.max(wp.sqrt(f1[2] * f2[2]), _EPS)

    contact_friction[tid] = vec5(gm_slide, gm_slide, gm_torsional, gm_rolling, gm_rolling)


@wp.kernel(enable_backward=False)
def update_max_penetration_kernel(
    nacon: wp.array(dtype=wp.int32),
    contact_dist: wp.array(dtype=wp.float32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=wp.int32),
    geom_to_shape: wp.array2d(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    n_bodies: int,
    # output: flat (n_bodies+1)*(n_bodies+1) array, indexed by (b0+1)*(n_bodies+1)+(b1+1)
    max_pen: wp.array(dtype=wp.float32),
):
    """GPU에서 contact별 침투깊이를 body-pair별 max로 atomic update."""
    tid = wp.tid()
    if tid >= nacon[0]:
        return

    pen = -contact_dist[tid]
    world = contact_worldid[tid]
    g0 = contact_geom[tid][0]
    g1 = contact_geom[tid][1]
    s0 = geom_to_shape[world, g0]
    s1 = geom_to_shape[world, g1]

    b0 = -1
    b1 = -1
    if s0 >= 0:
        b0 = shape_body[s0]
    if s1 >= 0:
        b1 = shape_body[s1]

    # 정렬: min, max
    ba = wp.min(b0, b1)
    bb = wp.max(b0, b1)

    stride = n_bodies + 1  # -1 → index 0, body 0 → index 1, ...
    idx = (ba + 1) * stride + (bb + 1)
    wp.atomic_max(max_pen, idx, pen)
