#!/usr/bin/env python
# coding: utf-8

# This file is part of pyDelPhi.
# Copyright (C) 2025 The pyDelPhi Project and contributors.
#
# pyDelPhi is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with pyDelPhi. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numba import njit, prange

from pydelphi.constants import (
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
    ConstDelPhiInts,
)
from pydelphi.space.core.vdw.internals import _calculate_strides

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

import numpy as np
from numba import njit
from numba.np.ufunc.parallel import get_thread_id


# -----------------------------------------------------------------------------
# LOG HELPERS (keep in this file; no extra module needed)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _reset_cpu_log_counts(log_count):
    for t in range(log_count.shape[0]):
        log_count[t] = 0


@njit(nogil=True, boundscheck=False, cache=True)
def _fill_neighbor_entity_media_6(
    idx1d: int,
    epsilon_dimension: int,
    index_discrete_epsilon_map_1d: np.ndarray,
    x_stride_x_3: int,
    y_stride_x_3: int,
    eid6: np.ndarray,  # shape (6,)
    mid6: np.ndarray,  # shape (6,)
):
    base3 = idx1d * 3

    # MUST match your serial/CPU midpoint offsets
    o0 = 0
    o1 = 1
    o2 = 2
    o3 = -x_stride_x_3
    o4 = -y_stride_x_3 + 1
    o5 = -1  # == (-3 + 2)

    # unrolled for Numba friendliness
    eps = index_discrete_epsilon_map_1d[base3 + o0]
    eid6[0] = eps % epsilon_dimension
    mid6[0] = eps // epsilon_dimension

    eps = index_discrete_epsilon_map_1d[base3 + o1]
    eid6[1] = eps % epsilon_dimension
    mid6[1] = eps // epsilon_dimension

    eps = index_discrete_epsilon_map_1d[base3 + o2]
    eid6[2] = eps % epsilon_dimension
    mid6[2] = eps // epsilon_dimension

    eps = index_discrete_epsilon_map_1d[base3 + o3]
    eid6[3] = eps % epsilon_dimension
    mid6[3] = eps // epsilon_dimension

    eps = index_discrete_epsilon_map_1d[base3 + o4]
    eid6[4] = eps % epsilon_dimension
    mid6[4] = eps // epsilon_dimension

    eps = index_discrete_epsilon_map_1d[base3 + o5]
    eid6[5] = eps % epsilon_dimension
    mid6[5] = eps // epsilon_dimension


@njit(cache=True, inline="always")
def _cpu_try_log_event(
    log_enabled: int,
    cap_per: int,
    log_count,  # (nthreads,)
    log_idx1d,  # (nthreads, cap)
    log_action,  # (nthreads, cap)
    log_color,  # (nthreads, cap)
    log_old_type,  # (nthreads, cap)
    log_new_type,  # (nthreads, cap)
    log_is_boundary,  # (nthreads, cap)
    log_is_external,  # (nthreads, cap)
    log_eid6,  # (nthreads, cap, 6) int16
    log_mid6,  # (nthreads, cap, 6) int16
    idx1d: int,
    action: int,
    color_id: int,
    old_type: int,
    new_type: int,
    is_boundary: int,
    is_external: int,
    eid6,  # (6,) int16
    mid6,  # (6,) int16
):
    if log_enabled == 0:
        return

    tid = get_thread_id()
    k = log_count[tid]
    if k >= cap_per:
        return

    log_idx1d[tid, k] = idx1d
    log_action[tid, k] = action
    log_color[tid, k] = color_id
    log_old_type[tid, k] = old_type
    log_new_type[tid, k] = new_type
    log_is_boundary[tid, k] = is_boundary
    log_is_external[tid, k] = is_external

    for j in range(6):
        log_eid6[tid, k, j] = eid6[j]
        log_mid6[tid, k, j] = mid6[j]

    log_count[tid] = k + 1


@njit(nogil=True, boundscheck=False, cache=True)
def _push_if_boundary_after_commit(
    idx1d: int,
    solute_bgp_type_1d: np.ndarray,
    visited_next: np.ndarray,  # int32[grid_npoints]
    stamp: int,  # int32
    new_bgps_idx1d: np.ndarray,  # int64[...] or int32[...]
    new_count: int,
    max_new: int,  # MUST match new_bgps_idx1d.shape[0]
    dtype_int=int,
):
    """
    If idx1d is boundary AFTER commit, push into new_bgps_idx1d unless already visited.
    No clearing: uses iteration stamp.
    """
    overflow_flag = dtype_int(0)

    if solute_bgp_type_1d[idx1d] != NON_SOLUTE_BOUNDARY:
        if visited_next[idx1d] != stamp:
            visited_next[idx1d] = stamp
            if new_count >= max_new:
                overflow_flag = dtype_int(1)
            else:
                new_bgps_idx1d[new_count] = idx1d
                new_count += 1

    return new_count, overflow_flag


@njit(nogil=True, boundscheck=False, cache=True)
def _classify_point_from_epsilon_map(
    idx1d: int,
    epsilon_dimension: int,
    index_discrete_epsilon_map_1d,
    x_stride_x_3: int,  # == (ny*nz)*3
    y_stride_x_3: int,  # == (nz)*3
    dtype_int=int,
):
    """
    Serial-equivalent classification (status logic):
      - is_external = 1 if ANY of the 6 midpoint media ids is 0
      - is_boundary = 1 if ANY adjacent midpoint media ids differ
                     (including v1 vs v6 check)
    """
    epsdim = dtype_int(epsilon_dimension)
    base = dtype_int(idx1d * 3)

    # Offsets (match serial):
    o1 = dtype_int(0)
    o2 = dtype_int(1)
    o3 = dtype_int(2)
    o4 = dtype_int(-x_stride_x_3)
    o5 = dtype_int(-y_stride_x_3 + 1)
    o6 = dtype_int(-1)  # == -3 + 2 since z_stride==1

    v1 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o1]) // epsdim)
    v2 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o2]) // epsdim)
    v3 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o3]) // epsdim)
    v4 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o4]) // epsdim)
    v5 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o5]) // epsdim)
    v6 = dtype_int(abs(index_discrete_epsilon_map_1d[base + o6]) // epsdim)

    is_external = dtype_int(0)
    if (v1 == 0) or (v2 == 0) or (v3 == 0) or (v4 == 0) or (v5 == 0) or (v6 == 0):
        is_external = dtype_int(1)

    is_boundary = dtype_int(0)
    if v1 != v6:
        is_boundary = dtype_int(1)
    if (v2 != v1) or (v3 != v2) or (v4 != v3) or (v5 != v4) or (v6 != v5):
        is_boundary = dtype_int(1)

    return is_boundary, is_external


# -----------------------------------------------------------------------------
# PASS-1: count touched points per BGP (7-point stencil: +/- neighbors + center)
# -----------------------------------------------------------------------------
@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _color_count_events(
    color_id: int,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices: np.ndarray,  # (Btot,3)
    grid_shape: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    dtype_int=int,
):
    """
    Per-color PASS-1 (parallel): count how many touched gridpoints we will emit
    per BGP in this color. This MUST match serial visitation:
      neighbor_index = 1..7 (inclusive)  => 6 neighbors + center.
    """
    B = boundary_point_end_index - boundary_point_start_index + 1
    counts = np.zeros(B, dtype=np.int32)

    nx = dtype_int(grid_shape[0])
    ny = dtype_int(grid_shape[1])
    nz = dtype_int(grid_shape[2])

    ny_nz = dtype_int(ny * nz)

    for bi in prange(B):
        bgp_index = boundary_point_start_index + bi
        ixyz = boundary_grid_indices[bgp_index - 1]
        ix = dtype_int(ixyz[0])
        iy = dtype_int(ixyz[1])
        iz = dtype_int(ixyz[2])

        # color filter
        if ((ix % 3) + 3 * (iy % 3) + 9 * (iz % 3)) != color_id:
            continue

        # center must still be marked boundary
        idx1d_center = ix * ny_nz + iy * dtype_int(nz) + iz
        if solute_bgp_type_1d[idx1d_center] == NON_SOLUTE_BOUNDARY:
            continue

        c = np.int32(0)

        # neighbor_index=1..6 bounds
        if ix + 1 < nx - 1:
            c += 1
        if iy + 1 < ny - 1:
            c += 1
        if iz + 1 < nz - 1:
            c += 1
        if ix - 1 > 0:
            c += 1
        if iy - 1 > 0:
            c += 1
        if iz - 1 > 0:
            c += 1

        # neighbor_index=7 (center) should be evaluated if center is interior
        if (
            (ix > 0)
            and (ix < nx - 1)
            and (iy > 0)
            and (iy < ny - 1)
            and (iz > 0)
            and (iz < nz - 1)
        ):
            c += 1

        counts[bi] = c

    total = np.int32(0)
    for i in range(B):
        total += counts[i]

    return counts, total


@njit(nogil=True, boundscheck=False, cache=True)
def _exclusive_prefix_sum_int32(counts: np.ndarray):
    """
    Exclusive prefix sum:
      offsets[i] = sum_{k < i} counts[k]
      offsets[n] = total
    """
    n = counts.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int32)
    s = np.int32(0)
    for i in range(n):
        offsets[i] = s
        s += counts[i]
    offsets[n] = s
    return offsets


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _color_fill_events(
    color_id: int,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices: np.ndarray,
    grid_shape: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    event_idx1d: np.ndarray,
    dtype_int=int,
):
    B = boundary_point_end_index - boundary_point_start_index + 1

    nx = dtype_int(grid_shape[0])
    ny = dtype_int(grid_shape[1])
    nz = dtype_int(grid_shape[2])

    ny_nz = dtype_int(ny * nz)

    for bi in prange(B):
        if counts[bi] == 0:
            continue

        bgp_index = boundary_point_start_index + bi
        ixyz = boundary_grid_indices[bgp_index - 1]
        ix = dtype_int(ixyz[0])
        iy = dtype_int(ixyz[1])
        iz = dtype_int(ixyz[2])

        if ((ix % 3) + 3 * (iy % 3) + 9 * (iz % 3)) != color_id:
            continue

        idx1d_center = ix * ny_nz + iy * dtype_int(nz) + iz

        # IMPORTANT: this gate MUST match PASS-1 if PASS-1 gated on "center is boundary"
        # If PASS-1 did gate but PASS-2 does not, you can overflow writes -> segfault.
        if solute_bgp_type_1d[idx1d_center] == NON_SOLUTE_BOUNDARY:
            continue

        w = offsets[bi]

        # 1 (+x)
        if ix + 1 < nx - 1:
            event_idx1d[w] = (ix + 1) * ny_nz + iy * dtype_int(nz) + iz
            w += 1
        # 2 (+y)
        if iy + 1 < ny - 1:
            event_idx1d[w] = ix * ny_nz + (iy + 1) * dtype_int(nz) + iz
            w += 1
        # 3 (+z)
        if iz + 1 < nz - 1:
            event_idx1d[w] = ix * ny_nz + iy * dtype_int(nz) + (iz + 1)
            w += 1
        # 4 (-x)
        if ix - 1 > 0:
            event_idx1d[w] = (ix - 1) * ny_nz + iy * dtype_int(nz) + iz
            w += 1
        # 5 (-y)
        if iy - 1 > 0:
            event_idx1d[w] = ix * ny_nz + (iy - 1) * dtype_int(nz) + iz
            w += 1
        # 6 (-z)
        if iz - 1 > 0:
            event_idx1d[w] = ix * ny_nz + iy * dtype_int(nz) + (iz - 1)
            w += 1

        # 7 (center)
        if (
            (ix > 0)
            and (ix < nx - 1)
            and (iy > 0)
            and (iy < ny - 1)
            and (iz > 0)
            and (iz < nz - 1)
        ):
            event_idx1d[w] = idx1d_center
            w += 1


# -----------------------------------------------------------------------------
# COMMIT: dedup touched idx1d, recompute (is_boundary,is_external) from epsilon map,
# apply serial update rules, update counts, and optionally build next frontier.
# -----------------------------------------------------------------------------
@njit(nogil=True, boundscheck=False, cache=True)
def _commit_events_and_build_next(
    *,
    event_idx1d: np.ndarray,  # touched idx1d stream
    num_events: int,
    grid_shape: np.ndarray,
    boundary_grid_indices: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    epsilon_dimension: int,
    num_discovered_bndy_grid_points: int,
    max_boundary_grid_points: int,
    num_external_boundary_points: int,
    added_boundary_points_increment: int,
    removed_boundary_points_increment: int,
    visited_next: np.ndarray,
    stamp: int,
    new_bgps_idx1d: np.ndarray,
    new_count: int,
    processed_stamp: np.ndarray,  # int32[grid_npoints]
    dtype_int=int,
    # ---- debug log (CPU, per-thread buffers) ----
    log_enabled: int = 0,
    log_color_id: int = -1,
    log_cap_per: int = 0,
    log_count=None,
    log_idx1d=None,
    log_action=None,
    log_color=None,
    log_old_type=None,
    log_new_type=None,
    log_is_boundary=None,
    log_is_external=None,
    log_eid6=None,
    log_mid6=None,
):
    """
    Serial-equivalent COMMIT over the *touched set* (deduped):
      - recompute boundary/external status from epsilon map
      - apply the exact serial rules in _update_neighbor_status (remove/extclr/create)
      - optionally build next frontier (points that are boundary after commit)
    """
    exec_status = dtype_int(0)
    overflow = dtype_int(0)

    z_stride = dtype_int(1)
    y_stride = dtype_int(grid_shape[2])
    x_stride = dtype_int(grid_shape[1]) * y_stride

    y_stride_x_3 = y_stride * 3
    x_stride_x_3 = x_stride * 3

    max_new = dtype_int(new_bgps_idx1d.shape[0])

    for e in range(num_events):
        idx1d = dtype_int(event_idx1d[e])

        # dedup touched points per color (stamp advanced per color by caller)
        if processed_stamp[idx1d] == stamp:
            continue
        processed_stamp[idx1d] = stamp

        slot = dtype_int(idx1d * 3)

        # ---- MUST be defined for all paths (Numba) ----
        action = dtype_int(4)  # 4 = noop
        old_type = dtype_int(solute_bgp_type_1d[idx1d])
        new_type = old_type

        # optional payloads (keep zeros unless you later fill real values)
        eid6 = np.empty(6, dtype=np.int16)
        mid6 = np.empty(6, dtype=np.int16)
        # for j in range(6):
        #     eid6[j] = np.int16(0)
        #     mid6[j] = np.int16(0)

        _fill_neighbor_entity_media_6(
            idx1d=int(idx1d),
            epsilon_dimension=int(epsilon_dimension),
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            x_stride_x_3=int(x_stride_x_3),
            y_stride_x_3=int(y_stride_x_3),
            eid6=eid6,
            mid6=mid6,
        )

        was_marked = dtype_int(0)
        old_ext = dtype_int(0)

        # note: use old_type captured above, not re-reading (keeps log consistent)
        if old_type != NON_SOLUTE_BOUNDARY:
            was_marked = dtype_int(1)
            old_ext = (old_type & SOLUTE_BOUNDARY_EXTERNAL) == SOLUTE_BOUNDARY_EXTERNAL

        is_boundary, is_external = _classify_point_from_epsilon_map(
            idx1d=idx1d,
            epsilon_dimension=epsilon_dimension,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            x_stride_x_3=x_stride_x_3,
            y_stride_x_3=y_stride_x_3,
            dtype_int=dtype_int,
        )

        # --- REMOVE: no longer boundary but was marked ---
        if (is_boundary == 0) and (was_marked == 1):
            num_external_boundary_points -= old_ext
            solute_bgp_type_1d[idx1d] = NON_SOLUTE_BOUNDARY
            removed_boundary_points_increment += 1
            action = dtype_int(2)

        else:
            # --- EXTCLR: still boundary, but should not be external ---
            if (
                (is_boundary == 1)
                and (was_marked == 1)
                and (old_ext == 1)
                and (is_external == 0)
            ):
                solute_bgp_type_1d[idx1d] = NON_SOLUTE_BOUNDARY
                num_external_boundary_points -= dtype_int(1)
                action = dtype_int(3)

        # --- CREATE: boundary but was not marked ---
        if (is_boundary == 1) and (was_marked == 0):
            action = dtype_int(1)
            added_boundary_points_increment += dtype_int(1)
            new_id = dtype_int(
                num_discovered_bndy_grid_points + added_boundary_points_increment
            )

            if new_id > max_boundary_grid_points:
                exec_status = EXIT_NJIT_FLAG
                break

            solute_bgp_type_1d[idx1d] = SOLUTE_BOUNDARY_ANY | (
                SOLUTE_BOUNDARY_EXTERNAL if is_external else np.uint8(0)
            )
            num_external_boundary_points += dtype_int(is_external)

            ix, iy, iz = _ijk_from_idx1d(
                idx1d, x_stride, y_stride, z_stride, dtype_int=dtype_int
            )
            boundary_grid_indices[new_id - 1, 0] = ix
            boundary_grid_indices[new_id - 1, 1] = iy
            boundary_grid_indices[new_id - 1, 2] = iz

        # frontier: push if boundary after commit
        new_count, overflow = _push_if_boundary_after_commit(
            idx1d=idx1d,
            solute_bgp_type_1d=solute_bgp_type_1d,
            visited_next=visited_next,
            stamp=stamp,
            new_bgps_idx1d=new_bgps_idx1d,
            new_count=new_count,
            max_new=max_new,
            dtype_int=dtype_int,
        )
        if overflow != 0:
            exec_status = EXIT_NJIT_FLAG
            break

        # compute final new_type for logging
        new_type = dtype_int(solute_bgp_type_1d[idx1d])

        if action != 4:
            _cpu_try_log_event(
                log_enabled,
                log_cap_per,
                log_count,
                log_idx1d,
                log_action,
                log_color,
                log_old_type,
                log_new_type,
                log_is_boundary,
                log_is_external,
                log_eid6,
                log_mid6,
                int(idx1d),
                int(action),
                int(log_color_id),
                int(old_type),
                int(new_type),
                int(is_boundary),
                int(is_external),
                eid6,
                mid6,
            )

    return (
        exec_status,
        boundary_grid_indices,
        solute_bgp_type_1d,
        num_external_boundary_points,
        added_boundary_points_increment,
        removed_boundary_points_increment,
        new_bgps_idx1d,
        new_count,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _finalize_frontier_centers_gridpt_status(
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices: np.ndarray,  # (Btot,3)
    grid_shape: np.ndarray,
    epsilon_dimension: int,
    index_discrete_epsilon_map_1d: np.ndarray,  # (grid_npoints*3,)
    solute_bgp_type_1d: np.ndarray,
    num_external_boundary_points: int,
    removed_boundary_points_increment: int,
    dtype_int=int,
):
    """
    Serial-equivalent of calling _update_gridpoint_boundary_status() for EACH frontier BGP center.

    This is *required* because serial does this even when the point never appears
    in neighbor-touched set (watch shows "NOT TOUCHED" otherwise).
    """
    epsilon_dimension = dtype_int(epsilon_dimension)

    x_stride, y_stride, z_stride, x_stride_x_3, y_stride_x_3, z_stride_x_3 = (
        _calculate_strides(grid_shape, dtype_int=dtype_int)
    )

    # neighbor midpoint offsets in the epsilon map (same as serial)
    o1 = dtype_int(0)
    o2 = dtype_int(1)
    o3 = dtype_int(2)
    o4 = dtype_int(-x_stride_x_3)
    o5 = dtype_int(-y_stride_x_3 + 1)
    o6 = dtype_int(-z_stride_x_3 + 2)

    for bgp in range(boundary_point_start_index, boundary_point_end_index + 1):
        ix = dtype_int(boundary_grid_indices[bgp - 1, 0])
        iy = dtype_int(boundary_grid_indices[bgp - 1, 1])
        iz = dtype_int(boundary_grid_indices[bgp - 1, 2])

        idx1d = dtype_int(ix * x_stride + iy * y_stride + iz * z_stride)
        slot = dtype_int(idx1d * 3)

        # If it's not currently marked as boundary, serial also skips the midpoint phase,
        # but _update_gridpoint_boundary_status is called only when it was processed as a BGP.
        # We are sweeping the frontier list itself, so we must still apply the status update
        # if it is/was marked. If it is fully unmarked, nothing to do.
        if solute_bgp_type_1d[idx1d] == NON_SOLUTE_BOUNDARY:
            continue

        old_ext = (
            solute_bgp_type_1d[idx1d] & SOLUTE_BOUNDARY_EXTERNAL
            == SOLUTE_BOUNDARY_EXTERNAL
        )

        # Read 6 midpoint media ids (abs() // epsdim), same as serial
        v1 = abs(index_discrete_epsilon_map_1d[slot + o1]) // epsilon_dimension
        v2 = abs(index_discrete_epsilon_map_1d[slot + o2]) // epsilon_dimension
        v3 = abs(index_discrete_epsilon_map_1d[slot + o3]) // epsilon_dimension
        v4 = abs(index_discrete_epsilon_map_1d[slot + o4]) // epsilon_dimension
        v5 = abs(index_discrete_epsilon_map_1d[slot + o5]) // epsilon_dimension
        v6 = abs(index_discrete_epsilon_map_1d[slot + o6]) // epsilon_dimension

        is_external = dtype_int(0)
        if (v1 == 0) or (v2 == 0) or (v3 == 0) or (v4 == 0) or (v5 == 0) or (v6 == 0):
            is_external = dtype_int(1)

        is_boundary = dtype_int(0)
        if v1 != v6:
            is_boundary = dtype_int(1)
        if (v2 != v1) or (v3 != v2) or (v4 != v3) or (v5 != v4) or (v6 != v5):
            is_boundary = dtype_int(1)

        # EXACT serial behavior in _update_gridpoint_boundary_status:
        # if (is_external==0) OR (is_boundary==0):
        #   decrement ext count by old_ext
        #   if is_boundary==1: update ext flag
        #   if is_boundary==0: clear boundary id/ext and ++removed
        if (is_external == 0) or (is_boundary == 0):
            num_external_boundary_points -= old_ext

            if is_boundary == 1:
                solute_bgp_type_1d[idx1d] = (
                    SOLUTE_BOUNDARY_ANY | SOLUTE_BOUNDARY_EXTERNAL
                )
            else:
                # solute_bgp_type_1d[slot + 1] = 0
                solute_bgp_type_1d[idx1d] = NON_SOLUTE_BOUNDARY
                removed_boundary_points_increment += 1

    return num_external_boundary_points, removed_boundary_points_increment


@njit(nogil=True, cache=True)
def _append_idx1d_if_boundary(
    idx1d,
    solute_bgp_type_1d,
    stamp_arr,
    stamp,
    out_idx1d,
    out_count,
    dtype_int=int,
):
    # boundary mark is slot+1
    if solute_bgp_type_1d[idx1d] == NON_SOLUTE_BOUNDARY:
        return out_count
    if stamp_arr[idx1d] == stamp:
        return out_count
    stamp_arr[idx1d] = stamp
    out_idx1d[out_count] = idx1d
    return out_count + 1


@njit(nogil=True, cache=True)
def _compact_boundary_grid_indices_simple(
    new_bgps_idx1d: np.ndarray,
    new_count: int,
    total_discovered_end_index: int,
    boundary_grid_indices: np.ndarray,  # (maxB,3)
    solute_bgp_type_1d: np.ndarray,  # channel +1 ids, +2 ext flag rebuilt here
    index_discrete_epsilon_map_1d: np.ndarray,
    epsilon_dimension: int,
    grid_shape: np.ndarray,
    stamp_arr: np.ndarray,  # int32[grid_npoints]
    stamp: int,
    dtype_int=int,
):
    """
    Compact boundary_grid_indices_1d by:
      - treating (ix,iy,iz)!=(0,0,0) as the ONLY truth for membership
      - removing holes
      - appending newly discovered points
      - de-duplicating (by stamp_arr)
      - rebuilding DBM contiguously:
          dbm[idx1d*3+1] = 1..out_count  (ID consistent with compacted list)
          dbm[idx1d*3+2] = is_external   (serial-equivalent from epsilon map)
    Returns:
      out_count (int)
    """
    grid_npoints = stamp_arr.shape[0]

    y_stride = dtype_int(grid_shape[2])  # nz
    x_stride = dtype_int(grid_shape[1]) * y_stride  # ny*nz

    x_stride_x_3 = dtype_int(x_stride * 3)  # (ny*nz)*3
    y_stride_x_3 = dtype_int(y_stride * 3)  # (nz)*3

    out_count = dtype_int(0)

    # ------------------------------------------------------------------
    # (0) clear DBM id+ext for all previously discovered (non-hole) points
    # ------------------------------------------------------------------
    for i in range(total_discovered_end_index):
        ix = boundary_grid_indices[i, 0]
        iy = boundary_grid_indices[i, 1]
        iz = boundary_grid_indices[i, 2]

        if ix == 0 and iy == 0 and iz == 0:
            continue

        idx1d = dtype_int(ix * x_stride + iy * y_stride + iz)
        if 0 <= idx1d < grid_npoints:
            base3 = dtype_int(idx1d * 3)
            # solute_bgp_type_1d[idx1d] = dtype_int(0)
            solute_bgp_type_1d[idx1d] = NON_SOLUTE_BOUNDARY

    # ------------------------------------------------------------------
    # (A1) keep all existing non-hole boundary points (dedup)
    # ------------------------------------------------------------------
    for i in range(total_discovered_end_index):
        ix = boundary_grid_indices[i, 0]
        iy = boundary_grid_indices[i, 1]
        iz = boundary_grid_indices[i, 2]

        if ix == 0 and iy == 0 and iz == 0:
            continue

        idx1d = dtype_int(ix * x_stride + iy * y_stride + iz)
        if idx1d < 0 or idx1d >= grid_npoints:
            continue

        if stamp_arr[idx1d] == stamp:
            continue
        stamp_arr[idx1d] = stamp

        boundary_grid_indices[out_count, 0] = ix
        boundary_grid_indices[out_count, 1] = iy
        boundary_grid_indices[out_count, 2] = iz

        # ext flag rebuilt deterministically from epsilon map
        _, is_external = _classify_point_from_epsilon_map(
            idx1d=idx1d,
            epsilon_dimension=epsilon_dimension,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            x_stride_x_3=x_stride_x_3,
            y_stride_x_3=y_stride_x_3,
            dtype_int=dtype_int,
        )
        solute_bgp_type_1d[idx1d] = (
            SOLUTE_BOUNDARY_EXTERNAL | SOLUTE_BOUNDARY_ANY
            if is_external
            else NON_SOLUTE_BOUNDARY
        )

        out_count += 1

    # ------------------------------------------------------------------
    # (A2) append newly discovered points (dedup)
    # ------------------------------------------------------------------
    for i in range(new_count):
        idx1d = dtype_int(new_bgps_idx1d[i])
        if idx1d < 0 or idx1d >= grid_npoints:
            continue

        if stamp_arr[idx1d] == stamp:
            continue
        stamp_arr[idx1d] = stamp

        ix = dtype_int(idx1d // x_stride)
        rem = dtype_int(idx1d - ix * x_stride)
        iy = dtype_int(rem // y_stride)
        iz = dtype_int(rem - iy * y_stride)

        boundary_grid_indices[out_count, 0] = ix
        boundary_grid_indices[out_count, 1] = iy
        boundary_grid_indices[out_count, 2] = iz

        _, is_external = _classify_point_from_epsilon_map(
            idx1d=idx1d,
            epsilon_dimension=epsilon_dimension,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            x_stride_x_3=x_stride_x_3,
            y_stride_x_3=y_stride_x_3,
            dtype_int=dtype_int,
        )
        solute_bgp_type_1d[idx1d] = (
            SOLUTE_BOUNDARY_EXTERNAL | SOLUTE_BOUNDARY_ANY
            if is_external
            else NON_SOLUTE_BOUNDARY
        )

        out_count += 1

    # ------------------------------------------------------------------
    # (A3) clear tail in the old discovered range
    # ------------------------------------------------------------------
    for i in range(out_count, total_discovered_end_index):
        boundary_grid_indices[i, 0] = dtype_int(0)
        boundary_grid_indices[i, 1] = dtype_int(0)
        boundary_grid_indices[i, 2] = dtype_int(0)

    return out_count


@njit(nogil=True, cache=True)
def _ijk_from_idx1d(
    idx1d: int,
    x_stride: int,
    y_stride: int,
    z_stride: int,
    dtype_int=int,
):
    """
    Convert idx1d to (ix, iy, iz) using precomputed strides.
    Assumes:
      z_stride = 1
      y_stride = nz
      x_stride = ny * nz
    """
    ix = dtype_int(idx1d // x_stride)
    rem = dtype_int(idx1d - ix * x_stride)
    iy = dtype_int(rem // y_stride)
    iz = dtype_int(rem - iy * y_stride)
    return ix, iy, iz


@njit(nogil=True, boundscheck=False, cache=True)
def _remap_epsilon_map_parallel_inplace(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map: np.ndarray,
    neighbor_index: int,
    epsilon_dimension: int,
    contact_region_media_id: int,
    neighbor_entity_id: int,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int=int,
    dtype_real=float,
):
    """
    IDENTICAL math to _remap_epsilon_map, but:
      - no prints
      - no return (in-place only)
    """
    index_1d_x_3 = (
        (grid_index_x + index_map[1][neighbor_index]) * stride_x_3
        + (grid_index_y + index_map[2][neighbor_index]) * stride_y_3
        + (grid_index_z + index_map[3][neighbor_index]) * 3
    )

    entity_id = neighbor_entity_id
    new_epsilon_value = dtype_real(
        entity_id + contact_region_media_id * epsilon_dimension
    )

    flag = index_map[4][neighbor_index]
    if flag == 1:
        index_discrete_epsilon_map_1d[index_1d_x_3] = new_epsilon_value
    elif flag == 2:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 1] = new_epsilon_value
    elif flag == 3:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 2] = new_epsilon_value


@njit(nogil=True, boundscheck=False, cache=True)
def _remap_possibly_sign_updated_epsilon_map_parallel_inplace(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map: np.ndarray,
    neighbor_index: int,
    epsilon_dimension: int,
    neighbor_entity_id: int,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int=int,
    dtype_real=float,
):
    """
    Parallel-safe version of _remap_possibly_sign_updated_epsilon_map:
      - same math
      - no prints
      - in-place (no return)
    """
    sign = dtype_real(1.0)

    index_1d_x_3 = (
        (grid_index_x + index_map[1][neighbor_index]) * stride_x_3
        + (grid_index_y + index_map[2][neighbor_index]) * stride_y_3
        + (grid_index_z + index_map[3][neighbor_index]) * 3
    )

    # pick the component addressed by flag (1/2/3)
    flag = index_map[4][neighbor_index]
    eps_idx = index_discrete_epsilon_map_1d[index_1d_x_3 + flag - 1]

    # IMPORTANT: keep same semantics as your serial code (Numba uses fmod-like remainder for floats)
    midpoint_original_entity_id = eps_idx % epsilon_dimension
    if midpoint_original_entity_id < 0:
        return  # already sign-updated => do nothing

    if neighbor_entity_id < 0:
        sign = dtype_real(-1.0)
        if midpoint_original_entity_id == 0:
            midpoint_original_entity_id = 1

    midpoint_media_id = abs(eps_idx) // epsilon_dimension
    new_eps = dtype_real(
        sign * (midpoint_original_entity_id + midpoint_media_id * epsilon_dimension)
    )

    if flag == 1:
        index_discrete_epsilon_map_1d[index_1d_x_3] = new_eps
    elif flag == 2:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 1] = new_eps
    elif flag == 3:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 2] = new_eps
