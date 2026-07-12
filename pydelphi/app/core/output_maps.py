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


from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pydelphi.foundation.enums import DielectricModel, SurfaceMethod
from pydelphi.config.global_runtime import vprint
from pydelphi.config.logging_config import DEBUG, ERROR, WARNING

import pydelphi.utils.io.writers as wrt
import pydelphi.utils.io.readers as rdr


# -------------------------
# Small pure helpers
# -------------------------
def get_phase_specific_filename(datafile: str, datalabel: str) -> str:
    """
    Insert `_{datalabel}` before file extension.
    """
    i = datafile.rindex(".")
    return datafile[:i] + f"_{datalabel}" + datafile[i:]


def construct_eps_map_title(surface_method, srfexp, m, sigma, datalabel: str) -> str:
    if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
        return f"gridpoint-epsmap({datalabel}): srfexp={srfexp} m={m} sigma={sigma}"
    else:
        return f"gridpoint-epsmap({datalabel})"


def construct_phi_map_title(srfexp, m, sigma, datalabel: str, surface_method) -> str:
    if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
        return f"phimap({datalabel}): srfexp={srfexp} m={m} sigma={sigma}"
    else:
        return f"phimap({datalabel})"


# -------------------------
# Cube writing + verification
# -------------------------
def write_cube_and_verify(
    *,
    filename: str,
    scale: float,
    grid_center: NDArray[np.floating],
    grid_shape: NDArray[np.integer],
    data_1d: NDArray[np.floating],
    content: str,
    format_type: str,
    binary_precision,
    verbosity: int,
    is_4d: bool = False,
    verify: bool = True,
) -> bool:
    """
    Writes a cube file (or 4D cube) and optionally verifies its content after reading.

    Notes:
      - 4D verification intentionally not implemented (matches current behavior).
      - `wrt` and `rdr` are injected modules/objects (so core does not depend on app).

    Returns:
      - bool success
    """
    if is_4d:
        wrt.write_cube_4d(
            filename,
            scale,
            grid_center,
            grid_shape,
            data_1d,
            content,
            binary_precision=binary_precision,
            format=format_type,
        )
        vprint(
            DEBUG,
            verbosity,
            f"4D cube written to {filename}. Verification not implemented.",
        )
        return True

    # Write 3D cube file
    wrt.write_cube(
        filename,
        scale,
        grid_center,
        grid_shape,
        data_1d,
        binary_precision=binary_precision,
        content=content,
        format=format_type,
    )

    if not verify:
        vprint(DEBUG, verbosity, f"Verification skipped for {filename}")
        return True

    # Read back the written cube file for verification
    (
        read_scale,
        read_grid_center,
        read_grid_shape,
        read_data_3d,
        read_origin_bohr,
        read_vectors_bohr,
        read_comment,
        read_data_type_comment,
        read_endianness,
        read_marker,
    ) = rdr.read_cube(filename, format=format_type)

    # Verification checks
    if not np.isclose(scale, read_scale):
        vprint(
            ERROR,
            verbosity,
            f"ERROR: Scale mismatch in {filename}: {scale} != {read_scale}",
        )
        return False

    if not np.allclose(grid_center, read_grid_center):
        vprint(
            ERROR,
            verbosity,
            f"ERROR: Grid center mismatch in {filename}: {grid_center} != {read_grid_center}",
        )
        return False

    if not np.array_equal(grid_shape, read_grid_shape):
        vprint(
            ERROR,
            verbosity,
            f"ERROR: Grid shape mismatch in {filename}: {grid_shape} != {read_grid_shape}",
        )
        return False

    if data_1d.ndim == 1:
        read_data_1d_ravelled = read_data_3d.ravel()
        if not np.allclose(data_1d, read_data_1d_ravelled):
            vprint(ERROR, verbosity, f"ERROR: Data mismatch in {filename}")
            max_diff = np.max(np.abs(data_1d - read_data_1d_ravelled))
            vprint(WARNING, verbosity, f"Max difference: {max_diff}")
            return False
    elif data_1d.ndim == 3:
        if not np.allclose(data_1d, read_data_3d):
            vprint(ERROR, verbosity, f"ERROR: Data mismatch in {filename}")
            max_diff = np.max(np.abs(data_1d - read_data_3d))
            vprint(WARNING, verbosity, f"Max difference: {max_diff}")
            return False

    # Keep the comment logic unchanged (it was intentionally not enforced)
    binary_data_type_str = (
        "binary float64 (Fortran unformatted, little-endian)"
        if data_1d.dtype == np.float64
        else "binary float32 (Fortran unformatted, little-endian)"
    )
    data_type_comment = (
        f"Data type: {data_1d.dtype}"
        if format_type != "phi"
        else f"Data type: {binary_data_type_str}"
    )
    written_data_type_comment = (
        f"Gaussian cube {content} "
        f"({'binary format Fortran unformatted, little-endian' if format_type == 'phi' else 'text format'}, "
        f"{data_type_comment})"
    )
    # Intentionally not comparing written_data_type_comment vs read_data_type_comment

    vprint(DEBUG, verbosity, f"Verification successful for {filename}")
    return True


# -------------------------
# Higher-level writers
# -------------------------
def write_spatial_maps(*, inp, ctx, verbosity: int) -> None:
    """
    Writes spatial maps (density and surface) to cube files and verifies them.
    """
    out_density = inp.get_param("out__density")
    dielectric_model = inp.get_param_value("dielectric_model")
    out_surf = inp.get_param("out__surf")

    # Gaussian density map
    if (
        dielectric_model.int_value == DielectricModel.GAUSSIAN.int_value
        and out_density
        and out_density.active
    ):
        density_format = out_density.get_attribute("format")
        density_bin_precision = (
            np.float32
            if out_density.get_attribute("precision") == "single"
            else np.float64
        )

        if out_density.get_attribute("point") in ("both", "grid"):
            content = (
                f"Gaussian density map: m={inp.get_param_value('gaussian_exponent')}, "
                f"sigma={inp.get_param_value('sigma')}"
            )
            ok = write_cube_and_verify(
                filename=out_density.get_attribute("file"),
                scale=ctx.scale,
                grid_center=ctx.grid_center,
                grid_shape=ctx.grid_shape,
                data_1d=ctx.gauss_density_map_1d,
                content=content,
                format_type=density_format,
                binary_precision=density_bin_precision,
                verbosity=verbosity,
            )
            if not ok:
                print("Verification failed for density map.")

        if out_density.get_attribute("point") in ("both", "mid"):
            datafile = get_phase_specific_filename(
                out_density.get_attribute("file"), "midpoint"
            )
            content = (
                f"Gaussian density map at midpoints: m={inp.get_param_value('gaussian_exponent')}, "
                f"sigma={inp.get_param_value('sigma')}"
            )
            data_shape_4d = np.array(list(ctx.grid_shape) + [3], dtype=int)
            ok = write_cube_and_verify(
                filename=datafile,
                scale=ctx.scale,
                grid_center=ctx.grid_center,
                grid_shape=data_shape_4d,
                data_1d=ctx.gauss_density_map_midpoints_1d,
                content=content,
                format_type=density_format,
                binary_precision=density_bin_precision,
                is_4d=True,
                verify=False,
                verbosity=verbosity,
            )
            if not ok:
                print("Verification failed for density map.")

    # Surface map
    if out_surf and out_surf.active:
        surf_format = out_surf.get_attribute("format")
        surf_bin_precision = (
            np.float32
            if out_surf.get_attribute("precision") == "single"
            else np.float64
        )
        surface_method = inp.get_param_value("surface_method")

        if surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
            # Keep existing content behavior (including current formatting style).
            content = (
                f"Surface map: srfexp={inp.get_param_value('surface_density_exponent')}, "
                "m={inp.get_param_value('gaussian_exponent')}, sigma={inp.get_param_value('sigma')}"
            )
            data_1d = ctx.surface_map_1d
        elif surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
            content = "Surface map also known as idebmap generated with GAUSSIONCUTOFF method."
            data_1d = ctx.ion_exclusion_map_1d
        elif surface_method.int_value == SurfaceMethod.VDW.int_value:
            content = "Surface map also known as idebmap generated with VDW method."
            data_1d = ctx.dielectric_boundary_map_1d
        else:
            return

        ok = write_cube_and_verify(
            filename=out_surf.get_attribute("file"),
            scale=ctx.scale,
            grid_center=ctx.grid_center,
            grid_shape=ctx.grid_shape,
            data_1d=data_1d,
            content=content,
            format_type=surf_format,
            binary_precision=surf_bin_precision,
            verbosity=verbosity,
        )
        if not ok:
            vprint(DEBUG, verbosity, "Verification failed for surface map.")


def write_phase_dependent_maps(*, inp, ctx, verbosity: int, isvacuum: bool) -> None:
    """
    Writes epsilon (grid + midpoint), phi, and zeta-phi maps for a phase.
    """
    out_eps = inp.get_param("out__eps")
    out_phi = inp.get_param("out__phi")
    out_zphi = inp.get_param("out__zphi")
    surface_method = inp.get_param_value("surface_method")
    srfexp = inp.get_param_value("surface_density_exponent")
    m = inp.get_param_value("gaussian_exponent")
    sigma = inp.get_param_value("sigma")
    is_zeta_on = inp.get_param_value("zeta_potential")

    # epsilon maps
    if out_eps and out_eps.active:
        point_type = out_eps.get_attribute("point")
        media_type = out_eps.get_attribute("media")
        eps_format = out_eps.get_attribute("format")
        eps_bin_precision = (
            np.float32 if out_eps.get_attribute("precision") == "single" else np.float64
        )

        if point_type in ("both", "grid"):
            datalabel = "vacuum" if isvacuum else "water"
            datafile = out_eps.get_attribute("file")
            if media_type == "both":
                datafile = get_phase_specific_filename(datafile, datalabel)

            datamap = ctx.epsilon_map_1d  # same for both phases
            epsmap_title = construct_eps_map_title(
                surface_method, srfexp, m, sigma, datalabel
            )

            if datamap is not None:
                write_cube_and_verify(
                    filename=datafile,
                    scale=ctx.scale,
                    grid_center=ctx.grid_center,
                    grid_shape=ctx.grid_shape,
                    data_1d=datamap,
                    content=epsmap_title,
                    format_type=eps_format,
                    binary_precision=eps_bin_precision,
                    verbosity=verbosity,
                )

        if point_type in ("both", "mid"):
            datalabel = "vacuum_mids" if isvacuum else "water_mids"
            datafile = out_eps.get_attribute("file")
            if media_type == "both":
                datafile = get_phase_specific_filename(datafile, datalabel)

            datamap = (
                ctx.epsilon_map_midpoints_vacuum_1d
                if isvacuum
                else ctx.epsilon_map_midpoints_water_1d
            )
            epsmap_title = construct_eps_map_title(
                surface_method, srfexp, m, sigma, datalabel
            )

            if datamap is not None:
                data_shape_4d = np.array(list(ctx.grid_shape) + [3], dtype=int)
                write_cube_and_verify(
                    filename=datafile,
                    scale=ctx.scale,
                    grid_center=ctx.grid_center,
                    grid_shape=data_shape_4d,
                    data_1d=datamap,
                    content=epsmap_title,
                    format_type=eps_format,
                    binary_precision=eps_bin_precision,
                    is_4d=True,
                    verify=False,
                    verbosity=verbosity,
                )

    # phi maps
    if out_phi and out_phi.active:
        datalabel = "vacuum" if isvacuum else "water"
        datafile = out_phi.get_attribute("file")
        phi_format = out_phi.get_attribute("format")
        phi_bin_precision = (
            np.float32 if out_phi.get_attribute("precision") == "single" else np.float64
        )

        if out_phi.get_attribute("media") == "both":
            datafile = get_phase_specific_filename(datafile, datalabel)

        datamap = ctx.phimap_in_vacuum if isvacuum else ctx.phimap_in_water
        phimap_title = construct_phi_map_title(
            srfexp, m, sigma, datalabel, surface_method
        )

        if datamap is not None:
            write_cube_and_verify(
                filename=datafile,
                scale=ctx.scale,
                grid_center=ctx.grid_center,
                grid_shape=ctx.grid_shape,
                data_1d=datamap,
                content=phimap_title,
                format_type=phi_format,
                binary_precision=phi_bin_precision,
                verbosity=verbosity,
            )

    # zeta-phi (water only)
    if is_zeta_on and (not isvacuum) and (out_zphi and out_zphi.active):
        datafile = out_zphi.get_attribute("file")
        # print(
        #     ctx.grid_center,
        #     ctx.zeta_surf_grid_coords,
        #     ctx.zeta_surf_grid_indices,
        #     ctx.num_zeta_surf_grid_coords,
        # )
        wrt.write_zphi(
            zphi_filename=datafile,
            grid_shape=ctx.grid_shape,
            grid_scale=ctx.scale,
            grid_center=ctx.grid_center,
            surf_grid_indices=ctx.zeta_surf_grid_indices,
            num_surf_grid_points=ctx.num_zeta_surf_grid_coords,
            phimap=ctx.phimap_in_water,
            surface_distance=inp.get_param_value("zeta_distance"),
            context_type="all_solute",
            context_label="all",
            context_num_atoms=getattr(ctx, "num_atoms", 0),
        )
        vprint(
            DEBUG,
            verbosity,
            f"Zeta-phi written to {datafile}. Verification not implemented.",
        )


def log_epsilon_map_shapes(
    *,
    verbosity: int,
    phase_name: str,
    epsilon_map,
    epsilon_map_midpoints,
) -> None:
    """
    Logs shapes of epsilon maps (kept behavior-compatible with the original).
    """
    if verbosity <= DEBUG:
        if epsilon_map is None:
            print("self.dc.epsilon_map_1d= None")
        else:
            print(f"self.dc.epsilon_map_1d.shape= {epsilon_map.shape}")

        epsilon_map_midpoints_attr_name = f"epsilon_map_midpoints_{phase_name}_1d"
        if epsilon_map_midpoints is None:
            print(f"self.dc.{epsilon_map_midpoints_attr_name}= None")
        else:
            print(
                f"self.dc.{epsilon_map_midpoints_attr_name}.shape= {epsilon_map_midpoints.shape}"
            )
