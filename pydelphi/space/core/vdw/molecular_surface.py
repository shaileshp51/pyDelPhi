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

#
# pyDelPhi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

from math import sqrt
import numpy as np

from pydelphi.constants import ATOMFIELD_CHARGE

#
# def msrf(
#     epsilon_dimension,
#     n_boundary_grid_points,
#     grid_scale,
#     grid_shape,
#     egrid,
#     index_discrete_epsilon_map_1d,
#     atndx,
#     atoms_data,
#     is_ibem_format,
#     output_filename,
#     grid_origin_parent_run,
#     num_vertices,
#     num_normals,
#     crd3d_cross_fn,
#     crd3d_sum_fn,
#     dtype_int,
#     dtype_real,
# ):
#     iv1, iv2, iv3 = 0, 0, 0
#     total_vertices, total_triangles, ntot2 = 0, 0, 0
#
#     max_vertices = n_boundary_grid_points * 2.2
#     max_triangles = 2 * max_vertices
#
#     for k in range(1, grid_shape[2] + 1):
#         for j in range(1, grid_shape[1] + 1):
#             for i in range(1, grid_shape[0] + 1):
#                 # changed from div mod, it should only serve the means
#                 egrid[i][j][k] = (
#                     index_discrete_epsilon_map_1d[i][j][k]
#                     // epsilon_dimension
#                 )
#
#     for i in range(1, grid_shape[0] + 1):
#         for j in range(1, grid_shape[1] + 1):
#             egrid[i][1][j][0] = egrid[i][1][j][1]
#             egrid[i][j][1][0] = egrid[i][j][1][2]
#
#     for k in range(2, grid_shape[2] + 1):
#         for j in range(2, grid_shape[1] + 1):
#             for i in range(2, grid_shape[0] + 1):
#                 iate = 0
#                 if egrid[i][j][k][0] > 0:
#                     iate = iate + 1
#                 if egrid[i][j][k][1] > 0:
#                     iate = iate + 1
#                 if egrid[i][j][k][2] > 0:
#                     iate = iate + 1
#                 if egrid[i - 1][j][k][0] > 0:
#                     iate = iate + 1
#                 if egrid[i][j - 1][k][1] > 0:
#                     iate = iate + 1
#                 if egrid[i][j][k - 1][2] > 0:
#                     iate = iate + 1
#
#                 if iate <= 3:
#                     egrid[i][j][k][0] = 0
#                 else:
#                     egrid[i][j][k][0] = 1
#
#     vertex_indices = np.zeros((max_triangles + 1, 3), dtype=dtype_int)
#     vertex_lines = np.zeros((max_triangles + 1, 3), dtype=dtype_int)
#
#     # ifdef VERBOSE
#     print("max_triangles,max_vertices= ", max_triangles, " ", max_vertices)
#     # endif
#
#     vdat1 = "./"
#
#     if total_vertices > max_vertices:
#         print("total_vertices = " ,  total_vertices ,  " > max_vertices = " ,  max_vertices)
#         print("increase max_vertices in msrf.f")
#         exit(0)
#
#     for ib in range(1, total_vertices + 1):
#         vertex_lines[ib] = vertex_lines[ib] / 2.0
#
#     total_triangles = total_triangles / 3
#
#     # ifdef VERBOSE
#     print("scaling vertices")
#     # endif
#
#     normal_lines = np.zeros((total_vertices + 1, 3), dtype=dtype_real)
#     vnorm2 = np.zeros((total_vertices + 1, 3), dtype=dtype_real)
#
#     # fix holes and make vertex to triangle arrays allocate hole-fixing arrays next hole-fixing variables
#     max_vertex_index = total_vertices * 2 if total_vertices < max_vertices // 2 else max_vertices
#
#     vtlen = np.zeros(max_vertex_index + 1, dtype=dtype_int)
#     vtpnt = np.zeros(max_vertex_index + 1, dtype=dtype_int)
#     vtlst = np.zeros(6 * max_vertex_index + 1, dtype=dtype_int)
#
#     fix = True if ntot2 > 0 else False
#
#     # while fix:
#     #     if ntot2 > 0:
#     #         fix = True
#     #     else:
#     #         fix = False
#
#     if total_triangles > max_triangles:
#         print("total_triangles = " ,  total_triangles ,  " > max_triangles = " ,  max_triangles)
#         print("increase max_triangles in msrf.f")
#         exit(0)
#
#     print("number of vertices = " ,  total_vertices)
#     print("number of triangles = " ,  total_triangles)
#
#     # calculate area
#     area = 0.0
#
#     for it in range(1, total_triangles + 1):
#         iv123 = vertex_indices[it]
#         v1 = vertex_lines[iv2] - vertex_lines[iv1]
#         v2 = vertex_lines[iv3] - vertex_lines[iv1]
#
#         # Vector product defined in operators_on_coordinates module
#         vxyz = crd3d_cross_fn(v1, v2)
#         vmg = sqrt(np.dot(vxyz, vxyz))
#         tar = vmg / 2.0
#         vxyz = normal_lines[iv1] + normal_lines[iv2] + normal_lines[iv3]
#         vmg = sqrt(np.dot(vxyz, vxyz))
#
#         vnorm2[iv1] = vnorm2[iv1] + (vxyz / vmg)
#         vnorm2[iv2] = vnorm2[iv2] + (vxyz / vmg)
#         vnorm2[iv3] = vnorm2[iv3] + (vxyz / vmg)
#
#         # calculate spherical triangle area if appropriate
#         ia1 = atndx[iv1]
#         ia2 = atndx[iv2]
#         ia3 = atndx[iv3]
#
#         if ia1 > 0:
#             if ia1 == ia2 and ia1 == ia3:
#                 atom1 = atoms_data[ia1]
#                 rad = atom1[ATOMFIELD_CHARGE]
#                 rad2 = rad * rad
#                 aa = crd3d_sum_fn((vertex_lines[iv2] - vertex_lines[iv1]) * (vertex_lines[iv2] - vertex_lines[iv1]))
#                 bb = crd3d_sum_fn((vertex_lines[iv3] - vertex_lines[iv2]) * (vertex_lines[iv3] - vertex_lines[iv2]))
#                 cc = crd3d_sum_fn((vertex_lines[iv1] - vertex_lines[iv3]) * (vertex_lines[iv1] - vertex_lines[iv3]))
#
#                 aa = np.acos(1.0 - aa / (2.0 * rad2))
#                 bb = np.acos(1.0 - bb / (2.0 * rad2))
#                 cc = np.acos(1.0 - cc / (2.0 * rad2))
#                 ss = (aa + bb + cc) * 0.5
#                 tne4 = sqrt(
#                     np.tan(ss * 0.5)
#                     * np.tan((ss - aa) * 0.5)
#                     * np.tan((ss - bb) * 0.5)
#                     * np.tan((ss - cc) * 0.5)
#                 )
#                 tar = 4.0 * np.atan(tne4) * rad2
#
#         area = area + tar
#
#     for i in range(1, total_vertices + 1):
#         vmg = sqrt(np.dot(vnorm2[i], vnorm2[i]))
#         vnorm2[i] = vnorm2[i] / vmg
#
#     print("MS area = " , area)
#     write_surface(
#         is_ibem_format,
#         output_filename, grid_shape, grid_origin_parent_run, grid_scale, num_vertices, num_normals,
#         total_vertices,
#         total_triangles,
#         max_vertex_index,
#         max_triangles,
#         vertex_lines,
#         triangle_index_lines,
#         normal_lines,
#     )
#
#
# def write_surface(
#     is_ibem_format: bool,
#     output_filename: str,
#     grid_shape,
#     grid_origin,
#     grid_scale: float,
#     num_vertices: int,
#     num_normals: int,
#     total_vertices: int,
#     total_triangles: int,
#     max_vertex_index: int,
#     max_triangle_index: int,
#     vertex_lines,
#     triangle_index_lines,
#     normal_lines,
# ):
#     """
#     Write surface data to a GRASP-format or IBEEM-format file.
#
#     Parameters:
#     ----------
#     is_ibem_format : bool
#         If True, writes in IBEEM format; otherwise, writes in GRASP format.
#     output_filename : str
#         Name of the file to write surface data to.
#     grid_shape : tuple of int
#         Shape of the parent grid (typically 3D), only the first dimension is used.
#     grid_origin : tuple of float
#         The origin of the grid in 3D space (x0, y0, z0).
#     grid_scale : float
#         The scaling factor from grid units to real units.
#     num_vertices : int
#         Number of vertices written in GRASP header.
#     num_normals : int
#         Number of normals written in GRASP header.
#     total_vertices : int
#         Total number of vertices in the data.
#     total_triangles : int
#         Total number of triangles in the data.
#     max_vertex_index : int
#         The maximum vertex index to write (likely = total_vertices, but allows padding).
#     max_triangle_index : int
#         The maximum triangle index to write.
#     vertex_lines : list of str
#         Lines representing vertex data (indexing starts from 1).
#     triangle_index_lines : list of str
#         Lines representing triangle indices (indexing starts from 1).
#     normal_lines : list of str
#         Lines representing vertex normal data (indexing starts from 1).
#     """
#     if not is_ibem_format:
#         with open(output_filename, "w") as file:
#             print("Writing GRASP file to", output_filename)
#             file.write("format=2\n")
#             file.write("vertices,normals,triangles\n\n")
#             file.write(f"{num_vertices:6}{num_normals:6}{grid_shape[0]:6}{grid_scale:12.6f}\n")
#             file.write(
#                 f"{grid_origin[0]:12.6f}{grid_origin[1]:12.6f}{grid_origin[2]:12.6f}\n"
#             )
#
#             print(f"Writing data for {total_vertices} vertices and {total_triangles} triangles")
#
#             for i in range(1, total_vertices + 1):
#                 file.write(vertex_lines[i] + "\n")
#             for i in range(1, total_vertices + 1):
#                 file.write(normal_lines[i] + "\n")
#             for i in range(1, max_triangle_index + 1):
#                 file.write(triangle_index_lines[i] + "\n")
#
#             print("Finished writing", output_filename)
#     else:
#         with open(output_filename, "w") as file:
#             file.write(f"{total_vertices} {total_triangles}\n")
#             for i in range(1, total_vertices + 1):
#                 file.write(vertex_lines[i] + "\n")
#             for i in range(1, total_triangles + 1):
#                 file.write(triangle_index_lines[i] + "\n")
#             for i in range(1, total_vertices + 1):
#                 file.write(normal_lines[i] + "\n")
#
#
#
# # def write_surface(
# #     ibem, filename, grid_shape, grid_origin_parent_run, scale, n_vertices, n_normals
# # ):
# #     if not ibem:
# #         # fname = "grasp.srf"
# #         surfile = open(filename, "w")
# #
# #         print("writing GRASP file to ", filename)
# #
# #         surfile.write("format=2\n")
# #         surfile.write("vertices,normals,triangles\n\n")
# #
# #         surfile.write(
# #             f"{n_vertices:6}{n_normals:6}{grid_shape[0]:6}{scale:12.6f}\n"
# #         )
# #         surfile.write(
# #             "      {0:12.6f}      {1:12.6f}      {2:12.6f}\n".format(
# #                 grid_origin_parent_run[0],
# #                 grid_origin_parent_run[1],
# #                 grid_origin_parent_run[2],
# #             )
# #         )
# #
# #         # ifdef VERBOSE
# #         print("writing data for", total_vertices, " vertices and", total_triangles, " triangles")
# #         # endif
# #
# #         for i in range(1, mxvtx + 1):
# #             surfile.write(vert[i] + "\n")
# #
# #         for i in range(1, total_vertices + 1):
# #             surfile.write(normal_lines[i] + "\n")
# #
# #         for i in range(1, max_triangle_index + 1):
# #             surfile.write(triangle_index_lines[i] + "\n")
# #
# #         surfile.close()
# #
# #         # ifdef VERBOSE
# #         print("finished writing ", filename)
# #         # endif
# #     else:
# #         surfile = open(filename, "w")
# #         surfile.write(f"{total_vertices} {total_triangles}\n")
# #
# #         for i in range(1, total_vertices + 1):
# #             surfile.write(f"{vert[i]}\n")
# #
# #         for i in range(1, total_triangles + 1):
# #             surfile.write(f"{triangle_index_lines[i]}\n")
# #
# #         for i in range(1, total_vertices + 1):
# #             surfile.write(f"{normal_lines[i]}\n")
# #
# #         surfile.close()


import numpy as np
import math


def write_surface(
    ibem,
    vtot,
    itot,
    mxvtx,
    mxtri,
    vert,
    vnorm,
    vindx,
    iGrid_nX,
    fScale,
    cOldMid,
    fname,
):
    """
    Write molecular surface to a file in either GRASP or BEM format.
    """
    if not ibem:
        filename = f"{fname}.srf"
        with open(filename, "w") as surfile:
            print(f"writing GRASP file to {filename}")
            surfile.write("format=2\n")
            surfile.write("vertices,normals,triangles\n\n")
            surfile.write(f"{vtot:6}{itot:6}{iGrid_nX:6}{fScale:12.6f}\n")
            surfile.write(f"{cOldMid[0]:12.6f}{cOldMid[1]:12.6f}{cOldMid[2]:12.6f}\n")
            for i in range(1, mxvtx + 1):
                surfile.write(" ".join(f"{x:12.6f}" for x in vert[i]) + "\n")
            for i in range(1, vtot + 1):
                surfile.write(" ".join(f"{x:12.6f}" for x in vnorm[i]) + "\n")
            for i in range(1, mxtri + 1):
                surfile.write(" ".join(str(x) for x in vindx[i]) + "\n")
        print(f"finished writing {filename}")
    else:
        filename = f"{fname}.srf"
        with open(filename, "w") as surfile:
            surfile.write(f"{vtot} {itot}\n")
            for i in range(1, vtot + 1):
                surfile.write(" ".join(f"{x:12.6f}" for x in vert[i]) + "\n")
            for i in range(1, itot + 1):
                surfile.write(" ".join(str(x) for x in vindx[i]) + "\n")
            for i in range(1, vtot + 1):
                surfile.write(" ".join(f"{x:12.6f}" for x in vnorm[i]) + "\n")


def msrf(
    iBoundNum,
    grid_shape,
    index_discrete_epsilon_map_1d,
    egrid,
    epsdim,
    vert,
    vindx,
    vnorm,
    vnorm2,
    atndx,
    sDelPhiPDB,
    fScale,
    cOldMid,
    ibem,
):
    vtot = len(vert) - 1
    itot = len(vindx) - 1
    ntot2 = 0

    mxvtx = int(iBoundNum * 2.2)
    mxtri = 2 * mxvtx

    # Normalize dielectric map
    for k in range(grid_shape[2]):
        for j in range(grid_shape[1]):
            for i in range(grid_shape[0]):
                egrid[i][j][k] = index_discrete_epsilon_map_1d[i][j][k] / epsdim

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            egrid[i][1][j][0] = egrid[i][1][j][1]
            egrid[i][j][1][0] = egrid[i][j][1][2]

    for k in range(1, grid_shape[2]):
        for j in range(1, grid_shape[1]):
            for i in range(1, grid_shape[0]):
                iate = 0
                if egrid[i][j][k][0] > 0:
                    iate += 1
                if egrid[i][j][k][1] > 0:
                    iate += 1
                if egrid[i][j][k][2] > 0:
                    iate += 1
                if egrid[i - 1][j][k][0] > 0:
                    iate += 1
                if egrid[i][j - 1][k][1] > 0:
                    iate += 1
                if egrid[i][j][k - 1][2] > 0:
                    iate += 1
                egrid[i][j][k][0] = 1 if iate > 3 else 0

    for ib in range(1, vtot + 1):
        vert[ib] = vert[ib] / 2.0

    itot = itot // 3

    imxvtx = vtot * 2 if vtot < mxvtx // 2 else mxvtx

    area = 0.0
    for it in range(1, itot + 1):
        iv1, iv2, iv3 = vindx[it]
        v1 = vert[iv2] - vert[iv1]
        v2 = vert[iv3] - vert[iv1]
        vxyz = np.cross(v1, v2)
        vmg = math.sqrt(np.dot(vxyz, vxyz))
        tar = vmg / 2.0
        vxyz = vnorm[iv1] + vnorm[iv2] + vnorm[iv3]
        vmg = math.sqrt(np.dot(vxyz, vxyz))

        if vmg != 0:
            unit_v = vxyz / vmg
            vnorm2[iv1] += unit_v
            vnorm2[iv2] += unit_v
            vnorm2[iv3] += unit_v

        ia1, ia2, ia3 = atndx[iv1], atndx[iv2], atndx[iv3]

        if ia1 > 0 and ia1 == ia2 == ia3:
            rad = sDelPhiPDB[ia1].radius
            rad2 = rad * rad
            aa = np.sum((vert[iv2] - vert[iv1]) ** 2)
            bb = np.sum((vert[iv3] - vert[iv2]) ** 2)
            cc = np.sum((vert[iv1] - vert[iv3]) ** 2)
            aa = math.acos(1.0 - aa / (2 * rad2))
            bb = math.acos(1.0 - bb / (2 * rad2))
            cc = math.acos(1.0 - cc / (2 * rad2))
            ss = 0.5 * (aa + bb + cc)
            tne4 = math.sqrt(
                math.tan(0.5 * ss)
                * math.tan(0.5 * (ss - aa))
                * math.tan(0.5 * (ss - bb))
                * math.tan(0.5 * (ss - cc))
            )
            tar = 4.0 * math.atan(tne4) * rad2

        area += tar

    for i in range(1, vtot + 1):
        norm = math.sqrt(np.dot(vnorm2[i], vnorm2[i]))
        if norm != 0:
            vnorm2[i] = vnorm2[i] / norm

    print(f"MS area = {area}")

    write_surface(
        ibem,
        vtot,
        itot,
        mxvtx,
        mxtri,
        vert,
        vnorm2,
        vindx,
        grid_shape[0],
        fScale,
        np.array([cOldMid.nX, cOldMid.nY, cOldMid.nZ]),
        fname="bem" if ibem else "grasp",
    )
