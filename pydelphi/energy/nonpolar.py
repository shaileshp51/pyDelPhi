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

# import numpy as np
# from numba import njit, prange
# import math
#
# # Define data types for Numba
# delphi_float = np.float64
# delphi_int = np.int64
# delphi_bool = np.bool_
#
# # --- Constants ---
# _EPSILON = 1e-12  # A small number to prevent division by zero or log of zero
# _PROB_THRESHOLD = 0.99  # Threshold for probability calculation in Gaussian overlap
#
# # Compile-time maximum order for inclusion-exclusion series
# # This defines the maximum depth of the DFS and sizes some arrays.
# # Even if max_overlaps_per_atom is higher, runtime order will be capped by this.
# _COMPILE_TIME_MAX_ORDER = 10
#
# # Heuristics for pre-allocating buffer sizes
# # These are generous upper bounds based on expected worst-case scenarios.
# # Adjusting these can impact memory usage and potentially very large system performance.
# _MAX_TOTAL_OVERLAPS_HEURISTIC = 0  # Placeholder, will be calculated based on num_atoms
# _MAX_RD_UPDATES = 0  # Placeholder, will be calculated
# _MAX_VOL_CONTRIBUTIONS = 0  # Placeholder, will be calculated
# _MAX_DFS_STACK_SIZE = 0  # Placeholder, will be calculated
# _MAX_COMMON_NB_LISTS = 0  # Placeholder, will be calculated
# _MAX_COMMON_NB_DATA_TOTAL = 0  # Placeholder, will be calculated
#
#
# # --- Mathematical Functions (Numba JITted) ---
#
# @njit(delphi_float(delphi_float, delphi_float), fastmath=True, cache=True)
# def _calculate_pairwise_overlap(r_ij, r_i_sq, r_j_sq):
#     """
#     Calculates the Gaussian overlap integral between two spheres.
#     Eq. 2.1 in "Numerical Calculation of Solvation Energy Using the Gaussian Model"
#     by Gilson et al., J. Comput. Chem. 1993, 14, 217-226.
#     """
#     exp_term = np.exp(-r_ij * r_ij / (r_i_sq + r_j_sq))
#     return (r_i_sq + r_j_sq) ** 1.5 * exp_term
#
#
# @njit(delphi_float(delphi_float, delphi_float, delphi_float), fastmath=True, cache=True)
# def _calculate_radii_derivative(r_ij, r_i_sq, r_j_sq):
#     """
#     Calculates the derivative of the Gaussian overlap integral with respect to radius.
#     Eq. 2.4 (for d/dr_i) in "Numerical Calculation of Solvation Energy Using the Gaussian Model"
#     by Gilson et al., J. Comput. Chem. 1993, 14, 217-226.
#     """
#     denominator = r_i_sq + r_j_sq
#     exp_term = np.exp(-r_ij * r_ij / denominator)
#     return (1.5 * (denominator) ** 0.5 - r_ij * r_ij * r_i_sq / (denominator ** 1.5)) * exp_term
#
#
# @njit(delphi_float(delphi_float), fastmath=True, cache=True)
# def _calculate_overlap_integral(s_val):
#     """
#     Calculates the integral of 1 - exp(-s^2) for the nonpolar term.
#     This corresponds to the term in Eq. 1.
#     """
#     if s_val < _EPSILON:
#         return 0.0
#     return s_val * np.sqrt(np.pi) / 2.0 * math.erf(s_val) + 0.5 * np.exp(-s_val * s_val) - 0.5
#
#
# @njit(delphi_float(delphi_float), fastmath=True, cache=True)
# def _calculate_d_integral_dr(s_val):
#     """
#     Calculates the derivative of the integral term with respect to radius (s_val).
#     This corresponds to the derivative of the term in Eq. 1.
#     """
#     return np.sqrt(np.pi) / 2.0 * math.erf(s_val) + s_val * np.exp(-s_val * s_val)
#
#
# # --- Core Numba JITted Functions ---
#
# @njit(delphi_int[:](delphi_int, delphi_int[:], delphi_int[:], delphi_int, delphi_int[:], delphi_int[:]), cache=True)
# def find_common_neighbors_of_list_in_place(
#         current_tuple_len, current_tuple_indices,
#         neighbor_list_indices, neighbor_list_ptr,
#         common_neighbors_buffer, common_neighbors_count_ptr
# ):
#     """
#     Finds common neighbors among a list of atoms without using Python sets.
#     The result is written into a pre-allocated buffer.
#
#     Parameters:
#     ----------
#     current_tuple_len : int
#         The number of atoms in the current_tuple_indices.
#     current_tuple_indices : np.ndarray (delphi_int)
#         Array of atom indices in the current overlap tuple.
#     neighbor_list_indices : np.ndarray (delphi_int)
#         Flat array of all atom neighbors.
#     neighbor_list_ptr : np.ndarray (delphi_int)
#         Pointer array for neighbor_list_indices, where neighbor_list_ptr[i]
#         is the start index for atom i's neighbors and neighbor_list_ptr[i+1]-1 is the end.
#     common_neighbors_buffer : np.ndarray (delphi_int)
#         Pre-allocated buffer to write the common neighbors into.
#     common_neighbors_count_ptr : np.ndarray (delphi_int)
#         A 1-element array acting as a pointer to the current count of common neighbors
#         written into the buffer. This is modified in-place.
#
#     Returns:
#     -------
#     The starting index of the common neighbors in the buffer.
#     """
#     if current_tuple_len == 0:
#         # This case should ideally not be reached if called correctly,
#         # but defensively return 0.
#         common_neighbors_count_ptr[0] = 0
#         return 0
#
#     atom0_idx = current_tuple_indices[0]
#     start_idx0 = neighbor_list_ptr[atom0_idx]
#     end_idx0 = neighbor_list_ptr[atom0_idx + 1]
#
#     current_common_neighbors_start_idx = common_neighbors_count_ptr[0]
#     local_count = 0
#
#     for i in range(start_idx0, end_idx0):
#         candidate_neighbor = neighbor_list_indices[i]
#         is_common = True
#         for j in range(1, current_tuple_len):
#             current_atom_idx = current_tuple_indices[j]
#             found_candidate_in_current_atom_neighbors = False
#             start_idx_j = neighbor_list_ptr[current_atom_idx]
#             end_idx_j = neighbor_list_ptr[current_atom_idx + 1]
#             for k in range(start_idx_j, end_idx_j):
#                 if neighbor_list_indices[k] == candidate_neighbor:
#                     found_candidate_in_current_atom_neighbors = True
#                     break
#             if not found_candidate_in_current_atom_neighbors:
#                 is_common = False
#                 break
#
#         # Also ensure candidate_neighbor is not already in current_tuple_indices
#         # to avoid self-referencing or duplicating tuple members.
#         # This is crucial for correctly building unique tuples.
#         for j in range(current_tuple_len):
#             if current_tuple_indices[j] == candidate_neighbor:
#                 is_common = False
#                 break
#
#         if is_common:
#             # Check for duplicates in common_neighbors_buffer to keep it unique
#             # (only necessary if multiple paths could lead to same common neighbor,
#             # which is not typically the case here for DFS building, but good practice).
#             # Given the DFS structure, the found candidate neighbors from atom0's list
#             # are generally distinct from each other.
#             common_neighbors_buffer[current_common_neighbors_start_idx + local_count] = candidate_neighbor
#             local_count += 1
#             if current_common_neighbors_start_idx + local_count >= len(common_neighbors_buffer):
#                 # This indicates an overflow in the pre-allocated buffer.
#                 # In production, this should trigger an error or larger heuristic.
#                 break  # or raise ValueError("Common neighbors buffer overflow!")
#
#     common_neighbors_count_ptr[0] += local_count
#     return current_common_neighbors_start_idx, local_count
#
#
# @njit(
#     (delphi_int[:], delphi_int[:], delphi_int[:], delphi_int, delphi_int[:],
#      delphi_int[:], delphi_int[:], delphi_int[:], delphi_float[:],
#      delphi_float[:], delphi_float[:], delphi_float[:], delphi_int[:],
#      delphi_int[:], delphi_int[:], delphi_int[:], delphi_int[:],
#      delphi_int[:], delphi_float[:], delphi_float[:], delphi_int[:],
#      delphi_int[:], delphi_int[:], delphi_int[:], delphi_float[:],
#      delphi_float[:]),
#     cache=True
# )
# def _generate_overlap_tuples_iterative(
#         start_atom_idx,
#         dfs_stack_atom_idx, dfs_stack_parent_idx, dfs_stack_ptr,
#         dfs_stack_common_nb_start_idx, dfs_stack_common_nb_count, dfs_stack_common_nb_curr_ptr,
#         overlap_region_flat_array, overlap_region_info_array,
#         atom_coords, radii, s_values_sq, inv_s_sq,
#         neighbor_list_indices, neighbor_list_ptr,
#         is_in_contact_matrix,
#         common_neighbors_buffer, common_neighbors_metadata, flat_common_neighbors_ptr,
#         radii_derivative_updates_flat, radii_derivative_updates_values,
#         radii_derivative_updates_overall_count,
#         total_volume_contributions_indices, total_volume_contributions_values,
#         total_volume_contributions_count
# ):
#     """
#     Generates unique overlap tuples using an iterative Depth-First Search (DFS) approach.
#     This function populates pre-allocated buffers for overlap regions, radii derivative updates,
#     and volume contributions. It avoids Python sets and recursive calls for Numba efficiency.
#     """
#     max_order_runtime = _COMPILE_TIME_MAX_ORDER  # This is set based on min(max_overlaps_per_atom, _COMPILE_TIME_MAX_ORDER)
#     # in the calling function, so use the constant here.
#
#     # Initialize DFS stack for the root atom
#     dfs_stack_atom_idx[0] = start_atom_idx
#     dfs_stack_parent_idx[0] = -1  # No parent for the root
#     dfs_stack_common_nb_start_idx[0] = 0  # Not used for root, but needs a value
#     dfs_stack_common_nb_count[0] = 0  # Not used for root
#     dfs_stack_common_nb_curr_ptr[0] = 0  # Not used for root
#     dfs_stack_ptr[0] = 1  # Stack has one element
#
#     while dfs_stack_ptr[0] > 0:
#         # Pop an item from stack
#         dfs_stack_ptr[0] -= 1
#         current_stack_idx = dfs_stack_ptr[0]
#
#         current_atom_idx = dfs_stack_atom_idx[current_stack_idx]
#         parent_stack_idx = dfs_stack_parent_idx[current_stack_idx]
#
#         # Reconstruct current tuple path
#         current_tuple_indices_temp = np.empty(max_order_runtime, dtype=delphi_int)
#         current_tuple_len = 0
#         temp_s_idx = current_stack_idx
#         while temp_s_idx != -1:
#             current_tuple_indices_temp[current_tuple_len] = dfs_stack_atom_idx[temp_s_idx]
#             current_tuple_len += 1
#             temp_s_idx = dfs_stack_parent_idx[temp_s_idx]
#
#         # Reverse the tuple to get correct order (root first)
#         current_tuple_indices = current_tuple_indices_temp[:current_tuple_len][::-1]
#
#         # --- Process current tuple ---
#         if current_tuple_len > 0:  # Should always be true after popping.
#             overlap_sign = (-1) ** (current_tuple_len + 1)  # Plus 1 for total overlap (union)
#
#             # Calculate combined S value for the current tuple
#             sum_s_sq_inv = 0.0
#             for i in range(current_tuple_len):
#                 sum_s_sq_inv += inv_s_sq[current_tuple_indices[i]]
#
#             # Prevent division by zero or extremely small numbers
#             if sum_s_sq_inv < _EPSILON:
#                 continue  # Skip this tuple if s_sq_inv is effectively zero
#
#             combined_s_sq = 1.0 / sum_s_sq_inv
#             combined_s = np.sqrt(combined_s_sq)
#
#             # --- Volume Contribution ---
#             vol_contrib = overlap_sign * _calculate_overlap_integral(combined_s) * combined_s_sq ** 1.5
#             _add_volume_contribution(
#                 vol_contrib, current_tuple_indices, current_tuple_len,
#                 total_volume_contributions_indices, total_volume_contributions_values,
#                 total_volume_contributions_count
#             )
#
#             # --- Radii Derivative Updates ---
#             for i in range(current_tuple_len):
#                 atom_idx = current_tuple_indices[i]
#                 d_s_d_r = combined_s_sq * inv_s_sq[atom_idx] * (1.0 / np.sqrt(s_values_sq[atom_idx]))
#
#                 # Check for NaNs or Infs from d_s_d_r calculation
#                 if not np.isfinite(d_s_d_r):
#                     continue  # Skip this derivative if d_s_d_r is not a finite number
#
#                 dr_contrib = overlap_sign * (
#                         _calculate_d_integral_dr(combined_s) * d_s_d_r * combined_s_sq ** 1.5 +
#                         1.5 * _calculate_overlap_integral(combined_s) * combined_s_sq ** 0.5 * d_s_d_r
#                 )
#                 _add_radii_derivative_update(
#                     atom_idx, dr_contrib, radii_derivative_updates_flat,
#                     radii_derivative_updates_values, radii_derivative_updates_overall_count
#                 )
#
#         # --- Expand tuple via common neighbors ---
#         if current_tuple_len < max_order_runtime:
#             common_neighbors_start_idx, num_common_neighbors = find_common_neighbors_of_list_in_place(
#                 current_tuple_len, current_tuple_indices,
#                 neighbor_list_indices, neighbor_list_ptr,
#                 common_neighbors_buffer, flat_common_neighbors_ptr
#             )
#
#             # Store metadata for common neighbors for this tuple's context
#             meta_ptr = common_neighbors_metadata[0]  # Use a dedicated pointer for metadata array
#             common_neighbors_metadata[meta_ptr, 0] = common_neighbors_start_idx
#             common_neighbors_metadata[meta_ptr, 1] = num_common_neighbors
#             common_neighbors_metadata[0] += 1  # Increment global metadata pointer
#
#             # Now, iterate through the common neighbors and push them to the stack
#             for i in range(num_common_neighbors):
#                 neighbor_atom_idx = common_neighbors_buffer[common_neighbors_start_idx + i]
#
#                 # Ensure neighbor_atom_idx has not been processed in this path already
#                 # (handled implicitly by find_common_neighbors_of_list_in_place now)
#                 # Ensure it is not smaller than any atom index in current_tuple to avoid duplicates
#                 # due to different ordering. This is the canonical ordering optimization.
#                 # The DFS explores in increasing order of atom index to prevent duplicate tuples.
#                 if current_tuple_len > 0 and neighbor_atom_idx <= current_tuple_indices[current_tuple_len - 1]:
#                     continue
#
#                 if dfs_stack_ptr[0] >= len(dfs_stack_atom_idx):
#                     # Stack overflow. Increase _MAX_DFS_STACK_SIZE heuristic.
#                     # print("DFS Stack Overflow!")
#                     break  # Stop expanding for this branch if stack is full.
#
#                 dfs_stack_atom_idx[dfs_stack_ptr[0]] = neighbor_atom_idx
#                 dfs_stack_parent_idx[dfs_stack_ptr[0]] = current_stack_idx  # Link to parent
#                 # No need to store common neighbor info for child, it will recompute
#                 dfs_stack_ptr[0] += 1
#
#
# @njit(
#     (delphi_float, delphi_int[:], delphi_int,
#      delphi_int[:, :], delphi_float[:], delphi_int[:]),
#     cache=True
# )
# def _add_volume_contribution(
#         contribution_value,
#         tuple_indices, tuple_len,
#         total_volume_contributions_indices,
#         total_volume_contributions_values,
#         total_volume_contributions_count
# ):
#     """Adds a volume contribution to the pre-allocated buffers."""
#     current_count = total_volume_contributions_count[0]
#     if current_count >= len(total_volume_contributions_values):
#         # Buffer overflow, print warning in dev, perhaps exit in prod
#         # print("Volume contributions buffer overflow!")
#         return
#
#     # Store tuple length and indices (assuming max tuple len is small)
#     total_volume_contributions_indices[current_count, 0] = tuple_len
#     for i in range(tuple_len):
#         total_volume_contributions_indices[current_count, i + 1] = tuple_indices[i]
#
#     total_volume_contributions_values[current_count] = contribution_value
#     total_volume_contributions_count[0] += 1
#
#
# @njit(
#     (delphi_int, delphi_float,
#      delphi_int[:], delphi_float[:], delphi_int[:]),
#     cache=True
# )
# def _add_radii_derivative_update(
#         atom_idx,
#         derivative_value,
#         radii_derivative_updates_flat,
#         radii_derivative_updates_values,
#         radii_derivative_updates_overall_count
# ):
#     """Adds a radii derivative update to the pre-allocated buffers."""
#     current_count = radii_derivative_updates_overall_count[0]
#     if current_count >= len(radii_derivative_updates_flat):
#         # Buffer overflow, print warning in dev, perhaps exit in prod
#         # print("Radii derivative updates buffer overflow!")
#         return
#
#     radii_derivative_updates_flat[current_count] = atom_idx
#     radii_derivative_updates_values[current_count] = derivative_value
#     radii_derivative_updates_overall_count[0] += 1
#
#
# @njit(
#     (delphi_int, delphi_float[:], delphi_float[:], delphi_float[:, :],
#      delphi_float, delphi_int, delphi_int, delphi_int[:], delphi_int[:],
#      delphi_bool[:, :], delphi_int[:], delphi_int[:]),
#     cache=True
# )
# def compute_atom_neighbors(
#         num_atoms, radii, atom_coords,
#         probe_radius,
#         neighbor_list_indices_buffer, neighbor_list_ptr_buffer,
#         is_in_contact_matrix,
#         contact_pairs_flat_array, contact_pairs_count_ptr
# ):
#     """
#     Computes direct neighbors for each atom based on their contact distance.
#     Populates `neighbor_list_indices_buffer` and `neighbor_list_ptr_buffer`.
#     Also populates `is_in_contact_matrix` and `contact_pairs_flat_array`.
#     """
#     current_neighbor_idx_in_flat = 0
#     current_contact_pair_idx = 0
#
#     neighbor_list_ptr_buffer[0] = 0  # First atom's neighbors start at index 0
#
#     for i in prange(num_atoms):  # Use prange for parallel loop
#         # Calculate contact distance for atom i
#         r_i_contact = radii[i] + probe_radius
#
#         for j in range(i + 1, num_atoms):  # Only consider j > i to avoid duplicates and self
#             # Calculate contact distance for atom j
#             r_j_contact = radii[j] + probe_radius
#
#             # Calculate squared distance between atom i and j
#             r_ij_sq = (atom_coords[i, 0] - atom_coords[j, 0]) ** 2 + \
#                       (atom_coords[i, 1] - atom_coords[j, 1]) ** 2 + \
#                       (atom_coords[i, 2] - atom_coords[j, 2]) ** 2
#
#             r_ij = np.sqrt(r_ij_sq)
#
#             # Check if atoms are in contact
#             if r_ij <= (r_i_contact + r_j_contact):
#                 is_in_contact_matrix[i, j] = True
#                 is_in_contact_matrix[j, i] = True  # Symmetric
#
#                 # Add to contact pairs flat array
#                 if current_contact_pair_idx < len(contact_pairs_flat_array) // 2:
#                     contact_pairs_flat_array[current_contact_pair_idx * 2] = i
#                     contact_pairs_flat_array[current_contact_pair_idx * 2 + 1] = j
#                     current_contact_pair_idx += 1
#
#                 # Add to neighbor lists (symmetric)
#                 if current_neighbor_idx_in_flat < len(neighbor_list_indices_buffer):
#                     neighbor_list_indices_buffer[current_neighbor_idx_in_flat] = j
#                     current_neighbor_idx_in_flat += 1
#                 else:
#                     # Overflow in neighbor_list_indices_buffer
#                     # In a robust system, you'd raise an error or resize
#                     pass  # Continue to avoid crashing, but this indicates a problem
#
#         # Update pointer for atom i+1 (where its neighbors will start)
#         # Sort neighbors for this atom for canonical common neighbor finding later
#         start_for_i = neighbor_list_ptr_buffer[i]
#         end_for_i = current_neighbor_idx_in_flat
#         neighbor_list_indices_buffer[start_for_i:end_for_i].sort()
#
#         if i + 1 < num_atoms:  # Ensure we don't go out of bounds for the ptr buffer
#             neighbor_list_ptr_buffer[i + 1] = current_neighbor_idx_in_flat
#
#     # Store the final count of unique contact pairs
#     contact_pairs_count_ptr[0] = current_contact_pair_idx
#
#
# @njit(
#     (delphi_int, delphi_int, delphi_int),  # num_atoms, max_order_runtime, max_num_neighbors_per_atom
#     cache=True
# )
# def _initialize_data_structures(num_atoms, max_order_runtime, max_num_neighbors_per_atom):
#     """
#     Initializes and returns all pre-allocated NumPy arrays needed for calculations.
#     This function calculates heuristic sizes based on num_atoms and max_order_runtime.
#     """
#     # Heuristics for buffer sizes (can be tuned)
#     # These are generous estimates to minimize reallocations and overflows.
#     global _MAX_TOTAL_OVERLAPS_HEURISTIC
#     global _MAX_RD_UPDATES
#     global _MAX_VOL_CONTRIBUTIONS
#     global _MAX_DFS_STACK_SIZE
#     global _MAX_COMMON_NB_LISTS
#     global _MAX_COMMON_NB_DATA_TOTAL
#
#     _MAX_TOTAL_OVERLAPS_HEURISTIC = int(
#         num_atoms * (max_order_runtime + 1) * 2)  # A bit arbitrary, but a safe upper bound
#     _MAX_RD_UPDATES = int(
#         _MAX_TOTAL_OVERLAPS_HEURISTIC * max_order_runtime * 2)  # Each overlap contributes up to max_order derivatives
#     _MAX_VOL_CONTRIBUTIONS = int(_MAX_TOTAL_OVERLAPS_HEURISTIC)  # One volume contribution per unique overlap
#     _MAX_DFS_STACK_SIZE = int(num_atoms * max_order_runtime * 2)  # Max depth * branching factor * cushion
#     _MAX_COMMON_NB_LISTS = int(_MAX_DFS_STACK_SIZE * 2)  # Each stack push can generate common neighbors
#     _MAX_COMMON_NB_DATA_TOTAL = int(
#         _MAX_COMMON_NB_LISTS * max_num_neighbors_per_atom * 2)  # Total size for flat common neighbors data
#
#     # --- Buffers for DFS and Overlap Tuple Generation ---
#     # DFS Stack: Stores current path and parent info
#     dfs_stack_atom_idx = np.empty(_MAX_DFS_STACK_SIZE, dtype=delphi_int)
#     dfs_stack_parent_idx = np.full(_MAX_DFS_STACK_SIZE, -1, dtype=delphi_int)
#     # Stack pointers for common neighbor data within the stack frame.
#     # Not strictly needed if common neighbors are recomputed on demand from current_tuple,
#     # but could be useful if wanting to cache common neighbor lists.
#     dfs_stack_common_nb_start_idx = np.empty(_MAX_DFS_STACK_SIZE, dtype=delphi_int)
#     dfs_stack_common_nb_count = np.empty(_MAX_DFS_STACK_SIZE, dtype=delphi_int)
#     dfs_stack_common_nb_curr_ptr = np.empty(_MAX_DFS_STACK_SIZE, dtype=delphi_int)  # Iterator for common neighbors
#     dfs_stack_ptr = np.array([0], dtype=delphi_int)  # Pointer for the stack top
#
#     # Common Neighbors Buffer (for `find_common_neighbors_of_list_in_place`)
#     # This is a flat array to store actual common neighbor indices for current tuple expansion.
#     common_neighbors_buffer = np.empty(_MAX_COMMON_NB_DATA_TOTAL, dtype=delphi_int)
#     # Metadata for common neighbors (stores start_idx and count for each list)
#     # common_neighbors_metadata stores [start_index, count] for common neighbor lists
#     common_neighbors_metadata = np.empty((_MAX_COMMON_NB_LISTS, 2), dtype=delphi_int)
#     flat_common_neighbors_ptr = np.array([0], dtype=delphi_int)  # Pointer for common_neighbors_buffer
#
#     # --- Buffers for Results (Volume and Radii Derivatives) ---
#     # Radii Derivative Updates: Store (atom_idx, value) pairs
#     radii_derivative_updates_flat = np.empty(_MAX_RD_UPDATES, dtype=delphi_int)
#     radii_derivative_updates_values = np.empty(_MAX_RD_UPDATES, dtype=delphi_float)
#     radii_derivative_updates_overall_count = np.array([0], dtype=delphi_int)
#
#     # Total Volume Contributions: Store (tuple_indices, value) pairs
#     # The `+1` is for storing the tuple_len at index 0. Max tuple len is max_order_runtime.
#     total_volume_contributions_indices = np.empty((_MAX_VOL_CONTRIBUTIONS, max_order_runtime + 1), dtype=delphi_int)
#     total_volume_contributions_values = np.empty(_MAX_VOL_CONTRIBUTIONS, dtype=delphi_float)
#     total_volume_contributions_count = np.array([0], dtype=delphi_int)
#
#     # --- Buffers for Atom Neighbors and Contact Pairs ---
#     # Max possible neighbors is num_atoms-1 for each atom, sum over all atoms.
#     # For neighbor_list_indices_buffer, a heuristic is `num_atoms * max_num_neighbors_per_atom`
#     neighbor_list_indices = np.empty(num_atoms * max_num_neighbors_per_atom * 2, dtype=delphi_int)  # *2 for cushion
#     neighbor_list_ptr = np.empty(num_atoms + 1, dtype=delphi_int)  # Stores start/end indices for each atom's neighbors
#
#     # Contact matrix for O(1) lookup: is_in_contact_matrix[i,j] = True if i and j are in contact
#     is_in_contact_matrix = np.full((num_atoms, num_atoms), False, dtype=delphi_bool)
#
#     # Flat array to store unique contact pairs (for debugging/verification if needed)
#     # Max unique pairs: num_atoms * (num_atoms - 1) / 2
#     contact_pairs_flat_array = np.empty(num_atoms * (num_atoms - 1),
#                                         dtype=delphi_int)  # *2 because each pair is 2 indices
#     contact_pairs_count_ptr = np.array([0], dtype=delphi_int)
#
#     return (
#         dfs_stack_atom_idx, dfs_stack_parent_idx, dfs_stack_ptr,
#         dfs_stack_common_nb_start_idx, dfs_stack_common_nb_count, dfs_stack_common_nb_curr_ptr,
#         common_neighbors_buffer, common_neighbors_metadata, flat_common_neighbors_ptr,
#         radii_derivative_updates_flat, radii_derivative_updates_values, radii_derivative_updates_overall_count,
#         total_volume_contributions_indices, total_volume_contributions_values, total_volume_contributions_count,
#         neighbor_list_indices, neighbor_list_ptr, is_in_contact_matrix,
#         contact_pairs_flat_array, contact_pairs_count_ptr
#     )
#
#
# # --- Main Public Function ---
#
# @njit(
#     (delphi_float[:], delphi_float[:, :], delphi_float,
#      delphi_float, delphi_float, delphi_float, delphi_int, delphi_int),
#     parallel=False, fastmath=True, cache=True
# )
# def compute_nonpolar_energy(
#         radii,
#         atom_coords,
#         probe_radius,
#         gamma,
#         s_cut_prob,
#         charge_density,
#         # This seems like an unused parameter for nonpolar energy. Kept for API compatibility if needed.
#         max_overlaps_per_atom=5,  # Runtime upper bound for inclusion-exclusion order
#         max_num_neighbors_per_atom=20  # Heuristic for neighbor list pre-allocation
# ):
#     """
#     Computes the nonpolar solvation energy and its derivatives with respect to atomic radii
#     using a Numba-optimized Gaussian overlap model.
#
#     Parameters:
#     -----------
#     radii : np.ndarray (delphi_float)
#         Array of atomic radii (in Angstroms).
#     atom_coords : np.ndarray (delphi_float, shape=(num_atoms, 3))
#         Array of atomic coordinates (in Angstroms).
#     probe_radius : delphi_float
#         Radius of the solvent probe (in Angstroms).
#     gamma : delphi_float
#         Surface tension constant (in units of energy/area, e.g., kcal/mol/A^2).
#     s_cut_prob : delphi_float
#         Probability threshold for defining the effective Gaussian radius 's'.
#         Determines how broad the Gaussian is (e.g., 0.99 for 99% probability).
#     charge_density : delphi_float
#         Placeholder parameter, typically for polar energy calculations. Not used here.
#     max_overlaps_per_atom : delphi_int, optional
#         Maximum number of overlapping neighbors an atom can have considered for calculation.
#         This sets the runtime order limit for the inclusion-exclusion series. Defaults to 5.
#     max_num_neighbors_per_atom : delphi_int, optional
#         Heuristic for pre-allocating neighbor lists. Should be an overestimate of
#         the maximum number of neighbors any single atom might have. Defaults to 20.
#
#     Returns:
#     --------
#     total_nonpolar_energy : delphi_float
#         The calculated nonpolar solvation energy.
#     radii_derivatives : np.ndarray (delphi_float)
#         Array of derivatives of the nonpolar energy with respect to each atom's radius.
#     """
#     num_atoms = len(radii)
#
#     # --- Pre-calculate s values and 1/s^2 ---
#     # Derived from P(r < R) = s_cut_prob for a Gaussian, R = radius + probe_radius
#     # R = s * sqrt(2 * ln(1 / (1 - s_cut_prob)))
#     # s = R / sqrt(2 * ln(1 / (1 - s_cut_prob)))
#     # constant_for_s = sqrt(2 * ln(1 / (1 - s_cut_prob)))
#     # If s_cut_prob is 0.99, 1 - s_cut_prob = 0.01, 1 / (1 - s_cut_prob) = 100
#     # ln(100) approx 4.605, sqrt(2 * 4.605) approx sqrt(9.21) approx 3.035
#     constant_for_s = np.sqrt(2.0 * np.log(1.0 / (1.0 - s_cut_prob + _EPSILON)))  # Add epsilon for robustness
#
#     s_values = (radii + probe_radius) / constant_for_s
#     s_values_sq = s_values * s_values
#
#     # Handle cases where s_values might be zero (e.g., if radius + probe_radius is zero)
#     inv_s_sq = np.zeros_like(s_values_sq)
#     non_zero_s_sq_mask = s_values_sq > _EPSILON
#     inv_s_sq[non_zero_s_sq_mask] = 1.0 / s_values_sq[non_zero_s_sq_mask]
#
#     # --- Determine runtime max_order ---
#     # The actual max_order used for calculations will be the minimum of the
#     # compile-time constant and the user-specified max_overlaps_per_atom.
#     max_order_runtime = min(_COMPILE_TIME_MAX_ORDER, max_overlaps_per_atom)
#     if max_order_runtime < 1:  # Must be at least 1 for single atom contributions
#         max_order_runtime = 1
#
#     # --- Initialize all data structures ---
#     (
#         dfs_stack_atom_idx, dfs_stack_parent_idx, dfs_stack_ptr,
#         dfs_stack_common_nb_start_idx, dfs_stack_common_nb_count, dfs_stack_common_nb_curr_ptr,
#         common_neighbors_buffer, common_neighbors_metadata, flat_common_neighbors_ptr,
#         radii_derivative_updates_flat, radii_derivative_updates_values, radii_derivative_updates_overall_count,
#         total_volume_contributions_indices, total_volume_contributions_values, total_volume_contributions_count,
#         neighbor_list_indices, neighbor_list_ptr, is_in_contact_matrix,
#         contact_pairs_flat_array, contact_pairs_count_ptr
#     ) = _initialize_data_structures(num_atoms, max_order_runtime, max_num_neighbors_per_atom)
#
#     # --- Pre-compute atom neighbors ---
#     # This step populates neighbor_list_indices, neighbor_list_ptr, and is_in_contact_matrix
#     compute_atom_neighbors(
#         num_atoms, radii, atom_coords, probe_radius,
#         neighbor_list_indices, neighbor_list_ptr,
#         is_in_contact_matrix, contact_pairs_flat_array, contact_pairs_count_ptr
#     )
#
#     # --- Main Loop: Generate Overlap Tuples for each atom as a root ---
#     # Each atom starts its own DFS traversal.
#     # The DFS is designed to ensure canonical ordering of tuples (indices always increasing)
#     # to avoid processing the same tuple multiple times.
#     for i in prange(num_atoms):  # Can be parallelized per atom (if DFS calls are independent)
#         # Reset DFS specific pointers for each root atom
#         dfs_stack_ptr[0] = 0
#         flat_common_neighbors_ptr[0] = 0
#         common_neighbors_metadata[0, 0] = 0  # Also reset the metadata pointer
#
#         _generate_overlap_tuples_iterative(
#             i,  # Current atom as root
#             dfs_stack_atom_idx, dfs_stack_parent_idx, dfs_stack_ptr,
#             dfs_stack_common_nb_start_idx, dfs_stack_common_nb_count, dfs_stack_common_nb_curr_ptr,
#             np.empty(0, dtype=delphi_int),  # overlap_region_flat_array - not used in this iteration
#             np.empty(0, dtype=delphi_float),  # overlap_region_info_array - not used in this iteration
#             atom_coords, radii, s_values_sq, inv_s_sq,
#             neighbor_list_indices, neighbor_list_ptr,
#             is_in_contact_matrix,
#             common_neighbors_buffer, common_neighbors_metadata, flat_common_neighbors_ptr,
#             radii_derivative_updates_flat, radii_derivative_updates_values,
#             radii_derivative_updates_overall_count,
#             total_volume_contributions_indices, total_volume_contributions_values,
#             total_volume_contributions_count
#         )
#
#     # --- Aggregate Results ---
#     total_nonpolar_energy = 0.0
#     for i in range(total_volume_contributions_count[0]):
#         total_nonpolar_energy += total_volume_contributions_values[i]
#
#     total_nonpolar_energy *= gamma
#
#     radii_derivatives = np.zeros(num_atoms, dtype=delphi_float)
#     for i in range(radii_derivative_updates_overall_count[0]):
#         atom_idx = radii_derivative_updates_flat[i]
#         value = radii_derivative_updates_values[i]
#         radii_derivatives[atom_idx] += value
#
#     radii_derivatives *= gamma
#
#     return total_nonpolar_energy, radii_derivatives
#
#
# # Example Usage (for testing purposes, not part of the core module)
# if __name__ == '__main__':
#     # Sample data
#     test_radii = np.array([1.5, 1.7, 1.8], dtype=delphi_float)
#     test_coords = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [2.5, 0.0, 0.0]
#     ], dtype=delphi_float)
#     test_probe_radius = 1.4
#     test_gamma = 0.005  # kcal/mol/A^2
#     test_s_cut_prob = 0.99
#     test_charge_density = 0.0  # Not used for nonpolar
#
#     # Run the computation
#     energy, derivatives = compute_nonpolar_energy(
#         test_radii, test_coords, test_probe_radius,
#         test_gamma, test_s_cut_prob, test_charge_density,
#         max_overlaps_per_atom=3,  # Example: limit to triplets
#         max_num_neighbors_per_atom=5  # Example: max 5 neighbors for any atom
#     )
#
#     print(f"Total Nonpolar Energy: {energy:.6f} kcal/mol")
#     print(f"Radii Derivatives: {derivatives} kcal/mol/A")
#
#     # Test with more atoms, ensure heuristics are sufficient
#     large_radii = np.random.rand(100) * 0.5 + 1.0  # Radii between 1.0 and 1.5
#     large_coords = np.random.rand(100, 3) * 10.0  # Coords in 10x10x10 cube
#
#     large_energy, large_derivatives = compute_nonpolar_energy(
#         large_radii, large_coords, test_probe_radius,
#         test_gamma, test_s_cut_prob, test_charge_density,
#         max_overlaps_per_atom=5,
#         max_num_neighbors_per_atom=15
#     )
#     print(f"\nLarge System Energy: {large_energy:.6f} kcal/mol")
#     # print(f"Large System Radii Derivatives (first 5): {large_derivatives[:5]} kcal/mol/A")
#
#     # Example of potential buffer overflow check (if assertions were added)
#     # You'd see warnings if _MAX_...HEURISTIC values were too small for a given input.


def calc_nonpolar_energy():
    return 0.0
