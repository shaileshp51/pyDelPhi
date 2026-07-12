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


class TrajectoryLogger:
    """
    Trajectory logger with clear separation of:
      - static_fields  : run-level invariants (printed once as commented header)
      - dynamic_fields : per-frame quantities (written as TSV columns)

    Per-frame input is a plain dict. Whatever is passed per frame and listed
    in dynamic_fields gets written per frame.
    """

    # -----------------------------
    # Defaults
    # -----------------------------
    DEFAULT_STATIC_FIELDS = (
        "platform",
        "device_identity",
        "num_threads",
        "precision",
        "num_atoms",
        "bio_model",
        "pb_approximation",
        "solver",
        "surface_method",
        "dielectric_model",
        "internal_dielectric",
        "external_dielectric",
        "solute_extrema",
        "gridbox_type",
        "boundary_condition",
        "max_linear_iteration",
        "max_rmsd",
        "max_delta_phi",
        "salt_concentration",
        "probe_radius",
        "probe_radius2",
        "ions_radii",
        "absolute_temperature",
        "scale",
    )

    DEFAULT_DYNAMIC_FIELDS = (
        "frame",
        "perfil",
        "gbmargin",
        "nx",
        "ny",
        "nz",
        "orig_x",
        "orig_y",
        "orig_z",
        "range_x",
        "range_y",
        "range_z",
        "ctr_nq_x",
        "ctr_nq_y",
        "ctr_nq_z",
        "ctr_pq_x",
        "ctr_pq_y",
        "ctr_pq_z",
        "ctr_x",
        "ctr_y",
        "ctr_z",
        "final_rms",
        "final_dphi",
        "total_iters",
        "status",
        "wall_s",
    )

    def __init__(
        self,
        fp,
        platform,
        inp,
        static_fields=None,
        dynamic_fields=None,
        stdout_fields=("frame", "final_rms", "final_dphi", "total_iters", "status"),
        float_format=".6g",
        print_stdout=True,
    ):
        self.fp = fp
        self.platform = platform
        self.inp = inp

        self.static_fields = (
            list(static_fields)
            if static_fields is not None
            else list(self.DEFAULT_STATIC_FIELDS)
        )
        self.dynamic_fields = (
            list(dynamic_fields)
            if dynamic_fields is not None
            else list(self.DEFAULT_DYNAMIC_FIELDS)
        )

        self.stdout_fields = tuple(stdout_fields)
        self.float_format = float_format
        self.print_stdout = print_stdout

        self._started = False

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def start(self, static_meta=dict()) -> None:
        """Write static header + dynamic TSV header once."""
        if self._started:
            return
        self._write_static_header(static_meta)
        self._write_dynamic_header()
        self._started = True

    # -----------------------------
    # Static header
    # -----------------------------
    def _write_static_header(self, static_meta) -> None:
        p = self.platform
        inp = self.inp

        lines = [
            "## header legend: '##' = documentation, '#' = static key=value metadata",
            "## pydelphi trajectory run",
            "## --- field abbreviations ---",
            "## ctr   : centroid",
            "## nq    : negative charge-weighted",
            "## pq    : positive charge-weighted",
            "## orig  : grid origin",
            "## range : coordinate extent (max - min)",
            "## rms   : root-mean-square residual",
            "## dphi  : max potential change per iteration",
            "## n_iter: total PB iterations",
            "## wall_s: wall-clock time (seconds)\n",
        ]

        for key in self.static_fields:
            if key == "platform":
                lines.append(f"# platform={p.active}")

            elif key == "num_threads":
                lines.append(f"# num_threads={p.names['cpu']['num_threads']}")

            elif key == "device_identity" and p.active == "cuda":
                dev_id = p.names["cuda"]["selected_id"]
                dev = p.names["cuda"]["device"][dev_id]
                lines.append(f"# device_identity={dev['device_identity']}")

            elif key == "precision":
                lines.append(f"# precision={p.precision}")

            else:
                # assume inp parameter
                try:
                    v = inp.get_param_value(key)
                except Exception:
                    continue
                lines.append(f"# {key}={v}")

        for line in lines:
            self.fp.write(line + "\n")

        for key, v in static_meta.items():
            if isinstance(v, float):
                value_str = f"{v:.6g}"
            else:
                value_str = str(v)

            self.fp.write(f"# {key}={value_str}" + "\n")

    # -----------------------------
    # Dynamic TSV header
    # -----------------------------
    def _write_dynamic_header(self) -> None:
        self.fp.write("\t".join(self.dynamic_fields) + "\n")
        self.fp.flush()

    # -----------------------------
    # Per-frame writing
    # -----------------------------
    def _fmt(self, v):
        if v is None:
            return ""
        if isinstance(v, float):
            return format(v, self.float_format)
        return str(v)

    def write_frame(self, frame_dict: dict) -> None:
        """
        frame_dict: per-frame data dict.
        Only keys listed in dynamic_fields are written (in that order).
        """
        if not self._started:
            raise RuntimeError(
                "TrajectoryLogger.start() must be called before write_frame()."
            )

        row = [self._fmt(frame_dict.get(k, "")) for k in self.dynamic_fields]
        self.fp.write("\t".join(row) + "\n")
        self.fp.flush()

        if self.print_stdout:
            parts = []
            for k in self.stdout_fields:
                if k in frame_dict:
                    v = frame_dict[k]
                    if isinstance(v, float) and k in ("final_rms", "final_dphi"):
                        parts.append(f"{k}={v:.3e}")
                    else:
                        parts.append(f"{k}={v}")
            if parts:
                print(" ".join(parts))
