"""relay-grid-dag: Near-field Programmable Relay Grid.

Sparse activation / selection of candidate relays for geometry-aware mutual-
information channel shaping. The near-field physics (spherical-wave LoS channels +
linear-Gaussian DAG specs) is a vendored, frozen copy of ``nearfield-dag`` under
:mod:`relay_grid_dag._nearfield`; MI is delegated to the public ``gaussian-dag``
K-recursion. This package adds the selection / activation layer on top.

Public API
----------
Physics (re-exported from the vendored engine):
    ``Scene``, ``mi``, ``multirelay_merge``, ``diamond``, ``single_link``,
    ``near_field_channel``, ``far_field_channel``, ``subcarrier_wavenumbers``,
    ``subcarrier_frequencies``, ``wideband_mi``, ``project_box``,
    ``project_min_separation``, ``viz``.
Grid + selection (this package):
    ``grid_coords``, ``build_candidate_scene``, ``subset_mi``, ``direct_only_mi``,
    ``select_greedy_mi``, ``select_exhaustive``, ``select_received_power``,
    ``select_distance``, ``swap_search``, ``random_subset_stats``,
    ``continuous_relays``, ``round_to_grid``, ``received_power_scores``.
"""
from __future__ import annotations

# --- vendored near-field physics (the gaussian-dag boundary) ---------------- #
from ._nearfield import (
    Scene, mi, multirelay_merge, diamond, single_link, ris_branch,
    near_field_channel, far_field_channel, distances,
    subcarrier_frequencies, subcarrier_wavenumbers, wideband_mi,
    af_gain, physical_channel, waterfilling_capacity,
    project_box, project_min_separation, repulsion_penalty,
    ula, movable, ArrayGeometry, DTYPE, RDTYPE, K_WAVE,
)
from ._nearfield import viz  # matplotlib-tier diagnostics (carrier_field, etc.)

# --- selection / activation layer (this package) ---------------------------- #
from .grid import grid_coords, build_candidate_scene
from .selection import (
    subset_mi, direct_only_mi, received_power_scores,
    select_received_power, select_distance, select_greedy_mi,
    select_exhaustive, swap_search, random_subset_stats,
    continuous_relays, round_to_grid,
)

# --- multi-pair interference layer (conditional MI via cmi-dag) -------------- #
from .multipair import (
    build_pair_scene, pair_rates, weighted_sum_rate, min_rate,
    select_greedy_sumrate, select_exhaustive_sumrate, received_power_pairs,
)

__version__ = "0.1.0"

__all__ = [
    # physics (vendored engine)
    "Scene", "mi", "multirelay_merge", "diamond", "single_link", "ris_branch",
    "near_field_channel", "far_field_channel", "distances",
    "subcarrier_frequencies", "subcarrier_wavenumbers", "wideband_mi",
    "af_gain", "physical_channel", "waterfilling_capacity",
    "project_box", "project_min_separation", "repulsion_penalty",
    "ula", "movable", "ArrayGeometry", "DTYPE", "RDTYPE", "K_WAVE", "viz",
    # grid + selection (this package)
    "grid_coords", "build_candidate_scene",
    "subset_mi", "direct_only_mi", "received_power_scores",
    "select_received_power", "select_distance", "select_greedy_mi",
    "select_exhaustive", "swap_search", "random_subset_stats",
    "continuous_relays", "round_to_grid",
    # multi-pair interference (this package, via cmi-dag)
    "build_pair_scene", "pair_rates", "weighted_sum_rate", "min_rate",
    "select_greedy_sumrate", "select_exhaustive_sumrate", "received_power_pairs",
]
