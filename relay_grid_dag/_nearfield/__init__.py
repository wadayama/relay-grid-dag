"""nearfield-dag: analytic near-field (spherical-wave) MIMO channels and
linear-Gaussian DAG specs for the gaussian-dag K-recursion.

Sister library to ``radio-twin`` (ray-traced channels): nearfield-dag is the
analytic front-end. It turns antenna element geometry into differentiable
channel matrices ``H_k`` and assembles linear-Gaussian DAGs; ``gaussian-dag``
then computes the log-det mutual information and its gradient. The two libraries
meet only at ``H_k`` -- all MI/optimization is delegated to gaussian-dag.
"""
from .channel import (
    DTYPE,
    K_WAVE,
    RDTYPE,
    distances,
    far_field_channel,
    near_field_channel,
)
from .dag import (
    diamond,
    mi,
    multirelay_merge,
    ris_branch,
    single_link,
)
from .geometry import ArrayGeometry, movable, ula
from .ofdm import (
    subcarrier_frequencies,
    subcarrier_wavenumbers,
    wideband_mi,
)
from .projections import (
    project_box,
    project_frobenius_ball,
    project_min_separation,
    project_total_power,
    repulsion_penalty,
)
from .relay import af_gain, physical_channel, waterfilling_capacity
from .scene import Scene

__version__ = "0.1.0"
__all__ = [
    "ArrayGeometry", "DTYPE", "K_WAVE", "RDTYPE", "Scene", "af_gain",
    "diamond", "distances", "far_field_channel", "mi", "movable",
    "multirelay_merge", "near_field_channel", "physical_channel", "project_box",
    "project_frobenius_ball", "project_min_separation", "project_total_power",
    "repulsion_penalty", "ris_branch", "single_link",
    "subcarrier_frequencies", "subcarrier_wavenumbers", "ula",
    "waterfilling_capacity", "wideband_mi",
]
