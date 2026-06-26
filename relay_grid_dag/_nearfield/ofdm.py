"""Wideband (OFDM) wrapper: subcarrier grid and weighted wideband sum-rate.

The single boundary this module adds is *frequency*. ``Scene`` stays pure
geometry and the DAG emitters stay per-subcarrier; here we build the subcarrier
wavenumbers from a fixed OFDM grid and sum the per-subcarrier mutual information
into the weighted wideband sum-rate

    U = sum_i w_i I(X_i; Y_i),

each term evaluated by the same K-recursion. Because every subcarrier reuses the
*same* movable centres (autograd leaves), one backward pass through ``U``
accumulates the position gradient over all subcarriers at once. The carrier
``f_c``, subcarrier count ``S`` and spacing ``df`` are fixed system parameters,
not design variables; the single-carrier case is ``S = 1`` (k = 2*pi).
"""
from __future__ import annotations

import torch

from .channel import K_WAVE
from .dag import mi


def subcarrier_frequencies(f_c: float, S: int, df: float) -> list[float]:
    """OFDM subcarrier frequencies ``f_i = f_c + (i - (S+1)/2) df``, ``i=1..S``.

    Centred on ``f_c`` (so ``S = 1`` gives ``[f_c]``); ``S = 3`` gives
    ``[f_c - df, f_c, f_c + df]``.
    """
    if S < 1:
        raise ValueError(f"S must be >= 1, got {S}")
    return [f_c + (i - (S + 1) / 2.0) * df for i in range(1, S + 1)]


def subcarrier_wavenumbers(f_c: float, S: int, df: float) -> list[float]:
    """Subcarrier wavenumbers ``k_i = 2*pi f_i / f_c`` for the grid above.

    Normalised by the carrier wavelength (lambda_c = 1), so ``k = 2*pi`` at
    ``f_c``. These are the values to pass as ``k_wave`` to a DAG emitter.
    """
    return [K_WAVE * f_i / f_c for f_i in subcarrier_frequencies(f_c, S, df)]


def wideband_mi(spec_fn, k_waves, *, weights=None, output_node=None,
                input_node=0, jitter=1e-9, bits=True) -> torch.Tensor:
    """Weighted wideband sum-rate ``U = sum_i w_i I(X_i; Y_i)`` (differentiable).

    ``spec_fn(k_wave) -> DagSpec`` builds the per-subcarrier DAG at the given
    wavenumber, e.g.::

        wideband_mi(lambda k: single_link(scene, "tx", "rx", k_wave=k),
                    subcarrier_wavenumbers(f_c=10.0, S=4, df=0.5))

    All subcarriers share the scene's movable centres, so a single
    ``U.backward()`` returns the position gradient summed over the band. With one
    wavenumber (and the default weight) this reduces to a plain single-carrier
    :func:`~nearfield_dag.dag.mi`. ``weights`` defaults to all ones.
    """
    ks = list(k_waves)
    if not ks:
        raise ValueError("k_waves must be non-empty")
    w = [1.0] * len(ks) if weights is None else list(weights)
    if len(w) != len(ks):
        raise ValueError(f"weights length {len(w)} != number of subcarriers {len(ks)}")
    total = None
    for w_i, k_i in zip(w, ks):
        term = w_i * mi(spec_fn(k_i), output_node=output_node,
                        input_node=input_node, jitter=jitter, bits=bits)
        total = term if total is None else total + term
    return total
