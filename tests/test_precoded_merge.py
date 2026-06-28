"""Step #1 regression / tie-point tests for the precoded merge channel (SPEC.md).

The precoded ``multirelay_merge`` (Tx precoder ``F`` and per-relay matrices
``W_l``) is a strict generalization of the isotropic, scalar-AF model. These
tests pin the *anchor*: the general code, evaluated at the scalar-AF / isotropic
special case, must reproduce the former model exactly. They are the bridge
between the old behavior and the new precoded engine (SPEC.md Sec. 4).

Run: uv run pytest -q tests/test_precoded_merge.py
"""
import math

import torch

import relay_grid_dag as rgd

P_TX = 100.0
P_RELAY = 100.0
SIGMA_R = 1.0


def _scene():
    s, names, coords = rgd.build_candidate_scene("near")
    return s, names


def test_defaults_unchanged_is_byte_identical():
    """Calling with the new kwargs left at their defaults yields the same MI as
    not passing them at all (backward compatibility of the signature)."""
    s, names = _scene()
    subset = [names[i] for i in (5, 12, 18)]
    mi_plain = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                           P_tx=P_TX, P_relay=P_RELAY))
    mi_kw = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                        P_tx=P_TX, P_relay=P_RELAY,
                                        precoder=None, relay_mats=None))
    assert torch.allclose(mi_plain, mi_kw)


def test_precoder_identity_reproduces_isotropic():
    """A scaled-identity Tx precoder F = sqrt(P_tx/n_tx) I (so X ~ CN(0,I) and the
    transmit covariance is the isotropic (P_tx/n_tx) I) reproduces the default
    isotropic-input MI."""
    s, names = _scene()
    subset = [names[i] for i in (3, 9, 20)]
    n_tx = s.n_ant("tx")
    F = math.sqrt(P_TX / n_tx) * torch.eye(n_tx, dtype=rgd.DTYPE)

    mi_iso = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                         P_tx=P_TX, P_relay=P_RELAY))
    mi_prec = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                          P_tx=P_TX, P_relay=P_RELAY, precoder=F))
    assert torch.allclose(mi_iso, mi_prec, atol=1e-9, rtol=0)


def test_relay_mats_scalar_reproduces_af():
    """Per-relay matrices set to the scalar AF gain, W_l = g_l I, reproduce the
    default scalar-AF MI (W = g I is the special case of the matrix relay)."""
    s, names = _scene()
    subset = [names[i] for i in (1, 7, 14, 22)]
    n_tx = s.n_ant("tx")
    input_cov = (P_TX / n_tx) * torch.eye(n_tx, dtype=rgd.DTYPE)

    relay_mats = []
    for nm in subset:
        H_sr = s.channel("tx", nm)
        g = rgd.af_gain(H_sr, input_cov, SIGMA_R, P_RELAY)
        relay_mats.append(g * torch.eye(s.n_ant(nm), dtype=rgd.DTYPE))

    mi_af = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                        P_tx=P_TX, P_relay=P_RELAY))
    mi_mat = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                         P_tx=P_TX, P_relay=P_RELAY,
                                         relay_mats=relay_mats))
    assert torch.allclose(mi_af, mi_mat, atol=1e-9, rtol=0)


def test_relay_matrices_keep_mi_differentiable():
    """A relay matrix that requires grad keeps the merge-channel MI a finite,
    differentiable scalar (precondition for the engine of SPEC.md Sec. 4)."""
    s, names = _scene()
    subset = [names[5]]
    n_l = s.n_ant(subset[0])
    # start from a scalar-AF-scaled identity so the power is sane
    input_cov = (P_TX / s.n_ant("tx")) * torch.eye(s.n_ant("tx"), dtype=rgd.DTYPE)
    g = rgd.af_gain(s.channel("tx", subset[0]), input_cov, SIGMA_R, P_RELAY)
    W = (g * torch.eye(n_l, dtype=rgd.DTYPE)).clone().requires_grad_(True)

    I = rgd.mi(rgd.multirelay_merge(s, "tx", subset, "rx",
                                    P_tx=P_TX, P_relay=P_RELAY, relay_mats=[W]))
    assert torch.isfinite(I) and I.item() > 0
    I.backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    assert W.grad.abs().sum() > 0      # the relay matrix actually moves the MI
