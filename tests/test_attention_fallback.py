import pytest
"""CPU tests for the flash-attn -> PyTorch SDPA varlen fallback."""
import sys

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from src.infer import flash_attn_varlen_sdpa, install_flash_attn_sdpa_fallback


def _pack(seqs):
    """seqs: list of (S, H, D) tensors -> packed (L, H, D) and cu_seqlens."""
    cu = [0]
    for s in seqs:
        cu.append(cu[-1] + s.shape[0])
    return torch.cat(seqs, dim=0), torch.tensor(cu, dtype=torch.int32)


def test_sdpa_varlen_matches_batched_sdpa_one_sequence():
    torch.manual_seed(0)
    q = torch.randn(5, 2, 4)
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    cu = torch.tensor([0, 5], dtype=torch.int32)
    got = flash_attn_varlen_sdpa(q, k, v, cu, cu)
    want = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
    ).squeeze(0).transpose(0, 1)
    assert got.shape == q.shape
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_sdpa_varlen_two_sequences_are_independent():
    torch.manual_seed(1)
    q1, k1, v1 = torch.randn(3, 2, 4), torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    q2, k2, v2 = torch.randn(4, 2, 4), torch.randn(4, 2, 4), torch.randn(4, 2, 4)
    q, cu_q = _pack([q1, q2])
    k, cu_k = _pack([k1, k2])
    v, cu_v = _pack([v1, v2])
    assert torch.equal(cu_q, cu_k) and torch.equal(cu_k, cu_v)
    got = flash_attn_varlen_sdpa(q, k, v, cu_q, cu_k)
    a = F.scaled_dot_product_attention(
        q1.transpose(0, 1).unsqueeze(0), k1.transpose(0, 1).unsqueeze(0), v1.transpose(0, 1).unsqueeze(0),
    ).squeeze(0).transpose(0, 1)
    b = F.scaled_dot_product_attention(
        q2.transpose(0, 1).unsqueeze(0), k2.transpose(0, 1).unsqueeze(0), v2.transpose(0, 1).unsqueeze(0),
    ).squeeze(0).transpose(0, 1)
    assert torch.allclose(got[:3], a, atol=1e-5, rtol=1e-4)
    assert torch.allclose(got[3:], b, atol=1e-5, rtol=1e-4)


def test_install_fallback_provides_flash_attn_varlen_func(monkeypatch):
    # Force the SDPA path even if a flash_attn package is present in the venv.
    monkeypatch.setattr("src.infer._flash_attn_has_device_kernel", lambda device_index=0: False)
    sys.modules.pop("flash_attn", None)
    msg = install_flash_attn_sdpa_fallback()
    assert msg and "SDPA" in msg
    import flash_attn
    assert flash_attn.flash_attn_varlen_func is flash_attn_varlen_sdpa
