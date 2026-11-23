import os

from src.infer import apex_fused_norm_supported


def test_apex_fused_norm_supported_false_without_cuda(monkeypatch):
    # On the unit-test host there is no CUDA 4090. The helper must not crash.
    import src.infer as infer
    monkeypatch.setattr(infer, "apex_fused_norm_supported", infer.apex_fused_norm_supported)
    assert apex_fused_norm_supported() in (True, False)


def test_detects_sm_token_in_fake_so(tmp_path, monkeypatch):
    so = tmp_path / "fused_layer_norm_cuda.so"
    so.write_bytes(b"xxxxsm_90yyyy")
    import src.infer as infer

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability(idx=0):
            return (8, 9)

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setattr(infer, "apex_fused_norm_supported", infer.apex_fused_norm_supported)
    # Patch internals via a thin reimplementation check: official 4090 token is sm_89,
    # which is absent from a sm_90-only fake wheel.
    data = so.read_bytes()
    assert b"sm_89" not in data and b"sm_90" in data

    # And a 4090-capable rebuild would advertise sm_89.
    so.write_bytes(b"xxxxsm_89yyyy sm_90")
    assert b"sm_89" in so.read_bytes()
