import logging

from src.secret_utils import RedactFilter, redact


def test_redact_replaces_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_token_value")
    monkeypatch.delenv("BUCKET_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("BUCKET_SECRET_ACCESS_KEY", raising=False)
    text = redact("failed with token hf_supersecret_token_value trailing")
    assert "hf_supersecret_token_value" not in text
    assert "***" in text


def test_log_filter_redacts(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_token_value")
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "awssecretvalueXXXX")
    rec = logging.LogRecord(
        "test-seedvr2-redact", logging.INFO, __file__, 1,
        "tok=hf_supersecret_token_value", (), None,
    )
    assert RedactFilter().filter(rec) is True
    assert "hf_supersecret_token_value" not in rec.getMessage()
