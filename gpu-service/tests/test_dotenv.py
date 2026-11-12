import os

from ohh_gpu.dotenv import load_dotenv


def test_reads_crlf_bom_and_keeps_existing_environment(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_bytes(
        b"\xef\xbb\xbfOHH_T_A=abc\r\n# c\r\n\r\nOHH_T_B=\"q\"\r\nOHH_T_C=file\r\n"
    )
    monkeypatch.delenv("OHH_T_A", raising=False)
    monkeypatch.delenv("OHH_T_B", raising=False)
    monkeypatch.setenv("OHH_T_C", "preset")
    load_dotenv(p)
    assert os.environ["OHH_T_A"] == "abc"
    assert os.environ["OHH_T_B"] == "q"
    assert os.environ["OHH_T_C"] == "preset"
    monkeypatch.delenv("OHH_T_A")
    monkeypatch.delenv("OHH_T_B")


def test_missing_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope")
