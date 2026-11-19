import pytest

from ncm_iwsp_gpt1.cli import main


def test_download_is_off_by_default(tmp_path):
    out = tmp_path / "should-not-exist.pt"
    with pytest.raises(SystemExit, match="download is off"):
        main(
            [
                "pack",
                "--config",
                "configs/gpt1_data.yaml",
                "--hf-dataset",
                "lucadiliello/bookcorpusopen",
                "--out",
                str(out),
            ]
        )
    assert not out.exists()
