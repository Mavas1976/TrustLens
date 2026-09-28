import json

import numpy as np
import pytest

from trustlens.report import TrustReport


@pytest.fixture
def sample_report():
    """Create a minimal TrustReport for testing."""
    results = {
        "calibration": {"ece": 0.02, "brier": 0.01},
        "failure": {"confidence_gap": {"gap": 0.15}},
    }
    X = np.random.rand(10, 2)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])  # 1 error
    y_prob = np.random.rand(10, 2)

    class MockModel:
        pass

    return TrustReport(results, MockModel(), X, y_true, y_pred, y_prob)


def test_save_json_creates_file(sample_report, tmp_path):
    path = tmp_path / "report.json"
    p = sample_report.save(str(path))

    assert path.exists()
    assert p == path.resolve()


def test_save_json_valid_content(sample_report, tmp_path):
    path = tmp_path / "report.json"
    sample_report.save(str(path))

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # New unified schema (PR 6)
    assert "results" in data
    assert "metadata" in data
    assert "trust_score" in data
    assert "calibration" in data["results"]
    assert data["results"]["calibration"]["ece"] == 0.02
    assert "failure" in data["results"]


def test_save_txt_creates_file(sample_report, tmp_path):
    path = tmp_path / "report.txt"
    p = sample_report.save(str(path))

    assert path.exists()
    assert p == path.resolve()

    content = path.read_text(encoding="utf-8")
    assert "TrustLens Analysis Report" in content
    assert "TRUST SCORE" in content
    assert "Calibration Analysis" in content


def test_save_directory_bundle(sample_report, tmp_path):
    # Test backward compatibility for directory saving
    out_dir = tmp_path / "my_report"
    sample_report.save(str(out_dir))

    assert out_dir.is_dir()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "trust_score.json").exists()


class TestSaveRobustness:
    """TL-27 / TL-20: save paths behave predictably."""

    @pytest.fixture
    def report(self):
        import numpy as np

        from trustlens import analyze

        y = np.array([0, 1] * 60)
        p = np.where(y == 1, 0.8, 0.2)
        return analyze(None, None, y, y_pred=y, y_prob=np.column_stack([1 - p, p]), verbose=False)

    def test_accepts_pathlib_and_dotted_directories(self, report, tmp_path):
        assert report.save(tmp_path / "r.json").is_file()
        assert report.save(tmp_path / "run_2026.09").is_dir()

    def test_rejects_unsupported_file_type(self, report, tmp_path):
        with pytest.raises(ValueError, match="Unsupported report file type"):
            report.save(tmp_path / "report.PNG")
        assert not (tmp_path / "report.PNG").exists()

    def test_overwrite_false_protects_existing_output(self, report, tmp_path):
        target = tmp_path / "r.json"
        report.save(target)
        with pytest.raises(FileExistsError):
            report.save(target, overwrite=False)

    def test_plots_create_missing_directories(self, report, tmp_path):
        target = tmp_path / "nested" / "deeper" / "summary.png"
        report.summary_plot(save_path=str(target), show=False)
        assert target.is_file()
