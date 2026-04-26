from pathlib import Path

from quiver.config import load_config


def test_load_example_config(tmp_path: Path):
    src = Path(__file__).resolve().parent.parent / "examples" / "quiver.toml"
    cfg = load_config(src)
    assert cfg.exploration.num_solutions == 10
    assert cfg.optimizer.method == "COBYLA"
    assert cfg.diversity.threshold == 0.25
    assert cfg.budget.max_gates == 80
    assert "qaoa" in cfg.ansatz.families
