from omegaconf import OmegaConf

def test_default_config_loads():
    cfg = OmegaConf.load("configs/default.yaml")
    assert "detection" in cfg
    assert "tracking" in cfg
    assert "occupancy" in cfg
    assert "reasoning" in cfg

def test_colab_overrides_merge():
    base = OmegaConf.load("configs/default.yaml")
    overrides = OmegaConf.load("configs/colab.yaml")
    merged = OmegaConf.merge(base, overrides)
    assert merged.detection.fp16 is True
