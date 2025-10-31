from pathlib import Path

def test_dirs_exist():
    base = Path('experiments/phase2')
    for sub in ['config','core','metrics','verify','runners','vendors','reports','tests']:
        assert (base / sub).exists()