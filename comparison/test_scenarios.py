import numpy as np
from comparison.scenarios import select_training, perturb, CORRUPTIONS


def test_nested_subsets_and_corruptions():
    labels = np.repeat(np.arange(8), 50)
    small, y = select_training(labels, "few2", 42)
    big, _ = select_training(labels, "few10", 42)
    assert set(small) <= set(big)
    assert np.all(np.bincount(y) == 2)
    idx, noisy = select_training(labels, "label_noise", 42)
    assert (noisy != labels[idx]).sum() == 80
    image = np.random.default_rng(0).integers(0, 255, (150, 150, 3), dtype=np.uint8)
    for kind in CORRUPTIONS:
        a = perturb(image, kind, 7)
        assert a.shape == image.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, perturb(image, kind, 7))


def test_augmentation_is_train_only_and_reproducible():
    from comparison.variants import training_views
    image = np.arange(150 * 150 * 3, dtype=np.uint8).reshape(150, 150, 3)
    before = image.copy()
    a, b = list(training_views(image, 3, 42, 7)), list(training_views(image, 3, 42, 7))
    assert len(a) == 4
    np.testing.assert_array_equal(a[0], before)
    np.testing.assert_array_equal(image, before)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_localization_layout_and_validation_threshold():
    from comparison.localization import best_dice_threshold, make_layouts, patch_scores_from_map, stitch_tile_maps
    labels = np.r_[np.zeros(8, dtype=int), np.ones(56, dtype=int)]
    layouts = make_layouts(labels, 4, tumors_per_mosaic=2, seed=7)
    indices = [i for layout in layouts for i in layout["indices"]]
    assert len(indices) == len(set(indices)) == 64
    assert all(sum(np.asarray(layout["labels"]) == 0) == 2 for layout in layouts)
    threshold, dice = best_dice_threshold([1, 1, 0, 0], [.9, .8, .2, .1])
    assert threshold == np.float32(.8) and dice == 1
    stitched = stitch_tile_maps([np.full((2, 2), value) for value in range(16)], tile_size=4)
    np.testing.assert_array_equal(patch_scores_from_map(stitched), np.arange(16))
    assert all(np.unique(stitched[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4]).size == 1 for row in range(4) for col in range(4))


def test_finish_propagates_failed_validation(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from comparison import finish
    root = tmp_path / "comparison"
    (root / "runs/smoke").mkdir(parents=True)
    monkeypatch.setattr(finish, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["finish.py", "--campaign", "smoke"])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if "pytest" in command else 0)
    monkeypatch.setattr(finish.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        finish.main()
    assert any(any(str(part).endswith("localization.py") for part in command) for command in calls)
    assert "pytest" in calls[-1]
    assert not any(any(str(part).endswith("report.py") for part in command) for command in calls)
