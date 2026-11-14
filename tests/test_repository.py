"""Guard compatibility imports and required assets after repository restructuring."""
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from amp import runner, fisher_router


class RepositoryTests(unittest.TestCase):
    def test_legacy_imports_share_the_canonical_module_state(self):
        legacy_runner = importlib.import_module("resnet_benchmark")
        legacy_fisher = importlib.import_module("fisher_router")
        self.assertIs(legacy_runner, runner)
        self.assertIs(legacy_fisher, fisher_router)
        with patch.object(legacy_runner, "BATCH_SIZE", 3):
            self.assertEqual(runner.BATCH_SIZE, 3)
        self.assertIs(legacy_fisher.LayerModel, fisher_router.LayerModel)

    def test_asset_paths_remain_relative_to_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(runner.ROOT, root)
        self.assertEqual(runner.DATA, root / "benchmark_data")
        self.assertEqual(runner.CHECKPOINT, root / "benchmark_data/resnet50-0676ba61.pth")
        self.assertEqual(runner.IMAGE_ROOT, root)

    def test_required_class_map_ships_with_the_source(self):
        mapping = json.loads((runner.DATA / "imagenet_class_index.json").read_text())
        self.assertEqual(set(mapping), {str(i) for i in range(1000)})
        self.assertEqual(len({entry[0] for entry in mapping.values()}), 1000)
        self.assertEqual(mapping["0"][0], "n01440764")


if __name__ == "__main__":
    unittest.main()
