"""Exercise the relocated wrapper without weights or network access."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class TestGeospatialIntegration(unittest.TestCase):
    def setUp(self):
        # Keep binary extensions outside the sys.modules rollback below.
        for name in ("cv2", "torch"):
            importlib.import_module(name)

        class Model:
            def __init__(self, model, **kwargs):
                self.model = model
                self.device = "cpu"

            def __call__(self, image, **kwargs):
                return "model-results"

        self.mask = np.ones((1, 12, 12), dtype=np.uint8)
        self.prompt = Mock(img=np.zeros((12, 12, 3), dtype=np.uint8))
        for name in ("everything_prompt", "point_prompt", "box_prompt", "text_prompt"):
            getattr(self.prompt, name).return_value = self.mask
        backend = types.ModuleType("fastsam")
        backend.FastSAM = Model
        backend.FastSAMPrompt = Mock(return_value=self.prompt)
        self.backend = backend
        common = types.ModuleType("samgeo.common")
        for name in (
            "array_to_image",
            "download_file",
            "raster_to_vector",
            "show_image",
            "temp_file_path",
        ):
            setattr(common, name, Mock())
        self.common = common
        modules = {
            "fastsam": backend,
            "samgeo": types.ModuleType("samgeo"),
            "samgeo.common": common,
        }
        with patch.dict(sys.modules, modules):
            sys.modules.pop("samgeo_fastsam", None)
            self.module = importlib.import_module("samgeo_fastsam")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name, "FastSAM-s.pt").touch()
        self.model = self.module.SamGeo(
            model="FastSAM-s.pt", checkpoint_dir=self.temp.name
        )
        self.image = Path(self.temp.name, "image.tif")
        self.image.touch()
        self.model.set_image(str(self.image), device="cpu")

    def test_model_uses_cached_checkpoint_path(self):
        self.assertEqual(self.model.model, str(Path(self.temp.name, "FastSAM-s.pt")))
        self.common.download_file.assert_not_called()

    def test_overlay_accepts_numpy_and_tensor_masks(self):
        torch = importlib.import_module("torch")
        expected = np.broadcast_to([30 / 255, 144 / 255, 1.0, 0.6], (12, 12, 4))
        for annotations in (self.mask, torch.from_numpy(self.mask)):
            with self.subTest(type=type(annotations).__name__):
                self.model.annotations = annotations
                np.testing.assert_allclose(self.model.fast_show_mask(), expected)

    def test_prompt_workflows_keep_their_arguments_and_masks(self):
        self.backend.FastSAMPrompt.assert_called_once_with(
            str(self.image), "model-results", device="cpu"
        )
        cases = (
            ("everything_prompt", (), {}),
            ("point_prompt", ([[2, 3], [7, 8]], [1, 0]), {}),
            ("box_prompt", (), {"bbox": [1, 2, 9, 10]}),
            ("text_prompt", ("buildings",), {}),
        )
        for name, args, kwargs in cases:
            with self.subTest(prompt=name):
                self.assertIs(getattr(self.model, name)(*args, **kwargs), self.mask)
        self.prompt.point_prompt.assert_called_once_with([[2, 3], [7, 8]], [1, 0])
        self.prompt.box_prompt.assert_called_once_with([1, 2, 9, 10], None)
        self.prompt.text_prompt.assert_called_once_with("buildings")
        self.common.download_file.assert_not_called()

    def test_mask_export_preserves_source_and_vector_options(self):
        self.model.everything_prompt()
        mask = self.model.save_masks(better_quality=False)
        np.testing.assert_array_equal(mask, np.full((12, 12), 255, dtype=np.uint8))
        self.model.save_masks(output="masks.tif", better_quality=False, compress="lzw")
        args, kwargs = self.common.array_to_image.call_args
        np.testing.assert_array_equal(args[0], mask)
        self.assertEqual(args[1:], ("masks.tif", str(self.image)))
        self.assertEqual(kwargs, {"compress": "lzw"})
        self.model.raster_to_vector(
            "masks.tif", "masks.gpkg", simplify_tolerance=0.5, dst_crs="EPSG:3857"
        )
        self.common.raster_to_vector.assert_called_once_with(
            "masks.tif", "masks.gpkg", simplify_tolerance=0.5, dst_crs="EPSG:3857"
        )


if __name__ == "__main__":
    unittest.main()
