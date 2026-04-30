import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.representations.registry import REPRESENTATION_REGISTRY, get_representation
from src.representations.traditional import (
    BinaryEventImageRepresentation,
    EventFrameRepresentation,
    TimeSurfaceRepresentation,
    TimestampImageRepresentation,
    VoxelGridRepresentation,
)


class TraditionalRepresentationTest(unittest.TestCase):
    def setUp(self):
        self.events = np.array(
            [
                [1, 1, 0, 1],
                [1, 1, 10, 1],
                [2, 1, 20, 0],
                [3, 1, 30, -1],
                [9, 9, 40, 1],
            ],
            dtype=np.float32,
        )
        self.config = {"height": 3, "width": 4, "bins": 3, "normalize": False, "max_events": 100}

    def test_event_frame_uses_positive_then_negative_channels(self):
        out = EventFrameRepresentation(self.config).build(self.events)
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertEqual(float(out[0, 1, 1]), 2.0)
        self.assertEqual(float(out[1, 1, 2]), 1.0)
        self.assertEqual(float(out[1, 1, 3]), 1.0)

    def test_binary_timestamp_and_time_surface_are_bounded(self):
        binary = BinaryEventImageRepresentation(self.config).build(self.events)
        timestamp = TimestampImageRepresentation({**self.config, "normalize": True}).build(self.events)
        surface = TimeSurfaceRepresentation({**self.config, "normalize": True, "tau_us": 10}).build(self.events)

        for out in (binary, timestamp, surface):
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)
            self.assertTrue(np.isfinite(out).all())

        self.assertAlmostEqual(float(timestamp[0, 1, 1]), 10.0 / 30.0, places=6)
        self.assertAlmostEqual(float(timestamp[1, 1, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(surface[1, 1, 3]), 1.0, places=6)

    def test_voxel_grid_preserves_event_mass(self):
        out = VoxelGridRepresentation(self.config).build(self.events)
        self.assertEqual(out.shape, (6, 3, 4))
        self.assertAlmostEqual(float(out[:3, 1, 1].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(out[3:, 1, 2].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(out[3:, 1, 3].sum()), 1.0, places=6)

    def test_empty_structured_and_registry_paths(self):
        dtype = [("x", "i4"), ("y", "i4"), ("t", "f8"), ("p", "i4")]
        structured = np.array([(1, 1, 0.0, 1), (2, 1, 10.0, 0)], dtype=dtype)
        frame = EventFrameRepresentation(self.config).build(structured)
        self.assertEqual(float(frame[0, 1, 1]), 1.0)
        self.assertEqual(float(frame[1, 1, 2]), 1.0)

        for cls, channels in [
            (EventFrameRepresentation, 2),
            (BinaryEventImageRepresentation, 2),
            (TimestampImageRepresentation, 2),
            (TimeSurfaceRepresentation, 2),
            (VoxelGridRepresentation, 6),
        ]:
            out = cls(self.config).build(np.zeros((0, 4), dtype=np.float32))
            self.assertEqual(out.shape, (channels, 3, 4))
            self.assertEqual(float(out.sum()), 0.0)

        for name in ["event_frame", "event_count", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"]:
            self.assertIn(name, REPRESENTATION_REGISTRY)
            self.assertIsNotNone(get_representation(name))

    def test_gen1_factory_accepts_traditional_names_without_detection_import(self):
        spec = importlib.util.spec_from_file_location(
            "gen1_representations_direct",
            Path("src/detection/gen1_representations.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name in ["event_frame", "event_count", "binary_event_image", "timestamp_image", "time_surface", "voxel_grid"]:
            rep = module.create_gen1_representation(name, {"height": 3, "width": 4, "bins": 3})
            out = rep.build(self.events)
            self.assertEqual(out.ndim, 3)
            self.assertEqual(out.shape[1:], (3, 4))
            self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
