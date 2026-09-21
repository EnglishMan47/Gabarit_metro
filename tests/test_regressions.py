import tempfile
import sqlite3
import unittest
from pathlib import Path

import numpy as np

from webapp.bag_store import BagStore
from webapp.config import DetectorConfig
from webapp.detector import detect_obstacles, visualization_sample
from webapp.scene_model import NormalSceneModel, encode_background_voxels, save_scene_model


class RegressionTests(unittest.TestCase):
    def test_model_grid_does_not_change_with_ui_parameters(self):
        cfg = DetectorConfig()
        points = np.array([[0.13, 22.13, -1.3]], dtype=np.float32)
        heights = np.array([1.13], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.npz'
            save_scene_model(path, encode_background_voxels(points, heights, cfg), np.array([5]), 10, 3, cfg)
            model = NormalSceneModel.load(path)
            self.assertTrue(model.match(points, heights, cfg.with_overrides(corridor_width=3.0, max_forward=150))[0])

    def test_far_object_is_processed_when_range_increases(self):
        rng = np.random.default_rng(81)
        points = rng.normal([0, 120, -1.1], [0.05, 0.05, 0.15], (100, 3)).astype(np.float32)
        self.assertTrue(detect_obstacles(points, DetectorConfig(max_forward=150))["raw_detected"])
        self.assertFalse(detect_obstacles(points, DetectorConfig(max_forward=90))["raw_detected"])

    def test_visualization_coordinates_are_short_decimals(self):
        points = np.array([[0.123456, 10.56789, -1.23456]], dtype=np.float32)
        self.assertEqual(visualization_sample(points, None)[0], [0.123, 10.568, -1.235, 0.0])

    def test_bag_metadata_counts_only_pointcloud_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.db3'
            with sqlite3.connect(path) as db:
                db.executescript("CREATE TABLE topics(id INTEGER, name TEXT, type TEXT); CREATE TABLE messages(id INTEGER, topic_id INTEGER, timestamp INTEGER, data BLOB);")
                db.executemany('INSERT INTO topics VALUES(?,?,?)', [(1,'/imu','sensor_msgs/msg/Imu'),(2,'/cloud','sensor_msgs/msg/PointCloud2')])
                db.executemany('INSERT INTO messages VALUES(?,?,?,?)', [(1,1,0,b''),(2,2,100,b''),(3,2,200,b'')])
            db.close()
            bag = BagStore(Path(directory)).list_bags()[0]
            self.assertEqual(bag['frame_count'], 2)
            self.assertEqual(bag['topic'], '/cloud')

    def test_nonfinite_config_is_rejected(self):
        with self.assertRaises(ValueError):
            DetectorConfig().with_overrides(corridor_center_x=float('nan'))
