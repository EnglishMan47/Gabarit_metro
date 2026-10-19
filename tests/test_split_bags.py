"""Read every SQLite segment of a ROS bag, including different topic IDs."""
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import yaml

from gabarit.rosbag import POINTCLOUD2, BagReader, discover_bags, open_bag
from tests.sim import encode_pointcloud2, tunnel_scan


def _chunk(path, topic_id, stamps):
    con = sqlite3.connect(path)
    con.executescript('CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT);'
                      'CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);')
    con.execute('INSERT INTO topics VALUES(?,?,?)', (topic_id, '/points', POINTCLOUD2))
    for i, stamp in enumerate(stamps):
        raw = tunnel_scan(cols=4, stamp=stamp)
        con.execute('INSERT INTO messages VALUES(?,?,?,?)', (i+1, topic_id, int(stamp*1e9), encode_pointcloud2(raw)))
    con.commit(); con.close()


def _metadata(folder, names):
    (folder/'metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':
                                                       {'relative_file_paths': names}}), encoding='utf-8')


def test_directory_reads_all_segments_globally_in_timestamp_order(tmp_path):
    bag = tmp_path/'split'; bag.mkdir()
    _chunk(bag/'ride_0.db3', 3, [1.0, 1.2])
    _chunk(bag/'ride_1.db3', 9, [1.1, 1.3])
    _metadata(bag, ['ride_1.db3', 'ride_0.db3'])
    info = open_bag(bag, tmp_path)
    assert info.frame_count == 4 and len(info.storage_paths) == 2 and info.id == 'split'
    assert info.start_ns == 1_000_000_000 and info.end_ns == 1_300_000_000
    assert info.public()['storage_files'] == 2
    reader = BagReader(info)
    try:
        frames = list(reader.iterate())
        assert [row[1] for row in frames] == [1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000]
        np.testing.assert_allclose([row[2].stamp for row in frames], [1., 1.1, 1.2, 1.3])
        assert reader.read(2)[0] == 1_200_000_000
    finally:
        reader.close()
    assert discover_bags(tmp_path)[0].frame_count == 4
    assert open_bag(bag/'ride_0.db3').frame_count == 2


def test_metadata_excludes_unlisted_sqlite_files(tmp_path):
    _chunk(tmp_path/'part0.db3', 1, [1.0])
    _chunk(tmp_path/'orphan.db3', 1, [9.0])
    _metadata(tmp_path, ['part0.db3'])
    assert open_bag(tmp_path).frame_count == 1


def test_incomplete_split_bag_is_not_silently_evaluated(tmp_path):
    _chunk(tmp_path/'part0.db3', 1, [1.0])
    _metadata(tmp_path, ['part0.db3', 'part1.db3'])
    with pytest.raises(FileNotFoundError):
        open_bag(tmp_path)
    assert discover_bags(tmp_path) == []


def test_metadata_cannot_escape_bag_directory(tmp_path):
    bag = tmp_path/'bag'; bag.mkdir()
    _chunk(tmp_path/'outside.db3', 1, [1.0])
    _metadata(bag, ['../outside.db3'])
    with pytest.raises(ValueError, match='outside'):
        open_bag(bag)
