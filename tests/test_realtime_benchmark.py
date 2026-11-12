"""A missing scan must not be hidden by a duplicate status message."""
import numpy as np
from scripts.benchmark_ros import summarize, header_stamps
import sqlite3
import struct
from types import SimpleNamespace


def rows(stamps):
    return [(i*.1, {'stamp':stamp,'latency_ms':70.,'ros_timing_ms':{'pipeline':65.}})
            for i,stamp in enumerate(stamps)]


def test_full_rate_counts_actual_device_stamps_with_different_storage_epoch():
    stamps=946684800.+np.arange(30)*.1
    report=summarize(rows(stamps),2,np.rint(stamps*1e9).astype(np.int64))
    assert report['steady_10hz_pass']
    assert report['missing_source_messages']==report['duplicate_source_messages']==0
    assert report['received_hz']==10.


def test_duplicate_does_not_hide_missing_input_even_when_message_count_matches():
    stamps=100.+np.arange(30)*.1
    received=stamps.copy();received[9]=received[8]
    report=summarize(rows(received),2,np.rint(stamps*1e9).astype(np.int64))
    assert report['missing_source_messages']==report['duplicate_source_messages']==1
    assert not report['steady_10hz_pass']


def test_processing_deadline_and_frame_loss_independently_fail():
    stamps=100.+np.arange(30)*.1
    messages=rows(stamps);messages[15][1]['latency_ms']=200.
    messages[16][1]['latency_ms']=200.
    assert not summarize(messages,2,np.rint(stamps*1e9).astype(np.int64))['steady_10hz_pass']
    messages=rows(np.delete(stamps,15))
    assert summarize(messages,2,np.rint(stamps*1e9).astype(np.int64))['missing_source_messages']==1


def test_a_gap_recorded_at_source_is_reported_separately_from_detector_loss():
    stamps=100.+np.arange(30)*.1
    stamps[15:]+=.4
    messages=[(float(stamp-stamps[0]),{'stamp':float(stamp),'latency_ms':70.}) for stamp in stamps]
    report=summarize(messages,2,np.rint(stamps*1e9).astype(np.int64))
    assert report['recorded_gap_count']==1 and report['missing_source_messages']==0
    assert report['source_realtime_pass'] and report['latency_10hz_budget_pass']
    assert not report['steady_10hz_pass']


def test_header_prefix_follows_storage_time_across_interleaved_split_files():
    connections=[];index=[]
    for source,times in enumerate([[0,4,5],[1,2,3]]):
        con=sqlite3.connect(':memory:');connections.append(con)
        con.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY,data BLOB)')
        for message,time in enumerate(times):
            data=b'\x00\x01\x00\x00'+struct.pack('<iI',100+time,0)
            con.execute('INSERT INTO messages VALUES (?,?)',(message,data))
            index.append((time,source,message))
    try:
        reader=SimpleNamespace(_cons=connections,_ids=sorted(index))
        np.testing.assert_array_equal(header_stamps(reader,4),np.arange(100,104,dtype=np.int64)*1_000_000_000)
    finally:
        for con in connections:con.close()
