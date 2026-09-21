"""Deterministic multi-frame tracking and obstacle event aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from webapp.config import DetectorConfig


@dataclass
class Track:
    id: int
    center: np.ndarray
    first_frame: int
    last_frame: int
    hits: int = 1
    consecutive_hits: int = 1
    confirmed_frame: int | None = None
    missing: int = 0
    confirmed: bool = False
    distances: list[float] = field(default_factory=list)


class TrackManager:
    def __init__(self, config: DetectorConfig, expected_step: int = 1):
        self.config = config
        self.expected_step = max(1, int(expected_step))
        self.tracks: dict[int, Track] = {}
        self.completed_events: list[dict[str, Any]] = []
        self.next_id = 1
        self.last_frame: int | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.completed_events.clear()
        self.next_id = 1
        self.last_frame = None

    def _close(self, track: Track) -> None:
        if not track.confirmed:
            return
        self.completed_events.append(
            {
                "track_id": track.id,
                "first_frame": track.first_frame,
                "confirmed_frame": track.confirmed_frame,
                "last_frame": track.last_frame,
                "frame_count": track.hits,
                "nearest_distance_m": min(track.distances) if track.distances else None,
            }
        )

    def update(self, frame_index: int, objects: list[dict[str, Any]]) -> dict[str, Any]:
        if self.last_frame is not None and frame_index != self.last_frame + self.expected_step:
            for track in self.tracks.values():
                self._close(track)
            self.tracks.clear()
        self.last_frame = frame_index
        candidates = [dict(item) for item in objects]
        centers = [
            np.asarray([item["center_m"]["x"], item["center_m"]["forward"]], dtype=np.float32)
            for item in candidates
        ]
        unmatched_tracks = set(self.tracks)
        unmatched_objects = set(range(len(candidates)))
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for object_index, center in enumerate(centers):
                distance = float(np.linalg.norm(track.center - center))
                if distance <= self.config.track_match_distance:
                    pairs.append((distance, track_id, object_index))
        for _, track_id, object_index in sorted(pairs):
            if track_id not in unmatched_tracks or object_index not in unmatched_objects:
                continue
            track = self.tracks[track_id]
            track.center = centers[object_index]
            track.last_frame = frame_index
            track.hits += 1
            track.consecutive_hits += 1
            track.missing = 0
            if not track.confirmed and track.consecutive_hits >= self.config.track_confirm_frames:
                track.confirmed = True
                track.confirmed_frame = frame_index
            track.distances.append(float(candidates[object_index]["distance_surface_m"]))
            candidates[object_index]["track_id"] = track_id
            candidates[object_index]["track_confirmed"] = track.confirmed
            unmatched_tracks.remove(track_id)
            unmatched_objects.remove(object_index)
        for object_index in sorted(unmatched_objects):
            item = candidates[object_index]
            track = Track(
                id=self.next_id,
                center=centers[object_index],
                first_frame=frame_index,
                last_frame=frame_index,
                confirmed=self.config.track_confirm_frames <= 1,
                confirmed_frame=frame_index if self.config.track_confirm_frames <= 1 else None,
                distances=[float(item["distance_surface_m"])],
            )
            self.tracks[track.id] = track
            item["track_id"] = track.id
            item["track_confirmed"] = track.confirmed
            self.next_id += 1
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missing += 1
            track.consecutive_hits = 0
            if track.missing > self.config.track_max_missing_frames:
                self._close(track)
                del self.tracks[track_id]
        active = [
            {
                "track_id": track.id,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "hits": track.hits,
                "missing": track.missing,
                "confirmed": track.confirmed,
                "nearest_distance_m": min(track.distances) if track.distances else None,
            }
            for track in self.tracks.values()
        ]
        stable_tracks = [
            track
            for track in self.tracks.values()
            if track.confirmed and track.missing < self.config.track_clear_frames
        ]
        stable_distances = [track.distances[-1] for track in stable_tracks if track.distances]
        return {
            "objects": candidates,
            "event_detected": bool(stable_tracks),
            "event_nearest_distance_m": min(stable_distances) if stable_distances else None,
            "active_tracks": active,
            "completed_events": list(self.completed_events),
        }

    def finalize(self) -> list[dict[str, Any]]:
        for track in self.tracks.values():
            self._close(track)
        self.tracks.clear()
        return list(self.completed_events)
