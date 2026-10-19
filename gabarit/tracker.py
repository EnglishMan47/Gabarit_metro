"""Temporal confirmation of obstacles.

Tracks live in track coordinates. Between frames a static object moves towards
the train by the ego displacement, so the prediction is ``s - delta``. An
obstacle is confirmed after ``confirm_hits`` associations within the last
``confirm_window`` frames and is kept while it is seen (hysteresis on drop).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from gabarit.config import Config
from gabarit.detect import Candidate, FrameGeometry
from gabarit.filtering import support_mask
from gabarit.track import TrackModel


@dataclass
class Track:
    id: int
    s: float
    d: float
    cand: Candidate
    history: list = field(default_factory=list)   # 1 = hit, 0 = miss (latest last)
    hits: int = 0
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    first_seen_s: float = 0.0
    confirmed_at_s: float | None = None
    confirmed_frame: int | None = None
    first_frame: int = 0
    score: float = 0.0
    v_rel: float = 0.0   # own motion along the track (m/frame), positive = away
    lat_hist: list = field(default_factory=list)
    body_history: list = field(default_factory=list)
    strict_history: list = field(default_factory=list)
    width_seen: float = 0.0  # transverse body evidence at least once; narrow rail-head echoes are insufficient
    resolved_history: list = field(default_factory=list)
    height_seen: float = 0.0
    resolved_size: tuple = ()
    resolved_points: int = 0
    level_height: tuple = ()
    wide_body_history: list = field(default_factory=list)

    def public(self) -> dict:
        c = self.cand.public()
        c.update({
            "track_id": self.id,
            "confirmed": self.confirmed,
            "age_frames": self.age,
            "hits": self.hits,
            "confidence": round(self.confidence, 3),
            "first_seen_m": round(self.first_seen_s, 1),
            "confirmed_at_m": None if self.confirmed_at_s is None else round(self.confirmed_at_s, 1),
            "missed": self.misses > 0,
        })
        return c

    @property
    def confidence(self) -> float:
        recent = self.history[-6:]
        rate = sum(recent) / max(len(recent), 1)
        return float(min(1.0, 0.6 * rate + 0.4 * min(self.score, 1.0)))


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame = 0

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    @staticmethod
    def _calibration_shift(t: Track, prior: TrackModel, current: TrackModel) -> tuple[float,float,float]:
        """Same predicted physical point, expressed in two estimated sensor poses."""
        old,new = prior.calibration,current.calibration
        keys = ('height_m','pitch_deg','roll_deg')
        if not old or not new or any(k not in old or k not in new for k in keys):
            return 0.,0.,0.
        if any(not math.isfinite(float(p[k])) for p in (old,new) for k in keys):
            return 0.,0.,0.
        if all(old[k] == new[k] for k in keys):
            return 0.,0.,0.
        f = .5*(t.cand.s_near+t.cand.s_far)
        x = t.d+float(np.interp(f,prior.s,prior.c))
        z = (float(np.mean(t.level_height)) if t.cand.rail_local and t.level_height else
             .5*(t.cand.h_min+t.cand.h_max)+float(np.interp(f,prior.s,prior.z)))
        a,b = math.radians(old['pitch_deg']),math.radians(old['roll_deg'])
        ca,sa,cb,sb = math.cos(a),math.sin(a),math.cos(b),math.sin(b)
        z2 = z-old['height_m']
        raw_x,z1 = x*cb-z2*sb,x*sb+z2*cb
        raw_f,raw_z = f*ca-z1*sa,f*sa+z1*ca
        a,b = math.radians(new['pitch_deg']),math.radians(new['roll_deg'])
        ca,sa,cb,sb = math.cos(a),math.sin(a),math.cos(b),math.sin(b)
        new_f = raw_f*ca+raw_z*sa
        z1 = -raw_f*sa+raw_z*ca
        return raw_x*cb+z1*sb-x,new_f-f,-raw_x*sb+z1*cb+new['height_m']-z

    def _confirmation(self, t: Track) -> tuple[int, int]:
        tc = self.cfg.tracking
        if t.cand.low:
            if t.cand.single_beam:
                return tc.low_single_confirm_window, tc.low_single_confirm_hits
            if self._low_body(t.cand):
                return tc.low_body_confirm_window, tc.low_body_confirm_hits
            return tc.low_confirm_window, tc.low_confirm_hits
        if t.s > tc.far_distance:
            return tc.far_confirm_window, tc.far_confirm_hits
        return tc.confirm_window, tc.confirm_hits

    def _low_body(self, c: Candidate) -> bool:
        det = self.cfg.detection
        return (c.low and not c.single_beam and c.n_points >= max(6, det.low_min_points)
                and c.n_rings >= 2 and c.d_max-c.d_min >= det.low_min_width
                # Fast confirmation is for a resolved shallow body. A steep
                # rail-profile excursion under dust must retain the 6/8 rule.
                and c.h_max <= min(det.low_max_height, .25) and c.size[2] <= .15
                and c.s_far-c.s_near <= 1.5
                # The rail band starts at4m and the baseline needs +/-5m.
                # Nearer bumps have a truncated reference: keep 6/8 there.
                and c.s_near >= 9.
                and c.s_near <= det.low_dense_range)

    def _has_body_extent(self, t: Track) -> bool:
        c = t.cand
        # A vertical-profile error can place a rail-head fragment just above
        # low_max_height. It still needs transverse body evidence before it
        # can confirm; crossing that height threshold is not new evidence.
        rail_fragment = (c.h_min < .35 and c.h_max < .6
                         and abs(abs(c.lateral)-self.cfg.track.gauge/2) < .2)
        return not (c.low or rail_fragment) or t.width_seen >= self.cfg.detection.low_min_width

    def _wide_low_body(self, c: Candidate) -> bool:
        return self._low_body(c) and c.n_points >= 12 and c.size[1] >= .25

    def _weak_continuation(self, t: Track, c: Candidate) -> bool:
        """Lower the score threshold only for an already confirmed, resolved body.

        Sparse reflections cannot create a new track here. The lower score
        may come from transparency or a less certain path; dense multi-beam
        measurements must still match the predicted object's extent closely.
        """
        if (not t.confirmed or c.rejected != 'low_score' or c.n_points < 12 or c.n_rings < 4
                or c.score < .5*self.cfg.detection.min_score):
            return False
        old = t.cand
        lateral_gap = max(c.d_min-old.d_max, old.d_min-c.d_max, 0.0)
        vertical_overlap = min(c.h_max, old.h_max)-max(c.h_min, old.h_min)
        return (abs(c.s_near-t.s) <= .5+.005*max(t.s, 0.)
                and lateral_gap <= .1+.001*max(t.s, 0.)
                and c.size[0] <= max(.75, 1.5*old.size[0])
                and c.size[2] >= .3*old.size[2]
                and vertical_overlap >= .25*min(c.size[2], old.size[2]))

    def _shape_compatible(self, t: Track, c: Candidate) -> bool:
        if not t.confirmed:
            return True
        old = t.cand
        if t.resolved_points >= 100 and c.n_points < max(12,.1*t.resolved_points):
            return False
        if old.low and old.size[2] < .2 and not c.low and c.h_min > .35 and c.size[2] >= .5:
            return False
        # A resolved upright body must not transfer its identity to a low
        # rail fragment at the same range during a temporary occlusion.
        if max(old.size[2], t.height_seen) >= .5 and c.size[2] < .4*max(old.size[2], t.height_seen):
            return False
        if (old.rail_local and c.s_near <= self.cfg.detection.low_dense_range
                and c.size[1] < self.cfg.detection.low_min_width):
            return False
        return True

    @staticmethod
    def _same_returns(a: Candidate, b: Candidate) -> bool:
        if a.point_index is None or b.point_index is None:
            return False
        count = np.intersect1d(a.point_index,b.point_index,assume_unique=True).size
        return count >= max(3,.5*min(len(a.point_index),len(b.point_index)))

    @staticmethod
    def _resolved(c: Candidate) -> bool:
        """Dense body evidence, independent of the number of temporal hits."""
        dense = c.n_points >= 30 and c.n_rings >= 6
        grounded = (c.s_near <= 60. and c.n_points >= 12 and c.n_rings >= 4
                    and c.h_min <= .35 and c.size[1] >= .15)
        slender = (c.n_points >= 6 and c.n_rings >= 4 and c.size[0] <= .35
                   and c.size[1] <= .12 and c.size[2] >= .7)
        shallow = ((c.low or c.h_max <= .8) and c.n_points >= 12 and c.n_rings >= 2
                   and c.size[1] >= .2 and c.size[0] <= 1.8 and c.size[2] <= .3 and c.h_max >= .08)
        upright = (not c.low and (dense or slender or grounded)
                   and .5 <= c.size[2] <= 3.3 and c.size[0] <= 2.0 and c.size[1] <= 1.5)
        return shallow or upright

    def _local_continuation(self, t: Track, lev: np.ndarray, ring: np.ndarray,
                            geo: FrameGeometry) -> Candidate | None:
        """Measure an established body even when dust joins it to infrastructure.

        This restricted search cannot initiate a track. It uses fresh spatially
        supported returns near the predicted body, including just outside a
        jittering estimated envelope. Empty space never renews a track.
        """
        if not t.confirmed or sum(t.resolved_history[-6:]) < 2 or not t.resolved_size:
            return None
        old = t.cand
        shallow = t.resolved_size[2] < .5
        if old.s_near > 60.0 or (shallow and not t.level_height):
            return None
        slender = t.resolved_size[1] <= .12 and t.resolved_size[0] <= .35
        min_points, min_rings = ((12,2) if shallow else (6,4) if slender
                                 else (12,4) if t.resolved_points < 30 else (30,6))
        height = lev[:,2] if shallow else geo.h
        h_min,h_max = t.level_height if shallow else (old.h_min,old.h_max)
        padding = max(.03,.2*(h_max-h_min)) if shallow else .25
        ds = .6 + .01*max(t.s, 0.)
        lateral_search = .3+.003*max(t.s,0.)
        ids = np.flatnonzero((geo.s >= old.s_near-ds) & (geo.s <= old.s_far+ds)
                             & (geo.d >= old.d_min-lateral_search) & (geo.d <= old.d_max+lateral_search)
                             & (height >= h_min-padding) & (height <= h_max+padding))
        if len(ids) < min_points:
            return None
        if self.cfg.filter.enabled:
            ids = ids[support_mask(lev[ids], self.cfg.filter)]
        # Use the dense upper part to locate the body in range. Rail returns
        # below it can otherwise pull the nearest bound forwards a little on
        # every update, accumulating an artificial obstacle velocity.
        upper = ids[height[ids] >= h_min+.3*(h_max-h_min)]
        if len(upper) < (3 if slender or shallow else 12):
            return None
        bins = np.floor(geo.s[upper]/.1).astype(np.int64)
        lo, hi = int(bins.min()), int(bins.max())
        counts = np.bincount(bins-lo,minlength=hi-lo+1)
        span = max(3, int(np.ceil((t.resolved_size[0]+.15)/.1)))
        summed = np.convolve(counts,np.ones(span,dtype=np.int64),mode='full')
        end = int(np.argmax(summed))+lo+1
        # A cropped wall also looks like a dense body. Check fresh returns
        # beyond the largest supported object length before renewing an ID.
        band = ((geo.d >= old.d_min-lateral_search) & (geo.d <= old.d_max+lateral_search)
                & (height >= h_min+.3*(h_max-h_min)) & (height <= h_max+padding))
        width = .1*span
        left, right = .1*(end-span), .1*end
        before = np.count_nonzero(band & (geo.s >= left-2.-width) & (geo.s < left-2.))
        after = np.count_nonzero(band & (geo.s > right+2.) & (geo.s <= right+2.+width))
        if min(before,after) >= .5*summed.max():
            return None
        ids = ids[(geo.s[ids] >= .1*(end-span)) & (geo.s[ids] < .1*end)]
        # Renew the observed body core instead of expanding a crop to every
        # neighbouring return. A partly visible wide object can retain its
        # proven identity without swallowing an adjacent wall or rail.
        lateral_bins = np.floor(geo.d[ids]/.05).astype(np.int64)
        if not len(lateral_bins):
            return None
        lateral_lo = int(lateral_bins.min())
        lateral_counts = np.bincount(lateral_bins-lateral_lo)
        lateral_span = max(6,int(np.ceil((t.resolved_size[1]+.1)/.05)))
        lateral_sums = np.convolve(lateral_counts,np.ones(lateral_span,dtype=np.int64),mode='full')
        peaks = np.flatnonzero(lateral_sums >= .95*lateral_sums.max())
        peak = peaks[np.argmin(np.abs(.05*(peaks+lateral_lo+1-.5*lateral_span)-t.d))]
        lateral_end = int(peak)+lateral_lo+1
        ids = ids[(geo.d[ids] >= .05*(lateral_end-lateral_span)) & (geo.d[ids] < .05*lateral_end)]
        if len(ids) < max(min_points, .3*t.resolved_points) or len(np.unique(ring[ids])) < min_rings:
            return None
        base_z = h_min-old.h_min if shallow else 0.
        s, d, h = geo.s[ids], geo.d[ids], height[ids]-base_z
        # Robust bounds prevent an isolated bridge from growing the search on
        # every frame. A coherent body must retain its vertical extent.
        bounds = np.quantile(np.column_stack((s,d,h)), [.02,.98], axis=0)
        size = bounds[1]-bounds[0]
        reference = t.resolved_size
        if (size[0] > min(2., max(.6, 1.25*reference[0]+.1))
                or size[1] > min(1.5, max(.3, (3. if shallow else 1.5)*reference[1]+.1))
                or size[2] < (.2 if shallow else .5)*reference[2] or size[2] > (.6 if shallow else min(3.3,1.5*reference[2]+.2))):
            return None
        if shallow and size[1] < .2:
            return None
        inner = np.all((np.column_stack((s,d,h)) >= bounds[0])
                       & (np.column_stack((s,d,h)) <= bounds[1]), axis=1)
        ids = ids[inner]
        if len(ids) < min_points or len(np.unique(ring[ids])) < min_rings:
            return None
        return Candidate(-t.id, float(bounds[0,0]), float(bounds[1,0]),
                         float(bounds[0,1]), float(bounds[1,1]), float(bounds[0,2]),float(bounds[1,2]),
                         float(np.linalg.norm(lev[ids],axis=1).min()),len(ids),len(np.unique(ring[ids])),
                         float(np.median(geo.d[ids])), t.score,
                         reasons=['продолжение подтверждённого тела по локальным точкам'], point_index=ids,
                         low=old.low if shallow else False,
                         rail_local=old.rail_local if shallow else False)

    @staticmethod
    def _predict_candidate(t: Track, shift: float) -> None:
        """Keep the reported box in the current frame, including on dropouts.

        Do not mutate a Candidate held by a previous FrameResult. Its point
        indices belong to an older cloud and must not be used for new labels.
        """
        c = t.cand
        s_near, s_far = c.s_near + shift, c.s_far + shift
        # Recover the perpendicular distance from the previous nearest return.
        # This preserves Euclidean range instead of subtracting a longitudinal
        # displacement from it (which is wrong for off-axis objects).
        perpendicular2 = max(0.0, c.distance * c.distance - c.s_near * c.s_near)
        t.cand = replace(c, s_near=s_near, s_far=s_far,
                         distance=float(np.sqrt(max(s_near, 0.0) ** 2 + perpendicular2)),
                         point_index=None)

    def update(self, cands: list[Candidate], delta: float, *, allow_single: bool = False,
               path_prior: TrackModel | None = None, path_current: TrackModel | None = None,
               lev: np.ndarray | None = None, ring: np.ndarray | None = None,
               geo: FrameGeometry | None = None) -> list[Track]:
        tc = self.cfg.tracking
        self.frame += 1
        good = [c for c in cands if c.rejected in (None, 'low_score')
                and (not c.single_beam or allow_single and abs(delta) <= 0.01)]
        # predict
        for t in self.tracks:
            shift = t.v_rel - delta
            t.s += shift
            self._predict_candidate(t, shift)
            if (path_prior is not None and path_current is not None
                    and t.s <= min(path_prior.observed_until,path_current.observed_until)+10.):
                # d/h are coordinates relative to a fitted path. A correction
                # of that path is not lateral motion of the physical object.
                # Rebase the previous box before associating fresh returns.
                mid = .5*(t.cand.s_near+t.cand.s_far)
                dx,df,dz = self._calibration_shift(t,path_prior,path_current)
                lateral = float(np.interp(mid,path_prior.s,path_prior.c)-np.interp(mid+df,path_current.s,path_current.c))+dx
                height = 0. if t.cand.rail_local else float(np.interp(mid,path_prior.s,path_prior.z)-np.interp(mid+df,path_current.s,path_current.z))+dz
                if np.isfinite(lateral) and np.isfinite(height):
                    t.s += df
                    self._predict_candidate(t,df)
                    if t.level_height:
                        t.level_height = tuple(v+dz for v in t.level_height)
                    t.d += lateral
                    t.lat_hist = [v+lateral for v in t.lat_hist]
                    t.cand = replace(t.cand,d_min=t.cand.d_min+lateral,d_max=t.cand.d_max+lateral,
                                     lateral=t.cand.lateral+lateral,
                                     h_min=t.cand.h_min+height,h_max=t.cand.h_max+height)
        local_owner = {}
        if lev is not None and ring is not None and geo is not None:
            for ti,t in enumerate(self.tracks):
                local = self._local_continuation(t,lev,ring,geo)
                if local is not None:
                    local_owner[len(good)] = ti
                    good.append(local)
        # greedy association by cost
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = tc.gate + 0.01 * max(t.s, 0.0)
            for ci, c in enumerate(good):
                if ci in local_owner and local_owner[ci] != ti:
                    continue
                ds = c.s_near - t.s
                dd = c.lateral - t.d
                pair_gate = min(gate, .6+.01*max(t.s, 0.)) if t.cand.low and c.low else gate
                # Nearest centres alone can merge an occluded person with a
                # neighbouring rail fragment and inherit its confirmed state.
                lateral_gap = max(c.d_min-t.cand.d_max, t.cand.d_min-c.d_max, 0.0)
                if (abs(ds) < pair_gate and abs(dd) < tc.lateral_gate + 0.004 * max(t.s, 0.0)
                        and lateral_gap <= .2+.003*max(t.s, 0.0)
                        and self._shape_compatible(t,c)
                        and (c.rejected is None or self._weak_continuation(t, c))):
                    pairs.append((abs(ds) / pair_gate + abs(dd) / tc.lateral_gate + (.05 if ci in local_owner else 0.), ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t = self.tracks[ti]
            c = good[ci]
            if c.rejected == 'low_score':
                c = replace(c, rejected=None, reasons=c.reasons+['продолжение подтверждённого тела по плотным согласованным измерениям'])
            # objects may move on their own (a walking person): follow the measurement
            innov = c.s_near - t.s
            t.v_rel = float(np.clip(t.v_rel + 0.4 * innov, -0.4, 0.4))
            t.s = c.s_near
            t.d = c.lateral
            t.lat_hist.append(c.lateral)
            del t.lat_hist[:-10]
            t.cand = c
            t.width_seen = max(t.width_seen, c.d_max-c.d_min)
            t.height_seen = max(t.height_seen, c.size[2])
            if lev is not None and c.point_index is not None:
                t.level_height = tuple(np.quantile(lev[c.point_index,2],[.02,.98]))
            t.resolved_history.append(int(ci in local_owner or self._resolved(c)))
            if ci not in local_owner and self._resolved(c):
                t.resolved_size = (*c.size[:2],max(c.size[2],t.resolved_size[2] if t.resolved_size else 0.))
                t.resolved_points = c.n_points
            t.score = 0.6 * t.score + 0.4 * c.score
            t.history.append(1)
            t.body_history.append(int(self._low_body(c)))
            t.wide_body_history.append(int(self._wide_low_body(c)))
            t.strict_history.append(int(c.strict_rail_support))
            t.hits += 1
            t.misses = 0
        for ti, t in enumerate(self.tracks):
            t.age += 1
            if ti not in used_t:
                t.history.append(0)
                t.body_history.append(0)
                t.wide_body_history.append(0)
                t.strict_history.append(0)
                t.resolved_history.append(0)
                t.misses += 1
            del t.history[:-max(tc.low_single_confirm_window, 12)]
            del t.body_history[:-max(tc.low_body_confirm_window, 12)]
            del t.wide_body_history[:-max(tc.low_body_confirm_window, 12)]
            del t.strict_history[:-max(tc.low_confirm_window, 12)]
            del t.resolved_history[:-12]
            win, need = self._confirmation(t)
            body = self._low_body(t.cand)
            evidence = t.body_history if body else (t.strict_history if t.cand.low and not t.cand.single_beam else t.history)
            recent = evidence[-win:]
            strict = not body or sum(bool(b and s) for b,s in
                                     zip(t.body_history[-win:],t.strict_history[-win:])) >= min(2,need)
            strict |= body and sum(t.wide_body_history[-win:]) >= need
            stable = t.s <= tc.far_distance or len(t.lat_hist) < 3 or float(np.std(t.lat_hist[-need:])) < tc.far_lateral_std
            if not t.confirmed and sum(recent) >= need and stable and strict and self._has_body_extent(t):
                t.confirmed = True
                t.confirmed_at_s = t.s
                t.confirmed_frame = self.frame
        # drop stale tracks
        # A local crop and a normal cluster may describe the same body. Exact
        # current-frame evidence can merge those identities; predicted boxes
        # alone cannot merge neighbouring or temporarily occluded objects.
        unique = []
        for t in sorted(self.tracks,key=lambda t:(not t.confirmed,-t.hits,t.id)):
            if t.misses == 0 and any(o.misses == 0 and self._same_returns(t.cand,o.cand) for o in unique):
                continue
            unique.append(t)
        self.tracks = unique
        keep = []
        for t in self.tracks:
            limit = tc.drop_misses if t.confirmed else (3 if t.cand.low or t.s > tc.far_distance else 2)
            if t.misses <= limit and t.s > -2.0:
                keep.append(t)
        self.tracks = keep
        # new tracks
        for ci, c in enumerate(good):
            if ci in used_c or ci in local_owner or c.rejected is not None:
                continue
            if any(t.misses == 0 and self._same_returns(t.cand,c) for t in self.tracks):
                continue
            t = Track(self.next_id, c.s_near, c.lateral, c, [1], 1, 0, 0, False, c.s_near,
                      first_frame=self.frame, score=c.score, lat_hist=[c.lateral],
                      body_history=[int(self._low_body(c))],strict_history=[int(c.strict_rail_support)], width_seen=c.d_max-c.d_min,
                      resolved_history=[int(self._resolved(c))],height_seen=c.size[2])
            t.wide_body_history = [int(self._wide_low_body(c))]
            if self._resolved(c):
                t.resolved_size = (*c.size[:2],max(c.size[2],t.resolved_size[2] if t.resolved_size else 0.))
                t.resolved_points = c.n_points
            if lev is not None and c.point_index is not None:
                t.level_height = tuple(np.quantile(lev[c.point_index,2],[.02,.98]))
            self.next_id += 1
            _, need = self._confirmation(t)
            if need <= 1 and self._has_body_extent(t):
                t.confirmed = True
                t.confirmed_at_s = t.s
                t.confirmed_frame = self.frame
            self.tracks.append(t)
        return self.tracks
