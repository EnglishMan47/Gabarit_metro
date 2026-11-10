// 3D view. Data arrive in the levelled lidar frame: x lateral (left +),
// f forward, z up (rail head = 0). Three.js uses X = f, Y = z, Z = -x.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { axisDistanceLabels, sideAxisLabelPosition, sceneRange } from '/static/js/winner_view_geometry.js';

const COLORS = {
  other: new THREE.Color('#5b7a99'),
  ground: new THREE.Color('#4b5563'),
  env: new THREE.Color('#fb923c'),
  cand: new THREE.Color('#f59e0b'),
  obst: new THREE.Color('#ef4444'),
};
const tv = (x, f, z) => new THREE.Vector3(f, z, -x);

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

function labelSprite(text, color = '#dbe5f0', scale = 1) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.userData.labelCanvas = c;
  s.userData.screenHeight = 24 * scale;
  updateLabel(s, text, color);
  s.scale.set(4 * scale, 1 * scale, 1);
  s.renderOrder = 10;
  return s;
}

function updateLabel(sprite, text, color) {
  if (sprite.userData.labelText === text && sprite.userData.labelColor === color) return;
  const c = sprite.userData.labelCanvas, g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.font = 'bold 34px Segoe UI, system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = 'rgba(8,12,17,0.72)';
  const w = Math.min(250, g.measureText(text).width + 24);
  g.beginPath(); g.roundRect((256 - w) / 2, 8, w, 48, 10); g.fill();
  g.fillStyle = color; g.fillText(text, 128, 33, 236);
  sprite.material.map.needsUpdate = true;
  sprite.userData.labelText = text;
  sprite.userData.labelColor = color;
}

export class Scene3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x070a0e, 120, 320);
    this.renderer.setClearColor(0x070a0e, 1);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enabled = false;
    this.camMode = 'cab';
    this.colorMode = 'class';
    this.layers = { points: true, envelope: true, path: true, learned: true, walls: false, markers: true, boxes: true, candidates: false };
    this.pointSize = 2;
    this.frame = null;
    this.status = 'idle';
    this._clock = 0;
    this._boxes = new Map();
    this._labels = new Map();
    this._freePose = null;
    this.targetTrackId = null;
    this._unitBox = new THREE.BoxGeometry(1, 1, 1);
    this._unitEdges = new THREE.EdgesGeometry(this._unitBox);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    // points
    this.points = new THREE.Points(new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true }));
    this.hot = new THREE.Points(new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true, depthTest: false }));
    this.hot.renderOrder = 5;
    this.scene.add(this.points, this.hot);
    // groups rebuilt every frame
    this.envGroup = new THREE.Group();
    this.pathGroup = new THREE.Group();
    this.wallGroup = new THREE.Group();
    this.boxGroup = new THREE.Group();
    this.markGroup = new THREE.Group();
    this.scene.add(this.envGroup, this.pathGroup, this.wallGroup, this.boxGroup, this.markGroup);
    // train (sensor carrier)
    const train = new THREE.Mesh(new THREE.BoxGeometry(6, 3.2, 2.7),
      new THREE.MeshBasicMaterial({ color: 0x1d4ed8, transparent: true, opacity: 0.35 }));
    train.position.copy(tv(0, -3.2, 1.45));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(train.geometry), new THREE.LineBasicMaterial({ color: 0x60a5fa }));
    train.add(edges);
    this.train = train;
    this.scene.add(train);
    const sensor = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.14, 16), new THREE.MeshBasicMaterial({ color: 0x38bdf8 }));
    sensor.position.copy(tv(0, 0, 1.1));
    this.sensor = sensor;
    this.scene.add(sensor);

    this.raycaster = new THREE.Raycaster();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this.setCamera('cab');
    const loop = (t) => { this._tick(t); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  _resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  setTheme(light) {
    // background + fog, and tunnel points a bit darker on the light background
    const bg = light ? 0xe4eaef : 0x070a0e;
    this.renderer.setClearColor(bg, 1);
    this.scene.fog.color.set(bg);
    COLORS.other.set(light ? '#4c6883' : '#5b7a99');
    COLORS.ground.set(light ? '#7b8594' : '#4b5563');
    this.light = light;
    if (this.lastPoints) this._fillPoints(this.lastPoints);
  }

  setCamera(mode) {
    if (this.camMode === 'free') this._freePose = {
      position: this.camera.position.clone(), target: this.controls.target.clone(), up: this.camera.up.clone(),
    };
    this.camMode = mode;
    this.controls.enabled = mode === 'free';
    if (mode === 'free') {
      if (this._freePose) {
        this.camera.position.copy(this._freePose.position); this.controls.target.copy(this._freePose.target);
        this.camera.up.copy(this._freePose.up);
      } else {
        this.camera.up.set(0, 1, 0);
        this.camera.position.set(-18, 16, 14);
        this.controls.target.set(40, 0, 0);
      }
      this.controls.update();
    }
    this._placeCamera(true);
  }

  resetCamera() {
    if (this.camMode === 'free') { this.camMode = ''; this._freePose = null; this.setCamera('free'); }
    else this._placeCamera(true);
  }

  focusObstacle(id) {
    this.targetTrackId = id;
    this.setCamera('target');
  }

  clear() {
    this.frame = this.lastPoints = null;
    this.targetTrackId = null;
    this.points.geometry.setDrawRange(0, 0); this.hot.geometry.setDrawRange(0, 0);
    [this.envGroup, this.pathGroup, this.wallGroup, this.markGroup].forEach((g) => this._clear(g));
    this._labels.clear();
    for (const entry of this._boxes.values()) {
      this.boxGroup.remove(entry.group); entry.edge.material.dispose(); entry.fill.material.dispose();
    }
    this._boxes.clear();
  }

  _placeCamera(force = false) {
    const m = this.camMode;
    // the cab camera sits inside the train model: its edges would cross the view
    this.train.visible = m !== 'cab';
    if (m === 'free') return;
    const h = (this.frame && this.frame.calibration && this.frame.calibration.height_m) || 1.1;
    const range = sceneRange(this.frame);
    let pos, look;
    if (m === 'cab') { pos = tv(0, -1.5, h + 1.4); look = tv(0, 70, 0.4); }
    else if (m === 'chase') { pos = tv(-4, -16, 7); look = tv(0, 45, 0.5); }
    else if (m === 'top') { pos = tv(0, range / 2, range * 1.12); look = tv(0, range / 2, 0); }
    else if (m === 'side') { pos = tv(-range * .7, range / 2, 10); look = tv(0, range / 2, 1); }
    else if (m === 'target') {
      const ob = this._targetObstacle();
      const s = ob ? ob.s_m : 60;
      const tr = this.frame && this.frame.track;
      const c = tr ? interp(tr.s, tr.c, s) : 0, z = tr ? interp(tr.s, tr.z, s) : 0;
      pos = tv(c - 5, s - 11, z + 4.5); look = tv(c + (ob ? ob.lateral_m : 0), s, z + 0.8);
    }
    if (m === 'top') this.camera.up.set(1, 0, 0); else this.camera.up.set(0, 1, 0);
    this.camera.position.copy(pos);
    this.camera.lookAt(look);
  }

  _targetObstacle() {
    if (!this.frame) return null;
    const all = [...(this.frame.obstacles || []), ...(this.frame.pending || [])];
    const selected = all.find((o) => o.track_id === this.targetTrackId);
    if (selected) return selected;
    if (all.length) { this.targetTrackId = all[0].track_id; return all[0]; }
    const syn = this.frame.synthetic || [];
    return syn.length ? { s_m: syn[0].s, lateral_m: syn[0].d } : null;
  }

  setLayer(name, on) {
    this.layers[name] = on;
    if (this.frame) this._rebuild();
  }

  setColorMode(m) { this.colorMode = m; if (this.lastPoints) this._fillPoints(this.lastPoints); }
  setPointSize(v) { this.pointSize = v; this.points.material.size = v; this.hot.material.size = v * 2.2; }

  update(frame, pts) {
    this.frame = frame;
    this.status = frame.status;
    this.lastPoints = pts;
    this.scene.fog.near = sceneRange(frame) * .85;
    this.scene.fog.far = sceneRange(frame) * 1.8;
    this._fillPoints(pts);
    this._rebuild();
    this._placeCamera();
  }

  _fillPoints(p) {
    const n = p.n;
    const attributes = (geometry, count) => {
      if (!geometry.getAttribute('position') || geometry.getAttribute('position').count < count) {
        const capacity = Math.max(count, geometry.getAttribute('position')?.count * 2 || 1024);
        if (geometry.getAttribute('position')) geometry.dispose();
        for (const name of ['position', 'color']) geometry.setAttribute(name,
          new THREE.BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage));
      }
      geometry.setDrawRange(0, count);
      return [geometry.getAttribute('position').array, geometry.getAttribute('color').array];
    };
    const [pos, col] = attributes(this.points.geometry, n);
    let hot = 0;
    for (let i = 0; i < n; i++) if (p.cls[i] >= 3) hot++;
    const [hpos, hcol] = attributes(this.hot.geometry, hot);
    const c = new THREE.Color();
    let k = 0;
    for (let i = 0; i < n; i++) {
      const x = p.xyz[3 * i], f = p.xyz[3 * i + 1], z = p.xyz[3 * i + 2];
      const cls = p.cls[i];
      if (this.colorMode === 'height') c.setHSL(0.66 - Math.min(Math.max((z + 0.5) / 5, 0), 1) * 0.66, 0.85, 0.55);
      else if (this.colorMode === 'intensity') c.setHSL(0.62 - Math.min(p.inten[i] / 90, 1) * 0.62, 0.9, 0.5);
      else if (cls === 1) c.copy(COLORS.ground);
      else if (cls === 2) c.copy(COLORS.env);
      else if (cls === 0) {   // tunnel: brightness by height (darker points on the light theme)
        const g = (this.light ? 0.2 : 0.45) + Math.min(Math.max(z / 5, 0), 1) * (this.light ? 0.3 : 0.35);
        c.setRGB(0.36 * g + 0.1, 0.48 * g + 0.1, 0.6 * g + 0.12);
      }
      if (cls >= 3) {
        const cc = cls === 4 ? COLORS.obst : COLORS.cand;
        hpos[3 * k] = f; hpos[3 * k + 1] = z; hpos[3 * k + 2] = -x;
        hcol[3 * k] = cc.r; hcol[3 * k + 1] = cc.g; hcol[3 * k + 2] = cc.b; k++;
        c.copy(cc);
      }
      pos[3 * i] = f; pos[3 * i + 1] = z; pos[3 * i + 2] = -x;
      col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b;
    }
    const g = this.points.geometry;
    g.getAttribute('position').needsUpdate = g.getAttribute('color').needsUpdate = true;
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), sceneRange(this.frame) + 50);
    const hg = this.hot.geometry;
    hg.getAttribute('position').needsUpdate = hg.getAttribute('color').needsUpdate = true;
    hg.boundingSphere = g.boundingSphere;
    this.points.visible = this.layers.points;
    this.hot.visible = this.layers.points;
  }

  _clear(group) {
    for (const ch of [...group.children]) {
      group.remove(ch);
      ch.traverse((o) => { o.geometry && o.geometry.dispose(); if (o.material) { o.material.map && o.material.map.dispose(); o.material.dispose(); } });
    }
  }

  _rebuild() {
    const fr = this.frame, tr = fr.track;
    [this.envGroup, this.pathGroup, this.wallGroup].forEach((g) => this._clear(g));
    const usedLabels = new Set(), usedBoxes = new Set();
    const label = (key, text, color, position, scale = 1) => {
      usedLabels.add(key);
      let sprite = this._labels.get(key);
      if (!sprite) { sprite = labelSprite(text, color, scale); this._labels.set(key, sprite); this.markGroup.add(sprite); }
      updateLabel(sprite, text, color); sprite.position.copy(position);
      sprite.userData.labelKey = key;
      return sprite;
    };
    const W = (fr.envelope && fr.envelope.width_m) || 2.1, H = (fr.envelope && fr.envelope.height_m) || 3.0;
    const obs = tr.observed_until || 0;
    const current = Number.isFinite(tr.current_observed_until) ? tr.current_observed_until :
      Number.isFinite(fr.current_observed_until_m) ? fr.current_observed_until_m : obs;
    const fresh = Math.max(0, Math.min(obs, current));
    const reliable = Number.isFinite(tr.uncertain_from) ? Math.max(0, Math.min(fresh, tr.uncertain_from)) : fresh;
    const held = fresh < obs - .1;
    const alarm = fr.status === 'OBSTACLE', check = fr.status === 'CHECK';
    const envColor = alarm ? 0xef4444 : check ? 0xf59e0b : 0x22c55e;
    const S = tr.s, C = tr.c, Z = tr.z;
    // beyond the observed range the path is only extrapolated and jitters from
    // frame to frame: draw only a short grey hint of its extrapolated tail (the
    // radar strip shows the unobserved part as hatching)
    const sDraw = Math.min(fr.analysis_range_m || 200, S[S.length - 1], Math.max(obs, 20) + 15);
    const segments = [];
    const segment = (start, end, kind) => {
      start = Math.max(3, start); end = Math.min(sDraw, end);
      if (end > start) segments.push({ start, end, kind });
    };
    segment(3, reliable, 'measured');
    segment(reliable, fresh, 'uncertain');
    segment(fresh, obs, 'held');
    segment(obs, sDraw, 'forecast');
    // --- envelope: translucent walls + roof edges, dimmer beyond the observed range
    if (this.layers.envelope) {
      const build = (s0, s1, opacity, color, dashed = false) => {
        const verts = [], idx = [];
        const lines = [];
        const pts = [];
        const stations = [];
        for (let s = s0; s < s1; s += 2) stations.push(s);
        stations.push(s1);
        for (const s of stations) {
          const c = interp(S, C, s), z = interp(S, Z, s);
          pts.push([s, c, z]);
        }
        pts.forEach(([s, c, z], i) => {
          const corners = [[c + W / 2, z], [c + W / 2, z + H], [c - W / 2, z + H], [c - W / 2, z]];
          corners.forEach(([x, zz]) => { const v = tv(x, s, zz); verts.push(v.x, v.y, v.z); });
          if (i > 0) {
            const a = (i - 1) * 4, b = i * 4;
            for (const [p, q] of [[0, 1], [1, 2], [2, 3]]) idx.push(a + p, b + p, b + q, a + p, b + q, a + q);
          }
        });
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
        g.setIndex(idx);
        const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
        this.envGroup.add(mesh);
        for (const k of [0, 1, 2, 3]) {
          const l = pts.map(([s, c, z]) => {
            const cs = [[c + W / 2, z], [c + W / 2, z + H], [c - W / 2, z + H], [c - W / 2, z]][k];
            return tv(cs[0], s, cs[1]);
          });
          const material = dashed ? new THREE.LineDashedMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 5), dashSize: .8, gapSize: .6 }) :
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 5) });
          const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(l), material);
          if (dashed) line.computeLineDistances();
          lines.push(line);
        }
        lines.forEach((l) => this.envGroup.add(l));
        // cross-section frames every 25 m
        for (let s = Math.ceil(s0 / 25) * 25; s <= s1; s += 25) {
          const c = interp(S, C, s), z = interp(S, Z, s);
          const ring = [[c + W / 2, z], [c + W / 2, z + H], [c - W / 2, z + H], [c - W / 2, z], [c + W / 2, z]].map(([x, zz]) => tv(x, s, zz));
          const material = dashed ? new THREE.LineDashedMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 4), dashSize: .8, gapSize: .6 }) :
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 4) });
          const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), material);
          if (dashed) line.computeLineDistances();
          this.envGroup.add(line);
        }
      };
      for (const part of segments) {
        const measured = part.kind === 'measured', forecast = part.kind === 'forecast';
        const color = forecast ? 0x64748b : measured ? envColor : alarm ? 0xef4444 : 0xf59e0b;
        build(part.start, part.end, measured ? .09 : forecast ? .025 : .055, color, !measured);
      }
    }
    // --- path centre and rails
    if (this.layers.path) {
      const mk = (pts, color, dashed) => {
        const g = new THREE.BufferGeometry().setFromPoints(pts);
        const m = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 1.5, gapSize: 1.0 }) : new THREE.LineBasicMaterial({ color });
        const l = new THREE.Line(g, m); if (dashed) l.computeLineDistances(); return l;
      };
      for (const part of segments) {
        const centre = [], rl = [], rr = [], stations = [];
        for (let s = part.start; s < part.end; s += 1) stations.push(s);
        stations.push(part.end);
        for (const s of stations) {
          const c = interp(S, C, s), z = interp(S, Z, s);
          centre.push(tv(c, s, z + .01)); rl.push(tv(c + .8, s, z)); rr.push(tv(c - .8, s, z));
        }
        const measured = part.kind === 'measured', forecast = part.kind === 'forecast';
        const color = measured ? 0x38bdf8 : forecast ? 0x64748b : 0xf59e0b;
        this.pathGroup.add(mk(centre, color, true), mk(rl, measured ? 0x94a3b8 : color, !measured), mk(rr, measured ? 0x94a3b8 : color, !measured));
      }
    }
    const learned = fr.models?.far_path?.refined_path;
    if (this.layers.learned && learned?.s?.length) {
      const points = [];
      learned.s.forEach((s, i) => {
        if (s <= obs && [s, learned.c[i], learned.z[i]].every(Number.isFinite))
          points.push(tv(learned.c[i], s, learned.z[i] + .06));
      });
      if (points.length > 1) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineDashedMaterial({ color: 0xc084fc, dashSize: 1.2, gapSize: .7, depthTest: false }));
        line.computeLineDistances(); line.renderOrder = 4; this.pathGroup.add(line);
      }
    }
    if (this.layers.walls) {
      for (const [key, color] of [['wl', 0xf87171], ['wr', 0x60a5fa]]) {
        const pts = [];
        (tr[key] || []).forEach((v, i) => { if (v !== null && Number.isFinite(v)) pts.push(tv(v, S[i], interp(S, Z, S[i]) + 1.2)); });
        if (pts.length > 1) this.wallGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color })));
      }
    }
    // --- distance markers
    if (this.layers.markers) {
      const pairs = [];
      for (let s = 25; s <= sDraw; s += 25) {
        const c = interp(S, C, s), z = interp(S, Z, s);
        pairs.push({ source_s_m: s, left_xyz: [c + .8, s, z], right_xyz: [c - .8, s, z] });
      }
      if (obs >= 4) {
        const c = interp(S, C, obs), z = interp(S, Z, obs);
        pairs.push({ source_s_m: obs, left_xyz: [c + .8, obs, z], right_xyz: [c - .8, obs, z] });
        pairs.sort((a, b) => a.source_s_m - b.source_s_m);
      }
      for (const pair of axisDistanceLabels(pairs, obs)) {
        const s = pair.source_s_m, position = sideAxisLabelPosition(pair, [-W / 2, W / 2]);
        label(Math.abs(s - obs) < .001 ? 'distance:observed' : `distance:${s}`,
          `${Number.isInteger(s) ? s : s.toFixed(1)} м`, s <= reliable ? '#b8cddd' : s <= obs ? '#fbbf24' : '#7c8fa3', tv(...position), .9);
      }
      if (obs >= 4) {
        const c = interp(S, C, obs), z = interp(S, Z, obs);
        const uncertain = reliable < fresh - .1;
        label('observed-end', `${held || fr.envelope_held ? 'Удержание' : uncertain ? 'Неуверенно' : 'Обзор'} ${obs.toFixed(0)} м`,
          held || fr.envelope_held || uncertain ? '#fbbf24' : '#93e8bd', tv(c, obs, z + H + .6), 1.05);
        if (held && fresh >= 4) {
          const fc = interp(S, C, fresh), fz = interp(S, Z, fresh);
          label('fresh-end', `Измерено ${fresh.toFixed(0)} м`, reliable < fresh - .1 ? '#fbbf24' : '#93e8bd', tv(fc, fresh, fz + H + .6), 1.05);
        }
      }
    }
    // --- boxes of obstacles / pending tracks / synthetic ground truth
    if (this.layers.boxes) {
      const addBox = (key, b, color, text, dashed = false, heading = 0) => {
        if (!b || ![...b.s, ...b.d, ...b.h].every(Number.isFinite)) return;
        usedBoxes.add(key);
        const s0 = b.s[0], s1 = Math.max(b.s[1], s0 + 0.3);
        const d0 = b.d[0], d1 = b.d[1], h0 = b.h[0], h1 = b.h[1];
        const center = (s0 + s1) / 2;
        const c = interp(S, C, center), z = interp(S, Z, center);
        const cx = c + (d0 + d1) / 2, cz = z + (h0 + h1) / 2;
        let entry = this._boxes.get(key);
        if (!entry || entry.dashed !== dashed) {
          if (entry) { this.boxGroup.remove(entry.group); entry.edge.material.dispose(); entry.fill.material.dispose(); }
          const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: .3, gapSize: .2, depthTest: false })
            : new THREE.LineBasicMaterial({ color, depthTest: false });
          const edge = new THREE.LineSegments(this._unitEdges, mat);
          if (dashed) edge.computeLineDistances();
          const fill = new THREE.Mesh(this._unitBox,
            new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .12, depthWrite: false }));
          const group = new THREE.Group(); group.add(edge, fill); group.renderOrder = 6;
          entry = { group, edge, fill, dashed }; this._boxes.set(key, entry); this.boxGroup.add(group);
        }
        entry.group.position.copy(tv(cx, center, cz)); entry.group.rotation.y = heading;
        entry.group.scale.set(s1 - s0 + .12, Math.max(h1 - h0, .1) + .12, Math.max(d1 - d0, .1) + .12);
        entry.edge.material.color.set(color); entry.fill.material.color.set(color); entry.fill.visible = !dashed;
        entry.edge.renderOrder = 6; entry.fill.renderOrder = 5;
        if (text) label(`box:${key}`, text, dashed ? '#e3c5ff' : color === 0xef4444 ? '#fecaca' : '#fde68a', tv(cx, s0, z + h1 + .9), 1.15);
      };
      (fr.obstacles || []).forEach((o) => addBox(`track:${o.track_id}`, o.box, 0xef4444, `${o.missed ? '≈ ' : ''}${o.distance_m.toFixed(1)} м · №${o.track_id}`, !!o.missed));
      (fr.pending || []).forEach((o) => addBox(`track:${o.track_id}`, o.box, 0xf59e0b, `? ${o.distance_m.toFixed(0)} м`, true));
      if (this.layers.candidates) (fr.candidates || []).filter((c) => c.rejected).forEach((c, i) => addBox(`candidate:${i}`, c.box, 0x64748b, null, true));
      (fr.synthetic || []).forEach((o, i) => {
        const w = o.width, h = o.height, l = o.length;
        const slope = interp(S, C, o.s + 0.5) - interp(S, C, o.s - 0.5);
        const heading = Math.atan(slope) + (o.yaw_deg || 0) * Math.PI / 180;
        addBox(`synthetic:${i}`, { s: [o.s - l / 2, o.s + l / 2], d: [o.d - w / 2, o.d + w / 2], h: [o.base, o.base + h] }, 0xc084fc, `тест ${o.s.toFixed(0)} м`, true, heading);
      });
    }
    for (const [key, entry] of this._boxes) if (!usedBoxes.has(key)) {
      this.boxGroup.remove(entry.group); entry.edge.material.dispose(); entry.fill.material.dispose(); this._boxes.delete(key);
    }
    for (const [key, sprite] of this._labels) if (!usedLabels.has(key)) {
      this.markGroup.remove(sprite); sprite.material.map.dispose(); sprite.material.dispose(); this._labels.delete(key);
    }
  }

  // screen point -> distance along the track / lateral offset on the corridor floor
  pickTrack(clientX, clientY) {
    if (!this.frame) return null;
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const p = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(plane, p)) return null;
    const s = p.x, x = -p.z;
    const tr = this.frame.track;
    if (s < 5 || s > (this.frame.analysis_range_m || 200)) return null;
    return { s, d: x - interp(tr.s, tr.c, s) };
  }

  _tick(t) {
    this._clock = t;
    if (this.controls.enabled) this.controls.update();
    this._updateLabels();
    this.renderer.render(this.scene, this.camera);
  }

  _updateLabels() {
    // Reserve screen space for actual detections first. Perspective compresses
    // distant axis stations into the horizon, even with constant-size labels.
    const width = this.canvas.clientWidth, height = Math.max(this.canvas.clientHeight, 1);
    this.camera.updateMatrixWorld();
    const visible = [];
    for (const [key, sprite] of this._labels) {
      const distance = this.camera.position.distanceTo(sprite.position);
      const size = 2 * distance * Math.tan(this.camera.fov * Math.PI / 360)
        * sprite.userData.screenHeight / height;
      sprite.scale.set(4 * size, size, 1);
      sprite.visible = false;
      const projected = sprite.position.clone().project(this.camera);
      if (![projected.x, projected.y, projected.z].every(Number.isFinite) ||
          projected.z < -1 || projected.z > 1 || Math.abs(projected.x) > 1.05 || Math.abs(projected.y) > 1.05) continue;
      const pixelHeight = sprite.userData.screenHeight;
      const x = (projected.x + 1) * .5 * width, y = (1 - projected.y) * .5 * height;
      const rect = [x - 2 * pixelHeight - 4, y - .5 * pixelHeight - 3,
                    x + 2 * pixelHeight + 4, y + .5 * pixelHeight + 3];
      const axis = key.startsWith('distance:');
      const priority = key.startsWith('box:track:') ? 4 : key.startsWith('box:') ? 3 : ['observed-end', 'fresh-end'].includes(key) ? 2 : 1;
      visible.push({sprite, rect, axis, priority, distance});
    }
    visible.sort((a, b) => b.priority - a.priority || a.distance - b.distance);
    const occupied = [];
    let axes = 0;
    const axisLimit = ['cab', 'chase', 'target'].includes(this.camMode) ? 2 : this.camMode === 'top' ? 8 : 5;
    for (const item of visible) {
      if (item.axis && axes >= axisLimit) continue;
      const r = item.rect;
      if (occupied.some((p) => r[0] < p[2] && r[2] > p[0] && r[1] < p[3] && r[3] > p[1])) continue;
      item.sprite.visible = true;
      occupied.push(r);
      if (item.axis) axes++;
    }
  }
}
