// 3D view. Data arrive in the levelled lidar frame: x lateral (left +),
// f forward, z up (rail head = 0). Three.js uses X = f, Y = z, Z = -x.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

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
  g.font = 'bold 34px Segoe UI, system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = 'rgba(8,12,17,0.72)';
  const w = Math.min(250, g.measureText(text).width + 24);
  g.beginPath(); g.roundRect((256 - w) / 2, 8, w, 48, 10); g.fill();
  g.fillStyle = color; g.fillText(text, 128, 33);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(4 * scale, 1 * scale, 1);
  s.renderOrder = 10;
  return s;
}

export class Scene3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x070a0e, 120, 320);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enabled = false;
    this.camMode = 'cab';
    this.colorMode = 'class';
    this.layers = { points: true, envelope: true, path: true, walls: false, markers: true, boxes: true, candidates: false };
    this.pointSize = 2;
    this.frame = null;
    this.status = 'idle';
    this._clock = 0;

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

  setCamera(mode) {
    this.camMode = mode;
    this.controls.enabled = mode === 'free';
    if (mode === 'free') {
      this.camera.position.set(-18, 16, 14);
      this.controls.target.set(40, 0, 0);
    }
    this._placeCamera(true);
  }

  _placeCamera(force = false) {
    const m = this.camMode;
    if (m === 'free') return;
    const h = (this.frame && this.frame.calibration && this.frame.calibration.height_m) || 1.1;
    let pos, look;
    if (m === 'cab') { pos = tv(0, -1.5, h + 1.4); look = tv(0, 70, 0.4); }
    else if (m === 'chase') { pos = tv(-4, -16, 7); look = tv(0, 45, 0.5); }
    else if (m === 'top') { pos = tv(0, 85, 150); look = tv(0, 85, 0); }
    else if (m === 'side') { pos = tv(-70, 70, 10); look = tv(0, 70, 1); }
    else if (m === 'target') {
      const ob = this._targetObstacle();
      const s = ob ? ob.s_m : 60;
      const tr = this.frame && this.frame.track;
      const c = tr ? interp(tr.s, tr.c, s) : 0;
      pos = tv(c - 5, s - 11, 4.5); look = tv(c + (ob ? ob.lateral_m : 0), s, 0.8);
    }
    if (m === 'top') this.camera.up.set(1, 0, 0); else this.camera.up.set(0, 1, 0);
    this.camera.position.copy(pos);
    this.camera.lookAt(look);
  }

  _targetObstacle() {
    if (!this.frame) return null;
    const all = [...(this.frame.obstacles || []), ...(this.frame.pending || [])];
    if (all.length) return all[0];
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
    this._fillPoints(pts);
    this._rebuild();
    this._placeCamera();
  }

  _fillPoints(p) {
    const n = p.n;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    let hot = 0;
    for (let i = 0; i < n; i++) if (p.cls[i] >= 3) hot++;
    const hpos = new Float32Array(hot * 3), hcol = new Float32Array(hot * 3);
    const c = new THREE.Color();
    let k = 0;
    for (let i = 0; i < n; i++) {
      const x = p.xyz[3 * i], f = p.xyz[3 * i + 1], z = p.xyz[3 * i + 2];
      const cls = p.cls[i];
      if (this.colorMode === 'height') c.setHSL(0.66 - Math.min(Math.max((z + 0.5) / 5, 0), 1) * 0.66, 0.85, 0.55);
      else if (this.colorMode === 'intensity') c.setHSL(0.62 - Math.min(p.inten[i] / 90, 1) * 0.62, 0.9, 0.5);
      else if (cls === 1) c.copy(COLORS.ground);
      else if (cls === 2) c.copy(COLORS.env);
      else if (cls === 0) { const g = 0.45 + Math.min(Math.max(z / 5, 0), 1) * 0.35; c.setRGB(0.36 * g + 0.1, 0.48 * g + 0.1, 0.6 * g + 0.12); }
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
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeBoundingSphere();
    const hg = this.hot.geometry;
    hg.setAttribute('position', new THREE.BufferAttribute(hpos, 3));
    hg.setAttribute('color', new THREE.BufferAttribute(hcol, 3));
    hg.computeBoundingSphere();
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
    [this.envGroup, this.pathGroup, this.wallGroup, this.boxGroup, this.markGroup].forEach((g) => this._clear(g));
    const W = (fr.envelope && fr.envelope.width_m) || 2.2, H = (fr.envelope && fr.envelope.height_m) || 3.0;
    const obs = tr.observed_until || 0;
    const alarm = fr.status === 'OBSTACLE', check = fr.status === 'CHECK';
    const envColor = alarm ? 0xef4444 : check ? 0xf59e0b : 0x22c55e;
    const S = tr.s, C = tr.c, Z = tr.z;
    // --- envelope: translucent walls + roof edges, dimmer beyond the observed range
    if (this.layers.envelope) {
      const build = (s0, s1, opacity, color) => {
        const verts = [], idx = [];
        const lines = [];
        const pts = [];
        for (let s = s0; s <= s1 + 1e-6; s += 2) {
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
          lines.push(new THREE.Line(new THREE.BufferGeometry().setFromPoints(l),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 5) })));
        }
        lines.forEach((l) => this.envGroup.add(l));
        // cross-section frames every 25 m
        for (let s = Math.ceil(s0 / 25) * 25; s <= s1; s += 25) {
          const c = interp(S, C, s), z = interp(S, Z, s);
          const ring = [[c + W / 2, z], [c + W / 2, z + H], [c - W / 2, z + H], [c - W / 2, z], [c + W / 2, z]].map(([x, zz]) => tv(x, s, zz));
          this.envGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 4) })));
        }
      };
      const sEnd = Math.min(200, S[S.length - 1]);
      build(3, Math.max(4, Math.min(obs, sEnd)), 0.09, envColor);
      if (obs < sEnd - 2) build(Math.max(3, obs), sEnd, 0.03, 0x64748b);
    }
    // --- path centre and rails
    if (this.layers.path) {
      const centre = [], rl = [], rr = [];
      for (let s = 3; s <= Math.min(S[S.length - 1], 200); s += 1) {
        const c = interp(S, C, s), z = interp(S, Z, s);
        centre.push(tv(c, s, z + 0.01)); rl.push(tv(c + 0.8, s, z)); rr.push(tv(c - 0.8, s, z));
      }
      const mk = (pts, color, dashed) => {
        const g = new THREE.BufferGeometry().setFromPoints(pts);
        const m = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 1.5, gapSize: 1.0 }) : new THREE.LineBasicMaterial({ color });
        const l = new THREE.Line(g, m); if (dashed) l.computeLineDistances(); return l;
      };
      this.pathGroup.add(mk(centre, 0x38bdf8, true), mk(rl, 0x94a3b8, false), mk(rr, 0x94a3b8, false));
    }
    if (this.layers.walls) {
      for (const [key, color] of [['wl', 0xf87171], ['wr', 0x60a5fa]]) {
        const pts = [];
        tr[key].forEach((v, i) => { if (v !== null) pts.push(tv(v, S[i], interp(S, Z, S[i]) + 1.2)); });
        if (pts.length > 1) this.wallGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color })));
      }
    }
    // --- distance markers
    if (this.layers.markers) {
      for (let s = 25; s <= 200; s += 25) {
        if (s > S[S.length - 1]) break;
        const c = interp(S, C, s), z = interp(S, Z, s);
        const sp = labelSprite(`${s} м`, s <= obs ? '#9fb3c8' : '#566476', s % 50 ? 0.9 : 1.2);
        sp.position.copy(tv(c - W / 2 - 1.8, s, z + 0.4));
        this.markGroup.add(sp);
        const tick = [tv(c - W / 2 - 0.6, s, z), tv(c + W / 2 + 0.6, s, z)];
        this.markGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(tick), new THREE.LineBasicMaterial({ color: 0x334155 })));
      }
    }
    // --- boxes of obstacles / pending tracks / synthetic ground truth
    if (this.layers.boxes) {
      const addBox = (b, color, label, dashed = false) => {
        const s0 = b.s[0], s1 = Math.max(b.s[1], s0 + 0.3);
        const d0 = b.d[0], d1 = b.d[1], h0 = b.h[0], h1 = b.h[1];
        const c = interp(S, C, s0), z = interp(S, Z, s0);
        const cx = c + (d0 + d1) / 2, cz = z + (h0 + h1) / 2;
        const geo = new THREE.BoxGeometry(s1 - s0 + 0.2, Math.max(h1 - h0, 0.1) + 0.2, Math.max(d1 - d0, 0.1) + 0.2);
        const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2, depthTest: false })
          : new THREE.LineBasicMaterial({ color, depthTest: false });
        const box = new THREE.LineSegments(new THREE.EdgesGeometry(geo), mat);
        if (dashed) box.computeLineDistances();
        box.position.copy(tv(cx, (s0 + s1) / 2, cz));
        box.renderOrder = 6;
        this.boxGroup.add(box);
        if (!dashed) {
          const fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false }));
          fill.position.copy(box.position);
          this.boxGroup.add(fill);
        }
        if (label) {
          const sp = labelSprite(label, dashed ? '#d8b4fe' : color === 0xef4444 ? '#fecaca' : '#fde68a', 1.1);
          sp.position.copy(tv(cx, s0, z + h1 + 0.9));
          this.boxGroup.add(sp);
        }
      };
      (fr.obstacles || []).forEach((o) => addBox(o.box, 0xef4444, `${o.distance_m.toFixed(1)} м`));
      (fr.pending || []).forEach((o) => addBox(o.box, 0xf59e0b, `? ${o.distance_m.toFixed(0)} м`));
      if (this.layers.candidates) (fr.candidates || []).filter((c) => c.rejected).forEach((c) => addBox(c.box, 0x64748b, null));
      (fr.synthetic || []).forEach((o) => {
        const w = o.width, h = o.height, l = o.length;
        addBox({ s: [o.s - l / 2, o.s + l / 2], d: [o.d - w / 2, o.d + w / 2], h: [o.base, o.base + h] }, 0xc084fc, `тест ${o.s.toFixed(0)} м`, true);
      });
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
    if (s < 5 || s > 200) return null;
    return { s, d: x - interp(tr.s, tr.c, s) };
  }

  _tick(t) {
    this._clock = t;
    if (this.controls.enabled) this.controls.update();
    // pulse obstacle boxes
    const k = 0.5 + 0.5 * Math.sin(t / 180);
    this.boxGroup.children.forEach((o) => { if (o.isMesh) o.material.opacity = 0.1 + 0.2 * k; });
    this.renderer.render(this.scene, this.camera);
  }
}
