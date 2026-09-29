import { Scene3D } from '/static/js/scene.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const scene = new Scene3D($('scene'));
const state = {
  ws: null, mode: 'playback', bags: [], bag: null, frames: 0, playing: false,
  frame: null, timeline: [], answered: new Map(), toastFor: null, seen: new Set(),
  perfHist: [], lastArrive: 0, fpsHist: [], shapes: {}, baseConfig: null,
  token: null, brake: null, autoBrake: true, exempt: new Set(), busy: false, noReconnect: false, pendingOpen: null,
};

// ------------------------------------------------------------------ modal
function showModal(title, text, actions = [], progress = null) {
  $('modalTitle').textContent = title;
  $('modalText').textContent = text;
  $('modalProgress').classList.toggle('hidden', progress === null);
  if (progress !== null) $('modalBar').style.width = `${Math.round(progress * 100)}%`;
  $('modalActions').innerHTML = '';
  actions.forEach(([label, fn, primary]) => {
    const b = document.createElement('button');
    b.textContent = label; if (primary) b.className = 'primary';
    b.onclick = fn; $('modalActions').appendChild(b);
  });
  $('modal').classList.remove('hidden');
}
function hideModal() { $('modal').classList.add('hidden'); }

// ------------------------------------------------------------------ socket
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  ws.onopen = () => { $('conn').className = 'conn on'; };
  ws.onclose = () => {
    $('conn').className = 'conn off';
    if (!state.noReconnect) setTimeout(connect, state.busy ? 10000 : 1500);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') onJson(JSON.parse(ev.data));
    else onFrame(ev.data);
  };
}
const send = (obj) => { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); };

function onJson(m) {
  if (m.type === 'hello') {
    state.mode = m.mode;
    $('liveBadge').classList.toggle('hidden', m.mode !== 'live');
    $('transport').classList.toggle('hidden', m.mode === 'live');
    if (m.mode === 'live') {   // no seeking / synthetic obstacles on a live ROS 2 stream
      $('btnScan').classList.add('hidden');
      document.querySelector('#tabs button[data-tab="test"]').classList.add('hidden');
      document.querySelector('#tabs button[data-tab="layers"]').click();
    }
    state.busy = false; state.token = m.token || null; hideModal();
    $('btnUpload').classList.toggle('hidden', m.mode === 'live');
    if (m.mode === 'playback') {
      fillBags(m.bags);
      const keep = state.bag && m.bags.find((b) => b.id === state.bag.id);
      const pref = keep || m.bags.find((b) => b.name.includes('obstacle')) || m.bags[0];
      if (pref) { $('bagSelect').value = pref.id; openBag(pref.id); }
    }
  } else if (m.type === 'busy') {
    state.busy = true;
    showModal('Сервис занят',
      `Сейчас с сервисом работает другой пользователь (${Math.max(1, Math.round(m.busy_for_s / 60))} мин). `
      + `Одновременно работает ${m.max_users === 1 ? 'один пользователь' : m.max_users + ' пользователя'}: каждому сеансу `
      + 'нужен свой процессор для обработки облаков в реальном времени. Повторная попытка каждые 10 секунд…');
  } else if (m.type === 'kicked') {
    state.noReconnect = true;
    showModal('Сеанс завершён', `${m.reason}. Нажмите «Подключиться», чтобы продолжить работу.`,
      [['Подключиться', () => { state.noReconnect = false; hideModal(); connect(); }, true]]);
  } else if (m.type === 'bags') {
    fillBags(m.bags);
    if (state.pendingOpen) { $('bagSelect').value = state.pendingOpen; openBag(state.pendingOpen); state.pendingOpen = null; }
    else if (state.bag) $('bagSelect').value = state.bag.id;
  } else if (m.type === 'brake') {
    state.brake = m.cleared ? null : m;
    if (m.cleared && m.released) state.released = { how: m.released, until: performance.now() + 6000 };
    if (m.cleared) state.exempt = new Set(m.exempt || []);
    if (state.frame) { renderStatus(state.frame); drawRadar(state.frame); }
  } else if (m.type === 'hold_hint') {
    flashHold();
  } else if (m.type === 'opened') {
    state.bag = m.bag; state.frames = m.frames; state.timeline = new Array(m.frames).fill(null);
    state.seen.clear(); state.answered.clear(); state.exempt = new Set(); state.brake = null;
    drawTimeline(); $('tlEvents').innerHTML = ''; $('scanState').textContent = '';
  } else if (m.type === 'state') {
    state.playing = m.playing; $('btnPlay').innerHTML = m.playing ? '&#10074;&#10074;' : '&#9654;';
  } else if (m.type === 'timeline') {
    m.items.forEach((it) => { state.timeline[it.i] = it; });
    $('scanState').textContent = m.done ? `проанализировано ${m.total} кадров` : `анализ… ${m.n || 0} / ${m.total}`;
    if (m.done) { $('btnScan').textContent = 'Анализ всего проезда'; renderEvents(); }
    drawTimeline();
  } else if (m.type === 'synthetic') {
    if (m.cleared) $('synList').innerHTML = '';
  } else if (m.type === 'journal') {
    loadJournal();
  } else if (m.type === 'config') {
    // applied
  } else if (m.type === 'error') {
    console.warn(m.message);
  }
}

function onFrame(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'GBR1') return;
  const jlen = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jlen)));
  let o = 8 + jlen;
  const n = dv.getUint32(o, true); o += 4;
  const xyz = new Float32Array(buf, o, n * 3); o += n * 12;
  const cls = new Uint8Array(buf, o, n); o += n;
  const inten = new Uint8Array(buf, o, n);
  const now = performance.now();
  if (state.lastArrive) state.fpsHist.push(1000 / (now - state.lastArrive));
  state.lastArrive = now;
  if (state.fpsHist.length > 20) state.fpsHist.shift();
  state.frame = meta;
  if (state.mode === 'live') {
    // a live stream has no length: the strip shows the last 60 s (600 frames)
    const LIVE_N = 600;
    state.frames = LIVE_N;
    state.timeline.push({ i: 0, status: meta.status, dist: meta.nearest_distance_m, v: meta.speed_kmh, ids: [], ms: meta.timing_ms.total });
    while (state.timeline.length > LIVE_N) state.timeline.shift();
    state.timeline.forEach((it, i) => { if (it) it.i = i; });
  } else {
    if (meta.frames) state.frames = meta.frames;
    if (state.timeline.length !== state.frames) state.timeline = new Array(state.frames).fill(null);
  }
  if (state.mode !== 'live' && meta.frame !== undefined && state.timeline.length > meta.frame) {
    const prev = state.timeline[meta.frame];
    state.timeline[meta.frame] = { i: meta.frame, status: meta.status, dist: meta.nearest_distance_m, v: meta.speed_kmh,
      ids: meta.obstacles.map((x) => x.track_id), ms: meta.timing_ms.total, ...(prev && prev.fromScan ? {} : {}) };
  }
  scene.update(meta, { n, xyz, cls, inten });
  renderStatus(meta);
  renderObstacles(meta);
  renderSynthetic(meta);
  renderPerf(meta);
  drawRadar(meta);
  drawTimeline();
  maybeToast(meta);
}

function fillBags(bags) {
  state.bags = bags;
  // grouped, readable titles; the technical bag name stays in the tooltip
  const order = ['С препятствиями', 'Штатные проезды (без препятствий)', 'Синтетические препятствия', 'Загруженные записи'];
  const groups = new Map();
  bags.forEach((b) => { const g = b.group || 'Записи'; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(b); });
  const keys = [...groups.keys()].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  $('bagSelect').innerHTML = keys.map((g) => `<optgroup label="${g}">${groups.get(g).map((b) =>
    `<option value="${b.id}" title="${b.name} · ${b.frames} кадров">${b.title || b.name} · ${b.duration_s.toFixed(0)} с</option>`).join('')}</optgroup>`).join('');
}

// ------------------------------------------------------------------ status
const SHAPE_RU = { person: 'человек', child: 'ребёнок', box_50cm: 'ящик', box_30cm: 'предмет на рельсе', barrel: 'бочка', trolley: 'тележка', cable: 'кабель', glass: 'стеклянный объект' };

function renderStatus(m) {
  const card = $('statusCard'), vp = $('viewport');
  let cls = 'state-ok', label = 'ПУТЬ СВОБОДЕН', main = `${fmt(m.free_distance_m)} м`, sub = 'путь проверен и свободен на это расстояние';
  if (m.status === 'OBSTACLE') {
    const o = m.obstacles[0];
    cls = 'state-bad'; label = 'ПРЕПЯТСТВИЕ — ТОРМОЖЕНИЕ';
    const ex = state.exempt || new Set();
    if (!state.autoBrake) label = 'ПРЕПЯТСТВИЕ — ТОРМОЖЕНИЕ ВЫКЛЮЧЕНО';
    else if (m.obstacles.length && m.obstacles.every((x) => ex.has(x.track_id))) {
      cls = 'state-warn'; label = 'ОБЪЕКТ ПРОПУЩЕН ОПЕРАТОРОМ';
    }
    main = `${fmt(m.nearest_distance_m, 1)} м`;
    const when = m.time_to_obstacle_s ? `через ${fmt(m.time_to_obstacle_s, 1)} с` : 'поезд стоит';
    sub = `ближайший из ${m.obstacles.length}: ${obKind(o)}, смещение ${fmt(o.lateral_m, 2)} м · ${when}`;
  } else if (m.status === 'CHECK') {
    const o = m.pending[0];
    cls = 'state-warn'; label = 'ПРОВЕРКА ОБЪЕКТА';
    main = `${fmt(o.distance_m, 0)} м`; sub = 'возможное препятствие подтверждается на следующих кадрах';
  } else if (m.status === 'DEGRADED') {
    cls = 'state-warn'; label = 'НЕТ ОПОРЫ ПУТИ'; main = '—'; sub = 'рельсы и стены не распознаны — проверьте установку лидара';
  }
  const br = state.brake;
  const rel = state.released && performance.now() < state.released.until ? state.released.how : '';
  if (!br && rel === 'auto') {
    sub = 'путь освободился — движение возобновлено автоматически';
  } else if (!br && rel === 'override') {
    sub = 'движение продолжено по решению оператора (записано в журнал)';
  }
  if (br && br.holding) {
    cls = 'state-halt'; label = 'ПОЕЗД ОСТАНОВЛЕН — ПУТЬ ЗАНЯТ';
    main = br.collision ? `${fmt(br.impact_kmh, 0)} км/ч у препятствия` : `${fmt(br.gap_m, 0)} м до препятствия`;
    sub = br.collision
      ? `объект обнаружен ближе тормозного пути (${fmt(br.stop_m, 0)} м); торможение снизило скорость с ${fmt(br.v0_kmh, 0)} км/ч`
      : `тормозной путь ${fmt(br.stop_m, 0)} м за ${fmt(br.time_s, 1)} с с ${fmt(br.v0_kmh, 0)} км/ч`;
  } else if (br) {
    if (br.active) {
      cls = 'state-bad'; label = 'ЭКСТРЕННОЕ ТОРМОЖЕНИЕ';
      main = `${fmt(br.v_kmh, 0)} км/ч`;
      sub = br.collision
        ? `объект слишком близко: остановиться не успевает, скорость у препятствия ~${fmt(br.impact_kmh, 0)} км/ч`
        : `остановка через ${fmt(Math.max(br.stop_m - br.travelled_m, 0), 0)} м, за ${fmt(br.gap_m, 0)} м до препятствия`;
    } else if (br.stopped) {
      cls = 'state-halt'; label = 'ПОЕЗД ОСТАНОВЛЕН';
      main = `${fmt(br.gap_m, 0)} м`;
      sub = `до препятствия · тормозной путь ${fmt(br.stop_m, 0)} м за ${fmt(br.time_s, 1)} с с ${fmt(br.v0_kmh, 0)} км/ч · ▶ — отпустить тормоз`;
    } else {
      cls = 'state-bad'; label = 'СТОЛКНОВЕНИЕ НЕ ПРЕДОТВРАЩЕНО';
      main = `${fmt(br.impact_kmh, 0)} км/ч`;
      sub = `объект обнаружен ближе тормозного пути (${fmt(br.stop_m, 0)} м); торможение снизило скорость с ${fmt(br.v0_kmh, 0)} км/ч`;
    }
  }
  const holding = !!(br && br.holding);
  $('holdBox').classList.toggle('hidden', !holding);
  if (holding) {
    const ids = (br.hold_ids || []).map((x) => `№${x}`).join(', ');
    $('holdText').innerHTML = `Поезд удерживается системой безопасности${ids ? ` (объект ${ids})` : ''}. `
      + '<b>Ничего нажимать не нужно:</b> когда путь освободится, движение возобновится автоматически.<br>'
      + 'Если вы <b>убедились</b> (камера, доклад машиниста или диспетчера), что объект не мешает движению, — '
      + 'нажмите «Продолжить движение». Отметка «Ложная тревога» тормоз не отпускает.';
  }
  $('btnPlay').classList.toggle('held', holding);
  $('btnPlay').title = holding ? 'Поезд удерживается: путь занят. Используйте «Продолжить движение» в карточке статуса'
    : 'Воспроизвести / пауза (пробел)';
  card.className = `status-card ${cls}`;
  $('statusLabel').textContent = label; $('statusMain').textContent = main; $('statusSub').textContent = sub;
  const note = $('statusNote');
  note.classList.toggle('hidden', !m.turnout_ahead_m);
  if (m.turnout_ahead_m) note.textContent = `Стрелка через ${fmt(m.turnout_ahead_m, 0)} м: маршрут за ней не определён`;
  const halted = br && !br.active;
  vp.classList.toggle('alarm', m.status === 'OBSTACLE' && !halted);
  vp.classList.toggle('check', m.status === 'CHECK' && !br);
  vp.classList.toggle('halt', !!halted && br.stopped);
  $('mSpeed').textContent = br ? fmt(br.v_kmh, 0) : fmt(m.speed_kmh, 0);
  $('mFree').textContent = fmt(m.free_distance_m, 0);
  $('mStop').textContent = br ? fmt(Math.max(br.stop_m - br.travelled_m, 0), 0) : fmt(m.stopping_distance_m, 0);
  $('mTtc').textContent = !br && m.time_to_obstacle_s ? fmt(m.time_to_obstacle_s, 1) : '—';
  const fps = state.fpsHist.length ? state.fpsHist.reduce((a, b) => a + b, 0) / state.fpsHist.length : null;
  const ms = m.timing_ms.total;
  $('sysInfo').textContent = `обработка ${fmt(ms, 0)} мс` + (state.playing || state.mode === 'live' ? ` · ${fmt(fps, 1)} кадр/с` : '');
  $('sysInfo').classList.toggle('slow', ms > 100);
  if (m.frames) $('frameLabel').textContent = `кадр ${m.frame + 1} / ${m.frames}`;
  $('timeLabel').textContent = m.bag_time_s !== undefined ? `${fmt(m.bag_time_s, 1)} с записи` : '';
  const cal = m.calibration || {};
  $('calibBox').innerHTML = `<b>Самокалибровка по рельсам</b><br>высота лидара над головкой рельса: ${fmt(cal.height_m, 2)} м<br>тангаж ${fmt(cal.pitch_deg, 2)}°, крен ${fmt(cal.roll_deg, 2)}°<br>путь подтверждён до ${fmt(m.observed_until_m, 0)} м, рельсы — до ${fmt(m.track.rail_until, 0)} м`;
}

function obKind(o) {
  if (o.transparent) return 'стеклянный объект';
  if (o.low) return 'низкий объект на рельсе';
  const h = o.box.h[1];
  if (h > 1.2) return 'высокий объект (человек?)';
  if (h > 0.5) return 'объект средней высоты';
  return 'низкий объект';
}

function renderObstacles(m) {
  const all = [...m.obstacles.map((o) => ({ ...o, conf: true })), ...m.pending.map((o) => ({ ...o, conf: false }))];
  $('obstacles').innerHTML = all.slice(0, 5).map((o) => {
    const key = `${state.bag ? state.bag.id : 'live'}#${o.track_id}`;
    const ans = state.answered.get(key);
    const verdict = ans ? `<div class="ob-verdict ${ans === 'confirmed' ? 'yes' : 'no'}">оператор: ${ans === 'confirmed' ? 'подтверждено' : 'ложная тревога'}</div>` : '';
    const actions = o.conf && !ans ? `<div class="ob-actions"><button data-v="confirmed" data-id="${o.track_id}">Подтвердить</button><button data-v="false" data-id="${o.track_id}">Ложная</button></div>` : '';
    return `<div class="ob ${o.conf ? '' : 'pending'} ${o.low ? 'low' : ''}" data-id="${o.track_id}">
      <div class="ob-top"><span class="ob-dist">${fmt(o.distance_m, 1)} м</span><span class="ob-kind">${o.conf ? 'подтверждено' : 'проверка'} · №${o.track_id}</span></div>
      <div class="ob-info">${obKind(o)} · смещение ${fmt(o.lateral_m, 2)} м · высота ${fmt(o.box.h[0], 2)}–${fmt(o.box.h[1], 2)} м</div>
      <div class="ob-info">${o.points} точек на ${o.rings} лучах · обнаружен с ${fmt(o.first_seen_m, 0)} м · уверенность ${fmt(o.confidence * 100, 0)}%</div>
      ${verdict}${actions}</div>`;
  }).join('');
  $('obstacles').querySelectorAll('button[data-v]').forEach((b) => b.onclick = (e) => {
    e.stopPropagation(); answer(Number(b.dataset.id), b.dataset.v);
  });
  $('obstacles').querySelectorAll('.ob').forEach((el) => el.onclick = () => { setCam('target'); });
}

function renderSynthetic(m) {
  const syn = m.synthetic || [];
  $('synList').innerHTML = syn.map((o) => `<div class="synitem"><span>${SHAPE_RU[o.shape] || o.shape}</span><span>${fmt(o.s, 1)} м, смещение ${fmt(o.d, 1)} м</span></div>`).join('');
}

// ------------------------------------------------------------------ operator
function answer(trackId, verdict) {
  const m = state.frame;
  const o = [...m.obstacles, ...m.pending].find((x) => x.track_id === trackId) || {};
  const key = `${state.bag ? state.bag.id : 'live'}#${trackId}`;
  state.answered.set(key, verdict);
  send({ cmd: 'feedback', bag: state.bag ? state.bag.id : 'live', frame: m.frame, track_id: trackId, verdict,
    distance_m: o.distance_m, lateral_m: o.lateral_m, height_m: o.box ? o.box.h[1] : null,
    synthetic: (m.synthetic || []).length > 0 });
  if (state.toastFor === key) hideToast();
  renderObstacles(m);
}

function toastText(o) {
  return `<b>${fmt(o.distance_m, 1)} м</b> — ${obKind(o)}, смещение от оси ${fmt(o.lateral_m, 2)} м.<br>Подтвердите обнаружение: решение попадёт в журнал оператора. Торможение выполняется автоматически и от ответа не зависит.`;
}

function maybeToast(m) {
  if (state.toastFor) {
    // keep the request in step with its object; drop it once the object is gone
    const cur = m.obstacles.find((x) => x.track_id === state.toastTrack);
    if (cur) $('toastBody').innerHTML = toastText(cur);
    else hideToast();
  }
  for (const o of m.obstacles) {
    const key = `${state.bag ? state.bag.id : 'live'}#${o.track_id}`;
    if (state.seen.has(key)) continue;
    state.seen.add(key);
    if (state.answered.has(key)) continue;
    state.toastFor = key;
    state.toastTrack = o.track_id;
    $('toastBody').innerHTML = toastText(o);
    $('toast').classList.remove('hidden');
    beep();
  }
}
function hideToast() { $('toast').classList.add('hidden'); state.toastFor = null; }
$('toastConfirm').onclick = () => answer(state.toastTrack, 'confirmed');
$('toastReject').onclick = () => answer(state.toastTrack, 'false');
$('toastLater').onclick = hideToast;

let audio = null;
function beep() {
  if (!$('soundOn').checked) return;
  audio = audio || new AudioContext();
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 880; o.type = 'square'; g.gain.value = 0.05;
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.25);
}

// ------------------------------------------------------------------ radar strip
function drawRadar(m) {
  const cv = $('radar'), g = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
  const k = devicePixelRatio, L = 40 * k, R = W - 20 * k, maxS = 200;
  const X = (s) => L + (R - L) * Math.min(Math.max(s, 0), maxS) / maxS;
  g.clearRect(0, 0, W, H);
  g.font = `${11 * k}px Segoe UI, sans-serif`;
  const y0 = 30 * k, y1 = 58 * k;
  // corridor band: observed (solid) / not observed (hatched)
  const obs = m.observed_until_m || 0;
  const col = m.status === 'OBSTACLE' ? 'rgba(239,68,68,' : m.status === 'CHECK' ? 'rgba(245,158,11,' : 'rgba(34,197,94,';
  g.fillStyle = col + '0.20)'; g.fillRect(X(0), y0, X(obs) - X(0), y1 - y0);
  g.strokeStyle = col + '0.8)'; g.lineWidth = 1.5 * k; g.strokeRect(X(0), y0, X(obs) - X(0), y1 - y0);
  g.save(); g.beginPath(); g.rect(X(obs), y0, X(maxS) - X(obs), y1 - y0); g.clip();
  g.strokeStyle = 'rgba(100,116,139,0.35)'; g.lineWidth = k;
  for (let x = X(obs) - 40 * k; x < R; x += 8 * k) { g.beginPath(); g.moveTo(x, y1); g.lineTo(x + 28 * k, y0); g.stroke(); }
  g.restore();
  // rails
  g.strokeStyle = css('--muted'); g.globalAlpha = 0.5; g.lineWidth = k;
  [0.38, 0.62].forEach((f) => { const y = y0 + (y1 - y0) * f; g.beginPath(); g.moveTo(X(0), y); g.lineTo(X(maxS), y); g.stroke(); });
  g.globalAlpha = 1;
  // ticks
  g.fillStyle = css('--muted'); g.textAlign = 'center';
  for (let s = 0; s <= maxS; s += 25) {
    g.fillRect(X(s), y1, k, 5 * k);
    g.fillText(`${s}`, X(s), y1 + 16 * k);
  }
  g.textAlign = 'left'; g.fillText('м', R + 4 * k, y1 + 16 * k);
  // stopping distance
  const stop = m.stopping_distance_m || 0;
  const info = css('--info');
  const br = state.brake;
  if (br && (br.active || br.stopped)) {   // where the braking train stops
    const at = Math.max(br.stop_m - br.travelled_m, 0);
    g.fillStyle = info; g.fillRect(X(at) - 2 * k, y0 - 6 * k, 4 * k, y1 - y0 + 8 * k);
    if (!br.stopped) g.fillText(`остановка через ${at.toFixed(0)} м`, Math.min(X(at) + 6 * k, R - 130 * k), y0 - 10 * k);
  } else if (stop > 1) {
    g.strokeStyle = info; g.setLineDash([4 * k, 3 * k]); g.lineWidth = 1.5 * k;
    g.beginPath(); g.moveTo(X(stop), y0 - 8 * k); g.lineTo(X(stop), y1 + 2 * k); g.stroke(); g.setLineDash([]);
    g.fillStyle = info; g.fillText(`тормозной путь ${stop.toFixed(0)} м`, Math.min(X(stop) + 4 * k, R - 120 * k), y0 - 10 * k);
  }
  // train
  g.fillStyle = '#3b82f6'; g.fillRect(L - 30 * k, y0 + 4 * k, 26 * k, y1 - y0 - 8 * k);
  // objects
  const tr = m.track;
  const yOf = (lat) => (y0 + y1) / 2 - Math.max(-1.4, Math.min(1.4, lat)) / 2.8 * (y1 - y0);
  (m.synthetic || []).forEach((o) => { g.strokeStyle = '#c084fc'; g.lineWidth = 2 * k; g.strokeRect(X(o.s) - 4 * k, yOf(o.d) - 4 * k, 8 * k, 8 * k); });
  m.pending.forEach((o) => { g.fillStyle = '#f59e0b'; g.beginPath(); g.arc(X(o.s_m), yOf(o.lateral_m), 5 * k, 0, 7); g.fill(); });
  m.obstacles.forEach((o) => {
    g.fillStyle = '#ef4444'; g.beginPath(); g.arc(X(o.s_m), yOf(o.lateral_m), 7 * k, 0, 7); g.fill();
    g.font = `bold ${13 * k}px Segoe UI, sans-serif`; g.fillText(`${o.distance_m.toFixed(1)} м`, X(o.s_m) + 10 * k, y0 - 8 * k);
    g.font = `${11 * k}px Segoe UI, sans-serif`;
  });
  g.fillStyle = css('--muted'); g.font = `${11 * k}px Segoe UI, sans-serif`;
  g.fillText('схема пути · заливка — проверенный участок', 12 * k, 16 * k);
  void tr;
}

// ------------------------------------------------------------------ timeline
const tlColors = () => ({ CLEAR: css('--tl-clear'), CHECK: css('--tl-check'), OBSTACLE: css('--tl-obst'), DEGRADED: css('--muted') });
function drawTimeline() {
  const cv = $('timeline'), g = cv.getContext('2d');
  const k = devicePixelRatio, W = cv.width = cv.clientWidth * k, H = cv.height = cv.clientHeight * k;
  g.clearRect(0, 0, W, H);
  const n = state.frames || 1;
  const bw = W / n;
  const TL_COLORS = tlColors();
  state.timeline.forEach((it, i) => {
    if (!it) return;
    g.fillStyle = TL_COLORS[it.status] || '#334155';
    const h = it.status === 'OBSTACLE' ? H * 0.62 : it.status === 'CHECK' ? H * 0.42 : H * 0.26;
    g.fillRect(i * bw, H - h - 14 * k, Math.max(bw, 1), h);
  });
  // speed curve
  g.strokeStyle = 'rgba(56,189,248,0.7)'; g.lineWidth = k; g.beginPath();
  let started = false;
  state.timeline.forEach((it, i) => { if (!it || it.v === undefined) return; const y = H - 14 * k - (H - 20 * k) * Math.min(it.v, 90) / 90; if (!started) { g.moveTo(i * bw, y); started = true; } else g.lineTo(i * bw, y); });
  g.stroke();
  // cursor
  const cur = state.mode === 'live' ? state.timeline.length - 1 : state.frame && state.frame.frame;
  if (cur !== undefined && cur !== null && cur >= 0) {
    const x = (cur + 0.5) * bw;
    g.fillStyle = css('--text'); g.fillRect(x - k, 0, 2 * k, H - 12 * k);
  }
  g.fillStyle = css('--muted'); g.font = `${10 * k}px Segoe UI, sans-serif`;
  g.fillText(state.mode === 'live' ? 'последние 60 с · статус по кадрам · голубая линия — скорость (км/ч)'
    : 'статус по кадрам · голубая линия — скорость (км/ч)', 4 * k, H - 2 * k);
}
$('timeline').onclick = (e) => {
  if (state.mode !== 'playback' || !state.frames) return;
  const r = e.target.getBoundingClientRect();
  const i = Math.floor((e.clientX - r.left) / r.width * state.frames);
  send({ cmd: 'seek', index: i });
};

function renderEvents() {
  const ev = new Map();
  state.timeline.forEach((it) => { if (it && it.ids) it.ids.forEach((id) => { if (!ev.has(id)) ev.set(id, it); }); });
  $('tlEvents').innerHTML = [...ev.entries()].map(([id, it]) => `<span class="tl-chip" data-i="${it.i}">кадр ${it.i}: ${fmt(it.dist, 0)} м</span>`).join('') || '<span class="scan-state">событий нет</span>';
  $('tlEvents').querySelectorAll('.tl-chip').forEach((c) => c.onclick = () => send({ cmd: 'seek', index: Math.max(0, Number(c.dataset.i) - 2) }));
}

// ------------------------------------------------------------------ perf
function renderPerf(m) {
  const t = m.timing_ms;
  const rows = [['decode', 'Декодирование'], ['track', 'Калибровка и путь'], ['egomotion', 'Эго-движение'], ['detect', 'Кандидаты'], ['tracking', 'Треки'], ['total', 'Всего']];
  $('perfBars').innerHTML = rows.map(([k, name]) => `<div class="perf-row"><span>${name}</span><div class="perf-bar"><i style="width:${Math.min(100, (t[k] || 0))}%;${k === 'total' ? 'background:' + ((t.total || 0) > 100 ? '#ef4444' : '#22c55e') : ''}"></i></div><span>${fmt(t[k], 1)} мс</span></div>`).join('');
  state.perfHist.push(t.total || 0);
  if (state.perfHist.length > 150) state.perfHist.shift();
  const cv = $('perfChart'), g = cv.getContext('2d');
  const k = devicePixelRatio, W = cv.width = cv.clientWidth * k, H = cv.height = cv.clientHeight * k;
  g.clearRect(0, 0, W, H);
  const y = (v) => H - (H - 8 * k) * Math.min(v, 200) / 200;
  g.strokeStyle = 'rgba(239,68,68,0.6)'; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(0, y(100)); g.lineTo(W, y(100)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = '#38bdf8'; g.lineWidth = 1.5 * k; g.beginPath();
  state.perfHist.forEach((v, i) => { const x = i / 150 * W; if (i) g.lineTo(x, y(v)); else g.moveTo(x, y(v)); });
  g.stroke();
  g.fillStyle = css('--muted'); g.font = `${10 * k}px sans-serif`; g.fillText('100 мс', 4 * k, y(100) - 3 * k);
}

// ------------------------------------------------------------------ journal
async function loadJournal() {
  try {
    const rows = await (await fetch('/api/journal')).json();
    $('journal').innerHTML = rows.slice(-200).reverse().map((r) => `<div class="jrow"><span class="${r.verdict === 'confirmed' ? 'v-confirmed' : 'v-false'}">${{ confirmed: 'ПОДТВЕРЖДЕНО', override: 'ДВИЖЕНИЕ ПРОДОЛЖЕНО ОПЕРАТОРОМ' }[r.verdict] || 'ЛОЖНАЯ ТРЕВОГА'}</span> · ${r.time.replace('T', ' ')}<br>${r.bag} · кадр ${r.frame} · №${r.track_id} · ${fmt(r.distance_m, 1)} м${r.synthetic ? ' · испытание' : ''}</div>`).join('') || '<span class="hint">решений оператора пока нет</span>';
  } catch (e) { /* offline */ }
}

// ------------------------------------------------------------------ controls
function openBag(id) { send({ cmd: 'open', bag: id }); }
$('bagSelect').onchange = (e) => openBag(e.target.value);
$('btnPlay').onclick = () => send({ cmd: state.playing ? 'pause' : 'play' });
$('btnNext').onclick = () => send({ cmd: 'step', delta: 1 });
$('btnPrev').onclick = () => send({ cmd: 'step', delta: -1 });
$('speedSelect').onchange = (e) => send({ cmd: 'speed', value: Number(e.target.value) });
$('btnScan').onclick = () => {
  if ($('btnScan').textContent.startsWith('Остановить')) { send({ cmd: 'scan_stop' }); $('btnScan').textContent = 'Анализ всего проезда'; return; }
  send({ cmd: 'scan' }); $('btnScan').textContent = 'Остановить анализ';
};

function setCam(mode) {
  scene.setCamera(mode);
  document.querySelectorAll('#cams button').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));
}
document.querySelectorAll('#cams button').forEach((b) => b.onclick = () => setCam(b.dataset.cam));

document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('hidden', t.id !== `tab-${b.dataset.tab}`));
  if (b.dataset.tab === 'log') loadJournal();
});

document.querySelectorAll('[data-layer]').forEach((c) => c.onchange = () => scene.setLayer(c.dataset.layer, c.checked));
$('colorMode').onchange = (e) => scene.setColorMode(e.target.value);
$('pointSize').oninput = (e) => scene.setPointSize(Number(e.target.value));

$('synDist').oninput = (e) => { $('synDistVal').textContent = e.target.value; };
$('synLat').oninput = (e) => { $('synLatVal').textContent = Number(e.target.value).toFixed(1); };
$('synAdd').onclick = () => send({ cmd: 'add_obstacle', shape: $('synShape').value, s: Number($('synDist').value), d: Number($('synLat').value) });
$('synClear').onclick = () => send({ cmd: 'clear_obstacles' });
$('scene').addEventListener('click', (e) => {
  if (!e.shiftKey) return;
  const p = scene.pickTrack(e.clientX, e.clientY);
  if (p) send({ cmd: 'add_obstacle', shape: $('synShape').value, s: p.s, d: Math.max(-1.5, Math.min(1.5, p.d)) });
});

const PRESETS = {
  balanced: {},
  strict: { tracking: { confirm_hits: 4, confirm_window: 5, far_confirm_hits: 8, far_confirm_window: 10 }, detection: { min_score: 0.65 } },
  sensitive: { tracking: { confirm_hits: 2, confirm_window: 3, far_confirm_hits: 4, far_confirm_window: 6 }, detection: { min_score: 0.4, max_range: 200 } },
};
[['pWidth', 'pWidthVal', 1], ['pHeight', 'pHeightVal', 1], ['pMinH', 'pMinHVal', 2], ['pRange', 'pRangeVal', 0]].forEach(([id, lab, d]) => {
  $(id).oninput = (e) => { $(lab).textContent = Number(e.target.value).toFixed(d); };
});
$('pApply').onclick = () => {
  const p = JSON.parse(JSON.stringify(PRESETS[$('preset').value]));
  p.envelope = { width: Number($('pWidth').value), height: Number($('pHeight').value), min_height: Number($('pMinH').value) };
  p.detection = { ...(p.detection || {}), max_range: Number($('pRange').value) };
  send({ cmd: 'config', overrides: p });
};
$('pReset').onclick = () => {
  $('preset').value = 'balanced';
  [['pWidth', 2.2], ['pHeight', 3.0], ['pMinH', 0.12], ['pRange', 200]].forEach(([id, v]) => { $(id).value = v; $(id).dispatchEvent(new Event('input')); });
  send({ cmd: 'config', overrides: {} });
};

// ------------------------------------------------------------------ theme
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('gabarit-theme', t); } catch (e) { /* private mode */ }
  $('btnTheme').innerHTML = t === 'light' ? '&#9790;' : '&#9788;';
  scene.setTheme(t === 'light');
  if (state.frame) { drawRadar(state.frame); renderPerf(state.frame); }
  drawTimeline();
}
$('btnTheme').onclick = () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
applyTheme(document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');

// ------------------------------------------------------------------ upload of a recording
$('btnUpload').onclick = () => $('uploadInput').click();
$('uploadInput').onchange = () => {
  const files = [...$('uploadInput').files];
  $('uploadInput').value = '';
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append('file', f, f.name));
  const total = files.reduce((a, f) => a + f.size, 0);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  if (state.token) xhr.setRequestHeader('X-Gabarit-Token', state.token);
  xhr.upload.onprogress = (e) => showModal('Загрузка записи',
    `${files.map((f) => f.name).join(', ')} — ${(e.loaded / 2 ** 20).toFixed(0)} из ${(total / 2 ** 20).toFixed(0)} МБ`, [],
    e.lengthComputable ? e.loaded / e.total : 0);
  xhr.onload = () => {
    let r = {};
    try { r = JSON.parse(xhr.responseText); } catch (e) { /* not json */ }
    if (xhr.status === 200 && r.ok) {
      hideModal();
      state.pendingOpen = r.bag ? r.bag.id : null;
      send({ cmd: 'list' });
    } else {
      showModal('Запись не загружена', r.error || `ошибка ${xhr.status}`, [['Закрыть', hideModal, true]]);
    }
  };
  xhr.onerror = () => showModal('Запись не загружена', 'соединение прервано', [['Закрыть', hideModal, true]]);
  showModal('Загрузка записи', files.map((f) => f.name).join(', '), [], 0);
  xhr.send(fd);
};

// braking mode on the main screen
function setBrakeMode(auto) {
  $('bmAuto').classList.toggle('on', auto);
  $('bmOff').classList.toggle('on', !auto);
  $('bmNote').classList.toggle('hidden', auto);
  state.autoBrake = auto;
  send({ cmd: 'autobrake', value: auto });
  if (state.frame) renderStatus(state.frame);   // the card reflects the mode at once, also on pause
}
$('bmAuto').onclick = () => setBrakeMode(true);
$('bmOff').onclick = () => {
  showModal('Выключить торможение?',
    'Поезд перестанет останавливаться перед подтверждёнными препятствиями — это режим просмотра записи, а не штатная работа. '
    + 'Тревоги и журнал продолжат работать.',
    [['Выключить', () => { hideModal(); setBrakeMode(false); }, true], ['Отмена', hideModal]]);
};

// held train: operator override
function flashHold() {
  const box = $('holdBox');
  box.classList.remove('flash'); void box.offsetWidth; box.classList.add('flash');
}
$('btnResume').onclick = () => {
  const br = state.brake;
  if (!br || !br.holding) return;
  const ids = (br.hold_ids || []).map((x) => `№${x}`).join(', ');
  showModal('Продолжить движение при обнаруженном объекте?',
    `Система видит на пути объект ${ids}. Продолжайте, только если убедились, что он не мешает движению `
    + '(ложное срабатывание, объект убран за габарит). Решение будет записано в журнал от вашего имени. '
    + 'Если впереди появится другое препятствие, поезд снова затормозит сам.',
    [['Да, путь свободен — продолжить', () => { hideModal(); send({ cmd: 'resume' }); }, true], ['Отмена', hideModal]]);
};

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); $('btnPlay').click(); }
  else if (e.code === 'ArrowRight') $('btnNext').click();
  else if (e.code === 'ArrowLeft') $('btnPrev').click();
  else if (/^Digit[1-6]$/.test(e.code)) setCam(['cab', 'chase', 'top', 'side', 'target', 'free'][Number(e.code.slice(5)) - 1]);
});
window.addEventListener('resize', () => { if (state.frame) drawRadar(state.frame); drawTimeline(); });

connect();
