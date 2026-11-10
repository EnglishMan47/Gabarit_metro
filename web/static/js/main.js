import { Scene3D } from '/static/js/scene.js';
import { coverage, decisionView, streamView, decodeFrame, finite, clamp } from '/static/js/viewer_state.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const scene = new Scene3D($('scene'));
const state = {
  ws: null, mode: 'playback', bags: [], bag: null, frames: 0, playing: false,
  frame: null, timeline: [], alarmIds: new Set(),
  perfHist: [], lastArrive: 0, fpsHist: [], shapes: {}, baseConfig: null, sensorProfiles: {},
  token: null, busy: false, noReconnect: false, pendingOpen: null,
  connected: false, frameInvalid: false, selectedObstacle: null,
};

function notify(message) {
  $('noticeText').textContent = String(message);
  $('notice').classList.remove('hidden');
}
$('noticeClose').onclick = () => $('notice').classList.add('hidden');

function updateStreamState() {
  const view = streamView({ connected: state.connected, mode: state.mode, playing: state.playing,
    hasFrame: Boolean(state.frame), lastArrive: state.lastArrive, now: performance.now() });
  if (state.frameInvalid) { view.stale = true; view.kind = 'off'; view.label = 'Кадр не прочитан · показаны предыдущие данные'; }
  $('streamState').className = `stream-state ${view.kind}`;
  $('streamState').textContent = view.label;
  $('viewport').classList.toggle('stale', view.stale);
  if (view.stale) {
    $('statusCard').className = 'status-card state-warn';
    $('statusLabel').textContent = 'НЕТ АКТУАЛЬНЫХ ДАННЫХ';
    $('statusSub').textContent = 'Показан последний полученный кадр; состояние пути сейчас неизвестно';
  } else if (state.frame) renderStatus(state.frame);
}
setInterval(updateStreamState, 1000);

function clearFrame() {
  state.frame = null; state.playing = false; state.frameInvalid = false;
  state.lastArrive = 0; state.fpsHist = []; state.perfHist = []; state.alarmIds.clear();
  state.selectedObstacle = null;
  scene.clear();
  $('statusCard').className = 'status-card state-idle';
  $('statusLabel').textContent = 'ОЖИДАНИЕ ДАННЫХ';
  $('statusMain').textContent = '—'; $('statusSub').textContent = 'запись открывается и детектор прогревается';
  $('statusNote').classList.add('hidden'); $('obstacles').replaceChildren(); $('synList').replaceChildren();
  ['mSpeed', 'mFree', 'mEnvelope', 'mTtc', 'frameLabel', 'coverageRange'].forEach((id) => $(id).textContent = '—');
  $('mFree').title = '';
  $('timeLabel').textContent = ''; $('calibBox').textContent = ''; $('sysInfo').textContent = '—';
  $('coverageFresh').style.width = $('coverageHeld').style.width = '0%';
  $('coverageDetail').textContent = 'Нет измерений';
  $('viewport').classList.remove('alarm', 'check');
  $('btnPlay').innerHTML = '&#9654;';
  $('perfBars').replaceChildren();
  for (const id of ['radar', 'perfChart']) { const canvas = $(id); canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); }
  updateStreamState();
}

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
  ws.onopen = () => { state.connected = true; $('conn').className = 'conn on'; updateStreamState(); };
  ws.onclose = () => {
    if (state.ws !== ws) return;
    state.connected = false;
    $('conn').className = 'conn off';
    updateStreamState();
    if (!state.noReconnect) setTimeout(connect, state.busy ? 10000 : 1500);
  };
  ws.onmessage = (ev) => {
    try {
      if (typeof ev.data === 'string') onJson(JSON.parse(ev.data));
      else onFrame(ev.data);
    } catch (error) {
      state.frameInvalid = true; notify(`Не удалось прочитать данные: ${error.message}`); updateStreamState();
    }
  };
}
const send = (obj) => {
  if (state.ws && state.ws.readyState === 1) { state.ws.send(JSON.stringify(obj)); return true; }
  notify('Нет соединения с сервером. Команда не отправлена.'); return false;
};

function onJson(m) {
  if (m.type === 'hello') {
    state.mode = m.mode;
    if (m.config) { state.baseConfig = m.config; setDetectorControls(m.config); }
    if (m.sensor_profiles) {
      state.sensorProfiles = m.sensor_profiles;
      $('sensorProfile').replaceChildren(...Object.entries(m.sensor_profiles).map(([key, p]) => {
        const option = document.createElement('option'); option.value = key; option.textContent = p.title; return option;
      }));
    }
    ['sensorProfile', 'pApply', 'pReset'].forEach((id) => { $(id).disabled = m.mode === 'live'; });
    $('liveBadge').classList.toggle('hidden', m.mode !== 'live');
    $('transport').classList.toggle('hidden', m.mode === 'live');
    $('previousEvent').disabled = $('nextEvent').disabled = m.mode === 'live';
    $('timelineHint').textContent = m.mode === 'live' ? 'История полученных кадров' : 'Нажмите на ленту, чтобы перейти к кадру';
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
  } else if (m.type === 'opened') {
    clearFrame();
    state.bag = m.bag; state.frames = m.frames; state.timeline = new Array(m.frames).fill(null);
    state.alarmIds.clear();
    drawTimeline(); $('tlEvents').innerHTML = ''; $('scanState').textContent = '';
  } else if (m.type === 'state') {
    state.playing = m.playing; $('btnPlay').innerHTML = m.playing ? '&#10074;&#10074;' : '&#9654;';
    $('btnPlay').setAttribute('aria-label', m.playing ? 'Пауза' : 'Воспроизвести');
    updateStreamState();
  } else if (m.type === 'timeline') {
    if (m.bag && state.bag && m.bag !== state.bag.id) return;
    m.items.forEach((it) => { state.timeline[it.i] = it; });
    $('scanState').textContent = m.done ? `проанализировано ${m.total} кадров` : `анализ… ${m.n || 0} / ${m.total}`;
    if (m.done) { $('btnScan').textContent = 'Анализ всего проезда'; renderEvents(); }
    drawTimeline();
  } else if (m.type === 'synthetic') {
    if (m.cleared) $('synList').innerHTML = '';

  } else if (m.type === 'config') {
    setDetectorControls(m.config);
  } else if (m.type === 'error') {
    notify(m.message);
  }
}

function onFrame(buf) {
  const { meta, points } = decodeFrame(buf);
  if (state.mode !== 'live' && meta.bag && state.bag && meta.bag !== state.bag.id) return;
  const now = performance.now();
  if (state.lastArrive && now > state.lastArrive && now - state.lastArrive < 3000) state.fpsHist.push(now - state.lastArrive);
  state.lastArrive = now;
  if (state.fpsHist.length > 20) state.fpsHist.shift();
  state.frame = meta;
  state.frameInvalid = false;
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
  scene.update(meta, points);
  renderStatus(meta);
  renderObstacles(meta);
  renderSynthetic(meta);
  renderPerf(meta);
  drawRadar(meta);
  drawTimeline();
  if (meta.obstacles.some((item) => !state.alarmIds.has(item.track_id))) renderEvents();
  signalAlarm(meta);
  updateStreamState();
}

function fillBags(bags) {
  state.bags = bags;
  // grouped, readable titles; the technical bag name stays in the tooltip
  const order = ['С препятствиями', 'Штатные проезды (без препятствий)', 'Синтетические препятствия', 'Записи без разметки', 'Загруженные записи'];
  const groups = new Map();
  bags.forEach((b) => { const g = b.group || 'Записи'; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(b); });
  const keys = [...groups.keys()].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  $('bagSelect').replaceChildren(...keys.map((g) => {
    const group = document.createElement('optgroup'); group.label = g;
    groups.get(g).forEach((b) => {
      const option = document.createElement('option'); option.value = b.id;
      option.title = `${b.name} · ${b.frames} кадров`;
      option.textContent = `${b.title || b.name} · ${fmt(b.duration_s)} с`;
      group.appendChild(option);
    });
    return group;
  }));
}

// ------------------------------------------------------------------ status
const SHAPE_RU = { person: 'человек', person_sitting: 'человек сидя', person_lying: 'человек лёжа', child: 'ребёнок', box_50cm: 'ящик', box_30cm: 'предмет на рельсе', barrel: 'бочка', trolley: 'тележка', cable: 'кабель', glass: 'стеклянный объект' };

function renderStatus(m) {
  const card = $('statusCard'), vp = $('viewport');
  const view = decisionView(m), area = view.area;
  let cls = `state-${view.kind}`, label = view.label, main = `${fmt(m.free_distance_m)} м`,
    sub = 'На измеренном участке подтверждённых препятствий нет';
  if (m.status === 'OBSTACLE') {
    const o = m.obstacles[0];
    cls = 'state-bad'; label = 'ПРЕПЯТСТВИЕ';
    main = `${fmt(m.nearest_distance_m, 1)} м`;
    const when = finite(m.time_to_obstacle_s) ? `до объекта ${fmt(m.time_to_obstacle_s, 1)} с` : 'время до объекта не определено';
    sub = o ? `ближайший из ${m.obstacles.length}: ${obKind(o)}, смещение ${fmt(o.lateral_m, 2)} м · ${when}` : 'Подтверждённое препятствие на пути';
  } else if (m.status === 'CHECK') {
    const o = m.pending[0];
    cls = 'state-warn'; label = 'ПРОВЕРКА ОБЪЕКТА';
    main = `${fmt(o?.distance_m, 0)} м`; sub = 'возможное препятствие подтверждается на следующих кадрах';
  } else if (m.status === 'DEGRADED') {
    cls = 'state-warn'; label = 'НЕТ ОПОРЫ ПУТИ'; main = '—'; sub = 'рельсы и стены не распознаны — проверьте установку лидара';
  } else if (finite(view.uncertainFrom)) {
    main = `с ${fmt(view.uncertainFrom)} м`; sub = 'Оценка дальней оси пути требует уточнения';
  } else if (area.held) {
    main = `${fmt(area.measured)} м`; sub = 'Свежие измерения до этой отметки; часть габарита продолжена по истории';
  } else if (view.kind !== 'ok') {
    main = '—'; sub = 'Данных недостаточно для вывода о свободном пути';
  }
  card.className = `status-card ${cls}`;
  $('statusLabel').textContent = label; $('statusMain').textContent = main; $('statusSub').textContent = sub;
  const note = $('statusNote');
  const notes = [];
  if (finite(m.track?.uncertain_from)) notes.push(`Ось пути неуверенна с ${fmt(m.track.uncertain_from)} м`);
  if (m.envelope_held) notes.push(`Габарит удержан по предыдущим кадрам (${m.envelope_coasted_frames}); текущие измерения — до ${fmt(m.observed_until_m, 0)} м`);
  if (m.turnout_ahead_m) notes.push(`Стрелка через ${fmt(m.turnout_ahead_m, 0)} м: маршрут за ней не определён`);
  note.classList.toggle('hidden', !notes.length); note.textContent = notes.join('. ');
  vp.classList.toggle('alarm', m.status === 'OBSTACLE');
  vp.classList.toggle('check', m.status === 'CHECK');
  $('mSpeed').textContent = fmt(m.speed_kmh, 0);
  $('mFree').textContent = finite(view.uncertainFrom) ? '—' : fmt(m.free_distance_m, 0);
  $('mFree').title = finite(view.uncertainFrom) ? 'Свободная дальность не подтверждена из-за неопределённой оси пути' : '';
  $('mEnvelope').textContent = fmt(m.envelope_until_m ?? m.observed_until_m, 0);
  $('mTtc').textContent = fmt(m.time_to_obstacle_s, 1);
  $('coverageRange').textContent = `${fmt(area.range)} м`;
  $('coverageFresh').style.width = `${100 * area.measured / area.range}%`;
  $('coverageHeld').style.width = `${100 * area.envelope / area.range}%`;
  $('coverageDetail').textContent = `Измерено ${fmt(area.measured)} м` + (area.held ? ` · удержано до ${fmt(area.envelope)} м` : '')
    + ` · далее нет наблюдения`;
  const fps = state.fpsHist.length ? 1000 * state.fpsHist.length / state.fpsHist.reduce((a, b) => a + b, 0) : null;
  const ms = m.timing_ms?.total;
  $('sysInfo').textContent = `обработка ${fmt(ms, 0)} мс` + (state.playing || state.mode === 'live' ? ` · ${fmt(fps, 1)} кадр/с` : '');
  $('sysInfo').classList.toggle('slow', ms > 100);
  if (m.frames) $('frameLabel').textContent = `кадр ${m.frame + 1} / ${m.frames}`;
  $('timeLabel').textContent = m.bag_time_s !== undefined ? `${fmt(m.bag_time_s, 1)} с записи` : '';
  const cal = m.calibration || {};
  $('calibBox').innerHTML = `<b>${cal.valid ? 'Самокалибровка по рельсам' : 'Начальная установка · калибровка не подтверждена'}</b><br>высота лидара над головкой рельса: ${fmt(cal.height_m, 2)} м<br>тангаж ${fmt(cal.pitch_deg, 2)}°, крен ${fmt(cal.roll_deg, 2)}°<br>измерения до ${fmt(area.measured)} м, габарит до ${fmt(area.envelope)} м<br>рельсы — до ${fmt(m.track?.rail_until)} м · фильтр убрал ${fmt((m.points || {}).filtered || 0)} изолированных возвратов`;
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
    const confidence = finite(o.confidence) ? clamp(o.confidence, 0, 1) : null;
    return `<button type="button" class="ob ${o.conf ? '' : 'pending'} ${o.low ? 'low' : ''} ${state.selectedObstacle === o.track_id ? 'selected' : ''}" data-id="${o.track_id}" title="Показать объект №${o.track_id} в 3D" aria-pressed="${state.selectedObstacle === o.track_id}">
      <div class="ob-top"><span class="ob-dist">${fmt(o.distance_m, 1)} м</span><span class="ob-kind">${o.conf ? 'подтверждено' : 'проверка'} · №${o.track_id}</span></div>
      <div class="ob-info">${obKind(o)} · смещение ${fmt(o.lateral_m, 2)} м · высота ${fmt(o.box.h[0], 2)}–${fmt(o.box.h[1], 2)} м</div>
      <div class="ob-info">${o.points} точек · ${o.rings} лучей · замечен с ${fmt(o.first_seen_m)} м</div>
      <div class="ob-confidence" title="Оценка устойчивости сопровождения, не вероятность класса"><span>Сопровождение ${fmt(confidence === null ? null : confidence * 100)}%</span><span>${o.missed ? 'удержан по истории' : 'виден в этом кадре'}</span></div>
      <div class="confidence-track"><i style="width:${confidence === null ? 0 : confidence * 100}%"></i></div>
      </button>`;
  }).join('');
  if (!all.length) $('obstacles').innerHTML = '<div class="empty-objects">Подтверждённых объектов и кандидатов нет</div>';
  $('obstacles').querySelectorAll('.ob').forEach((el) => el.onclick = () => {
    state.selectedObstacle = Number(el.dataset.id);
    scene.focusObstacle(state.selectedObstacle);
    syncCam('target'); renderObstacles(state.frame);
  });
}

function renderSynthetic(m) {
  const syn = m.synthetic || [];
  $('synList').innerHTML = syn.map((o) => `<div class="synitem"><span>${SHAPE_RU[o.shape] || o.shape}</span><span>${fmt(o.s, 1)} м, смещение ${fmt(o.d, 1)} м</span></div>`).join('');
}

let alarmAudio = null;
$('soundOn').onchange = () => {
  if (!$('soundOn').checked) return;
  alarmAudio = alarmAudio || new AudioContext();
  alarmAudio.resume().catch(() => {});
};
function signalAlarm(m) {
  const active = new Set(m.obstacles.map((o) => o.track_id));
  if ($('soundOn').checked && alarmAudio && [...active].some((id) => !state.alarmIds.has(id))) {
    const tone = alarmAudio.createOscillator(), gain = alarmAudio.createGain();
    tone.frequency.value = 880; tone.type = 'square'; gain.gain.value = 0.05;
    tone.connect(gain).connect(alarmAudio.destination);
    tone.onended = () => { tone.disconnect(); gain.disconnect(); };
    tone.start(); tone.stop(alarmAudio.currentTime + 0.25);
  }
  state.alarmIds = active;
}

function drawRadar(m) {
  const cv = $('radar'), g = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
  const k = devicePixelRatio, L = 40 * k, R = W - 20 * k, maxS = m.analysis_range_m || 200;
  cv.title = `Схема пути: 0–${maxS} м`;
  const X = (s) => L + (R - L) * Math.min(Math.max(s, 0), maxS) / maxS;
  g.clearRect(0, 0, W, H);
  g.font = `${11 * k}px Segoe UI, sans-serif`;
  const y0 = 34 * k, y1 = 62 * k;
  // corridor band: observed (solid) / not observed (hatched)
  const obs = m.observed_until_m || 0;
  const col = decisionView(m).kind === 'bad' ? 'rgba(239,68,68,' : decisionView(m).kind === 'ok' ? 'rgba(34,197,94,' : 'rgba(245,158,11,';
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
  g.fillStyle = css('--muted');
  const tickStep = maxS > 250 ? 50 : 25;
  const ticks = Array.from({ length: Math.floor(maxS / tickStep) + 1 }, (_, index) => index * tickStep);
  if (ticks[ticks.length - 1] !== maxS) {
    if (maxS - ticks[ticks.length - 1] < tickStep * .55) ticks.pop();
    ticks.push(maxS);
  }
  for (const s of ticks) {
    g.fillRect(X(s), y1, k, 5 * k);
    g.textAlign = s === maxS ? 'right' : s === 0 ? 'left' : 'center';
    g.fillText(s === maxS ? `${s} м` : `${s}`, X(s), y1 + 16 * k);
  }
  g.textAlign = 'left';
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
  g.fillText('схема пути · заливка — текущие измерения', 12 * k, 12 * k);
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
  g.fillText(state.mode === 'live' ? 'последние 600 полученных кадров · голубая линия — скорость (км/ч)'
    : 'статус по кадрам · голубая линия — скорость (км/ч)', 4 * k, H - 2 * k);
}
$('timeline').onclick = (e) => {
  if (state.mode !== 'playback' || !state.frames) return;
  const r = e.target.getBoundingClientRect();
  const i = clamp(Math.floor((e.clientX - r.left) / r.width * state.frames), 0, state.frames - 1);
  send({ cmd: 'seek', index: i });
};

function renderEvents() {
  const ev = new Map();
  state.timeline.forEach((it) => { if (it && it.ids) it.ids.forEach((id) => { if (!ev.has(id)) ev.set(id, it); }); });
  // Like Sakhalin Ai's event review, use actual buttons with a frame label and
  // an accessible target. The server's detections determine these markers.
  $('tlEvents').innerHTML = [...ev.entries()].map(([id, it]) => `<button type="button" class="tl-chip" data-i="${it.i}" aria-label="Событие №${id}, кадр ${it.i + 1}, ${fmt(it.dist)} метров">кадр ${it.i + 1}: ${fmt(it.dist)} м</button>`).join('') || '<span class="scan-state">событий пока нет</span>';
  $('tlEvents').querySelectorAll('.tl-chip').forEach((c) => c.onclick = () => send({ cmd: 'seek', index: Math.max(0, Number(c.dataset.i) - 2) }));
}
function seekEvent(direction) {
  if (state.mode !== 'playback' || !state.frames) return;
  const starts = [];
  let previous = false;
  state.timeline.forEach((item, index) => {
    const alarm = item?.status === 'OBSTACLE';
    if (alarm && !previous) starts.push(index);
    previous = alarm;
  });
  const current = state.frame?.frame ?? -1;
  const next = direction > 0 ? starts.find((index) => index > current) : starts.filter((index) => index < current).pop();
  if (next === undefined) { notify(direction > 0 ? 'Дальше нет известных событий. Выполните анализ проезда.' : 'Предыдущих известных событий нет.'); return; }
  send({ cmd: 'seek', index: next });
}
$('previousEvent').onclick = () => seekEvent(-1);
$('nextEvent').onclick = () => seekEvent(1);

// ------------------------------------------------------------------ perf
function renderPerf(m, record = true) {
  const t = m.timing_ms;
  const rows = [['decode', 'Преобразование облака'], ['track', 'Калибровка и путь'], ['egomotion', 'Эго-движение'], ['filter', 'Фильтр'], ['detect', 'Кандидаты'], ['tracking', 'Треки'],
    ...[['neural_path', 'Доп. оценка оси'], ['learned_detection', 'Доп. поиск'], ['learned_tracking', 'Доп. сопровождение'], ['component_model', 'Оценка кластеров']].filter(([key]) => t[key] > 0), ['total', 'Всего']];
  $('perfBars').innerHTML = rows.map(([k, name]) => `<div class="perf-row"><span>${name}</span><div class="perf-bar"><i style="width:${Math.min(100, (t[k] || 0))}%;${k === 'total' ? 'background:' + ((t.total || 0) > 100 ? '#ef4444' : '#22c55e') : ''}"></i></div><span>${fmt(t[k], 1)} мс</span></div>`).join('');
  if (record && finite(t.total)) state.perfHist.push(t.total);
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

// ------------------------------------------------------------------ controls
function openBag(id) {
  if (!send({ cmd: 'open', bag: id })) return;
  clearFrame(); state.frames = 0; state.timeline = []; state.bag = state.bags.find((bag) => bag.id === id) || null;
  drawTimeline(); $('tlEvents').replaceChildren(); $('scanState').textContent = '';
}
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
  syncCam(mode);
}
function syncCam(mode) {
  document.querySelectorAll('#cams button').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));
}
scene.onCameraChange = syncCam;
$('resetCamera').onclick = () => scene.resetCamera();
document.querySelectorAll('#cams button[data-cam]').forEach((b) => b.onclick = () => setCam(b.dataset.cam));

document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('hidden', t.id !== `tab-${b.dataset.tab}`));
});

document.querySelectorAll('[data-layer]').forEach((c) => c.onchange = () => scene.setLayer(c.dataset.layer, c.checked));
function layerPreset(raw) {
  const normal = new Set(['points', 'envelope', 'path', 'learned', 'markers', 'boxes']);
  document.querySelectorAll('[data-layer]').forEach((input) => {
    input.checked = raw ? input.dataset.layer === 'points' : normal.has(input.dataset.layer);
    scene.setLayer(input.dataset.layer, input.checked);
  });
  if (raw) { $('colorMode').value = 'intensity'; scene.setColorMode('intensity'); }
}
$('rawOnly').onclick = () => layerPreset(true);
$('defaultLayers').onclick = () => { layerPreset(false); $('colorMode').value = 'class'; scene.setColorMode('class'); };
$('colorMode').onchange = (e) => scene.setColorMode(e.target.value);
$('pointSize').oninput = (e) => scene.setPointSize(Number(e.target.value));

$('synDist').oninput = (e) => { $('synDistVal').textContent = e.target.value; };
$('synLat').oninput = (e) => { $('synLatVal').textContent = Number(e.target.value).toFixed(1); };
$('synYaw').oninput = (e) => { $('synYawVal').textContent = e.target.value; };
$('synAdd').onclick = () => send({ cmd: 'add_obstacle', shape: $('synShape').value, s: Number($('synDist').value), d: Number($('synLat').value), yaw_deg: Number($('synYaw').value) });
$('synClear').onclick = () => send({ cmd: 'clear_obstacles' });
$('scene').addEventListener('click', (e) => {
  if (!e.shiftKey) return;
  const p = scene.pickTrack(e.clientX, e.clientY);
  if (p) send({ cmd: 'add_obstacle', shape: $('synShape').value, s: p.s, d: Math.max(-1.5, Math.min(1.5, p.d)), yaw_deg: Number($('synYaw').value) });
});

const PRESETS = {
  balanced: {},
  strict: { tracking: { confirm_hits: 4, confirm_window: 5, far_confirm_hits: 8, far_confirm_window: 10 }, detection: { min_score: 0.65 } },
  sensitive: { tracking: { confirm_hits: 2, confirm_window: 3, far_confirm_hits: 4, far_confirm_window: 6 }, detection: { min_score: 0.4 } },
};
function setDetectorControls(cfg) {
  [['pWidth', cfg.envelope.width], ['pHeight', cfg.envelope.height],
    ['pMinH', cfg.envelope.min_height], ['pRange', Math.min(cfg.track.horizon, cfg.detection.max_range)]].forEach(([id, value]) => {
    $(id).value = value; $(id).dispatchEvent(new Event('input'));
  });
}
[['pWidth', 'pWidthVal', 1], ['pHeight', 'pHeightVal', 1], ['pMinH', 'pMinHVal', 2], ['pRange', 'pRangeVal', 0]].forEach(([id, lab, d]) => {
  $(id).oninput = (e) => { $(lab).textContent = Number(e.target.value).toFixed(d); };
});
$('pApply').onclick = () => {
  const p = JSON.parse(JSON.stringify(PRESETS[$('preset').value]));
  p.envelope = { width: Number($('pWidth').value), height: Number($('pHeight').value), min_height: Number($('pMinH').value) };
  p.detection = { ...(p.detection || {}), max_range: Number($('pRange').value) };
  p.track = { horizon: Number($('pRange').value) };
  const profile = state.sensorProfiles[$('sensorProfile').value];
  if (profile) p.sensor = profile.sensor;
  send({ cmd: 'config', overrides: p });
};
$('sensorProfile').onchange = () => {
  const profile = state.sensorProfiles[$('sensorProfile').value];
  if (profile) { $('pRange').value = profile.max_range_m; $('pRange').dispatchEvent(new Event('input')); }
};
$('pReset').onclick = () => {
  $('preset').value = 'balanced';
  $('sensorProfile').value = 'current';
  if (state.baseConfig) setDetectorControls(state.baseConfig);
  send({ cmd: 'config', overrides: {} });
};

// ------------------------------------------------------------------ theme
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('gabarit-theme', t); } catch (e) { /* private mode */ }
  $('btnTheme').innerHTML = t === 'light' ? '&#9790;' : '&#9788;';
  scene.setTheme(t === 'light');
  if (state.frame) { drawRadar(state.frame); renderPerf(state.frame, false); }
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

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.altKey || e.metaKey || e.repeat || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)
      || document.activeElement.isContentEditable || !$('modal').classList.contains('hidden')) return;
  if (e.code === 'KeyR') { e.preventDefault(); scene.resetCamera(); }
  if (e.code === 'Space') { e.preventDefault(); $('btnPlay').click(); }
  else if (e.code === 'ArrowRight') $('btnNext').click();
  else if (e.code === 'ArrowLeft') $('btnPrev').click();
  else if (/^Digit[1-6]$/.test(e.code)) setCam(['cab', 'chase', 'top', 'side', 'target', 'free'][Number(e.code.slice(5)) - 1]);
});
window.addEventListener('resize', () => { if (state.frame) { drawRadar(state.frame); renderPerf(state.frame, false); } drawTimeline(); });

connect();
