import { LidarViewer3D } from "./viewer3d.js";

const elements = Object.fromEntries([
  "bagSelect", "bagMeta", "frameInput", "frameSlider", "previousButton", "nextButton", "playButton",
  "corridorWidth", "corridorCenter", "floorZ", "maxForward", "applyButton", "pathStatus", "statusText",
  "statusDetail", "distanceValue", "objectCount", "processingValue", "frameBadge", "pointBadge", "topCanvas",
  "sideCanvas", "incidentText", "confirmButton", "dismissButton", "eventLog", "clearLogButton", "connectionDot",
  "connectionText", "toast", "analysisMeta", "objectRows", "trackBadge", "eventFilter",
  "inferenceMode", "inferenceStatus", "threeViewer"
].map(id => [id, document.getElementById(id)]));

const state = { bags: [], frame: null, capabilities: null, playing: false, loading: false, eventCount: 0, request: null, requestId: 0, renderedAt: null, fps: null };
const sessionId = crypto.randomUUID();
const bagNames = {
  doubleT_obstacle: "Двойной тоннель — препятствие",
  doubleT_platform: "Двойной тоннель — платформа",
  roundT_doubleT: "Круглый → двойной тоннель",
  roundT_pressureGate_roundT: "Круглый тоннель — гермоворота",
  roundT_squareT_pressureGate_squareT: "Круглый → прямоугольный — гермоворота",
  squareT_platform_squareT_switch: "Прямоугольный тоннель — платформа и стрелка"
};
const backendNames = {
  cuda: "Видеокарта NVIDIA",
  cpu: "Процессор",
  geometry: "Геометрический алгоритм"
};
const fallbackNames = {
  model_missing: "модель не найдена",
  onnxruntime_missing: "модуль нейросети недоступен",
  cuda_unavailable: "видеокарта недоступна",
  cpu_unavailable: "процессорный режим недоступен",
  session_failed: "не удалось запустить видеокарту",
  inference_failed: "ошибка выполнения модели",
  provider_unavailable: "вычислительный модуль недоступен",
  bev_missing: "не удалось подготовить данные"
};
const rejectionNames = {
  background_occupancy: "совпали с фоном",
  rail_like: "похожи на рельсы",
  infrastructure_shape: "похожи на инфраструктуру"
};
let playbackTimer = null;
let viewer3d = null;
try {
  viewer3d = new LidarViewer3D(elements.threeViewer);
} catch (error) {
  elements.threeViewer.hidden = true;
  elements.topCanvas.hidden = false;
  console.warn("Three.js viewer unavailable; using Canvas fallback", error);
}

function setConnection(ok, text) {
  elements.connectionText.textContent = text;
  elements.connectionDot.style.background = ok ? "#57e6a1" : "#ffbd59";
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => elements.toast.classList.remove("show"), 2600);
}

function currentBag() { return state.bags.find(bag => bag.id === elements.bagSelect.value); }
function clampFrame(value) {
  const bag = currentBag();
  return Math.max(0, Math.min(bag ? bag.frame_count - 1 : 0, Math.floor(Number(value) || 0)));
}

async function loadBags() {
  const response = await fetch("/api/bags");
  if (!response.ok) throw new Error("Не удалось получить список записей лидара");
  state.bags = (await response.json()).bags;
  elements.bagSelect.innerHTML = state.bags.map(bag => `<option value="${escapeHtml(bag.id)}">${escapeHtml(bagNames[bag.name] || bag.name)}</option>`).join("");
  if (!state.bags.length) {
    elements.bagMeta.textContent = "В каталоге данных не найдено файлов .db3";
    return;
  }
  updateBag();
}

async function loadCapabilities() {
  const response = await fetch("/api/capabilities");
  if (!response.ok) throw new Error("Не удалось проверить доступные способы обработки");
  state.capabilities = await response.json();
  const caps = state.capabilities;
  elements.inferenceStatus.classList.toggle("warning", !caps.model_available);
  if (!caps.model_available) {
    elements.inferenceStatus.innerHTML = "Используется <strong>геометрический алгоритм</strong>: модель не найдена";
  } else if (caps.recommended_mode === "gpu") {
    elements.inferenceStatus.innerHTML = "Рекомендуется <strong>видеокарта</strong>";
  } else if (caps.recommended_mode === "cpu") {
    elements.inferenceStatus.innerHTML = "Рекомендуется <strong>процессор</strong>";
  } else {
    elements.inferenceStatus.innerHTML = "Модель недоступна; используется <strong>геометрический алгоритм</strong>";
  }
}

function updateBag() {
  togglePlayback(false);
  const bag = currentBag();
  if (!bag) return;
  elements.frameSlider.max = bag.frame_count - 1;
  elements.frameInput.max = bag.frame_count - 1;
  elements.frameSlider.value = 0;
  elements.frameInput.value = 0;
  const sourceName = bag.topic.includes("hesai128") ? "Лидар Hesai 128" : "Облако точек лидара";
  elements.bagMeta.innerHTML = `<span class="meta-primary">${bag.frame_count.toLocaleString("ru-RU")} кадров · ${bag.duration_s.toFixed(1)} с</span><span class="meta-secondary" title="${escapeHtml(bag.topic)}">Источник: ${sourceName}</span>`;
  loadFrame(0);
}

function queryForFrame(index) {
  const params = new URLSearchParams({
    bag: elements.bagSelect.value,
    session: sessionId,
    index: String(index),
    corridor_width: elements.corridorWidth.value,
    corridor_center_x: elements.corridorCenter.value,
    floor_z: elements.floorZ.value,
    max_forward: elements.maxForward.value,
    inference_mode: elements.inferenceMode.value,
    response: "packet"
  });
  return `/api/frame?${params}`;
}

async function decodeFrameResponse(response) {
  const contentType = response.headers.get("Content-Type") || "";
  if (!contentType.includes("application/vnd.metro-lidar.frame")) return response.json();
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 4) throw new Error("Сервер вернул повреждённый кадр");
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  if (headerLength <= 0 || 4 + headerLength > buffer.byteLength) throw new Error("Некорректный заголовок кадра");
  const header = new TextDecoder("utf-8").decode(new Uint8Array(buffer, 4, headerLength));
  const payload = JSON.parse(header);
  const pointOffset = (4 + headerLength + 3) & ~3;
  const expectedFloats = payload.point_count * payload.point_stride;
  if (pointOffset + expectedFloats * 4 !== buffer.byteLength) throw new Error("Некорректный размер массива точек");
  payload.points = new Float32Array(buffer, pointOffset, expectedFloats);
  return payload;
}

async function loadFrame(value) {
  clearTimeout(playbackTimer);
  if (!currentBag()) return;
  const index = clampFrame(value);
  if (state.request) state.request.abort();
  const controller = new AbortController();
  const requestId = ++state.requestId;
  state.request = controller;
  state.loading = true;
  setConnection(false, "Кадр обрабатывается…");
  try {
    const response = await fetch(queryForFrame(index), { signal: controller.signal });
    const payload = await decodeFrameResponse(response);
    if (!response.ok) throw new Error(payload.error || "Ошибка обработки кадра");
    if (requestId !== state.requestId) return;
    state.frame = payload;
    elements.frameInput.value = index;
    elements.frameSlider.value = index;
    render(payload);
    setConnection(true, "Сервер подключён");
  } catch (error) {
    if (error.name === "AbortError") return;
    setConnection(false, "Ошибка подключения");
    toast(error.message);
    state.playing = false;
    elements.playButton.textContent = "▶ Запустить просмотр";
  } finally {
    if (requestId === state.requestId) {
      state.loading = false;
      state.request = null;
    }
  }
  if (state.playing && requestId === state.requestId) {
    const next = index + 1;
    if (next < currentBag().frame_count) playbackTimer = setTimeout(() => {
      if (state.playing && requestId === state.requestId) loadFrame(next);
    }, 0);
    else togglePlayback(false);
  }
}

function render(frame) {
  const renderedAt = performance.now();
  if (state.renderedAt != null) state.fps = 1000 / Math.max(1, renderedAt - state.renderedAt);
  state.renderedAt = renderedAt;
  const danger = frame.event_detected;
  const candidate = frame.raw_detected;
  elements.pathStatus.classList.toggle("danger", danger);
  elements.pathStatus.classList.toggle("safe", !danger);
  elements.statusText.textContent = danger ? "Обнаружено препятствие" : candidate ? "Объект проверяется" : "Путь свободен";
  elements.statusDetail.textContent = danger ? "Объект устойчиво обнаружен в нескольких кадрах" : candidate ? "Алгоритм проверяет объект по последовательности кадров" : "В контролируемом коридоре объектов нет";
  const shownDistance = frame.event_nearest_distance_m ?? frame.nearest_distance_m;
  elements.distanceValue.textContent = shownDistance == null ? "—" : shownDistance.toFixed(1);
  elements.objectCount.textContent = frame.objects.length;
  elements.processingValue.textContent = frame.processing_ms.toFixed(0);
  elements.frameBadge.textContent = `Кадр ${frame.frame_index}`;
  elements.pointBadge.textContent = `${frame.valid_point_count.toLocaleString("ru-RU")} точек`;
  elements.incidentText.textContent = candidate
    ? `Объектов на проверке: ${frame.objects.length}. Ближайший находится на расстоянии ${frame.nearest_distance_m.toFixed(1)} м. ${danger ? "Событие подтверждено алгоритмом." : "Идёт проверка по следующим кадрам."}`
    : danger ? "Событие временно сохраняется после краткого пропуска объекта. Показано последнее измерение." : "Препятствия не обнаружены. Решение оператора сейчас не требуется.";
  elements.confirmButton.disabled = !candidate;
  elements.dismissButton.disabled = !candidate;
  const rejectedEntries = Object.entries(frame.diagnostics.rejected || {});
  const rejectedCount = rejectedEntries.reduce((sum, [, count]) => sum + count, 0);
  const rejectedDetail = rejectedEntries.map(([name, count]) => `${rejectionNames[name] || name}: ${count}`).join("; ") || "нет отфильтрованных групп";
  const inference = frame.inference || { active_backend: "geometry", model_used: false, fallback_reason: null };
  const backendName = backendNames[inference.active_backend] || inference.active_backend;
  const fallbackName = fallbackNames[inference.fallback_reason] || inference.fallback_reason;
  const fallbackText = fallbackName ? ` — ${fallbackName}` : "";
  elements.inferenceStatus.classList.toggle("warning", Boolean(inference.fallback_reason));
  elements.inferenceStatus.innerHTML = `Используется: <strong>${escapeHtml(backendName)}</strong>${escapeHtml(fallbackText)}`;
  const modelTime = inference.model_latency_ms == null ? "—" : `${inference.model_latency_ms.toFixed(1)} мс`;
  const railCoverage = `${(frame.path_coverage * 100).toFixed(0)}%${frame.diagnostics.path_fallback_used ? " · резерв" : ""}`;
  elements.analysisMeta.innerHTML = [
    diagnosticCard("Обработка", backendName, "accent", fallbackName || "Фактически используемый способ обработки"),
    diagnosticCard("Нейросеть", modelTime),
    diagnosticCard("Покрытие пола", `${(frame.floor_coverage * 100).toFixed(0)}%`),
    diagnosticCard("Рельсы", railCoverage, frame.diagnostics.path_fallback_used ? "warning" : ""),
    diagnosticCard("Исходных точек", frame.raw_point_count.toLocaleString("ru-RU")),
    diagnosticCard("Обработано", frame.valid_point_count.toLocaleString("ru-RU")),
    diagnosticCard("Показано", frame.point_count.toLocaleString("ru-RU")),
    diagnosticCard("Декодирование", `${frame.decode_ms.toFixed(0)} мс`),
    diagnosticCard("Частота показа", state.fps == null ? "—" : `${state.fps.toFixed(1)} кадр/с`),
    diagnosticCard("Фильтр фона", rejectedCount ? `${rejectedCount} групп` : "нет", rejectedCount ? "warning" : "", rejectedDetail)
  ].join("");
  const confirmedTracks = frame.active_tracks.filter(track => track.confirmed && track.missing === 0).length;
  elements.trackBadge.textContent = confirmedTracks === 1 ? "1 подтверждён" : `${confirmedTracks} подтверждено`;
  elements.objectRows.innerHTML = frame.objects.length ? frame.objects.map(object => `<tr><td>${object.track_id ?? "—"}</td><td>${object.distance_surface_m.toFixed(1)} м</td><td>${object.point_count.toLocaleString("ru-RU")}</td><td>${object.track_confirmed ? "Подтверждён" : "На проверке"}</td></tr>`).join("") : '<tr><td colspan="4">Объекты не обнаружены</td></tr>';
  if (viewer3d) viewer3d.updateFrame(frame); else drawTop(frame);
  drawSide(frame);
}

function visitPoints(frame, callback) {
  const points = frame.points;
  if (points instanceof Float32Array) {
    for (let index = 0; index < points.length; index += 4) callback(points[index], points[index + 1], points[index + 2], points[index + 3]);
    return;
  }
  for (const [x, forward, z, intensity] of points) callback(x, forward, z, intensity);
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function grid(ctx, width, height, columns = 8, rows = 8) {
  ctx.fillStyle = "#050c0b"; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(130,174,161,.12)"; ctx.lineWidth = 1;
  for (let i = 1; i < columns; i++) { const x = i * width / columns; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let i = 1; i < rows; i++) { const y = i * height / rows; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
}

function pointColor(z, alpha = .62) {
  const normalized = Math.max(0, Math.min(1, (z + 2.5) / 4.5));
  const hue = 174 + normalized * 42;
  return `hsla(${hue},75%,65%,${alpha})`;
}

function drawTop(frame) {
  const { ctx, width, height } = prepareCanvas(elements.topCanvas); grid(ctx, width, height);
  const xLimit = 8, maxF = frame.corridor.max_forward_m;
  const px = x => (x + xLimit) / (2 * xLimit) * width;
  const py = forward => height - forward / maxF * height;
  const half = frame.corridor.width_m / 2;
  const profile = frame.path_profile.filter(item => item.forward_m <= maxF);
  ctx.fillStyle = "rgba(87,230,161,.07)"; ctx.beginPath();
  profile.forEach((item, index) => { const x = px(item.center_x_m - half), y = py(item.forward_m); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
  [...profile].reverse().forEach(item => ctx.lineTo(px(item.center_x_m + half), py(item.forward_m))); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(87,230,161,.55)"; ctx.setLineDash([7, 7]);
  [-half, half].forEach(offset => { ctx.beginPath(); profile.forEach((item, index) => { const x = px(item.center_x_m + offset), y = py(item.forward_m); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); }); ctx.setLineDash([]);
  ctx.fillStyle = "rgba(105,183,255,.85)";
  profile.filter(item => item.rail_confident).forEach(item => { ctx.fillRect(px(item.left_rail_x_m) - 1, py(item.forward_m) - 1, 3, 3); ctx.fillRect(px(item.right_rail_x_m) - 1, py(item.forward_m) - 1, 3, 3); });
  visitPoints(frame, (x, forward, z) => {
    ctx.fillStyle = pointColor(z); ctx.fillRect(px(x), py(forward), 1.3, 1.3);
  });
  for (const object of frame.objects) {
    ctx.strokeStyle = object.track_confirmed ? "#ff6272" : "#ffbd59"; ctx.lineWidth = 2;
    const x1 = px(object.min_m.x), x2 = px(object.max_m.x), y1 = py(object.max_m.forward), y2 = py(object.min_m.forward);
    ctx.strokeRect(x1, y1, Math.max(4, x2 - x1), Math.max(4, y2 - y1));
  }
}

function drawSide(frame) {
  const { ctx, width, height } = prepareCanvas(elements.sideCanvas); grid(ctx, width, height);
  const maxF = frame.corridor.max_forward_m, minZ = -3, maxZ = 2;
  const px = forward => forward / maxF * width;
  const py = z => height - (z - minZ) / (maxZ - minZ) * height;
  ctx.fillStyle = "rgba(87,230,161,.06)"; ctx.fillRect(0, py(frame.corridor.floor_z_m + frame.corridor.height_m), width, py(frame.corridor.floor_z_m) - py(frame.corridor.floor_z_m + frame.corridor.height_m));
  ctx.strokeStyle = "rgba(87,230,161,.72)"; ctx.lineWidth = 1.5; ctx.beginPath();
  frame.floor_profile.forEach((item, index) => { const x = px(item.forward_m), y = py(item.z_m); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
  visitPoints(frame, (_x, forward, z) => { ctx.fillStyle = pointColor(z); ctx.fillRect(px(forward), py(z), 1.3, 1.3); });
  for (const object of frame.objects) {
    ctx.strokeStyle = object.track_confirmed ? "#ff6272" : "#ffbd59"; ctx.lineWidth = 2;
    const x1 = px(object.min_m.forward), x2 = px(object.max_m.forward), y1 = py(object.max_m.z), y2 = py(object.min_m.z);
    ctx.strokeRect(x1, y1, Math.max(4, x2 - x1), Math.max(4, y2 - y1));
  }
  ctx.font = "11px Segoe UI, sans-serif";
  ctx.fillStyle = "rgba(190, 218, 209, .78)";
  ctx.textAlign = "center";
  const distanceStep = Math.max(10, Math.round(maxF / 4 / 10) * 10);
  for (let distance = 0; distance <= maxF; distance += distanceStep) {
    ctx.fillText(`${distance} м`, px(distance), height - 8);
  }
  ctx.textAlign = "left";
  for (let z = minZ; z <= maxZ; z += 1) ctx.fillText(`${z} м`, 7, Math.max(13, py(z) - 4));
}

function togglePlayback(force) {
  clearTimeout(playbackTimer);
  state.playing = force ?? !state.playing;
  elements.playButton.textContent = state.playing ? "■ Остановить просмотр" : "▶ Запустить просмотр";
  if (state.playing && !state.loading) loadFrame(Number(elements.frameInput.value));
}

async function sendFeedback(decision) {
  if (!state.frame) return;
  const response = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
    bag: state.frame.bag.id, frame_index: state.frame.frame_index, decision,
    nearest_distance_m: state.frame.nearest_distance_m
  }) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Не удалось сохранить решение");
  addEvent(payload);
  toast(decision === "confirmed" ? "Тревога подтверждена" : "Тревога отклонена");
}

function addEvent(record) {
  if (!state.eventCount) elements.eventLog.innerHTML = "";
  state.eventCount += 1;
  const item = document.createElement("li");
  item.dataset.decision = record.decision;
  item.innerHTML = `<strong>Кадр ${record.frame_index}</strong> — ${record.decision === "confirmed" ? "подтверждено" : "отклонено"}`;
  elements.eventLog.prepend(item);
  applyEventFilter();
}

function applyEventFilter() {
  const selected = elements.eventFilter.value;
  elements.eventLog.querySelectorAll("li[data-decision]").forEach(item => {
    item.hidden = selected !== "all" && item.dataset.decision !== selected;
  });
}

async function loadFeedback() {
  const response = await fetch("/api/feedback");
  if (!response.ok) return;
  const records = (await response.json()).records;
  records.forEach(addEvent);
}

function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]); }

function diagnosticCard(label, value, className = "", title = "") {
  const titleAttribute = title ? ` title="${escapeHtml(title)}"` : "";
  return `<div class="analysis-item ${escapeHtml(className)}"${titleAttribute}><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

elements.bagSelect.addEventListener("change", updateBag);
elements.previousButton.addEventListener("click", () => loadFrame(clampFrame(Number(elements.frameInput.value) - 1)));
elements.nextButton.addEventListener("click", () => loadFrame(clampFrame(Number(elements.frameInput.value) + 1)));
elements.frameInput.addEventListener("change", event => loadFrame(event.target.value));
elements.frameSlider.addEventListener("change", event => loadFrame(event.target.value));
elements.playButton.addEventListener("click", () => togglePlayback());
elements.applyButton.addEventListener("click", () => loadFrame(elements.frameInput.value));
elements.confirmButton.addEventListener("click", () => sendFeedback("confirmed").catch(error => toast(error.message)));
elements.dismissButton.addEventListener("click", () => sendFeedback("dismissed").catch(error => toast(error.message)));
elements.clearLogButton.addEventListener("click", () => { state.eventCount = 0; elements.eventLog.innerHTML = '<li class="empty-log">Событий пока нет</li>'; });
elements.eventFilter.addEventListener("change", applyEventFilter);
elements.inferenceMode.addEventListener("change", () => loadFrame(elements.frameInput.value));
window.addEventListener("resize", () => { viewer3d?.resize(); if (state.frame && !viewer3d) render(state.frame); });

Promise.all([loadCapabilities(), loadBags(), loadFeedback()]).then(() => setConnection(true, "Сервер подключён")).catch(error => { setConnection(false, "Сервер недоступен"); toast(error.message); });
