// Operator-state rules inspired by Sakhalin Ai's explicit UNKNOWN and
// Attention Please's separate measured/held range. These are display rules;
// detection and temporal confirmation always remain on the server.
export const finite = (value) => typeof value === 'number' && Number.isFinite(value);
export const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

export function coverage(frame) {
  const range = finite(frame.analysis_range_m) && frame.analysis_range_m > 0 ? frame.analysis_range_m : 200;
  const measured = clamp(finite(frame.current_observed_until_m) ? frame.current_observed_until_m
    : finite(frame.observed_until_m) ? frame.observed_until_m : 0, 0, range);
  const envelope = clamp(finite(frame.envelope_until_m) ? frame.envelope_until_m : measured, measured, range);
  return { range, measured, envelope, held: Boolean(frame.envelope_held) || envelope > measured + 0.5 };
}

export function decisionView(frame) {
  const area = coverage(frame);
  if (frame.status === 'OBSTACLE') return { kind: 'bad', label: 'ПРЕПЯТСТВИЕ', area };
  if (frame.status === 'CHECK') return { kind: 'warn', label: 'ПРОВЕРКА ОБЪЕКТА', area };
  if (frame.status === 'DEGRADED') return { kind: 'warn', label: 'НЕТ ОПОРЫ ПУТИ', area };
  if (frame.status !== 'CLEAR' || area.measured <= 0) return { kind: 'warn', label: 'НЕДОСТАТОЧНО ДАННЫХ', area };
  const uncertainFrom = frame.track?.uncertain_from;
  if (finite(uncertainFrom) && uncertainFrom >= 0) {
    return { kind: 'warn', label: 'ГАБАРИТ УТОЧНЯЕТСЯ', area, uncertainFrom };
  }
  if (area.held) return { kind: 'warn', label: 'ГАБАРИТ УДЕРЖАН', area };
  return { kind: 'ok', label: 'ПРОВЕРЕНО ДО', area };
}

export function streamView({ connected, mode, playing, hasFrame, lastArrive, now }) {
  if (!connected) return { kind: 'off', label: 'Нет соединения', stale: hasFrame };
  if (!hasFrame) return { kind: 'wait', label: 'Ожидание облака', stale: false };
  const age = Math.max(0, now - lastArrive);
  if ((mode === 'live' || playing) && age > 3000) {
    return { kind: 'off', label: `Нет нового кадра ${Math.floor(age / 1000)} с`, stale: true };
  }
  if (mode !== 'live' && !playing) return { kind: 'pause', label: 'Запись на паузе · последний кадр', stale: false };
  return { kind: 'on', label: mode === 'live' ? 'Живой поток' : 'Воспроизведение', stale: false };
}

export function decodeFrame(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 12) throw new Error('Неполный кадр облака');
  const data = new DataView(buffer);
  if (data.getUint32(0, false) !== 0x47425231) throw new Error('Неизвестный формат облака');
  const jsonLength = data.getUint32(4, true);
  const offset = 8 + jsonLength;
  if (jsonLength < 2 || offset % 4 || offset + 4 > buffer.byteLength) throw new Error('Повреждён заголовок облака');
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, jsonLength)));
  const n = data.getUint32(offset, true);
  if (offset + 4 + n * 14 !== buffer.byteLength) throw new Error('Неполный массив точек облака');
  if (!meta || !meta.track || !Array.isArray(meta.obstacles) || !Array.isArray(meta.pending)) {
    throw new Error('В кадре отсутствуют данные детектора');
  }
  const path = meta.track;
  if (!Array.isArray(path.s) || !Array.isArray(path.c) || !Array.isArray(path.z)
      || path.s.length < 2 || path.c.length !== path.s.length || path.z.length !== path.s.length
      || [...path.s, ...path.c, ...path.z].some((value) => !finite(value))) {
    throw new Error('В кадре повреждена геометрия пути');
  }
  return { meta, points: {
    n, xyz: new Float32Array(buffer, offset + 4, n * 3),
    cls: new Uint8Array(buffer, offset + 4 + n * 12, n),
    inten: new Uint8Array(buffer, offset + 4 + n * 13, n),
  } };
}
