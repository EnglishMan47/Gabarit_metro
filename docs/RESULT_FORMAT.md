# Формат результата детектора

Версия контракта: `metro-lidar-evaluation-v1`. Все расстояния указаны в метрах, время обработки — в миллисекундах, временные метки bag — в наносекундах.

## HTTP-ответ кадра

`GET /api/frame?bag=<id>&index=<n>` возвращает:

- `raw_detected` — в текущем кадре есть хотя бы один геометрический кандидат;
- `event_detected` и `obstacle_detected` — кандидат подтверждён временным трекером;
- `nearest_distance_m` — расстояние до ближайшей видимой поверхности сырого кандидата;
- `event_nearest_distance_m` — ближайшее расстояние по последним наблюдениям активных подтверждённых треков; при кратком пропуске используется последнее измерение;
- `objects` — объекты с `track_id`, центром, bbox, размерами, числом точек, высотой над полом, долей совпадения с фоном и причинами;
- `floor_profile` — локальная высота пола по продольным сегментам;
- `path_profile` — центр коридора и распознанные рельсы по сегментам;
- `active_tracks` — состояние временных треков;
- `diagnostics` — причины удаления кластеров и признаки резервной геометрии;
- `points` — прореженное облако только для визуализации.

Пример объекта:

```json
{
  "track_id": 3,
  "track_confirmed": true,
  "point_count": 184,
  "distance_surface_m": 54.91,
  "center_m": {"x": 0.31, "forward": 55.62, "z": -1.71},
  "min_m": {"x": -0.12, "forward": 55.31, "z": -2.18},
  "max_m": {"x": 0.72, "forward": 55.94, "z": -0.72},
  "height_above_floor_m": {"min": 0.12, "max": 1.58},
  "background_fraction": 0.08,
  "reasons": ["compact_cluster", "above_floor", "background_deviation"]
}
```

## Пакетная оценка

Команда:

```powershell
python scripts/evaluate_detector.py --step 1
```

создаёт `artifacts/evaluation/final-summary.json` и по одному JSONL-файлу на bag в `artifacts/evaluation/final-frames/`. Каждая строка JSONL самодостаточна и содержит номер кадра, временную метку, сырую и устойчивую тревогу, расстояние, времена обработки, объекты и диагностику.

## Обратная связь оператора

`POST /api/feedback` принимает `bag`, `frame_index`, `decision` (`confirmed` или `dismissed`), необязательные `nearest_distance_m` и `comment`. Записи добавляются в `artifacts/operator_feedback.jsonl`. `GET /api/export?format=json` и `format=csv` возвращают журнал для выгрузки.
