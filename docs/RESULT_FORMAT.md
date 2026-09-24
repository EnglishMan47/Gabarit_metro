# Формат результата

## Топики ROS 2

| Топик | Тип | Содержание |
|---|---|---|
| `/gabarit/obstacle` | `std_msgs/Bool` | `true` — впереди подтверждённое препятствие, нужно тормозить |
| `/gabarit/distance` | `std_msgs/Float32` | расстояние до ближайшего препятствия, м (`NaN`, если путь свободен) |
| `/gabarit/free_distance` | `std_msgs/Float32` | путь проверен и свободен до этого расстояния, м |
| `/gabarit/status` | `std_msgs/String` | полный JSON-результат кадра (ниже) |
| `/gabarit/markers` | `visualization_msgs/MarkerArray` | габарит, ось пути, рамки и подписи препятствий для RViz2 |
| `/gabarit/obstacle_points` | `sensor_msgs/PointCloud2` | точки подтверждённых препятствий |

Все геометрические выходы — в системе координат входного облака (`header.frame_id`).

## JSON кадра (`/gabarit/status`, веб-интерфейс, `scripts/evaluate.py`)

```json
{
  "frame": 57,
  "stamp": 1788354628.81,
  "status": "OBSTACLE",              // CLEAR | CHECK | OBSTACLE | DEGRADED
  "obstacle": true,
  "nearest_distance_m": 55.6,        // евклидово расстояние от лидара до ближайшей точки объекта
  "nearest_track_distance_m": 55.5,  // вдоль пути
  "free_distance_m": 55.5,           // до препятствия или до конца проверенного пути
  "observed_until_m": 107.0,         // путь подтверждён измерениями до
  "speed_kmh": 0.0,                  // скорость поезда по лидару
  "stopping_distance_m": 0.0,        // тормозной путь (реакция 1,5 с, замедление 1,1 м/с²), справочно
  "time_to_obstacle_s": null,
  "turnout_ahead_m": null,           // распознанная впереди стрелка (маршрут на ней не определён)
  "obstacles": [{
    "track_id": 3, "confirmed": true,
    "distance_m": 55.6, "s_m": 55.5, "lateral_m": 0.35,
    "box": {"s": [55.5, 55.9], "d": [0.1, 0.6], "h": [0.18, 1.55]},  // вдоль пути / поперёк / над головкой рельса
    "size_m": [0.4, 0.5, 1.37],
    "points": 112, "rings": 12,
    "low": false,                    // true — низкий объект на рельсе
    "confidence": 0.98,
    "first_seen_m": 55.7,            // на каком расстоянии объект впервые стал кандидатом
    "confirmed_at_m": 55.6,
    "reasons": ["112 точек на 12 лучах"]
  }],
  "pending": [],                     // кандидаты, ещё не подтверждённые (статус CHECK)
  "candidates": [ ... ],             // все кластеры кадра с причиной отбраковки ("rejected")
  "calibration": {"height_m": 1.48, "pitch_deg": -0.9, "roll_deg": -3.5, "valid": true},
  "timing_ms": {"decode": 20.1, "track": 30.2, "egomotion": 2.4, "detect": 14.9, "tracking": 0.1, "total": 67.7}
}
```

`scripts/evaluate.py` сохраняет такие записи построчно (`frames/<bag>.jsonl`) и сводку по записям (`summary.json`: кадры с тревогой, события, задержки p50/p95).

## Журнал оператора

Решения оператора в веб-интерфейсе («Подтвердить» / «Ложная тревога») пишутся в `artifacts/operator_journal.jsonl` и выгружаются в CSV (`/api/journal.csv`): время, запись, кадр, номер трека, решение, расстояние, смещение, высота, признак испытательного объекта.
