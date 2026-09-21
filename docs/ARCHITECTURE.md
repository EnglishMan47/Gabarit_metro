# Архитектура локального веб-решения

```text
.db3 -> BagStore/CDR decoder -> detector -> tracker -> HTTP JSON -> Canvas UI
                                  |            |             |
                         normal_scene.npz   events      feedback JSONL
                                  |
                              evaluator -> JSONL + summary
```

- `webapp/bag_store.py` находит bag и выдаёт нормализованные кадры.
- `webapp/detector.py` оценивает пол и путь, формирует и фильтрует кластеры.
- `webapp/scene_model.py` хранит занятость штатной сцены.
- `webapp/tracking.py` присваивает стабильные идентификаторы и подтверждает событие.
- `webapp/server.py` предоставляет API, статические файлы, журнал и экспорт.
- `webapp/static/` содержит автономный интерфейс без CDN.
- `scripts/evaluate_detector.py` запускает тот же алгоритм без браузера.

Приложение работает локально на Windows и использует только Python и NumPy. ROS, Docker, база данных сервера, интернет и внешние API не требуются.
