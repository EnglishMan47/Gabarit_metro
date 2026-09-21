# ЛЦТ 2026 — задача №5

Система обнаружения посторонних объектов для беспилотных поездов в тоннеле метро по данным 3D-лидара.

Главные документы: [план доведения](FINALIZATION_PLAN.md), [архитектура](docs/ARCHITECTURE.md), [алгоритм](docs/ALGORITHM.md), [формат результата](docs/RESULT_FORMAT.md), [веб-интерфейс](docs/WEB_APP.md), [оценка качества](docs/EVALUATION.md) и [текущее состояние](docs/STATUS.md).

По новому указанию организаторов итогом является рабочее веб-приложение без Docker и ROS. Приложение напрямую читает SQLite3 bag, декодирует `PointCloud2`, запускает геометрический детектор и показывает результат в браузере.

## Запуск веб-приложения

```powershell
.\scripts\setup_web.ps1 -Inference gpu
.\scripts\run_web.ps1
```

Для компьютера без NVIDIA используйте `-Inference cpu`; минимальный геометрический вариант устанавливается с `-Inference geometry`.

Откройте `http://127.0.0.1:8000`. Каталог данных по умолчанию: `data/extracted/for_hackathon`.

После установки можно запускать приложение двойным щелчком по `start_web.cmd`. Окно сервера должно оставаться открытым. Если сайт не отвечает, убедитесь, что в нём нет ошибки занятого порта или отсутствующего каталога данных.

Статус готовности и актуальные ограничения: [RELEASE_READINESS.md](docs/RELEASE_READINESS.md). Прежние отметки реализации этапов не означают прохождение целевых критериев качества.

Для другого каталога:

```powershell
.\scripts\run_web.ps1 -DataRoot "D:\lidar\for_hackathon"
```

## Проверка и пакетная оценка

Автономная установка из локальных пакетов и результаты проверки чистого окружения описаны в [INSTALLATION_CHECK.md](docs/INSTALLATION_CHECK.md).

`requirements.txt` содержит только обязательную runtime-зависимость NumPy. Пока ONNX-модель отсутствует, приложение автоматически использует геометрический режим. После обучения модели доступны отдельные варианты установки: `requirements-inference-cpu.txt` для CPU, `requirements-inference-gpu.txt` для NVIDIA GPU и `requirements-training.txt` для обучения. CPU- и GPU-пакеты ONNX Runtime не устанавливаются одновременно. Подробный контракт переключения описан в [HYBRID_INFERENCE.md](docs/HYBRID_INFERENCE.md).

Веб-интерфейс запрашивает кадр в бинарном формате: метаданные остаются JSON, а точки передаются как little-endian `Float32Array`. Это уменьшает сетевой ответ без изменения вычислений детектора.

Развёрнутое для проекта WSL-окружение использует Ubuntu 24.04 и изолированный Python в `/root/.venvs/metro-bev`. Проверка GPU и запуск экспериментального обучения:

```powershell
.\scripts\setup_training_wsl.ps1
.\scripts\train_bev_wsl.ps1 -Device cuda -Epochs 20 -BatchSize 16
```

Скрипт сначала выполняет реальную матричную операцию на CUDA, затем обучает модель, проверяет корректность ONNX и сохраняет `models/bev_segmentation.onnx`. До появления отдельного положительного проезда эта модель помечена как `development_only` и используется в интерфейсе только как диагностический слой; решение о тревоге остаётся за геометрическим детектором.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\build_normal_scene.py --step 10
.\.venv\Scripts\python.exe scripts\evaluate_detector.py --step 1
```

Evaluator сохраняет сводку в `artifacts/evaluation/final-summary.json`, а покадровые результаты — в `artifacts/evaluation/final-frames/`. Эти runtime-артефакты не предназначены для Git.

## Инструменты данных

Инвентаризация файлов — Python 3.11+, без сторонних зависимостей:

```powershell
python scripts/audit_dataset.py --input "ПУТЬ_К_ДАТАСЕТУ" --output artifacts/data_audit.json
```

Инструмент рекурсивно инвентаризирует обычные файлы, считает размеры и SHA-256, показывает распределение по расширениям, дубликаты и ошибки чтения. Он не распаковывает архивы, не интерпретирует координаты и не считает метрики детекции. Символические ссылки и Windows junction пропускаются.

Проверка реальных `PointCloud2` напрямую из SQLite3 bag требует NumPy:

```powershell
python scripts/inspect_pointcloud_bag.py data/extracted/for_hackathon --frames 3 --output artifacts/pointcloud_inspection.json
python scripts/render_pointcloud_frame.py data/extracted/for_hackathon/doubleT_obstacle/doubleT_obstacle_0.db3 --index 100 --output artifacts/frame.png --lateral -3 3 --forward 0 90 --height -2 2.5
```

Первый скрипт проверяет CDR, поля, размеры и диапазоны координат; второй строит три диагностические проекции в PNG без ROS и графических библиотек. Основная интерактивная проверка выполняется через локальное веб-приложение.

Воспроизводимая предварительная разметка положительного bag:

```powershell
.\.venv\Scripts\python.exe scripts\annotate_positive_bag.py data\extracted\for_hackathon\doubleT_obstacle\doubleT_obstacle_0.db3 --output annotations\doubleT_obstacle.json
```

Исходные материалы помещать в `data/raw/`, результаты — в `artifacts/`. Приватная переписка хранится в `private/`; эти каталоги исключены из Git.
