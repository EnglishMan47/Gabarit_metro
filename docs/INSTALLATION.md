# Установка и запуск

Инструкция для текущей реализации на 12.11.2026. Основной режим требует Python 3.10+ и четыре пакета из `requirements.txt`; ROS, Node.js, PyTorch и scikit-learn для обычного просмотра не нужны. Проверки интерфейса используют Node.js, обучение собственного леса — отдельную среду.

## Python

Windows PowerShell из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
.\.venv\Scripts\python.exe -m gabarit check --data data/extracted
.\.venv\Scripts\python.exe -m gabarit web --data data/extracted --port 8000
```

Linux или macOS:

```sh
bash install.sh
.venv/bin/python -m gabarit check --data data/extracted
.venv/bin/python -m gabarit web --data data/extracted --port 8000
```

[Открыть просмотр](http://localhost:8000). `start_web.cmd` и `start_web.sh` запускают тот же интерфейс. Для своих данных передайте `--data <каталог>`; флаг разрешено повторять. Интернет нужен при установке зависимостей и получении необязательных весов, после этого просмотр работает локально. Внешние bag-файлы не скачиваются установщиком.

## Проверка без внешних данных

В активированной Python-среде:

```sh
python scripts/make_sample_bag.py results/smoke/input/sample
python scripts/evaluate.py results/smoke/input --out results/smoke/evaluation
```

Генератор создаёт 30 кадров. Ожидается одно событие препятствия: объект в коридоре примерно на 45 м; боковой объект игнорируется. В `results/smoke/evaluation/summary.json` проверьте `frames: 30` и `event_count: 1` у единственной записи.

## Профили датчиков

| Облако | Конфигурация | Оси и предел анализа |
|---|---|---|
| Предоставленные Pandar bag | `config/gabarit.yaml` | вперёд `-y`, вверх `+z`, 200 м |
| Falcon в системе ROS | `config/falcon_prime_ros.yaml` | вперёд `+x`, вверх `+z`, 350 м |
| Falcon в исходных осях | `config/falcon_prime.yaml` | вперёд `+z`, вверх `+x`, 350 м |

Профиль выбирают по координатам конкретной записи, а не только по названию лидара. Для `falcon_test_second_dark` проверен Falcon ROS; `new_data` использует Pandar. В браузере примените профиль до начала оценки. [Подробнее о датчиках](SENSORS.md).

```sh
python -m gabarit web --data data/extracted/falcon_test --config config/falcon_prime_ros.yaml
```

## Docker и ROS 2

Требуется работающий Docker Engine с Linux-контейнерами. Образ использует ROS 2 Humble, NumPy 1.x для совместимости с ROS, CycloneDDS и программный рендер RViz.

```sh
docker compose config --quiet
docker compose up -d --build
docker compose exec gabarit gabarit bag list
docker compose exec gabarit gabarit bag play /data/extracted/for_hackathon/doubleT_obstacle --cache
```

[RViz](http://localhost:6080/vnc.html?autoconnect=1&resize=scale) и [web](http://localhost:8080). `data` монтируется только для чтения, `results` — в `/output`. Другой каталог данных задаётся `GABARIT_BAGS` в `.env`. Переключение записи сбрасывает историю детектора:

```sh
docker compose exec gabarit gabarit bag switch /data/extracted/falcon_test/falcon_test_second_dark --config /opt/gabarit/config/falcon_prime_ros.yaml
docker compose exec gabarit gabarit bag pause
docker compose exec gabarit gabarit bag rate 0.25
docker compose exec gabarit gabarit bag resume
```

При возврате к Pandar явно передайте `--config /opt/gabarit/config/gabarit.yaml`. Кеш по умолчанию ограничен 16 ГБ и занимает место сверх исходной записи. Для `new_data` около 84 ГБ используйте исходный путь без `--cache` или Linux/WSL ext4. Подробности: [DOCKER_GUI.md](DOCKER_GUI.md).

На хосте Ubuntu 22.04 с ROS 2 Humble: `bash ros2/run_ros2.sh`; в отдельной ROS-консоли — `ros2 bag play <bag>`. Узел обнаруживает PointCloud2-топики автоматически, либо задаётся `input_topic:=/lidar_points`. Для живого лидара требуется общая доступная ROS/DDS-сеть; проброса порта веб-интерфейса для этого недостаточно.

## Дополнительные модели

Собственный лес уже находится в `gabarit/models/component_forest_v1.json`; scikit-learn для его исполнения не требуется:

```sh
python -m gabarit web --data data/extracted --config config/component_trained_fast.yaml
```

Это отдельный Pandar-профиль с двумя наблюдениями при подтверждённой поддержке модели. Базовый профиль не меняется автоматически. Консервативный вариант с тремя наблюдениями — `component_trained.yaml`. [Профили и ограничения](MODELS.md).

Для FarBevNet:

```sh
python scripts/fetch_path_models.py
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-neural.txt
python -m gabarit web --data data/extracted --config config/learned_pandar.yaml
```

Для Falcon ROS используйте `config/learned_falcon_ros.yaml`. Контейнер с CPU-PyTorch запускается через `docker compose -f compose.yaml -f compose.neural.yaml up -d --build`. CUDA-вариант добавляет `-f compose.cuda.yaml` и требует совместимого NVIDIA Container Toolkit. Проверка YAML не подтверждает доступность GPU; измеренная скорость нейросетевого режима пока не удовлетворяет целевому бюджету на всех кадрах. Веса не входят в Git. Условия распространения Attention Please не уточнены: см. [происхождение файлов](../THIRD_PARTY_NOTICES.md).

## Тесты и диагностика

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q tests
```

Для JavaScript-проверок установите Node.js 18+; CI GitHub использует 22. Проверка оригинального C++ леса требует компилятора C++; необязательный checkpoint-тест — PyTorch и скачанных весов. Результат 12.11: 419 passed, 1 skipped. В ограниченной Windows-среде тестам нужны доступный временный каталог и локальные сокеты asyncio.

```sh
docker compose logs --tail 80
docker compose exec gabarit gabarit bag log
docker compose exec gabarit gabarit bag status
docker compose down
```

Пустое облако: проверьте наличие всех частей bag, выбранные оси и топик. Сбой сборки: проверьте Docker Engine, место на диске и установку зависимостей. Задержки: смотрите ядро, callback ROS и частоту источника отдельно. [Текущие результаты](VALIDATION_2026_11_12.md).
