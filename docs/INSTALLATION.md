# Установка и запуск — обновление 07.11.2026

Для показа экспертам используйте Docker: в одном образе находятся детектор,
ROS 2 Humble, RViz2 и рабочий стол в браузере. Локальная установка Python
даёт веб-просмотр и пакетную обработку SQLite bag без ROS.

В проверке от 19.10 устойчивые 10 Гц на машине доработки не были достигнуты:
измерено около 5,9 сообщения/с в ROS с RViz и 7–9 кадров/с для ядра в двух
изолированных Windows-прогонах. Для плавного показа используйте `--rate 0.5`;
это режим просмотра, а не подтверждение производительности 10 Гц.
Подробные условия и ограничения — в [отчёте проверки](DETECTION_VALIDATION_2026_10_19.md).

После ускорения от 07.11 контрольный запуск ROS + веб + RViz получил около
8,3 сообщения/с на первых 100 кадрах Pandar; параллельно выполнялся пакетный
нейросетевой прогон. Это рабочая проверка запуска, выполнение 10 Гц на всех
записях пока не подтверждено.

## Docker Desktop / Linux

Нужны Git, Docker Engine с Compose v2 либо Docker Desktop в режиме Linux
containers. На Windows включите WSL 2 и запустите Docker Desktop. Проверьте:

```sh
docker version
docker compose version
```

Из корня клонированного репозитория:

```sh
docker compose up -d --build
docker compose exec -T gabarit gabarit test
docker compose exec gabarit gabarit bag list
```

Откройте [RViz/noVNC](http://localhost:6080/vnc.html?autoconnect=1&resize=scale)
и [детектор](http://localhost:8080). В RViz видны облако, габарит и препятствия.
Данные в контейнере доступны только для чтения. Результаты из `/output`
попадают в локальную папку `results`.

После обновления исходников обязательно повторите `docker compose up -d --build`:
образ содержит копию Python-кода, а папка `config` подключается отдельно.
Изменение файлов на хосте само по себе не обновляет код работающего образа.
Файл `.gitattributes` сохраняет Linux-скрипты с окончаниями строк LF, в том
числе при клонировании на Windows; его также нужно включить в публикацию.

## Куда положить записи

Распакованные bag-каталоги складывайте в `data/extracted`. Сохраняйте вместе
`metadata.yaml` и все перечисленные в нём `.db3`. Архивы можно хранить в
`data/source`; ни архивы, ни облака, ни локальные результаты в Git не входят.
Новый набор с маркерами расположен в `data/extracted/finalist_markers_20261019`.

Чтобы подключить другую папку, создайте `.env` рядом с `compose.yaml`:

```dotenv
GABARIT_BAGS=D:/datasets/metro
```

Папка должна существовать. На Windows в `.env` используйте прямые `/`.
После изменения подключения выполните `docker compose up -d`.

| Записи | Профиль в контейнере |
| --- | --- |
| `for_hackathon/*`, `new_data`, `cloud_with_fake_obj`, `synthetic_*` обоих новых архивов | `/opt/gabarit/config/gabarit.yaml` |
| `falcon_test_second_dark` | `/opt/gabarit/config/falcon_prime_ros.yaml` |
| Falcon в исходных осях производителя, +z вперёд, +x вверх | `/opt/gabarit/config/falcon_prime.yaml` |

`new_data` имеет оси Pandar; слово «новый» не означает профиль Falcon.
Веб-просмотр без ROS читает SQLite `.db3`; MCAP воспроизводится штатным
rosbag2 внутри Docker. Топики разметки не подаются на вход детектору.

## Управление показом

Веб-просмотр поддерживает выбор цели нажатием на карточку объекта, свободную
камеру мышью и сброс камеры кнопкой ↺ или клавишей `R`. Во вкладке «Вид»
доступны чистое облако, основные слои и отдельная дополнительная ось.
Внизу можно переходить к событиям проезда. Пошаговый просмотр переводит
проигрыватель на паузу.

Зелёным показана свежая часть габарита с надёжной осью; янтарным — неуверенная
или удерживаемая часть, серым — прогноз. При разрыве связи последний кадр
помечается устаревшим. «Сопровождение, %» на карточке оценивает устойчивость
трека, а не вероятность распознанного класса. Для Falcon выберите профиль
датчика во вкладке «Детектор» и нажмите «Применить».

```sh
docker compose exec gabarit gabarit bag play /data/extracted/finalist_markers_20261019/synthetic_switch_cloud_with_markers --config /opt/gabarit/config/gabarit.yaml --paused --rate 0.25
docker compose exec gabarit gabarit bag next
docker compose exec gabarit gabarit bag resume
docker compose exec gabarit gabarit bag pause
docker compose exec gabarit gabarit bag rate 0.5
docker compose exec gabarit gabarit bag switch /data/extracted/falcon_test/falcon_test_second_dark --config /opt/gabarit/config/falcon_prime_ros.yaml --paused
docker compose exec gabarit gabarit bag status
docker compose exec gabarit gabarit bag stop
```

При возврате с Falcon на Pandar передавайте `--config .../gabarit.yaml` явно:
профиль сохраняется после переключения. Контейнер и RViz перезапускать не нужно.
`next` выдаёт одно ROS-сообщение. В записи с разметкой облако и MarkerArray —
разные сообщения, поэтому один шаг не всегда меняет облако.

Для просмотра исходной разметки на паузе в RViz смените `Fixed Frame` на
`hesai_lidar`, отключите отображения `Lidar` и `Envelope and obstacles`
(они используют `gabarit_view`), добавьте `PointCloud2` → `/lidar_points` и два
`MarkerArray` → `/synthetic_obstacle_markers` и `/gabarit/markers`. Так исходная
разметка и результат детектора показываются в одинаковых координатах датчика.
Стандартный вид с выровненным габаритом восстанавливается повторным открытием
конфигурации `/opt/gabarit/ros_ws/install/gabarit_ros/share/gabarit_ros/rviz/gabarit.rviz`.

Для медленного диска можно добавить `--cache`. Бюджет кеша по умолчанию
16 ГБ; требуется дополнительное свободное место размером с выбранную запись.
Для `new_data` размером около 84 ГБ используйте исходный путь без `--cache`
или Linux/WSL ext4. Не увеличивайте бюджет кеша без достаточного места на хосте.

Обработка автоматически подтверждает препятствие; управление торможением,
тормозной путь и запрос подтверждения у машиниста удалены. Тревога не ставит
воспроизведение на паузу. Пауза и скорость относятся к просмотру записи.

## Без Docker

Нужен Python 3.10+ с `venv` и доступ к PyPI на время установки. Проверенная
локальная среда — Python 3.12; контейнер использует Python 3.10 и NumPy 1.26.

Windows, PowerShell из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
.\start_web.cmd
.\.venv\Scripts\python.exe -m gabarit check --data data/extracted
```

Linux / macOS:

```sh
bash install.sh
bash start_web.sh
.venv/bin/python -m gabarit check --data data/extracted
```

При отсутствии модуля `venv` в Ubuntu установите `python3-venv`. Установщики
останавливаются при ошибке зависимостей или проверки окружения. Для тестов
отдельно установите `requirements-dev.txt`; для рабочего запуска достаточно
`requirements.txt`. Откройте [веб-просмотр](http://localhost:8000). Выберите
запись и профиль датчика, примените настройки и начните воспроизведение.

## Проверка и диагностика

### Дополнительные модели

Обычные команды установки и `docker compose up -d --build` используют CPU.
Компактный лес Sakhalin Ai поставляется с исходниками и включён в режиме
`models.component_policy: score`: оценки доступны в JSON, решения геометрии
сохраняются. Для явного включения положительной помощи используйте
`config/component_assist.yaml` (оси Pandar). Этот профиль проходит те же
геометрические и временные проверки; он требует отдельной оценки на ваших
записях. Для Falcon сохраняйте его профиль датчика и добавляйте раздел `models`.

Для дополнительной ветки Attention Please сначала получите проверенные веса:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_path_models.py
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-neural.txt
.\.venv\Scripts\python.exe -m gabarit web --data data/extracted --config config/learned_pandar.yaml
```

Для записи `falcon_test_second_dark` используйте
`config/learned_falcon_ros.yaml`. Параметры обоих профилей можно менять через
редактор конфигурации веб-интерфейса. Для CUDA установите подходящую сборку
PyTorch из официального индекса вместо CPU-сборки.

Контейнер с CPU-сборкой PyTorch, RViz и подключёнными весами:

```sh
docker compose -f compose.yaml -f compose.neural.yaml config --quiet
docker compose -f compose.yaml -f compose.neural.yaml up -d --build
docker compose -f compose.yaml -f compose.neural.yaml exec gabarit gabarit bag list
```

Для Falcon задайте `GABARIT_CONFIG_FILE=learned_falcon_ros.yaml` в `.env`
перед запуском либо передайте `--config /opt/gabarit/config/learned_falcon_ros.yaml`
команде `gabarit bag play`. Нейросетевая ветка используется дополнительно;
геометрическая ветка продолжает сопровождение. Без PyTorch или весов она
отключается с причиной в `models.far_path`, основной детектор продолжает работу.
Измеренный CPU-inference сети занимает около 130–190 мс на один запуск;
поэтому этот профиль не подтверждает обработку каждого кадра за 100 мс.

Возврат к стандартному образу: `docker compose up -d --build`.
Происхождение перенесённых файлов и лицензии сохранены в
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

### Проверка контейнера

```sh
docker compose config --quiet
docker compose logs --tail 80
docker compose exec gabarit gabarit bag log
docker compose exec gabarit gabarit shell
```

В ROS-консоли доступны обычные `ros2 bag info`, `ros2 bag play`, `ros2 topic
echo /gabarit/status --once`. Дополнительные способы запуска, X11 и управление
штатным проигрывателем описаны в [DOCKER_GUI.md](DOCKER_GUI.md).

Если контейнер не запускается, проверьте работающий Docker Engine, свободное
место и порты 6080/8080. Для других портов задайте `GABARIT_GUI_PORT` и
`GABARIT_WEB_PORT` в `.env`. При пустом облаке проверьте `bag log`, наличие
всех частей записи, топик PointCloud2 и профиль координат.

Остановка: `docker compose down`. Данные и результаты сохраняются.
Текущие результаты и ограничения — в [отчёте проверки](DETECTION_VALIDATION_2026_10_19.md),
команды отправки кода — в [PUBLISHING.md](PUBLISHING.md).
