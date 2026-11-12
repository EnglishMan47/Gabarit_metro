Актуализация 12.11.2026: текущие результаты и ограничения приведены в [VALIDATION_2026_11_12.md](VALIDATION_2026_11_12.md), модели — в [MODELS.md](MODELS.md), команды оценки — в [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md). Числа прежних версий ниже сохраняются как история и не описывают текущую версию.

# Docker: ROS 2, RViz и переключение записей

Краткая инструкция установки с нуля и таблица профилей новых записей:
[INSTALLATION.md](INSTALLATION.md). После обновления кода пересоберите образ
через `docker compose up -d --build`.

Контейнер содержит ROS 2 Humble, RViz2, SQLite3/MCAP и поддержку zstd. Основной режим `workspace` оставляет детектор и RViz запущенными, а записи выбираются отдельными командами. Браузерный рабочий стол через noVNC работает без внешнего X-сервера; он подходит для Linux и Windows Docker Desktop в режиме Linux containers.

## Запуск

В корне репозитория:

```sh
docker compose build
docker compose up -d
docker compose logs -f
```

По умолчанию в `/data` доступна папка `data` проекта. Для другой папки создайте рядом с `compose.yaml` файл `.env`, например:

```dotenv
GABARIT_BAGS=C:/Users/Lev Kalitikov/Desktop/new_dataset
```

На Linux укажите абсолютный Linux-путь. Папка должна существовать; в ней могут находиться несколько записей в подпапках. Для Windows используйте `/` в пути. Распакуйте архивы заранее. Папка монтируется только для чтения.

После запуска откройте:

- [RViz в браузере](http://localhost:6080/vnc.html?autoconnect=1&resize=scale).
- [Веб-интерфейс детектора](http://localhost:8080).

GUI публикуется только на `127.0.0.1`. Для удалённого сервера используйте SSH-туннель. Порты можно изменить в `.env`: `GABARIT_GUI_PORT=6081`, `GABARIT_WEB_PORT=8081`.

## Выбор, пауза и скорость без перезапуска контейнера

```sh
docker compose exec gabarit gabarit bag list
docker compose exec gabarit gabarit bag play /data/first_recording
docker compose exec gabarit gabarit bag pause
docker compose exec gabarit gabarit bag next
docker compose exec gabarit gabarit bag rate 0.25
docker compose exec gabarit gabarit bag resume
docker compose exec gabarit gabarit bag status
docker compose exec gabarit gabarit bag switch /data/second_recording --rate 0.5
docker compose exec gabarit gabarit bag stop
docker compose exec gabarit gabarit bag log
```

`next` воспроизводит одно ROS-сообщение при паузе: оно не обязательно является облаком точек. Скорость меняется через штатный сервис rosbag2 и не требует повторного чтения записи с начала. При завершении записи контейнер продолжает работать. `status` показывает состояние процесса, паузу, скорость, имя записи и путь к журналу; `log` выводит последние 16 000 символов журнала.

Для запуска сразу на паузе:

```sh
docker compose exec gabarit gabarit bag play /data/recording --paused --rate 0.25
```

Можно передать параметры штатного проигрывателя после `--`:

```sh
docker compose exec gabarit gabarit bag play /data/recording -- --loop --start-offset 10
```

`bag list` распознаёт каталоги с `metadata.yaml`, отдельные `.db3` и `.mcap`. Для разделённой записи с `metadata.yaml` выбирайте её каталог. Если одинаковое короткое имя есть в нескольких папках, задайте полный путь внутри `/data`.

Папки, содержащие только резервную копию metadata без данных, не включаются
в список и не создают ложную неоднозначность коротких имён.

Переключение останавливает всю группу процессов предыдущего проигрывателя. Новый rosbag2 запускается на паузе; затем `/gabarit/reset` сбрасывает калибровку, историю пути, трекер препятствий и маркеры. Только после успешного сброса воспроизведение продолжается. Ошибка сброса останавливает новый проигрыватель. Состояние старого датасета не должно попадать в новый; повтор записи также обрабатывает сброс времени в основном pipeline.

## Выбор профиля лидара

По умолчанию используется `config/gabarit.yaml` для предоставленных ранее записей Pandar. Для Falcon с координатами производителя (`z` вперёд, `x` вверх):

```sh
docker compose exec gabarit gabarit bag switch /data/falcon_recording --config /opt/gabarit/config/falcon_prime.yaml
```

Если драйвер уже перевёл Falcon в ROS-координаты (`x` вперёд, `z` вверх), используйте `falcon_prime_ros.yaml`. **Именно этот вариант проверен в предоставленном `falcon_test_second_dark`.** Убедитесь в осях облака: модель датчика сама по себе не определяет преобразования, выполненные драйвером. Профиль можно менять при каждом `play/switch`; для возврата к Pandar передайте `--config /opt/gabarit/config/gabarit.yaml`. Если параметр отсутствует, сохраняется текущий профиль.

Чтобы выбрать стартовый профиль для всего контейнера, добавьте в `.env`:

```dotenv
GABARIT_CONFIG_FILE=falcon_prime.yaml
```

В RViz облако `/gabarit/view_cloud` и габарит `/gabarit/view_markers` имеют общие координаты `gabarit_view`: `x` вперёд, `y` влево, `z` вверх; ноль высоты — плоскость головок рельсов. Смена имени исходного топика или frame не требует перенастройки RViz. Для графики фон прорежен до бюджета 30 тысяч точек, а точки кандидатов и препятствий сохраняются все; алгоритм обрабатывает полное рабочее облако. Обычные `/gabarit/markers` и `/gabarit/obstacle_points` сохраняют исходную систему координат датчика для интеграции с другими ROS-узлами.

## Штатный ros2 bag play из консоли

Консоль с подготовленным ROS-окружением:

```sh
docker compose exec gabarit gabarit shell
```

Внутри контейнера:

```sh
gabarit bag stop
ros2 service call /gabarit/reset std_srvs/srv/Trigger '{}'
ros2 bag play /data/recording --clock 100 --rate 0.5 --read-ahead-queue-size 5
```

Штатное управление с клавиатуры: пробел — пауза/продолжение, стрелки вверх/вниз — изменение скорости, вправо — следующее сообщение при паузе, `Ctrl+C` — остановка. Альтернатива: `gabarit play /data/recording --clock 100 --rate 0.5`. Ручной проигрыватель не принадлежит менеджеру `gabarit bag`; остановите его `Ctrl+C` перед запуском другой записи. Для автоматического сброса и переключения используйте команды `gabarit bag`.

Из второй ROS-консоли можно вызвать штатные сервисы:

```sh
ros2 service call /rosbag2_player/pause rosbag2_interfaces/srv/Pause '{}'
ros2 service call /rosbag2_player/set_rate rosbag2_interfaces/srv/SetRate '{rate: 0.25}'
ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume '{}'
```

Сервисы и параметры сверены с [исходным кодом rosbag2 для Humble](https://github.com/ros2/rosbag2/tree/humble/rosbag2_interfaces/srv) и [CLI Humble](https://github.com/ros2/rosbag2/blob/humble/ros2bag/ros2bag/verb/play.py).

## Linux: отдельное окно X11

Браузерный режим используется по умолчанию. Для локального X11 можно передать Xauthority cookie без глобального открытия X-сервера:

```sh
touch /tmp/gabarit.xauth
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f /tmp/gabarit.xauth nmerge -
docker compose -f compose.yaml -f docker/compose.x11.yaml up -d
```

Требуется установленный `xauth` и доступный X11/XWayland `DISPLAY`. При другом пути cookie задайте `GABARIT_XAUTHORITY`. Для остановки используйте те же два `-f`. В режиме X11 порт 6080 не обслуживается; веб-интерфейс детектора остаётся на 8080. Windows Desktop использует браузерный режим.

## Ресурсы, диагностика и остановка

SQLite через bind mount папки Windows может читать большие облака намного
медленнее алгоритма. Менеджер автоматически передаёт SQLite-проигрывателю
`config/rosbag_read.yaml`: mmap до 2 ГБ и кеш страниц 64 МБ. Пользовательский
`--storage-config-file` после `--` имеет приоритет. Формат настройки проверен
по [исходному коду rosbag2 Humble](https://github.com/ros2/rosbag2/blob/humble/rosbag2_storage_default_plugins/src/rosbag2_storage_default_plugins/sqlite/sqlite_storage.cpp).

Для плотных записей на Docker Desktop используйте:

```sh
docker compose exec gabarit gabarit bag play /data/extracted/for_hackathon/doubleT_obstacle --cache
```

`--cache` копирует только выбранную запись в Linux-том `bag-cache`, затем
запускает обычный rosbag2. Копия повторно используется при следующих запусках,
если пути, размеры и время изменения исходных файлов совпадают. Исходные данные
остаются доступными только для чтения. Завершённые кеши сохраняются после
`docker compose down`; они занимают дополнительное место размером с запись.
Перед копированием проверяется свободное место Linux-тома с резервом 512 МБ.
Общий бюджет кеша по умолчанию — 16 ГБ; его можно изменить через
`GABARIT_CACHE_MAX_GB` в `.env`, предварительно проверив свободное место на
диске хоста (виртуальный диск Docker может показывать больший доступный объём).
Копирование
выполняется до старта проигрывателя и может занять время; RViz и контейнер
остаются работающими. Для `new_data` копия потребует около 84 ГБ: при нехватке
места используйте исходный путь или разместите данные на Linux/WSL ext4.

Частоту проигрывателя проверяйте отдельно от `timing_ms.total`: сообщение
`Message queue starved` указывает на задержки чтения, даже если детектор
укладывается в 100 мс.

Параметры по умолчанию: 1 ГБ `/dev/shm`, программный OpenGL (`LIBGL_ALWAYS_SOFTWARE=1`), очередь чтения rosbag2 — 5 сообщений. Это ограничивает расход памяти на больших облаках. Для оценки частоты алгоритма отделяйте вычисление pipeline от вывода GUI; на слабом CPU программный рендеринг может конкурировать с детектором. Для запуска без GUI задайте `GABARIT_GUI=off` в `.env`.

```sh
docker compose logs --tail 100
docker compose exec gabarit gabarit bag log
docker compose exec gabarit gabarit shell
# внутри ROS-консоли:
ros2 topic echo /gabarit/status --once
ros2 topic hz /gabarit/status
```

При отсутствии облаков проверьте `bag log`, выбранный профиль и наличие `PointCloud2` в `ros2 bag info /data/recording`. Автообнаружение топиков работает и на паузе независимо от `/clock`. Для live-лидара используйте `gabarit detect` (без sim time); основная Compose-конфигурация рассчитана на проигрыватель и детектор в одном контейнере.

Входной QoS автоматически учитывает профиль издателя: для надёжного потока используется Reliable, для Best Effort — совместимая подписка Best Effort. При смене записи с другим QoS узел переподписывается. Для принудительного выбора задайте в `.env` `GABARIT_INPUT_QOS=reliable` или `GABARIT_INPUT_QOS=best_effort`; обычное значение — `auto`. Совместимость основана на [правилах QoS Humble](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst).

```sh
docker compose down
```

Supervisor посылает сигналы всей группе каждого дочернего процесса и ограничивает ожидание завершения. Compose использует init для уборки завершённых detached-процессов и даёт 45 секунд на корректную остановку. Без Compose используйте `docker run --init --stop-timeout 45 ...`.

## Проверки и пределы проверки

Изолированные регрессионные тесты `tests/test_docker_playback.py` и `tests/test_ros_visualization.py` выполняются без ROS: они проверяют последовательность paused/reset/resume, ошибки переключения, защиту от повторного использования PID, очистку процессов, соответствие осей и сохранение маркеров габарита. `docker compose config --quiet` проверяет конфигурацию Compose.

На машине доработки 19.10 собран образ и пройдено **206 тестов** внутри Linux/ROS Humble
(NumPy 1.26). Проверены рабочий стол noVNC, отрисовка облака и габарита в RViz,
выходные ROS-топики, пауза/продолжение, изменение скорости, шаг и переключение
Pandar → Falcon → Pandar с заменой конфигурации без перезапуска контейнера,
включая новую запись `synthetic_switch_cloud_with_markers`. Актуальные результаты —
в [отчёте 19.10](DETECTION_VALIDATION_2026_10_19.md); прежние измерения
частоты и скриншоты — в [отчёте 18.10](FINALIST_UPDATE.md).
Повторите проверку на машине показа: скорость диска, DDS и программный
рендеринг зависят от окружения.
