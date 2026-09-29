# Облачное демо в Яндекс Облаке

Демо: **http://62.84.115.61** — веб-интерфейс на виртуальной машине Yandex Compute Cloud (Container Optimized Image), панель администратора — `/admin` (логин `admin`, пароль задаётся при развёртывании). При первом запуске на сервере создаётся демонстрационная запись с препятствием; свои записи загружаются кнопкой «+ Загрузить».

В облаке недоступны Docker Hub и репозиторий пакетов ROS (`packages.ros.org`), поэтому для демо собирается лёгкий образ только с веб-интерфейсом — [`Dockerfile.web`](Dockerfile.web) на `python:3.11-slim` с зеркала `mirror.gcr.io`. Полный образ с ROS 2 ([`../Dockerfile`](../Dockerfile)) — для локального запуска.

## Как развёрнуто

1. `deploy/yc-setup.sh` — реестр `gabarit`, сервисные аккаунты, статический IP, группа безопасности (порты 80 и 22), ВМ `gabarit-demo` (2 vCPU, 4 ГБ, SSD 50 ГБ): `CORES=2 MEMORY=4 DISK_GB=50 GABARIT_ADMIN_PASSWORD=<пароль> bash deploy/yc-setup.sh`.
2. Сборка на самой ВМ, без SSH и без локального Docker: код (`git archive`) кладётся в приватный бакет Object Storage, ВМ получает временную ссылку и спецификацию с контейнером-сборщиком (`mirror.gcr.io/library/docker:27-cli`), который собирает `Dockerfile.web`, отправляет образ в реестр и пишет ход работы в последовательный порт (`yc compute instance get-serial-port-output --name gabarit-demo`).
3. Спецификация [`docker-compose.yaml`](docker-compose.yaml) с образом, закреплённым по digest, применяется командой `yc compute instance update-container --name gabarit-demo --docker-compose-file <файл>` (агент ВМ реагирует именно на неё, а не на изменение метаданных).

Если с вашей машины SSH и сайт доступны, то же самое делает `deploy/vm-build.sh` (архив по SSH, сборка на ВМ).

Сервисное подключение SourceCraft видит только облака организации, к которой привязан репозиторий, а секреты репозитория в CI хакатона не подставляются, поэтому автоматическая выкатка из CI ([`ci-cloud.yaml`](ci-cloud.yaml)) не включена.

Почему ВМ, а не Serverless Containers: там запрос ограничен 3,5 МБ, а записи проездов — гигабайты, и обработка идёт через долгое WebSocket-соединение.

```
git archive ──► Object Storage (приватный бакет) ──► ВМ gabarit-demo: контейнер-сборщик
                                                    │  docker build -f deploy/Dockerfile.web
                                                    ▼
                          Container Registry ◄── push   ──► сайт :80 (образ по digest)
```

## Работа

- Интерфейс: `http://<IP>`, панель администратора: `http://<IP>/admin` (логин `admin`, пароль из секрета).
- Одновременно работает один пользователь, остальные видят «Сервис занят».
- Данные на ВМ: `/var/lib/gabarit/bags` (скопированные записи), `/var/lib/gabarit/uploads` (загруженные из браузера), `/var/lib/gabarit/artifacts` (журнал оператора, версии параметров); переживают обновления образа.
- SSH: `ssh -i ~/.ssh/gabarit_demo yc-user@<IP>`, логи: `sudo docker logs -f gabarit`.
- Остановить ВМ, когда демо не нужно: `yc compute instance stop gabarit-demo` (диск и IP оплачиваются и при остановленной ВМ), запустить: `yc compute instance start gabarit-demo`.
- MLflow в облаке не устанавливается: оценки пишутся в локальный журнал администратора.
