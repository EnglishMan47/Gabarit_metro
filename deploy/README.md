# Облачное демо в Яндекс Облаке

Веб-интерфейс работает на виртуальной машине Yandex Compute Cloud (Container Optimized Image). Workflow `deploy-demo` (`deploy/ci-cloud.yaml`, после настройки облака переносится в `.sourcecraft/ci.yaml`) собирает Docker-образ при каждом push в `main`, отправляет его в Container Registry, и ВМ перезапускает контейнер с новым образом.

Проверка и запуск решения в CI без облака описаны в разделе «CI/CD в SourceCraft» [README](../README.md).

Почему ВМ, а не Serverless Containers: там запрос ограничен 3,5 МБ, а записи проездов — гигабайты, и обработка идёт через долгое WebSocket-соединение.

```
push в main ──► SourceCraft CI ──► Container Registry ──► ВМ gabarit-demo :80
                (сборка образа)     cr.yandex/<id>/gabarit   веб-интерфейс, админка,
                                                              /var/lib/gabarit (записи, журнал)
```

## Однократная настройка

1. Аккаунт в Яндекс Облаке с платёжным аккаунтом (грант нового пользователя или грант SourceCraft) и каталог.
2. Yandex Cloud CLI: [установка](https://yandex.cloud/ru/docs/cli/quickstart), затем `yc init` (вход, облако, каталог, зона `ru-central1-a`).
3. Из корня проекта в Git Bash (пароль — латиница и цифры):

   ```bash
   GABARIT_ADMIN_PASSWORD=<пароль> bash deploy/yc-setup.sh
   ```

   Скрипт создаёт реестр `gabarit`, сервисные аккаунты `gabarit-ci` (CI: push образов, обновление ВМ) и `gabarit-vm` (ВМ: загрузка образов), статический IP, группу безопасности (порты 80 и 22) и ВМ `gabarit-demo` (4 vCPU, 8 ГБ, SSD 100 ГБ), а также вписывает id реестра в `deploy/ci-cloud.yaml`. Повторный запуск безопасен.
4. SourceCraft → настройки репозитория:
   - **Сервисные подключения**: создать `gabarit-demo` — каталог из вывода скрипта, сервисный аккаунт `gabarit-ci`;
   - **Секреты**: `GABARIT_ADMIN_PASSWORD` — тот же пароль.
5. Перенести workflow `deploy-demo` и блок `tokens` из `deploy/ci-cloud.yaml` в `.sourcecraft/ci.yaml`, закоммитить и отправить: `git push sourcecraft main`. Ход сборки — в разделе CI/CD репозитория.
6. Записи проездов:

   ```bash
   bash deploy/upload-bag.sh data/extracted/for_hackathon/doubleT_obstacle
   ```

   или кнопкой «+ Загрузить» в интерфейсе.

## Работа

- Интерфейс: `http://<IP>`, панель администратора: `http://<IP>/admin` (логин `admin`, пароль из секрета).
- Одновременно работает один пользователь, остальные видят «Сервис занят».
- Данные на ВМ: `/var/lib/gabarit/bags` (скопированные записи), `/var/lib/gabarit/uploads` (загруженные из браузера), `/var/lib/gabarit/artifacts` (журнал оператора, версии параметров); переживают обновления образа.
- SSH: `ssh -i ~/.ssh/gabarit_demo yc-user@<IP>`, логи: `sudo docker logs -f gabarit`.
- Остановить ВМ, когда демо не нужно: `yc compute instance stop gabarit-demo` (диск и IP оплачиваются и при остановленной ВМ), запустить: `yc compute instance start gabarit-demo`.
- MLflow в облаке не устанавливается: оценки пишутся в локальный журнал администратора.
