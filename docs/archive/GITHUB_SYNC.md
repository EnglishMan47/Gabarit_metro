# Создание локального Git-репозитория и синхронизация с GitHub

Инструкция подготовлена для папки `C:\Users\Lev Kalitikov\Desktop\hacatone` и PowerShell. Сейчас эта папка ещё не является Git-репозиторием. Git установлен, GitHub CLI (`gh`) отсутствует, поэтому ниже используется обычный Git и HTTPS.

## 1. Что хранить в GitHub

В репозиторий следует отправлять исходный код, конфигурации, документацию, тесты и небольшие воспроизводимые результаты. Не следует отправлять исходный датасет, приватный экспорт Telegram, локальные окружения, пароли и большие записи Q&A.

Текущий `.gitignore` уже исключает:

- `data/`, `датасет/` и `датасет.zip`;
- `private/` с экспортом Telegram;
- `artifacts/` с локальными результатами;
- `.venv/`, `.models/`, кэши Python и `.env`;
- исходное видео `Город 5. Департамент транспорта.mp4`.

Файл `5. ДепТранспорта.pdf` весит около 2,7 МБ и технически может храниться в репозитории. Перед публикацией нужно убедиться, что правила организаторов разрешают распространять ТЗ. Если нет, добавить его точное имя в `.gitignore`.

GitHub не принимает обычным Git отдельные файлы больше 100 МиБ. Для обязательных крупных моделей или демонстрационного видео можно отдельно рассмотреть Git LFS, но датасет объёмом несколько гигабайт лучше хранить во внешнем хранилище, а в репозитории оставить инструкцию получения и контрольные суммы.

## 2. Создать пустой репозиторий на GitHub

На GitHub:

1. Нажать **New repository**.
2. Указать владельца и имя, например `metro-lidar-obstacle-detection`.
3. Выбрать `Private`, если публикация материалов соревнования не разрешена явно.
4. Не добавлять README, `.gitignore` и лицензию: эти файлы уже существуют локально.
5. Нажать **Create repository** и скопировать HTTPS-адрес вида `https://github.com/OWNER/REPO.git`.

Пустой удалённый репозиторий — самый простой вариант для существующей локальной папки: не потребуется объединять независимые истории.

## 3. Настроить имя и почту автора

Проверить текущие значения:

```powershell
git config --global user.name
git config --global user.email
```

Если значения пустые или неверные, задать их:

```powershell
git config --global user.name "ИМЯ ФАМИЛИЯ"
git config --global user.email "EMAIL_ИЗ_GITHUB"
```

Вместо личной почты можно использовать GitHub `noreply`-адрес из настроек аккаунта.

## 4. Создать локальный репозиторий и первый коммит

В PowerShell:

```powershell
cd "C:\Users\Lev Kalitikov\Desktop\hacatone"
git init -b main
git status --short --ignored
git add .
git status --short
git diff --cached --stat
git diff --cached
git commit -m "Initialize metro lidar project"
```

Перед `commit` обязательно проверить `git status --short` и `git diff --cached`. В списке не должно быть датасета, видео, `.env`, файлов из `private/`, `.venv/` или `.models/`.

Если ненужный файл только подготовлен к коммиту:

```powershell
git restore --staged -- "ПУТЬ_К_ФАЙЛУ"
```

После этого добавить путь в `.gitignore`, если он всегда должен оставаться локальным.

## 5. Связать локальный и удалённый репозитории

Подставить настоящий HTTPS-адрес:

```powershell
git remote add origin https://github.com/OWNER/REPO.git
git remote -v
git push -u origin main
```

При первом `push` Git Credential Manager обычно откроет вход через браузер. Пароль аккаунта GitHub для HTTPS не используется; нужно войти через браузер либо применить personal access token, если этого потребует настройка организации.

Ключ `-u` связывает локальную ветку `main` с `origin/main`. После первого раза достаточно команд `git pull` и `git push`.

## 6. Ежедневная синхронизация

Перед началом работы:

```powershell
cd "C:\Users\Lev Kalitikov\Desktop\hacatone"
git status
git pull --rebase origin main
```

После законченного логического изменения:

```powershell
git status --short
git add README.md docs scripts
git diff --cached
git commit -m "Describe completed change"
git push
```

Предпочтительно добавлять конкретные файлы или каталоги, а не без проверки выполнять `git add .`. Коммит должен описывать один законченный результат, например:

```powershell
git commit -m "Add PointCloud2 bag decoder"
git commit -m "Document dataset coordinate ranges"
git commit -m "Implement geometric obstacle baseline"
```

Команда `git pull --rebase` сначала получает изменения команды и затем переносит поверх них локальные коммиты. Это сохраняет прямую историю без лишнего merge-коммита.

## 7. Работа через отдельную ветку

Для крупной функции:

```powershell
git switch main
git pull --rebase origin main
git switch -c feature/geometric-baseline
```

После разработки:

```powershell
git add scripts docs README.md
git commit -m "Implement geometric obstacle baseline"
git push -u origin feature/geometric-baseline
```

Затем на GitHub создать Pull Request из `feature/geometric-baseline` в `main`. После слияния:

```powershell
git switch main
git pull --rebase origin main
git branch -d feature/geometric-baseline
```

Удалить удалённую ветку можно через интерфейс Pull Request или командой:

```powershell
git push origin --delete feature/geometric-baseline
```

## 8. Разрешение конфликтов при `pull --rebase`

Если Git сообщает о конфликте:

```powershell
git status
```

Открыть перечисленные файлы, оставить правильное содержимое и удалить маркеры `<<<<<<<`, `=======`, `>>>>>>>`. Затем:

```powershell
git add "ПУТЬ_К_ИСПРАВЛЕННОМУ_ФАЙЛУ"
git rebase --continue
```

Повторять до завершения. Если объединение начато ошибочно:

```powershell
git rebase --abort
```

Не использовать `git push --force` в общей ветке `main`. Если изменение опубликованной ветки действительно требует перезаписи истории, сначала согласовать это с командой и применять более безопасный `--force-with-lease`.

## 9. Если удалённый репозиторий уже содержит коммиты

Если на GitHub уже есть README, код или коммиты команды, безопаснее не создавать независимую историю в текущей папке:

1. Клонировать репозиторий в соседнюю пустую папку.
2. Скопировать в неё исходники и документацию этого проекта без игнорируемых данных.
3. Проверить изменения, сделать коммит и отправить его.

```powershell
cd "C:\Users\Lev Kalitikov\Desktop"
git clone https://github.com/OWNER/REPO.git hacatone-github
cd hacatone-github
git status
```

После копирования файлов:

```powershell
git status --short
git add .
git diff --cached
git commit -m "Add local project files"
git push
```

Не копировать `.git` из одной папки в другую. Если удалённая история важна, этот способ надёжнее, чем принудительная отправка локальной ветки.

## 10. Контрольная проверка

После первой отправки:

```powershell
git status
git branch -vv
git remote -v
git log --oneline --decorate -5
```

Ожидаемый результат:

- рабочее дерево чистое;
- локальная `main` отслеживает `origin/main`;
- последний локальный коммит виден на странице GitHub;
- датасет, видео, приватная переписка, окружение и секреты отсутствуют в репозитории.

Если секрет уже попал в коммит или был отправлен на GitHub, одного удаления файла недостаточно: секрет нужно сразу отозвать или заменить, а историю очистить отдельно.
