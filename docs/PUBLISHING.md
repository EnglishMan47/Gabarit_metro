# Отправка изменений в GitHub и SourceCraft

Это один локальный Git-репозиторий с двумя remote:

- `origin`: https://github.com/EnglishMan47/metro-lidar-detection.git
- `sourcecraft`: https://git.sourcecraft.dev/lct-hackaton-2026/case-05-metro-lidar-detection-team-56

Текущая ветка — `main`. Один коммит отправляется в оба репозитория. Команды
ниже предназначены для PowerShell; автоматическая отправка не выполнялась.

```powershell
Set-Location -LiteralPath 'C:\Users\Lev Kalitikov\Desktop\hacatone'
git status --short
git add -- .gitattributes .dockerignore Dockerfile README.md compose.yaml config docker docs gabarit install.ps1 install.sh start_web.cmd start_web.sh ros2 scripts tests web ':(exclude)docs/C1_PLAN.md' ':(exclude)docs/PROJECT_EXECUTION_PLAN.md'
git diff --cached --check
git diff --cached --stat
git commit -m "Prepare finalist release: robust LiDAR detection, GUI Docker and dataset validation"
git push origin main
git push sourcecraft main
```

Перед `commit` убедитесь, что `git diff --cached --check` не сообщает ошибок,
а список файлов соответствует доработке. Два локальных документа планирования
не добавляются этой командой. Датасеты, записи встреч, виртуальное окружение,
артефакты экспериментов, `results` и `.env` исключены; не используйте `git add -f`
для них. В Git идут код, конфигурация, тесты и документация с итоговыми числами.

Если remote ещё не настроен в другой копии проекта, добавьте недостающий:

```powershell
git remote add origin https://github.com/EnglishMan47/metro-lidar-detection.git
git remote add sourcecraft https://git@git.sourcecraft.dev/lct-hackaton-2026/case-05-metro-lidar-detection-team-56.git
```

В текущей копии оба уже настроены, повторять `remote add` не нужно.
Для push потребуются права и авторизация в соответствующем сервисе.
Если сервер отклоняет push из-за новых коммитов, выполните `git fetch origin`
и `git fetch sourcecraft`, сравните историю и согласуйте изменения. Обычный
push не перезаписывает чужую историю; `--force` для этой публикации не нужен.

Проверить отправленные вершины:

```powershell
git rev-parse HEAD
git ls-remote origin refs/heads/main
git ls-remote sourcecraft refs/heads/main
```

Оба серверных SHA должны совпадать с локальным HEAD после успешной отправки.
SourceCraft запускает workflow `check` при push в `main`; нужно дождаться его
результата в CI/CD. Локально перед публикацией проверены Compose, сборка образа
и тесты Windows/ROS; это не заменяет подтверждение завершения серверного CI.
