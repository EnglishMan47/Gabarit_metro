# Сохранение и отправка изменений

Локальная ветка `main`, GitHub remote `origin`: [EnglishMan47/metro-lidar-detection](https://github.com/EnglishMan47/metro-lidar-detection). Дополнительный remote `sourcecraft` уже настроен. Команды выполняет владелец репозитория; автоматическая отправка в этом обновлении не выполнялась.

## Коммит доработок кода

```powershell
Set-Location -LiteralPath 'C:\Users\Lev Kalitikov\Desktop\hacatone'
git status --short
git add -- .gitattributes .sourcecraft/ci.yaml .github Dockerfile compose.yaml compose.cuda.yaml docker/workspace.py gabarit config scripts tests ros2 web requirements-training.txt training
git diff --cached --check
git diff --cached --stat
git commit -m "Improve detection, realtime processing and regression checks"
git push origin main
```

`training` содержит экспорт признаков и отчёты собственного леса, необходимые для повторного обучения без bag. Веса собственного леса находятся в `gabarit/models`; чужой checkpoint в `.models` не добавляется. Если этот коммит уже создан, повторять его не нужно.

## Коммит документации и каркаса статьи

```powershell
git add -- README.md deploy/README.md docs/INSTALLATION.md docs/USER_GUIDE.md docs/WEB_UI.md docs/SENSORS.md docs/ARCHITECTURE.md docs/ALGORITHM.md docs/MODELS.md docs/DATA_AND_EVALUATION.md docs/VALIDATION_2026_11_12.md docs/validation docs/RESULT_FORMAT.md docs/RELATED_WORK.md docs/PUBLISHING.md docs/EXPERIMENTS.md docs/DOCKER_GUI.md docs/EVALUATION_FINALIST.md docs/FINALIST_UPDATE.md docs/publication
git diff --cached --check
git diff --cached --stat
git commit -m "Refresh project documentation and add publication draft"
git push origin main
```

Список намеренно задаёт конкретные материалы документации. Исследовательские планы С1/С1–С4, отдельный прежний DOCX анализа конкурентов и произвольные локальные изображения этой командой не добавляются. Документация использует существующие отслеживаемые снимки `docs/screenshots/ui_20261107`.

Проверьте полный список уже staged-файлов перед коммитом: `git add` не убирает файлы, добавленные ранее другой командой. При отсутствии изменений Git сообщит `nothing to commit`; тогда остаётся отправить существующий коммит.

## Второй remote и сверка

```powershell
git push sourcecraft main
git rev-parse HEAD
git ls-remote origin refs/heads/main
git ls-remote sourcecraft refs/heads/main
```

После успешной отправки соответствующий серверный SHA должен совпадать с HEAD. Если push отклонён из-за новой удалённой истории, выполните `git fetch origin`, затем `git log --oneline --left-right main...origin/main` и согласуйте изменения. `--force` для обычной публикации не нужен.

Не добавляйте `data`, `private`, `.env`, окружения и большие локальные артефакты через `git add -f`. Файлы Attention Please уже присутствуют в проекте с указанием источника, но лицензия зафиксированной версии не объявлена. Коммит и push сами по себе не устраняют этот открытый вопрос распространения.

После push проверьте GitHub Actions и SourceCraft CI. Наличие YAML и локальных успешных тестов не означает, что удалённый CI уже завершён. Затем проверьте установку и запуск нового выпуска из чистого клона.
