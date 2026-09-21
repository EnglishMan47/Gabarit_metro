"""Fill the required slides of the organizers' PowerPoint template.

The downloaded template stays in artifacts/ and is not redistributed in Git.
Team identity fields are intentionally left blank until the captain supplies them.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "artifacts" / "release" / "official-template"
OUTPUT = ROOT / "artifacts" / "release" / "metro-lidar-official-template-draft.pptx"
PURPLE = RGBColor(72, 12, 113)
DARK = RGBColor(28, 29, 34)
WHITE = RGBColor(255, 255, 255)


def set_text(shape, content: str, *, size: int, color: RGBColor = DARK, bold: bool = False) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_top = 0
    frame.margin_bottom = 0
    for index, line in enumerate(content.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.space_after = Pt(9)
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color


def add_cover_text(slide, text: str, *, x: float, y: float, width: float, height: float, size: int, bold: bool) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    set_text(shape, text, size=size, color=WHITE, bold=bold)


def main() -> None:
    matches = list(TEMPLATE_DIR.rglob("*.pptx"))
    if len(matches) != 1:
        raise SystemExit("Ожидается один официальный шаблон PPTX в artifacts/release/official-template")
    deck = Presentation(matches[0])
    if len(deck.slides) < 11:
        raise SystemExit("В шаблоне нет обязательных слайдов 7–11")

    # Remove only the explanatory and design-example slides. Keep the five
    # required slides with their layouts, graphics, logos and slide numbers.
    for index in reversed(range(len(deck.slides))):
        if index not in range(6, 11):
            slide_id = deck.slides._sldIdLst[index]
            deck.part.drop_rel(slide_id.rId)
            deck.slides._sldIdLst.remove(slide_id)

    cover, summary, team, story, solution = deck.slides
    add_cover_text(cover, "КОНТРОЛЬ ПУТИ\nПО ЛИДАРУ", x=0.65, y=0.56, width=5.4, height=1.4, size=32, bold=True)
    add_cover_text(cover, "Обнаружение препятствий в тоннеле метро\nЛокальное веб-приложение", x=0.65, y=2.65, width=8.5, height=1.2, size=25, bold=False)
    add_cover_text(cover, "ЛЦТ 2026 · задача №5", x=0.65, y=6.30, width=5.4, height=0.5, size=17, bold=True)

    set_text(summary.shapes[2], "Краткое описание решения", size=19, color=PURPLE, bold=True)
    set_text(summary.shapes[11], "Приложение читает облака точек из SQLite bag, показывает путь в 3D и сообщает расстояние до устойчивого препятствия внутри габарита.", size=16)
    set_text(summary.shapes[10], "Уникальность решения", size=19, color=PURPLE, bold=True)
    set_text(summary.shapes[3], "Прямое чтение bag без Docker и ROS; локальные пол и путь, временные треки, диагностическая BEV-модель на GPU или CPU.", size=16)
    set_text(summary.shapes[5], "О команде", size=19, color=PURPLE, bold=True)
    set_text(summary.shapes[4], "", size=16)

    set_text(team.shapes[20], "КОМАНДА", size=21, color=WHITE, bold=True)
    for index in (1, 2, 4, 5, 7, 8, 10, 11, 21, 22):
        set_text(team.shapes[index], "", size=16)

    set_text(story.shapes[2], "ИСТОРИЯ КОМАНДЫ", size=21, color=WHITE, bold=True)
    for index in (3, 5, 7):
        set_text(story.shapes[index], "", size=16)
    set_text(story.shapes[4], "Краткая история команды", size=18, color=PURPLE, bold=True)
    set_text(story.shapes[8], "Почему выбрали эту задачу", size=18, color=PURPLE, bold=True)
    set_text(story.shapes[6], "Какие сложности преодолели", size=18, color=PURPLE, bold=True)

    set_text(solution.shapes[4],
             "SQLite bag → CDR → очистка облака точек.\n"
             "Локальный пол, габарит пути, кластеры и модель штатной сцены.\n"
             "Трекер подтверждает объект за три последовательных кадра.\n"
             "BEV-сегментация ONNX работает как диагностика; тревогу определяет геометрия.\n"
             "2 488 кадров проверено; 36 ложных событий на штатных записях остаются ограничением.",
             size=18)
    set_text(solution.shapes[5],
             "Оператор видит 3D-облако, профиль, боксы и расстояние; решения сохраняются в журнал.\n"
             "Приложение запускается локально без внешнего API и может работать с GPU или CPU.\n"
             "Для внедрения нужны новые размеченные проезды, снижение ложных тревог и испытания на независимых данных.",
             size=19)
    set_text(solution.shapes[6], "КОРОТКО О РЕШЕНИИ", size=22, color=WHITE, bold=True)
    set_text(solution.shapes[7], "Техническая суть решения", size=22, color=PURPLE, bold=True)
    set_text(solution.shapes[8], "Применение и развитие", size=22, color=PURPLE, bold=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    deck.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
