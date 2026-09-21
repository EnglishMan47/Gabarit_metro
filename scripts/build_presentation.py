"""Create an editable content draft; the official organizer template is required for submission."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'artifacts/evaluation/release-summary.json').read_text(encoding='utf-8'))
    overall = report['overall']
    negatives = sum(b['event_count'] for n,b in report['bags'].items() if n != 'doubleT_obstacle')
    slides = [
        ('Metro Lidar Control', 'ЛЦТ 2026 · задача №5\nЛокальное веб-приложение обнаружения препятствий\nСодержательный черновик: нужен официальный шаблон.'),
        ('Задача', 'Обнаружить посторонний объект внутри габарита пути.\nОпределить расстояние до ближайшей видимой поверхности.\nПоказать результат оператору в браузере.'),
        ('Предоставленные данные', '6 записей · 2 488 кадров · около 250 секунд\n307 200 или 921 600 исходных точек на кадр\nПоложительная разметка предварительная, подготовлена командой.'),
        ('Архитектура', 'SQLite bag → CDR → нормализация координат\nПол и путь → кластеры → фон → временные треки\nHTTP JSON → браузер → журнал и экспорт'),
        ('Геометрия', 'Высота над локальным полом вместо постоянного Z.\nКонсервативная оценка рельсов и резервный центр пути.\nРазмер вокселя и минимальное число точек зависят от дальности.'),
        ('Модель нормальной сцены', 'Занятость вокселей на штатных записях.\nПоложительная запись не участвует в построении фона.\nЕдиные параметры для всех записей; нейросеть не используется.'),
        ('Результаты — предварительная разметка', 'Основной человек внутри габарита: 140/140\nКандидат предмета: 201/201\np95 ошибки центра: около 0,40 м\nЭто внутренняя проверка, не независимый тест.'),
        ('Скорость и ресурсы', f"Все {overall['processed_frames']} кадров обработаны.\nДетектор p95: {overall['detector_p95_ms']:.1f} мс\nВычислительный кадр p95: {overall['total_p95_ms']:.1f} мс\nПик памяти процесса: {overall['peak_process_memory_mb']:.1f} МБ"),
        ('Интерфейс', 'Выбор записи и кадра, воспроизведение.\nДве проекции, bbox, расстояние и устойчивые треки.\nПодтверждение оператором и экспорт JSON/CSV.\nАвтономный запуск после установки NumPy.'),
        ('Ограничения и проверка обобщения', f'Штатные записи: {negatives} устойчивых тревожных интервалов.\nОтложенный roundT_doubleT: тревога в 202/252 кадрах.\nВнутренняя цель по ложным тревогам не достигнута.\nТребуется улучшение отделения инфраструктуры от препятствий.'),
        ('Комплект и следующие действия', 'Исходники, конфигурация, фон, тесты и воспроизводимая оценка.\nИнструкция запуска, архитектура и сценарий демонстрации.\nПеред сдачей: официальный шаблон, видео интерфейса, устранение открытых критериев качества.'),
    ]
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    for index, (title, body) in enumerate(slides, 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string('091714')
        for text, top, size, color in ((title, .75, 32, '57E6A1'), (body, 2.0, 23, 'EDF8F4'), (f'{index:02}  |  Черновик содержания · 20.09.2026', 6.9, 12, '91AAA2')):
            box = slide.shapes.add_textbox(Inches(.8), Inches(top), Inches(11.7), Inches(4.7 if top == 2.0 else .6))
            for line_index, line in enumerate(text.splitlines()):
                paragraph = box.text_frame.paragraphs[0] if line_index == 0 else box.text_frame.add_paragraph()
                paragraph.text = line
                paragraph.font.name = 'Arial'
                paragraph.font.size = Pt(size)
                paragraph.font.color.rgb = RGBColor.from_string(color)
                paragraph.space_after = Pt(18)
    output = ROOT / 'artifacts/release/presentation-content-draft.pptx'
    output.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(output)
    print(output)


if __name__ == '__main__':
    main()
