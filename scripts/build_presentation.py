#!/usr/bin/env python3
"""Build the presentation on the official LCT 2026 template.

Keeps the mandatory template slides 7-11 (title, solution/team, team members,
team history, short solution) and fills content slides built from the
template layouts. Team data are left for the captain where they are unknown.

    python scripts/build_presentation.py --template "<ЛЦТ2026 Шаблон презентации.pptx>" --out artifacts/release/gabarit.pptx
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "artifacts" / "release_v2"
TEAM = "[заполняет капитан]"


def set_lines(shape, lines, bold_first=False, size=None):
    """Replace text keeping the formatting of the first paragraph / run."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    r0 = p0.runs[0] if p0.runs else None
    ppr = copy.deepcopy(p0._p.pPr) if p0._p.pPr is not None else None
    rpr = copy.deepcopy(r0._r.rPr) if r0 is not None and r0._r.rPr is not None else None
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    for i, line in enumerate(lines):
        p = p0 if i == 0 else tf.add_paragraph()
        if i and ppr is not None:
            if p._p.pPr is not None:
                p._p.remove(p._p.pPr)
            p._p.insert(0, copy.deepcopy(ppr))
        run = p.add_run()
        run.text = line
        if rpr is not None:
            run._r.insert(0, copy.deepcopy(rpr))
        if size:
            run.font.size = Pt(size)
        if bold_first and i == 0:
            run.font.bold = True


def ph(slide, idx):
    for s in slide.placeholders:
        if s.placeholder_format.idx == idx:
            return s
    raise KeyError(idx)


def by_text(slide, start):
    for s in slide.shapes:
        if s.has_text_frame and s.text_frame.text.strip().startswith(start):
            return s
    raise KeyError(start)


def fill(shape, text, size=None):
    tf = shape.text_frame
    lines = text if isinstance(text, list) else [text]
    tf.text = lines[0]
    for line in lines[1:]:
        tf.add_paragraph().text = line
    if size:
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.size = Pt(size)


def plural(n, one: str, few: str, many: str) -> str:
    """Russian plural form: 1 событие, 4 события, 13 событий."""
    if not isinstance(n, int):
        return many
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def load_numbers() -> dict:
    n = {"fp_events": "—", "fp_minutes": "—", "clean_frames": "—", "p95": "—", "person_med": "—", "person_max": "—",
         "child_med": "—", "box_med": "—", "trolley_med": "—", "synthetic": {}}
    syn = ROOT / "artifacts" / "synthetic_final" / "summary.json"
    if syn.exists():
        s = json.loads(syn.read_text(encoding="utf-8"))
        n["synthetic"] = s
        r0 = lambda v: int(round(v)) if v is not None else "—"
        n["person_med"] = r0(s["person"]["confirm_distance_median_m"])
        n["person_max"] = r0(s["person"]["confirm_distance_max_m"])
        n["child_med"] = r0(s["child"]["confirm_distance_median_m"])
        n["box_med"] = r0(s["box_50cm"]["confirm_distance_median_m"])
        n["trolley_med"] = r0(s["trolley"]["confirm_distance_median_m"])
    ev = ROOT / "artifacts" / "eval_final" / "summary.json"
    if ev.exists():
        r = json.loads(ev.read_text(encoding="utf-8"))
        clean = [b for b in r["bags"] if "obstacle" not in b["bag"]]
        n["fp_events"] = sum(b["event_count"] for b in clean)
        n["clean_frames"] = sum(b["frames"] for b in clean)
        n["fp_minutes"] = str(round(sum(b["frames"] for b in clean) / 600.0, 1)).replace(".", ",")
        n["p95"] = round(max(b["total_p95_ms"] for b in r["bags"]), 0)
    tm = ROOT / "artifacts" / "timing" / "summary.json"   # clean timing run (no other load)
    if tm.exists():
        r = json.loads(tm.read_text(encoding="utf-8"))
        n["p95"] = int(round(max(b["total_p95_ms"] for b in r["bags"])))
        n["p95_typical"] = int(round(min(b["total_p95_ms"] for b in r["bags"])))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "release" / "gabarit-presentation.pptx")
    args = ap.parse_args()
    N = load_numbers()
    prs = Presentation(args.template)
    S = {i + 1: s for i, s in enumerate(prs.slides)}

    # ---- 7: title
    s = S[7]
    fill(ph(s, 0), ["ГАБАРИТ"])
    fill(ph(s, 12), "Обнаружение препятствий перед беспилотным поездом метро по данным 3D-лидара · задача №5")

    # ---- 8: solution + team
    s = S[8]
    fill(ph(s, 0), ["ГАБАРИТ"])
    set_lines(by_text(s, "В чем суть"), [
        "Система «видит» путь впереди поезда по 3D-лидару: сама находит рельсы, продолжает путь по стенам тоннеля "
        "до 150–200 м и выдаёт «путь свободен на X м» или «препятствие, Y м» в ROS 2 и в интерфейс оператора."])
    set_lines(by_text(s, "Что делает"), [
        "Не учится на внешнем виде объектов: ищет всё, что нарушает свободный габарит. Самокалибровка лидара, "
        "скорость по лидару, испытание синтетическими препятствиями прямо в интерфейсе."])
    ph(s, 10).insert_picture(str(IMG / "ui-target.png"))
    set_lines(by_text(s, "Капитан"), [f"Капитан: {TEAM}", f"Кол-во участников: {TEAM}", f"Краткое описание: {TEAM}",
                                      f"Город и регион: {TEAM}"])

    # ---- 10: team history (technical part is known)
    s = S[10]
    set_lines(ph(s, 27), [TEAM])
    set_lines(by_text(s, "Что вас вдохновило"), [
        "Задача без размеченных данных и с «пустым» тоннелем: интересно решить её инженерно — описать нормальный "
        "тоннель и находить всё, что в него не вписывается."])
    set_lines(by_text(s, "Расскажите о самых"), [
        "Главная трудность — ложные тревоги на кривых, у платформ и стрелок: стены попадали в габарит. Решили "
        "слежением за стенами фильтрами Калмана и правилами «нормального тоннеля». Второе — нет скорости поезда: "
        "восстановили её по текстуре тоннеля."])

    # ---- 11: short solution
    s = S[11]
    set_lines(ph(s, 38), [
        "ROS 2 Humble узел в Docker: PointCloud2 → /gabarit/obstacle, /distance, /status, маркеры RViz2",
        "Самокалибровка лидара по рельсам (колея 1520 мм)",
        "Путь до 150–200 м: рельсы + фильтры Калмана по стенам, полу, своду",
        "Габарит 2,2 × 3,0 м, отбраковка «нормального тоннеля», подтверждение во времени",
        "Отдельный детектор предметов на рельсе",
        f"Задержка p95 ≈ {N['p95']} мс, только CPU и NumPy",
    ], size=13)
    set_lines(ph(s, 42), [
        "Модуль безопасности для беспилотных поездов метро и МЦД; может работать как «второй машинист» и на пилотируемых составах",
        "Интерфейс оператора: подтверждение тревог, журнал, разбор проезда",
        "Испытательный полигон: синтетические препятствия в реальных записях для приёмки и регрессии",
        "Развитие: слияние с камерой и картой пути, учёт маршрута на стрелках, GPU-ускорение",
    ], size=13)

    # ---- 24: problem / idea / result
    s = S[24]
    fill(ph(s, 0), "ПРОБЛЕМА → ИДЕЯ → РЕЗУЛЬТАТ")
    fill(ph(s, 26), ["Проблема", "Тоннель однороден, препятствий в данных почти нет, разметки нет. Классификатор "
                     "«человек/не человек» не обучить, а ложная тревога останавливает поезд."], size=12)
    fill(ph(s, 31), ["Идея", "Не распознавать объекты, а знать, где путь: габарит свободен по построению. "
                     "Всё, что внутри и не является тоннелем, — препятствие."], size=12)
    fill(ph(s, 32), ["Результат", f"Человек подтверждается в среднем на {N['person_med']} м (до {N['person_max']} м), "
                     f"ребёнок и тележка ~{N['child_med']} м, предмет на рельсе — 56 м в реальных данных."], size=12)

    # ---- 25: pipeline stages
    s = S[25]
    fill(ph(s, 0), "КАК ЭТО РАБОТАЕТ")
    stages = [
        ("Самокалибровка", "Рельсы — два пика высоты на расстоянии колеи; по ним высота, тангаж, крен лидара"),
        ("Путь впереди", "Рельсы до ~50 м, дальше фильтры Калмана по стенам, полу и своду + сглаживание"),
        ("Эго-движение", "Скорость по корреляции текстуры тоннеля между кадрами"),
        ("Габарит", "Кластеры внутри 2,2 × 3,0 м; отбраковка стен, колонн, края платформы"),
        ("Подтверждение", "3 из 4 кадров вблизи, 6 из 8 вдали; запрос оператору"),
    ]
    for (title, text), (a, b) in zip(stages, [(26, 27), (28, 29), (30, 31), (32, 33), (34, 35)]):
        fill(ph(s, a), title, size=14)
        fill(ph(s, b), text, size=11)

    # ---- 14: interface picture
    s = S[14]
    fill(ph(s, 0), "ИНТЕРФЕЙС")
    fill(ph(s, 14), [
        "3D-сцена тоннеля: облако, габарит, ось пути, отметки дальности",
        "«Путь свободен на X м» / «Препятствие Y м», скорость, тормозной путь, время до препятствия",
        "Подтверждение тревоги оператором и журнал решений",
        "Испытание: поставить виртуального человека на путь и увидеть, с какого расстояния он найден",
        "Анализ всего проезда на таймлайне",
    ], size=13)
    ph(s, 10).insert_picture(str(IMG / "ui-synthetic-far.png"))

    # ---- 21: statistics with chart
    s = S[21]
    fill(ph(s, 0), "РЕЗУЛЬТАТЫ")
    chart = next(sh for sh in s.shapes if sh.has_chart).chart
    cd = CategoryChartData()
    cd.categories = ["Человек", "Ребёнок", "Тележка", "Ящик 0,5 м"]
    cd.add_series("Медиана, м", [N["person_med"] or 0, N["child_med"] or 0, N["trolley_med"] or 0, N["box_med"] or 0])
    syn = N["synthetic"]
    cd.add_series("Максимум, м", [round(syn.get(k, {}).get("confirm_distance_max_m") or 0)
                                  for k in ("person", "child", "trolley", "box_50cm")])
    chart.replace_data(cd)
    chart.has_title = True
    chart.chart_title.text_frame.text = "Дистанция подтверждения, м (синтетика в реальных проездах)"
    fill(ph(s, 21), "100%", size=24)
    fill(ph(s, 18), "обнаружение испытательных объектов (64 сближения, 4 типа)", size=12)
    fill(ph(s, 22), f"{N['fp_events']} {plural(N['fp_events'], 'событие', 'события', 'событий')}", size=24)
    fill(ph(s, 23), f"ложных тревог на {N['clean_frames']} кадрах ({N['fp_minutes']} мин) пяти чистых проездов; "
                    "10 из них — стоянка у станции за стрелкой (было 36)", size=12)
    fill(ph(s, 24), f"{N.get('p95_typical', N['p95'])}–{N['p95']} мс", size=24)
    fill(ph(s, 25), "задержка p95 на кадр (1 ядро CPU); в Docker 10 Гц без пропусков", size=12)

    # ---- 27: two pictures (real and synthetic)
    s = S[27]
    fill(ph(s, 0), "НАЙДЕНО НА РЕАЛЬНЫХ ДАННЫХ")
    ph(s, 14).insert_picture(str(IMG / "ui-obstacle.png"))
    ph(s, 18).insert_picture(str(IMG / "ui-rail-object.png"))
    fill(ph(s, 15), "Человек переходит путь: 55,6 м, подтверждён за 0,2 с после входа в габарит; оператор подтверждает тревогу", size=12)
    fill(ph(s, 16), "Предмет на рельсе высотой ~0,25 м: 56,4 м — найден отдельным детектором выступов над головкой рельса", size=12)

    for grp in [sh for sh in s.shapes if sh.shape_type == 6]:   # browser mock-ups: address bar
        for sub in grp.shapes:
            if sub.has_text_frame and "lider.com" in sub.text_frame.text:
                set_lines(sub, ["localhost:8000"])

    # ---- 17: engineering
    s = S[17]
    fill(ph(s, 0), "ГОТОВО К ПЕРЕДАЧЕ ИНЖЕНЕРУ МЕТРО")
    cols = [
        ("ROS 2", ["docker build → docker run → ros2 bag play", "топики /gabarit/*, маркеры RViz2", "CycloneDDS для облаков 8–24 МБ"]),
        ("Интерфейс", ["работает без интернета", "live из ROS 2 или прямое чтение bag", "подтверждение, журнал, испытания"]),
        ("Качество", ["тесты на синтетическом тоннеле", "пакетная оценка и оценка дальности", "все параметры в YAML"]),
    ]
    for (num, head, body), (i1, i2, i3) in zip([("01", *cols[0]), ("02", *cols[1]), ("03", *cols[2])],
                                               [(49, 37, 38), (50, 39, 40), (51, 41, 42)]):
        fill(ph(s, i1), num, size=20)
        fill(ph(s, i2), head, size=16)
        fill(ph(s, i3), body, size=12)

    # ---- 15: what did not work
    s = S[15]
    fill(ph(s, 0), "ЧТО НЕ СРАБОТАЛО")
    items = [
        "Прямой коридор по оси лидара → стены на кривых в габарите (36 ложных событий у прежней версии)",
        "Воксельная модель «нормальной сцены» из штатных проездов → запоминает места, не обобщается",
        "Нейросеть BEV на одном проезде с препятствием → переобучение, оставили геометрию",
        "Однократная подгонка пути полиномом → зигзаги; заменили фильтрами Калмана + сглаживанием",
        "Многопоточный BLAS → решение малых систем в 100 раз медленнее; зафиксирован 1 поток",
    ]
    for idx, text in zip([15, 16, 17, 18, 19], items):
        fill(ph(s, idx), text, size=12)

    # ---- 20: next steps / conclusion
    s = S[20]
    fill(ph(s, 0), "ИТОГ")
    fill(ph(s, 14), [
        "Поезд видит путь: «свободно на X м» или «препятствие, Y м».",
        f"Человек — в среднем за {N['person_med']} м, предмет на рельсе — 56 м.",
        "Без размеченных данных и без привязки к месту: только инварианты тоннеля.",
    ], size=14)
    nxt = ["Учёт маршрута на стрелках по данным поезда", "Слияние с камерой для классификации",
           "Карта пути для дальних кривых", "Ускорение на GPU / C++", "Испытания на новых проездах заказчика"]
    for idx, text in zip([15, 16, 17, 18, 19], nxt):
        fill(ph(s, idx), text, size=13)

    # ---- order and cleanup: keep 7-11, then content slides
    order = [7, 8, 9, 10, 11, 24, 25, 14, 21, 27, 17, 15, 20]
    sld = prs.slides._sldIdLst
    ids = list(sld)
    keep = [ids[i - 1] for i in order]
    for el in ids:
        sld.remove(el)
        if el not in keep:
            # drop the relationship too: unreachable slides/media are not saved
            prs.part.drop_rel(el.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
    for el in keep:
        sld.append(el)
    # static slide-number text boxes of the template -> real position
    for pos, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if not sh.is_placeholder and sh.name.startswith("Номер слайда") and sh.has_text_frame:
                set_lines(sh, [str(pos)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
