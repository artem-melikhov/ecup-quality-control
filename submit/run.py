# -*- coding: utf-8 -*-
"""Контейнерное решение задачи «Контроль качества» E-CUP 2026.

БАД: TF-IDF с NB-взвешиванием и логистической регрессией.
ЛВ: два текстовых классификатора с логистическим объединением.
Готовые каналы объединяются стеком с учётом доступного времени.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
_SH = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
GEMMA = [os.path.join(_SH, "google/gemma-4-E4B-it"), "google/gemma-4-E4B-it"]
QWEN = [os.path.join(_SH, "Qwen/Qwen3.5-4B"), "Qwen/Qwen3.5-4B"]
DL_GEM, DL_QWEN = 9 * 60, 16 * 60   # дедлайн gemma-стадии и Qwen-стадии; лимит паблика 20 мин
ATTR_KEYS = ["комплектац", "состав", "топлив", "тип баллона", "баллон", "расход газа", "пьезоподжиг",
             "характеристик", "применение", "назначение", "особенност", "материал", "в комплекте", "входит"]
RULES = [
    ("Пиротехника и дым", r"пиротех|цветной дым|дымовая шашка|дымов|хлопушк|фейерверк|салют|бенгальск|страйкбол|граната"),
    ("Розжиг и твёрдое топливо", r"розжиг|сухое горючее|сухого горючего|брикет|роллы|растопк|топливо"),
    ("Уголь", r"уголь|угли |углём|углем"), ("Спички", r"спичк|спичеч"), ("Свечи", r"свеч"),
    ("Благовония/аромапалочки", r"благовон|шалфей|пало санто|аромапалочк"),
    ("Зажигалки и бензин", r"зажигалк|бензин"),
    ("Газ: горелки/баллоны/плиты", r"газов|баллон|пропан|бутан|горелк|плитк|плита|резак|паяльн"),
    ("Мангалы/грили/аксессуары", r"мангал|гриль|барбекю|шампур|казан|коптил|опахал|решетк|решётк"),
    ("Краски Холи", r"холи|краск"),
]
COMMENTS = {
    ("БАД", 1): ("Товар распознан как корректно размещённая биологически активная добавка: название и "
                 "описание содержат характерные признаки БАД, карточка соответствует требованиям категории."),
    ("БАД", 0): ("Карточка в категории БАД не подтверждена как корректная: по названию и описанию товар "
                 "ближе к спортивному питанию или продукции без обязательной маркировки БАД."),
    ("Легковоспламеняющиеся", 1): ("Товар отнесён к легковоспламеняющимся: по названию и описанию он содержит "
                                   "горючее вещество, газ, топливо или пиротехнический заряд либо горючее "
                                   "входит в комплект поставки."),
    ("Легковоспламеняющиеся", 0): ("Признаков легковоспламеняющегося товара не найдено: это устройство или "
                                   "изделие без горючего содержимого в комплекте, горючий материал здесь "
                                   "не является самостоятельным товаром."),
}
SYS_RET = ("Ты воспроизводишь историческую разметку маркетплейса: пометил бы разметчик этот товар как "
           "легковоспламеняющийся? Общий принцип: решает то, что физически есть в самом товаре или в его "
           "комплекте (горючее вещество, газ, топливо, пиротехнический заряд, спички), а не то, для чего "
           "товар предназначен: пустые устройства для огня размечают как НЕ легковоспламеняющиеся. "
           "Ниже даны примеры разметки похожих товаров — ориентируйся прежде всего на них, "
           "они отражают реальную конвенцию. Ответ одним словом: да или нет.")


def subcat(name, desc):
    t = str(name).lower()
    for nm, pat in RULES:
        if re.search(pat, t):
            return nm
    t2 = str(desc)[:600].lower()
    for nm, pat in RULES:
        if re.search(pat, t2):
            return nm
    return "Прочее"


def attr_field(h):
    t = re.sub(r"<[^>]+>", "\n", str(h)).lower()
    return " ".join(t[m.start():m.start() + 140] for k in ATTR_KEYS for m in re.finditer(re.escape(k), t))


def tfidf_blocks(tr_df, te_df, ytr, cols_cfg, C=4.0):
    Xs_tr, Xs_te = [], []
    for col, wr, cr, mdf, mf in cols_cfg:
        wv = TfidfVectorizer(ngram_range=wr, min_df=mdf[0], max_features=mf[0], sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cr, min_df=mdf[1], max_features=mf[1], sublinear_tf=True)
        Xs_tr.append(sparse.hstack([wv.fit_transform(tr_df[col]), cv.fit_transform(tr_df[col])]).tocsr())
        Xs_te.append(sparse.hstack([wv.transform(te_df[col]), cv.transform(te_df[col])]).tocsr())
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)
    m.fit(sparse.hstack(Xs_tr).tocsr(), ytr)
    return m.predict_proba(sparse.hstack(Xs_te).tocsr())[:, 1]


def tfidf_blocks_nb(tr_df, te_df, ytr, cols_cfg, C=4.0):
    """Обучение TF-IDF + логрег с NB-весами, рассчитанными только на трейне."""
    Xs_tr, Xs_te = [], []
    for col, wr, cr, mdf, mf in cols_cfg:
        wv = TfidfVectorizer(ngram_range=wr, min_df=mdf[0], max_features=mf[0], sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cr, min_df=mdf[1], max_features=mf[1], sublinear_tf=True)
        Xs_tr.append(sparse.hstack([wv.fit_transform(tr_df[col]), cv.fit_transform(tr_df[col])]).tocsr())
        Xs_te.append(sparse.hstack([wv.transform(te_df[col]), cv.transform(te_df[col])]).tocsr())
    Xtr = sparse.hstack(Xs_tr).tocsr(); Xte = sparse.hstack(Xs_te).tocsr()
    yy = np.asarray(ytr)
    pc = np.asarray(Xtr[yy == 1].sum(0)).ravel() + 1.0
    nc = np.asarray(Xtr[yy == 0].sum(0)).ravel() + 1.0
    r = np.abs(np.log((pc / pc.sum()) / (nc / nc.sum())))
    r = np.maximum(r, 1e-6); r = r / r.mean()          # нормировка: не двигаем эффективное C
    R = sparse.diags(r)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)
    m.fit(Xtr @ R, yy)
    return m.predict_proba(Xte @ R)[:, 1]


CFG3 = [("f_txt", (1, 2), (3, 5), (2, 3), (200_000, 300_000)),
        ("f_attr", (1, 3), (3, 5), (2, 2), (150_000, 200_000)),
        ("f_nm", (1, 3), (2, 4), (2, 2), (100_000, 150_000))]
CFG1 = [("f_txt", (1, 2), (3, 5), (2, 3), (200_000, 300_000))]
BAD_RATE = 578.0 / 921.0
STACK_PRIORITY = ("all7", "no_emb6", "no_emb_no_few", "core4", "gem_only", "text_only")


def log_odds(probabilities):
    """Перевести вероятности в лог-оддсы с защитой от нуля и единицы."""
    p = np.clip(np.asarray(probabilities, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prepare_features(df):
    """Добавить текстовые поля, используемые классификаторами и поиском соседей."""
    df["desc_raw"] = df["description"].fillna("").astype(str)
    df["desc"] = df["desc_raw"].str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True)
    df["f_txt"] = (df["name"].astype(str) + " \n " + df["desc_raw"]).str.lower()
    df["f_attr"] = [attr_field(x) for x in df["desc_raw"]]
    df["f_nm"] = df["name"].astype(str).str.lower()


def select_bad_labels(probabilities):
    """Выбрать верхнюю фиксированную долю карточек БАД по вероятности."""
    k_take = int(round(BAD_RATE * len(probabilities)))
    labels = np.zeros(len(probabilities), dtype=int)
    labels[np.argsort(-probabilities)[:k_take]] = 1
    return labels


def combine_signals(signals, params):
    """Использовать первый стек, для которого полностью рассчитаны все каналы."""
    have = set(signals)
    for name in STACK_PRIORITY:
        cfg = params[name]
        if set(cfg["cols"]) <= have:
            z = cfg["intercept"] + sum(
                c * log_odds(signals[col]) for c, col in zip(np.array(cfg["coef"]), cfg["cols"])
            )
            return 1 / (1 + np.exp(-z)), name
    raise ValueError("Нет коэффициентов для доступных каналов")


def select_lv_labels(probabilities):
    """Выбрать размер положительного класса по максимуму ожидаемого F1."""
    order = np.argsort(-probabilities)
    counts = np.arange(1, len(probabilities) + 1)
    expected_tp = np.cumsum(probabilities[order])
    k = int(np.argmax(2 * expected_tp / (counts + probabilities.sum())))
    labels = np.zeros(len(probabilities), dtype=int)
    labels[order[:k + 1]] = 1
    return labels


def build_submission(test, predictions):
    """Сформировать id/result с конкурсными тегами и шаблонами комментариев."""
    results = [
        f"<комментарий>{COMMENTS[(str(test.loc[i, 'category']), int(predictions.loc[i]))]}"
        f"<вердикт>{'не бан' if predictions.loc[i] == 1 else 'бан'}"
        for i in test.index
    ]
    return pd.DataFrame({"id": test["id"].values, "result": results})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data_path", "--test-data-path", "-i", dest="test_data_path")
    ap.add_argument("--output_path", "--output-path", "-o", dest="output_path")
    a = ap.parse_args()
    t0 = time.time(); log = print
    np.random.seed(SEED)
    P = json.load(open(os.path.join(HERE, "params.json"), encoding="utf-8"))
    train = pd.read_csv(os.path.join(HERE, "train_data.csv"), index_col=0)
    test = pd.read_csv(a.test_data_path)
    log(f"train {len(train)}, test {len(test)}")
    for df in (train, test):
        prepare_features(df)
    preds = pd.Series(0, index=test.index, dtype=int)

    m_bad = (test["category"] == "БАД").values
    if m_bad.sum():
        trb = train[train["category"] == "БАД"]
        p = tfidf_blocks_nb(trb, test.loc[m_bad], trb["label"].values, CFG1, C=4.0)
        log(f"[{time.time()-t0:.0f}s] БАД: NB-взвешенная модель обучена на {len(trb)}")
        sel = select_bad_labels(p)
        preds[m_bad] = sel
        log(f"[{time.time()-t0:.0f}s] БАД: {len(p)} карточек, взято {int(sel.sum())} (доля {BAD_RATE})")
        log(f"[{time.time()-t0:.0f}s] БАД: {int(m_bad.sum())}, позитивов {int(preds[m_bad].sum())}")

    m_lv = (test["category"] == "Легковоспламеняющиеся").values
    if m_lv.sum():
        sub = test[m_lv].reset_index(drop=True)
        trl = train[train["category"] == "Легковоспламеняющиеся"]
        S = {"attr": tfidf_blocks(trl, sub, trl["label"].values, CFG3),
             "tfidf": tfidf_blocks(trl, sub, trl["label"].values, CFG1)}
        log(f"[{time.time()-t0:.0f}s] текстовые сигналы готовы")


        p, stack_name = combine_signals(S, P)
        log(f"стек «{stack_name}»: {P[stack_name]['cols']}")
        lp_ = select_lv_labels(p)
        preds[m_lv] = lp_
        log(f"[{time.time()-t0:.0f}s] ЛВ: DTA взял {lp_.sum()}/{len(lp_)} ({lp_.mean():.2%})")

    norm_key = lambda n, d: re.sub(r"\s+", " ", (str(n) + " " + str(d)).lower()).strip()
    dup = train.groupby([norm_key(n, d) for n, d in zip(train["name"], train["desc_raw"])])["label"].mean()
    n_over = 0
    for i in test.index:
        k2 = norm_key(test.loc[i, "name"], test.loc[i, "desc_raw"])
        if k2 in dup.index:
            m2 = dup.loc[k2]
            if m2 > 0.5 and preds.loc[i] != 1: preds.loc[i] = 1; n_over += 1
            elif m2 < 0.5 and preds.loc[i] != 0: preds.loc[i] = 0; n_over += 1
    log(f"дубль-оверрайдов: {n_over}")
    submission = build_submission(test, preds)
    submission.to_csv(a.output_path, index=False)
    log(f"готово за {time.time()-t0:.0f}s: {len(submission)} строк")


if __name__ == "__main__":
    main()
