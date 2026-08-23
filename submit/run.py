# -*- coding: utf-8 -*-
"""Контейнерное решение задачи «Контроль качества» E-CUP 2026.

БАД: TF-IDF с NB-взвешиванием и логистической регрессией.
ЛВ: текстовые классификаторы и замороженные LLM с примерами из трейна.
Готовые каналы объединяются стеком с учётом доступного времени.
"""
import argparse
from html.parser import HTMLParser
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
COMMENT_PREFIXES = {
    ("БАД", 1): "По текстовым признакам модель отнесла товар к категории БАД.",
    ("БАД", 0): "По текстовым признакам модель не подтвердила принадлежность товара к БАД.",
    ("Легковоспламеняющиеся", 1): "Модель отнесла товар к легковоспламеняющимся по совокупности текстовых признаков.",
    ("Легковоспламеняющиеся", 0): "По совокупности текстовых признаков модель не отнесла товар к легковоспламеняющимся.",
}
EVIDENCE_PATTERNS = {
    "БАД": re.compile(
        r"\bбад\b|биологически\s+активн\w*|dietary\s+supplement|food\s+supplement|"
        r"витамин\w*|минерал\w*|спортивн\w*\s+питани\w*|протеин\w*", re.I
    ),
    "Легковоспламеняющиеся": re.compile(
        r"пиротех\w*|спич\w*|топлив\w*|горюч\w*|\bгаз\w*|баллон\w*|"
        r"бензин\w*|уголь\w*|угли\b|розжиг\w*|зажигал\w*|свеч\w*", re.I
    ),
}
SYS_RET = ("Ты воспроизводишь историческую разметку маркетплейса: пометил бы разметчик этот товар как "
           "легковоспламеняющийся? Общий принцип: решает то, что физически есть в самом товаре или в его "
           "комплекте (горючее вещество, газ, топливо, пиротехнический заряд, спички), а не то, для чего "
           "товар предназначен: пустые устройства для огня размечают как НЕ легковоспламеняющиеся. "
           "Ниже даны примеры разметки похожих товаров — ориентируйся прежде всего на них, "
           "они отражают реальную конвенцию. Ответ одним словом: да или нет.")


class _CardTextParser(HTMLParser):
    """Извлечь видимый текст, сохранив границы слов на месте HTML-тегов."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden.append(tag)
        if not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    """Очистить разметку и HTML-сущности, затем нормализовать пробелы."""
    if value is None or pd.isna(value):
        return ""
    parser = _CardTextParser()
    parser.feed(str(value))
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def select_demo_indices(indices, similarities, groups, subcategories, target, limit, seen):
    """Сначала соседи той же подкатегории, затем остальные; одна карточка на группу."""
    ranked = np.argsort(-similarities, kind="stable")
    if target != "Прочее":
        same = subcategories[indices[ranked]] == target
        ranked = np.concatenate((ranked[same], ranked[~same]))
    selected = []
    for j in indices[ranked]:
        if len(selected) >= limit:
            break
        if groups[j] in seen:
            continue
        seen.add(groups[j])
        selected.append(int(j))
    return selected


def build_comment(category, label, name, description):
    """Добавить наблюдаемый фрагмент текста, не выдавая упоминание за доказательство."""
    prefix = COMMENT_PREFIXES[(category, int(label))]
    text = clean_text(name) + ". " + clean_text(description)
    # В цитате не должно быть конкурсных тегов или переносов строк.
    text = text.replace("<", "‹").replace(">", "›")
    match = EVIDENCE_PATTERNS[category].search(text)
    if match:
        start = max(0, match.start() - 35)
        end = min(len(text), match.end() + 45, start + 110)
        excerpt = ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")
        detail = f" В тексте карточки: «{excerpt}». Упоминание учитывается в контексте описания."
    else:
        detail = " Отдельные ключевые упоминания не выделены; решение принято по совокупности текстовых признаков."
    return (prefix + detail)[:300]


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
    """Выделить атрибуты из уже очищенного описания."""
    t = str(h).lower()
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
    df["desc"] = df["desc_raw"].map(clean_text)
    df["name_text"] = df["name"].map(clean_text)
    df["f_txt"] = (df["name_text"] + " \n " + df["desc"]).str.lower()
    df["f_attr"] = [attr_field(x) for x in df["desc"]]
    df["f_nm"] = df["name_text"].str.lower()


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
    """Сформировать id/result с конкурсными тегами и текстовыми наблюдениями."""
    results = [
        "<комментарий>" + build_comment(
            str(test.loc[i, "category"]), predictions.loc[i],
            test.loc[i, "name"], test.loc[i, "description"],
        ) +
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

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise RuntimeError("нет GPU")
            os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            # Не повторяем почти-дубли в примерах для LLM.
            gmap = json.load(open(os.path.join(HERE, "lv_groups.json"), encoding="utf-8"))
            tl = trl.copy()
            tl["grp"] = [int(gmap.get(str(int(v)), -1)) for v in tl["id"]]
            if (np.array(tl["grp"]) == -1).all():
                key = lambda n, d: re.sub(r"\s+", " ", (str(n) + " " + str(d)).lower()).strip()
                tl["grp"] = pd.factorize([key(n, d) for n, d in zip(tl["name"], tl["desc"])])[0]
                log("группы из lv_groups.json не сопоставились — дедуплицирую по точному ключу")
            tl = tl.reset_index(drop=True)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  max_features=200_000, sublinear_tf=True)
            Xc = normalize(vec.fit_transform((tl["name_text"] + " " + tl["desc"]).str.lower()))
            Xq = normalize(vec.transform((sub["name_text"] + " " + sub["desc"]).str.lower()))
            pi = np.where(tl["label"].values == 1)[0]; ni = np.where(tl["label"].values == 0)[0]
            sp = (Xq @ Xc[pi].T).toarray(); sn = (Xq @ Xc[ni].T).toarray()
            rng = np.random.default_rng(SEED)

            grpv = tl["grp"].values
            train_subcats = np.array([subcat(n, d) for n, d in zip(tl["name_text"], tl["desc"])])
            subs = [subcat(n, d) for n, d in zip(sub["name_text"], sub["desc"])]

            def demos_prompt(r, kk):
                dm, seen = [], set()
                for idxs, sims in ((pi, sp[r]), (ni, sn[r])):
                    chosen = select_demo_indices(
                        idxs, sims, grpv, train_subcats, subs[r], kk, seen
                    )
                    for j in chosen:
                        dm.append((str(tl.loc[j, "name_text"]), str(tl.loc[j, "desc"])[:180], int(tl.loc[j, "label"])))
                rng.shuffle(dm)
                lines = [f"- {n} || {d} -> {'да' if l else 'нет'}" for n, d, l in dm]
                return ("Примеры разметки похожих товаров:\n" + "\n".join(lines) +
                        f"\n\nТовар: {sub.iloc[r]['name_text']}\nОписание: {str(sub.iloc[r]['desc'])[:400]}\n\nОтвет:")

            def run_model(model_paths, passes, deadline, tag):
                """Одна загрузка модели — несколько проходов. Экономит минуты на загрузке."""
                path = next((p_ for p_ in model_paths if os.path.exists(p_)), model_paths[-1])
                loc = os.path.exists(path)
                tok = AutoTokenizer.from_pretrained(path, local_files_only=loc, trust_remote_code=True)
                tok.padding_side = "left"
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",
                                                           local_files_only=loc, trust_remote_code=True).eval()
                yi = sorted({tok.encode(x, add_special_tokens=False)[0] for x in ["да", "Да", " да", " Да"]})
                no = sorted({tok.encode(x, add_special_tokens=False)[0] for x in ["нет", "Нет", " нет", " Нет"]})
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                MBT, MBS = (60000, 32) if gb > 40 else (6000, 3)
                log(f"[{time.time()-t0:.0f}s] {tag}: модель {path}, батч до {MBT} токенов / {MBS}")
                res = {}
                for pname, sysmsg, userfn in passes:
                    if time.time() - t0 > deadline:
                        log(f"  {pname}: пропущен по дедлайну"); continue
                    pr = []
                    for r in range(len(sub)):
                        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": userfn(r)}]
                        try:
                            pr.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                              enable_thinking=False))
                        except TypeError:
                            pr.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                    ln = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in pr]
                    out = np.full(len(sub), np.nan)
                    order = sorted(range(len(sub)), key=lambda i2: ln[i2]); i2 = 0
                    while i2 < len(order):
                        if time.time() - t0 > deadline:
                            log(f"  {pname}: дедлайн на {int(np.isfinite(out).sum())}/{len(sub)}"); break
                        bt, mx = [], 0
                        while i2 + len(bt) < len(order) and len(bt) < MBS:
                            j2 = order[i2 + len(bt)]; nm2 = max(mx, ln[j2])
                            if bt and nm2 * (len(bt) + 1) > MBT: break
                            bt.append(j2); mx = nm2
                        i2 += len(bt)
                        enc = tok([pr[j2] for j2 in bt], return_tensors="pt", padding=True,
                                  add_special_tokens=False).to("cuda")
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                            try:
                                lg = mdl(**enc, logits_to_keep=1).logits[:, -1, :].float()
                            except TypeError:
                                lg = mdl(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
                        lp = torch.log_softmax(lg, dim=-1)
                        sc = (torch.logsumexp(lp[:, yi], 1) - torch.logsumexp(lp[:, no], 1)).cpu().numpy()
                        for j2, v in zip(bt, sc):
                            out[j2] = 1 / (1 + np.exp(-v))
                    if np.isfinite(out).all():
                        res[pname] = out
                        log(f"[{time.time()-t0:.0f}s] {pname}: готов")
                    else:
                        log(f"[{time.time()-t0:.0f}s] {pname}: неполный, отброшен")
                del mdl; torch.cuda.empty_cache()
                return res

            fs = json.load(open(os.path.join(HERE, "fewshot_submit.json"), encoding="utf-8"))
            import importlib.util as _iu
            _sp = _iu.spec_from_file_location("prompts", os.path.join(HERE, "prompts.py"))
            _pm = _iu.module_from_spec(_sp); _sp.loader.exec_module(_pm)
            INSTR = getattr(_pm, "SUBCAT_INSTRUCTIONS", {})

            def fewuser(r):
                b_ = subs[r]; d = fs.get(b_, [])
                lines = [f"- {clean_text(n)} || {clean_text(dd)} -> {'да' if l else 'нет'}" for n, dd, l in d]
                head = INSTR.get(b_, "")
                return ((head + "\n\n" if head else "") +
                        ("Примеры разметки этой подкатегории:\n" + "\n".join(lines) + "\n\n" if lines else "") +
                        f"Название: {sub.iloc[r]['name_text']}\nОписание: {str(sub.iloc[r]['desc'])[:400]}\n\nОтвет:")

            S.update(run_model(GEMMA, [("gemret", SYS_RET, lambda r: demos_prompt(r, 4)),
                                       ("ret8", SYS_RET, lambda r: demos_prompt(r, 8))], DL_GEM, "gemma"))
            S.update(run_model(QWEN, [("qret", SYS_RET, lambda r: demos_prompt(r, 4)),
                                      ("few", _pm.SYSTEM_BASE, fewuser)], DL_QWEN, "qwen"))
        except Exception as e:
            log(f"LLM недоступна ({type(e).__name__}: {e})")

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
