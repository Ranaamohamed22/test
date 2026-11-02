import io
import json
import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="تقييم الإفادات", page_icon="📋", layout="wide")
st.title("📋 مساعد تقييم الإفادات")
st.write("ارفعي ملف الموديل الذي يحتوي على الإفادات وتقييماتها، ثم ارفعي ملف الإفادات الجديدة. سيُضاف التقييم إلى نسخة جديدة من ملف Excel.")

st.info("قاعدة أساسية: المكالمة الناجحة = 1، وغير الناجحة = 0. إذا كانت الإفادة تحتوي على وعد سابق بالسداد لكن آخر متابعة كانت لا يرد/لا يجيب/مغلق/بريد صوتي، فالتقييم النهائي = 0.")

with st.sidebar:
    st.header("إعدادات الذكاء الاصطناعي")
    api_key = st.text_input("OpenAI API key", type="password", help="مفتاح API الخاص بك. لا تضعيه داخل الكود أو على GitHub.")
    model_name = st.text_input("اسم الموديل", value="gpt-4o-mini")
    st.caption("يتطلب التقييم الذكي اتصال إنترنت ومفتاح OpenAI API صالحًا. استخدام الـ API قد يترتب عليه تكلفة حسب الاستخدام.")

reference_file = st.file_uploader("1) ارفعي ملف الموديل (الإفادات + التقييم 0/1)", type=["xlsx", "xls", "csv"], key="reference")
target_file = st.file_uploader("2) ارفعي ملف الإفادات المطلوب تقييمها", type=["xlsx", "xls", "csv"], key="target")


def read_table(uploaded_file) -> tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        # Try UTF-8 first, then common Arabic Windows encoding.
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="cp1256")
        return df, "csv"
    return pd.read_excel(io.BytesIO(raw)), "excel"


def normalize_col(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def find_column(df: pd.DataFrame, choices: list[str]) -> str | None:
    normalized = {normalize_col(c): c for c in df.columns}
    for choice in choices:
        if normalize_col(choice) in normalized:
            return normalized[normalize_col(choice)]
    # Soft match for common column variants.
    for c in df.columns:
        n = normalize_col(c)
        if any(normalize_col(ch) in n for ch in choices):
            return c
    return None


def make_examples(df: pd.DataFrame, note_col: str, label_col: str) -> list[dict[str, Any]]:
    examples = []
    for _, row in df.iterrows():
        note = str(row.get(note_col, "") or "").strip()
        label = str(row.get(label_col, "")).strip()
        if not note or note.casefold() in {"nan", "none"}:
            continue
        try:
            y = int(float(label))
        except (TypeError, ValueError):
            continue
        if y in (0, 1):
            examples.append({"note": note[:1000], "label": y})
    # Keep a balanced sample if the model file is large.
    if len(examples) > 50:
        zeros = [x for x in examples if x["label"] == 0]
        ones = [x for x in examples if x["label"] == 1]
        examples = zeros[:25] + ones[:25]
    return examples


def classify_batch(client: OpenAI, model: str, notes: list[str], examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = """أنت مساعد متخصص في تقييم إفادات متابعة تحصيل الديون. صنّف كل إفادة برقم واحد فقط: 1 أو 0.

القواعد:
- 1 = حدث تواصل فعلي مع العميل، بما في ذلك الرد أو المناقشة أو الرفض أو الوعد بالسداد أو شرح حالة العميل/مراجعة البنك/الجدولة، ما لم تكن آخر نتيجة متابعة تثبت عدم الوصول للعميل.
- 0 = لم يحدث تواصل ناجح في آخر متابعة: لا يرد، لا يجيب، لايرد، لايجيب، الجوال/الخط مغلق، الخط لا يشبك/يفصل، بريد صوتي، المكالمة لا تتم، أو ما شابه.
- قاعدة أولوية حاسمة: اقرأ الإفادة كاملة وحدد نتيجة آخر متابعة زمنيًا/المذكورة في نهاية النص. إذا ورد وعد سابق أو سداد سابق ثم ذُكر في النهاية أن العميل الآن لا يرد/لا يجيب/مغلق/بريد صوتي، فالتقييم النهائي 0.
- مثال: «وعد بالسداد ولا يرد / جاري المتابعة» = 0.
- مثال: «سدد جزءًا من مبلغ المعالجة والان لا يجيب» = 0.
- مثال: «العميل وعد بالسداد اليوم، لكن في آخر المتابعة لا يرد» = 0.
- إذا كان العميل رد ورفض السداد، فهذا يظل 1 لأن التواصل نجح.
- لا تجعل كلمة «معالج» وحدها دليل نجاح؛ اتبع أمثلة الموديل المرجعي.
- استخدم ملف الموديل كأمثلة مرجعية، والقواعد أعلاه مقدمة عند تعارضها مع تعميم غير دقيق.
- لا تخترع تفاصيل غير موجودة. عند الغموض، اختر أقرب تصنيف للأمثلة المرجعية.

أعد JSON فقط بهذا الشكل: {\"results\":[{\"row\":1,\"label\":0,\"reason\":\"سبب مختصر بالعربية\"}]}.
السبب يجب أن يكون مختصرًا وواضحًا، والتقييم يجب أن يكون 0 أو 1 فقط."""
    user_payload = {
        "reference_examples": examples,
        "notes_to_classify": [{"row": i + 1, "note": n} for i, n in enumerate(notes)],
    }
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    results = parsed.get("results", [])
    mapped = {}
    for item in results:
        try:
            row_no = int(item.get("row"))
            label = int(item.get("label"))
            if label in (0, 1):
                mapped[row_no] = {"label": label, "reason": str(item.get("reason", "")).strip()}
        except (TypeError, ValueError):
            continue
    output = []
    for i, note in enumerate(notes, start=1):
        output.append(mapped.get(i, {"label": 0, "reason": "لم يتم استلام نتيجة واضحة من النموذج؛ يُرجى المراجعة اليدوية."}))
    return output

if reference_file and target_file:
    try:
        ref_df, ref_kind = read_table(reference_file)
        target_df, target_kind = read_table(target_file)
        ref_note_col = find_column(ref_df, ["Notes", "الإفادة", "الافادة", "نص الإفادة", "نص الافادة"])
        ref_label_col = find_column(ref_df, ["التقييم", "التقيم", "التقيم ", "Evaluation", "Label", "Score"])
        target_note_col = find_column(target_df, ["Notes", "الإفادة", "الافادة", "نص الإفادة", "نص الافادة"])

        if not ref_note_col or not ref_label_col:
            st.error("مش قادر أحدد أعمدة الموديل. لازم يحتوي على عمود للإفادة (مثل Notes) وعمود للتقييم (0 أو 1).")
        elif not target_note_col:
            st.error("مش قادر أحدد عمود الإفادات في الملف المطلوب تقييمه. سمّي العمود Notes أو الإفادة أو نص الإفادة.")
        else:
            examples = make_examples(ref_df, ref_note_col, ref_label_col)
            if not examples:
                st.error("ملف الموديل لا يحتوي على أمثلة صالحة بتقييم 0 أو 1.")
            else:
                st.success(f"تمت قراءة {len(examples)} مثال مرجعي. عمود الإفادات في الملف الجديد: {target_note_col}.")
                st.caption(f"عدد الصفوف في الملف المطلوب تقييمه: {len(target_df)}")
                if st.button("ابدئي تقييم الإفادات", type="primary", disabled=not api_key.strip()):
                    client = OpenAI(api_key=api_key.strip())
                    notes = ["" if pd.isna(x) else str(x).strip() for x in target_df[target_note_col].tolist()]
                    all_results = []
                    progress = st.progress(0)
                    status = st.empty()
                    batch_size = 15
                    try:
                        for start in range(0, len(notes), batch_size):
                            batch = notes[start:start + batch_size]
                            status.write(f"جاري تقييم الصفوف {start + 1} إلى {min(start + batch_size, len(notes))}...")
                            batch_results = classify_batch(client, model_name.strip() or "gpt-4o-mini", batch, examples)
                            all_results.extend(batch_results)
                            progress.progress(min(1.0, len(all_results) / max(len(notes), 1)))
                        # Update/create the evaluation column while preserving all original columns.
                        label_col = find_column(target_df, ["التقييم", "التقيم", "Evaluation", "Label", "Score"]) or "التقييم"
                        reason_col = "سبب التقييم"
                        target_df[label_col] = [r["label"] for r in all_results]
                        target_df[reason_col] = [r["reason"] for r in all_results]
                        output = io.BytesIO()
                        if target_kind == "csv":
                            target_df.to_csv(output, index=False, encoding="utf-8-sig")
                            mime = "text/csv"
                            out_name = "نتيجة_تقييم_الإفادات.csv"
                        else:
                            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                                target_df.to_excel(writer, index=False, sheet_name="تقييم الإفادات")
                            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            out_name = "نتيجة_تقييم_الإفادات.xlsx"
                        status.success("اكتمل التقييم. راجعي عينة من النتائج قبل الاعتماد النهائي.")
                        st.dataframe(target_df.head(30), use_container_width=True)
                        st.download_button("⬇️ تحميل الشيت بعد التقييم", data=output.getvalue(), file_name=out_name, mime=mime)
                        st.warning("التقييم آلي وقد يخطئ، خصوصًا في الإفادات الغامضة أو المكتوبة باختصارات. راجعي النتائج قبل استخدامها في تقارير العمل.")
                    except Exception as exc:
                        st.error(f"حصل خطأ أثناء التقييم: {exc}")
    except Exception as exc:
        st.error(f"تعذر قراءة أحد الملفين: {exc}")
else:
    st.caption("ارفعي الملفين للبدء. ملف الموديل هو المرجع الذي يحتوي على Notes والتقييم 0/1.")
