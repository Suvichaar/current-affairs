import io
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html  # inline HTML viewer
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI
import boto3
import pandas as pd  # CSV/Excel export

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(
    page_title="Notes → OCR → Quiz → AMP",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Notes/Quiz OCR → GPT Structuring → AMP Web Story")
st.caption("Upload notes image(s) or a pre-made quiz image (or JSON), plus an AMP HTML template → download timestamped final HTML. Saves uploads to S3 and exports MCQs to CSV/Excel (6 per image).")

# ===========================
# Secrets / Config
# ===========================
try:
    # Azure (OCR)
    AZURE_DI_ENDPOINT = st.secrets["AZURE_DI_ENDPOINT"]
    AZURE_API_KEY     = st.secrets["AZURE_DI_API_KEY"]  # rename in secrets.toml to avoid collisions

    # Azure OpenAI (GPT)
    AZURE_OPENAI_ENDPOINT     = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_API_VERSION  = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_API_KEY      = st.secrets.get("AZURE_OPENAI_API_KEY", AZURE_API_KEY)
    GPT_DEPLOYMENT            = st.secrets.get("GPT_DEPLOYMENT", "gpt-4")

    # AWS / S3
    AWS_ACCESS_KEY_ID     = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_REGION            = st.secrets.get("AWS_REGION", "ap-south-1")
    AWS_BUCKET            = st.secrets.get("AWS_BUCKET", "suvichaarapp")

    # Final HTML path + CDN
    HTML_S3_PREFIX = st.secrets.get("HTML_S3_PREFIX", "webstory-html")
    CDN_HTML_BASE  = st.secrets.get("CDN_HTML_BASE", "https://stories.suvichaar.org/")

    # Uploaded images path + CDN
    IMAGES_S3_PREFIX = st.secrets.get("IMAGES_S3_PREFIX", "notes-uploads")
    CDN_MEDIA_BASE   = st.secrets.get("CDN_MEDIA_BASE", CDN_HTML_BASE)

    # Artifact prefixes (JSON/CSV/XLSX/Placeholders)
    QUIZ_JSON_PREFIX    = st.secrets.get("QUIZ_JSON_PREFIX", "quiz-json")
    QUIZ_CSV_PREFIX     = st.secrets.get("QUIZ_CSV_PREFIX", "quiz-csv")
    PLACEHOLDERS_PREFIX = st.secrets.get("PLACEHOLDERS_PREFIX", "quiz-placeholders")
except Exception:
    st.error("Missing secrets. Please set required Azure and AWS keys in .streamlit/secrets.toml")
    st.stop()

# ===========================
# Clients
# ===========================
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY)
)

gpt_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

# ===========================
# Prompts
# ===========================
SYSTEM_PROMPT_OCR_TO_QA = """
You receive OCR text that already contains multiple-choice questions in Hindi or English.
Each question has options (A)-(D), a single correct answer, and ideally an explanation.

Return a JSON object:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    },
    ...
  ]
}

- Return EXACTLY SIX questions total if there are more; pick the six clearest and most complete.
- If explanations are missing, write a concise 1–2 sentence explanation grounded in the text.
- Preserve the original language (Hindi stays Hindi, English stays English).
- Ensure valid JSON only.
"""

SYSTEM_PROMPT_NOTES_TO_QA = """
You are given raw study notes text (could be Hindi or English). Generate exactly SIX high-quality
multiple-choice questions (MCQs) that are strictly grounded in these notes.

For each question:
- Provide four options labeled A–D.
- Ensure exactly one correct option.
- Add a 1–2 sentence explanation that justifies the correct answer using the notes.

Respond ONLY with valid JSON in this schema:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    },
    ...
  ]
}

Language: Use the same language as the notes (auto-detect). Keep questions concise and unambiguous.
"""

SYSTEM_PROMPT_QA_TO_PLACEHOLDERS = """
You are given a JSON object with key "questions": a list where each item has:
- question (string)
- options: {"A":..., "B":..., "C":..., "D":...}
- correct_option (A/B/C/D)
- explanation (string)

Produce a single flat JSON object with EXACTLY these keys (sensible short defaults if missing).
Use the SAME language as the input questions (auto-detect; Hindi → Hindi, English → English).

pagetitle, storytitle, typeofquiz, potraitcoverurl,
s1title1, s1text1,

s2questionHeading, s2question1,
s2option1, s2option1attr, s2option2, s2option2attr,
s2option3, s2option3attr, s2option4, s2option4attr,
s2attachment1,

s3questionHeading, s3question1,
s3option1, s3option1attr, s3option2, s3option2attr,
s3option3, s3option3attr, s3option4, s3option4attr,
s3attachment1,

s4questionHeading, s4question1,
s4option1, s4option1attr, s4option2, s4option2attr,
s4option3, s4option3attr, s4option4, s4option4attr,
s4attachment1,

s5questionHeading, s5question1,
s5option1, s5option1attr, s5option2, s5option2attr,
s5option3, s5option3attr, s5option4, s5option4attr,
s5attachment1,

s6questionHeading, s6question1,
s6option1, s6option1attr, s6option2, s6option2attr,
s6option3, s6option3attr, s6option4, s6option4attr,
s6attachment1,

results_bg_image, results_prompt_text, results1_text, results2_text, results3_text

Mapping rules:
- We only need FIVE questions in the AMP template: map questions[0] → s2*, questions[1] → s3*, … questions[4] → s6*.
- If there are SIX questions, ignore the 6th when filling the template; but keep it in CSV/Excel export.
- sNquestion1 ← questions[N-2].question  (N=2..6)
- sNoption1..4 ← options A..D text
- For the correct option, set sNoptionKattr to the string "correct"; for others set "".
- sNattachment1 ← explanation for that question
- sNquestionHeading ← "Question {N-1}" (or language-appropriate equivalent)
- pagetitle/storytitle: create short, relevant titles from the overall content.
- typeofquiz: "Educational" (or language-appropriate equivalent) if unknown.
- s1title1: a 2–5 word intro title; s1text1: 1–2 sentence intro.
- results_*: short friendly strings in the same language. results_bg_image: "" if none.

Return only the JSON object.
""".strip()

# ===========================
# Helpers
# ===========================
def clean_model_json(txt: str) -> str:
    """Remove code fences if model returns ```json ... ``` or ``` ... ```."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return txt.strip()

def ocr_extract(image_bytes: bytes) -> str:
    """OCR via Azure Document Intelligence prebuilt-read for one image."""
    poller = di_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=image_bytes
    )
    result = poller.result()
    if getattr(result, "paragraphs", None):
        return "\n".join([p.content for p in result.paragraphs]).strip()
    if getattr(result, "content", None):
        return result.content.strip()
    lines = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            if getattr(line, "content", None):
                lines.append(line.content)
    return "\n".join(lines).strip()

def ocr_extract_each(files):
    """Return list of (image_idx, text) after OCR for each uploaded file."""
    out = []
    for idx, f in enumerate(files, start=1):
        txt = ocr_extract(f.getvalue())
        out.append((idx, txt.strip()))
    return out

def gpt_ocr_text_to_questions(raw_text: str) -> dict:
    """Convert OCR text that already contains questions into structured questions JSON (pick 6)."""
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_OCR_TO_QA},
            {"role": "user", "content": raw_text}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    q = data.get("questions", [])
    if len(q) > 6:
        data["questions"] = q[:6]
    return data

def gpt_notes_to_questions(notes_text: str) -> dict:
    """Generate 6 MCQs from raw notes text."""
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_NOTES_TO_QA},
            {"role": "user", "content": notes_text}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    q = data.get("questions", [])
    if len(q) > 6:
        data["questions"] = q[:6]
    return data

def build_per_image_question_sets(texts_by_image, mode: str):
    """
    texts_by_image: list of (image_idx, text)
    mode: "notes" or "quiz"
    Returns: [{"image_idx": int, "questions_data": {...}}, ...]
    """
    packs = []
    for image_idx, txt in texts_by_image:
        if not txt:
            continue
        if mode == "notes":
            qd = gpt_notes_to_questions(txt)
        else:
            qd = gpt_ocr_text_to_questions(txt)
        packs.append({"image_idx": image_idx, "questions_data": qd})
    return packs

def gpt_questions_to_placeholders(questions_data: dict) -> dict:
    """Map structured questions JSON into flat placeholder JSON for AMP template (uses first 5)."""
    q = questions_data.get("questions", [])
    sub = {"questions": q[:5]} if len(q) > 5 else {"questions": q}
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_QA_TO_PLACEHOLDERS},
            {"role": "user", "content": json.dumps(sub, ensure_ascii=False)}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def build_attr_value(key: str, val: str) -> str:
    """s2option3attr + 'correct' → 'option-3-correct', else passthrough/empty."""
    if not key.endswith("attr") or not val:
        return ""
    m = re.match(r"s(\d+)option(\d)attr$", key)
    if m and val.strip().lower() == "correct":
        return f"option-{m.group(2)}-correct"
    return val

def fill_template(template: str, data: dict) -> str:
    """Replace {{key}} and {{key|safe}} using placeholder data, handling *attr keys specially."""
    rendered = {}
    for k, v in data.items():
        rendered[k] = build_attr_value(k, str(v)) if k.endswith("attr") else ("" if v is None else str(v))
    html = template
    for k, v in rendered.items():
        html = html.replace(f"{{{{{k}}}}}", v)
        html = html.replace(f"{{{{{k}|safe}}}}", v)
    return html

# ---------- S3 helpers ----------
def upload_html_to_s3(html_text: str, filename: str):
    """Upload HTML to S3 and return (s3_key, cdn_url)."""
    if not filename.lower().endswith(".html"):
        filename = f"{filename}.html"
    s3_key = f"{HTML_S3_PREFIX.strip('/')}/{filename}" if HTML_S3_PREFIX else filename
    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=html_text.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
        CacheControl="public, max-age=300",
        ContentDisposition=f'inline; filename="{filename}"',
    )
    cdn_url = f"{CDN_HTML_BASE.rstrip('/')}/{s3_key}"
    return s3_key, cdn_url

def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> str:
    """Generic uploader; returns CDN URL."""
    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=300",
        ContentDisposition=f'inline; filename="{Path(key).name}"',
    )
    return f"{CDN_HTML_BASE.rstrip('/')}/{key}"

def upload_images_to_s3(files):
    """Upload a list of uploaded images to S3. Returns list of (key, cdn_url)."""
    ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3 = get_s3_client()
    out = []
    for i, f in enumerate(files, start=1):
        raw = f.getvalue()
        ct = "image/jpeg"
        name_lower = f.name.lower()
        if name_lower.endswith(".png"):
            ct = "image/png"
        elif name_lower.endswith(".webp"):
            ct = "image/webp"
        elif name_lower.endswith((".tif", ".tiff")):
            ct = "image/tiff"

        key = f"{IMAGES_S3_PREFIX.strip('/')}/{ts_folder}/{i:02d}_{Path(f.name).name}"
        s3.put_object(
            Bucket=AWS_BUCKET,
            Key=key,
            Body=raw,
            ContentType=ct,
            CacheControl="public, max-age=86400",
            ContentDisposition=f'inline; filename="{Path(f.name).name}"',
        )
        url = f"{CDN_MEDIA_BASE.rstrip('/')}/{key}"
        out.append((key, url))
    return out

def upload_json_to_s3(obj: dict, prefix: str, base_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"{prefix.strip('/')}/{base_name}_{ts}.json"
    return upload_bytes_to_s3(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), key, "application/json")

def upload_df_csv_xlsx_to_s3(df: pd.DataFrame, prefix: str, base_name: str) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    urls = {}

    # CSV
    csv_key = f"{prefix.strip('/')}/{base_name}_{ts}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    urls["csv"] = upload_bytes_to_s3(csv_bytes, csv_key, "text/csv")

    # XLSX
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="MCQs", index=False)
    xlsx_key = f"{prefix.strip('/')}/{base_name}_{ts}.xlsx"
    urls["xlsx"] = upload_bytes_to_s3(
        xlsx_buf.getvalue(),
        xlsx_key,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return urls

# ---------- Wide DataFrame builder (exact columns you requested) ----------
def questions_to_wide_dataframe(per_image_packs, image_urls, max_q=6) -> pd.DataFrame:
    """
    Build one row per image with columns:
    image_url,
    question1, s1option1, s1option2, s1option3, s1option4, correct_option1, explaination1,
    question2, s2option1, ..., correct_option2, explaination2,
    ...
    up to `max_q` (default 6).
    """
    idx_to_url = {i + 1: (image_urls[i] if i < len(image_urls) else "") for i in range(len(image_urls))}
    rows = []

    for pack in sorted(per_image_packs, key=lambda x: x["image_idx"]):
        img_idx = pack["image_idx"]
        img_url = idx_to_url.get(img_idx, "")
        qs = pack["questions_data"].get("questions", []) or []

        row = {"image_url": img_url}

        # Fill each question block
        for qn in range(1, max_q + 1):
            q = qs[qn - 1] if qn - 1 < len(qs) else {}
            opts = q.get("options", {}) if q else {}
            correct = (q.get("correct_option", "") if q else "").strip()

            row[f"question{qn}"] = q.get("question", "") if q else ""
            row[f"s{qn}option1"] = opts.get("A", "")
            row[f"s{qn}option2"] = opts.get("B", "")
            row[f"s{qn}option3"] = opts.get("C", "")
            row[f"s{qn}option4"] = opts.get("D", "")
            row[f"correct_option{qn}"] = correct
            # keep user's requested spelling
            row[f"explaination{qn}"] = q.get("explanation", "") if q else ""

        rows.append(row)

    # Column order exactly as requested
    cols = ["image_url"]
    for qn in range(1, max_q + 1):
        cols += [
            f"question{qn}",
            f"s{qn}option1", f"s{qn}option2", f"s{qn}option3", f"s{qn}option4",
            f"correct_option{qn}",
            f"explaination{qn}",
        ]
    df = pd.DataFrame(rows, columns=cols)
    return df

def df_download_buttons(df: pd.DataFrame, base_name: str = "mcqs_export"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download MCQs (CSV)",
        data=csv_bytes,
        file_name=f"{base_name}.csv",
        mime="text/csv"
    )
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="MCQs", index=False)
    st.download_button(
        "⬇️ Download MCQs (Excel)",
        data=xlsx_buf.getvalue(),
        file_name=f"{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===========================
# UI
# ===========================
tab_all, = st.tabs(["All-in-one Builder"])

with tab_all:
    st.subheader("Build final AMP HTML from image(s) or structured JSON")
    st.caption("Pick an input source, upload AMP HTML template, and download the final HTML with a timestamped filename.")

    mode = st.radio(
        "Choose input",
        [
            "Notes image(s) (OCR → generate quiz JSON)",
            "Quiz image(s) (OCR → parse existing MCQs)",
            "Structured JSON (skip OCR)"
        ],
        horizontal=False
    )

    up_tpl = st.file_uploader("📎 Upload AMP HTML template (.html)", type=["html", "htm"], key="tpl")
    show_debug = st.toggle("Show OCR / JSON previews", value=False)

    uploaded_image_urls = []        # CDN URLs of uploaded source images
    per_image_packs = []            # [{"image_idx": 1, "questions_data": {...}}, ...]

    # ---------- Notes images mode ----------
    if mode == "Notes image(s) (OCR → generate quiz JSON)":
        up_imgs = st.file_uploader(
            "📎 Upload notes image(s) (JPG/PNG/WebP/TIFF) — multiple allowed",
            type=["jpg", "jpeg", "png", "webp", "tiff"],
            accept_multiple_files=True,
            key="notes_imgs"
        )
        if up_imgs:
            try:
                with st.spinner("☁️ Uploading source images to S3…"):
                    k_urls = upload_images_to_s3(up_imgs)
                    uploaded_image_urls = [u for _, u in k_urls]
                st.success(f"Uploaded {len(uploaded_image_urls)} image(s) to S3.")
                if show_debug:
                    st.write("Source image URLs:")
                    for u in uploaded_image_urls:
                        st.write(u)
            except Exception as e:
                st.error(f"Failed to upload images to S3: {e}")
                st.stop()

            if show_debug:
                for i, f in enumerate(up_imgs, start=1):
                    try:
                        st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                 caption=f"Notes page {i}", use_container_width=True)
                    except Exception:
                        pass

            try:
                with st.spinner("🔍 OCR (per image)…"):
                    texts_by_image = ocr_extract_each(up_imgs)  # [(idx, text), ...]
                if not any(t for _, t in texts_by_image):
                    st.error("OCR returned empty text for all images.")
                    st.stop()

                with st.spinner("📝 Generating 6 MCQs per image (notes)…"):
                    per_image_packs = build_per_image_question_sets(texts_by_image, mode="notes")
            except Exception as e:
                st.error(f"Failed to process notes → quizzes: {e}")
                st.stop()

    # ---------- Quiz images mode ----------
    elif mode == "Quiz image(s) (OCR → parse existing MCQs)":
        up_imgs = st.file_uploader(
            "📎 Upload quiz image(s) (JPG/PNG/WebP/TIFF) — multiple allowed",
            type=["jpg", "jpeg", "png", "webp", "tiff"],
            accept_multiple_files=True,
            key="quiz_imgs"
        )
        if up_imgs:
            try:
                with st.spinner("☁️ Uploading source images to S3…"):
                    k_urls = upload_images_to_s3(up_imgs)
                    uploaded_image_urls = [u for _, u in k_urls]
                st.success(f"Uploaded {len(uploaded_image_urls)} image(s) to S3.")
                if show_debug:
                    st.write("Source image URLs:")
                    for u in uploaded_image_urls:
                        st.write(u)
            except Exception as e:
                st.error(f"Failed to upload images to S3: {e}")
                st.stop()

            if show_debug:
                for i, f in enumerate(up_imgs, start=1):
                    try:
                        st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                 caption=f"Quiz page {i}", use_container_width=True)
                    except Exception:
                        pass

            try:
                with st.spinner("🔍 OCR (per image)…"):
                    texts_by_image = ocr_extract_each(up_imgs)
                if not any(t for _, t in texts_by_image):
                    st.error("OCR returned empty text for all images.")
                    st.stop()

                with st.spinner("🤖 Parsing 6 MCQs per image (quiz)…"):
                    per_image_packs = build_per_image_question_sets(texts_by_image, mode="quiz")
            except Exception as e:
                st.error(f"Failed to process quiz images → quizzes: {e}")
                st.stop()

    # ---------- Structured JSON mode ----------
    else:
        up_json = st.file_uploader("📎 Upload structured questions JSON", type=["json"], key="json")
        if up_json:
            try:
                questions_data = json.loads(up_json.getvalue().decode("utf-8"))
                per_image_packs = [{"image_idx": 1, "questions_data": questions_data}]
                if show_debug:
                    with st.expander("🧱 Structured Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:12000], language="json")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

    # ===========================
    # Export MCQs (WIDE, one row per image) + S3
    # ===========================
    if per_image_packs:
        try:
            df = questions_to_wide_dataframe(per_image_packs, uploaded_image_urls, max_q=6)
            st.markdown("### 📊 MCQs (one row per image, wide format)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Local downloads
            df_download_buttons(df, base_name=f"mcqs_wide_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Upload the wide CSV/XLSX + the raw JSON packs to S3
            st.markdown("### ☁️ S3 artifacts")
            combined_json = {"packs": per_image_packs}
            json_url = upload_json_to_s3(combined_json, prefix=QUIZ_JSON_PREFIX, base_name="questions_all_images")
            file_urls = upload_df_csv_xlsx_to_s3(df, prefix=QUIZ_CSV_PREFIX, base_name="mcqs_wide_all_images")

            st.success("Uploaded quiz artifacts to S3")
            st.write("**All-Images Questions JSON (raw):**", json_url)
            st.write("**CSV (wide):**", file_urls["csv"])
            st.write("**Excel (wide):**", file_urls["xlsx"])

        except Exception as e:
            st.warning(f"Could not build/upload wide CSV/Excel artifacts: {e}")

    # ===========================
    # Build AMP from one image's questions
    # ===========================
    selected_image_idx = None
    if per_image_packs:
        indices = [p["image_idx"] for p in sorted(per_image_packs, key=lambda x: x["image_idx"])]
        selected_image_idx = st.selectbox("Choose image for AMP placeholders (first 5 Qs):", indices, index=0)

    build = st.button("🛠️ Build final HTML", disabled=not (per_image_packs and up_tpl and selected_image_idx is not None))

    if build and per_image_packs and up_tpl and selected_image_idx is not None:
        try:
            # pick the selected pack
            pack = next(p for p in per_image_packs if p["image_idx"] == selected_image_idx)
            questions_for_amp = pack["questions_data"]

            with st.spinner("🧩 Generating placeholders (first 5 Qs)…"):
                placeholders = gpt_questions_to_placeholders(questions_for_amp)
                if show_debug:
                    with st.expander("🧩 Placeholder JSON"):
                        st.code(json.dumps(placeholders, ensure_ascii=False, indent=2)[:12000], language="json")

            # Optional: upload placeholders JSON for audit
            try:
                placeholders_url = upload_json_to_s3(placeholders, prefix=PLACEHOLDERS_PREFIX, base_name=f"placeholders_img{selected_image_idx}")
                st.write("**Placeholders JSON (used for replacement):**", placeholders_url)
            except Exception as e:
                st.warning(f"Could not upload placeholders JSON: {e}")

            # merge into HTML
            template_html = up_tpl.getvalue().decode("utf-8")
            final_html = fill_template(template_html, placeholders)

            # save local + upload
            ts_name = f"final_quiz_img{selected_image_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(ts_name).write_text(final_html, encoding="utf-8")
            with st.spinner("☁️ Uploading final HTML to S3…"):
                s3_key, cdn_url = upload_html_to_s3(final_html, ts_name)

            st.success(f"✅ Final HTML generated and uploaded to S3: s3://{AWS_BUCKET}/{s3_key}")
            st.markdown(f"**CDN URL:** {cdn_url}")

            with st.expander("🔍 HTML Preview (source)"):
                st.code(final_html[:120000], language="html")

            st.download_button(
                "⬇️ Download final HTML",
                data=final_html.encode("utf-8"),
                file_name=ts_name,
                mime="text/html"
            )

            # Inline preview (AMP may be limited in sandbox)
            st.markdown("### 👀 Live HTML Preview")
            h = st.slider("Preview height (px)", min_value=400, max_value=1600, value=900, step=50)
            full_width = st.checkbox("Force full viewport width (100vw)", value=True)
            style = f"width: {'100vw' if full_width else '100%'}; height: {h}px; border: 0; margin: 0; padding: 0;"
            st_html(final_html, height=h, scrolling=True) if not full_width else st_html(
                f'<div style="position:relative;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;{style}">{final_html}</div>',
                height=h,
                scrolling=True
            )

            st.info("AMP pages may not fully render inside Streamlit due to CSP/sandbox. For a faithful view, open in a real browser or the CDN URL.")
        except Exception as e:
            st.error(f"Build failed: {e}")
    elif not (up_tpl and per_image_packs):
        st.info("Upload images/JSON **and** a template to enable the Build button.")
