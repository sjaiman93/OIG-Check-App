"""
Master Federal Compliance Auditor
==================================
Bulk screens a candidate roster against OIG LEIE, SAM.gov Exclusions,
and the full OFAC SDN + Alternate + Consolidated + Alternate name sets,
issuing an automated preliminary screening PDF per cleared candidate.

--------------------------------------------------------------------
DEPLOYMENT — STREAMLIT COMMUNITY CLOUD
--------------------------------------------------------------------
1. Save as `app.py` in a GitHub repo.
2. requirements.txt:
       streamlit
       pandas
       numpy
       reportlab
       rapidfuzz
       pytz
       openpyxl
3. https://share.streamlit.io -> New app -> point to repo -> Deploy.
4. Restrict access under Settings -> Sharing (this handles PII +
   federal watchlist data — do not leave it public).

--------------------------------------------------------------------
DEPLOYMENT — REPLIT
--------------------------------------------------------------------
1. New Replit (Python) -> paste into main.py.
2. Add requirements.txt (same as above).
3. .replit file:
       run = "streamlit run main.py --server.port 8080 --server.address 0.0.0.0"
4. Run -> use webview URL, or Replit Deployments for an always-on link.
--------------------------------------------------------------------
"""

import io
import uuid
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from rapidfuzz import fuzz, process
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

st.set_page_config(page_title="Master Federal Compliance Auditor", layout="wide")
EASTERN = pytz.timezone("US/Eastern")
OFAC_THRESHOLD = 85
OFAC_CDIST_CHUNK_SIZE = 5000  # tune down if you hit memory limits on your host
OFAC_FORMAL_LABEL = "U.S. Treasury Specially Designated Nationals and Consolidated Sanctions Lists"

# =====================================================================
# Loaders
# =====================================================================

@st.cache_data(show_spinner=False)
def load_oig(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_sam(file_bytes: bytes) -> pd.DataFrame:
    """SAM.gov exclusions extract. Column names vary by export vintage,
    so we map common variants to a canonical First/Last."""
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    first_candidates = ["first", "firstname", "first_name", "first name"]
    last_candidates = ["last", "lastname", "last_name", "last name"]

    first_col = next((c for c in first_candidates if c in df.columns), None)
    last_col = next((c for c in last_candidates if c in df.columns), None)

    if first_col is None or last_col is None:
        raise ValueError(
            "Could not locate First/Last name columns in the SAM.gov extract. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.rename(columns={first_col: "sam_first", last_col: "sam_last"})
    return df


def _extract_ofac_names_fixed(file_bytes: bytes, name_col_idx: int, source_label: str) -> pd.DataFrame:
    """Load a headerless OFAC flat file and pull the name field from a
    hardcoded, known-good column position:
      - SDN.CSV / CONS_PRIM.CSV -> name is column index 1
      - ALT.CSV  / CONS_ALT.CSV -> name is column index 3
    """
    df = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str, low_memory=False)

    if df.shape[1] <= name_col_idx:
        raise ValueError(
            f"{source_label}: expected a name column at index {name_col_idx}, "
            f"but the file only has {df.shape[1]} columns. Confirm this is the "
            "correct, unmodified Treasury export."
        )

    names = df[name_col_idx].astype(str).str.strip()
    out = pd.DataFrame({"ofac_name": names, "ofac_source": source_label})
    out = out[out["ofac_name"].str.len() > 0]
    out = out[~out["ofac_name"].str.lower().isin(["nan", "none", "-0-"])]
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ofac(sdn_bytes, alt_bytes, cons_prim_bytes, cons_alt_bytes) -> pd.DataFrame:
    """Combine all four OFAC files into one normalized lookup frame,
    each row tagged with which specific source file it came from.
    Column positions are hardcoded per the known Treasury flat-file layout."""
    frames = [
        _extract_ofac_names_fixed(sdn_bytes, name_col_idx=1, source_label="OFAC SDN (Primary)"),
        _extract_ofac_names_fixed(alt_bytes, name_col_idx=3, source_label="OFAC SDN (Alternate)"),
        _extract_ofac_names_fixed(cons_prim_bytes, name_col_idx=1, source_label="OFAC Consolidated (Primary)"),
        _extract_ofac_names_fixed(cons_alt_bytes, name_col_idx=3, source_label="OFAC Consolidated (Alternate)"),
    ]
    combined = pd.concat(frames, ignore_index=True)

    # De-duplicate exact repeated names to shrink the matching array — first
    # occurrence's source label wins for display purposes.
    deduped = combined.drop_duplicates(subset="ofac_name").reset_index(drop=True)
    return deduped


def load_roster(uploaded_file) -> pd.DataFrame:
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file, dtype=str)
    else:
        df = pd.read_excel(uploaded_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    required = {"Primary Name", "Aliases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Roster is missing required column(s): {', '.join(missing)}")

    df["Primary Name"] = df["Primary Name"].fillna("").astype(str).str.strip()
    df["Aliases"] = df["Aliases"].fillna("").astype(str).str.strip()
    return df


def split_name(full_name: str):
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0].upper(), parts[0].upper()
    return parts[0].upper(), parts[-1].upper()


# =====================================================================
# Exact-match logic (OIG, SAM)
# =====================================================================

def check_oig(name: str, oig_df: pd.DataFrame):
    first, last = split_name(name)
    if first is None:
        return []
    result = oig_df[
        (oig_df["lastname"].astype(str).str.upper() == last) &
        (oig_df["firstname"].astype(str).str.upper() == first)
    ]
    return result.to_dict("records")


def check_sam(name: str, sam_df: pd.DataFrame):
    first, last = split_name(name)
    if first is None:
        return []
    result = sam_df[
        (sam_df["sam_last"].astype(str).str.upper() == last) &
        (sam_df["sam_first"].astype(str).str.upper() == first)
    ]
    return result.to_dict("records")


# =====================================================================
# Batch OFAC fuzzy matching via rapidfuzz.process.cdist
# =====================================================================

def batch_ofac_match(queries: list, choices: list, threshold: int = OFAC_THRESHOLD,
                      chunk_size: int = OFAC_CDIST_CHUNK_SIZE):
    """
    Vectorized fuzzy matching of every query name against the full OFAC
    choices array using rapidfuzz.process.cdist, chunked over `choices` so
    the score matrix never has to hold (n_queries x n_choices) all at once
    in memory.

    Returns two numpy arrays aligned to `queries`:
      best_score[i]  -> highest token_set_ratio score for queries[i]
      best_idx[i]    -> index into `choices` of that best match (-1 if none)
    """
    n_q = len(queries)
    if n_q == 0 or not choices:
        return np.zeros(0), np.full(0, -1)

    best_score = np.zeros(n_q, dtype=float)
    best_idx = np.full(n_q, -1, dtype=int)

    for start in range(0, len(choices), chunk_size):
        chunk = choices[start:start + chunk_size]
        scores = process.cdist(queries, chunk, scorer=fuzz.token_set_ratio, workers=-1)
        chunk_best_idx = scores.argmax(axis=1)
        chunk_best_score = scores[np.arange(n_q), chunk_best_idx]

        better = chunk_best_score > best_score
        best_score[better] = chunk_best_score[better]
        best_idx[better] = chunk_best_idx[better] + start

    below_threshold = best_score < threshold
    best_idx[below_threshold] = -1
    return best_score, best_idx


def run_batch_audit(roster_df: pd.DataFrame, oig_df, sam_df, ofac_df):
    """
    Two-pass audit:
      Pass 1 (fast, per-row): OIG + SAM exact matches.
      Pass 2 (batched): every candidate/alias name across the ENTIRE
        roster is deduplicated and run through a single cdist batch pass
        against OFAC.
    """
    flat_names = []
    for idx, row in roster_df.iterrows():
        names = [row["Primary Name"]] + [a.strip() for a in row["Aliases"].split(",") if a.strip()]
        for name in names:
            if name:
                flat_names.append((idx, name))

    exceptions_by_row = {idx: [] for idx in roster_df.index}
    unique_names = sorted(set(name for _, name in flat_names))

    oig_hits_by_name = {}
    sam_hits_by_name = {}
    for name in unique_names:
        oig_hits_by_name[name] = check_oig(name, oig_df)
        sam_hits_by_name[name] = check_sam(name, sam_df)

    for idx, name in flat_names:
        for rec in oig_hits_by_name[name]:
            exceptions_by_row[idx].append({
                "Matched Name": name,
                "Source": "OIG LEIE",
                "Match Score": "Exact",
                "Detail": f"{rec.get('excltype', 'N/A')} / {rec.get('state', 'N/A')}",
            })
        for rec in sam_hits_by_name[name]:
            exceptions_by_row[idx].append({
                "Matched Name": name,
                "Source": "SAM.gov",
                "Match Score": "Exact",
                "Detail": "SAM.gov Exclusions Public Extract",
            })

    ofac_choices = ofac_df["ofac_name"].str.upper().tolist()
    query_list = [n.upper() for n in unique_names]

    best_score, best_idx = batch_ofac_match(query_list, ofac_choices, threshold=OFAC_THRESHOLD)

    ofac_result_by_name = {}
    for name, score, idx_in_choices in zip(unique_names, best_score, best_idx):
        if idx_in_choices >= 0:
            row = ofac_df.iloc[idx_in_choices]
            ofac_result_by_name[name] = {
                "matched_name": row["ofac_name"],
                "source": row["ofac_source"],
                "score": round(float(score), 1),
            }

    for idx, name in flat_names:
        hit = ofac_result_by_name.get(name)
        if hit:
            exceptions_by_row[idx].append({
                "Matched Name": name,
                "Source": hit["source"],
                "Match Score": f"{hit['score']}%",
                "Detail": f"Fuzzy match to '{hit['matched_name']}'",
            })

    return exceptions_by_row


# =====================================================================
# PDF report
# =====================================================================

def build_report_pdf(candidate_name: str, aliases: str, audit_id: str,
                      oig_file_date: str, sam_file_date: str, ofac_file_date: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=17, spaceAfter=4,
        textColor=colors.HexColor("#1a3d63"),
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading3"], fontSize=12, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1a3d63"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#555555"), spaceAfter=10,
    )
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=9.5, leading=13)
    vintage_style = ParagraphStyle(
        "VintageStyle", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#8a5a00"), spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "FooterStyle", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=colors.HexColor("#666666"),
    )

    now_eastern = datetime.now(EASTERN).strftime("%B %d, %Y %I:%M %p %Z")
    aliases_display = aliases if aliases else "None provided"

    elements = [
        Paragraph("Automated Preliminary Name Screening Report", title_style),
        Paragraph("Internal RPO Compliance System &mdash; Automated Screening Record", subtitle_style),
        HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=1),
        Spacer(1, 10),
    ]

    candidate_table = Table([
        ["Candidate Name:", candidate_name],
        ["Aliases Screened:", aliases_display],
        ["Timestamp:", now_eastern],
        ["Audit ID:", audit_id],
    ], colWidths=[1.7 * inch, 4.3 * inch])
    candidate_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(candidate_table)
    elements.append(Spacer(1, 6))

    def db_section(title, status_line, source_line, vintage_label, vintage_value):
        block = [
            Paragraph(title, section_style),
            Paragraph(f"<b>Status:</b> {status_line}", body_style),
            Paragraph(f"<b>Source:</b> {source_line}", body_style),
        ]
        vintage_display = vintage_value.strip() if vintage_value and vintage_value.strip() else "Not specified"
        block.append(Paragraph(f"<b>{vintage_label}:</b> {vintage_display}", vintage_style))
        return block

    elements += db_section(
        "OIG LEIE",
        "CLEARED &mdash; No exact first/last name match found.",
        "OIG LEIE Database",
        "Data As Of",
        oig_file_date,
    )
    elements += db_section(
        "SAM.gov Exclusions",
        "CLEARED &mdash; No exact first/last name match found.",
        "SAM Exclusions Public Extract",
        "Data As Of",
        sam_file_date,
    )
    elements += db_section(
        "OFAC Sanctions (SDN &amp; Consolidated, Primary &amp; Alternate)",
        f"CLEARED &mdash; No matches at or above {OFAC_THRESHOLD}% fuzzy-match confidence threshold.",
        OFAC_FORMAL_LABEL,
        "Data As Of",
        ofac_file_date,
    )

    elements.append(Spacer(1, 18))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=1))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "This is a preliminary name-string match only. Definitive clearance requires "
        "manual SSN/DOB verification on official federal portals. Data vintage for each "
        "source database is stated above as entered by the reviewing analyst at the time "
        "of upload; it reflects the file's stated currency, not an independent verification "
        "of that date by this system.",
        footer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


# =====================================================================
# UI
# =====================================================================

st.title("🛡️ Master Federal Compliance Auditor")
st.caption("Bulk screening against OIG LEIE, SAM.gov Exclusions, and the full OFAC SDN/Consolidated name sets.")

with st.sidebar:
    st.header("📅 Data Vintage")
    st.caption("Enter the 'as of' date shown on each source file. This prints on every generated certificate.")
    oig_file_date = st.text_input("OIG LEIE File Date", placeholder="e.g., 2026-08-15")
    sam_file_date = st.text_input("SAM.gov File Date", placeholder="e.g., 2026-08-15")
    ofac_file_date = st.text_input("OFAC File Date", placeholder="e.g., 2026-08-15")

st.subheader("Source Files")
r1c1, r1c2 = st.columns(2)
with r1c1:
    oig_file = st.file_uploader("1. OIG LEIE Database (CSV)", type=["csv"])
with r1c2:
    sam_file = st.file_uploader("2. SAM.gov Exclusions Public Extract (CSV)", type=["csv"])

r2c1, r2c2 = st.columns(2)
with r2c1:
    ofac_sdn_file = st.file_uploader("3. OFAC SDN List — Primary (SDN.CSV)", type=["csv"])
with r2c2:
    ofac_alt_file = st.file_uploader("4. OFAC SDN List — Alternate (ALT.CSV)", type=["csv"])

r3c1, r3c2 = st.columns(2)
with r3c1:
    ofac_cons_prim_file = st.file_uploader("5. OFAC Consolidated List — Primary (CONS_PRIM.CSV)", type=["csv"])
with r3c2:
    ofac_cons_alt_file = st.file_uploader("6. OFAC Consolidated List — Alternate (CONS_ALT.CSV)", type=["csv"])

roster_file = st.file_uploader("7. Candidate Roster (CSV/Excel)", type=["csv", "xlsx", "xls"])

all_uploaded = all([
    oig_file, sam_file, ofac_sdn_file, ofac_alt_file,
    ofac_cons_prim_file, ofac_cons_alt_file, roster_file,
])
all_dates_entered = all([oig_file_date.strip(), sam_file_date.strip(), ofac_file_date.strip()])

st.divider()
run_audit = st.button(
    "▶️ Run Master Audit", type="primary", use_container_width=True,
    disabled=not all_uploaded,
)
if not all_uploaded:
    st.caption("All seven files are required before the audit can run.")
elif not all_dates_entered:
    st.warning(
        "One or more data-vintage dates in the sidebar are blank. The audit will still "
        "run, but generated certificates will show 'Not specified' for any missing date — "
        "fill these in for a complete compliance record."
    )

if run_audit:
    with st.spinner("Loading source databases..."):
        try:
            oig_df = load_oig(oig_file.getvalue())
            sam_df = load_sam(sam_file.getvalue())
            ofac_df = load_ofac(
                ofac_sdn_file.getvalue(),
                ofac_alt_file.getvalue(),
                ofac_cons_prim_file.getvalue(),
                ofac_cons_alt_file.getvalue(),
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if not {"firstname", "lastname"}.issubset(oig_df.columns):
            st.error("OIG file is missing expected 'firstname'/'lastname' columns after normalization.")
            st.stop()

    try:
        roster_df = load_roster(roster_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"Running batched audit across {len(roster_df)} candidates..."):
        exceptions_by_row = run_batch_audit(roster_df, oig_df, sam_df, ofac_df)

    all_exceptions = []
    cleared_candidates = []
    for idx, row in roster_df.iterrows():
        row_exceptions = exceptions_by_row.get(idx, [])
        if row_exceptions:
            for exc in row_exceptions:
                all_exceptions.append({"Candidate": row["Primary Name"], **exc})
        else:
            cleared_candidates.append((row["Primary Name"], row["Aliases"]))

    st.session_state["exceptions_df"] = pd.DataFrame(all_exceptions)
    st.session_state["cleared_candidates"] = cleared_candidates
    st.session_state["total_processed"] = len(roster_df)
    st.session_state["audit_run_at"] = datetime.now(EASTERN).strftime("%Y%m%d_%H%M%S")
    # Freeze the vintage dates used for this run so the zip stays consistent
    # even if someone edits the sidebar fields before generating certificates.
    st.session_state["oig_file_date"] = oig_file_date
    st.session_state["sam_file_date"] = sam_file_date
    st.session_state["ofac_file_date"] = ofac_file_date
    st.session_state.pop("zip_bytes", None)

# =====================================================================
# Results
# =====================================================================

if "total_processed" in st.session_state:
    total = st.session_state["total_processed"]
    cleared_candidates = st.session_state["cleared_candidates"]
    exceptions_df = st.session_state["exceptions_df"]
    flagged_count = exceptions_df["Candidate"].nunique() if not exceptions_df.empty else 0

    st.subheader("📊 Audit Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Processed", total)
    m2.metric("Cleared", len(cleared_candidates))
    m3.metric("Flagged Exceptions", flagged_count)

    st.subheader("🚩 Exceptions — Manual Review Required")
    if exceptions_df.empty:
        st.success("No exceptions in this batch.")
    else:
        st.dataframe(
            exceptions_df[["Candidate", "Matched Name", "Source", "Match Score", "Detail"]],
            use_container_width=True, hide_index=True,
        )
        st.warning(
            "Exceptions require manual verification directly on the official LEIE, "
            "SAM.gov, and OFAC portals using an identifier beyond name (e.g., DOB or SSN) "
            "before any placement decision — name-only matches, especially fuzzy OFAC hits, "
            "can be false positives."
        )

    st.subheader("📄 Preliminary Screening Reports (Cleared Candidates)")
    if cleared_candidates:
        if st.button("🏗️ Generate & Package Reports"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, aliases in cleared_candidates:
                    audit_id = str(uuid.uuid4())
                    pdf_bytes = build_report_pdf(
                        name, aliases, audit_id,
                        oig_file_date=st.session_state.get("oig_file_date", ""),
                        sam_file_date=st.session_state.get("sam_file_date", ""),
                        ofac_file_date=st.session_state.get("ofac_file_date", ""),
                    )
                    safe_name = "".join(c if c.isalnum() else "_" for c in name)
                    zf.writestr(f"{safe_name}_{audit_id}.pdf", pdf_bytes)
            zip_buffer.seek(0)
            st.session_state["zip_bytes"] = zip_buffer.read()

        if "zip_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Master Audit Reports (.zip)",
                data=st.session_state["zip_bytes"],
                file_name=f"Preliminary_Screening_Reports_{st.session_state['audit_run_at']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
    else:
        st.info("No cleared candidates to generate reports for.")
