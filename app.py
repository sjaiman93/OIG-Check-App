"""
Master Federal Compliance Auditor
==================================
Bulk screens a candidate roster against OIG LEIE, SAM.gov Exclusions,
and OFAC (SDN + Consolidated Non-SDN) lists, and issues a certified
audit PDF per cleared candidate.

--------------------------------------------------------------------
DEPLOYMENT — STREAMLIT COMMUNITY CLOUD
--------------------------------------------------------------------
1. Save as `app.py` in a GitHub repo.
2. requirements.txt:
       streamlit
       pandas
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


@st.cache_data(show_spinner=False)
def load_ofac(sdn_bytes: bytes, consolidated_bytes: bytes) -> pd.DataFrame:
    """Combine SDN + Consolidated lists into one lookup frame with a
    single normalized 'ofac_name' column, tagged by source list."""
    sdn = pd.read_csv(io.BytesIO(sdn_bytes), dtype=str, low_memory=False)
    con = pd.read_csv(io.BytesIO(consolidated_bytes), dtype=str, low_memory=False)

    sdn.columns = [c.strip().lower() for c in sdn.columns]
    con.columns = [c.strip().lower() for c in con.columns]

    name_candidates = ["name", "sdn_name", "primary_name", "entity_name", "full_name"]

    def find_name_col(df):
        col = next((c for c in name_candidates if c in df.columns), None)
        if col is None:
            # fall back to the first object/string-like column
            col = df.columns[0]
        return col

    sdn_name_col = find_name_col(sdn)
    con_name_col = find_name_col(con)

    sdn_out = pd.DataFrame({
        "ofac_name": sdn[sdn_name_col].astype(str).str.strip(),
        "ofac_source": "OFAC SDN",
    })
    con_out = pd.DataFrame({
        "ofac_name": con[con_name_col].astype(str).str.strip(),
        "ofac_source": "OFAC Consolidated Non-SDN",
    })

    combined = pd.concat([sdn_out, con_out], ignore_index=True)
    combined = combined[combined["ofac_name"].str.len() > 0].reset_index(drop=True)
    return combined


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
# Matching logic
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


def check_ofac(name: str, ofac_df: pd.DataFrame, choices: list):
    """rapidfuzz fuzzy match against combined OFAC name list, threshold 85."""
    if not name or not choices:
        return None
    match = process.extractOne(name.upper(), choices, scorer=fuzz.token_sort_ratio)
    if match is None:
        return None
    matched_text, score, idx = match
    if score >= OFAC_THRESHOLD:
        row = ofac_df.iloc[idx]
        return {
            "matched_name": row["ofac_name"],
            "source": row["ofac_source"],
            "score": round(score, 1),
        }
    return None


def audit_candidate(primary_name: str, aliases: str, oig_df, sam_df, ofac_df, ofac_choices):
    """Checks primary + every alias against all three sources.
    Returns a list of exception dicts (empty list = cleared)."""
    names_to_check = [primary_name] + [a.strip() for a in aliases.split(",") if a.strip()]
    exceptions = []

    for name in names_to_check:
        if not name:
            continue

        for rec in check_oig(name, oig_df):
            exceptions.append({
                "Matched Name": name,
                "Source": "OIG LEIE",
                "Match Score": "Exact",
                "Detail": f"{rec.get('excltype', 'N/A')} / {rec.get('state', 'N/A')}",
            })

        for rec in check_sam(name, sam_df):
            exceptions.append({
                "Matched Name": name,
                "Source": "SAM.gov",
                "Match Score": "Exact",
                "Detail": "SAM.gov Exclusions Public Extract",
            })

        ofac_hit = check_ofac(name, ofac_df, ofac_choices)
        if ofac_hit:
            exceptions.append({
                "Matched Name": name,
                "Source": ofac_hit["source"],
                "Match Score": f"{ofac_hit['score']}%",
                "Detail": f"Fuzzy match to '{ofac_hit['matched_name']}'",
            })

    return exceptions


# =====================================================================
# PDF certificate
# =====================================================================

def build_certificate_pdf(candidate_name: str, aliases: str, audit_id: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=4,
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
    footer_style = ParagraphStyle(
        "FooterStyle", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=colors.HexColor("#666666"),
    )

    now_eastern = datetime.now(EASTERN).strftime("%B %d, %Y %I:%M %p %Z")
    aliases_display = aliases if aliases else "None provided"

    elements = [
        Paragraph("Certified Master Federal Compliance Audit", title_style),
        Paragraph("Internal RPO Compliance System &mdash; Automated Screening Record", subtitle_style),
        HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=1),
        Spacer(1, 10),
    ]

    candidate_table = Table([
        ["Candidate Name:", candidate_name],
        ["Aliases Screened:", aliases_display],
        ["Audit Timestamp:", now_eastern],
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

    def db_section(title, status_line, source_line):
        return [
            Paragraph(title, section_style),
            Paragraph(f"<b>Status:</b> {status_line}", body_style),
            Paragraph(f"<b>Source:</b> {source_line}", body_style),
        ]

    elements += db_section(
        "Database 1: OIG LEIE",
        "CLEARED &mdash; No exact first/last name match found.",
        "OIG LEIE Database",
    )
    elements += db_section(
        "Database 2: SAM.gov Exclusions",
        "CLEARED &mdash; No exact first/last name match found.",
        "SAM Exclusions Public Extract V2",
    )
    elements += db_section(
        "Database 3: OFAC Sanctions (SDN &amp; Consolidated)",
        f"CLEARED &mdash; No matches at or above {OFAC_THRESHOLD}% fuzzy-match confidence threshold.",
        "OFAC SDN List &amp; OFAC Consolidated Non-SDN List",
    )

    elements.append(Spacer(1, 18))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=1))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "This certificate is generated automatically by the internal RPO Compliance "
        "System for internal recordkeeping purposes only. Screening is performed by "
        "automated name-matching against the source lists identified above at the "
        "timestamp shown; it does not include Social Security Number, Date of Birth, "
        "or NPI-level verification and does not constitute a legal determination of "
        "exclusion or sanctions status. Any candidate flagged as an exception requires "
        "manual review and independent verification directly on the official federal "
        "portals (LEIE, SAM.gov, and OFAC) prior to placement, using identifiers beyond "
        "name alone.",
        footer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


# =====================================================================
# UI
# =====================================================================

st.title("🛡️ Master Federal Compliance Auditor")
st.caption("Bulk screening against OIG LEIE, SAM.gov Exclusions, and OFAC SDN / Consolidated lists.")

c1, c2, c3 = st.columns(3)
with c1:
    oig_file = st.file_uploader("1. OIG LEIE Database (CSV)", type=["csv"])
with c2:
    sam_file = st.file_uploader("2. SAM.gov Exclusions Extract (CSV)", type=["csv"])
with c3:
    roster_file = st.file_uploader("5. Candidate Roster (CSV/Excel)", type=["csv", "xlsx", "xls"])

c4, c5 = st.columns(2)
with c4:
    ofac_sdn_file = st.file_uploader("3. OFAC SDN List (CSV)", type=["csv"])
with c5:
    ofac_con_file = st.file_uploader("4. OFAC Consolidated Non-SDN List (CSV)", type=["csv"])

all_uploaded = all([oig_file, sam_file, ofac_sdn_file, ofac_con_file, roster_file])

st.divider()
run_audit = st.button(
    "▶️ Run Master Audit", type="primary", use_container_width=True,
    disabled=not all_uploaded,
)
if not all_uploaded:
    st.caption("All five files are required before the audit can run.")

if run_audit:
    with st.spinner("Loading source databases..."):
        try:
            oig_df = load_oig(oig_file.getvalue())
            sam_df = load_sam(sam_file.getvalue())
            ofac_df = load_ofac(ofac_sdn_file.getvalue(), ofac_con_file.getvalue())
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if not {"firstname", "lastname"}.issubset(oig_df.columns):
            st.error("OIG file is missing expected 'firstname'/'lastname' columns after normalization.")
            st.stop()

        ofac_choices = ofac_df["ofac_name"].str.upper().tolist()

    try:
        roster_df = load_roster(roster_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    all_exceptions = []
    cleared_candidates = []  # (primary_name, aliases)
    total = len(roster_df)
    progress = st.progress(0, text="Running master audit...")

    for i in range(total):
        record = roster_df.iloc[i]
        primary_name = record["Primary Name"]
        aliases = record["Aliases"]

        exceptions = audit_candidate(primary_name, aliases, oig_df, sam_df, ofac_df, ofac_choices)

        if exceptions:
            for exc in exceptions:
                all_exceptions.append({"Candidate": primary_name, **exc})
        else:
            cleared_candidates.append((primary_name, aliases))

        progress.progress((i + 1) / total, text=f"Running master audit... ({i + 1}/{total})")

    progress.empty()

    st.session_state["exceptions_df"] = pd.DataFrame(all_exceptions)
    st.session_state["cleared_candidates"] = cleared_candidates
    st.session_state["total_processed"] = total
    st.session_state["audit_run_at"] = datetime.now(EASTERN).strftime("%Y%m%d_%H%M%S")
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

    st.subheader("📄 Certified Master Audit Certificates (Cleared Candidates)")
    if cleared_candidates:
        if st.button("🏗️ Generate & Package Certificates"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, aliases in cleared_candidates:
                    audit_id = str(uuid.uuid4())
                    pdf_bytes = build_certificate_pdf(name, aliases, audit_id)
                    safe_name = "".join(c if c.isalnum() else "_" for c in name)
                    zf.writestr(f"{safe_name}_{audit_id}.pdf", pdf_bytes)
            zip_buffer.seek(0)
            st.session_state["zip_bytes"] = zip_buffer.read()

        if "zip_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Master Audit Certificates (.zip)",
                data=st.session_state["zip_bytes"],
                file_name=f"Master_Federal_Compliance_Audit_{st.session_state['audit_run_at']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
    else:
        st.info("No cleared candidates to generate certificates for.")
