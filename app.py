"""
OIG LEIE Bulk Compliance Auditor
=================================
Internal tool for RPO operations teams to bulk-screen candidate rosters
against the OIG List of Excluded Individuals/Entities (LEIE).

--------------------------------------------------------------------
DEPLOYMENT — STREAMLIT COMMUNITY CLOUD (recommended, free HTTPS link)
--------------------------------------------------------------------
1. Save this file as `app.py` in a GitHub repo.
2. Add a `requirements.txt` in the same repo with:
       streamlit
       pandas
       reportlab
       openpyxl
3. Go to https://share.streamlit.io -> "New app" -> point to your
   repo/branch, set Main file path = app.py -> Deploy.
4. You get a shareable link like https://<yourapp>.streamlit.app
   In "Settings -> Sharing", restrict access to specific emails —
   this app handles candidate PII, don't leave it public.

--------------------------------------------------------------------
DEPLOYMENT — REPLIT
--------------------------------------------------------------------
1. New Replit -> Python template -> paste this into main.py.
2. Add requirements.txt (same list as above).
3. Add a `.replit` file containing:
       run = "streamlit run main.py --server.port 8080 --server.address 0.0.0.0"
4. Click Run -> use the webview URL as your shareable link.
   For an always-on link, use Replit Deployments (paid) instead of
   the free webview, which sleeps when idle.

NOTE: Both uploaded files contain sensitive PII. Don't commit real
candidate/LEIE data to a public repo. Confirm your hosting tier's
log/data retention policy before pointing this at production data.
--------------------------------------------------------------------
"""

import io
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                 TableStyle)

st.set_page_config(page_title="OIG LEIE Bulk Compliance Auditor", layout="wide")

# =====================================================================
# Core logic
# =====================================================================

@st.cache_data(show_spinner=False)
def load_oig(file_bytes: bytes) -> pd.DataFrame:
    """Parse the OIG LEIE CSV: strip whitespace, lowercase columns."""
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_roster(uploaded_file) -> pd.DataFrame:
    """Parse the candidate roster (CSV or Excel)."""
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
    """'First Middle Last' -> (FIRST, LAST). Last token = last name."""
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0].upper(), parts[0].upper()
    return parts[0].upper(), parts[-1].upper()


def check_name_against_oig(full_name: str, oig_df: pd.DataFrame) -> list:
    """Validated exact-match logic against firstname/lastname columns."""
    if not full_name:
        return []
    first_name, last_name = split_name(full_name)
    if first_name is None:
        return []

    result = oig_df[
        (oig_df["lastname"].astype(str).str.upper() == last_name) &
        (oig_df["firstname"].astype(str).str.upper() == first_name)
    ]
    return result.to_dict("records")


def audit_candidate(primary_name: str, aliases: str, oig_df: pd.DataFrame):
    """Checks primary name + every comma-separated alias.
    Returns (is_flagged, matched_name, list_of_match_records)."""
    names_to_check = [primary_name] + [a.strip() for a in aliases.split(",") if a.strip()]
    for name in names_to_check:
        matches = check_name_against_oig(name, oig_df)
        if matches:
            return True, name, matches
    return False, None, []


def build_certificate_pdf(candidate_name: str, audit_id: str) -> bytes:
    """1-page 'Certified Clear' audit PDF for a cleared candidate."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1 * inch, bottomMargin=1 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20, spaceAfter=6,
        textColor=colors.HexColor("#1a3d63"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#555555"), spaceAfter=20,
    )
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=11, leading=16)

    elements = [
        Paragraph("Certified OIG LEIE Compliance Audit", title_style),
        Paragraph(
            "Office of Inspector General &mdash; List of Excluded Individuals/Entities Screening",
            subtitle_style,
        ),
        Spacer(1, 12),
    ]

    info_table_data = [
        ["Candidate Name:", candidate_name],
        ["Audit ID:", audit_id],
        ["Screening Date:", datetime.now().strftime("%B %d, %Y %H:%M")],
        ["Result:", "CLEARED — No Match Found in OIG LEIE Database"],
    ]
    info_table = Table(info_table_data, colWidths=[1.8 * inch, 4.2 * inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor("#1a7a3d")),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "This certifies that the individual named above was screened against the "
        "official OIG List of Excluded Individuals/Entities (LEIE) on the date "
        "indicated. No exact first-name/last-name match was identified at the time "
        "of screening. This certificate is generated for internal compliance "
        "recordkeeping and does not constitute a legal opinion.",
        body_style,
    ))
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("Generated automatically by the RPO Compliance Audit System.", subtitle_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


# =====================================================================
# UI
# =====================================================================

st.title("🛡️ OIG LEIE Bulk Compliance Auditor")
st.caption("Upload the OIG LEIE database and a candidate roster to run a bulk exclusion check.")

col1, col2 = st.columns(2)
with col1:
    oig_file = st.file_uploader("1. Upload OIG LEIE Database (CSV, ~15MB)", type=["csv"])
with col2:
    roster_file = st.file_uploader("2. Upload Candidate Roster (CSV or Excel)", type=["csv", "xlsx", "xls"])

st.divider()
run_audit = st.button(
    "▶️ Run Bulk Audit", type="primary", use_container_width=True,
    disabled=not (oig_file and roster_file),
)

if run_audit:
    with st.spinner("Loading OIG database..."):
        oig_df = load_oig(oig_file.getvalue())
        if not {"firstname", "lastname"}.issubset(oig_df.columns):
            st.error(
                "The uploaded OIG file doesn't have the expected 'firstname'/'lastname' "
                "columns after normalization. Please check the file."
            )
            st.stop()

    try:
        roster_df = load_roster(roster_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    flagged_rows = []
    cleared_names = []
    total = len(roster_df)
    progress = st.progress(0, text="Auditing candidates...")

    for i in range(total):
        record = roster_df.iloc[i]
        primary_name = record["Primary Name"]
        aliases = record["Aliases"]

        is_flagged, matched_name, matches = audit_candidate(primary_name, aliases, oig_df)
        if is_flagged:
            for m in matches:
                flagged_rows.append({
                    "Candidate": primary_name,
                    "Matched Name On File": matched_name,
                    "State": m.get("state", "N/A"),
                    "Specialty": m.get("specialty", "N/A"),
                    "Exclusion Type": m.get("excltype", "N/A"),
                    "Exclusion Date": m.get("excldate", "N/A"),
                })
        else:
            cleared_names.append(primary_name)

        progress.progress((i + 1) / total, text=f"Auditing candidates... ({i + 1}/{total})")

    progress.empty()

    st.session_state["flagged_df"] = pd.DataFrame(flagged_rows)
    st.session_state["cleared_names"] = cleared_names
    st.session_state["total_processed"] = total
    st.session_state["audit_run_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.pop("zip_bytes", None)  # invalidate any stale zip from a prior run

# =====================================================================
# Results
# =====================================================================

if "total_processed" in st.session_state:
    total = st.session_state["total_processed"]
    cleared_names = st.session_state["cleared_names"]
    flagged_df = st.session_state["flagged_df"]
    flagged_candidate_count = flagged_df["Candidate"].nunique() if not flagged_df.empty else 0

    st.subheader("📊 Audit Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Processed", total)
    m2.metric("Cleared", len(cleared_names))
    m3.metric("Flagged", flagged_candidate_count)

    st.subheader("🚩 Flagged Candidates — Manual Review Required")
    if flagged_df.empty:
        st.success("No candidates were flagged in this batch.")
    else:
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)

    st.subheader("📄 Certified Audit Certificates (Cleared Candidates)")
    if cleared_names:
        if st.button("🏗️ Generate & Package Certificates"):
            zip_buffer = io.BytesIO()
            audit_batch_id = st.session_state["audit_run_at"]
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, name in enumerate(cleared_names, start=1):
                    audit_id = f"AUD-{audit_batch_id}-{idx:04d}"
                    pdf_bytes = build_certificate_pdf(name, audit_id)
                    safe_name = "".join(c if c.isalnum() else "_" for c in name)
                    zf.writestr(f"{safe_name}_{audit_id}.pdf", pdf_bytes)
            zip_buffer.seek(0)
            st.session_state["zip_bytes"] = zip_buffer.read()

        if "zip_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Audit Certificates (.zip)",
                data=st.session_state["zip_bytes"],
                file_name=f"OIG_Audit_Certificates_{st.session_state['audit_run_at']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
    else:
        st.info("No cleared candidates to generate certificates for.")
