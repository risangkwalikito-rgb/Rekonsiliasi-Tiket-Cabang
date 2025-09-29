# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana
- Multi-file upload untuk kedua sumber
- Parser uang/tanggal robust (format Eropa & serial Excel)
- Filter Tiket: St Bayar='paid' & Bank='ESPAY'
- Kolom Tanggal pada hasil SELALU mengikuti rentang tanggal parameter
"""

from __future__ import annotations

import io
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ---------------- Utilities ----------------

def _parse_money(val) -> float:
    """Terima 1.234.567 | 1,234.567 | 50.300,00 | (1.000) | 1.000- | IDR 1,000 CR | -2.500."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.number)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.endswith("-"):
        neg, s = True, s[:-1].strip()

    s = re.sub(r"(idr|rp|cr|dr)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^0-9\.,\-]", "", s).strip()

    if s.startswith("-"):
        neg, s = True, s[1:].strip()

    last_dot = s.rfind(".")
    last_com = s.rfind(",")

    if last_dot == -1 and last_com == -1:
        num_s = s
    elif last_dot > last_com:
        num_s = s.replace(",", "")
    else:
        num_s = s.replace(".", "").replace(",", ".")

    try:
        num = float(num_s)
    except Exception:
        num_s = s.replace(".", "").replace(",", "")
        num = float(num_s) if num_s else 0.0

    return -num if neg else num


def _to_num(sr: pd.Series) -> pd.Series:
    return sr.apply(_parse_money).astype(float)


def _to_date(val) -> Optional[pd.Timestamp]:
    """Parse string/datetime + Excel serial (days since 1899-12-30)."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, np.number)):
        if not np.isfinite(val):
            return None
        if 1 <= float(val) <= 100000:
            try:
                base = pd.Timestamp("1899-12-30")
                return (base + pd.to_timedelta(float(val), unit="D")).normalize()
            except Exception:
                pass
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(val).normalize()
    s = str(val).strip()
    if not s:
        return None
    for dayfirst in (True, False):
        try:
            d = dtparser.parse(s, dayfirst=dayfirst, fuzzy=True)
            return pd.Timestamp(d.date())
        except Exception:
            continue
    return None


def _read_any(uploaded_file) -> pd.DataFrame:
    """CSV: autodetect delimiter & read as text; Excel via openpyxl."""
    if not uploaded_file:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(
                        uploaded_file,
                        encoding=enc,
                        sep=None,
                        engine="python",
                        dtype=str,
                        na_filter=False,
                    )
                except Exception:
                    continue
            st.error(f"CSV gagal dibaca: {uploaded_file.name}. Simpan ulang sebagai UTF-8.")
            return pd.DataFrame()
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca {uploaded_file.name}: {e}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty:
        return None
    cols = [c for c in df.columns if isinstance(c, str)]
    m = {c.lower().strip(): c for c in cols}
    for n in names:
        if n.lower().strip() in m:
            return m[n.lower().strip()]
    for n in names:
        key = n.lower().strip()
        for c in cols:
            if key in c.lower():
                return c
    return None


def _idr_fmt(n: float) -> str:
    if pd.isna(n):
        return "-"
    neg = n < 0
    s = f"{abs(int(round(n))):,}".replace(",", ".")
    return f"({s})" if neg else s


def _concat_files(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = _read_any(f)
        if not df.empty:
            df["__source__"] = f.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _get_date_range_input() -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Ambil nilai date_input yang kadang bisa tuple/list/single date."""
    val = st.date_input("Periode", value=None)
    start = end = None
    if isinstance(val, tuple) and len(val) == 2:
        start, end = val
    elif isinstance(val, list) and len(val) == 2:
        start, end = val
    # Convert to Timestamp
    if start and end:
        return pd.Timestamp(start), pd.Timestamp(end)
    return None, None


# ---------------- App ----------------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    tiket_files = st.file_uploader(
        "Tiket Detail (Excel .xls/.xlsx)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
    )
    settle_files = st.file_uploader(
        "Settlement Dana (CSV/Excel)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
    )

    st.header("2) Parameter Tanggal (WAJIB)")
    st.caption("Kolom Tanggal pada hasil akan mengikuti rentang ini.")
    start_date, end_date = _get_date_range_input()

    st.header("3) Opsi")
    show_preview = st.checkbox("Tampilkan pratinjau", value=False)
    show_debug = st.checkbox("Debug parsing", value=False)
    go = st.button("Proses", type="primary", use_container_width=True)

tiket_df = _concat_files(tiket_files)
settle_df = _concat_files(settle_files)

if show_preview:
    st.subheader("Pratinjau")
    if not tiket_df.empty:
        st.markdown(f"Tiket Detail (rows: {len(tiket_df)})"); st.dataframe(tiket_df.head(50), use_container_width=True)
    if not settle_df.empty:
        st.markdown(f"Settlement Dana (rows: {len(settle_df)})"); st.dataframe(settle_df.head(50), use_container_width=True)

if go:
    # Validasi tanggal
    if not (start_date and end_date and start_date <= end_date):
        st.error("Mohon pilih rentang tanggal yang valid (Start ≤ End).")
        st.stop()

    # Mapping tetap sesuai spesifikasi
    t_date = _find_col(tiket_df, ["Action date"])
    t_amt  = _find_col(tiket_df, ["tarif"])
    t_stat = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status"])
    t_bank = _find_col(tiket_df, ["Bank", "Payment Channel", "channel"])

    s_date = _find_col(settle_df, ["Transaction Date"])
    s_amt  = _find_col(settle_df, ["Settlement Amount"])

    missing = []
    for name, col, src in [
        ("Action date", t_date, "Tiket Detail"),
        ("tarif", t_amt, "Tiket Detail"),
        ("St Bayar", t_stat, "Tiket Detail"),
        ("Bank", t_bank, "Tiket Detail"),
        ("Transaction Date", s_date, "Settlement Dana"),
        ("Settlement Amount", s_amt, "Settlement Dana"),
    ]:
        if col is None:
            missing.append(f"{src}: {name}")
    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing))
        st.stop()

    # --- Tiket Detail ---
    td = tiket_df.copy()
    td[t_date] = td[t_date].apply(_to_date)
    td = td[~td[t_date].isna()]
    # filter status/bank
    td_stat = td[t_stat].astype(str).str.strip().str.lower()
    td_bank = td[t_bank].astype(str).str.strip().str.lower()
    td = td[td_stat.eq("paid") & td_bank.eq("espay")]
    # filter rentang tanggal PARAMETER
    td = td[(td[t_date] >= start_date) & (td[t_date] <= end_date)]
    td[t_amt] = _to_num(td[t_amt])
    tiket_by_date = td.groupby(td[t_date])[t_amt].sum()

    # --- Settlement Dana ---
    sd = settle_df.copy()
    sd[s_date] = sd[s_date].apply(_to_date)
    sd = sd[~sd[s_date].isna()]
    # filter rentang tanggal PARAMETER
    sd = sd[(sd[s_date] >= start_date) & (sd[s_date] <= end_date)]
    if show_debug:
        st.info(f"Contoh Settlement Amount (raw → parsed): {sd[s_amt].head(3).tolist()} → {_to_num(sd[s_amt].head(3)).tolist()}")
    sd[s_amt] = _to_num(sd[s_amt])
    settle_by_date = sd.groupby(sd[s_date])[s_amt].sum()

    # --- Bangun kolom Tanggal dari PARAMETER ---
    all_dates = [d.date() for d in pd.date_range(start_date, end_date, freq="D")]
    idx = pd.Index(all_dates, name="Tanggal")

    # Reindex ke tanggal parameter
    tiket_by_date.index = pd.to_datetime(tiket_by_date.index).date
    settle_by_date.index = pd.to_datetime(settle_by_date.index).date
    tiket_series = tiket_by_date.reindex(idx, fill_value=0.0)
    settle_series = settle_by_date.reindex(idx, fill_value=0.0)

    final = pd.DataFrame(index=idx)
    final["Tiket Detail"] = tiket_series.values
    final["Settlement Dana"] = settle_series.values
    final["Selisih"] = final["Tiket Detail"] - final["Settlement Dana"]

    # View + total
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    total_row = pd.DataFrame(
        [{"No": "", "Tanggal": "TOTAL",
          "Tiket Detail": final["Tiket Detail"].sum(),
          "Settlement Dana": final["Settlement Dana"].sum(),
          "Selisih": final["Selisih"].sum()}]
    )
    view_total = pd.concat([view, total_row], ignore_index=True)

    fmt = view_total.copy()
    for c in ["Tiket Detail", "Settlement Dana", "Selisih"]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Export
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        view_total.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "Unduh Excel",
        data=bio.getvalue(),
        file_name="rekonsiliasi_tiket_vs_settlement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
