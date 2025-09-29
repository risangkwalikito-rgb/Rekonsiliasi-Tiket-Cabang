# app.py
# -*- coding: utf-8 -*-
"""
Demo: 1 file → 2 tabel (atas & bawah) di Streamlit.
Contoh: Settlement Dana dipisah menjadi BCA VA Online vs Non-BCA.

Jalankan:
  pip install streamlit pandas openpyxl python-dateutil
  streamlit run app.py
"""

from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ---------- Utilities (ringkas & robust) ----------

def read_any(uploaded_file) -> pd.DataFrame:
    """CSV autodetect delimiter & dtype=str; Excel via openpyxl."""
    if not uploaded_file:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(
                        uploaded_file, encoding=enc,
                        sep=None, engine="python",
                        dtype=str, na_filter=False
                    )
                except Exception:
                    continue
            st.error("CSV gagal dibaca. Coba simpan ulang sebagai UTF-8.")
            return pd.DataFrame()
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()


def parse_money(val) -> float:
    """Terima 1.234.567 | 1,234.567 | 50.300,00 | (1.000) | 1.000- | IDR 1,000 CR | -2.500."""
    if val is None or (isinstance(val, float) and np.isnan(val)): return 0.0
    if isinstance(val, (int, float, np.number)): return float(val)
    s = str(val).strip()
    if not s: return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"): neg, s = True, s[1:-1].strip()
    if s.endswith("-"): neg, s = True, s[:-1].strip()
    s = re.sub(r"(idr|rp|cr|dr)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^0-9\.,\-]", "", s).strip()
    if s.startswith("-"): neg, s = True, s[1:].strip()
    last_dot, last_com = s.rfind("."), s.rfind(",")
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


def to_num(sr: pd.Series) -> pd.Series:
    return sr.apply(parse_money).astype(float)


def to_date(val) -> Optional[pd.Timestamp]:
    """String/datetime + Excel serial (days since 1899-12-30)."""
    if pd.isna(val): return None
    if isinstance(val, (int, float, np.number)):
        if np.isfinite(val) and 1 <= float(val) <= 100000:
            base = pd.Timestamp("1899-12-30")
            return (base + pd.to_timedelta(float(val), unit="D")).normalize()
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(val).normalize()
    s = str(val).strip()
    for dayfirst in (True, False):
        try:
            d = dtparser.parse(s, dayfirst=dayfirst, fuzzy=True)
            return pd.Timestamp(d.date())
        except Exception:
            continue
    return None


def idr_fmt(n: float) -> str:
    if pd.isna(n): return "-"
    neg = n < 0
    s = f"{abs(int(round(n))):,}".replace(",", ".")
    return f"({s})" if neg else s


# ---------- App ----------

st.set_page_config(page_title="1 File → 2 Tabel", layout="wide")
st.title("Contoh: 1 File → 2 Tabel di Streamlit")

uploaded = st.file_uploader("Upload Settlement (CSV/Excel)", type=["csv", "xls", "xlsx"])
go = st.button("Proses", type="primary")

if go:
    df = read_any(uploaded)
    if df.empty:
        st.warning("Data kosong / gagal dibaca.")
        st.stop()

    # Map kolom yang diperlukan (ubah jika nama berbeda)
    col_date = next((c for c in df.columns if c.lower() == "transaction date" or "transaction date" in c.lower()), None)
    col_amt  = next((c for c in df.columns if c.lower() == "settlement amount" or "settlement amount" in c.lower()), None)
    col_prod = next((c for c in df.columns if c.lower() == "product name" or "product name" in c.lower()), None)

    missing = [n for n, c in [("Transaction Date", col_date), ("Settlement Amount", col_amt), ("Product Name", col_prod)] if c is None]
    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.stop()

    # Normalisasi
    work = df.copy()
    work[col_date] = work[col_date].apply(to_date)
    work = work[~work[col_date].isna()]
    work[col_amt] = to_num(work[col_amt])
    prod_norm = work[col_prod].astype(str).str.strip().str.lower()

    # Split: BCA vs Non-BCA
    bca_mask = prod_norm.eq("bca va online")
    bca = work[bca_mask].groupby(work[bca_mask][col_date])[col_amt].sum().rename("Settlement Dana BCA").sort_index()
    nonbca = work[~bca_mask].groupby(work[~bca_mask][col_date])[col_amt].sum().rename("Settlement Dana Non BCA").sort_index()

    # Tabel 1 (atas): BCA
    st.subheader("Tabel 1 — Settlement Dana BCA (Product Name = 'BCA VA Online')")
    tbl1 = bca.reset_index().rename(columns={col_date: "Tanggal"})
    view1 = tbl1.copy()
    view1["Settlement Dana BCA"] = view1["Settlement Dana BCA"].apply(idr_fmt)
    st.dataframe(view1, use_container_width=True, hide_index=True)

    xls1 = io.BytesIO()
    with pd.ExcelWriter(xls1, engine="openpyxl") as xw:
        tbl1.to_excel(xw, index=False, sheet_name="BCA")
        view1.to_excel(xw, index=False, sheet_name="BCA_View")
    st.download_button("Unduh Tabel BCA", xls1.getvalue(), "settlement_bca.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Tabel 2 (bawah): Non-BCA
    st.subheader("Tabel 2 — Settlement Dana Non BCA (selain 'BCA VA Online')")
    tbl2 = nonbca.reset_index().rename(columns={col_date: "Tanggal"})
    view2 = tbl2.copy()
    view2["Settlement Dana Non BCA"] = view2["Settlement Dana Non BCA"].apply(idr_fmt)
    st.dataframe(view2, use_container_width=True, hide_index=True)

    xls2 = io.BytesIO()
    with pd.ExcelWriter(xls2, engine="openpyxl") as xw:
        tbl2.to_excel(xw, index=False, sheet_name="NonBCA")
        view2.to_excel(xw, index=False, sheet_name="NonBCA_View")
    st.download_button("Unduh Tabel Non-BCA", xls2.getvalue(), "settlement_nonbca.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
