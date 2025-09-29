# app.py
# -*- coding: utf-8 -*-
"""
1 file → 2 tabel atas-bawah (Settlement: BCA VA Online vs Non-BCA)
- Parameter bulan & tahun; tanggal tabel selalu 1..akhir bulan
- CSV/Excel, parsing angka & tanggal robust
"""

from __future__ import annotations

import io
import re
import calendar
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ---------- Utilities ----------

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
            st.error("CSV gagal dibaca. Simpan ulang sebagai UTF-8.")
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


def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty: return None
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


def month_selector() -> Tuple[int, int]:
    from datetime import date
    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [
        ("01", "Januari"), ("02", "Februari"), ("03", "Maret"), ("04", "April"),
        ("05", "Mei"), ("06", "Juni"), ("07", "Juli"), ("08", "Agustus"),
        ("09", "September"), ("10", "Oktober"), ("11", "November"), ("12", "Desember"),
    ]
    c1, c2 = st.columns(2)
    with c1:
        year = st.selectbox("Tahun", years, index=years.index(today.year))
    with c2:
        sel = st.selectbox("Bulan", months, index=int(today.strftime("%m")) - 1, format_func=lambda x: x[1])
        month = int(sel[0])
    return year, month


# ---------- App ----------

st.set_page_config(page_title="1 File → 2 Tabel (Atas-Bawah)", layout="wide")
st.title("1 File → 2 Tabel (BCA VA Online di atas, Non-BCA di bawah)")

with st.sidebar:
    st.header("Upload & Parameter")
    file = st.file_uploader("Upload Settlement (CSV/Excel)", type=["csv", "xls", "xlsx"])
    y, m = month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode: {month_start.date()} s/d {month_end.date()}")
    go = st.button("Proses", type="primary", use_container_width=True)

if go:
    df = read_any(file)
    if df.empty:
        st.warning("Data kosong / gagal dibaca.")
        st.stop()

    # Map kolom
    col_date = find_col(df, ["Transaction Date"])
    col_amt  = find_col(df, ["Settlement Amount"])
    col_prod = find_col(df, ["Product Name", "Product"])
    missing = [n for n, c in [("Transaction Date", col_date), ("Settlement Amount", col_amt), ("Product Name", col_prod)] if c is None]
    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.stop()

    # Normalisasi & filter ke bulan
    work = df.copy()
    work[col_date] = work[col_date].apply(to_date)
    work = work[~work[col_date].isna()]
    work = work[(work[col_date] >= month_start) & (work[col_date] <= month_end)]
    work[col_amt] = to_num(work[col_amt])

    # Split
    prod_norm = work[col_prod].astype(str).str.strip().str.lower()
    bca_mask = prod_norm.eq("bca va online")

    # Agregasi per tanggal
    bca = work[bca_mask].groupby(work[bca_mask][col_date])[col_amt].sum().rename("Settlement Dana BCA")
    nonbca = work[~bca_mask].groupby(work[~bca_mask][col_date])[col_amt].sum().rename("Settlement Dana Non BCA")

    # Bentuk index tanggal 1..akhir bulan, reindex ke parameter
    idx = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")
    bca.index = pd.to_datetime(bca.index).date if len(bca.index) else bca.index
    nonbca.index = pd.to_datetime(nonbca.index).date if len(nonbca.index) else nonbca.index
    bca = bca.reindex(idx, fill_value=0.0)
    nonbca = nonbca.reindex(idx, fill_value=0.0)

    # ===== TABEL ATAS =====
    st.subheader("Tabel ATAS — Settlement Dana BCA (Product Name = 'BCA VA Online')")
    tbl1 = pd.DataFrame({"Tanggal": idx, "Settlement Dana BCA": bca.values})
    view1 = tbl1.copy(); view1["Settlement Dana BCA"] = view1["Settlement Dana BCA"].apply(idr_fmt)
    st.dataframe(view1, use_container_width=True, hide_index=True)

    # ===== TABEL BAWAH =====
    st.subheader("Tabel BAWAH — Settlement Dana Non BCA (selain 'BCA VA Online')")
    tbl2 = pd.DataFrame({"Tanggal": idx, "Settlement Dana Non BCA": nonbca.values})
    view2 = tbl2.copy(); view2["Settlement Dana Non BCA"] = view2["Settlement Dana Non BCA"].apply(idr_fmt)
    st.dataframe(view2, use_container_width=True, hide_index=True)

    # Unduh Excel (kedua tabel)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        tbl1.to_excel(xw, index=False, sheet_name="BCA")
        view1.to_excel(xw, index=False, sheet_name="BCA_View")
        tbl2.to_excel(xw, index=False, sheet_name="NonBCA")
        view2.to_excel(xw, index=False, sheet_name="NonBCA_View")
    st.download_button(
        "Unduh Excel (kedua tabel)",
        data=out.getvalue(),
        file_name=f"settlement_bca_nonbca_{y}-{m:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
