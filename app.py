# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi Sederhana: Tiket Detail vs Settlement Dana

Spesifikasi:
- Tiket Detail (Excel):
  Tanggal  = kolom "Action date"
  Nominal  = kolom "tarif"
  Filter   = "St Bayar" == "paid" (case-insensitive) AND "Bank" == "ESPAY"
- Settlement Dana (CSV/Excel):
  Tanggal  = kolom "Transaction Date"
  Nominal  = kolom "Settlement Amount"
- Selisih  = Tiket Detail - Settlement Dana

Jalankan:
  pip install streamlit pandas numpy openpyxl python-dateutil
  streamlit run app.py
"""

from __future__ import annotations

import io
import re
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ---------- Utilities ----------

_NON_NUM_RE = re.compile(r"[^\d\-]")

def _parse_rupiah(val) -> float:
    """Parser angka: terima 1.234.567, 1,234,567, (1.000), -2.500, dll. Gagal -> 0.0."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", ".")
    s = _NON_NUM_RE.sub("", s)
    if not s or s == "-":
        return 0.0
    try:
        num = float(s)
    except Exception:
        return 0.0
    return -num if (neg or s.startswith("-")) else num

def _to_num(sr: pd.Series) -> pd.Series:
    return sr.apply(_parse_rupiah).astype(float)

def _to_date(val) -> Optional[pd.Timestamp]:
    """Normalisasi ke tanggal (tanpa jam)."""
    if pd.isna(val):
        return None
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

def _idr_fmt(n: float) -> str:
    """Format IDR sederhana: pemisah ribuan '.' & negatif pakai ()."""
    if pd.isna(n):
        return "-"
    neg = n < 0
    s = f"{abs(int(round(n))):,}".replace(",", ".")
    return f"({s})" if neg else s

def _read_any(uploaded_file) -> pd.DataFrame:
    """Baca CSV kalau nama file berakhiran .csv, selain itu coba Excel."""
    if not uploaded_file:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
                try:
                    return pd.read_csv(uploaded_file, encoding=enc)
                except Exception:
                    uploaded_file.seek(0)
            st.error("CSV gagal dibaca. Coba simpan ulang sebagai UTF-8.")
            return pd.DataFrame()
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()

def _find_col(df: pd.DataFrame, *candidates: Iterable[str]) -> Optional[str]:
    """Cari kolom by-case-insensitive exact or relaxed contains."""
    if df.empty:
        return None
    cols = [c for c in df.columns if isinstance(c, str)]
    norm = {c.lower().strip(): c for c in cols}
    # exact (case-insensitive)
    for cands in candidates:
        for c in cands:
            key = c.lower().strip()
            if key in norm:
                return norm[key]
    # relaxed contains
    lowers = {c: o for c, o in ((c.lower(), c) for c in cols)}
    for cands in candidates:
        for c in cands:
            key = c.lower().strip()
            for lc, orig in lowers.items():
                if key in lc:
                    return orig
    return None


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber")
    tiket_file = st.file_uploader("Tiket Detail (Excel .xls/.xlsx)", type=["xls", "xlsx"])
    settle_file = st.file_uploader("Settlement Dana (CSV/Excel)", type=["csv", "xls", "xlsx"])

    st.header("2) Opsi")
    show_preview = st.checkbox("Tampilkan pratinjau data", value=False)
    go = st.button("Proses", type="primary", use_container_width=True)

# Baca data
tiket_df = _read_any(tiket_file)
settle_df = _read_any(settle_file)

if show_preview:
    st.subheader("Pratinjau")
    if not tiket_df.empty:
        st.markdown("Tiket Detail")
        st.dataframe(tiket_df.head(50), use_container_width=True)
    if not settle_df.empty:
        st.markdown("Settlement Dana")
        st.dataframe(settle_df.head(50), use_container_width=True)

# ---------- Processing ----------
if go:
    # Tiket Detail mapping fix
    t_date_col = _find_col(tiket_df, ["Action date"])
    t_nom_col  = _find_col(tiket_df, ["tarif"])
    t_stat_col = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status"])
    t_bank_col = _find_col(tiket_df, ["Bank", "Payment Channel", "channel"])

    missing_tiket = [name for name, col in [
        ("Action date", t_date_col),
        ("tarif", t_nom_col),
        ("St Bayar", t_stat_col),
        ("Bank", t_bank_col),
    ] if col is None]

    # Settlement mapping fix
    s_date_col = _find_col(settle_df, ["Transaction Date"])
    s_nom_col  = _find_col(settle_df, ["Settlement Amount"])

    missing_settle = [name for name, col in [
        ("Transaction Date", s_date_col),
        ("Settlement Amount", s_nom_col),
    ] if col is None]

    if tiket_df.empty:
        st.error("File Tiket Detail belum diupload.")
        st.stop()
    if settle_df.empty:
        st.error("File Settlement Dana belum diupload.")
        st.stop()
    if missing_tiket:
        st.error("Kolom Tiket Detail tidak ditemukan: " + ", ".join(missing_tiket))
        st.stop()
    if missing_settle:
        st.error("Kolom Settlement Dana tidak ditemukan: " + ", ".join(missing_settle))
        st.stop()

    # Tiket Detail: filter paid & ESPAY
    td = tiket_df.copy()
    td[t_date_col] = td[t_date_col].apply(_to_date)
    td = td[~td[t_date_col].isna()]

    # status bayar == paid
    td_stat = td[t_stat_col].astype(str).str.strip().str.lower()
    td = td[td_stat.eq("paid")]

    # bank == ESPAY
    td_bank = td[t_bank_col].astype(str).str.strip().str.lower()
    td = td[td_bank.eq("espay")]

    # nominal
    td[t_nom_col] = _to_num(td[t_nom_col])

    tiket_per_date = td.groupby(td[t_date_col])[[t_nom_col]].sum().squeeze()
    tiket_per_date.index = pd.to_datetime(tiket_per_date.index).date
    tiket_per_date.name = "Tiket Detail"

    # Settlement Dana
    sd = settle_df.copy()
    sd[s_date_col] = sd[s_date_col].apply(_to_date)
    sd = sd[~sd[s_date_col].isna()]
    sd[s_nom_col] = _to_num(sd[s_nom_col])

    settle_per_date = sd.groupby(sd[s_date_col])[[s_nom_col]].sum().squeeze()
    settle_per_date.index = pd.to_datetime(settle_per_date.index).date
    settle_per_date.name = "Settlement Dana"

    # Gabung & hitung selisih
    all_dates = sorted(set(tiket_per_date.index) | set(settle_per_date.index))
    final = pd.DataFrame(index=pd.Index(all_dates, name="Tanggal"))
    final["Tiket Detail"] = tiket_per_date.reindex(all_dates).fillna(0.0)
    final["Settlement Dana"] = settle_per_date.reindex(all_dates).fillna(0.0)
    final["Selisih"] = final["Tiket Detail"] - final["Settlement Dana"]

    # Render
    final_reset = final.reset_index()
    final_reset.insert(0, "No", range(1, len(final_reset) + 1))

    fmt = final_reset.copy()
    for c in ["Tiket Detail", "Settlement Dana", "Selisih"]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Unduh Excel
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        final_reset.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "Unduh Excel",
        data=bio.getvalue(),
        file_name="rekonsiliasi_tiket_vs_settlement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
