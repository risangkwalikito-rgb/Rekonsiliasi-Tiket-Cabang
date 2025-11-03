# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana
- Parameter tanggal = Bulan & Tahun; hasil selalu 1..akhir bulan itu
- Multi-file upload (Tiket Excel; Settlement CSV/Excel)
- Parser uang/tanggal robust (format Eropa & serial Excel)
- Tiket difilter: St Bayar mengandung 'paid/success/sukses/settled/lunas' & Bank mengandung 'ESPAY'
- Settlement Dana: KEMBALI seperti semula (Transaction Date + Settlement Amount/L)
- Tambahan kolom:
    * Settlement BCA:    Amount=L, tanggal=Settlement Date(E), Product Name(P) == "BCA VA Online"
    * Settlement Non BCA:Amount=L, tanggal=Settlement Date(E), Product Name(P) != "BCA VA Online"
    * Total Settlement:  Settlement BCA + Settlement Non BCA
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

def _parse_money(val) -> float:
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
    last_dot = s.rfind("."); last_com = s.rfind(",")
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

_ddmmyyyy = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")
def _to_date(val) -> Optional[pd.Timestamp]:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, np.number)):
        if not np.isfinite(val): return None
        if 1 <= float(val) <= 100000:
            try:
                base = pd.Timestamp("1899-12-30")
                return (base + pd.to_timedelta(float(val), unit="D")).normalize()
            except Exception:
                pass
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        try: return pd.to_datetime(val).normalize()
        except Exception: return None
    s = str(val).strip()
    if not s: return None
    m = _ddmmyyyy.search(s)
    if m:
        d, M, y = m.groups()
        if len(y) == 2: y = "20" + y
        try: return pd.Timestamp(year=int(y), month=int(M), day=int(d))
        except Exception: pass
    for dayfirst in (True, False):
        try:
            d = dtparser.parse(s, dayfirst=dayfirst, fuzzy=True)
            return pd.Timestamp(d.date())
        except Exception:
            continue
    return None

def _read_any(uploaded_file) -> pd.DataFrame:
    if not uploaded_file: return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig","utf-8","cp1252","iso-8859-1"):
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python", dtype=str, na_filter=False)
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
    if df.empty: return None
    cols = [c for c in df.columns jika isinstance(c, str)]
    m = {c.lower().strip().lstrip("\ufeff"): c for c in cols}
    for n in names:
        key = n.lower().strip()
        if key in m: return m[key]
    for n in names:
        key = n.lower().strip()
        for c in cols:
            if key in c.lower(): return c
    return None

def _idr_fmt(n: float) -> str:
    if pd.isna(n): return "-"
    neg = n < 0
    s = f"{abs(int(round(n))):,}".replace(",", ".")
    return f"({s})" if neg else s

def _concat_files(files) -> pd.DataFrame:
    if not files: return pd.DataFrame()
    frames = []
    for f in files:
        df = _read_any(f)
        if not df.empty:
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            df["__source__"] = f.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _month_selector() -> Tuple[int, int]:
    from datetime import date
    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [("01","Januari"),("02","Februari"),("03","Maret"),("04","April"),
              ("05","Mei"),("06","Juni"),("07","Juli"),("08","Agustus"),
              ("09","September"),("10","Oktober"),("11","November"),("12","Desember")]
    col1, col2 = st.columns(2)
    with col1: year = st.selectbox("Tahun", years, index=years.index(today.year))
    with col2:
        month_label = st.selectbox("Bulan", months, index=int(today.strftime("%m"))-1, format_func=lambda x: x[1])
        month = int(month_label[0])
    return year, month

# ---------- App ----------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    tiket_files = st.file_uploader("Tiket Detail (Excel .xls/.xlsx)", type=["xls","xlsx"], accept_multiple_files=True)
    settle_files = st.file_uploader("Settlement Dana (CSV/Excel)", type=["csv","xls","xlsx"], accept_multiple_files=True)

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end   = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode dipakai: {month_start.date()} s/d {month_end.date()}")

    st.header("3) Opsi")
    show_preview = st.checkbox("Tampilkan pratinjau", value=False)
    show_debug   = st.checkbox("Debug parsing", value=False)
    go = st.button("Proses", type="primary", use_container_width=True)

tiket_df  = _concat_files(tiket_files)
settle_df = _concat_files(settle_files)

if show_preview:
    st.subheader("Pratinjau")
    if not tiket_df.empty:
        st.markdown(f"Tiket Detail (rows: {len(tiket_df)})"); st.dataframe(tiket_df.head(50), use_container_width=True)
    if not settle_df.empty:
        st.markdown(f"Settlement Dana (rows: {len(settle_df)})"); st.dataframe(settle_df.head(50), use_container_width=True)

if go:
    # --- Tiket: tanggal prioritas Action/Action Date ---
    t_date = _find_col(tiket_df, [
        "Action/Action Date","Action Date","Action","Action date",
        "Paid Date","Payment Date","Tanggal Bayar","Tanggal"
    ])
    t_amt  = _find_col(tiket_df, ["tarif","amount","nominal","jumlah","total"])
    t_stat = _find_col(tiket_df, ["St Bayar","Status Bayar","status","status bayar"])
    t_bank = _find_col(tiket_df, ["Bank","Payment Channel","channel","payment method"])

    # --- Settlement Dana (SEPERTI SEMULA): Transaction Date + Settlement Amount ---
    s_date_legacy = _find_col(settle_df, ["Transaction Date","Tanggal Transaksi","Tanggal"])
    s_amt_legacy  = _find_col(settle_df, ["Settlement Amount","Amount","Nominal","Jumlah"])
    if s_amt_legacy is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_legacy = settle_df.columns[11]  # fallback kolom L
    if s_date_legacy is None:
        s_date_legacy = _find_col(settle_df, ["Settlement Date","Tanggal Settlement","Settle Date","Tanggal"])
        if s_date_legacy is None and not settle_df.empty and len(settle_df.columns) >= 5:
            s_date_legacy = settle_df.columns[4]  # fallback kolom E

    # --- Untuk BCA/Non-BCA: wajib E (Settlement Date) + L (Settlement Amount) + P (Product Name) ---
    s_date_E = _find_col(settle_df, ["Settlement Date","Tanggal Settlement","Settle Date","Tanggal"])
    s_amt_L  = _find_col(settle_df, ["Settlement Amount","Amount","Nominal","Jumlah"])
    if s_amt_L is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_L = settle_df.columns[11]  # kolom L
    s_prod_P = _find_col(settle_df, ["Product Name","Produk","Nama Produk"])
    if s_date_E is None and not settle_df.empty and len(settle_df.columns) >= 5:
        s_date_E = settle_df.columns[4]   # kolom E
    if s_prod_P is None and not settle_df.empty and len(settle_df.columns) >= 16:
        s_prod_P = settle_df.columns[15]  # kolom P

    # --- Validasi minimal kolom ---
    missing = []
    for name, col, src in [
        ("Action/Action Date", t_date, "Tiket Detail"),
        ("tarif/amount", t_amt, "Tiket Detail"),
        ("St Bayar/Status", t_stat, "Tiket Detail"),
        ("Bank/Channel", t_bank, "Tiket Detail"),
        ("Transaction Date/Tanggal Transaksi", s_date_legacy, "Settlement Dana (utama)"),
        ("Settlement Amount/L", s_amt_legacy, "Settlement Dana (utama)"),
        ("Settlement Date/E", s_date_E, "BCA/Non-BCA"),
        ("Settlement Amount/L", s_amt_L, "BCA/Non-BCA"),
        ("Product Name/P", s_prod_P, "BCA/Non-BCA"),
    ]:
        if col is None: missing.append(f"{src}: {name}")
    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing))
        st.stop()

    # --- Tiket Detail ---
    td = tiket_df.copy()
    td[t_date] = td[t_date].apply(_to_date)
    td = td[~td[t_date].isna()]
    td_stat_norm = td[t_stat].astype(str).str.strip().str.lower()
    td_bank_norm = td[t_bank].astype(str).str.strip().str.lower()
    paid_mask  = td_stat_norm.str.contains(r"\bpaid\b|\bsuccess\b|sukses|settled|lunas", na=False)
    espay_mask = td_bank_norm.str.contains("espay", na=False)
    td = td[paid_mask & espay_mask]
    td = td[(td[t_date] >= month_start) & (td[t_date] <= month_end)]
    td[t_amt] = _to_num(td[t_amt])
    tiket_by_date = td.groupby(td[t_date].dt.date, dropna=True)[t_amt].sum()

    # --- Settlement Dana (SEPERTI SEMULA) ---
    sd_main = settle_df.copy()
    sd_main[s_date_legacy] = sd_main[s_date_legacy].apply(_to_date)
    sd_main = sd_main[~sd_main[s_date_legacy].isna()]
    sd_main = sd_main[(sd_main[s_date_legacy] >= month_start) & (sd_main[s_date_legacy] <= month_end)]
    sd_main[s_amt_legacy] = _to_num(sd_main[s_amt_legacy])
    settle_by_date_total = sd_main.groupby(sd_main[s_date_legacy].dt.date, dropna=True)[s_amt_legacy].sum()

    # --- Settlement BCA / Non BCA (berdasar E,L,P) ---
    sd_bca = settle_df.copy()
    sd_bca[s_date_E] = sd_bca[s_date_E].apply(_to_date)
    sd_bca = sd_bca[~sd_bca[s_date_E].isna()]
    sd_bca = sd_bca[(sd_bca[s_date_E] >= month_start) & (sd_bca[s_date_E] <= month_end)]
    sd_bca[s_amt_L] = _to_num(sd_bca[s_amt_L])

    prod_norm = sd_bca[s_prod_P].astype(str).str.strip().str.casefold()
    bca_mask  = (prod_norm == "bca va online".casefold())

    settle_by_date_bca     = sd_bca.loc[bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()
    settle_by_date_non_bca = sd_bca.loc[~bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()

    # --- Index tanggal (1..akhir bulan) ---
    idx = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")
    tiket_series   = tiket_by_date.reindex(idx, fill_value=0.0)
    settle_series  = settle_by_date_total.reindex(idx, fill_value=0.0)     # Settlement Dana (legacy)
    bca_series     = settle_by_date_bca.reindex(idx, fill_value=0.0)
    non_bca_series = settle_by_date_non_bca.reindex(idx, fill_value=0.0)
    total_settle_series = bca_series + non_bca_series                      # ← kolom baru

    # --- Final table ---
    final = pd.DataFrame(index=idx)
    final["Tiket Detail ESPAY"] = tiket_series.values
    final["Settlement Dana"]    = settle_series.values
    final["Selisih"]            = final["Tiket Detail ESPAY"] - final["Settlement Dana"]
    final["Settlement BCA"]     = bca_series.values
    final["Settlement Non BCA"] = non_bca_series.values
    final["Total Settlement"]   = total_settle_series.values               # ← kolom baru

    # View + total
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    total_row = pd.DataFrame([{
        "No": "", "Tanggal": "TOTAL",
        "Tiket Detail ESPAY": final["Tiket Detail ESPAY"].sum(),
        "Settlement Dana":    final["Settlement Dana"].sum(),
        "Selisih":            final["Selisih"].sum(),
        "Settlement BCA":     final["Settlement BCA"].sum(),
        "Settlement Non BCA": final["Settlement Non BCA"].sum(),
        "Total Settlement":   final["Total Settlement"].sum(),             # ← total baru
    }])
    view_total = pd.concat([view, total_row], ignore_index=True)

    # Format tampilan uang
    fmt = view_total.copy()
    for c in ["Tiket Detail ESPAY","Settlement Dana","Selisih","Settlement BCA","Settlement Non BCA","Total Settlement"]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Debug info (opsional)
    if show_debug:
        st.caption(f"Settlement Dana pakai: Tgl=**{s_date_legacy}**, Amount=**{s_amt_legacy}** (legacy)")
        st.caption(f"BCA/Non-BCA pakai: Tgl=**{s_date_E}** (E), Amount=**{s_amt_L}** (L), Product=**{s_prod_P}** (P)")

    # Export
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        view_total.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "Unduh Excel",
        data=bio.getvalue(),
        file_name=f"rekonsiliasi_{y}-{m:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
