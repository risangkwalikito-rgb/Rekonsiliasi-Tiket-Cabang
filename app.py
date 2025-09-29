# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana
- Bulan & Tahun → tanggal 1..akhir bulan
- Multi-file upload (Tiket Excel; Settlement CSV/Excel)
- Tiket: St Bayar='paid' & Bank='ESPAY'
- Split Settlement: BCA (Product Name persis) vs Non-BCA
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

# ---------------- Utilities ----------------

def _parse_money(val) -> float:
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
    if last_dot == -1 and last_com == -1: num_s = s
    elif last_dot > last_com:             num_s = s.replace(",", "")
    else:                                  num_s = s.replace(".", "").replace(",", ".")
    try: num = float(num_s)
    except Exception:
        num_s = s.replace(".", "").replace(",", ""); num = float(num_s) if num_s else 0.0
    return -num if neg else num

def _to_num(sr: pd.Series) -> pd.Series: return sr.apply(_parse_money).astype(float)

def _to_date(val) -> Optional[pd.Timestamp]:
    if pd.isna(val): return None
    if isinstance(val, (int, float, np.number)):
        if np.isfinite(val) and 1 <= float(val) <= 100000:
            base = pd.Timestamp("1899-12-30"); return (base + pd.to_timedelta(float(val), unit="D")).normalize()
        return None
    if isinstance(val, (pd.Timestamp, np.datetime64)): return pd.to_datetime(val).normalize()
    s = str(val).strip()
    for dayfirst in (True, False):
        try: d = dtparser.parse(s, dayfirst=dayfirst, fuzzy=True); return pd.Timestamp(d.date())
        except Exception: continue
    return None

def _read_any(uploaded_file) -> pd.DataFrame:
    if not uploaded_file: return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python", dtype=str, na_filter=False)
                except Exception: continue
            st.error(f"CSV gagal dibaca: {uploaded_file.name}."); return pd.DataFrame()
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca {uploaded_file.name}: {e}"); return pd.DataFrame()

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty: return None
    cols = [c for c in df.columns if isinstance(c, str)]
    norm = {c.lower().strip(): c for c in cols}
    for n in names:
        k = n.lower().strip()
        if k in norm: return norm[k]
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
    c1, c2 = st.columns(2)
    with c1: year = st.selectbox("Tahun", years, index=years.index(today.year))
    with c2: sel = st.selectbox("Bulan", months, index=int(today.strftime("%m"))-1, format_func=lambda x: x[1]); month = int(sel[0])
    return year, month

def _norm_label(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)   # kompres spasi
    return s

# ---------------- App ----------------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    tiket_files  = st.file_uploader("Tiket Detail (Excel .xls/.xlsx)", type=["xls","xlsx"], accept_multiple_files=True)
    settle_files = st.file_uploader("Settlement Dana (CSV/Excel)", type=["csv","xls","xlsx"], accept_multiple_files=True)

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end   = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode: {month_start.date()} s/d {month_end.date()}")

    st.header("3) Ketentuan BCA")
    bca_exact = st.text_input(
        "Nama persis Product Name untuk BCA",
        value="BCA VA Online",
        help="Cocok persis (case-insensitive, spasi di-normalisasi). Contoh: 'BCA VA Online'."
    )

    st.header("4) Opsi")
    show_preview = st.checkbox("Tampilkan pratinjau", value=False)
    show_debug   = st.checkbox("Debug Product Name", value=False)
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
    # Kolom yang dipakai
    t_date = _find_col(tiket_df,  ["Action date"])
    t_amt  = _find_col(tiket_df,  ["tarif"])
    t_stat = _find_col(tiket_df,  ["St Bayar","Status Bayar","status"])
    t_bank = _find_col(tiket_df,  ["Bank","Payment Channel","channel"])

    s_date = _find_col(settle_df, ["Transaction Date"])
    s_amt  = _find_col(settle_df, ["Settlement Amount"])
    s_prod = _find_col(settle_df, ["Product Name","Product"])

    missing = []
    for name, col, src in [
        ("Action date", t_date, "Tiket Detail"),
        ("tarif", t_amt, "Tiket Detail"),
        ("St Bayar", t_stat, "Tiket Detail"),
        ("Bank",  t_bank, "Tiket Detail"),
        ("Transaction Date", s_date, "Settlement Dana"),
        ("Settlement Amount", s_amt, "Settlement Dana"),
    ]:
        if col is None: missing.append(f"{src}: {name}")
    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing)); st.stop()
    if s_prod is None:
        st.warning("Kolom 'Product Name' tidak ditemukan. Kolom BCA/Non-BCA akan 0.")

    # Tiket Detail (paid & ESPAY)
    td = tiket_df.copy()
    td[t_date] = td[t_date].apply(_to_date)
    td = td[~td[t_date].isna()]
    td_stat = td[t_stat].astype(str).str.strip().str.lower()
    td_bank = td[t_bank].astype(str).str.strip().str.lower()
    td = td[td_stat.eq("paid") & td_bank.eq("espay")]
    td = td[(td[t_date] >= month_start) & (td[t_date] <= month_end)]
    td[t_amt] = _to_num(td[t_amt])
    tiket_by_date = td.groupby(td[t_date])[t_amt].sum()
    tiket_by_date.index = pd.to_datetime(tiket_by_date.index).date

    # Settlement total
    sd = settle_df.copy()
    sd[s_date] = sd[s_date].apply(_to_date)
    sd = sd[~sd[s_date].isna()]
    sd = sd[(sd[s_date] >= month_start) & (sd[s_date] <= month_end)]
    sd[s_amt] = _to_num(sd[s_amt])
    settle_total = sd.groupby(sd[s_date])[s_amt].sum()
    settle_total.index = pd.to_datetime(settle_total.index).date

    # BCA (exact match setelah normalisasi); Non-BCA = total - BCA
    if s_prod is not None:
        target = _norm_label(bca_exact)
        prod_norm = sd[s_prod].apply(_norm_label)
        bca_mask = prod_norm.eq(target)
        settle_bca = sd[bca_mask].groupby(sd[bca_mask][s_date])[s_amt].sum() if bca_mask.any() else pd.Series(dtype=float)
    else:
        settle_bca = pd.Series(dtype=float)

    # Bentuk tanggal parameter & reindex
    idx = pd.Index(pd.date_range(month_start, month_end, "D").date, name="Tanggal")

    def _reidx(s: pd.Series) -> pd.Series:
        if not isinstance(s, pd.Series): s = pd.Series(dtype=float)
        if len(getattr(s, "index", [])): s.index = pd.to_datetime(s.index).date
        return s.reindex(idx, fill_value=0.0)

    tiket_series = _reidx(tiket_by_date)
    total_series = _reidx(settle_total)
    bca_series   = _reidx(settle_bca)
    nonbca_series = total_series - bca_series  # konsisten

    # Tabel final (kolom lama + 2 kolom baru)
    final = pd.DataFrame(index=idx)
    final["Tiket Detail ESPAY"]      = tiket_series.values
    final["Settlement Dana ESPAY"]   = total_series.values
    final["Selisih"]                 = final["Tiket Detail ESPAY"] - final["Settlement Dana ESPAY"]
    final["Settlement Dana BCA"]     = bca_series.values
    final["Settlement Dana Non BCA"] = nonbca_series.values

    # Tampilkan
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    total_row = pd.DataFrame([{
        "No": "",
        "Tanggal": "TOTAL",
        "Tiket Detail ESPAY": final["Tiket Detail ESPAY"].sum(),
        "Settlement Dana ESPAY": final["Settlement Dana ESPAY"].sum(),
        "Selisih": final["Selisih"].sum(),
        "Settlement Dana BCA": final["Settlement Dana BCA"].sum(),
        "Settlement Dana Non BCA": final["Settlement Dana Non BCA"].sum(),
    }])
    view_total = pd.concat([view, total_row], ignore_index=True)

    fmt = view_total.copy()
    for c in ["Tiket Detail ESPAY","Settlement Dana ESPAY","Selisih","Settlement Dana BCA","Settlement Dana Non BCA"]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Debug Product Name
    if show_debug and s_prod is not None:
        st.caption("Top Product Name (dist: bulan dipilih) + apakah masuk BCA (exact match)")
        dbg = sd[[s_prod, s_amt, s_date]].copy()
        dbg["_norm"] = dbg[s_prod].apply(_norm_label)
        dbg["_is_bca"] = dbg["_norm"].eq(_norm_label(bca_exact))
        st.dataframe(
            dbg.groupby(["_norm","_is_bca"]).size().sort_values(ascending=False).head(30).rename("rows").reset_index(),
            use_container_width=True
        )

    # Export Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        view_total.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "Unduh Excel",
        data=out.getvalue(),
        file_name=f"rekonsiliasi_{y}-{m:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
