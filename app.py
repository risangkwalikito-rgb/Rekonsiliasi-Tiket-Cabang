# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana
- Parameter tanggal = Bulan & Tahun; hasil 1..akhir bulan itu
- Multi-file upload (Tiket Excel; Settlement CSV/Excel)
- Parser uang/tanggal robust (format Eropa & serial Excel)
- Tiket difilter: St Bayar='paid' & Bank='ESPAY'
- Tambahan kolom: Settlement Dana BCA & Settlement Dana Non BCA (berdasarkan Product Name)
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


def _to_date(val) -> Optional[pd.Timestamp]:
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
            df["__source__"] = f.name  # debug
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _month_selector() -> Tuple[int, int]:
    from datetime import date
    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [
        ("01", "Januari"), ("02", "Februari"), ("03", "Maret"), ("04", "April"),
        ("05", "Mei"), ("06", "Juni"), ("07", "Juli"), ("08", "Agustus"),
        ("09", "September"), ("10", "Oktober"), ("11", "November"), ("12", "Desember"),
    ]
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Tahun", years, index=years.index(today.year))
    with col2:
        month_label = st.selectbox("Bulan", months, index=int(today.strftime("%m")) - 1, format_func=lambda x: x[1])
        month = int(month_label[0])
    return year, month


# ---------- App ----------

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

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode dipakai: {month_start.date()} s/d {month_end.date()}")

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
    # Mapping
    t_date = _find_col(tiket_df, ["Action date"])
    t_amt  = _find_col(tiket_df, ["tarif"])
    t_stat = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status"])
    t_bank = _find_col(tiket_df, ["Bank", "Payment Channel", "channel"])

    s_date = _find_col(settle_df, ["Transaction Date"])
    s_amt  = _find_col(settle_df, ["Settlement Amount"])
    s_prod = _find_col(settle_df, ["Product Name", "Product"])

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
    # Product Name opsional: kalau tidak ada, kolom BCA/Non-BCA akan 0
    if s_prod is None:
        st.warning("Kolom 'Product Name' tidak ditemukan. Kolom Settlement BCA/Non BCA akan diisi 0.")

    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing))
        st.stop()

    # --- Tiket Detail ---
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

    # --- Settlement Dana (Total ESPAY) ---
    sd = settle_df.copy()
    sd[s_date] = sd[s_date].apply(_to_date)
    sd = sd[~sd[s_date].isna()]
    sd = sd[(sd[s_date] >= month_start) & (sd[s_date] <= month_end)]
    if show_debug:
        st.info(f"Contoh Settlement Amount (raw → parsed): {sd[s_amt].head(3).tolist()} → {_to_num(sd[s_amt].head(3)).tolist()}")
    sd[s_amt] = _to_num(sd[s_amt])
    settle_by_date = sd.groupby(sd[s_date])[s_amt].sum()
    settle_by_date.index = pd.to_datetime(settle_by_date.index).date

    # --- Split BCA vs Non-BCA (berdasarkan Product Name) ---
    if s_prod is not None:
        prod_norm = sd[s_prod].astype(str).str.strip().str.lower()
        bca_mask = prod_norm.eq("bca va online")
        settle_bca = sd[bca_mask].groupby(sd[bca_mask][s_date])[s_amt].sum() if bca_mask.any() else pd.Series(dtype=float)
        settle_nonbca = sd[~bca_mask].groupby(sd[~bca_mask][s_date])[s_amt].sum() if (~bca_mask).any() else pd.Series(dtype=float)
    else:
        settle_bca = pd.Series(dtype=float); settle_nonbca = pd.Series(dtype=float)

    # --- Index tanggal dari parameter (1..akhir bulan) ---
    idx = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")

    def _reidx(s: pd.Series) -> pd.Series:
        if not isinstance(s, pd.Series):
            s = pd.Series(dtype=float)
        if len(s.index):
            s.index = pd.to_datetime(s.index).date
        return s.reindex(idx, fill_value=0.0)

    tiket_series   = _reidx(tiket_by_date)
    settle_series  = _reidx(settle_by_date)
    bca_series     = _reidx(settle_bca)
    nonbca_series  = _reidx(settle_nonbca)

    # --- Tabel utama (TIDAK mengubah kolom lama; hanya menambah 2 kolom baru) ---
    final = pd.DataFrame(index=idx)
    final["Tiket Detail ESPAY"]     = tiket_series.values
    final["Settlement Dana ESPAY"]  = settle_series.values
    final["Settlement Dana BCA"]    = bca_series.values
    final["Settlement Dana Non BCA"]= nonbca_series.values
    final["Selisih"]                = final["Tiket Detail ESPAY"] - final["Settlement Dana ESPAY"]

    # View + total
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    total_row = pd.DataFrame([{
        "No": "",
        "Tanggal": "TOTAL",
        "Tiket Detail ESPAY": final["Tiket Detail ESPAY"].sum(),
        "Settlement Dana ESPAY": final["Settlement Dana ESPAY"].sum(),
        "Settlement Dana BCA": final["Settlement Dana BCA"].sum(),
        "Settlement Dana Non BCA": final["Settlement Dana Non BCA"].sum(),
        "Selisih": final["Selisih"].sum(),
    }])
    view_total = pd.concat([view, total_row], ignore_index=True)

    fmt = view_total.copy()
    money_cols = [
        "Tiket Detail ESPAY",
        "Settlement Dana ESPAY",
        "Settlement Dana BCA",
        "Settlement Dana Non BCA",
        "Selisih",
    ]
    for c in money_cols:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

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
