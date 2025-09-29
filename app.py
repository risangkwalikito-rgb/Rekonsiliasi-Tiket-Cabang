Streamlit app: Rekonsiliasi Otomatis (Tiket Detail • Settlement • Rekening Koran)

Fungsi:
- Upload 3 file (Excel Tiket Detail, CSV Settlement, Excel Rekening Koran)
- Mapping kolom fleksibel (tanggal, amount, channel/bank/keterangan)
- Normalisasi angka (format Indonesia) & tanggal
- Agregasi per tanggal: ESPAY, Cash, Settlement BCA/Non-BCA, Uang Masuk BCA/Non-BCA/Cash
- Hitung Selisih & Total, tampilkan tabel, unduh Excel

Dependensi:
  pip install streamlit pandas numpy openpyxl python-dateutil
Jalankan:
  streamlit run app.py
"""

from __future__ import annotations

import io
import re
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ---------------- Utilities ----------------

NON_NUM_RE = re.compile(r"[^\d\-]")

def parse_rupiah_to_number(val) -> float:
    """
    Robust IDR parser: accepts '1.234.567', '1,234,567', '(1.000)', strings with symbols.
    Returns float; on failure -> 0.0.
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    s = s.replace(',', '.')   # unify decimal mark
    s = NON_NUM_RE.sub('', s) # keep digits & minus
    if not s or s == '-':
        return 0.0
    try:
        num = float(s)
    except Exception:
        return 0.0
    return -num if (neg or s.startswith('-')) else num


def normalize_numeric_series(sr: pd.Series) -> pd.Series:
    """Vectorized parser for rupiah-like strings -> float."""
    return sr.apply(parse_rupiah_to_number).astype(float)


def normalize_date(val):
    """Convert many date formats to pandas Timestamp (date-only)."""
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


def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find first column whose name contains any candidate substring (case-insensitive)."""
    cols = [c for c in df.columns if isinstance(c, str)]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        for k, orig in low.items():
            if cand in k:
                return orig
    return None


def idr_fmt(n: float) -> str:
    """Format integer-like to ID style (thousands '.') and parentheses for negatives."""
    if pd.isna(n):
        return "-"
    neg = n < 0
    n_abs = abs(int(round(n)))
    s = f"{n_abs:,}".replace(",", ".")
    return f"({s})" if neg else s


def sum_by_date(
    df: pd.DataFrame,
    date_col: str,
    amt_col: str,
    flt_col: Optional[str] = None,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> pd.Series:
    """
    Sum amount per date with optional keyword filter on flt_col.
    include > exclude precedence. Case-insensitive substring match.
    """
    if df.empty:
        return pd.Series(dtype=float)

    work = df.copy()
    work[date_col] = work[date_col].apply(normalize_date)
    work = work[~work[date_col].isna()]
    work[amt_col] = normalize_numeric_series(work[amt_col])

    if flt_col:
        col = work[flt_col].astype(str).str.lower()
        if include_keywords:
            pats = [re.escape(k.lower()) for k in include_keywords if k]
            if pats:
                work = work[col.str.contains("|".join(pats), na=False)]
        if exclude_keywords:
            pats = [re.escape(k.lower()) for k in exclude_keywords if k]
            if pats:
                work = work[~col.str.contains("|".join(pats), na=False)]

    grp = work.groupby(work[date_col])[[amt_col]].sum().squeeze()
    grp.index = pd.to_datetime(grp.index).date
    return grp


def kw_list(s: str) -> List[str]:
    return [k.strip() for k in (s or "").split(",") if k.strip()]


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Rekonsiliasi Otomatis", layout="wide")
st.title("📊 Rekonsiliasi Otomatis — Tiket Detail • Settlement • Rekening Koran")

with st.sidebar:
    st.header("1) Upload Sumber Data")
    tiket_file = st.file_uploader("Tiket Detail (Excel)", type=["xls", "xlsx"])
    settle_file = st.file_uploader("Settlement Dana Masuk (CSV)", type=["csv"])
    koran_file = st.file_uploader("Rekening Koran (Excel)", type=["xls", "xlsx"])

def try_read_excel(uploaded_file) -> pd.DataFrame:
    if not uploaded_file:
        return pd.DataFrame()
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca Excel: {e}")
        return pd.DataFrame()

def try_read_csv(uploaded_file) -> pd.DataFrame:
    if not uploaded_file:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    st.error("Gagal membaca CSV (coba simpan ulang sebagai UTF-8).")
    return pd.DataFrame()

tiket_df = try_read_excel(tiket_file)
settle_df = try_read_csv(settle_file)
koran_df = try_read_excel(koran_file)

def mapping_controls(df: pd.DataFrame, title: str):
    if df.empty:
        st.info(f"Unggah file {title} terlebih dahulu.")
        return None, None, None
    with st.expander(f"🔧 Mapping kolom: {title}", expanded=False):
        cols = df.columns.tolist()
        auto_date = detect_column(df, ("tanggal", "date", "tgl", "waktu", "posting", "settle"))
        auto_amt  = detect_column(df, ("amount", "nominal", "jumlah", "nilai", "total", "debit", "kredit", "credit"))
        auto_ch   = detect_column(df, ("channel", "metode", "method", "tipe", "type", "bank", "acquirer", "source", "desc", "keterangan"))

        date_col = st.selectbox("Kolom tanggal", options=cols, index=(cols.index(auto_date) if auto_date in cols else 0), key=title+"_date")
        amt_col  = st.selectbox("Kolom amount/nominal", options=cols, index=(cols.index(auto_amt) if auto_amt in cols else 0), key=title+"_amt")
        ch_col   = st.selectbox("Kolom channel/bank/keterangan (opsional)", options=["<tidak ada>"] + cols, index=(0 if not auto_ch else cols.index(auto_ch)+1), key=title+"_ch")
        ch_col   = None if ch_col == "<tidak ada>" else ch_col
    return date_col, amt_col, ch_col

t_date, t_amt, t_ch = mapping_controls(tiket_df, "Tiket Detail")
s_date, s_amt, s_ch = mapping_controls(settle_df, "Settlement")
k_date, k_amt, k_ch = mapping_controls(koran_df, "Rekening Koran")

with st.sidebar:
    st.header("2) Kata Kunci Kategori")
    st.caption("Pisahkan dengan koma; cocok sebagian; case-insensitive.")
    kw_espay = st.text_input("ESPAY (Tiket/Settlement)", value="espay, va espay")
    kw_cash  = st.text_input("CASH (Tiket/Koran)", value="cash, tunai, setor tunai")
    kw_bca   = st.text_input("BCA (Settlement/Koran)", value="bca")
    kw_nonbca= st.text_input("Non-BCA (Settlement/Koran)", value="non bca, bri, bni, mandiri, cimb, permata, danamon")

    show_preview = st.checkbox("Tampilkan pratinjau sumber data", value=False)
    go = st.button("🚀 Proses Rekonsiliasi", type="primary", use_container_width=True)

if show_preview:
    st.subheader("Pratinjau Sumber Data")
    if not tiket_df.empty:
        st.markdown("**Tiket Detail**")
        st.dataframe(tiket_df.head(50), use_container_width=True)
    if not settle_df.empty:
        st.markdown("**Settlement**")
        st.dataframe(settle_df.head(50), use_container_width=True)
    if not koran_df.empty:
        st.markdown("**Rekening Koran**")
        st.dataframe(koran_df.head(50), use_container_width=True)

if go:
    missing = []
    if tiket_df.empty or not (t_date and t_amt): missing.append("Tiket Detail")
    if settle_df.empty or not (s_date and s_amt): missing.append("Settlement")
    if koran_df.empty or not (k_date and k_amt): missing.append("Rekening Koran")
    if missing:
        st.error("Lengkapi unggahan & mapping: " + ", ".join(missing))
        st.stop()

    # Build keyword lists
    espay_keys = kw_list(kw_espay)
    cash_keys  = kw_list(kw_cash)
    bca_keys   = kw_list(kw_bca)
    nonbca_keys= kw_list(kw_nonbca)

    # --- Aggregations ---
    tiket_espay = sum_by_date(tiket_df, t_date, t_amt, t_ch, include_keywords=espay_keys)
    tiket_cash  = sum_by_date(tiket_df, t_date, t_amt, t_ch, include_keywords=cash_keys)

    settle_espay = sum_by_date(settle_df, s_date, s_amt, s_ch, include_keywords=espay_keys)
    settle_bca   = sum_by_date(settle_df, s_date, s_amt, s_ch, include_keywords=bca_keys)
    if nonbca_keys:
        settle_nonbca = sum_by_date(settle_df, s_date, s_amt, s_ch, include_keywords=nonbca_keys)
    else:
        settle_nonbca = sum_by_date(settle_df, s_date, s_amt, s_ch, exclude_keywords=bca_keys)

    koran_norm = koran_df.copy()
    koran_norm[k_date] = koran_norm[k_date].apply(normalize_date)
    koran_norm[k_amt] = normalize_numeric_series(koran_norm[k_amt])

    in_bca    = sum_by_date(koran_norm, k_date, k_amt, k_ch, include_keywords=bca_keys)
    in_nonbca = sum_by_date(koran_norm, k_date, k_amt, k_ch, include_keywords=nonbca_keys)
    in_cash   = sum_by_date(koran_norm, k_date, k_amt, k_ch, include_keywords=cash_keys)

    # --- Combine final table ---
    all_dates = sorted(
        set(tiket_espay.index)
        | set(tiket_cash.index)
        | set(settle_espay.index)
        | set(settle_bca.index)
        | set(settle_nonbca.index)
        | set(in_bca.index)
        | set(in_nonbca.index)
        | set(in_cash.index)
    )
    final = pd.DataFrame(index=pd.Index(all_dates, name="Tanggal"))
    final["Tiket Detail - Espay"] = tiket_espay.reindex(all_dates).fillna(0.0)
    final["Settlement - ESPAY"] = settle_espay.reindex(all_dates).fillna(0.0)
    final["SELISIH TIKET DETAIL - SETTLEMENT"] = final["Tiket Detail - Espay"] - final["Settlement - ESPAY"]
    final["Tiket Detail Cash"] = tiket_cash.reindex(all_dates).fillna(0.0)
    final["Settlement - BCA"] = settle_bca.reindex(all_dates).fillna(0.0)
    final["Settlement - Non BCA"] = settle_nonbca.reindex(all_dates).fillna(0.0)
    final["Total Settlement"] = final["Settlement - BCA"] + final["Settlement - Non BCA"]
    final["Uang Masuk - BCA"] = in_bca.reindex(all_dates).fillna(0.0)
    final["Uang Masuk - Non BCA"] = in_nonbca.reindex(all_dates).fillna(0.0)
    final["Uang Masuk - Cash"] = in_cash.reindex(all_dates).fillna(0.0)

    # Render formatted table
    final_reset = final.reset_index()
    final_reset.insert(0, "No", range(1, len(final_reset) + 1))

    fmt = final_reset.copy()
    money_cols = [c for c in fmt.columns if c not in ("No", "Tanggal")]
    for c in money_cols:
        fmt[c] = fmt[c].apply(idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Download: Excel with raw + view sheet
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        final_reset.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "⬇️ Unduh Excel",
        data=out.getvalue(),
        file_name="rekonsiliasi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.caption("Tips: sesuaikan Kata Kunci & Mapping jika label bank/channel berbeda.")
