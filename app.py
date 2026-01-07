# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana (FAST + FIX FORMAT)
Optimasi utama:
- Baca & concat file hanya saat tombol "Proses" ditekan (hindari rerun Streamlit mahal)
- Cache pembacaan file berbasis (filename, bytes), termasuk isi ZIP
- Parsing tanggal & uang semi-vectorized (fallback apply hanya untuk sisa yang gagal)
- Groupby pakai Timestamp normalize (hindari .dt.date object)
- Detail Tiket & Detail Settlement pakai pivot_table (1x scan)

Fix error TypeError saat format:
- _idr_fmt tahan banting (non-angka tidak dibandingkan <0)
- Kolom display df2/df3 dipaksa jadi MultiIndex konsisten, NO/Tanggal jadi ("","NO") & ("","Tanggal")
- Format Rupiah hanya untuk kolom numeric
"""

from __future__ import annotations

import io
import re
import zipfile
import calendar
import unicodedata
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

# =========================
# Utilities
# =========================

def _parse_money(val) -> float:
    """Fallback parser uang per-sel (dipakai hanya untuk kasus yang gagal di parser cepat)."""
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

def _to_num_fast(sr: pd.Series) -> pd.Series:
    """Parser uang cepat (vector-ish). Fallback apply hanya untuk sisa yang gagal."""
    if sr is None:
        return pd.Series(dtype=float)

    if pd.api.types.is_numeric_dtype(sr):
        return pd.to_numeric(sr, errors="coerce").fillna(0.0).astype(float)

    s = sr.astype(str).str.strip()
    s2 = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    neg = s.str.match(r"^\(.*\)$") | s.str.endswith("-") | s.str.startswith("-")

    x = s2.fillna("")
    x = x.str.replace(r"[()]", "", regex=True)
    x = x.str.replace(r"(?i)\b(idr|rp|cr|dr)\b", "", regex=True)
    x = x.str.replace("-", "", regex=False)
    x = x.str.replace(r"[^0-9\.,]", "", regex=True)

    last_dot = x.str.rfind(".")
    last_com = x.str.rfind(",")

    dot_major = last_dot > last_com
    out = pd.Series(np.nan, index=sr.index, dtype=float)

    if dot_major.any():
        out.loc[dot_major] = pd.to_numeric(x.loc[dot_major].str.replace(",", "", regex=False), errors="coerce")
    if (~dot_major).any():
        out.loc[~dot_major] = pd.to_numeric(
            x.loc[~dot_major].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    need_fallback = out.isna() & s2.notna()
    if need_fallback.any():
        out.loc[need_fallback] = s2.loc[need_fallback].apply(_parse_money)

    out = out.fillna(0.0)
    out.loc[neg] = -out.loc[neg].abs()
    return out.astype(float)

_ddmmyyyy = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")

def _to_date_series_fast(sr: pd.Series) -> pd.Series:
    """Parser tanggal cepat: excel serial + pandas to_datetime + fallback dateutil untuk sisa."""
    if sr is None:
        return pd.Series(dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(sr):
        return pd.to_datetime(sr, errors="coerce").dt.normalize()

    s = sr.copy()

    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=sr.index)

    mask_serial = num.between(1, 100000)
    if mask_serial.any():
        out.loc[mask_serial] = (pd.Timestamp("1899-12-30") + pd.to_timedelta(num[mask_serial], unit="D")).dt.normalize()

    mask_rest = ~mask_serial
    if mask_rest.any():
        ss = s.loc[mask_rest].astype(str).str.strip()
        parsed = pd.to_datetime(ss, errors="coerce", dayfirst=True)
        remain = parsed.isna() & ss.ne("")
        if remain.any():
            parsed2 = pd.to_datetime(ss[remain], errors="coerce", dayfirst=False)
            parsed.loc[remain] = parsed2
        out.loc[mask_rest] = parsed.dt.normalize()

    need_fallback = out.isna() & sr.notna()
    if need_fallback.any():
        def _fallback_one(v):
            if pd.isna(v):
                return pd.NaT
            s0 = str(v).strip()
            if not s0:
                return pd.NaT
            m0 = _ddmmyyyy.search(s0)
            if m0:
                d, M, y = m0.groups()
                if len(y) == 2:
                    y = "20" + y
                try:
                    return pd.Timestamp(year=int(y), month=int(M), day=int(d))
                except Exception:
                    pass
            for dayfirst in (True, False):
                try:
                    d0 = dtparser.parse(s0, dayfirst=dayfirst, fuzzy=True)
                    return pd.Timestamp(d0.date())
                except Exception:
                    continue
            return pd.NaT
        out.loc[need_fallback] = sr.loc[need_fallback].apply(_fallback_one)

    return pd.to_datetime(out, errors="coerce").dt.normalize()

def _norm_str(val) -> str:
    s = "" if val is None else str(val)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

def _norm_series_simple(sr: pd.Series) -> pd.Series:
    return sr.astype(str).str.strip().str.casefold()

def _idr_fmt(val) -> str:
    """Format angka ke IDR. Kalau val bukan angka (tanggal/teks), kembalikan string apa adanya."""
    if val is None:
        return "-"
    if isinstance(val, float) and np.isnan(val):
        return "-"
    try:
        n = float(val)
    except Exception:
        return str(val)
    neg = n < 0
    s = f"{abs(int(round(n))):,}".replace(",", ".")
    return f"({s})" if neg else s

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns if isinstance(c, str)]
    m = {c.lower().strip().lstrip("\ufeff"): c for c in cols}
    for n in names:
        key = n.lower().strip()
        if key in m:
            return m[key]
    for n in names:
        key = n.lower().strip()
        for c in cols:
            if key in c.lower():
                return c
    return None

def _read_any(file_like) -> pd.DataFrame:
    """Baca CSV/Excel. .xls → xlrd; fallback pyexcel-xls; terakhir coba openpyxl."""
    if not file_like:
        return pd.DataFrame()
    name = getattr(file_like, "name", "unknown").lower()

    try:
        if name.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
                try:
                    file_like.seek(0)
                    return pd.read_csv(
                        file_like,
                        encoding=enc,
                        sep=None,
                        engine="python",
                        dtype=str,
                        na_filter=False,
                    )
                except Exception:
                    continue
            st.error(f"CSV gagal dibaca: {getattr(file_like,'name','(no name)')}. Simpan ulang sebagai UTF-8.")
            return pd.DataFrame()

        elif name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            file_like.seek(0)
            return pd.read_excel(file_like, engine="openpyxl")

        elif name.endswith(".xls"):
            try:
                file_like.seek(0)
                return pd.read_excel(file_like, engine="xlrd")
            except Exception:
                pass
            try:
                file_like.seek(0)
                raw = file_like.read()
                from pyexcel_xls import get_data
                book = get_data(io.BytesIO(raw))
                for _sh, rows in book.items():
                    if not rows:
                        continue
                    header = [str(x).strip() if x is not None else "" for x in rows[0]]
                    body = rows[1:] if len(rows) > 1 else []
                    return pd.DataFrame(body, columns=header)
            except Exception:
                pass
            try:
                file_like.seek(0)
                return pd.read_excel(file_like, engine="openpyxl")
            except Exception:
                st.error("Gagal membaca file .xls. Pasang 'xlrd' atau 'pyexcel-xls', atau simpan ulang ke .xlsx.")
                return pd.DataFrame()
        else:
            file_like.seek(0)
            return pd.read_excel(file_like)

    except ImportError:
        st.error("Dukungan .xls perlu paket 'xlrd' atau 'pyexcel-xls'. Tambahkan di requirements.txt.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca {getattr(file_like,'name','(no name)')}: {e}")
        return pd.DataFrame()

# =========================
# ZIP expand + Caching
# =========================

@st.cache_data(show_spinner=False)
def _read_any_bytes(name: str, data: bytes) -> pd.DataFrame:
    bio = io.BytesIO(data)
    bio.name = name
    return _read_any(bio)

def _expand_zip_to_pairs(files) -> List[Tuple[str, bytes]]:
    """Return list of (name, bytes); if zip, expand allowed ext."""
    out: List[Tuple[str, bytes]] = []
    allow_ext = (".csv", ".xls", ".xlsx")

    for f in (files or []):
        try:
            f.seek(0)
            data = f.read()
        except Exception:
            data = f.getvalue()

        fname = (getattr(f, "name", "") or "").lower()
        if fname.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner = info.filename
                        inner_lower = inner.lower()
                        if inner_lower.startswith("__macosx/") or inner_lower.endswith(".ds_store"):
                            continue
                        if not inner_lower.endswith(allow_ext):
                            continue
                        out.append((f"{getattr(f,'name','zip')}::{inner}", zf.read(info)))
            except Exception as e:
                st.warning(f"Gagal ekstrak ZIP {getattr(f,'name','(zip)')}: {e}")
        else:
            out.append((getattr(f, "name", "file"), data))
    return out

def _concat_files_cached(pairs: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, data in (pairs or []):
        df = _read_any_bytes(name, data)
        if df is not None and not df.empty:
            df = df.copy()
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            df["__source__"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _promote_header(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _is_header_row(sr: pd.Series) -> bool:
        vals = [str(v).strip().lower() for v in sr.fillna("")]
        keys = ["date", "tanggal", "transaction date", "tgl",
                "remark", "keterangan", "description", "deskripsi",
                "credit", "kredit", "cr", "amount", "jumlah"]
        score = sum(any(k in v for k in keys) for v in vals)
        return score >= 2

    for r in (12, 13):
        if r < len(df) and _is_header_row(df.iloc[r]):
            cols = [str(x).strip() for x in df.iloc[r].tolist()]
            out = df.iloc[r + 1 :].copy()
            out.columns = cols
            out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
            return out

    scan_max = min(50, len(df))
    for r in range(scan_max):
        if _is_header_row(df.iloc[r]):
            cols = [str(x).strip() for x in df.iloc[r].tolist()]
            out = df.iloc[r + 1 :].copy()
            out.columns = cols
            out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
            return out

    return df

def _concat_rk_non_cached(pairs: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, data in (pairs or []):
        df = _read_any_bytes(name, data)
        if df is None or df.empty:
            continue
        df = _promote_header(df)
        if df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        df["__source__"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# =========================
# Month selector
# =========================

def _month_selector() -> Tuple[int, int]:
    from datetime import date
    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [
        ("01", "Januari"), ("02", "Februari"), ("03", "Maret"), ("04", "April"),
        ("05", "Mei"), ("06", "Juni"), ("07", "Juli"), ("08", "Agustus"),
        ("09", "September"), ("10", "Oktober"), ("11", "November"), ("12", "Desember")
    ]
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Tahun", years, index=years.index(today.year))
    with col2:
        month_label = st.selectbox("Bulan", months, index=int(today.strftime("%m")) - 1, format_func=lambda x: x[1])
        month = int(month_label[0])
    return year, month

def _force_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan semua columns menjadi MultiIndex tuple, NO/Tanggal jadi ("","NO") & ("","Tanggal") bila string."""
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append(c)
        else:
            cols.append(("", str(c)))
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df

# =========================
# App
# =========================

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    tiket_files = st.file_uploader("Tiket Detail (Excel .xls/.xlsx/.zip)", type=["xls", "xlsx", "zip"], accept_multiple_files=True)
    settle_files = st.file_uploader("Settlement Dana (CSV/Excel/.zip)", type=["csv", "xls", "xlsx", "zip"], accept_multiple_files=True)
    st.divider()
    st.header("Rekening Koran (opsional, multi-file)")
    rk_bca_files = st.file_uploader("Rekening Koran BCA (CSV/Excel/.zip)", type=["csv", "xls", "xlsx", "zip"], accept_multiple_files=True)
    rk_non_files = st.file_uploader("Rekening Koran Non BCA (CSV/Excel/.zip)", type=["csv", "xls", "xlsx", "zip"], accept_multiple_files=True)

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode dipakai: {month_start.date()} s/d {month_end.date()}")

    go = st.button("Proses", type="primary", use_container_width=True)

if go:
    # =========================
    # Read inputs (ONLY when go)
    # =========================
    tiket_pairs = _expand_zip_to_pairs(tiket_files)
    settle_pairs = _expand_zip_to_pairs(settle_files)
    rk_bca_pairs = _expand_zip_to_pairs(rk_bca_files)
    rk_non_pairs = _expand_zip_to_pairs(rk_non_files)

    tiket_df = _concat_files_cached(tiket_pairs)
    settle_df = _concat_files_cached(settle_pairs)
    rk_bca_df = _concat_files_cached(rk_bca_pairs)
    rk_non_df = _concat_rk_non_cached(rk_non_pairs)

    if tiket_df.empty:
        st.error("Tiket Detail kosong / belum diupload.")
        st.stop()
    if settle_df.empty:
        st.error("Settlement Dana kosong / belum diupload.")
        st.stop()

    # ---------------------- Tiket Detail (TABEL 1) ----------------------
    t_date_action = _find_col(tiket_df, ["Action/Action Date", "Action Date", "Action", "Action date"])
    t_amt_tarif = _find_col(tiket_df, ["Tarif", "tarif"])
    t_stat = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status", "status bayar"])
    t_bank = _find_col(tiket_df, ["Bank", "Payment Channel", "channel", "payment method"])

    if t_date_action is None:
        st.error("Kolom tanggal 'Action Date' tidak ditemukan pada Tiket Detail.")
        st.stop()
    if t_amt_tarif is None:
        st.error("Kolom nominal 'Tarif' tidak ditemukan pada Tiket Detail.")
        st.stop()
    if t_stat is None or t_bank is None:
        st.error("Kolom 'St Bayar' atau 'Bank' tidak ditemukan pada Tiket Detail.")
        st.stop()

    # ---------------------- Settlement (utama/legacy) ----------------------
    s_date_legacy = _find_col(settle_df, ["Transaction Date", "Tanggal Transaksi", "Tanggal"])
    s_amt_legacy = _find_col(settle_df, ["Settlement Amount", "Amount", "Nominal", "Jumlah"])
    if s_amt_legacy is None and len(settle_df.columns) >= 12:
        s_amt_legacy = settle_df.columns[11]  # kolom L
    if s_date_legacy is None:
        s_date_legacy = _find_col(settle_df, ["Settlement Date", "Tanggal Settlement", "Settle Date", "Tanggal", "Setle Date"])
        if s_date_legacy is None and len(settle_df.columns) >= 5:
            s_date_legacy = settle_df.columns[4]  # kolom E

    if s_date_legacy is None or s_amt_legacy is None:
        st.error("Kolom wajib Settlement Dana tidak ditemukan (tanggal/amount).")
        st.stop()

    # Untuk split BCA/Non-BCA dari E/L/P
    s_date_E = _find_col(settle_df, ["Settlement Date", "Tanggal Settlement", "Settle Date", "Tanggal"]) or (settle_df.columns[4] if len(settle_df.columns) >= 5 else None)
    s_amt_L = _find_col(settle_df, ["Settlement Amount", "Amount", "Nominal", "Jumlah"]) or (settle_df.columns[11] if len(settle_df.columns) >= 12 else None)
    s_prod_P = _find_col(settle_df, ["Product Name", "Produk", "Nama Produk"]) or (settle_df.columns[15] if len(settle_df.columns) >= 16 else None)

    # =========================
    # TABEL 1: TIKET DETAIL ESPAY (Bank=ESPAY, St Bayar=paid)
    # =========================
    td = tiket_df[[t_date_action, t_amt_tarif, t_bank, t_stat]].copy()
    td[t_date_action] = _to_date_series_fast(td[t_date_action])
    td = td[td[t_date_action].notna()]

    bank_norm = _norm_series_simple(td[t_bank])
    stat_norm = _norm_series_simple(td[t_stat])
    td = td[(bank_norm == "espay") & (stat_norm == "paid")]

    td = td[(td[t_date_action] >= month_start) & (td[t_date_action] <= month_end)]
    td[t_amt_tarif] = _to_num_fast(td[t_amt_tarif])

    td = td.drop_duplicates()
    td["_DATE"] = td[t_date_action].dt.normalize()
    tiket_by_date = td.groupby("_DATE", sort=False)[t_amt_tarif].sum()

    # =========================
    # Settlement (utama)
    # =========================
    sd_main = settle_df[[s_date_legacy, s_amt_legacy]].copy()
    sd_main[s_date_legacy] = _to_date_series_fast(sd_main[s_date_legacy])
    sd_main = sd_main[sd_main[s_date_legacy].notna()]
    sd_main = sd_main[(sd_main[s_date_legacy] >= month_start) & (sd_main[s_date_legacy] <= month_end)]
    sd_main[s_amt_legacy] = _to_num_fast(sd_main[s_amt_legacy])
    sd_main["_DATE"] = sd_main[s_date_legacy].dt.normalize()
    settle_by_date_total = sd_main.groupby("_DATE", sort=False)[s_amt_legacy].sum()

    # =========================
    # Settlement split BCA / Non-BCA (E/L/P)
    # =========================
    bca_series = pd.Series(dtype=float)
    non_bca_series = pd.Series(dtype=float)
    if s_date_E and s_amt_L and s_prod_P:
        sd_split = settle_df[[s_date_E, s_amt_L, s_prod_P]].copy()
        sd_split[s_date_E] = _to_date_series_fast(sd_split[s_date_E])
        sd_split = sd_split[sd_split[s_date_E].notna()]
        sd_split = sd_split[(sd_split[s_date_E] >= month_start) & (sd_split[s_date_E] <= month_end)]
        sd_split[s_amt_L] = _to_num_fast(sd_split[s_amt_L])

        prod_norm = sd_split[s_prod_P].astype(str).str.strip().str.casefold()
        bca_mask = (prod_norm == "bca va online".casefold())

        sd_split["_DATE"] = sd_split[s_date_E].dt.normalize()
        bca_series = sd_split.loc[bca_mask].groupby("_DATE", sort=False)[s_amt_L].sum()
        non_bca_series = sd_split.loc[~bca_mask].groupby("_DATE", sort=False)[s_amt_L].sum()

    # =========================
    # RK: Uang Masuk BCA (filter "mrc")
    # =========================
    uang_masuk_bca = pd.Series(dtype=float)
    if not rk_bca_df.empty:
        rk_tgl_bca = _find_col(rk_bca_df, ["Tanggal", "Date", "Tgl", "Transaction Date"])
        rk_amt_bca = _find_col(rk_bca_df, ["mutasi", "amount", "kredit", "credit", "cr"])
        rk_ket_bca = _find_col(rk_bca_df, ["Keterangan", "Remark", "Deskripsi", "Description"])
        if rk_tgl_bca and rk_amt_bca and rk_ket_bca:
            bca = rk_bca_df[[rk_tgl_bca, rk_amt_bca, rk_ket_bca]].copy()
            bca[rk_tgl_bca] = _to_date_series_fast(bca[rk_tgl_bca])
            bca = bca[bca[rk_tgl_bca].notna()]
            bca = bca[(bca[rk_tgl_bca] >= month_start) & (bca[rk_tgl_bca] <= month_end)]
            ket = bca[rk_ket_bca].astype(str).str.strip().str.casefold()
            bca = bca[ket.str.contains("mrc", na=False)]
            bca[rk_amt_bca] = _to_num_fast(bca[rk_amt_bca])
            bca["_DATE"] = bca[rk_tgl_bca].dt.normalize()
            uang_masuk_bca = bca.groupby("_DATE", sort=False)[rk_amt_bca].sum()

    # =========================
    # RK: Uang Masuk Non-BCA (filter "mrc")
    # =========================
    uang_masuk_non = pd.Series(dtype=float)
    if not rk_non_df.empty:
        rk_tgl_non = _find_col(rk_non_df, ["Date", "Tanggal", "Transaction Date", "Tgl"])
        rk_amt_non = _find_col(rk_non_df, ["credit", "kredit", "cr", "amount"])
        rk_rem_non = _find_col(rk_non_df, ["Remark", "Keterangan", "Description", "Deskripsi"])
        if rk_tgl_non and rk_amt_non and rk_rem_non:
            nb = rk_non_df[[rk_tgl_non, rk_amt_non, rk_rem_non]].copy()
            nb[rk_tgl_non] = _to_date_series_fast(nb[rk_tgl_non])
            nb = nb[nb[rk_tgl_non].notna()]
            nb = nb[(nb[rk_tgl_non] >= month_start) & (nb[rk_tgl_non] <= month_end)]
            rem = nb[rk_rem_non].astype(str).str.strip().str.casefold()
            nb = nb[rem.str.contains("mrc", na=False)]
            nb[rk_amt_non] = _to_num_fast(nb[rk_amt_non])
            nb["_DATE"] = nb[rk_tgl_non].dt.normalize()
            uang_masuk_non = nb.groupby("_DATE", sort=False)[rk_amt_non].sum()

    # =========================
    # Index tanggal 1..akhir bulan (DatetimeIndex)
    # =========================
    idx = pd.date_range(month_start, month_end, freq="D")

    tiket_series = tiket_by_date.reindex(idx, fill_value=0.0)
    settle_series = settle_by_date_total.reindex(idx, fill_value=0.0)

    bca_series = bca_series.reindex(idx, fill_value=0.0)
    non_bca_series = non_bca_series.reindex(idx, fill_value=0.0)
    total_settle_ser = (bca_series + non_bca_series).reindex(idx, fill_value=0.0)

    uang_masuk_bca_ser = uang_masuk_bca.reindex(idx, fill_value=0.0)
    uang_masuk_non_ser = uang_masuk_non.reindex(idx, fill_value=0.0)
    total_uang_masuk_ser = (uang_masuk_bca_ser + uang_masuk_non_ser).reindex(idx, fill_value=0.0)

    # =========================
    # Final reconciliation table
    # =========================
    final = pd.DataFrame(index=idx)
    final["TIKET DETAIL ESPAY"] = tiket_series.values
    final["SETTLEMENT DANA ESPAY"] = settle_series.values
    final["SELISIH TIKET DETAIL - SETTLEMENT"] = final["TIKET DETAIL ESPAY"] - final["SETTLEMENT DANA ESPAY"]
    final["SETTLEMENT BCA"] = bca_series.values
    final["SETTLEMENT NON BCA"] = non_bca_series.values
    final["TOTAL SETTLEMENT"] = total_settle_ser.values
    final["UANG MASUK BCA"] = uang_masuk_bca_ser.values
    final["UANG MASUK NON BCA"] = uang_masuk_non_ser.values
    final["TOTAL UANG MASUK"] = total_uang_masuk_ser.values
    final["SELISIH SETTLEMENT - UANG MASUK"] = final["TOTAL SETTLEMENT"] - final["TOTAL UANG MASUK"]

    view = final.reset_index().rename(columns={"index": "TANGGAL"})
    view["TANGGAL"] = pd.to_datetime(view["TANGGAL"]).dt.date
    view.insert(0, "NO", range(1, len(view) + 1))

    total_row = pd.DataFrame([{
        "NO": "",
        "TANGGAL": "TOTAL",
        "TIKET DETAIL ESPAY": float(final["TIKET DETAIL ESPAY"].sum()),
        "SETTLEMENT DANA ESPAY": float(final["SETTLEMENT DANA ESPAY"].sum()),
        "SELISIH TIKET DETAIL - SETTLEMENT": float(final["SELISIH TIKET DETAIL - SETTLEMENT"].sum()),
        "SETTLEMENT BCA": float(final["SETTLEMENT BCA"].sum()),
        "SETTLEMENT NON BCA": float(final["SETTLEMENT NON BCA"].sum()),
        "TOTAL SETTLEMENT": float(final["TOTAL SETTLEMENT"].sum()),
        "UANG MASUK BCA": float(final["UANG MASUK BCA"].sum()),
        "UANG MASUK NON BCA": float(final["UANG MASUK NON BCA"].sum()),
        "TOTAL UANG MASUK": float(final["TOTAL UANG MASUK"].sum()),
        "SELISIH SETTLEMENT - UANG MASUK": float(final["SELISIH SETTLEMENT - UANG MASUK"].sum()),
    }])

    view_total = pd.concat([view, total_row], ignore_index=True)

    ordered_cols = [
        "NO", "TANGGAL",
        "TIKET DETAIL ESPAY", "SETTLEMENT DANA ESPAY", "SELISIH TIKET DETAIL - SETTLEMENT",
        "SETTLEMENT BCA", "SETTLEMENT NON BCA", "TOTAL SETTLEMENT",
        "UANG MASUK BCA", "UANG MASUK NON BCA", "TOTAL UANG MASUK",
        "SELISIH SETTLEMENT - UANG MASUK",
    ]
    view_total = view_total.loc[:, ordered_cols]

    fmt = view_total.copy()
    for c in ordered_cols:
        if c in ("NO", "TANGGAL"):
            continue
        if pd.api.types.is_numeric_dtype(fmt[c]):
            fmt[c] = fmt[c].map(_idr_fmt)
        else:
            fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # =========================
    # Export Rekonsiliasi
    # =========================
    from openpyxl.styles import Alignment, Font

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        view_total.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")

        wb = xw.book
        ws = wb["Rekonsiliasi_View"]
        ws.insert_rows(1)

        sub_headers = [
            "NO", "TANGGAL",
            "TIKET DETAIL ESPAY", "SETTLEMENT DANA ESPAY", "SELISIH TIKET DETAIL - SETTLEMENT",
            "BCA", "NON BCA", "TOTAL SETTLEMENT",
            "BCA", "NON BCA", "TOTAL UANG MASUK",
            "SELISIH SETTLEMENT - UANG MASUK",
        ]
        top_headers = [
            "NO", "TANGGAL",
            "TIKET DETAIL ESPAY", "SETTLEMENT DANA ESPAY", "SELISIH TIKET DETAIL - SETTLEMENT",
            "SETTLEMENT", "SETTLEMENT", "TOTAL SETTLEMENT",
            "UANG MASUK", "UANG MASUK", "TOTAL UANG MASUK",
            "SELISIH SETTLEMENT - UANG MASUK",
        ]
        for col_idx, (top, sub) in enumerate(zip(top_headers, sub_headers), start=1):
            ws.cell(row=1, column=col_idx, value=top)
            ws.cell(row=2, column=col_idx, value=sub)

        ws.merge_cells(start_row=1, start_column=6, end_row=1, end_column=7)   # SETTLEMENT
        ws.merge_cells(start_row=1, start_column=9, end_row=1, end_column=10)  # UANG MASUK

        max_col = ws.max_column
        for c in range(1, max_col + 1):
            ws.cell(row=1, column=c).font = Font(bold=True)
            ws.cell(row=2, column=c).font = Font(bold=True)
            ws.cell(row=1, column=c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.cell(row=2, column=c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 22
        ws.row_dimensions[2].height = 22

    st.download_button(
        "Unduh Excel",
        data=bio.getvalue(),
        file_name=f"rekonsiliasi_{y}-{m:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # ======================================================================
    # TABEL BARU: DETAIL TIKET (GO SHOW × SUB-KATEGORI)  [SEMUA STATUS]
    # ======================================================================

    def _col_by_letter_local(df: pd.DataFrame, letters: str) -> Optional[str]:
        if df is None or df.empty:
            return None
        s = letters.strip().upper()
        if not s:
            return None
        n = 0
        for ch in s:
            if not ("A" <= ch <= "Z"):
                return None
            n = n * 26 + (ord(ch) - ord("A") + 1)
        idx0 = n - 1
        return df.columns[idx0] if 0 <= idx0 < len(df.columns) else None

    type_main_col = _find_col(tiket_df, ["Type", "Tipe", "Jenis"]) or _col_by_letter_local(tiket_df, "B")   # GO SHOW / ONLINE
    bank_col      = _find_col(tiket_df, ["Bank", "Payment Channel", "channel", "payment method"]) or _col_by_letter_local(tiket_df, "I")
    type_sub_col  = (
        _find_col(tiket_df, [
            "Payment Type", "Channel Type", "Transaction Type", "Sub Type",
            "Tipe", "Tipe Pembayaran", "Jenis Pembayaran", "Kategori", "Metode", "Product Type"
        ]) or _col_by_letter_local(tiket_df, "J")
    )
    date_col  = _find_col(tiket_df, ["Action/Action Date", "Action Date", "Action", "Action date"]) or _col_by_letter_local(tiket_df, "AG")
    tarif_col = _find_col(tiket_df, ["Tarif", "tarif"]) or _col_by_letter_local(tiket_df, "Y")

    required_missing = [n for n, c in [
        ("TYPE (kolom B)", type_main_col),
        ("BANK (kolom I)", bank_col),
        ("SUB-TIPE (kolom J)", type_sub_col),
        ("ACTION DATE (kolom AG)", date_col),
        ("TARIF (kolom Y)", tarif_col),
    ] if c is None]

    if required_missing:
        st.warning("Kolom wajib untuk tabel 'Detail Tiket (GO SHOW/ONLINE)' belum lengkap: " + ", ".join(required_missing))
    else:
        tix = tiket_df[[type_main_col, bank_col, type_sub_col, date_col, tarif_col]].copy()
        tix[date_col] = _to_date_series_fast(tix[date_col])
        tix = tix[tix[date_col].notna()]
        tix = tix[(tix[date_col] >= month_start) & (tix[date_col] <= month_end)]
        tix[tarif_col] = _to_num_fast(tix[tarif_col])

        main_norm = tix[type_main_col].astype(str).str.strip().str.casefold()
        sub_norm  = tix[type_sub_col].astype(str).str.strip().str.casefold()
        bank_norm = tix[bank_col].astype(str).str.strip().str.casefold()

        m_go_show = (main_norm == "go show") | main_norm.str.contains(r"\bgo\s*show\b", na=False)
        m_online  = (main_norm == "online")  | main_norm.str.contains(r"\bonline\b", na=False)

        m_prepaid  = (sub_norm == "prepaid") | sub_norm.str.contains(r"\bprepaid\b", na=False)
        m_emoney   = (sub_norm == "e-money") | sub_norm.str.contains(r"\be[-\s]*money\b|\bemoney\b", na=False)
        m_varetail = sub_norm.str.contains(r"virtual\s*account", na=False) & sub_norm.str.contains(r"gerai|retail", na=False)
        m_cash     = (sub_norm == "cash") | sub_norm.str.contains(r"\bcash\b", na=False)

        label = pd.Series("", index=tix.index)

        # GO SHOW
        label.loc[m_go_show & m_prepaid & (bank_norm == "bca")]      = "PREPAID - BCA"
        label.loc[m_go_show & m_prepaid & (bank_norm == "bri")]      = "PREPAID - BRI"
        label.loc[m_go_show & m_prepaid & (bank_norm == "bni")]      = "PREPAID - BNI"
        label.loc[m_go_show & m_prepaid & (bank_norm == "mandiri")]  = "PREPAID - MANDIRI"
        label.loc[m_go_show & m_emoney  & (bank_norm == "espay")]    = "E-MONEY - ESPAY"
        label.loc[m_go_show & m_varetail & (bank_norm == "espay")]   = "VIRTUAL ACCOUNT DAN GERAI RETAIL - ESPAY"
        label.loc[m_go_show & m_cash    & (bank_norm == "asdp")]     = "CASH - ASDP"

        # ONLINE
        label.loc[m_online & m_emoney & (bank_norm == "espay")]      = "E-MONEY - ESPAY"
        label.loc[m_online & m_varetail & (bank_norm == "espay")]    = "VIRTUAL ACCOUNT & GERAI RETAIL - ESPAY"
        label.loc[m_online & m_cash & (bank_norm == "asdp")]         = "CASH - ASDP"

        main_grp = pd.Series(np.where(m_go_show, "GO SHOW", np.where(m_online, "ONLINE", "")), index=tix.index)
        keep = (main_grp != "") & (label != "")
        tix2 = tix.loc[keep].copy()
        tix2["_MAIN"] = main_grp.loc[keep].values
        tix2["_LABEL"] = label.loc[keep].values
        tix2["_DATE"] = tix2[date_col].dt.normalize()

        idx2 = pd.date_range(month_start, month_end, freq="D")

        pt = tix2.pivot_table(
            index="_DATE",
            columns=["_MAIN", "_LABEL"],
            values=tarif_col,
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(idx2, fill_value=0.0)

        gs_order = [
            "PREPAID - BCA", "PREPAID - BRI", "PREPAID - BNI", "PREPAID - MANDIRI",
            "E-MONEY - ESPAY", "VIRTUAL ACCOUNT DAN GERAI RETAIL - ESPAY", "CASH - ASDP",
        ]
        on_order = [
            "E-MONEY - ESPAY", "VIRTUAL ACCOUNT & GERAI RETAIL - ESPAY", "CASH - ASDP",
        ]

        def _ensure_cols(main_name: str, labels_list: List[str]):
            for lab in labels_list:
                if (main_name, lab) not in pt.columns:
                    pt[(main_name, lab)] = 0.0

        _ensure_cols("GO SHOW", gs_order)
        _ensure_cols("ONLINE", on_order)

        # build output df2 (MultiIndex columns)
        gs = pt["GO SHOW"][gs_order].copy()
        on = pt["ONLINE"][on_order].copy()

        gs_subtotal = gs.sum(axis=1)
        on_subtotal = on.sum(axis=1)
        grand_total = gs_subtotal + on_subtotal

        df2 = pd.DataFrame(index=idx2)
        for c in gs_order:
            df2[("GO SHOW", c)] = gs[c].values
        df2[("GO SHOW", "SUBTOTAL")] = gs_subtotal.values
        for c in on_order:
            df2[("ONLINE", c)] = on[c].values
        df2[("ONLINE", "SUBTOTAL")] = on_subtotal.values
        df2[("GRAND TOTAL", "GRAND TOTAL")] = grand_total.values

        st.subheader("Detail Tiket per Tanggal — TYPE: GO SHOW & ONLINE × SUB-TIPE (J) [SEMUA STATUS]")

        df2_view = df2.reset_index().rename(columns={"index": "Tanggal"})
        df2_view["Tanggal"] = pd.to_datetime(df2_view["Tanggal"]).dt.date
        df2_view.insert(0, "NO", range(1, len(df2_view) + 1))
        df2_view = _force_multiindex_cols(df2_view)  # <-- FIX: konsisten MultiIndex

        # total row (pakai MultiIndex key)
        total_row = {("", "NO"): "", ("", "Tanggal"): "TOTAL"}
        for col in df2.columns:
            total_row[col] = float(df2[col].sum())
        df2_view = pd.concat([df2_view, pd.DataFrame([total_row])], ignore_index=True)

        # format rupiah hanya numeric & skip NO/Tanggal
        df2_fmt = df2_view.copy()
        for col in df2_fmt.columns:
            if col in [("", "NO"), ("", "Tanggal")]:
                continue
            if pd.api.types.is_numeric_dtype(df2_fmt[col]):
                df2_fmt[col] = df2_fmt[col].map(_idr_fmt)

        st.dataframe(df2_fmt, use_container_width=True, hide_index=True)

    # ======================================================================
    # TABEL: DETAIL SETTLEMENT REPORT (pivot)
    # ======================================================================
    st.subheader("DETAIL SETTLEMENT REPORT")

    s_order = _find_col(settle_df, ["Order ID", "OrderId", "Order Number", "Order No", "OrderID", "order id"])
    need = [("Settlement Date (E)", s_date_E), ("Settlement Amount (L)", s_amt_L), ("Product Name (P)", s_prod_P), ("Order ID", s_order)]
    miss = [n for n, c in need if c is None]
    if miss:
        st.warning("Kolom untuk 'DETAIL SETTLEMENT REPORT' belum lengkap: " + ", ".join(miss))
    else:
        sd = settle_df[[s_date_E, s_amt_L, s_prod_P, s_order]].copy()
        sd[s_date_E] = _to_date_series_fast(sd[s_date_E])
        sd = sd[sd[s_date_E].notna()]
        sd = sd[(sd[s_date_E] >= month_start) & (sd[s_date_E] <= month_end)]
        sd[s_amt_L] = _to_num_fast(sd[s_amt_L])

        prod_norm = sd[s_prod_P].astype(str).str.strip().str.casefold()
        order_norm = sd[s_order].astype(str).str.strip().str.casefold()

        go_show_mask = order_norm.str.endswith("_ord") | (~order_norm.str.startswith("ord") & order_norm.str.endswith("ord"))
        online_mask = order_norm.str.startswith("ord")

        main = np.where(go_show_mask, "GO SHOW", np.where(online_mask, "ONLINE", ""))
        has_va = prod_norm.str.contains("va", na=False)
        is_bca_va = (prod_norm == "bca va online".casefold())
        label = np.where(has_va & is_bca_va, "VIRTUAL ACCOUNT - BCA",
                 np.where(has_va & ~is_bca_va, "VIRTUAL ACCOUNT - NON BCA",
                 "E-MONEY"))

        keep = (main != "")
        sd2 = sd.loc[keep].copy()
        sd2["_MAIN"] = main[keep]
        sd2["_LABEL"] = label[keep]
        sd2["_DATE"] = sd2[s_date_E].dt.normalize()

        idx_set = pd.date_range(month_start, month_end, freq="D")

        pt2 = sd2.pivot_table(
            index="_DATE",
            columns=["_MAIN", "_LABEL"],
            values=s_amt_L,
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(idx_set, fill_value=0.0)

        det_order = ["VIRTUAL ACCOUNT - BCA", "VIRTUAL ACCOUNT - NON BCA", "E-MONEY"]
        for mname in ["GO SHOW", "ONLINE"]:
            for lab in det_order:
                if (mname, lab) not in pt2.columns:
                    pt2[(mname, lab)] = 0.0

        df3 = pd.DataFrame(index=idx_set)
        for lab in det_order:
            df3[("GO SHOW", lab)] = pt2["GO SHOW"][lab].values
        for lab in det_order:
            df3[("ONLINE", lab)] = pt2["ONLINE"][lab].values

        df3_view = df3.reset_index().rename(columns={"index": "Tanggal"})
        df3_view["Tanggal"] = pd.to_datetime(df3_view["Tanggal"]).dt.date
        df3_view.insert(0, "NO", range(1, len(df3_view) + 1))
        df3_view = _force_multiindex_cols(df3_view)  # <-- FIX

        total_row = {("", "NO"): "", ("", "Tanggal"): "TOTAL"}
        for col in df3.columns:
            total_row[col] = float(df3[col].sum())
        df3_view = pd.concat([df3_view, pd.DataFrame([total_row])], ignore_index=True)

        df3_fmt = df3_view.copy()
        for col in df3_fmt.columns:
            if col in [("", "NO"), ("", "Tanggal")]:
                continue
            if pd.api.types.is_numeric_dtype(df3_fmt[col]):
                df3_fmt[col] = df3_fmt[col].map(_idr_fmt)

        st.dataframe(df3_fmt, use_container_width=True, hide_index=True)

        # ------------------- Download Excel (Detail Settlement) -------------------
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter

        # versi flat untuk excel raw
        raw_flat = df3.reset_index().rename(columns={"index": "Tanggal"})
        raw_flat["Tanggal"] = pd.to_datetime(raw_flat["Tanggal"]).dt.date

        flat_map = {}
        for col in df3.columns:
            main_name, lab = col
            prefix = "GS|" if main_name == "GO SHOW" else "ON|"
            flat_map[col] = prefix + lab

        df3_excel = pd.DataFrame({"Tanggal": raw_flat["Tanggal"]})
        for col in df3.columns:
            df3_excel[flat_map[col]] = df3[col].values

        df3_excel.insert(0, "NO", range(1, len(df3_excel) + 1))

        total_row_excel = {"NO": "", "Tanggal": "TOTAL"}
        for c in df3_excel.columns:
            if c in ("NO", "Tanggal"):
                continue
            total_row_excel[c] = float(df3_excel[c].sum())
        df3_excel = pd.concat([df3_excel, pd.DataFrame([total_row_excel])], ignore_index=True)

        df3_excel_fmt = df3_excel.copy()
        for c in df3_excel_fmt.columns:
            if c in ("NO", "Tanggal"):
                continue
            if pd.api.types.is_numeric_dtype(df3_excel_fmt[c]):
                df3_excel_fmt[c] = df3_excel_fmt[c].map(_idr_fmt)

        bio_settle = io.BytesIO()
        with pd.ExcelWriter(bio_settle, engine="openpyxl") as xw3:
            df3_excel.to_excel(xw3, index=False, sheet_name="Detail_Settlement")

            wsname3 = "Detail_Settlement_View"
            df3_excel_fmt.to_excel(xw3, index=False, header=False, sheet_name=wsname3, startrow=2)

            wb3 = xw3.book
            ws3 = wb3[wsname3]

            cols3 = list(df3_excel.columns)
            top_headers = []
            sub_headers = []
            for c in cols3:
                if c in ("NO", "Tanggal"):
                    top_headers.append("")
                    sub_headers.append(c)
                elif c.startswith("GS|"):
                    top_headers.append("GO SHOW")
                    sub_headers.append(c[3:])
                elif c.startswith("ON|"):
                    top_headers.append("ONLINE")
                    sub_headers.append(c[3:])
                else:
                    top_headers.append("")
                    sub_headers.append(c)

            for j, (top, sub) in enumerate(zip(top_headers, sub_headers), start=1):
                ws3.cell(row=1, column=j, value=top)
                ws3.cell(row=2, column=j, value=sub)

            def _merge_same_run(labels, row_idx):
                start = 0
                while start < len(labels):
                    end = start
                    while end + 1 < len(labels) and labels[end + 1] == labels[start]:
                        end += 1
                    label0 = labels[start]
                    if label0 not in ("", None) and end >= start:
                        ws3.merge_cells(start_row=row_idx, start_column=start + 1,
                                        end_row=row_idx, end_column=end + 1)
                    start = end + 1

            _merge_same_run(top_headers, row_idx=1)

            for j, top in enumerate(top_headers, start=1):
                if top == "":
                    ws3.merge_cells(start_row=1, start_column=j, end_row=2, end_column=j)

            max_col = len(cols3)
            for ccol in range(1, max_col + 1):
                ws3.cell(row=1, column=ccol).font = Font(bold=True)
                ws3.cell(row=2, column=ccol).font = Font(bold=True)
                ws3.cell(row=1, column=ccol).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                ws3.cell(row=2, column=ccol).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws3.row_dimensions[1].height = 22
            ws3.row_dimensions[2].height = 22

            sample_rows = min(50, df3_excel_fmt.shape[0])
            for idx_col, col_name in enumerate(cols3, start=1):
                max_len = max(len(str(col_name)), len(str(sub_headers[idx_col - 1])), len(str(top_headers[idx_col - 1])))
                for r in range(3, 3 + sample_rows):
                    v = ws3.cell(row=r, column=idx_col).value
                    if v is not None:
                        max_len = max(max_len, len(str(v)))
                ws3.column_dimensions[get_column_letter(idx_col)].width = min(max(10, max_len + 2), 45)

        st.download_button(
            "Unduh Excel (Detail Settlement)",
            data=bio_settle.getvalue(),
            file_name=f"detail_settlement_{y}-{m:02d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
