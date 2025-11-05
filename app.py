# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana
- Parameter tanggal = Bulan & Tahun; hasil selalu 1..akhir bulan itu
- Multi-file upload:
    * Tiket Detail (Excel .xls/.xlsx atau .zip berisi .xls/.xlsx/.csv)
    * Settlement Dana (CSV/Excel atau .zip berisi .xls/.xlsx/.csv)
    * Rekening Koran BCA (CSV/Excel atau .zip berisi .xls/.xlsx/.csv)
    * Rekening Koran Non BCA (CSV/Excel atau .zip berisi .xls/.xlsx/.csv)
- Parser uang/tanggal robust (format Eropa & serial Excel)
- TABEL 1: TIKET DETAIL ESPAY = SUM(Tarif) dgn Action Date, Bank='ESPAY', St Bayar='paid'
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

# --- Text normalizer (hapus aksen/diakritik, lowercase) ---
def _norm_str(val) -> str:
    s = "" if val is None else str(val)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

# --- Date parser (day-first + Excel serial) ---
_ddmmyyyy = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")

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
        try:
            return pd.to_datetime(val).normalize()
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    m = _ddmmyyyy.search(s)
    if m:
        d, M, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        try:
            return pd.Timestamp(year=int(y), month=int(M), day=int(d))
        except Exception:
            pass
    for dayfirst in (True, False):
        try:
            d = dtparser.parse(s, dayfirst=dayfirst, fuzzy=True)
            return pd.Timestamp(d.date())
        except Exception:
            continue
    return None

def _read_any(uploaded_file) -> pd.DataFrame:
    """Baca CSV/Excel. .xls → xlrd; fallback pyexcel-xls; terakhir coba openpyxl (kalau salah ekstensi)."""
    if not uploaded_file:
        return pd.DataFrame()
    name = uploaded_file.name.lower()

    try:
        # CSV
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

        # Excel modern
        elif name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="openpyxl")

        # Excel lama
        elif name.endswith(".xls"):
            try:
                uploaded_file.seek(0)
                return pd.read_excel(uploaded_file, engine="xlrd")
            except Exception:
                pass
            try:
                uploaded_file.seek(0)
                raw = uploaded_file.read()
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
                uploaded_file.seek(0)
                return pd.read_excel(uploaded_file, engine="openpyxl")
            except Exception:
                st.error("Gagal membaca file .xls. Pasang 'xlrd' atau 'pyexcel-xls', atau simpan ulang ke .xlsx.")
                return pd.DataFrame()

        else:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)

    except ImportError:
        st.error("Dukungan .xls perlu paket 'xlrd' atau 'pyexcel-xls'. Tambahkan di requirements.txt.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca {uploaded_file.name}: {e}")
        return pd.DataFrame()

def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df.empty:
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
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            df["__source__"] = f.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==== Ekstraksi ZIP → list file-like (BytesIO) yang punya .name ====
def _expand_zip(files):
    """Terima list UploadedFile, kembalikan list file-like:
       - Jika item .zip → ekstrak file di dalamnya (hanya *.csv, *.xls, *.xlsx)
       - Jika bukan .zip → tetap dikembalikan
    """
    if not files:
        return []
    out = []
    allow_ext = (".csv", ".xls", ".xlsx")
    for f in files:
        fname = (f.name or "").lower()
        if fname.endswith(".zip"):
            try:
                f.seek(0)
                with zipfile.ZipFile(f) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner = info.filename
                        inner_lower = inner.lower()
                        if not inner_lower.endswith(allow_ext):
                            continue
                        # abaikan file sistem/mac
                        if inner_lower.startswith("__macosx/") or inner_lower.endswith(".ds_store"):
                            continue
                        data = zf.read(info)
                        bio = io.BytesIO(data)
                        bio.name = f"{f.name}::{inner}"  # agar _read_any tahu ekstensinya
                        out.append(bio)
            except Exception as e:
                st.warning(f"Gagal ekstrak ZIP {f.name}: {e}")
        else:
            out.append(f)
    return out

# ==== KHUSUS RK NON BCA: angkat baris 13/14 menjadi header bila perlu ====
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
            out = df.iloc[r+1:].copy()
            out.columns = cols
            out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
            return out

    scan_max = min(50, len(df))
    for r in range(scan_max):
        if _is_header_row(df.iloc[r]):
            cols = [str(x).strip() for x in df.iloc[r].tolist()]
            out = df.iloc[r+1:].copy()
            out.columns = cols
            out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
            return out

    return df

def _concat_rk_non(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = _read_any(f)
        if df.empty:
            continue
        df = _promote_header(df)
        if df.empty:
            continue
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        df["__source__"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _month_selector() -> Tuple[int, int]:
    from datetime import date
    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [
        ("01","Januari"), ("02","Februari"), ("03","Maret"), ("04","April"),
        ("05","Mei"), ("06","Juni"), ("07","Juli"), ("08","Agustus"),
        ("09","September"), ("10","Oktober"), ("11","November"), ("12","Desember")
    ]
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Tahun", years, index=years.index(today.year))
    with col2:
        month_label = st.selectbox("Bulan", months, index=int(today.strftime("%m"))-1, format_func=lambda x: x[1])
        month = int(month_label[0])
    return year, month

# ---------- App ----------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    # Tambahkan 'zip' di semua uploader yang relevan
    tiket_files = st.file_uploader("Tiket Detail (Excel .xls/.xlsx/.zip)", type=["xls","xlsx","zip"], accept_multiple_files=True)
    settle_files = st.file_uploader("Settlement Dana (CSV/Excel/.zip)", type=["csv","xls","xlsx","zip"], accept_multiple_files=True)
    st.divider()
    st.header("Rekening Koran (opsional, multi-file)")
    rk_bca_files = st.file_uploader("Rekening Koran BCA (CSV/Excel/.zip)", type=["csv","xls","xlsx","zip"], accept_multiple_files=True)
    rk_non_files = st.file_uploader("Rekening Koran Non BCA (CSV/Excel/.zip)", type=["csv","xls","xlsx","zip"], accept_multiple_files=True)

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end   = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode dipakai: {month_start.date()} s/d {month_end.date()}")

    # 3) Proses — filter disembunyikan
    show_preview = False
    show_debug   = False
    go = st.button("Proses", type="primary", use_container_width=True)

# ==== Ekspansi ZIP sebelum dibaca ====
tiket_inputs  = _expand_zip(tiket_files)
settle_inputs = _expand_zip(settle_files)
rk_bca_inputs = _expand_zip(rk_bca_files)
rk_non_inputs = _expand_zip(rk_non_files)

tiket_df   = _concat_files(tiket_inputs)
settle_df  = _concat_files(settle_inputs)
rk_bca_df  = _concat_files(rk_bca_inputs)
rk_non_df  = _concat_rk_non(rk_non_inputs)   # KHUSUS Non BCA → promote header 13/14

if go:
    # ---------------------- Tiket Detail (TABEL 1) ----------------------
    # Kolom tanggal UTAMA: Action Date
    t_date_action = _find_col(tiket_df, ["Action/Action Date","Action Date","Action","Action date"])
    if t_date_action is None:
        st.error("Kolom tanggal 'Action Date' tidak ditemukan pada Tiket Detail.")
        st.stop()

    # Nominal: wajib 'Tarif'
    t_amt_tarif = _find_col(tiket_df, ["Tarif","tarif"])
    if t_amt_tarif is None:
        st.error("Kolom nominal 'Tarif' tidak ditemukan pada Tiket Detail.")
        st.stop()

    # Status & Bank
    t_stat = _find_col(tiket_df, ["St Bayar","Status Bayar","status","status bayar"])
    t_bank = _find_col(tiket_df, ["Bank","Payment Channel","channel","payment method"])
    if t_stat is None or t_bank is None:
        st.error("Kolom 'St Bayar' atau 'Bank' tidak ditemukan pada Tiket Detail.")
        st.stop()

    # ---------------------- Mapping lain (untuk bagian lain aplikasi) ----------------------
    # Settlement Dana (utama/semula)
    s_date_legacy = _find_col(settle_df, ["Transaction Date","Tanggal Transaksi","Tanggal"])
    s_amt_legacy  = _find_col(settle_df, ["Settlement Amount","Amount","Nominal","Jumlah"])
    if s_amt_legacy is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_legacy = settle_df.columns[11]  # kolom L
    if s_date_legacy is None:
        s_date_legacy = _find_col(settle_df, ["Settlement Date","Tanggal Settlement","Settle Date","Tanggal","Setle Date"])
        if s_date_legacy is None and not settle_df.empty and len(settle_df.columns) >= 5:
            s_date_legacy = settle_df.columns[4]  # kolom E

    # Untuk BCA/Non-BCA (pakai E/L/P)
    s_date_E = _find_col(settle_df, ["Settlement Date","Tanggal Settlement","Settle Date","Tanggal"])
    s_amt_L  = _find_col(settle_df, ["Settlement Amount","Amount","Nominal","Jumlah"])
    if s_amt_L is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_L = settle_df.columns[11]
    s_prod_P = _find_col(settle_df, ["Product Name","Produk","Nama Produk"])
    if s_date_E is None and not settle_df.empty and len(settle_df.columns) >= 5:
        s_date_E = settle_df.columns[4]
    if s_prod_P is None and not settle_df.empty and len(settle_df.columns) >= 16:
        s_prod_P = settle_df.columns[15]

    # Validasi minimal Settlement
    missing = []
    for name, col, src in [
        ("Transaction Date/Tanggal Transaksi", s_date_legacy, "Settlement Dana (utama)"),
        ("Settlement Amount/L", s_amt_legacy, "Settlement Dana (utama)"),
    ]:
        if col is None:
            missing.append(f"{src}: {name}")
    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing))
        st.stop()

    # ------------------  TABEL 1: TIKET DETAIL ESPAY  -------------------
    td = tiket_df.copy()
    td[t_date_action] = td[t_date_action].apply(_to_date)
    td = td[~td[t_date_action].isna()]

    # Normalisasi Bank & Status
    td_bank_norm = td[t_bank].apply(_norm_str)
    td_stat_norm = td[t_stat].apply(_norm_str)

    # Filter ketat: Bank == 'espay' dan St Bayar == 'paid'
    bank_mask = td_bank_norm.eq("espay")
    paid_mask = td_stat_norm.eq("paid")
    td = td[bank_mask & paid_mask]

    # Range bulan
    td = td[(td[t_date_action] >= month_start) & (td[t_date_action] <= month_end)]

    # Nominal dari kolom 'Tarif'
    td[t_amt_tarif] = _to_num(td[t_amt_tarif])

    # HAPUS DUPLIKAT IDENTIK (aman saat multi-file)
    td = td.drop_duplicates()

    # Agregasi per tanggal (Action Date)
    tiket_by_date = td.groupby(td[t_date_action].dt.date, dropna=True)[t_amt_tarif].sum()

    # ------------------  Settlement Dana (utama/semula) ------------------
    sd_main = settle_df.copy()
    sd_main[s_date_legacy] = sd_main[s_date_legacy].apply(_to_date)
    sd_main = sd_main[~sd_main[s_date_legacy].isna()]
    sd_main = sd_main[(sd_main[s_date_legacy] >= month_start) & (sd_main[s_date_legacy] <= month_end)]
    sd_main[s_amt_legacy] = _to_num(sd_main[s_amt_legacy])
    settle_by_date_total = sd_main.groupby(sd_main[s_date_legacy].dt.date, dropna=True)[s_amt_legacy].sum()

    # --- Settlement BCA / Non BCA dari E/L/P (untuk split) ---
    bca_series = pd.Series(dtype=float)
    non_bca_series = pd.Series(dtype=float)
    if not settle_df.empty and s_date_E and s_amt_L and s_prod_P:
        sd_bca = settle_df.copy()
        sd_bca[s_date_E] = sd_bca[s_date_E].apply(_to_date)
        sd_bca = sd_bca[~sd_bca[s_date_E].isna()]
        sd_bca = sd_bca[(sd_bca[s_date_E] >= month_start) & (sd_bca[s_date_E] <= month_end)]
        sd_bca[s_amt_L] = _to_num(sd_bca[s_amt_L])
        prod_norm = sd_bca[s_prod_P].astype(str).str.strip().str.casefold()
        bca_mask  = (prod_norm == "bca va online".casefold())
        settle_by_date_bca     = sd_bca.loc[bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()
        settle_by_date_non_bca = sd_bca.loc[~bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()
        bca_series     = settle_by_date_bca
        non_bca_series = settle_by_date_non_bca

    # --- RK: Uang Masuk BCA ---
    uang_masuk_bca = pd.Series(dtype=float)
    if not rk_bca_df.empty:
        rk_tgl_bca  = _find_col(rk_bca_df, ["Tanggal","Date","Tgl","Transaction Date"])
        rk_amt_bca  = _find_col(rk_bca_df, ["mutasi","amount","kredit","credit","cr"])
        rk_ket_bca  = _find_col(rk_bca_df, ["Keterangan","Remark","Deskripsi","Description"])
        if rk_tgl_bca and rk_amt_bca and rk_ket_bca:
            bca = rk_bca_df.copy()
            bca[rk_tgl_bca] = bca[rk_tgl_bca].apply(_to_date)
            bca = bca[~bca[rk_tgl_bca].isna()]
            bca = bca[(bca[rk_tgl_bca] >= month_start) & (bca[rk_tgl_bca] <= month_end)]
            ket_norm = bca[rk_ket_bca].astype(str).str.strip().str.lower()
            mrc_mask = ket_norm.str.contains("mrc", na=False)
            bca = bca[mrc_mask]
            bca[rk_amt_bca] = _to_num(bca[rk_amt_bca])
            uang_masuk_bca = bca.groupby(bca[rk_tgl_bca].dt.date, dropna=True)[rk_amt_bca].sum()

    # --- RK: Uang Masuk NON BCA ---
    uang_masuk_non = pd.Series(dtype=float)
    if not rk_non_df.empty:
        rk_tgl_non  = _find_col(rk_non_df, ["Date","Tanggal","Transaction Date","Tgl"])
        rk_amt_non  = _find_col(rk_non_df, ["credit","kredit","cr","amount"])
        rk_rem_non  = _find_col(rk_non_df, ["Remark","Keterangan","Description","Deskripsi"])
        if rk_tgl_non and rk_amt_non and rk_rem_non:
            nb = rk_non_df.copy()
            nb[rk_tgl_non] = nb[rk_tgl_non].apply(_to_date)
            nb = nb[~nb[rk_tgl_non].isna()]
            nb = nb[(nb[rk_tgl_non] >= month_start) & (nb[rk_tgl_non] <= month_end)]
            rem_norm = nb[rk_rem_non].astype(str).str.strip().str.lower()
            mrc_mask = rem_norm.str.contains("mrc", na=False)
            nb = nb[mrc_mask]
            nb[rk_amt_non] = _to_num(nb[rk_amt_non])
            uang_masuk_non = nb.groupby(nb[rk_tgl_non].dt.date, dropna=True)[rk_amt_non].sum()

    # --- Index tanggal (1..akhir bulan) ---
    idx = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")
    tiket_series        = tiket_by_date.reindex(idx, fill_value=0.0)
    settle_series       = settle_by_date_total.reindex(idx, fill_value=0.0)
    bca_series          = bca_series.reindex(idx, fill_value=0.0)
    non_bca_series      = non_bca_series.reindex(idx, fill_value=0.0)
    total_settle_ser    = (bca_series + non_bca_series).reindex(idx, fill_value=0.0)
    uang_masuk_bca_ser  = uang_masuk_bca.reindex(idx, fill_value=0.0)
    uang_masuk_non_ser  = uang_masuk_non.reindex(idx, fill_value=0.0)
    total_uang_masuk_ser = (uang_masuk_bca_ser + uang_masuk_non_ser).reindex(idx, fill_value=0.0)

    # --- Final table (rename + kapital) ---
    final = pd.DataFrame(index=idx)
    final["TIKET DETAIL ESPAY"]                 = tiket_series.values
    final["SETTLEMENT DANA ESPAY"]             = settle_series.values
    final["SELISIH TIKET DETAIL - SETTLEMENT"] = final["TIKET DETAIL ESPAY"] - final["SETTLEMENT DANA ESPAY"]
    final["SETTLEMENT BCA"]                    = bca_series.values
    final["SETTLEMENT NON BCA"]                = non_bca_series.values
    final["TOTAL SETTLEMENT"]                  = total_settle_ser.values
    final["UANG MASUK BCA"]                    = uang_masuk_bca_ser.values
    final["UANG MASUK NON BCA"]                = uang_masuk_non_ser.values
    final["TOTAL UANG MASUK"]                  = total_uang_masuk_ser.values
    final["SELISIH SETTLEMENT - UANG MASUK"]   = final["TOTAL SETTLEMENT"] - final["TOTAL UANG MASUK"]

    # -------- View + total (tabel utama) --------
    view = final.reset_index()
    idx_col_name = view.columns[0]
    view = view.rename(columns={idx_col_name: "TANGGAL"})
    view.insert(0, "NO", range(1, len(view) + 1))

    total_row = pd.DataFrame([{
        "NO": "",
        "TANGGAL": "TOTAL",
        "TIKET DETAIL ESPAY": final["TIKET DETAIL ESPAY"].sum(),
        "SETTLEMENT DANA ESPAY": final["SETTLEMENT DANA ESPAY"].sum(),
        "SELISIH TIKET DETAIL - SETTLEMENT": final["SELISIH TIKET DETAIL - SETTLEMENT"].sum(),
        "SETTLEMENT BCA": final["SETTLEMENT BCA"].sum(),
        "SETTLEMENT NON BCA": final["SETTLEMENT NON BCA"].sum(),
        "TOTAL SETTLEMENT": final["TOTAL SETTLEMENT"].sum(),
        "UANG MASUK BCA": final["UANG MASUK BCA"].sum(),
        "UANG MASUK NON BCA": final["UANG MASUK NON BCA"].sum(),
        "TOTAL UANG MASUK": final["TOTAL UANG MASUK"].sum(),
        "SELISIH SETTLEMENT - UANG MASUK": final["SELISIH SETTLEMENT - UANG MASUK"].sum(),
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
    for c in [
        "TIKET DETAIL ESPAY","SETTLEMENT DANA ESPAY","SELISIH TIKET DETAIL - SETTLEMENT",
        "SETTLEMENT BCA","SETTLEMENT NON BCA","TOTAL SETTLEMENT",
        "UANG MASUK BCA","UANG MASUK NON BCA","TOTAL UANG MASUK",
        "SELISIH SETTLEMENT - UANG MASUK",
    ]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # ---------- Export ke Excel + MERGE HEADER ----------
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

        # Merge header level-1
        ws.merge_cells(start_row=1, start_column=6, end_row=1, end_column=7)   # SETTLEMENT
        ws.merge_cells(start_row=1, start_column=9, end_row=1, end_column=10)  # UANG MASUK

        # Style
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
        data=bio.getvalue(),  # penting: getvalue() bukan get_value()
        file_name=f"rekonsiliasi_{y}-{m:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

          # ======================================================================
    # ===========  TABEL BARU: DETAIL TIKET (GO SHOW × SUB-KATEGORI)  ======
    # ======================================================================

    # Helper: ambil nama kolom berdasarkan huruf (A..Z, AA..)
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
        idx = n - 1
        return df.columns[idx] if 0 <= idx < len(df.columns) else None

    # Kolom sumber (pakai header kalau ada; fallback ke huruf kolom)
    type_main_col = _find_col(tiket_df, ["Type","Tipe","Jenis"]) or _col_by_letter_local(tiket_df, "B")   # GO SHOW / ONLINE
    bank_col      = _find_col(tiket_df, ["Bank","Payment Channel","channel","payment method"]) or _col_by_letter_local(tiket_df, "I")
    type_sub_col  = (
        _find_col(tiket_df, [
            "Payment Type","Channel Type","Transaction Type","Sub Type",
            "Tipe","Tipe Pembayaran","Jenis Pembayaran","Kategori","Metode","Product Type"
        ]) or _col_by_letter_local(tiket_df, "J")
    )
    date_col      = _find_col(tiket_df, ["Action/Action Date","Action Date","Action","Action date"]) or _col_by_letter_local(tiket_df, "AG")
    tarif_col     = _find_col(tiket_df, ["Tarif","tarif"]) or _col_by_letter_local(tiket_df, "Y")

    required_missing = [n for n, c in [
        ("TYPE (kolom B)", type_main_col),
        ("BANK (kolom I)", bank_col),
        ("TIPE / SUB-TIPE (kolom J)", type_sub_col),
        ("ACTION DATE (kolom AG)", date_col),
        ("TARIF (kolom Y)", tarif_col),
    ] if c is None]

    if required_missing:
        st.warning("Kolom wajib untuk tabel 'Detail Tiket (GO SHOW/ONLINE)' belum lengkap: " + ", ".join(required_missing))
    else:
        tix = tiket_df.copy()
        # Tanggal dari kolom AG
        tix[date_col] = tix[date_col].apply(_to_date)
        tix = tix[~tix[date_col].isna()]
        tix = tix[(tix[date_col] >= month_start) & (tix[date_col] <= month_end)]

        # >>> TIDAK MEMPERDULIKAN ST BAYAR (paid/unpaid sama-sama dihitung)

        # Normalisasi teks
        main_norm = tix[type_main_col].apply(_norm_str)   # kolom B (GO SHOW/ONLINE)
        sub_norm  = tix[type_sub_col].apply(_norm_str)    # kolom J
        bank_norm = tix[bank_col].apply(_norm_str)        # kolom I

        # Amount dari kolom Y = Tarif
        tix[tarif_col] = _to_num(tix[tarif_col])

        # Mask Type
        m_go_show = main_norm.str.fullmatch(r"go\s*show", case=False, na=False) | main_norm.str.contains(r"\bgo\s*show\b", na=False)
        m_online  = main_norm.str.fullmatch(r"online", case=False, na=False)   | main_norm.str.contains(r"\bonline\b",    na=False)

        # Mask Sub-Tipe
        m_prepaid  = sub_norm.str.fullmatch(r"prepaid", na=False) | sub_norm.str.contains(r"\bprepaid\b", na=False)
        m_emoney   = sub_norm.str.fullmatch(r"e[\-\s]*money", na=False) | sub_norm.str.contains(r"\be[\-\s]*money\b|\bemoney\b", na=False)
        m_varetail = (
            sub_norm.str.fullmatch(r"virtual account dan gerai retail", na=False) |
            sub_norm.str.contains(r"virtual\s*account", na=False)
        ) & sub_norm.str.contains(r"gerai|retail", na=False)

        # ================= GO SHOW =================
        tix_gs = tix.loc[m_go_show].drop_duplicates(subset=[date_col, type_main_col, type_sub_col, bank_col, tarif_col])

        s_gs_prepaid_bca     = tix_gs.loc[m_prepaid  & (bank_norm == "bca")    ].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_bri     = tix_gs.loc[m_prepaid  & (bank_norm == "bri")    ].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_bni     = tix_gs.loc[m_prepaid  & (bank_norm == "bni")    ].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_mandiri = tix_gs.loc[m_prepaid  & (bank_norm == "mandiri")].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_emoney_espay    = tix_gs.loc[m_emoney   & (bank_norm == "espay")  ].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_varetail_espay  = tix_gs.loc[m_varetail & (bank_norm == "espay")  ].groupby(tix_gs[date_col].dt.date, dropna=True)[tarif_col].sum()

        # ================= ONLINE =================
        # Khusus ONLINE: Bank harus ESPAY
        tix_on = tix.loc[m_online & (bank_norm == "espay")].drop_duplicates(subset=[date_col, type_main_col, type_sub_col, bank_col, tarif_col])

        s_on_emoney_espay   = tix_on.loc[m_emoney  ].groupby(tix_on[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_on_varetail_espay = tix_on.loc[m_varetail].groupby(tix_on[date_col].dt.date, dropna=True)[tarif_col].sum()

        # Reindex ke kalender bulan
        idx2 = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")

        # Pakai key unik sementara supaya tidak tabrakan nama kolom,
        # lalu MultiIndex akan menaruhnya di bawah "GO SHOW" dan "ONLINE"
        go_show_cols = {
            "GS|PREPAID - BCA":     s_gs_prepaid_bca.reindex(idx2, fill_value=0.0),
            "GS|PREPAID - BRI":     s_gs_prepaid_bri.reindex(idx2, fill_value=0.0),
            "GS|PREPAID - BNI":     s_gs_prepaid_bni.reindex(idx2, fill_value=0.0),
            "GS|PREPAID - MANDIRI": s_gs_prepaid_mandiri.reindex(idx2, fill_value=0.0),
            "GS|E-MONEY - ESPAY":   s_gs_emoney_espay.reindex(idx2, fill_value=0.0),
            "GS|VIRTUAL ACCOUNT DAN GERAI RETAIL - ESPAY": s_gs_varetail_espay.reindex(idx2, fill_value=0.0),
        }
        online_cols = {
            "ON|E-MONEY - ESPAY":                      s_on_emoney_espay.reindex(idx2, fill_value=0.0),
            "ON|VIRTUAL ACCOUNT & GERAI RETAIL - ESPAY": s_on_varetail_espay.reindex(idx2, fill_value=0.0),
        }

        detail_mix = pd.DataFrame(index=idx2)
        for k, ser in {**go_show_cols, **online_cols}.items():
            detail_mix[k] = ser.values

        # ===== Tampilkan dengan MERGED HEADER "GO SHOW" dan "ONLINE" =====
        st.subheader("Detail Tiket per Tanggal — TYPE: GO SHOW & ONLINE × SUB-TIPE (J) [SEMUA STATUS]")
        df2 = detail_mix.reset_index()
        df2.insert(0, "NO", range(1, len(df2) + 1))

        # Subtotal (TOTAL)
        total_row = {"NO": "", "Tanggal": "TOTAL"}
        for k in detail_mix.columns:
            total_row[k] = float(detail_mix[k].sum())
        df2 = pd.concat([df2, pd.DataFrame([total_row])], ignore_index=True)

        # Format rupiah untuk kolom numerik
        from pandas.api.types import is_numeric_dtype
        df2_fmt = df2.copy()
        for c in df2_fmt.columns:
            if c in ("NO", "Tanggal"):
                continue
            if is_numeric_dtype(df2_fmt[c]):
                df2_fmt[c] = df2_fmt[c].apply(_idr_fmt)

        # Siapkan MultiIndex header
        def _strip_prefix(col_name: str) -> str:
            return col_name.split("|", 1)[1] if "|" in col_name else col_name

        # urutan kolom final: NO, Tanggal, GO SHOW..., ONLINE...
        ordered_keys = list(go_show_cols.keys()) + list(online_cols.keys())
        df2_fmt = df2_fmt[["NO", "Tanggal"] + ordered_keys]

        top = [("", "NO"), ("", "Tanggal")] \
              + [("GO SHOW", _strip_prefix(k)) for k in go_show_cols.keys()] \
              + [("ONLINE",  _strip_prefix(k)) for k in online_cols.keys()]

        df2_fmt_mi = df2_fmt.copy()
        df2_fmt_mi.columns = pd.MultiIndex.from_tuples(top)

        st.dataframe(df2_fmt_mi, use_container_width=True, hide_index=True)
