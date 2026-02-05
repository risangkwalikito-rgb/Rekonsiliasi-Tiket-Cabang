from __future__ import annotations

import calendar
import io
import re
import unicodedata
import zipfile
from typing import List, Optional, Tuple

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


def _norm_str(val) -> str:
    s = "" if val is None else str(val)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


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


def _fill_action_from_created(df: pd.DataFrame, action_col: str, created_col: str) -> pd.DataFrame:
    if df is None or df.empty or not action_col or not created_col:
        return df

    action_raw = df[action_col]
    action_str = action_raw.astype(str).str.strip().str.lower()
    empty_mask = action_raw.isna() | action_str.eq("") | action_str.eq("nan") | action_str.eq("none")
    if not empty_mask.any():
        return df

    created_str = df[created_col].astype(str).str.strip()

    dt_dayfirst = pd.to_datetime(created_str, errors="coerce", dayfirst=True)
    dt_monthfirst = pd.to_datetime(created_str, errors="coerce", dayfirst=False)
    created_dt = dt_dayfirst.fillna(dt_monthfirst)

    filled = created_dt.dt.strftime("%d/%m/%Y")

    fallback = created_str.str.slice(0, 10)
    ymd = fallback.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    fallback_norm = fallback.copy()
    fallback_norm.loc[ymd] = (
        fallback.loc[ymd].str.slice(8, 10) + "/"
        + fallback.loc[ymd].str.slice(5, 7) + "/"
        + fallback.loc[ymd].str.slice(0, 4)
    )

    filled = filled.where(created_dt.notna(), fallback_norm)

    df.loc[empty_mask, action_col] = filled.loc[empty_mask]
    return df


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

        if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="openpyxl")

        if name.endswith(".xls"):
            try:
                uploaded_file.seek(0)
                return pd.read_excel(uploaded_file, engine="xlrd")
            except Exception:
                pass
            try:
                uploaded_file.seek(0)
                raw = uploaded_file.read()
                from pyexcel_xls import get_data  # type: ignore

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


def _idr_fmt(val) -> str:
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


def _style_right(df: pd.DataFrame):
    if df is None or getattr(df, "empty", True):
        return df

    cols = list(df.columns)

    def _is_no(col):
        return (isinstance(col, tuple) and str(col[-1]) == "NO") or (not isinstance(col, tuple) and str(col) == "NO")

    def _is_tgl(col):
        return (isinstance(col, tuple) and str(col[-1]) in ("TANGGAL", "Tanggal")) or (
            not isinstance(col, tuple) and str(col) in ("TANGGAL", "Tanggal")
        )

    no_cols = [c for c in cols if _is_no(c)]
    tgl_cols = [c for c in cols if _is_tgl(c)]
    right_cols = [c for c in cols if c not in set(no_cols + tgl_cols)]

    sty = df.style
    sty = sty.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

    if no_cols:
        sty = sty.set_properties(subset=no_cols, **{"text-align": "center"})
    if tgl_cols:
        sty = sty.set_properties(subset=tgl_cols, **{"text-align": "left"})
    if right_cols:
        sty = sty.set_properties(subset=right_cols, **{"text-align": "right"})

    return sty


def _concat_files(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = _read_any(f)
        if not df.empty:
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            df["__source__"] = getattr(f, "name", "file")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _expand_zip(files):
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
                        if inner_lower.startswith("__macosx/") or inner_lower.endswith(".ds_store"):
                            continue
                        data = zf.read(info)
                        bio = io.BytesIO(data)
                        bio.name = f"{f.name}::{inner}"
                        out.append(bio)
            except Exception as e:
                st.warning(f"Gagal ekstrak ZIP {f.name}: {e}")
        else:
            out.append(f)
    return out


def _promote_header(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _is_header_row(sr: pd.Series) -> bool:
        vals = [str(v).strip().lower() for v in sr.fillna("")]
        keys = [
            "date",
            "tanggal",
            "transaction date",
            "tgl",
            "remark",
            "keterangan",
            "description",
            "deskripsi",
            "credit",
            "kredit",
            "cr",
            "amount",
            "jumlah",
        ]
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
        df["__source__"] = getattr(f, "name", "file")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _month_selector() -> Tuple[int, int]:
    from datetime import date

    today = date.today()
    years = list(range(today.year - 5, today.year + 2))
    months = [
        ("01", "Januari"),
        ("02", "Februari"),
        ("03", "Maret"),
        ("04", "April"),
        ("05", "Mei"),
        ("06", "Juni"),
        ("07", "Juli"),
        ("08", "Agustus"),
        ("09", "September"),
        ("10", "Oktober"),
        ("11", "November"),
        ("12", "Desember"),
    ]
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Tahun", years, index=years.index(today.year))
    with col2:
        month_label = st.selectbox("Bulan", months, index=int(today.strftime("%m")) - 1, format_func=lambda x: x[1])
        month = int(month_label[0])
    return year, month


def _norm_order_id(x) -> str:
    s = "" if x is None else str(x)
    s = s.strip().casefold()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z_]", "", s)
    if s.endswith("ord") and not s.endswith("_ord") and not s.startswith("ord"):
        s = s[:-3] + "_ord"
    return s


# ---------- App ----------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] td { text-align: right !important; }
    div[data-testid="stDataFrame"] th { text-align: center !important; }
    div[data-testid="stDataFrame"] td:nth-child(1) { text-align: center !important; }
    div[data-testid="stDataFrame"] td:nth-child(2) { text-align: left !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "HASIL" not in st.session_state:
    st.session_state["HASIL"] = {}

with st.sidebar:
    st.header("1) Upload Sumber (multi-file)")
    tiket_files = st.file_uploader(
        "Tiket Detail (CSV/Excel .csv/.xls/.xlsx/.zip)",
        type=["csv", "xls", "xlsx", "zip"],
        accept_multiple_files=True,
    )
    settle_files = st.file_uploader(
        "Settlement Dana (CSV/Excel/.zip)",
        type=["csv", "xls", "xlsx", "zip"],
        accept_multiple_files=True,
    )
    st.divider()
    st.header("Rekening Koran (opsional, multi-file)")
    rk_bca_files = st.file_uploader(
        "Rekening Koran BCA (CSV/Excel/.zip)",
        type=["csv", "xls", "xlsx", "zip"],
        accept_multiple_files=True,
    )
    rk_non_files = st.file_uploader(
        "Rekening Koran Non BCA (CSV/Excel/.zip)",
        type=["csv", "xls", "xlsx", "zip"],
        accept_multiple_files=True,
    )

    st.header("2) Parameter Bulan & Tahun (WAJIB)")
    y, m = _month_selector()
    month_start = pd.Timestamp(y, m, 1)
    month_end = pd.Timestamp(y, m, calendar.monthrange(y, m)[1])
    st.caption(f"Periode dipakai: {month_start.date()} s/d {month_end.date()}")

    go = st.button("Proses", type="primary", use_container_width=True)

    if st.session_state["HASIL"]:
        if st.button("Reset hasil (hapus tampilan tersimpan)", type="secondary", use_container_width=True):
            st.session_state["HASIL"] = {}
            st.rerun()

if go:
    tiket_inputs = _expand_zip(tiket_files)
    settle_inputs = _expand_zip(settle_files)
    rk_bca_inputs = _expand_zip(rk_bca_files)
    rk_non_inputs = _expand_zip(rk_non_files)

    tiket_df = _concat_files(tiket_inputs)
    settle_df = _concat_files(settle_inputs)
    rk_bca_df = _concat_files(rk_bca_inputs)
    rk_non_df = _concat_rk_non(rk_non_inputs)

    if tiket_df.empty:
        st.error("Tiket Detail kosong / belum diupload.")
        st.stop()
    if settle_df.empty:
        st.error("Settlement Dana kosong / belum diupload.")
        st.stop()

    # ---------------------- Tiket Detail (TABEL 1) ----------------------
    t_date_action = _find_col(tiket_df, ["Action Date", "Action"])
    if t_date_action is None:
        st.error("Kolom tanggal 'Action Date' / 'Action' tidak ditemukan pada Tiket Detail.")
        st.stop()

    t_amt_tarif = _find_col(tiket_df, ["Tarif", "tarif"])
    if t_amt_tarif is None:
        st.error("Kolom nominal 'Tarif' tidak ditemukan pada Tiket Detail.")
        st.stop()

    t_stat = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status", "status bayar"])
    t_bank = _find_col(tiket_df, ["Bank", "Payment Channel", "channel", "payment method"])
    if t_stat is None or t_bank is None:
        st.error("Kolom 'St Bayar' atau 'Bank' tidak ditemukan pada Tiket Detail.")
        st.stop()

    # ---------------------- Mapping Settlement (utama/legacy) ----------------------
    s_date_legacy = _find_col(settle_df, ["Transaction Date", "Tanggal Transaksi", "Tanggal"])
    s_amt_legacy = _find_col(settle_df, ["Settlement Amount", "Amount", "Nominal", "Jumlah"])
    if s_amt_legacy is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_legacy = settle_df.columns[11]
    if s_date_legacy is None:
        s_date_legacy = _find_col(settle_df, ["Settlement Date", "Tanggal Settlement", "Settle Date", "Tanggal", "Setle Date"])
        if s_date_legacy is None and not settle_df.empty and len(settle_df.columns) >= 5:
            s_date_legacy = settle_df.columns[4]

    if s_date_legacy is None or s_amt_legacy is None:
        st.error("Kolom wajib Settlement Dana tidak ditemukan (tanggal/amount).")
        st.stop()

    # Untuk BCA/Non-BCA (pakai E/L/P)
    s_date_E = _find_col(settle_df, ["Settlement Date", "Tanggal Settlement", "Settle Date", "Tanggal"])
    s_amt_L = _find_col(settle_df, ["Settlement Amount", "Amount", "Nominal", "Jumlah"])
    if s_amt_L is None and not settle_df.empty and len(settle_df.columns) >= 12:
        s_amt_L = settle_df.columns[11]
    s_prod_P = _find_col(settle_df, ["Product Name", "Produk", "Nama Produk"])
    if s_date_E is None and not settle_df.empty and len(settle_df.columns) >= 5:
        s_date_E = settle_df.columns[4]
    if s_prod_P is None and not settle_df.empty and len(settle_df.columns) >= 16:
        s_prod_P = settle_df.columns[15]

    # ------------------  TABEL 1: TIKET DETAIL ESPAY (PAID ONLY) -------------------
    td = tiket_df.copy()
    t_created = _find_col(tiket_df, ["Created", "Created Date", "Created At", "Created Time"])
    if t_created is not None:
        td = _fill_action_from_created(td, t_date_action, t_created)

    td[t_date_action] = pd.to_datetime(td[t_date_action].apply(_to_date), errors="coerce")
    td = td[~td[t_date_action].isna()]

    td_bank_norm = td[t_bank].apply(_norm_str)
    td_stat_norm = td[t_stat].apply(_norm_str)
    td = td[td_bank_norm.eq("espay") & td_stat_norm.eq("paid")]
    td = td[(td[t_date_action] >= month_start) & (td[t_date_action] <= month_end)]

    td[t_amt_tarif] = _to_num(td[t_amt_tarif])

    # FIX: ABAIKAN DUPLICATE UNTUK TIKET DETAIL (jangan drop_duplicates)
    tiket_by_date = td.groupby(td[t_date_action].dt.date, dropna=True)[t_amt_tarif].sum()

    # ------------------  Settlement Dana (utama/legacy) ------------------
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
        bca_mask = prod_norm.eq("bca va online".casefold())
        settle_by_date_bca = sd_bca.loc[bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()
        settle_by_date_non_bca = sd_bca.loc[~bca_mask].groupby(sd_bca[s_date_E].dt.date, dropna=True)[s_amt_L].sum()
        bca_series = settle_by_date_bca
        non_bca_series = settle_by_date_non_bca

    # --- RK: Uang Masuk BCA ---
    uang_masuk_bca = pd.Series(dtype=float)
    if not rk_bca_df.empty:
        rk_tgl_bca = _find_col(rk_bca_df, ["Tanggal", "Date", "Tgl", "Transaction Date"])
        rk_amt_bca = _find_col(rk_bca_df, ["mutasi", "amount", "kredit", "credit", "cr"])
        rk_ket_bca = _find_col(rk_bca_df, ["Keterangan", "Remark", "Deskripsi", "Description"])
        if rk_tgl_bca and rk_amt_bca and rk_ket_bca:
            bca = rk_bca_df.copy()
            bca[rk_tgl_bca] = bca[rk_tgl_bca].apply(_to_date)
            bca = bca[~bca[rk_tgl_bca].isna()]
            bca = bca[(bca[rk_tgl_bca] >= month_start) & (bca[rk_tgl_bca] <= month_end)]
            ket_norm = bca[rk_ket_bca].astype(str).str.strip().str.lower()
            bca = bca[ket_norm.str.contains("mrc", na=False)]
            bca[rk_amt_bca] = _to_num(bca[rk_amt_bca])
            uang_masuk_bca = bca.groupby(bca[rk_tgl_bca].dt.date, dropna=True)[rk_amt_bca].sum()

    # --- RK: Uang Masuk NON BCA ---
    uang_masuk_non = pd.Series(dtype=float)
    if not rk_non_df.empty:
        rk_tgl_non = _find_col(rk_non_df, ["Date", "Tanggal", "Transaction Date", "Tgl"])
        rk_amt_non = _find_col(rk_non_df, ["credit", "kredit", "cr", "amount"])
        rk_rem_non = _find_col(rk_non_df, ["Remark", "Keterangan", "Description", "Deskripsi"])
        if rk_tgl_non and rk_amt_non and rk_rem_non:
            nb = rk_non_df.copy()
            nb[rk_tgl_non] = nb[rk_tgl_non].apply(_to_date)
            nb = nb[~nb[rk_tgl_non].isna()]
            nb = nb[(nb[rk_tgl_non] >= month_start) & (nb[rk_tgl_non] <= month_end)]
            rem_norm = nb[rk_rem_non].astype(str).str.strip().str.lower()
            nb = nb[rem_norm.str.contains("mrc", na=False)]
            nb[rk_amt_non] = _to_num(nb[rk_amt_non])
            uang_masuk_non = nb.groupby(nb[rk_tgl_non].dt.date, dropna=True)[rk_amt_non].sum()

    # --- Index tanggal (1..akhir bulan) ---
    idx = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")
    tiket_series = tiket_by_date.reindex(idx, fill_value=0.0)
    settle_series = settle_by_date_total.reindex(idx, fill_value=0.0)
    bca_series = bca_series.reindex(idx, fill_value=0.0)
    non_bca_series = non_bca_series.reindex(idx, fill_value=0.0)
    total_settle_ser = (bca_series + non_bca_series).reindex(idx, fill_value=0.0)
    uang_masuk_bca_ser = uang_masuk_bca.reindex(idx, fill_value=0.0)
    uang_masuk_non_ser = uang_masuk_non.reindex(idx, fill_value=0.0)
    total_uang_masuk_ser = (uang_masuk_bca_ser + uang_masuk_non_ser).reindex(idx, fill_value=0.0)

    final = pd.DataFrame(index=idx)
    final["TIKET DETAIL ESPAY"] = tiket_series.values
    final["SETTLEMENT DANA ESPAY"] = settle_series.values
    final["SELISIH TIKET DETAIL - SETTLEMENT"] = final["TIKET DETAIL ESPAY"] - final["SETTLEMENT DANA ESPAY"]
    final["SETTLEMENT BCA"] = bca_series.values
    final["SETTLEMENT NON BCA"] = non_bca_series.values
    final["TOTAL SETTLEMENT"] = (bca_series.values + non_bca_series.values)
    final["UANG MASUK BCA"] = uang_masuk_bca_ser.values
    final["UANG MASUK NON BCA"] = uang_masuk_non_ser.values
    final["TOTAL UANG MASUK"] = total_uang_masuk_ser.values
    final["SELISIH SETTLEMENT - UANG MASUK"] = final["TOTAL SETTLEMENT"] - final["TOTAL UANG MASUK"]

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
    for c in ordered_cols:
        if c in ("NO", "TANGGAL"):
            continue
        fmt[c] = fmt[c].apply(_idr_fmt)

    # =========================
    # EXPORT EXCEL REKON
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

        ws.merge_cells(start_row=1, start_column=6, end_row=1, end_column=7)
        ws.merge_cells(start_row=1, start_column=9, end_row=1, end_column=10)

        max_col = ws.max_column
        for c in range(1, max_col + 1):
            ws.cell(row=1, column=c).font = Font(bold=True)
            ws.cell(row=2, column=c).font = Font(bold=True)
            ws.cell(row=1, column=c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.cell(row=2, column=c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 22
        ws.row_dimensions[2].height = 22

    # ======================================================================
    # ===========  TABEL: DETAIL TIKET (GO SHOW × SUB-KATEGORI)  ===========
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

    type_main_col = _find_col(tiket_df, ["Type", "Tipe", "Jenis"]) or _col_by_letter_local(tiket_df, "B")
    bank_col = _find_col(tiket_df, ["Bank", "Payment Channel", "channel", "payment method"]) or _col_by_letter_local(tiket_df, "I")
    type_sub_col = (
        _find_col(
            tiket_df,
            [
                "Payment Type", "Channel Type", "Transaction Type", "Sub Type",
                "Tipe", "Tipe Pembayaran", "Jenis Pembayaran", "Kategori", "Metode", "Product Type",
            ],
        )
        or _col_by_letter_local(tiket_df, "J")
    )
    date_col = _find_col(tiket_df, ["Action Date", "Action"]) or _col_by_letter_local(tiket_df, "AG")
    tarif_col = _find_col(tiket_df, ["Tarif", "tarif"]) or _col_by_letter_local(tiket_df, "Y")
    status_col = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status", "status bayar"])

    required_missing = [
        n for n, c in [
            ("TYPE (kolom B)", type_main_col),
            ("BANK (kolom I)", bank_col),
            ("TIPE / SUB-TIPE (kolom J)", type_sub_col),
            ("ACTION/Action Date (kolom AG)", date_col),
            ("TARIF (kolom Y)", tarif_col),
            ("ST BAYAR / STATUS BAYAR", status_col),
        ]
        if c is None
    ]

    df2_fmt_mi = None
    if required_missing:
        st.warning("Kolom wajib untuk tabel 'Detail Tiket (GO SHOW/ONLINE) [PAID]' belum lengkap: " + ", ".join(required_missing))
    else:
        tix = tiket_df.copy()
        if t_created is not None:
            tix = _fill_action_from_created(tix, date_col, t_created)

        tix[date_col] = pd.to_datetime(tix[date_col].apply(_to_date), errors="coerce")
        tix = tix[~tix[date_col].isna()]
        tix = tix[(tix[date_col] >= month_start) & (tix[date_col] <= month_end)]

        stat_norm = tix[status_col].apply(_norm_str)
        tix = tix[stat_norm.eq("paid") | stat_norm.str.contains(r"\bpaid\b", na=False)]

        main_norm_all = tix[type_main_col].apply(_norm_str)
        sub_norm_all = tix[type_sub_col].apply(_norm_str)
        bank_norm_all = tix[bank_col].apply(_norm_str)

        tix[tarif_col] = _to_num(tix[tarif_col])

        m_go_show = (main_norm_all == "go show") | main_norm_all.str.contains(r"\bgo\s*show\b", na=False)
        m_online = (main_norm_all == "online") | main_norm_all.str.contains(r"\bonline\b", na=False)

        m_prepaid_all = (sub_norm_all == "prepaid") | sub_norm_all.str.contains(r"\bprepaid\b", na=False)
        m_emoney_all = (sub_norm_all == "e-money") | sub_norm_all.str.contains(r"\be[-\s]*money\b|\bemoney\b", na=False)
        m_varetail_all = sub_norm_all.str.contains(r"virtual\s*account", na=False) & sub_norm_all.str.contains(r"gerai|retail", na=False)
        m_cash_all = (sub_norm_all == "cash") | sub_norm_all.str.contains(r"\bcash\b", na=False)

        # TRANSFER (kolom J mengandung "transfer")
        m_tf_all = sub_norm_all.str.contains(r"\btransfer\b", na=False)

        idx2 = pd.Index(pd.date_range(month_start, month_end, freq="D").date, name="Tanggal")

        # ---------------- GO SHOW existing ----------------
        s_gs_prepaid_bca = tix.loc[m_go_show & m_prepaid_all & bank_norm_all.eq("bca")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_bri = tix.loc[m_go_show & m_prepaid_all & bank_norm_all.eq("bri")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_bni = tix.loc[m_go_show & m_prepaid_all & bank_norm_all.eq("bni")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_prepaid_mandiri = tix.loc[m_go_show & m_prepaid_all & bank_norm_all.eq("mandiri")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_emoney_espay = tix.loc[m_go_show & m_emoney_all & bank_norm_all.eq("espay")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_varetail_espay = tix.loc[m_go_show & m_varetail_all & bank_norm_all.eq("espay")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_cash_asdp = tix.loc[m_go_show & m_cash_all & bank_norm_all.eq("asdp")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()

        # ---------------- GO SHOW TRANSFER (BARU) ----------------
        m_bank_ptpos = bank_norm_all.str.contains(r"\b(pt\.?\s*pos|ptpos|pos)\b", na=False)

        s_gs_tf_ptpos = tix.loc[m_go_show & m_tf_all & m_bank_ptpos].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_tf_bri = tix.loc[m_go_show & m_tf_all & bank_norm_all.eq("bri")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_tf_bni = tix.loc[m_go_show & m_tf_all & bank_norm_all.eq("bni")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_tf_mandiri = tix.loc[m_go_show & m_tf_all & bank_norm_all.eq("mandiri")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_gs_tf_bca = tix.loc[m_go_show & m_tf_all & bank_norm_all.eq("bca")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()

        go_show_cols = {
            "PREPAID - BCA": s_gs_prepaid_bca.reindex(idx2, fill_value=0.0),
            "PREPAID - BRI": s_gs_prepaid_bri.reindex(idx2, fill_value=0.0),
            "PREPAID - BNI": s_gs_prepaid_bni.reindex(idx2, fill_value=0.0),
            "PREPAID - MANDIRI": s_gs_prepaid_mandiri.reindex(idx2, fill_value=0.0),

            # KOLOM BARU (TRANSFER) MASUK GRUP GO SHOW
            "TF - PT.POS": s_gs_tf_ptpos.reindex(idx2, fill_value=0.0),
            "TF - BRI": s_gs_tf_bri.reindex(idx2, fill_value=0.0),
            "TF - BNI": s_gs_tf_bni.reindex(idx2, fill_value=0.0),
            "TF - MANDIRI": s_gs_tf_mandiri.reindex(idx2, fill_value=0.0),
            "TF - BCA": s_gs_tf_bca.reindex(idx2, fill_value=0.0),

            "E-MONEY - ESPAY": s_gs_emoney_espay.reindex(idx2, fill_value=0.0),
            "VIRTUAL ACCOUNT DAN GERAI RETAIL - ESPAY": s_gs_varetail_espay.reindex(idx2, fill_value=0.0),
            "CASH - ASDP": s_gs_cash_asdp.reindex(idx2, fill_value=0.0),
        }

        # ---------------- ONLINE existing ----------------
        m_emoney_on = (sub_norm_all == "e-money") | sub_norm_all.str.contains(r"\be[-\s]*money\b|\bemoney\b", na=False)
        m_varetail_on = sub_norm_all.str.contains(r"virtual\s*account", na=False) & sub_norm_all.str.contains(r"gerai|retail", na=False)
        m_cash_on = (sub_norm_all == "cash") | sub_norm_all.str.contains(r"\bcash\b", na=False)
        m_bank_espay_on = bank_norm_all.eq("espay")

        s_on_emoney_espay = tix.loc[m_online & m_bank_espay_on & m_emoney_on].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_on_varetail_espay = tix.loc[m_online & m_bank_espay_on & m_varetail_on].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()
        s_on_cash_asdp = tix.loc[m_online & m_cash_on & bank_norm_all.eq("asdp")].groupby(tix[date_col].dt.date, dropna=True)[tarif_col].sum()

        online_cols = {
            "E-MONEY - ESPAY": s_on_emoney_espay.reindex(idx2, fill_value=0.0),
            "VIRTUAL ACCOUNT & GERAI RETAIL - ESPAY": s_on_varetail_espay.reindex(idx2, fill_value=0.0),
            "CASH - ASDP": s_on_cash_asdp.reindex(idx2, fill_value=0.0),
        }

        detail_mix = pd.DataFrame(index=idx2)

        for k, ser in go_show_cols.items():
            detail_mix[f"GS|{k}"] = ser.values
        detail_mix["GS|SUBTOTAL"] = pd.DataFrame(go_show_cols, index=idx2).sum(axis=1).values

        for k, ser in online_cols.items():
            detail_mix[f"ON|{k}"] = ser.values
        detail_mix["ON|SUBTOTAL"] = pd.DataFrame(online_cols, index=idx2).sum(axis=1).values

        detail_mix["GT|GRAND TOTAL"] = detail_mix["GS|SUBTOTAL"] + detail_mix["ON|SUBTOTAL"]

        df2 = detail_mix.reset_index()
        df2.insert(0, "NO", range(1, len(df2) + 1))

        total_row2 = {"NO": "", "Tanggal": "TOTAL"}
        for k in detail_mix.columns:
            total_row2[k] = float(detail_mix[k].sum())
        df2 = pd.concat([df2, pd.DataFrame([total_row2])], ignore_index=True)

        df2_fmt = df2.copy()
        for c in df2_fmt.columns:
            if c in ("NO", "Tanggal"):
                continue
            df2_fmt[c] = df2_fmt[c].apply(_idr_fmt)

        def _strip_prefix(col_name: str) -> tuple[str, str]:
            if col_name.startswith("GS|"):
                return ("GO SHOW", col_name[3:])
            if col_name.startswith("ON|"):
                return ("ONLINE", col_name[3:])
            if col_name.startswith("GT|"):
                return ("GRAND TOTAL", "")
            return ("", col_name)

        ordered_keys = [k for k in detail_mix.columns if k.startswith("GS|") and k != "GS|SUBTOTAL"] + ["GS|SUBTOTAL"]
        ordered_keys += [k for k in detail_mix.columns if k.startswith("ON|") and k != "ON|SUBTOTAL"] + ["ON|SUBTOTAL"]
        ordered_keys += ["GT|GRAND TOTAL"]

        df2_fmt = df2_fmt[["NO", "Tanggal"] + ordered_keys]
        top = [("", "NO"), ("", "Tanggal")] + [_strip_prefix(k) for k in ordered_keys]
        df2_fmt_mi = df2_fmt.copy()
        df2_fmt_mi.columns = pd.MultiIndex.from_tuples(top)

    # ======================================================================
    # ===================  SUMMARY SELISIH (ORDER ID)  =====================
    # ======================================================================

    periode = f"{y}-{m:02d}"
    st.session_state["HASIL"]["rekon"] = {"periode": periode, "table": fmt, "excel_bytes": bio.getvalue()}
    if df2_fmt_mi is not None:
        st.session_state["HASIL"]["detail_tiket"] = {"periode": periode, "table": df2_fmt_mi}
    else:
        st.session_state["HASIL"].pop("detail_tiket", None)

    # SUMMARY: cari Only Ticket vs Settlement berdasarkan Order ID (tanpa tampil tabel besar)
    t_order = _find_col(tiket_df, ["Order ID", "OrderId", "OrderID", "Order Number", "Order No", "order id"])
    s_order = _find_col(settle_df, ["Order ID", "OrderId", "OrderID", "Order Number", "Order No", "order id"])

    s_date_for_oid = s_date_E or s_date_legacy
    s_amt_for_oid = s_amt_L or s_amt_legacy

    if t_order and s_order and s_date_for_oid and s_amt_for_oid:
        # Tiket ESPAY paid dalam periode
        tix_oid = tiket_df.copy()
        if t_created is not None:
            tix_oid = _fill_action_from_created(tix_oid, t_date_action, t_created)
        tix_oid[t_date_action] = pd.to_datetime(tix_oid[t_date_action].apply(_to_date), errors="coerce")
        tix_oid = tix_oid[~tix_oid[t_date_action].isna()]
        tix_oid = tix_oid[(tix_oid[t_date_action] >= month_start) & (tix_oid[t_date_action] <= month_end)]
        tix_oid = tix_oid[tix_oid[t_bank].apply(_norm_str).eq("espay") & tix_oid[t_stat].apply(_norm_str).eq("paid")]
        tix_oid[t_amt_tarif] = _to_num(tix_oid[t_amt_tarif])
        tix_oid["__oid_key__"] = tix_oid[t_order].apply(_norm_order_id)

        tiket_oid = tix_oid.groupby("__oid_key__", dropna=False).agg(
            TIKET_TARIF=(t_amt_tarif, "sum"),
        ).reset_index()

        # Settlement dalam periode
        sd_oid = settle_df.copy()
        sd_oid[s_date_for_oid] = pd.to_datetime(sd_oid[s_date_for_oid].apply(_to_date), errors="coerce")
        sd_oid = sd_oid[~sd_oid[s_date_for_oid].isna()]
        sd_oid = sd_oid[(sd_oid[s_date_for_oid] >= month_start) & (sd_oid[s_date_for_oid] <= month_end)]
        sd_oid[s_amt_for_oid] = _to_num(sd_oid[s_amt_for_oid])
        sd_oid["__oid_key__"] = sd_oid[s_order].apply(_norm_order_id)

        settle_oid = sd_oid.groupby("__oid_key__", dropna=False).agg(
            SETTLE_AMOUNT=(s_amt_for_oid, "sum"),
        ).reset_index()

        rekon_oid = tiket_oid.merge(settle_oid, on="__oid_key__", how="outer")
        rekon_oid["TIKET_TARIF"] = rekon_oid["TIKET_TARIF"].fillna(0.0)
        rekon_oid["SETTLE_AMOUNT"] = rekon_oid["SETTLE_AMOUNT"].fillna(0.0)
        rekon_oid["SELISIH"] = rekon_oid["TIKET_TARIF"] - rekon_oid["SETTLE_AMOUNT"]

        has_t = rekon_oid["TIKET_TARIF"].abs() > 0
        has_s = rekon_oid["SETTLE_AMOUNT"].abs() > 0

        status = np.where(
            has_t & ~has_s, "Only Ticket",
            np.where(~has_t & has_s, "Only Settlement",
                     np.where(rekon_oid["SELISIH"].abs() < 0.5, "Match", "Mismatch"))
        )
        rekon_oid["STATUS"] = status

        summary_selisih = {
            "TOTAL TIKET (ESPAY)": float(rekon_oid["TIKET_TARIF"].sum()),
            "TOTAL SETTLEMENT (ESPAY)": float(rekon_oid["SETTLE_AMOUNT"].sum()),
            "SELISIH TOTAL (TIKET - SETTLEMENT)": float(rekon_oid["SELISIH"].sum()),
            "ABS SELISIH TOTAL": float(rekon_oid["SELISIH"].abs().sum()),
            "ONLY TICKET (COUNT)": int((rekon_oid["STATUS"] == "Only Ticket").sum()),
            "ONLY TICKET (NOMINAL)": float(rekon_oid.loc[rekon_oid["STATUS"] == "Only Ticket", "TIKET_TARIF"].sum()),
            "ONLY SETTLEMENT (COUNT)": int((rekon_oid["STATUS"] == "Only Settlement").sum()),
            "ONLY SETTLEMENT (NOMINAL)": float(rekon_oid.loc[rekon_oid["STATUS"] == "Only Settlement", "SETTLE_AMOUNT"].sum()),
            "MISMATCH (COUNT)": int((rekon_oid["STATUS"] == "Mismatch").sum()),
            "MISMATCH (ABS NOMINAL)": float(rekon_oid.loc[rekon_oid["STATUS"] == "Mismatch", "SELISIH"].abs().sum()),
        }

        bio_sum = io.BytesIO()
        with pd.ExcelWriter(bio_sum, engine="openpyxl") as xw_sum:
            pd.DataFrame([summary_selisih]).to_excel(xw_sum, index=False, sheet_name="Summary")

        st.session_state["HASIL"]["summary_selisih"] = {
            "periode": periode,
            "summary": summary_selisih,
            "excel_bytes": bio_sum.getvalue(),
        }
    else:
        st.session_state["HASIL"].pop("summary_selisih", None)

    st.success("Proses selesai. Hasil tersimpan (klik download tidak perlu proses ulang).")


# =========================
# RENDER HASIL TERSIMPAN
# =========================

hasil = st.session_state.get("HASIL", {})

if "rekon" in hasil:
    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.caption(f"Periode tersimpan: {hasil['rekon']['periode']}")
    st.dataframe(_style_right(hasil["rekon"]["table"]), use_container_width=True, hide_index=True)

    st.download_button(
        "Unduh Excel Rekonsiliasi",
        data=hasil["rekon"]["excel_bytes"],
        file_name=f"rekonsiliasi_{hasil['rekon']['periode']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_rekon",
    )

if "detail_tiket" in hasil:
    st.subheader("Detail Tiket per Tanggal — TYPE: GO SHOW & ONLINE × SUB-TIPE (J) [HANYA PAID]")
    st.caption(f"Periode tersimpan: {hasil['detail_tiket']['periode']}")
    st.dataframe(_style_right(hasil["detail_tiket"]["table"]), use_container_width=True, hide_index=True)

if "summary_selisih" in hasil:
    st.subheader("Summary Selisih Tiket Detail ESPAY vs Settlement ESPAY (berdasarkan Order ID)")
    st.caption(f"Periode tersimpan: {hasil['summary_selisih']['periode']}")

    s = hasil["summary_selisih"]["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL TIKET (ESPAY)", _idr_fmt(s["TOTAL TIKET (ESPAY)"]))
    c2.metric("TOTAL SETTLEMENT (ESPAY)", _idr_fmt(s["TOTAL SETTLEMENT (ESPAY)"]))
    c3.metric("SELISIH TOTAL (TIKET - SETTLEMENT)", _idr_fmt(s["SELISIH TOTAL (TIKET - SETTLEMENT)"]))
    c4.metric("ABS SELISIH TOTAL", _idr_fmt(s["ABS SELISIH TOTAL"]))

    st.markdown("### Ringkasan Status Order ID")
    st.write({
        "ONLY TICKET (COUNT)": s["ONLY TICKET (COUNT)"],
        "ONLY TICKET (NOMINAL)": _idr_fmt(s["ONLY TICKET (NOMINAL)"]),
        "ONLY SETTLEMENT (COUNT)": s["ONLY SETTLEMENT (COUNT)"],
        "ONLY SETTLEMENT (NOMINAL)": _idr_fmt(s["ONLY SETTLEMENT (NOMINAL)"]),
        "MISMATCH (COUNT)": s["MISMATCH (COUNT)"],
        "MISMATCH (ABS NOMINAL)": _idr_fmt(s["MISMATCH (ABS NOMINAL)"]),
    })

    st.download_button(
        "Unduh Excel Summary Selisih",
        data=hasil["summary_selisih"]["excel_bytes"],
        file_name=f"summary_selisih_{hasil['summary_selisih']['periode']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_summary_selisih",
    )

if not hasil:
    st.info("Silakan upload file, pilih bulan-tahun, lalu klik **Proses**.")
