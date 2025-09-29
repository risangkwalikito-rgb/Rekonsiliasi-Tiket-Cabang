# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana (versi robust parser)
"""

from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

# ---------------- Utilities ----------------

def _parse_money(val) -> float:
    """
    Parser uang yang robust:
    - Terima: 1.234.567 | 1,234,567.89 | (1.000) | 1.000- | IDR 1,000 CR | -2.500
    - Mengabaikan teks seperti 'IDR', 'CR', 'DR', spasi, 'Rp', dll.
    - Menentukan pemisah desimal saat hanya ada salah satu dari '.' atau ','.
    - Jika ambiguitas (dua-duanya muncul), diasumsikan keduanya pemisah ribuan -> integer.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.number)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    # Negatif: parentheses atau trailing minus
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.endswith("-"):
        neg, s = True, s[:-1].strip()

    # Buang label umum
    s = re.sub(r"(idr|rp|cr|dr)", "", s, flags=re.IGNORECASE)

    # Sisakan digit, titik, koma, dan leading minus (kalau ada)
    s = re.sub(r"[^0-9\.,\-]", "", s).strip()

    # Jika masih ada minus di awal, tandai negatif
    if s.startswith("-"):
        neg, s = True, s[1:].strip()

    # Deteksi pola desimal
    dot = s.count(".")
    comma = s.count(",")

    if (dot == 1 and comma == 0) or (dot == 0 and comma == 1):
        # Ada satu kandidat desimal
        if comma == 1:
            # contoh: 123.456,78  atau 1,23 -> asumsikan ',' desimal
            s = s.replace(".", "")  # hapus ribuan
            s = s.replace(",", ".")  # ganti desimal ke '.'
        else:
            # '.' desimal, hapus koma ribuan
            s = s.replace(",", "")
        try:
            num = float(s)
        except Exception:
            num = 0.0
    else:
        # Ambigu atau tanpa desimal -> treat as integer ribuan
        s = s.replace(".", "").replace(",", "")
        try:
            num = float(s) if s else 0.0
        except Exception:
            num = 0.0

    return -num if neg else num


def _to_num(sr: pd.Series) -> pd.Series:
    return sr.apply(_parse_money).astype(float)


def _to_date(val) -> Optional[pd.Timestamp]:
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


def _read_any(uploaded_file) -> pd.DataFrame:
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
            st.error("CSV gagal dibaca. Simpan ulang sebagai UTF-8.")
            return pd.DataFrame()
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    if df.empty:
        return None
    cols = [c for c in df.columns if isinstance(c, str)]
    m = {c.lower().strip(): c for c in cols}
    # exact
    for n in names:
        if n.lower().strip() in m:
            return m[n.lower().strip()]
    # contains
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


# ---------------- App ----------------

st.set_page_config(page_title="Rekonsiliasi Tiket vs Settlement", layout="wide")
st.title("Rekonsiliasi: Tiket Detail vs Settlement Dana")

with st.sidebar:
    st.header("1) Upload Sumber")
    tiket_file = st.file_uploader("Tiket Detail (Excel .xls/.xlsx)", type=["xls", "xlsx"])
    settle_file = st.file_uploader("Settlement Dana (CSV/Excel)", type=["csv", "xls", "xlsx"])

    st.header("2) Opsi")
    show_preview = st.checkbox("Tampilkan pratinjau data", value=False)
    show_debug = st.checkbox("Debug parsing", value=False)
    go = st.button("Proses", type="primary", use_container_width=True)

tiket_df = _read_any(tiket_file)
settle_df = _read_any(settle_file)

if show_preview:
    st.subheader("Pratinjau")
    if not tiket_df.empty:
        st.markdown("Tiket Detail"); st.dataframe(tiket_df.head(50), use_container_width=True)
    if not settle_df.empty:
        st.markdown("Settlement Dana"); st.dataframe(settle_df.head(50), use_container_width=True)

if go:
    # Tiket Detail (fixed columns)
    t_date = _find_col(tiket_df, ["Action date"])
    t_amt  = _find_col(tiket_df, ["tarif"])
    t_stat = _find_col(tiket_df, ["St Bayar", "Status Bayar", "status"])
    t_bank = _find_col(tiket_df, ["Bank", "Payment Channel", "channel"])

    s_date = _find_col(settle_df, ["Transaction Date"])
    s_amt  = _find_col(settle_df, ["Settlement Amount"])

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
    if missing:
        st.error("Kolom wajib tidak ditemukan → " + "; ".join(missing))
        st.stop()

    # Tiket filter
    td = tiket_df.copy()
    td[t_date] = td[t_date].apply(_to_date)
    td = td[~td[t_date].isna()]
    td_stat = td[t_stat].astype(str).str.strip().str.lower()
    td_bank = td[t_bank].astype(str).str.strip().str.lower()
    td = td[td_stat.eq("paid") & td_bank.eq("espay")]
    td[t_amt] = _to_num(td[t_amt])

    tiket_by_date = td.groupby(td[t_date])[[t_amt]].sum().squeeze()
    tiket_by_date.index = pd.to_datetime(tiket_by_date.index).date
    tiket_by_date.name = "Tiket Detail"

    # Settlement
    sd = settle_df.copy()
    sd[s_date] = sd[s_date].apply(_to_date)
    sd = sd[~sd[s_date].isna()]
    orig_examples = sd[s_amt].astype(str).head(5).tolist()
    sd[s_amt] = _to_num(sd[s_amt])
    parsed_examples = sd[s_amt].head(5).tolist()

    settle_by_date = sd.groupby(sd[s_date])[[s_amt]].sum().squeeze()
    settle_by_date.index = pd.to_datetime(settle_by_date.index).date
    settle_by_date.name = "Settlement Dana"

    # Debug info
    if show_debug:
        st.info(
            f"Rows valid Tiket Detail: {len(td)} | Settlement: {len(sd)}\n\n"
            f"Contoh Settlement Amount (asli → parsed):\n"
            + "\n".join(f"- {a} → {p}" for a, p in zip(orig_examples, parsed_examples))
        )

    # Merge + diff
    all_dates = sorted(set(tiket_by_date.index) | set(settle_by_date.index))
    final = pd.DataFrame(index=pd.Index(all_dates, name="Tanggal"))
    final["Tiket Detail"] = tiket_by_date.reindex(all_dates).fillna(0.0)
    final["Settlement Dana"] = settle_by_date.reindex(all_dates).fillna(0.0)
    final["Selisih"] = final["Tiket Detail"] - final["Settlement Dana"]

    # View
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    fmt = view.copy()
    for c in ["Tiket Detail", "Settlement Dana", "Selisih"]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # Download
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        view.to_excel(xw, index=False, sheet_name="Rekonsiliasi")
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
    st.download_button(
        "Unduh Excel",
        data=bio.getvalue(),
        file_name="rekonsiliasi_tiket_vs_settlement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
