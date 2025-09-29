# app.py
# -*- coding: utf-8 -*-
"""
Rekonsiliasi: Tiket Detail vs Settlement Dana (robust CSV + money/date parsing)
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
    - Terima: 1.234.567 | 1,234,567.89 | 50.300,00 | (1.000) | 1.000- | IDR 1,000 CR | -2.500
    - Jika ada '.' dan ',' sekaligus → gunakan SEPARATOR TERAKHIR sebagai desimal.
      Contoh: '50.300,00' → ',' desimal → hasil 50300.0
              '1,234.56'  → '.' desimal → hasil 1234.56
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, (int, float, np.number)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    neg = False
    # parentheses atau trailing minus
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.endswith("-"):
        neg, s = True, s[:-1].strip()

    # buang label umum (mata uang / CR/DR)
    s = re.sub(r"(idr|rp|cr|dr)", "", s, flags=re.IGNORECASE)
    # sisakan digit, titik, koma, minus
    s = re.sub(r"[^0-9\.,\-]", "", s).strip()

    # minus di depan
    if s.startswith("-"):
        neg, s = True, s[1:].strip()

    # Hitung separator
    last_dot = s.rfind(".")
    last_com = s.rfind(",")

    if last_dot == -1 and last_com == -1:
        # hanya digit
        num_s = s
    elif last_dot > last_com:
        # '.' sebagai desimal → hapus semua koma ribuan
        num_s = s.replace(",", "")
    else:
        # ',' sebagai desimal → hapus semua titik ribuan, lalu ganti ',' → '.'
        num_s = s.replace(".", "").replace(",", ".")

    try:
        num = float(num_s)
    except Exception:
        # fallback: buang semua pemisah
        num_s = s.replace(".", "").replace(",", "")
        num = float(num_s) if num_s else 0.0

    return -num if neg else num


def _to_num(sr: pd.Series) -> pd.Series:
    return sr.apply(_parse_money).astype(float)


def _to_date(val) -> Optional[pd.Timestamp]:
    """
    Parse berbagai format tanggal + dukung serial Excel (days since 1899-12-30).
    """
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
    """
    CSV: auto-detect delimiter, baca sebagai teks murni (hindari konversi otomatis Excel-locale).
    Excel: openpyxl.
    """
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
                        sep=None,           # auto-detect delimiter
                        engine="python",
                        dtype=str,          # simpan mentah
                        na_filter=False,    # jangan ubah "" jadi NaN
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


def _find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
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
