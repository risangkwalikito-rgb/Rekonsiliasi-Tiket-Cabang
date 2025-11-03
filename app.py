# ... (semua kode Anda sebelumnya tetap sama persis sampai bagian pembuatan 'final')

    # --- Final table ---
    final = pd.DataFrame(index=idx)
    final["Tiket Detail ESPAY"] = tiket_series.values
    final["Settlement Dana"]    = settle_series.values
    final["Selisih"]            = final["Tiket Detail ESPAY"] - final["Settlement Dana"]
    final["Settlement BCA"]     = bca_series.values
    final["Settlement Non BCA"] = non_bca_series.values
    final["Total Settlement"]   = (final["Settlement BCA"] + final["Settlement Non BCA"]).values
    final["Uang Masuk BCA"]     = uang_masuk_bca_ser.values
    final["Uang Masuk Non BCA"] = uang_masuk_non_ser.values
    final["Total Uang Masuk"]   = (final["Uang Masuk BCA"] + final["Uang Masuk Non BCA"]).values

    # ---- RENAME -> UPPERCASE sesuai permintaan ----
    rename_map = {
        "Tiket Detail ESPAY": "TIKET DETAIL ESPAY",
        "Settlement Dana": "SETTLEMENT DANA ESPAY",
        "Selisih": "SELISIH TIKET DETAIL - SETTLEMENT",
        "Settlement BCA": "SETTLEMENT BCA",
        "Settlement Non BCA": "SETTLEMENT NON BCA",
        "Total Settlement": "TOTAL SETTLEMENT",
        "Uang Masuk BCA": "UANG MASUK BCA",
        "Uang Masuk Non BCA": "UANG MASUK NON BCA",
        "Total Uang Masuk": "TOTAL UANG MASUK",
    }
    final = final.rename(columns=rename_map)

    # View + total
    view = final.reset_index()
    view.insert(0, "No", range(1, len(view) + 1))
    view = view.rename(columns={"index": "Tanggal"})
    view = view.rename(columns={"No": "NO", "Tanggal": "TANGGAL"})

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
    }])
    view_total = pd.concat([view, total_row], ignore_index=True)

    # Format tampilan uang (untuk table di layar)
    fmt = view_total.copy()
    for c in [
        "TIKET DETAIL ESPAY","SETTLEMENT DANA ESPAY","SELISIH TIKET DETAIL - SETTLEMENT",
        "SETTLEMENT BCA","SETTLEMENT NON BCA","TOTAL SETTLEMENT",
        "UANG MASUK BCA","UANG MASUK NON BCA","TOTAL UANG MASUK"
    ]:
        fmt[c] = fmt[c].apply(_idr_fmt)

    st.subheader("Hasil Rekonsiliasi per Tanggal (mengikuti bulan parameter)")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # ---------- Export ke Excel + MERGE HEADER ----------
    from openpyxl.styles import Alignment, Font

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        # Sheet raw (angka)
        view_total.to_excel(xw, index=False, sheet_name="Rekonsiliasi")

        # Sheet tampilan (string terformat)
        fmt.to_excel(xw, index=False, sheet_name="Rekonsiliasi_View")
        wb = xw.book
        ws = wb["Rekonsiliasi_View"]

        # Sisipkan baris header utama di atas baris header yang sudah ada
        ws.insert_rows(1)

        # Header level-2 (sub) yang kita inginkan di baris ke-2
        sub_headers = [
            "NO", "TANGGAL",
            "TIKET DETAIL ESPAY", "SETTLEMENT DANA ESPAY", "SELISIH TIKET DETAIL - SETTLEMENT",
            "BCA", "NON BCA", "TOTAL SETTLEMENT",
            "BCA", "NON BCA", "TOTAL UANG MASUK",
        ]
        # Header level-1 (utama) di baris ke-1 (yang akan di-merge sebagian)
        top_headers = [
            "NO", "TANGGAL",
            "TIKET DETAIL ESPAY", "SETTLEMENT DANA ESPAY", "SELISIH TIKET DETAIL - SETTLEMENT",
            "SETTLEMENT", "SETTLEMENT", "TOTAL SETTLEMENT",
            "UANG MASUK", "UANG MASUK", "TOTAL UANG MASUK",
        ]

        # Tulis header utama & sub
        for col_idx, (top, sub) in enumerate(zip(top_headers, sub_headers), start=1):
            ws.cell(row=1, column=col_idx, value=top)
            ws.cell(row=2, column=col_idx, value=sub)

        # Merge header: SETTLEMENT (F1:G1) dan UANG MASUK (I1:J1)
        # (posisi sesuai urutan kolom di atas)
        ws.merge_cells(start_row=1, start_column=6, end_row=1, end_column=7)   # SETTLEMENT
        ws.merge_cells(start_row=1, start_column=9, end_row=1, end_column=10) # UANG MASUK

        # Styling header
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
