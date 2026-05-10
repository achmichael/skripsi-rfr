import pandas as pd


def diagnose_correlation(df, target="Tagihan_Rata_Rata_3Bulan_Rp"):
    corr_cols = [
        "Estimasi_Tagihan_Energi_Bulanan_Rp",
        "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp",
        "Estimasi_Biaya_Energi_Bulanan_Rp",
        "Total_Energi_Semua_kWhPerBulan",
        "Rasio_Energi_Bulanan_Per_Daya_VA",
        "Daya_Listrik_Rumah_VA",
        "Jumlah_Anggota_Keluarga",
    ]

    existing = [col for col in corr_cols if col in df.columns]
    if target not in df.columns or not existing:
        print(f"Diagnostik korelasi dilewati. Target/fitur tidak tersedia: {target}")
        return

    subset = df[existing + [target]].copy()
    subset = subset.apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna()

    print(f"Baris valid untuk korelasi: {len(subset)} dari {len(df)}")
    if subset.empty:
        print("Diagnostik korelasi dilewati karena tidak ada baris numerik valid.")
        return

    corr = subset.corr()
    print("\nKorelasi terhadap target:")
    print(corr[target].sort_values(ascending=False))

    print("\nJumlah nilai non-numerik per kolom:")
    for col in existing + [target]:
        n_invalid = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n_invalid > 0:
            print(f"  {col}: {n_invalid} nilai tidak valid")

    print("\nNilai unik target:", df[target].nunique())
    print("\nDistribusi nilai target:")
    print(df[target].value_counts().sort_index().head(20))
