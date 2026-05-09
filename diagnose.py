import pandas as pd

# Korelasi fitur estimasi tagihan vs target aktual
def diagnose_correlation(df):
    corr_cols = [
        "Estimasi_Tagihan_Energi_Bulanan_Rp",
        "Total_Energi_Semua_kWhPerBulan",
        "Rasio_Energi_Bulanan_Per_Daya_VA",
        "Daya_Listrik_Rumah_VA",
        "Jumlah_Anggota_Keluarga",
    ]

    target = "Rata_Rata_Tagihan_Per_Bulan_Rp"
    existing = [c for c in corr_cols if c in df.columns]

    # ✅ Fix: konversi ke numerik dulu, string → NaN, lalu drop baris NaN
    subset = df[existing + [target]].copy()
    subset = subset.apply(pd.to_numeric, errors="coerce")  # 'Tidak tahu' → NaN
    subset = subset.dropna()

    print(f"Baris valid untuk korelasi: {len(subset)} dari {len(df)}")

    corr = subset.corr()
    print("\nKorelasi terhadap target:")
    print(corr[target].sort_values(ascending=False))

    # Bonus: cek seberapa banyak nilai yang hilang per kolom
    print("\nJumlah nilai non-numerik per kolom:")
    for col in existing + [target]:
        n_invalid = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n_invalid > 0:
            print(f"  {col}: {n_invalid} nilai tidak valid")

    # Cek distribusi target
    print("\nNilai unik target:", df[target].nunique())
    print("\nDistribusi nilai target:")
    print(df[target].value_counts().sort_index().head(20))