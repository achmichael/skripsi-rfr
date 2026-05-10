# Configuration for Random Forest Regression from scratch
# This file stores model parameters, preprocessing settings,
# dataset targets, selected features, and unused columns.

config = {
    "random_forest": {
        # Number of trees in the forest
        "n_estimators": 100,

        # Maximum depth of each decision tree
        # None means the tree grows until stopping criteria are met
        "max_depth": 16,

        # Minimum number of samples required to split an internal node
        "min_samples_split": 5,

        # Minimum number of samples required to be in a leaf node
        "min_samples_leaf": 2,

        # Number of random features considered at each split
        # Options: "sqrt", "third", "all", or integer value
        "max_features": "third",

        # Bootstrap sampling for each tree
        "bootstrap": True,

        # Random seed for reproducibility
        "random_state": 42
    },

    "data_preprocessing": {
        # Train-test split configuration
        "test_size": 0.2,
        "shuffle": True,
        "random_state": 42,

        # Missing value handling
        "numeric_missing_strategy": "median",
        "categorical_missing_strategy": "most_frequent",

        # Outlier handling
        "handle_outliers": True,
        "outlier_method": "iqr",

        # Scaling is optional for Random Forest
        # Random Forest does not require scaling, so False is preferred
        "scaling": False,
        "clip_target": False,
        
        # Encoding configuration
        "encoding": {
            "nominal": "one_hot",
            "ordinal": "label_encoding"
        }
    },

    "target_cleaning": {
        "pascabayar": {
            "enabled": True,
            "small_bill_threshold": 1000,
            "small_bill_multiplier": 1000
        }
    },

    "cross_validation": {
        "enabled": True,
        "n_splits": 5
    },

    "target_transform": {
        "prabayar": "none",
        "pascabayar": "none"
    },

    "paths": {
        "raw_data": {
            "prabayar": "data/raw/prabayar.csv",
            "pascabayar": "data/raw/pascabayar.csv"
        },

        "processed_data": {
            "prabayar": "data/processed/prabayar_processed.csv",
            "pascabayar": "data/processed/pascabayar_processed.csv"
        },

        "split_data": {
            "prabayar_train": "data/split/prabayar_train.csv",
            "prabayar_test": "data/split/prabayar_test.csv",
            "pascabayar_train": "data/split/pascabayar_train.csv",
            "pascabayar_test": "data/split/pascabayar_test.csv"
        },

        "models": {
            "prabayar": "outputs/models/random_forest_prabayar.json",
            "pascabayar": "outputs/models/random_forest_pascabayar.json"
        },

        "results": {
            "prabayar": "outputs/results/prabayar_predictions.csv",
            "pascabayar": "outputs/results/pascabayar_predictions.csv"
        },

        "reports": {
            "prabayar": "outputs/reports/evaluation_prabayar.txt",
            "pascabayar": "outputs/reports/evaluation_pascabayar.txt"
        }
    },
    
    "target": {
        "prabayar": "Token_Habis_Dalam_Hari",
        "pascabayar": "Tagihan_Rata_Rata_3Bulan_Rp"
    },

    "cols_to_drop": {
        # Duplikat alias
        "Estimasi_Energi_Bulanan_kWh",          # = Total_Energi_Semua_kWhPerBulan
        "Rata_Rata_Energi_Harian_Dari_Bulanan_kWh",  # = kWhPerBulan / 30

        # Versi harian sudah ada, bulan hanya × 30 (tidak tambah info)
        "Total_Energi_Alat_Lain_kWhPerBulan",
        "Total_Energi_Utama_kWhPerBulan",

        # Zero importance — RF tidak pernah memakainya
        "Alat_Lain_1_EstimasiJamPerHari",
        "Alat_Lain_1_Energi_kWhPerHari",
        "Alat_Lain_2_EstimasiJamPerHari",
        "Alat_Lain_2_Energi_kWhPerHari",
        "Alat_Lain_3_EstimasiJamPerHari",
        "Alat_Lain_3_Energi_kWhPerHari",
        "Total_Energi_Alat_Lain_kWhPerHari",

        # VA dan kVA adalah hal yang sama (× 1000)
        "Daya_Listrik_Rumah_kVA",

        # Duplikat subsidi
        "Status_Subsidi_Listrik_Non Subsidi",   # inverse dari kolom Subsidi
    },

    "features": {
        "prabayar": [
            "Kota/Kabupaten",
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Nominal_Token_Terakhir_Rp",
            "Frekuensi_Isi_Token_Per_Bulan",
            "Kulkas_Jumlah",
            "Kulkas_Kategori",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_Kategori",
            "TV_EstimasiWattPerUnit",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_Kategori",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_Kategori",
            "Kipas_EstimasiWattPerUnit",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_Kategori",
            "RiceCooker_EstimasiWattPerUnit",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_Kategori",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_1_EstimasiWatt",
            "Alat_Lain_1_EstimasiJamPerHari",
            "Alat_Lain_1_Energi_kWhPerHari",

            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_2_EstimasiWatt",
            "Alat_Lain_2_EstimasiJamPerHari",
            "Alat_Lain_2_Energi_kWhPerHari",

            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Kategori",
            "Alat_Lain_3_EstimasiWatt",
            "Alat_Lain_3_EstimasiJamPerHari",
            "Alat_Lain_3_Energi_kWhPerHari",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari"
        ],

        "pascabayar": [
            "Kota/Kabupaten",
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Bulan_Tagihan",
            "Tagihan_Relatif_Stabil",

            "Kulkas_Jumlah",
            "Kulkas_Kategori",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_Kategori",
            "TV_EstimasiWattPerUnit",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_Kategori",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_Kategori",
            "Kipas_EstimasiWattPerUnit",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_Kategori",
            "RiceCooker_EstimasiWattPerUnit",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_Kategori",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_Ada",

            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_1_EstimasiWatt",
            "Alat_Lain_1_EstimasiJamPerHari",
            "Alat_Lain_1_Energi_kWhPerHari",

            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_2_EstimasiWatt",
            "Alat_Lain_2_EstimasiJamPerHari",
            "Alat_Lain_2_Energi_kWhPerHari",

            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Kategori",
            "Alat_Lain_3_EstimasiWatt",
            "Alat_Lain_3_EstimasiJamPerHari",
            "Alat_Lain_3_Energi_kWhPerHari",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari"
        ]
    },

    "one_hot_columns": [
        "Kota/Kabupaten",
        "Status_Subsidi_Listrik",
        "Kulkas_Kategori",
        "TV_Kategori",
        "AC_Kategori",
        "Kipas_Kategori",
        "RiceCooker_Kategori",
        "MesinCuci_Kategori",
        "Alat_Lain_Ada",
        "Alat_Lain_1_Jenis",
        "Alat_Lain_1_Kategori",
        "Alat_Lain_2_Jenis",
        "Alat_Lain_2_Kategori",
        "Alat_Lain_3_Jenis",
        "Alat_Lain_3_Kategori"
    ],

    "unused_features": {
        "prabayar": [
            "Timestamp",
            "Nama/Inisial",
            "Kota/Kabupaten",
            "Jenis_Listrik",

            # Optional: redundant Wh columns because kWh columns are already used
            "Kulkas_Energi_WhPerHari",
            "TV_Energi_WhPerHari",
            "AC_Energi_WhPerHari",
            "Kipas_Energi_WhPerHari",
            "RiceCooker_Energi_WhPerHari",
            "MesinCuci_Energi_WhPerHari",
            "Alat_Lain_1_Energi_WhPerHari",
            "Alat_Lain_2_Energi_WhPerHari",
            "Alat_Lain_3_Energi_WhPerHari",
            "Total_Energi_Alat_Lain_WhPerHari",
            "Total_Energi_Utama_WhPerHari",
            "Total_Energi_Semua_WhPerHari"
        ],

        "pascabayar": [
            "Timestamp",
            "Nama/Inisial",
            "Kota/Kabupaten",
            "Jenis_Listrik",

            # Optional: redundant Wh columns because kWh columns are already used
            "Kulkas_Energi_WhPerHari",
            "TV_Energi_WhPerHari",
            "AC_Energi_WhPerHari",
            "Kipas_Energi_WhPerHari",
            "RiceCooker_Energi_WhPerHari",
            "MesinCuci_Energi_WhPerHari",
            "Alat_Lain_1_Energi_WhPerHari",
            "Alat_Lain_2_Energi_WhPerHari",
            "Alat_Lain_3_Energi_WhPerHari",
            "Total_Energi_Alat_Lain_WhPerHari",
            "Total_Energi_Utama_WhPerHari",
            "Total_Energi_Semua_WhPerHari"
        ]
    },

    "categorical_features": {
        "prabayar": [
            "Kota/Kabupaten",
            "Status_Subsidi_Listrik",
            "Kulkas_Kategori",
            "TV_Kategori",
            "AC_Kategori",
            "Kipas_Kategori",
            "RiceCooker_Kategori",
            "MesinCuci_Kategori",
            "Alat_Lain_Ada",
            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Kategori"
        ],

        "pascabayar": [
            "Kota/Kabupaten",
            "Status_Subsidi_Listrik",
            "Bulan_Tagihan",
            "Tagihan_Relatif_Stabil",
            "Kulkas_Kategori",
            "TV_Kategori",
            "AC_Kategori",
            "Kipas_Kategori",
            "RiceCooker_Kategori",
            "MesinCuci_Kategori",
            "Alat_Lain_Ada",
            "Alat_Lain_1_Jenis",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_2_Jenis",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_3_Jenis",
            "Alat_Lain_3_Kategori"
        ]
    },

    "numeric_features": {
        "prabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
            "Nominal_Token_Terakhir_Rp",
            "Frekuensi_Isi_Token_Per_Bulan",

            "Kulkas_Jumlah",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_EstimasiWattPerUnit",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_EstimasiWattPerUnit",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_EstimasiWattPerUnit",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_1_EstimasiWatt",
            "Alat_Lain_1_EstimasiJamPerHari",
            "Alat_Lain_1_Energi_kWhPerHari",

            "Alat_Lain_2_EstimasiWatt",
            "Alat_Lain_2_EstimasiJamPerHari",
            "Alat_Lain_2_Energi_kWhPerHari",

            "Alat_Lain_3_EstimasiWatt",
            "Alat_Lain_3_EstimasiJamPerHari",
            "Alat_Lain_3_Energi_kWhPerHari",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari"
        ],

        "pascabayar": [
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",

            "Kulkas_Jumlah",
            "Kulkas_EstimasiWattPerUnit",
            "Kulkas_EstimasiJamPerHari",
            "Kulkas_Energi_kWhPerHari",

            "TV_Jumlah",
            "TV_EstimasiWattPerUnit",
            "TV_EstimasiJamPerHari",
            "TV_Energi_kWhPerHari",

            "AC_Jumlah",
            "AC_EstimasiWattPerUnit",
            "AC_EstimasiJamPerHari",
            "AC_Energi_kWhPerHari",

            "Kipas_Jumlah",
            "Kipas_EstimasiWattPerUnit",
            "Kipas_EstimasiJamPerHari",
            "Kipas_Energi_kWhPerHari",

            "RiceCooker_Jumlah",
            "RiceCooker_EstimasiWattPerUnit",
            "RiceCooker_EstimasiJamPerHari",
            "RiceCooker_Energi_kWhPerHari",

            "MesinCuci_Jumlah",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam",
            "MesinCuci_Energi_kWhPerHari",

            "Alat_Lain_1_EstimasiWatt",
            "Alat_Lain_1_EstimasiJamPerHari",
            "Alat_Lain_1_Energi_kWhPerHari",

            "Alat_Lain_2_EstimasiWatt",
            "Alat_Lain_2_EstimasiJamPerHari",
            "Alat_Lain_2_Energi_kWhPerHari",

            "Alat_Lain_3_EstimasiWatt",
            "Alat_Lain_3_EstimasiJamPerHari",
            "Alat_Lain_3_Energi_kWhPerHari",

            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari"
        ]
    },

    "devices": [
        "Kulkas",
        "TV",
        "AC",
        "Kipas",
        "RiceCooker",
        "MesinCuci",
        "Alat_Lain_1",
        "Alat_Lain_2",
        "Alat_Lain_3"
    ],

    "month_mapping": {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    },

    "evaluation_metrics": [
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "MAPE",
        "WAPE",
        "NMAE",
        "NRMSE"
    ]
}

# Dataset pascabayar baru memakai target rata-rata tagihan 3 bulan.
# Kolom tagihan per bulan tidak dipakai sebagai fitur karena target ini
# dihitung langsung dari kolom-kolom tersebut.
PASCABAYAR_CATEGORICAL_FEATURES = [
    "Status_Subsidi_Listrik",
    "Bulan_Tagihan",
    "Sumber_Angka_Tagihan",
    "Tagihan_Relatif_Stabil",
    "Kulkas_Kategori",
    "TV_Kategori",
    "AC_PK_Kategori",
    "AC_Kategori",
    "Kipas_Kategori",
    "RiceCooker_Kategori",
    "MesinCuci_Kategori",
    "Alat_Lain_Ada",
    "Alat_Lain_1_Jenis",
    "Alat_Lain_1_Kategori",
    "Alat_Lain_2_Jenis",
    "Alat_Lain_2_Kategori",
    "Alat_Lain_3_Jenis",
    "Alat_Lain_3_Kategori",
]

PASCABAYAR_NUMERIC_FEATURES = [
    "Jumlah_Anggota_Keluarga",
    "Daya_Listrik_Rumah_VA",
    "Pemakaian_Bulan_Terakhir_kWh",
    "Pemakaian_2_Bulan_Lalu_kWh",
    "Pemakaian_3_Bulan_Lalu_kWh",
    "Pemakaian_Rata_Rata_3Bulan_kWh",
    "Jumlah_Bulan_Tagihan_Terisi",
    "Jumlah_Bulan_kWh_Terisi",
    "Kulkas_Jumlah",
    "Kulkas_EstimasiWattPerUnit",
    "Kulkas_EstimasiJamPerHari",
    "Kulkas_Energi_kWhPerHari",
    "TV_Jumlah",
    "TV_EstimasiWattPerUnit",
    "TV_EstimasiJamPerHari",
    "TV_Energi_kWhPerHari",
    "AC_Jumlah",
    "AC_EstimasiWattPerUnit",
    "AC_EstimasiJamPerHari",
    "AC_Energi_kWhPerHari",
    "Kipas_Jumlah",
    "Kipas_EstimasiWattPerUnit",
    "Kipas_EstimasiJamPerHari",
    "Kipas_Energi_kWhPerHari",
    "RiceCooker_Jumlah",
    "RiceCooker_EstimasiWattPerUnit",
    "RiceCooker_EstimasiJamPerHari",
    "RiceCooker_Energi_kWhPerHari",
    "MesinCuci_Jumlah",
    "MesinCuci_EstimasiWattPerUnit",
    "MesinCuci_EstimasiFrekuensiPerMinggu",
    "MesinCuci_EstimasiDurasiSekaliPakaiJam",
    "MesinCuci_Energi_kWhPerHari",
    "Alat_Lain_1_Jumlah",
    "Alat_Lain_1_EstimasiWatt",
    "Alat_Lain_1_EstimasiJamPerHari",
    "Alat_Lain_1_Energi_kWhPerHari",
    "Alat_Lain_2_Jumlah",
    "Alat_Lain_2_EstimasiWatt",
    "Alat_Lain_2_EstimasiJamPerHari",
    "Alat_Lain_2_Energi_kWhPerHari",
    "Alat_Lain_3_Jumlah",
    "Alat_Lain_3_EstimasiWatt",
    "Alat_Lain_3_EstimasiJamPerHari",
    "Alat_Lain_3_Energi_kWhPerHari",
    "Total_Energi_Alat_Lain_kWhPerHari",
    "Total_Energi_Utama_kWhPerHari",
    "Total_Energi_Semua_kWhPerHari",
    "Total_Energi_Semua_kWhPerBulan",
    "Estimasi_Tarif_Per_kWh_Rp",
    "Estimasi_Biaya_Energi_Bulanan_Rp",
    "Estimasi_Tagihan_Dengan_PPJ_Admin_Rp",
    "Bulan_Tagihan_Sin",
    "Bulan_Tagihan_Cos",
    "Status_Subsidi_Flag",
    "Tarif_Estimasi_RpPerkWh",
    "Rasio_Energi_Bulanan_Per_Daya_VA",
    "Rasio_Estimasi_Tagihan_Per_Daya_VA",
]

config["features"]["pascabayar"] = (
    PASCABAYAR_CATEGORICAL_FEATURES + PASCABAYAR_NUMERIC_FEATURES
)
config["categorical_features"]["pascabayar"] = PASCABAYAR_CATEGORICAL_FEATURES
config["numeric_features"]["pascabayar"] = PASCABAYAR_NUMERIC_FEATURES
config["unused_features"]["pascabayar"] = [
    "Timestamp",
    "Nama/Inisial",
    "Kota/Kabupaten",
    "Jenis_Listrik",
    "Tagihan_Bulan_Terakhir_Rp",
    "Tagihan_2_Bulan_Lalu_Rp",
    "Tagihan_3_Bulan_Lalu_Rp",
    "Kulkas_Energi_WhPerHari",
    "TV_Energi_WhPerHari",
    "AC_Energi_WhPerHari",
    "Kipas_Energi_WhPerHari",
    "RiceCooker_Energi_WhPerHari",
    "MesinCuci_Energi_WhPerHari",
    "Alat_Lain_1_Energi_WhPerHari",
    "Alat_Lain_2_Energi_WhPerHari",
    "Alat_Lain_3_Energi_WhPerHari",
    "Total_Energi_Alat_Lain_WhPerHari",
    "Total_Energi_Utama_WhPerHari",
    "Total_Energi_Semua_WhPerHari",
]

config["feature_selection"] = {
    "prabayar": False,
    "pascabayar": True,
}

config["diagnostics"] = {
    "enabled": False,
}

if "AC_PK_Kategori" not in config["categorical_features"]["prabayar"]:
    config["categorical_features"]["prabayar"].append("AC_PK_Kategori")
if "AC_PK_Kategori" not in config["one_hot_columns"]:
    config["one_hot_columns"].append("AC_PK_Kategori")
