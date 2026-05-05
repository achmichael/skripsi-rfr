# Configuration for Random Forest Regression from scratch
# This file stores model parameters, preprocessing settings,
# dataset targets, selected features, and unused columns.

config = {
    "random_forest": {
        # Number of trees in the forest
        "n_estimators": 300,

        # Maximum depth of each decision tree
        # None means the tree grows until stopping criteria are met
        "max_depth": 12,

        # Minimum number of samples required to split an internal node
        "min_samples_split": 5,

        # Minimum number of samples required to be in a leaf node
        "min_samples_leaf": 3,

        # Number of random features considered at each split
        # Options: "sqrt", "third", "all", or integer value
        "max_features": "sqrt",

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
        "pascabayar": "Rata_Rata_Tagihan_Per_Bulan_Rp"
    },

    "features": {
        "prabayar": [
            "Kota/Kabupaten",
            "Jumlah_Anggota_Keluarga",
            "Daya_Listrik_Rumah_VA",
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
        "R2"
    ]
}