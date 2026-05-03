# this code is for feature engineering, which is the process of creating new features from existing data to improve the performance of models random forest regression
from utils.config import config

class FeatureEngineer:
    def __init__(self, df, dataset_type="prabayar"):
        self.df = df.copy()
        self.dataset_type = dataset_type.lower()
        
    def calculate_daily_energy(self):
        """
        Rumus:
        Energy (wh) = jumlah x watt x jam
        Energy (kwh) = Energy (wh) / 1000
        """

        if len(config['devices']) != 0:
            for device in config['devices']:
                jumlah_col = f"{device}_Jumlah"
                watt_col = f"{device}_EstimasiWattPerUnit"
                jam_col = f"{device}_EstimasiJamPerHari"
                
                if all(col in self.df.columns for col in [jumlah_col, watt_col, jam_col]):
                    wh_col = f"{device}_Energi_WhPerHari"
                    kwh_col = f"{device}_Energi_kWhPerHari"

                    self.df[wh_col] = (
                        self.df[jumlah_col] *
                        self.df[watt_col] *
                        self.df[jam_col]
                    )

                    self.df[kwh_col] = self.df[wh_col] / 1000
        
        return self

    def calculate_washing_machine_energy(self):
        """
        Rumus:
            (jumlah × watt × frekuensi_per_minggu × durasi) / 7
        """
        
        required_cols = [
            "MesinCuci_Jumlah",
            "MesinCuci_EstimasiWattPerUnit",
            "MesinCuci_EstimasiFrekuensiPerMinggu",
            "MesinCuci_EstimasiDurasiSekaliPakaiJam"
        ]

        if all(col in self.df.columns for col in required_cols):
            self.df["MesinCuci_Energi_WhPerHari"] = (
                self.df["MesinCuci_Jumlah"] *
                self.df["MesinCuci_EstimasiWattPerUnit"] *
                self.df["MesinCuci_EstimasiFrekuensiPerMinggu"] *
                self.df["MesinCuci_EstimasiDurasiSekaliPakaiJam"]
            ) / 7

            self.df["MesinCuci_Energi_KwhPerHari"] = (
                self.df["MesinCuci_Energi_WhPerHari"] / 1000
            )

        return self

    def calculate_other_devices(self):
        for i in range (1, 4):
            jumlah = f"Alat_Lain_{i}_Jumlah"
            watt = f"Alat_Lain_{i}_EstimasiWattPerUnit"
            jam = f"Alat_Lain_{i}_EstimasiJamPerHari"

            if all(col in self.df.columns for col in [jumlah, watt, jam]):

                wh = f"Alat_Lain_{i}_Energi_WhPerHari"
                kwh = f"Alat_Lain_{i}_Energi_kWhPerHari"

                self.df[wh] = (
                    self.df[jumlah] *
                    self.df[watt] *
                    self.df[jam]
                )

                self.df[kwh] = self.df[wh] / 1000

        return self

    def total_energy(self):
        main_cols = [
            "Kulkas_Energi_kWhPerHari",
            "MesinCuci_Energi_kWhPerHari",
            "TV_Energi_kWhPerHari",
            "AC_Energi_kWhPerHari",
            "Kulkas_Energi_kWhPerHari",
            "MesinCuci_Energi_kwhPerHari",
            "RiceCooker_Energi_kWhPerHari",
        ]

        main_exists = [c for c in main_cols if c in self.df.columns]

        other_cols = [
            f"Alat_Lain_{i}_Energi_kWhPerHari"
            for i in range(1, 4)
            if f"Alat_Lain_{i}_Energi_kWhPerHari" in self.df.columns
        ]

        if len(other_cols) > 0:
            self.df["Total_Energi_Alat_Lain_kWhPerHari"] = self.df[main_exists + other_cols].sum(axis=1)
        else:
            self.df["Total_Energi_Alat_Lain_kWhPerHari"] = 0
            print("No 'Alat Lain' energy columns found. Total_Energi_Alat_Lain_kWhPerHari set to 0.")
        
        self.df["Total_Energi_Semua_kWhPerHari"] = (
            self.df["Total_Energi_Utama_kWhPerHari"] +
            self.df["Total_Energi_Alat_Lain_kWhPerHari"]
        )

        return self


    def prepaid_features(self):
        if "Nominal_Token_Terakhir_Rp" in self.df.columns:
            self.df["Rasio_Token_Terhadap_Energi"] = (
                self.df["Nominal_Token_Terakhir_Rp"] /
                (self.df["Total_Energi_Semua_kWhPerHari"] + 1e-6)
            )

            self.df["Estimasi_Energi_Bulanan_kWh"] = (
                self.df["Total_Energi_Semua_kWhPerHari"] * 30
            )

            if "Frekuensi_Isi_Token_Per_Bulan" in self.df.columns:
                self.df["Estimasi_Pengeluaran_Token_Bulanan"] = (
                    self.df["Nominal_Token_Terakhir_Rp"] *
                    self.df["Frekuensi_Isi_Token_Per_Bulan"]
                )

        # additional features to capture interaction between energy consumption and token usage
            self.df["Rasio_Frekuensi_Token_Energi"] = (
                self.df["Nominal_Token_Terakhir_Rp"] /
                (self.df["Total_Energi_Semua_kWhPerHari"] + 1e-6)
            )
        
        return self
    
    def postpaid_features(self):
        self.df["Estimasi_Energi_Bulanan_kWh"] = self.df["Total_Energi_Semua_kWhPerHari"] * 30

        if "Bulan_Tagihan" in self.df.columns:
            self.df["Bulan_Tagihan"] = self.df["Bulan_Tagihan"].map(config['month_mapping'])

        # additional features to capture interaction between energy consumption and billing cycle
        if "Bulan_Tagihan" in self.df.columns:
            self.df["Rasio_Bulan_Tagihan_Energi"] = (
                self.df["Bulan_Tagihan"] /
                (self.df["Total_Energi_Semua_kWhPerHari"] + 1e-6)
            )

        return self
    
    # dropout redundant features that are not needed for modeling
    def drop_wh_columns(self):
        wh_cols = [col for col in self.df.columns if col.endswith("_Energi_WhPerHari")]
        self.df.drop(columns=wh_cols, inplace=True)
        print(f"Dropped redundant Wh columns: {wh_cols}")
        return self

    # pipeline for feature engineering
    def engineer_features(self):
        (
            self.calculate_daily_energy()
            .calculate_washing_machine_energy()
            .calculate_washing_machine_energy()
            .total_energy()
        )

        if self.dataset_type == "prabayar":
            self.prepaid_features()
        elif self.dataset_type == "pascabayar":
            self.postpaid_features()

        self.drop_wh_columns()

        return self.df
    