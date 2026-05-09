import numpy as np

# this code is for feature engineering, which is the process of creating new features from existing data to improve the performance of models random forest regression
from utils.config import config

class FeatureEngineer:
    DAYS_PER_MONTH = 30

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

            self.df["MesinCuci_Energi_kWhPerHari"] = (
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
            "TV_Energi_kWhPerHari",
            "AC_Energi_kWhPerHari",
            "Kipas_Energi_kWhPerHari",
            "RiceCooker_Energi_kWhPerHari",
            "MesinCuci_Energi_kWhPerHari",
        ]

        main_exists = [c for c in main_cols if c in self.df.columns]

        other_cols = [
            f"Alat_Lain_{i}_Energi_kWhPerHari"
            for i in range(1, 4)
            if f"Alat_Lain_{i}_Energi_kWhPerHari" in self.df.columns
        ]

        if len(other_cols) > 0:
            self.df["Total_Energi_Alat_Lain_kWhPerHari"] = self.df[other_cols].sum(axis=1)
        else:
            self.df["Total_Energi_Alat_Lain_kWhPerHari"] = 0
            print("No 'Alat Lain' energy columns found. Total_Energi_Alat_Lain_kWhPerHari set to 0.")
        
        self.df["Total_Energi_Utama_kWhPerHari"] = self.df[main_exists].sum(axis=1) if len(main_exists) > 0 else 0

        self.df["Total_Energi_Semua_kWhPerHari"] = (
            self.df["Total_Energi_Utama_kWhPerHari"] +
            self.df["Total_Energi_Alat_Lain_kWhPerHari"]
        )

        return self

    def monthly_energy_features(self):
        energy_columns = [
            "Total_Energi_Utama_kWhPerHari",
            "Total_Energi_Alat_Lain_kWhPerHari",
            "Total_Energi_Semua_kWhPerHari",
        ]

        for daily_col in energy_columns:
            if daily_col in self.df.columns:
                monthly_col = daily_col.replace("kWhPerHari", "kWhPerBulan")
                self.df[monthly_col] = self.df[daily_col] * self.DAYS_PER_MONTH

        if "Total_Energi_Semua_kWhPerBulan" in self.df.columns:
            self.df["Estimasi_Energi_Bulanan_kWh"] = self.df["Total_Energi_Semua_kWhPerBulan"]

        return self

    def prepaid_features(self):
        required_cols = [
            "Nominal_Token_Terakhir_Rp",
            "Frekuensi_Isi_Token_Per_Bulan",
            "Total_Energi_Semua_kWhPerBulan",
        ]

        if all(col in self.df.columns for col in required_cols):
            safe_frequency = self.df["Frekuensi_Isi_Token_Per_Bulan"].replace(0, 1e-6)
            safe_monthly_energy = self.df["Total_Energi_Semua_kWhPerBulan"] + 1e-6

            self.df["Estimasi_Energi_Per_Transaksi_kWh"] = (
                self.df["Total_Energi_Semua_kWhPerBulan"] / safe_frequency
            )

            self.df["Estimasi_Pengeluaran_Token_Bulanan"] = (
                self.df["Nominal_Token_Terakhir_Rp"] *
                self.df["Frekuensi_Isi_Token_Per_Bulan"]
            )

            self.df["Rasio_Token_Terhadap_Energi"] = (
                self.df["Nominal_Token_Terakhir_Rp"] /
                (self.df["Estimasi_Energi_Per_Transaksi_kWh"] + 1e-6)
            )

            # tambahkan logging untuk mengetahui apakah terdapat baris yang mempunyai nilai 0 atau NaN pada kolom Daya_Listrik_Rumah_VA
            zero_or_nan_rows = self.df[(self.df["Daya_Listrik_Rumah_VA"].isnull())]
            if not zero_or_nan_rows.empty:
                print(f"Found {len(zero_or_nan_rows)} rows with 0 or NaN values in 'Daya_Listrik_Rumah_VA'.")

            self.df["Rasio_Token_Daya_VA"] = self.df["Nominal_Token_Terakhir_Rp"] / self.df["Daya_Listrik_Rumah_VA"]
            
            self.df["Rasio_Pengeluaran_Token_Terhadap_Energi_Bulanan"] = (
                self.df["Estimasi_Pengeluaran_Token_Bulanan"] /
                safe_monthly_energy
            )

            self.df["Estimasi_Durasi_Token_Dari_Frekuensi_Hari"] = (
                self.DAYS_PER_MONTH / safe_frequency
            )
        
        return self
    
    def estimate_bill_amount(self, kwh, daya_va):
        tarif_map = {
            (True, 450): 415,
            (True, 900): 605,
            (False, 900): 1352,
            (False, 1300): 1444.70,
            (False, 2200): 1444.70,
            (False, 3500): 1699.53,
            (False, 4400): 1699.53,
        }

        tarif = tarif_map.get(daya_va)
        if tarif is None:
            if daya_va <= 450:
                tarif = tarif_map[(True, 450)]
            elif daya_va <= 900:
                tarif = tarif_map[(True, 900)]
            elif daya_va <= 1300:
                tarif = tarif_map[(False, 1300)]
            elif daya_va <= 2200:
                tarif = tarif_map[(False, 2200)]
            elif daya_va <= 3500:
                tarif = tarif_map[(False, 3500)]
            else:
                tarif = tarif_map[(False, 4400)]
        
        base_bill = kwh * tarif 
        ppj_rate = 0.03
        ppj = base_bill * ppj_rate
        admin_fee = 3000

        return base_bill + ppj + admin_fee

    def postpaid_features(self):
        if "Bulan_Tagihan" in self.df.columns:
            month_number = self.df["Bulan_Tagihan"].map(config['month_mapping'])
            month_mode = month_number.mode(dropna=True)
            month_fallback = 1 if month_mode.empty else month_mode.iloc[0]
            self.df["Bulan_Tagihan"] = month_number.fillna(month_fallback)

        if "Bulan_Tagihan" in self.df.columns:
            self.df["Siklus_Tagihan_Hari"] = self.DAYS_PER_MONTH
            self.df["Bulan_Tagihan_Sin"] = np.sin(2 * np.pi * self.df["Bulan_Tagihan"] / 12)
            self.df["Bulan_Tagihan_Cos"] = np.cos(2 * np.pi * self.df["Bulan_Tagihan"] / 12)
            self.df["Rata_Rata_Energi_Harian_Dari_Bulanan_kWh"] = (
                self.df["Total_Energi_Semua_kWhPerBulan"] /
                self.df["Siklus_Tagihan_Hari"]
            )

        if "Daya_Listrik_Rumah_VA" in self.df.columns:
            safe_power = self.df["Daya_Listrik_Rumah_VA"].replace(0, 1e-6)
            self.df["Daya_Listrik_Rumah_kVA"] = self.df["Daya_Listrik_Rumah_VA"] / 1000

            if "Total_Energi_Semua_kWhPerBulan" in self.df.columns:
                self.df["Rasio_Energi_Bulanan_Per_Daya_VA"] = (
                    self.df["Total_Energi_Semua_kWhPerBulan"] / safe_power
                )

        if all(col in self.df.columns for col in [
            "Daya_Listrik_Rumah_VA",
            "Status_Subsidi_Listrik",
            "Total_Energi_Semua_kWhPerBulan",
        ]):
            subsidy_text = self.df["Status_Subsidi_Listrik"].astype(str).str.lower()
            is_subsidized = subsidy_text.eq("subsidi")
            self.df["Status_Subsidi_Flag"] = is_subsidized.astype(int)

            tariff = np.select(
                [
                    is_subsidized & self.df["Daya_Listrik_Rumah_VA"].le(450),
                    is_subsidized & self.df["Daya_Listrik_Rumah_VA"].le(900),
                    self.df["Daya_Listrik_Rumah_VA"].le(900),
                    self.df["Daya_Listrik_Rumah_VA"].le(1300),
                ],
                [
                    415,
                    605,
                    1352,
                    1445,
                ],
                default=1445,
            )

            self.df["Tarif_Estimasi_RpPerkWh"] = tariff
            self.df["Estimasi_Tagihan_Energi_Bulanan_Rp"] = (
                self.df["Total_Energi_Semua_kWhPerBulan"] *
                self.df["Tarif_Estimasi_RpPerkWh"]
            )
            self.df["Rasio_Estimasi_Tagihan_Per_Daya_VA"] = (
                self.df["Estimasi_Tagihan_Energi_Bulanan_Rp"] /
                self.df["Daya_Listrik_Rumah_VA"].replace(0, 1e-6)
            )

        return self
    
    # dropout redundant features that are not needed for modeling
    def drop_wh_columns(self):
        wh_cols = [col for col in self.df.columns if col.endswith("_Energi_WhPerHari")]
        self.df.drop(columns=wh_cols, inplace=True)
        return self

    # pipeline for feature engineering
    def engineer_features(self):
        (
            self.calculate_daily_energy()
            .calculate_washing_machine_energy()
            .calculate_other_devices()
            .total_energy()
            .monthly_energy_features()
        )

        if self.dataset_type == "prabayar":
            self.prepaid_features()
        elif self.dataset_type == "pascabayar":
            self.postpaid_features()

        self.drop_wh_columns()

        return self.df
    
