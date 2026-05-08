import json
import mimetypes
import os
import pickle
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
WEB_DIR = ROOT_DIR / "web"
RESULTS_DIR = ROOT_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.preprocessor import Preprocessor
from utils.config import config
from utils.core import inverse_transform_target


MAIN_DEVICES = {
    "Kulkas": {
        "label": "Kulkas",
        "default_category": "Tidak ada",
        "usage_categories": ["Tidak ada", "Kecil / 1 pintu", "Sedang / 2 pintu"],
        "default_watt": 120,
        "default_hours": 24,
    },
    "TV": {
        "label": "TV",
        "default_category": "Tidak ada / tidak digunakan",
        "usage_categories": [
            "Tidak ada / tidak digunakan",
            "Jarang, kurang dari 2 jam per hari",
            "Sedang, sekitar 2-5 jam per hari",
            "Sering, sekitar 6-10 jam per hari",
            "Sangat sering, lebih dari 10 jam per hari",
        ],
        "default_watt": 80,
        "default_hours": 3.5,
    },
    "AC": {
        "label": "AC",
        "default_category": "Tidak ada / tidak digunakan",
        "usage_categories": [
            "Tidak ada / tidak digunakan",
            "Jarang, kurang dari 2 jam per hari",
            "Sedang, sekitar 2-5 jam per hari",
            "Sering, sekitar 6-10 jam per hari",
            "Sangat sering, lebih dari 10 jam per hari",
        ],
        "default_watt": 600,
        "default_hours": 3.5,
    },
    "Kipas": {
        "label": "Kipas",
        "default_category": "Tidak ada / tidak digunakan",
        "usage_categories": [
            "Tidak ada / tidak digunakan",
            "Jarang, kurang dari 2 jam per hari",
            "Sedang, sekitar 2-5 jam per hari",
            "Sering, sekitar 6-10 jam per hari",
            "Sangat sering, lebih dari 10 jam per hari",
        ],
        "default_watt": 45,
        "default_hours": 3.5,
    },
    "RiceCooker": {
        "label": "Rice Cooker",
        "default_category": "Tidak ada / tidak digunakan",
        "usage_categories": [
            "Tidak ada / tidak digunakan",
            "Jarang, kurang dari 2 jam per hari",
            "Sedang, sekitar 2-5 jam per hari",
            "Sering, sekitar 6-10 jam per hari",
            "Sangat sering, lebih dari 10 jam per hari",
        ],
        "default_watt": 300,
        "default_hours": 3.5,
    },
    "MesinCuci": {
        "label": "Mesin Cuci",
        "default_category": "Tidak ada / tidak digunakan",
        "usage_categories": [
            "Tidak ada / tidak digunakan",
            "Jarang, 1-2 kali per minggu",
            "Sedang, 3-4 kali per minggu",
            "Sering, 5-6 kali per minggu",
            "Sangat sering, hampir setiap hari",
        ],
        "default_watt": 350,
        "default_frequency": 3.5,
        "default_duration": 1.5,
    },
}

OTHER_DEVICE_TYPES = [
    "Tidak diisi",
    "Charger HP/perangkat kecil",
    "Komputer/Laptop",
    "Dispenser",
    "Setrika",
    "Pompa air",
    "Blender/Mixer",
    "Oven/Microwave",
    "Lainnya",
]


def to_number(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class PredictionService:
    def __init__(self):
        self.datasets = {}
        for dataset_type in ["prabayar", "pascabayar"]:
            self.datasets[dataset_type] = self._load_dataset_context(dataset_type)

    def _load_dataset_context(self, dataset_type):
        raw_path = ROOT_DIR / config["paths"]["raw_data"][dataset_type]
        raw_df = pd.read_csv(raw_path)

        preprocessor = Preprocessor(dataset_type)
        cleaned_df = preprocessor.clean_raw_dataset(raw_df, dataset_type)
        preprocessor.fit(cleaned_df)

        return {
            "raw_df": cleaned_df,
            "preprocessor": preprocessor,
            "model": self._load_latest_model(dataset_type),
            "defaults": self._build_defaults(cleaned_df, dataset_type),
        }

    def _load_latest_model(self, dataset_type):
        model_paths = sorted(
            RESULTS_DIR.glob(f"{dataset_type}_*.pkl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not model_paths:
            raise FileNotFoundError(f"Tidak ada model {dataset_type} di folder results.")

        with model_paths[0].open("rb") as file:
            model = CompatibleUnpickler(file).load()

        model.model_path_ = str(model_paths[0])
        return model

    def _build_defaults(self, df, dataset_type):
        defaults = {}
        numeric_columns = set(config["numeric_features"].get(dataset_type, []))
        numeric_columns.add(config["target"][dataset_type])

        for col in df.columns:
            if col in numeric_columns:
                series = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                    .replace("", pd.NA)
                )
                numeric = pd.to_numeric(series, errors="coerce")
                value = numeric.median()
                defaults[col] = 0 if pd.isna(value) else float(value)
            else:
                mode = df[col].mode(dropna=True)
                defaults[col] = "" if mode.empty else mode.iloc[0]

        return defaults

    def metadata(self):
        payload = {
            "datasets": {},
            "main_devices": MAIN_DEVICES,
            "other_device_types": OTHER_DEVICE_TYPES,
            "months": list(config["month_mapping"].keys()),
            "power_options": [450, 900, 1300, 2200],
        }

        for dataset_type, context in self.datasets.items():
            model = context["model"]
            payload["datasets"][dataset_type] = {
                "target": config["target"][dataset_type],
                "model_path": getattr(model, "model_path_", ""),
                "feature_count": len(getattr(model, "feature_names_", [])),
            }

        return payload

    def predict(self, payload):
        dataset_type = payload.get("dataset_type")
        if dataset_type not in self.datasets:
            raise ValueError("dataset_type harus prabayar atau pascabayar.")

        context = self.datasets[dataset_type]
        record = dict(context["defaults"])
        record.update(self._build_record(payload, dataset_type))
        self._calculate_energy_columns(record)

        input_df = pd.DataFrame([record])
        processed = context["preprocessor"].transform(input_df)
        target_column = config["target"][dataset_type]
        features = processed.drop(columns=[target_column], errors="ignore")

        model = context["model"]
        model_features = getattr(model, "feature_names_", list(features.columns))
        features = features.reindex(columns=model_features, fill_value=0)

        prediction = float(model.predict(features)[0])
        prediction = float(inverse_transform_target(np.array([prediction]), dataset_type)[0])

        return {
            "dataset_type": dataset_type,
            "prediction": max(0.0, prediction),
            "formatted_prediction": self._format_prediction(dataset_type, prediction),
            "unit": "hari" if dataset_type == "prabayar" else "rupiah",
            "model_path": getattr(model, "model_path_", ""),
            "feature_count": len(model_features),
            "energy": {
                "main_kwh_per_day": float(record.get("Total_Energi_Utama_kWhPerHari", 0)),
                "other_kwh_per_day": float(record.get("Total_Energi_Alat_Lain_kWhPerHari", 0)),
                "total_kwh_per_day": float(record.get("Total_Energi_Semua_kWhPerHari", 0)),
                "total_kwh_per_month": float(record.get("Total_Energi_Semua_kWhPerHari", 0)) * 30,
            },
        }

    def _build_record(self, payload, dataset_type):
        general = payload.get("general", {})
        record = {
            "Jenis_Listrik": "Prabayar (Token/Pulsa)" if dataset_type == "prabayar" else "Pascabayar (Bulanan)",
            "Jumlah_Anggota_Keluarga": to_number(general.get("jumlah_anggota"), 1),
            "Daya_Listrik_Rumah_VA": to_number(general.get("daya_listrik"), 900),
            "Status_Subsidi_Listrik": general.get("status_subsidi", "Non Subsidi"),
        }

        if dataset_type == "prabayar":
            record.update(
                {
                    "Nominal_Token_Terakhir_Rp": to_number(general.get("nominal_token"), 100000),
                    "Frekuensi_Isi_Token_Per_Bulan": to_number(general.get("frekuensi_token"), 2),
                }
            )
        else:
            record.update(
                {
                    "Bulan_Tagihan": general.get("bulan_tagihan", "Januari"),
                    "Tagihan_Relatif_Stabil": general.get("tagihan_stabil", "Ya"),
                }
            )

        for device, meta in MAIN_DEVICES.items():
            data = payload.get("devices", {}).get(device, {})
            jumlah = max(0.0, to_number(data.get("jumlah"), 0))
            record[f"{device}_Jumlah"] = jumlah

            if jumlah <= 0:
                record[f"{device}_Kategori"] = meta["default_category"]
                record[f"{device}_EstimasiWattPerUnit"] = 0
                record[f"{device}_EstimasiJamPerHari"] = 0
                record[f"{device}_EstimasiFrekuensiPerMinggu"] = 0
                record[f"{device}_EstimasiDurasiSekaliPakaiJam"] = 0
                continue

            category = data.get("kategori") or meta["usage_categories"][-2]
            if category == meta["default_category"]:
                category = meta["usage_categories"][1]

            record[f"{device}_Kategori"] = category
            record[f"{device}_EstimasiWattPerUnit"] = to_number(data.get("watt"), meta["default_watt"])

            if device == "MesinCuci":
                record[f"{device}_EstimasiFrekuensiPerMinggu"] = to_number(
                    data.get("frekuensi"), meta["default_frequency"]
                )
                record[f"{device}_EstimasiDurasiSekaliPakaiJam"] = to_number(
                    data.get("durasi"), meta["default_duration"]
                )
            else:
                record[f"{device}_EstimasiJamPerHari"] = to_number(data.get("jam"), meta["default_hours"])

        other_devices = payload.get("other_devices", {})
        has_other = other_devices.get("enabled", False)
        record["Alat_Lain_Ada"] = "Ya" if has_other else "Tidak"

        items = other_devices.get("items", [])
        for index in range(1, 4):
            item = items[index - 1] if index <= len(items) else {}
            prefix = f"Alat_Lain_{index}"
            if not has_other:
                item = {}

            jenis = item.get("jenis") or "Tidak diisi"
            watt = max(0.0, to_number(item.get("watt"), 0))
            jam = max(0.0, to_number(item.get("jam"), 0))

            record[f"{prefix}_Jenis"] = jenis if watt > 0 and jam > 0 else "Tidak diisi"
            record[f"{prefix}_Kategori"] = "Tidak diisi"
            record[f"{prefix}_EstimasiWatt"] = watt
            record[f"{prefix}_EstimasiJamPerHari"] = jam
            record[f"{prefix}_Energi_WhPerHari"] = watt * jam
            record[f"{prefix}_Energi_kWhPerHari"] = watt * jam / 1000

        return record

    def _calculate_energy_columns(self, record):
        main_total = 0.0

        for device in ["Kulkas", "TV", "AC", "Kipas", "RiceCooker"]:
            jumlah = to_number(record.get(f"{device}_Jumlah"), 0)
            watt = to_number(record.get(f"{device}_EstimasiWattPerUnit"), 0)
            jam = to_number(record.get(f"{device}_EstimasiJamPerHari"), 0)
            wh = jumlah * watt * jam
            record[f"{device}_Energi_WhPerHari"] = wh
            record[f"{device}_Energi_kWhPerHari"] = wh / 1000
            main_total += wh / 1000

        jumlah = to_number(record.get("MesinCuci_Jumlah"), 0)
        watt = to_number(record.get("MesinCuci_EstimasiWattPerUnit"), 0)
        frekuensi = to_number(record.get("MesinCuci_EstimasiFrekuensiPerMinggu"), 0)
        durasi = to_number(record.get("MesinCuci_EstimasiDurasiSekaliPakaiJam"), 0)
        mesin_cuci_wh = jumlah * watt * frekuensi * durasi / 7
        record["MesinCuci_Energi_WhPerHari"] = mesin_cuci_wh
        record["MesinCuci_Energi_kWhPerHari"] = mesin_cuci_wh / 1000
        main_total += mesin_cuci_wh / 1000

        other_total = sum(
            to_number(record.get(f"Alat_Lain_{index}_Energi_kWhPerHari"), 0)
            for index in range(1, 4)
        )

        record["Total_Energi_Utama_WhPerHari"] = main_total * 1000
        record["Total_Energi_Utama_kWhPerHari"] = main_total
        record["Total_Energi_Alat_Lain_WhPerHari"] = other_total * 1000
        record["Total_Energi_Alat_Lain_kWhPerHari"] = other_total
        record["Total_Energi_Semua_WhPerHari"] = (main_total + other_total) * 1000
        record["Total_Energi_Semua_kWhPerHari"] = main_total + other_total

    def _format_prediction(self, dataset_type, value):
        value = max(0.0, value)
        if dataset_type == "prabayar":
            return f"{value:.1f} hari"
        return f"Rp {value:,.0f}".replace(",", ".")


SERVICE = PredictionService()


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/api/metadata":
            self._send_json(SERVICE.metadata())
            return

        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            path = "/index.html"

        file_path = (WEB_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.exists():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404, "Endpoint not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = SERVICE.predict(payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), WebHandler)
    print(f"Web app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
