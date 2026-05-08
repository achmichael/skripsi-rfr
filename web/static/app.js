const DEVICE_CONFIG = {
  Kulkas: {
    label: "Kulkas",
    emptyCategory: "Tidak ada",
    categories: ["Tidak ada", "Kecil / 1 pintu", "Sedang / 2 pintu"],
    watt: 120,
    jam: 24
  },
  TV: {
    label: "TV",
    emptyCategory: "Tidak ada / tidak digunakan",
    categories: [
      "Tidak ada / tidak digunakan",
      "Jarang, kurang dari 2 jam per hari",
      "Sedang, sekitar 2-5 jam per hari",
      "Sering, sekitar 6-10 jam per hari",
      "Sangat sering, lebih dari 10 jam per hari"
    ],
    watt: 80,
    jam: 3.5
  },
  AC: {
    label: "AC",
    emptyCategory: "Tidak ada / tidak digunakan",
    categories: [
      "Tidak ada / tidak digunakan",
      "Jarang, kurang dari 2 jam per hari",
      "Sedang, sekitar 2-5 jam per hari",
      "Sering, sekitar 6-10 jam per hari",
      "Sangat sering, lebih dari 10 jam per hari"
    ],
    watt: 600,
    jam: 3.5
  },
  Kipas: {
    label: "Kipas",
    emptyCategory: "Tidak ada / tidak digunakan",
    categories: [
      "Tidak ada / tidak digunakan",
      "Jarang, kurang dari 2 jam per hari",
      "Sedang, sekitar 2-5 jam per hari",
      "Sering, sekitar 6-10 jam per hari",
      "Sangat sering, lebih dari 10 jam per hari"
    ],
    watt: 45,
    jam: 3.5
  },
  RiceCooker: {
    label: "Rice Cooker",
    emptyCategory: "Tidak ada / tidak digunakan",
    categories: [
      "Tidak ada / tidak digunakan",
      "Jarang, kurang dari 2 jam per hari",
      "Sedang, sekitar 2-5 jam per hari",
      "Sering, sekitar 6-10 jam per hari",
      "Sangat sering, lebih dari 10 jam per hari"
    ],
    watt: 300,
    jam: 3.5
  },
  MesinCuci: {
    label: "Mesin Cuci",
    emptyCategory: "Tidak ada / tidak digunakan",
    categories: [
      "Tidak ada / tidak digunakan",
      "Jarang, 1-2 kali per minggu",
      "Sedang, 3-4 kali per minggu",
      "Sering, 5-6 kali per minggu",
      "Sangat sering, hampir setiap hari"
    ],
    watt: 350,
    frekuensi: 3.5,
    durasi: 1.5
  }
};

const OTHER_TYPES = [
  "Tidak diisi",
  "Charger HP/perangkat kecil",
  "Komputer/Laptop",
  "Dispenser",
  "Setrika",
  "Pompa air",
  "Blender/Mixer",
  "Oven/Microwave",
  "Lainnya"
];

let activeDataset = "prabayar";
let metadata = null;

const form = document.querySelector("#predictionForm");
const deviceGrid = document.querySelector("#deviceGrid");
const otherGrid = document.querySelector("#otherGrid");
const otherEnabled = document.querySelector("#otherEnabled");
const resultValue = document.querySelector("#resultValue");
const resultLabel = document.querySelector("#resultLabel");
const dailyEnergy = document.querySelector("#dailyEnergy");
const monthlyEnergy = document.querySelector("#monthlyEnergy");
const activeModel = document.querySelector("#activeModel");
const modelStatus = document.querySelector("#modelStatus");

function optionList(values, selected) {
  return values
    .map((value) => `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(value)}</option>`)
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderDevices() {
  deviceGrid.innerHTML = Object.entries(DEVICE_CONFIG)
    .map(([key, config]) => {
      const isWasher = key === "MesinCuci";
      const usageFields = isWasher
        ? `
          <div class="mini-grid three">
            <label class="mini-field">
              <span>Watt</span>
              <input data-device="${key}" data-field="watt" type="number" min="0" step="1" value="${config.watt}">
            </label>
            <label class="mini-field">
              <span>Kali per minggu</span>
              <input data-device="${key}" data-field="frekuensi" type="number" min="0" step="0.5" value="${config.frekuensi}">
            </label>
            <label class="mini-field">
              <span>Durasi</span>
              <input data-device="${key}" data-field="durasi" type="number" min="0" step="0.5" value="${config.durasi}">
            </label>
          </div>`
        : `
          <div class="mini-grid">
            <label class="mini-field">
              <span>Watt</span>
              <input data-device="${key}" data-field="watt" type="number" min="0" step="1" value="${config.watt}">
            </label>
            <label class="mini-field">
              <span>Jam per hari</span>
              <input data-device="${key}" data-field="jam" type="number" min="0" step="0.5" value="${config.jam}">
            </label>
          </div>`;

      return `
        <article class="device-card" data-device-card="${key}">
          <div class="device-title">
            <strong>${config.label}</strong>
            <span class="usage-chip" data-device-chip="${key}">Aktif</span>
          </div>
          <div class="mini-grid">
            <label class="mini-field">
              <span>Jumlah</span>
              <input data-device="${key}" data-field="jumlah" type="number" min="0" step="1" value="${key === "AC" ? 0 : 1}">
            </label>
            <label class="mini-field">
              <span>Kategori</span>
              <select data-device="${key}" data-field="kategori">
                ${optionList(config.categories, key === "AC" ? config.emptyCategory : config.categories[Math.min(2, config.categories.length - 1)])}
              </select>
            </label>
          </div>
          ${usageFields}
        </article>
      `;
    })
    .join("");

  deviceGrid.querySelectorAll("input, select").forEach((input) => {
    input.addEventListener("input", syncDeviceState);
    input.addEventListener("change", syncDeviceState);
  });
  syncDeviceState();
}

function renderOtherDevices() {
  otherGrid.innerHTML = [1, 2, 3]
    .map((index) => `
      <article class="other-card">
        <strong>Alat lain ${index}</strong>
        <label class="mini-field">
          <span>Jenis</span>
          <select data-other="${index}" data-field="jenis">
            ${optionList(OTHER_TYPES, "Tidak diisi")}
          </select>
        </label>
        <div class="mini-grid">
          <label class="mini-field">
            <span>Watt</span>
            <input data-other="${index}" data-field="watt" type="number" min="0" step="1" value="0">
          </label>
          <label class="mini-field">
            <span>Jam per hari</span>
            <input data-other="${index}" data-field="jam" type="number" min="0" step="0.5" value="0">
          </label>
        </div>
      </article>
    `)
    .join("");
  syncOtherState();
}

function syncDeviceState() {
  Object.entries(DEVICE_CONFIG).forEach(([key, config]) => {
    const jumlah = Number(getDeviceInput(key, "jumlah").value || 0);
    const active = jumlah > 0;
    const category = getDeviceInput(key, "kategori");
    const chip = document.querySelector(`[data-device-chip="${key}"]`);

    category.disabled = !active;
    if (!active) {
      category.value = config.emptyCategory;
    } else if (category.value === config.emptyCategory) {
      category.value = config.categories[Math.min(1, config.categories.length - 1)];
    }

    ["watt", "jam", "frekuensi", "durasi"].forEach((field) => {
      const input = getDeviceInput(key, field);
      if (!input) return;
      input.disabled = !active;
      if (!active) input.value = 0;
    });

    chip.textContent = active ? "Aktif" : "Tidak aktif";
  });
}

function syncOtherState() {
  const enabled = otherEnabled.checked;
  otherGrid.classList.toggle("disabled", !enabled);
  otherGrid.querySelectorAll("input, select").forEach((input) => {
    input.disabled = !enabled;
  });
}

function getDeviceInput(device, field) {
  return document.querySelector(`[data-device="${device}"][data-field="${field}"]`);
}

function setDataset(dataset) {
  activeDataset = dataset;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.dataset === dataset);
  });
  document.querySelectorAll(".dataset-only").forEach((section) => {
    section.classList.toggle("hidden", section.dataset.show !== dataset);
  });
  updateActiveModel();
}

function collectPayload() {
  const formData = new FormData(form);
  const devices = {};
  Object.keys(DEVICE_CONFIG).forEach((key) => {
    const data = {};
    document.querySelectorAll(`[data-device="${key}"]`).forEach((input) => {
      data[input.dataset.field] = input.value;
    });
    devices[key] = data;
  });

  const items = [1, 2, 3].map((index) => {
    const item = {};
    document.querySelectorAll(`[data-other="${index}"]`).forEach((input) => {
      item[input.dataset.field] = input.value;
    });
    return item;
  });

  return {
    dataset_type: activeDataset,
    general: {
      jumlah_anggota: formData.get("jumlah_anggota"),
      daya_listrik: formData.get("daya_listrik"),
      status_subsidi: formData.get("status_subsidi"),
      nominal_token: formData.get("nominal_token"),
      frekuensi_token: formData.get("frekuensi_token"),
      bulan_tagihan: formData.get("bulan_tagihan"),
      tagihan_stabil: formData.get("tagihan_stabil")
    },
    devices,
    other_devices: {
      enabled: otherEnabled.checked,
      items
    }
  };
}

async function submitPrediction(event) {
  event.preventDefault();
  syncDeviceState();
  resultValue.textContent = "Menghitung";
  resultLabel.textContent = "Model sedang memproses input.";
  resultLabel.classList.remove("error");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload())
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediksi gagal.");
    }
    renderResult(data);
  } catch (error) {
    resultValue.textContent = "-";
    resultLabel.textContent = error.message;
    resultLabel.classList.add("error");
  }
}

function renderResult(data) {
  resultValue.textContent = data.formatted_prediction;
  resultLabel.textContent = data.dataset_type === "prabayar"
    ? "Estimasi durasi token listrik."
    : "Estimasi rata-rata tagihan bulanan.";
  resultLabel.classList.remove("error");
  dailyEnergy.textContent = `${data.energy.total_kwh_per_day.toFixed(2)} kWh`;
  monthlyEnergy.textContent = `${data.energy.total_kwh_per_month.toFixed(1)} kWh`;
  activeModel.textContent = data.model_path.split(/[\\/]/).pop();
}

function resetForm() {
  form.reset();
  setDataset("prabayar");
  renderDevices();
  renderOtherDevices();
  resultValue.textContent = "-";
  resultLabel.textContent = "Belum ada hasil prediksi.";
  resultLabel.classList.remove("error");
  dailyEnergy.textContent = "-";
  monthlyEnergy.textContent = "-";
  updateActiveModel();
}

async function loadMetadata() {
  try {
    const response = await fetch("/api/metadata");
    metadata = await response.json();
    modelStatus.textContent = "Model siap digunakan";
    updateActiveModel();
  } catch (error) {
    modelStatus.textContent = "Model belum siap";
    modelStatus.classList.add("error");
  }
}

function updateActiveModel() {
  if (!metadata || !metadata.datasets || !metadata.datasets[activeDataset]) {
    activeModel.textContent = "-";
    return;
  }
  const modelPath = metadata.datasets[activeDataset].model_path || "";
  activeModel.textContent = modelPath.split(/[\\/]/).pop() || "-";
}

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => setDataset(button.dataset.dataset));
});

otherEnabled.addEventListener("change", syncOtherState);
form.addEventListener("submit", submitPrediction);
document.querySelector("#resetButton").addEventListener("click", resetForm);

renderDevices();
renderOtherDevices();
loadMetadata();
