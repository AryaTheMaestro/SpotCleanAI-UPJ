"""Konfigurasi terpusat SpotClean AI.

Semua nilai yang dapat berubah (model, threshold, batas unggahan, pemetaan kelas)
dikelola di sini, terpisah dari kode aplikasi (SRS 11.3).

Beberapa nilai dapat ditimpa lewat variabel lingkungan sehingga staging dan
produksi dapat berbeda tanpa membangun ulang image:

    SPOTCLEAN_MODEL_PATH       lokasi berkas bobot model (default: models/best.pt)
    SPOTCLEAN_CONF_THRESHOLD   ambang confidence (default: 0.25)
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Lokasi bobot model hasil pelatihan (best.pt), disimpan terpisah dari kode.
MODEL_PATH = Path(os.environ.get("SPOTCLEAN_MODEL_PATH", BASE_DIR / "models" / "best.pt"))

# Ambang confidence (usulan awal, sesuaikan setelah uji).
CONF_THRESHOLD = float(os.environ.get("SPOTCLEAN_CONF_THRESHOLD", "0.25"))

# Sama dengan ukuran input saat pelatihan.
IMG_SIZE = 640

# Batas unggahan (usulan awal). Samakan dengan maxUploadSize di .streamlit/config.toml.
MAX_UPLOAD_MB = 10

# Batas jumlah piksel gambar (lebar x tinggi) agar tidak membebani memori server.
MAX_IMAGE_PIXELS = 25_000_000

# Format yang didukung (SRS 8.2): hanya foto/gambar.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}  # format sebenarnya yang dibaca Pillow

# Pemetaan kelas sesuai data.yaml saat pelatihan.
CLASS_NAMES = {
    0: "crack",
    1: "mold",
    2: "peeling paint",
    3: "stain",
    4: "trash",
}

# Warna kotak per kelas (RGB).
CLASS_COLORS = {
    0: (230, 57, 70),
    1: (42, 157, 143),
    2: (244, 162, 97),
    3: (69, 123, 157),
    4: (131, 56, 236),
}
