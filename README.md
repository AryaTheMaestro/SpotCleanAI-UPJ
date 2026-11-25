# SpotClean AI

Aplikasi web mobile berbasis Streamlit untuk mendeteksi gangguan kebersihan dan kondisi
fasilitas gedung dari foto, memakai model YOLO11n. Pengguna mengunggah foto, lalu aplikasi
menampilkan bounding box, label kelas, confidence score, dan jumlah deteksi per kelas.

> Hasil deteksi adalah alat bantu inspeksi dan wajib ditinjau petugas. Bukan keputusan otomatis.

## Fitur

- Unggah foto JPG, JPEG, atau PNG (maksimal 10 MB); format dan isi berkas divalidasi.
- Deteksi 5 kelas: `crack`, `mold`, `peeling paint`, `stain`, `trash`.
- Visualisasi bounding box dengan label dan confidence, kartu jumlah per kelas, dan unduh gambar hasil.
- Confidence dan IoU dapat diatur langsung di halaman web.
- Pesan yang jelas bila tidak ada objek terdeteksi atau model gagal dimuat.
- Siap dijalankan lewat Docker di belakang reverse proxy HTTPS.

## Performa model

Hasil pada data uji (575 gambar), pelatihan 8 epoch, ukuran input 640:

| Kelas | Precision | Recall | mAP@50 |
| --- | --- | --- | --- |
| crack | 0,698 | 0,506 | 0,595 |
| mold | 0,904 | 0,507 | 0,745 |
| peeling paint | 0,823 | 0,786 | 0,822 |
| stain | 0,688 | 0,701 | 0,708 |
| trash | 0,714 | 0,568 | 0,610 |
| **Semua** | **0,766** | **0,614** | **0,696** |

mAP@50-95 keseluruhan 0,440. Ini adalah baseline awal; nilai tersebut belum memenuhi seluruh
target di SRS (mAP@50 ≥ 0,70 dan recall ≥ 0,70 per kelas). Waktu inferensi sekitar 4,5 ms per
gambar pada GPU Tesla T4.

## Jalankan lokal

Prasyarat: Python 3.11 dan berkas model `best.pt` di folder `models/`.

**Cara cepat (disarankan)**

- Windows: klik dua kali `run.bat`, atau jalankan `.\run.bat` di terminal.
- macOS/Linux: `bash run.sh`

Skrip membuat virtual environment dan memasang paket otomatis pada pemakaian pertama
(beberapa menit), lalu membuka aplikasi di http://localhost:8501. Tidak perlu
`activate`, jadi aman dari masalah Execution Policy di PowerShell.

**Cara manual**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (cmd): .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Untuk uji dari smartphone di Wi-Fi yang sama:
`streamlit run app.py --server.address 0.0.0.0`, lalu buka `http://<IP-komputer>:8501`.

## Mengatur ambang deteksi

Di halaman web, buka **Pengaturan deteksi**:

| Pengaturan | Fungsi |
| --- | --- |
| Confidence minimum | Prediksi di bawah nilai ini disembunyikan. Naikkan untuk mengurangi tebakan ragu-ragu. |
| Batas tumpang tindih kotak (IoU) | Kotak sekelas yang tumpang tindihnya melebihi nilai ini dianggap satu objek. Turunkan bila muncul kotak ganda. |

Setelah satu foto dideteksi, menggeser slider langsung memperbarui hasil. Nilai awal
diatur di `config.py` atau lewat variabel lingkungan.

## Docker

```bash
# server tanpa GPU (image lebih kecil):
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t spotclean-ai:1.0 .

docker run -d --name spotclean \
  -p 127.0.0.1:8501:8501 \
  -v "$(pwd)/models:/app/models:ro" \
  --restart unless-stopped \
  spotclean-ai:1.0
```

Pasang reverse proxy HTTPS (mis. Caddy) di depan port 8501; jangan membuka 8501 ke publik.

## Konfigurasi

Nilai bawaan ada di `config.py` dan bisa diganti lewat variabel lingkungan:

| Variabel | Bawaan | Fungsi |
| --- | --- | --- |
| `SPOTCLEAN_MODEL_PATH` | `models/best.pt` | Lokasi bobot model |
| `SPOTCLEAN_CONF_THRESHOLD` | `0.25` | Nilai awal slider confidence |
| `SPOTCLEAN_IOU_THRESHOLD` | `0.50` | Nilai awal slider IoU (NMS) |

## Struktur proyek

| Berkas | Fungsi |
| --- | --- |
| `app.py` | Antarmuka, validasi input, inferensi, visualisasi, statistik |
| `config.py` | Model, threshold, batas unggahan, pemetaan kelas |
| `models/best.pt` | Bobot model (tidak ikut repositori, lihat `models/README.txt`) |
| `.streamlit/config.toml` | Pengaturan server Streamlit |
| `run.bat`, `run.sh` | Skrip sekali klik untuk menjalankan aplikasi |
| `Dockerfile` | Image untuk staging dan produksi |

## Model dan data

Model dilatih dari dataset Roboflow "SpotClean" v2 (5 kelas, gambar 512×512, pembagian
80:10:10) dengan Ultralytics YOLO11n. Bobot tidak disertakan di repositori; salin berkas
`best.pt` hasil pelatihan ke folder `models/`.

## Lisensi

Belum ditentukan. Perhatikan bahwa Ultralytics dirilis dengan lisensi AGPL-3.0, yang berlaku
juga bagi aplikasi yang memakainya.
