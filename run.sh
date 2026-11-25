#!/usr/bin/env bash
# SpotClean AI - jalankan aplikasi (macOS/Linux).  Pemakaian:  ./run.sh   atau   bash run.sh
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3 tidak ditemukan. Pasang Python 3.11 terlebih dahulu."; exit 1; }

if [ ! -f models/best.pt ]; then
  echo "[PERINGATAN] models/best.pt belum ada. Aplikasi tetap berjalan, tetapi deteksi akan gagal."
  echo "Salin spotclean_yolov11_best.pt dari Google Drive ke folder models/ dan ganti namanya menjadi best.pt."
  echo
fi

if [ ! -x .venv/bin/python ]; then
  echo "Membuat virtual environment..."
  "$PY" -m venv .venv
fi

if [ ! -f .venv/.installed ]; then
  echo "Memasang paket (pertama kali butuh beberapa menit)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed
fi

echo
echo "Menjalankan SpotClean AI di http://localhost:8501  (Ctrl+C untuk berhenti)"
exec .venv/bin/python -m streamlit run app.py "$@"
