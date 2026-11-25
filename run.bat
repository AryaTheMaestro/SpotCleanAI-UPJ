@echo off
REM SpotClean AI - jalankan aplikasi (Windows). Klik dua kali atau jalankan dari terminal.
REM Tidak memakai "activate", jadi tidak terkena masalah Execution Policy PowerShell.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python tidak ditemukan. Pasang Python 3.11 dari python.org dan centang "Add Python to PATH".
    pause
    exit /b 1
)

if not exist "models\best.pt" (
    echo [PERINGATAN] models\best.pt belum ada. Aplikasi tetap berjalan, tetapi deteksi akan gagal.
    echo Salin spotclean_yolov11_best.pt dari Google Drive ke folder models dan ganti namanya menjadi best.pt.
    echo.
)

if not exist ".venv\Scripts\python.exe" (
    echo Membuat virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

if not exist ".venv\.installed" (
    echo Memasang paket - pertama kali butuh beberapa menit...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    echo ok> ".venv\.installed"
)

echo.
echo Menjalankan SpotClean AI di http://localhost:8501  (tutup jendela ini untuk berhenti)
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:error
echo.
echo Terjadi kesalahan. Baca pesan di atas, lalu coba lagi.
pause
exit /b 1
