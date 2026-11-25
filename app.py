"""SpotClean AI - antarmuka mobile web (Streamlit).

Alur (sesuai diagram arsitektur):
    unggah foto -> validasi input -> inferensi YOLO11 -> visualisasi + statistik -> hasil

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import config


class InputError(Exception):
    """File yang diunggah tidak valid (format, ukuran, atau tidak terbaca)."""


class InferenceError(Exception):
    """Model tidak dapat dimuat atau proses inferensi gagal."""


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 dalam piksel


# --------------------------------------------------------------------------- #
# Validasi input
# --------------------------------------------------------------------------- #
def validate_upload(filename: str, data: bytes) -> Image.Image:
    """Periksa format, ukuran, dan keterbacaan berkas; kembalikan gambar RGB.

    Memeriksa isi berkas (bukan hanya ekstensi), sehingga PDF/MP4 yang diganti
    namanya menjadi .jpg tetap ditolak.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in config.ALLOWED_EXTENSIONS))
        raise InputError(f"Format '{ext or 'tanpa ekstensi'}' tidak didukung. Gunakan {allowed}.")

    if not data:
        raise InputError("Berkas kosong.")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise InputError(f"Ukuran berkas melebihi batas {config.MAX_UPLOAD_MB} MB.")

    try:
        with Image.open(BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            probe.verify()
    except Exception as exc:  # berkas rusak / bukan gambar
        raise InputError("Berkas rusak atau bukan gambar yang dapat dibaca.") from exc

    if fmt not in config.ALLOWED_FORMATS:
        raise InputError(f"Isi berkas berformat {fmt}, bukan JPG/JPEG/PNG.")

    if width * height > config.MAX_IMAGE_PIXELS:
        raise InputError("Resolusi gambar terlalu besar untuk diproses.")

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()  # memicu galat bila berkas terpotong
            img = ImageOps.exif_transpose(img)  # foto ponsel sering memakai rotasi EXIF
            return img.convert("RGB")
    except Exception as exc:
        raise InputError("Berkas rusak atau bukan gambar yang dapat dibaca.") from exc


# --------------------------------------------------------------------------- #
# Model dan inferensi
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str):
    """Muat model sekali dan pakai ulang untuk semua unggahan."""
    model_file = Path(path)
    if not model_file.is_file():
        raise InferenceError(f"Berkas model tidak ditemukan: {model_file}")
    try:
        from ultralytics import YOLO

        return YOLO(str(model_file))
    except Exception as exc:
        raise InferenceError(f"Model gagal dimuat: {exc}") from exc


def check_class_mapping(model) -> list[str]:
    """Bandingkan nama kelas pada model dengan CLASS_NAMES di config.py."""
    names = getattr(model, "names", None)
    if not isinstance(names, dict):
        return []
    model_names = {int(k): str(v).strip().lower() for k, v in names.items()}
    config_names = {k: v.strip().lower() for k, v in config.CLASS_NAMES.items()}
    if model_names != config_names:
        return [
            "Nama kelas pada model tidak sama dengan CLASS_NAMES di config.py. "
            f"Model: {model_names}; config: {config_names}."
        ]
    return []


def _to_numpy(value) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def detect(model, image: Image.Image) -> list[Detection]:
    """Jalankan inferensi dan kembalikan prediksi, diurutkan dari confidence tertinggi."""
    try:
        # Ultralytics menganggap array numpy berformat BGR, sedangkan gambar kita RGB.
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        results = model.predict(
            source=bgr,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        confs = _to_numpy(boxes.conf).reshape(-1)
        classes = _to_numpy(boxes.cls).reshape(-1)
    except Exception as exc:
        raise InferenceError(f"Inferensi gagal: {exc}") from exc

    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        class_id = int(cls)
        detections.append(
            Detection(
                class_id=class_id,
                label=config.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                confidence=float(conf),
                box=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# --------------------------------------------------------------------------- #
# Statistik dan visualisasi
# --------------------------------------------------------------------------- #
def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """Jumlah prediksi per kelas; kelas tanpa deteksi bernilai 0 (US-04, US-05)."""
    counts = {name: 0 for name in config.CLASS_NAMES.values()}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Gambar bounding box, label kelas, dan confidence score pada salinan gambar."""
    canvas = np.array(image.convert("RGB"))  # salinan, RGB
    height, width = canvas.shape[:2]
    short_side = min(height, width)
    thickness = max(2, round(short_side / 300))
    font_scale = max(0.5, short_side / 900)

    for det in detections:
        x1, y1, x2, y2 = det.box
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        color = config.CLASS_COLORS.get(det.class_id, (255, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        text = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = y1 - th - baseline - 4
        if top < 0:  # kotak menempel di tepi atas: letakkan label di dalam kotak
            top = y1
        cv2.rectangle(canvas, (x1, top), (min(x1 + tw + 6, width - 1), top + th + baseline + 4), color, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 3, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return Image.fromarray(canvas)


# --------------------------------------------------------------------------- #
# Antarmuka
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="SpotClean AI", page_icon="🧹", layout="centered")
    st.title("SpotClean AI")
    st.caption(
        "Deteksi crack, mold, peeling paint, stain, dan trash dari foto fasilitas. "
        "Hasil adalah bantuan inspeksi dan wajib ditinjau petugas."
    )

    # Tanpa pembatasan 'type' agar semua kesalahan format ditangani validate_upload
    # dengan pesan yang seragam.
    uploaded = st.file_uploader("Unggah foto (JPG, JPEG, PNG)")
    if uploaded is None:
        st.info("Unggah satu foto untuk memulai.")
        return

    try:
        image = validate_upload(uploaded.name, uploaded.getvalue())
    except InputError as exc:
        st.error(f"Input tidak valid: {exc}")
        return

    st.image(image, caption=uploaded.name)

    if not st.button("Deteksi gangguan", type="primary"):
        return

    start = time.perf_counter()
    try:
        with st.spinner("Memproses..."):
            model = load_model(str(config.MODEL_PATH))
            detections = detect(model, image)
    except InferenceError as exc:
        st.error(f"Proses deteksi gagal: {exc}")
        return
    elapsed = time.perf_counter() - start

    for warning in check_class_mapping(model):
        st.warning(warning)

    st.subheader("Hasil deteksi")
    if detections:
        st.image(draw_detections(image, detections), caption="Hasil deteksi")
    else:
        # Hasil kosong adalah hasil normal, bukan kegagalan (US-05).
        st.info(
            "Tidak ada objek yang memenuhi kriteria deteksi "
            f"(confidence ≥ {config.CONF_THRESHOLD:.2f})."
        )

    st.markdown("**Jumlah per kelas**")
    st.table([{"Kelas": name, "Jumlah": n} for name, n in count_by_class(detections).items()])

    if detections:
        with st.expander("Detail prediksi"):
            st.table(
                [
                    {"Kelas": d.label, "Confidence": f"{d.confidence:.2f}", "Posisi (x1,y1,x2,y2)": str(d.box)}
                    for d in detections
                ]
            )

    st.caption(f"Waktu proses {elapsed:.2f} detik · threshold {config.CONF_THRESHOLD:.2f}")


if __name__ == "__main__":
    main()
