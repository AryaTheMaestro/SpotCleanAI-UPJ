"""SpotClean AI - antarmuka mobile web (Streamlit).

Alur (sesuai diagram arsitektur):
    unggah foto -> validasi input -> inferensi YOLO11 -> visualisasi + statistik -> hasil

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
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


def detect(
    model,
    image: Image.Image,
    conf: float | None = None,
    iou: float | None = None,
) -> list[Detection]:
    """Jalankan inferensi dan kembalikan prediksi, diurutkan dari confidence tertinggi.

    conf  : ambang confidence minimum (default config.CONF_THRESHOLD)
    iou   : ambang IoU untuk NMS (default config.IOU_THRESHOLD)
    """
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou
    try:
        # Ultralytics menganggap array numpy berformat BGR, sedangkan gambar kita RGB.
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        results = model.predict(
            source=bgr,
            imgsz=config.IMG_SIZE,
            conf=conf,
            iou=iou,
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
CSS = """
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.sc-hero {padding: 1.1rem 1.25rem; border-radius: 14px; margin-bottom: 1rem;
          background: linear-gradient(135deg, #2a9d8f 0%, #264653 100%); color: #fff;}
.sc-hero h1 {margin: 0; padding: 0; font-size: 1.8rem; color: #fff;}
.sc-hero p {margin: .35rem 0 0 0; opacity: .92; font-size: .92rem;}
.sc-cards {display: flex; flex-wrap: wrap; gap: .5rem; margin: .25rem 0 1rem 0;}
.sc-card {flex: 1 1 100px; padding: .6rem .7rem; border-radius: 10px;
          background: rgba(128,128,128,.12); text-align: center;}
.sc-n {font-size: 1.6rem; font-weight: 700; line-height: 1.2;}
.sc-l {font-size: .8rem; opacity: .8;}
.sc-zero {opacity: .55;}
</style>
"""


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _reset_thresholds() -> None:
    st.session_state["conf_threshold"] = _clamp(config.CONF_THRESHOLD, config.CONF_RANGE)
    st.session_state["iou_threshold"] = _clamp(config.IOU_THRESHOLD, config.IOU_RANGE)


def threshold_controls() -> tuple[float, float]:
    """Slider ambang deteksi; nilainya bertahan selama sesi dan dapat dikembalikan ke bawaan."""
    if "conf_threshold" not in st.session_state or "iou_threshold" not in st.session_state:
        _reset_thresholds()

    with st.expander("⚙️ Pengaturan deteksi"):
        conf = st.slider(
            "Confidence minimum",
            min_value=config.CONF_RANGE[0],
            max_value=config.CONF_RANGE[1],
            step=0.05,
            key="conf_threshold",
            help="Hanya prediksi dengan keyakinan di atas nilai ini yang ditampilkan. "
            "Naikkan untuk mengurangi prediksi ragu-ragu; turunkan agar lebih banyak objek terdeteksi.",
        )
        iou = st.slider(
            "Batas tumpang tindih kotak (IoU)",
            min_value=config.IOU_RANGE[0],
            max_value=config.IOU_RANGE[1],
            step=0.05,
            key="iou_threshold",
            help="Dua kotak sekelas yang tumpang tindihnya melebihi nilai ini dianggap objek yang sama, "
            "dan yang keyakinannya lebih rendah dibuang. Turunkan bila muncul banyak kotak ganda.",
        )
        st.button("Kembalikan ke bawaan", on_click=_reset_thresholds)
    return float(conf), float(iou)


def run_detection(image: Image.Image, key: tuple, conf: float, iou: float) -> dict:
    """Jalankan deteksi; hasil terakhir disimpan di sesi agar tidak diulang tanpa perubahan."""
    cached = st.session_state.get("result")
    if cached and cached["key"] == key:
        return cached

    start = time.perf_counter()
    with st.spinner("Memproses..."):
        model = load_model(str(config.MODEL_PATH))
        detections = detect(model, image, conf=conf, iou=iou)
    result = {
        "key": key,
        "detections": detections,
        "elapsed": time.perf_counter() - start,
        "warnings": check_class_mapping(model),
        "conf": conf,
        "iou": iou,
    }
    st.session_state["result"] = result
    return result


def count_cards_html(counts: dict[str, int]) -> str:
    colors = {name: config.CLASS_COLORS.get(cid, (150, 150, 150)) for cid, name in config.CLASS_NAMES.items()}
    cards = []
    for name, n in counts.items():
        r, g, b = colors.get(name, (150, 150, 150))
        zero = " sc-zero" if n == 0 else ""
        cards.append(
            f'<div class="sc-card{zero}" style="border-top:4px solid rgb({r},{g},{b})">'
            f'<div class="sc-n">{n}</div><div class="sc-l">{name}</div></div>'
        )
    return '<div class="sc-cards">' + "".join(cards) + "</div>"


def show_results(image: Image.Image, result: dict) -> None:
    detections: list[Detection] = result["detections"]

    for warning in result["warnings"]:
        st.warning(warning)

    st.subheader("Hasil deteksi")
    counts = count_by_class(detections)
    total = len(detections)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total deteksi", total)
    top = max(counts, key=counts.get) if total else "-"
    col2.metric("Kelas terbanyak", top)
    col3.metric("Waktu proses", f"{result['elapsed']:.2f} dtk")

    if detections:
        annotated = draw_detections(image, detections)
        st.image(annotated, caption="Hasil deteksi")
        buffer = BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button("Unduh gambar hasil", buffer.getvalue(), file_name="spotclean_hasil.png", mime="image/png")
    else:
        # Hasil kosong adalah hasil normal, bukan kegagalan (US-05).
        st.info(
            "Tidak ada objek yang memenuhi kriteria deteksi "
            f"(confidence ≥ {result['conf']:.2f}). Coba turunkan confidence minimum di Pengaturan deteksi."
        )

    st.markdown("**Jumlah per kelas**")
    st.markdown(count_cards_html(counts), unsafe_allow_html=True)

    if detections:
        with st.expander("Detail prediksi"):
            st.table(
                [
                    {"Kelas": d.label, "Confidence": f"{d.confidence:.2f}", "Posisi (x1,y1,x2,y2)": str(d.box)}
                    for d in detections
                ]
            )

    st.caption(f"Confidence ≥ {result['conf']:.2f} · IoU {result['iou']:.2f}")


def main() -> None:
    st.set_page_config(page_title="SpotClean AI", page_icon="🧹", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="sc-hero"><h1>🧹 SpotClean AI</h1>'
        "<p>Deteksi crack, mold, peeling paint, stain, dan trash dari foto fasilitas. "
        "Hasil adalah bantuan inspeksi dan wajib ditinjau petugas.</p></div>",
        unsafe_allow_html=True,
    )

    conf, iou = threshold_controls()

    # Tanpa pembatasan 'type' agar semua kesalahan format ditangani validate_upload
    # dengan pesan yang seragam.
    uploaded = st.file_uploader("Unggah foto (JPG, JPEG, PNG)")
    if uploaded is None:
        st.info("Unggah satu foto untuk memulai.")
        return

    data = uploaded.getvalue()
    try:
        image = validate_upload(uploaded.name, data)
    except InputError as exc:
        st.error(f"Input tidak valid: {exc}")
        return

    st.image(image, caption=uploaded.name)

    file_key = hashlib.md5(data).hexdigest()
    if st.button("Deteksi gangguan", type="primary"):
        st.session_state["active_file"] = file_key
    if st.session_state.get("active_file") != file_key:
        return

    # Setelah foto ini dideteksi sekali, menggeser slider langsung memperbarui hasil.
    try:
        result = run_detection(image, (file_key, round(conf, 3), round(iou, 3)), conf, iou)
    except InferenceError as exc:
        st.error(f"Proses deteksi gagal: {exc}")
        return

    show_results(image, result)


if __name__ == "__main__":
    main()
