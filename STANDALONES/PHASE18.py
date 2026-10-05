"""PHASE 18 - Face verification as the system's START passkey.

Standalone version of the PySide6 face dialog in COBOT_GUI.py. It copies the
same theme, preview and controls without importing COBOT_GUI, YOLO or robot code.
Uses the project-root faces_db.npz and InsightFace.

Run:       python STANDALONES/PHASE18.py
Test:      python STANDALONES/PHASE18.py --selftest
Controls:  Cancel / Escape / close the dialog to cancel.

face_verified starts False and becomes True only after the same enrolled face
is the only face detected for three consecutive frames. verify_face() returns
that boolean too, so a caller can use: face_verified = verify_face().
Each new attempt resets the result. Unknown faces, cancellation and errors
leave it False. Recognition does not check liveness; photos may be accepted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import os
import sys
import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QSizePolicy,
)


BASE_DIR = Path(__file__).resolve().parent.parent
FACE_DB_PATH = BASE_DIR / "faces_db.npz"
FACE_THRESHOLD = 0.50
FACE_UNLOCK_FRAMES = 3
FACE_UNLOCK_TIMEOUT_S = 30.0
CAM_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FLIP_VIEW = True
BGR_CYAN = (238, 211, 34)
BGR_WARN = (11, 158, 245)
BGR_WHITE = (245, 245, 245)
BGR_DARK = (20, 16, 12)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Results of the most recent attempt. Import the module to read updated values:
# import PHASE18; PHASE18.verify_face(); print(PHASE18.face_verified)
face_verified = False
verified_name = None


class VerificationStreak:
    """Require one consistent identity; any missing/unknown/extra face resets it."""

    def __init__(self):
        self.candidate = None
        self.count = 0

    def update(self, names):
        name = names[0] if len(names) == 1 else None
        if not name or name == "Unknown":
            self.candidate, self.count = None, 0
            return False
        self.count = self.count + 1 if name == self.candidate else 1
        self.candidate = name
        return self.count >= FACE_UNLOCK_FRAMES


class C:
    """Central colour palette."""
    BG          = "#080b15"
    BG2         = "#0d1322"
    PANEL       = "#0f1626"
    CARD        = "#121a2e"
    CARD2       = "#0e1525"
    BORDER      = "#283556"
    BORDER_SOFT = "#1b2440"
    TRACK       = "#1d2742"
    TEXT        = "#eef2ff"
    MUTED       = "#8c96b6"
    FAINT       = "#525d80"

    BLUE        = "#3b82f6"
    CYAN        = "#22d3ee"
    VIOLET      = "#a39abd"
    GREEN       = "#22c55e"
    AMBER       = "#f59e0b"
    PINK        = "#ec4899"
    RED         = "#ef4444"
    ORANGE      = "#fb923c"
    TEAL        = "#2dd4bf"

    ACCENT      = BLUE
    SAFE        = GREEN
    DANGER      = RED
    WARN        = AMBER


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        color: {C.TEXT};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
    }}
    QWidget#root {{
        background: qlineargradient(x1:0, y1:0, x2:0.6, y2:1,
            stop:0 #080b15, stop:1 #0c1426);
    }}
    QFrame#card {{
        background: {C.CARD};
        border: 1px solid {C.BORDER_SOFT};
        border-radius: 13px;
    }}
    QFrame#dock {{
        background: {C.BG2};
        border: 1px solid {C.BORDER_SOFT};
        border-radius: 14px;
    }}
    QFrame#header {{
        background: #ffffff;
        border: 1px solid #6f4ed1;
        border-radius: 14px;
    }}
    QLabel#h1 {{ font-size: 20px; font-weight: 900; letter-spacing: 1px;
                 color: #1a0640; }}
    QLabel#h2 {{ font-size: 11px; font-weight: 700; color: #3a1d7a; letter-spacing: 3px; }}
    QLabel#section {{ color: {C.CYAN}; font-size: 11px; font-weight: 800;
                      letter-spacing: 2px; }}
    QLabel#panelTitle {{ font-size: 12px; font-weight: 800; letter-spacing: 1px; }}

    QPushButton {{
        background: {C.CARD}; border: 1px solid {C.BORDER}; border-radius: 9px;
        padding: 9px 14px; font-weight: 600; color: {C.TEXT};
    }}
    QPushButton:hover {{ border: 1px solid {C.CYAN}; }}
    QPushButton:pressed {{ background: {C.PANEL}; }}
    QPushButton:disabled {{ color: {C.FAINT}; border-color: {C.BORDER_SOFT}; }}
    QPushButton#primary {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #10b981, stop:1 #22d3ee);
        color: #05231a; border: none; font-weight: 800; font-size: 14px;
    }}
    QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #14d39b, stop:1 #4be3f7); }}
    QPushButton#danger {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #ef4444, stop:1 #ec4899);
        color: #2b0710; border: none; font-weight: 800; font-size: 14px;
    }}

    QComboBox {{ background: {C.CARD}; border: 1px solid {C.BORDER};
        border-radius: 8px; padding: 6px 10px; }}
    QComboBox:hover {{ border: 1px solid {C.CYAN}; }}
    QComboBox QAbstractItemView {{ background: {C.PANEL};
        border: 1px solid {C.BORDER}; selection-background-color: {C.BLUE};
        outline: none; }}

    QCheckBox {{ color: {C.TEXT}; spacing: 8px; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px;
        border: 1px solid {C.BORDER}; background: {C.CARD}; }}
    QCheckBox::indicator:checked {{ background: {C.CYAN}; border-color: {C.CYAN}; }}

    QSlider::groove:horizontal {{ height: 5px; background: {C.TRACK};
        border-radius: 3px; }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {C.BLUE}, stop:1 {C.CYAN});
        border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: #ffffff; width: 15px; height: 15px;
        margin: -6px 0; border-radius: 8px; }}
    QSlider::handle:horizontal:hover {{ background: {C.CYAN}; }}

    QPlainTextEdit {{ background: {C.CARD2}; border: 1px solid {C.BORDER_SOFT};
        border-radius: 10px; color: {C.MUTED};
        font-family: 'Consolas', 'Cascadia Mono', monospace; font-size: 11px;
        padding: 6px; }}
    """


def qc(h: str, a: int = 255) -> QColor:
    c = QColor(h); c.setAlpha(a); return c


def _label(frame, text, x, y, bg, fg=BGR_WHITE, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 8, y), bg, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 4, y - 5), FONT, scale, fg, 1, cv2.LINE_AA)


class FaceRecognizer:


    def __init__(self, db_path, threshold, device):
        self.available = False
        self.error = ""
        self.threshold = threshold
        self.names = ()
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            # Read numeric arrays only; do not allow pickled objects in the database.
            with np.load(db_path, allow_pickle=False) as data:
                names = data["names"].astype(str)
                embeddings = data["embeddings"].astype(np.float32)
            if (names.ndim != 1 or len(names) == 0 or embeddings.ndim != 2
                    or embeddings.shape != (len(names), 512)
                    or not np.all(np.isfinite(embeddings))
                    or any(not n.strip() or n == "Unknown" for n in names)):
                raise ValueError("Face database is empty or invalid. Run build_database.py.")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError("Face database contains an invalid embedding.")
            embeddings /= norms
    
            self.names = names
            self._db = embeddings

            avail = ort.get_available_providers()
            if device == "cuda" and "CUDAExecutionProvider" in avail:
                providers, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
            else:
                providers, ctx = ["CPUExecutionProvider"], -1
            self.app = FaceAnalysis(name="buffalo_l",
                                    allowed_modules=["detection", "recognition"],
                                    providers=providers)
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))
            self.available = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def identify(self, frame):
        """Return [(name, score, (x1, y1, x2, y2)), ...] for faces in `frame`."""
        if not self.available:
            return []
        out = []
        for face in self.app.get(frame):
            emb = face.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-9)
            sims = self._db @ emb
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            name = str(self.names[idx]) if score >= self.threshold else "Unknown"
            x1, y1, x2, y2 = face.bbox.astype(int)
            out.append((name, score, (x1, y1, x2, y2)))
        return out


def _draw_faces(frame, faces):
    for name, score, (x1, y1, x2, y2) in faces:
        known = name != "Unknown"
        col = BGR_CYAN if known else BGR_WARN
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        _label(frame, f"{name}", x1, max(0, y1 - 4), BGR_DARK, col)


def bgr_to_qimage(frame_bgr) -> QImage:
    """OpenCV BGR -> deep-copied QImage (safe cross-thread handoff)."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class VideoView(QWidget):


    rightClicked = Signal()

    def __init__(self, placeholder="NO SIGNAL", parent=None, lock_aspect=True):
        super().__init__(parent)
        self._img = None
        self._lock_aspect = lock_aspect
        self._placeholder = placeholder
        self._estop = False
        self._flashing = False
        self._flash_on = False
        self._flash_blink = QTimer(self)
        self._flash_blink.setInterval(220)
        self._flash_blink.timeout.connect(self._toggle_flash)
        self._flash_stop = QTimer(self)
        self._flash_stop.setSingleShot(True)
        self._flash_stop.timeout.connect(self.stop_flash)
        self.setMinimumSize(320, 240)
        self._aspect = FRAME_WIDTH / FRAME_HEIGHT
        if lock_aspect:
            sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            sp.setHeightForWidth(True)
            self.setSizePolicy(sp)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

    def hasHeightForWidth(self):
        return self._lock_aspect

    def heightForWidth(self, w):
        return int(round(w / self._aspect)) if self._aspect > 0 else w

    def set_image(self, img):
        self._img = img
        if (self._lock_aspect and img is not None and not img.isNull()
                and img.height() > 0):
            a = img.width() / img.height()
            if abs(a - self._aspect) > 1e-3:
                self._aspect = a
                self.updateGeometry()
        self.update()
    def set_placeholder(self, t): self._placeholder = t; self._img = None; self.update()
    def clear(self): self._img = None; self.update()
    def set_estop(self, on): self._estop = bool(on); self.update()

    def _toggle_flash(self): self._flash_on = not self._flash_on; self.update()

    def start_flash(self, duration_ms=5000):
        """Blink a red DANGER overlay for `duration_ms`, then stop on its own."""
        self._flashing = True; self._flash_on = True
        self._flash_blink.start()
        self._flash_stop.start(int(duration_ms))
        self.update()

    def stop_flash(self):
        self._flash_blink.stop(); self._flash_stop.stop()
        self._flashing = False; self._flash_on = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.rightClicked.emit(); e.accept()
        else:
            super().mousePressEvent(e)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        path = QPainterPath(); path.addRoundedRect(QRectF(r), 10, 10)
        p.setClipPath(path)
        p.fillRect(r, qc("#06080f"))
        if self._img is not None and not self._img.isNull():
            s = self._img.scaled(self.size(), Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
            p.drawImage((self.width() - s.width()) // 2,
                        (self.height() - s.height()) // 2, s)
        else:
            p.setPen(qc(C.FAINT))
            f = QFont("Segoe UI", 11); f.setLetterSpacing(QFont.AbsoluteSpacing, 2)
            p.setFont(f); p.drawText(r, Qt.AlignCenter, self._placeholder)
        p.setClipping(False)
        if self._estop:
            p.setPen(QPen(qc(C.RED), 3)); p.setBrush(Qt.NoBrush); p.drawPath(path)
            p.setClipPath(path)
            band = QRectF(r.left(), r.top(), r.width(), 34)
            p.fillRect(band, qc(C.RED, 220))
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter,
                       "EMERGENCY STOP  -  right-click to release")
            p.setClipping(False)
        else:
            p.setPen(QPen(qc(C.BORDER), 1)); p.setBrush(Qt.NoBrush); p.drawPath(path)
        if self._flashing and self._flash_on:
            p.setClipPath(path)
            p.fillRect(r, qc(C.RED, 95))
            p.setPen(QPen(qc(C.RED), 6)); p.setBrush(Qt.NoBrush); p.drawPath(path)
            band = QRectF(r.left(), r.bottom() - 33, r.width(), 33)
            p.fillRect(band, qc(C.RED, 235))
            p.setPen(qc("#ffffff"))
            f = QFont("Segoe UI", 12, QFont.Black)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1)
            p.setFont(f)
            p.drawText(band, Qt.AlignCenter, "!  RED ZONE  -  DANGER  !")
            p.setClipping(False)


class FaceUnlockWorker(QThread):
    preview = Signal(QImage)
    status = Signal(str)

    def __init__(self, cam_index, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.identity = None

    def run(self):
        cap = None
        try:
            self.status.emit("Loading face recognition...")
            face = FaceRecognizer(FACE_DB_PATH, FACE_THRESHOLD, "cpu")
            if not face.available:
                self.status.emit(f"Face verification unavailable: {face.error}")
                return
            if not len(face.names):
                self.status.emit("Face database is empty. Enrol a face first.")
                return
            if self.isInterruptionRequested():
                return
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.cam_index)
            if not cap.isOpened():
                self.status.emit("Cannot open camera for face verification.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            deadline = time.monotonic() + FACE_UNLOCK_TIMEOUT_S
            streak = VerificationStreak()
            while not self.isInterruptionRequested():
                if time.monotonic() >= deadline:
                    self.status.emit("Verification timed out. Close and try again.")
                    return
                ok, frame = cap.read()
                if not ok:
                    self.status.emit("Camera unavailable. START remains locked.")
                    return
                if FLIP_VIEW:
                    frame = cv2.flip(frame, 1)
                faces = face.identify(frame)
                accepted = streak.update([name for name, _score, _box in faces])
                if streak.candidate:
                    self.status.emit(f"Verifying {streak.candidate}... {streak.count}/{FACE_UNLOCK_FRAMES}")
                else:
                    self.status.emit("Show one enrolled face to unlock START.")
                _draw_faces(frame, faces)
                self.preview.emit(bgr_to_qimage(frame))
                if accepted:
                    if not self.isInterruptionRequested():
                        self.identity = streak.candidate
                    return
        except Exception as exc:
            self.status.emit(f"Face verification failed: {exc}")
        finally:
            if cap is not None:
                cap.release()


class FaceUnlockDialog(QDialog):
    def __init__(self, cam_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Verify your face")
        self.resize(660, 560)
        self.setStyleSheet(f"QDialog {{ background: {C.BG}; }}")
        self.identity = None
        self._cancelled = False
        layout = QVBoxLayout(self)
        self.view = VideoView("LOADING FACE RECOGNITION")
        self.message = QLabel("Loading face recognition...")
        self.message.setWordWrap(True)
        note = QLabel("Any face enrolled in faces_db.npz can unlock this system.\n"
                      "Camera matching does not verify liveness; a photo may unlock it.")
        note.setWordWrap(True)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.message)
        layout.addWidget(note)
        layout.addWidget(self.cancel)
        self.worker = FaceUnlockWorker(cam_index, self)
        self.worker.preview.connect(self.view.set_image)
        self.worker.status.connect(self.message.setText)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        if self._cancelled:
            super().reject()
        elif self.worker.identity is not None:
            self.identity = self.worker.identity
            self.accept()
        else:
            self.cancel.setText("Close")

    def reject(self):
        self._cancelled = True
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.message.setText("Closing camera; please wait...")
        else:
            super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()


def verify_face(cam_index=CAM_INDEX):
    """Run the modal face dialog and return True only when verification succeeds.

    Recognition runs in a worker thread; the dialog remains responsive while
    loading models. Cancellation waits for that worker to release the camera.
    """
    global face_verified, verified_name
    face_verified, verified_name = False, None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = FaceUnlockDialog(cam_index)
    dialog.setStyleSheet(build_stylesheet() + f"\nQDialog {{ background: {C.BG}; }}")
    dialog.setFont(QFont("Segoe UI", 10))
    try:
        if dialog.exec() == QDialog.Accepted and dialog.identity:
            face_verified = True
            verified_name = dialog.identity
            print(f"Access granted: {verified_name}")
        return face_verified
    finally:
        dialog.deleteLater()
        app.processEvents()



def selftest():
    """Exercise the decision rule without models, camera, or hardware."""
    streak = VerificationStreak()
    assert not streak.update(["Alice"])
    assert not streak.update(["Alice"])
    assert streak.update(["Alice"])
    for interruption in ([], ["Unknown"], ["Alice", "Bob"]):
        assert not streak.update(interruption)
        assert streak.count == 0
        assert not streak.update(["Alice"])
        assert not streak.update(["Alice"])
    assert not streak.update(["Bob"])
    assert streak.count == 1
    assert not streak.update(["Bob"])
    assert streak.update(["Bob"])
    print("[selftest] PASS: consistent identity, unknown, absent, multiple and changed faces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAM_INDEX)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    verify_face(args.camera)
    print(f"face_verified = {face_verified}")
    # Future system integration:
    # if face_verified:
    #     enable_start_button()


if __name__ == "__main__":
    main()
