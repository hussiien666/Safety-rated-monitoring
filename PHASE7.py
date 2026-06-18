from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "faces_db.npz"
THRESHOLD = 0.60

# Auto-select GPU only if CUDA is really available
available_providers = ort.get_available_providers()

if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0
else:
    providers = ["CPUExecutionProvider"]
    ctx_id = -1

print("Using providers:", providers)

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=providers
)

app.prepare(ctx_id=ctx_id, det_size=(640, 640))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_database():
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}\n"
            "Run build_database.py first."
        )

    data = np.load(DB_FILE, allow_pickle=False)

    names = data["names"]
    embeddings = data["embeddings"]

    return names, embeddings


def recognize_face(face_embedding, names, known_embeddings):
    best_name = "Unknown"
    best_score = -1

    for name, known_embedding in zip(names, known_embeddings):
        score = cosine_similarity(face_embedding, known_embedding)

        if score > best_score:
            best_score = score
            best_name = str(name)

    if best_score >= THRESHOLD:
        return best_name, best_score

    return "Unknown", best_score


names, known_embeddings = load_database()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from webcam.")
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        name, score = recognize_face(face.embedding, names, known_embeddings)

        label = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()