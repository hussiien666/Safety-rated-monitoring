from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
DB_FILE = BASE_DIR / "faces_db.npz"

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))


def get_largest_face(img):
    faces = app.get(img)

    if not faces:
        return None

    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )


names = []
embeddings = []

if not KNOWN_DIR.exists():
    print(f"Folder not found: {KNOWN_DIR}")
    exit()

for person_dir in KNOWN_DIR.iterdir():
    if not person_dir.is_dir():
        continue

    person_embeddings = []

    for img_path in person_dir.glob("*.*"):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        face = get_largest_face(img)

        if face is None:
            print(f"No face found in: {img_path}")
            continue

        person_embeddings.append(face.embedding)

    if person_embeddings:
        avg_embedding = np.mean(person_embeddings, axis=0)

        names.append(person_dir.name)
        embeddings.append(avg_embedding)

        print(f"Saved {person_dir.name}: {len(person_embeddings)} photos")

if not embeddings:
    print("No faces found. Database not created.")
else:
    np.savez_compressed(
        DB_FILE,
        names=np.array(names),
        embeddings=np.array(embeddings)
    )

    print(f"Database saved to: {DB_FILE}")