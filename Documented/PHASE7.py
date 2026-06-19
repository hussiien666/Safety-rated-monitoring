"""
PHASE7.py
=========
Live face-recognition demo built on the database created by build_database.py.

Flow:
  1. Pick the best ONNX Runtime provider (GPU if CUDA is present, else CPU).
  2. Load the InsightFace 'buffalo_l' model + the saved faces_db.npz.
  3. For every webcam frame, detect faces, compute each face's embedding, and
     compare it (cosine similarity) against every known person.
  4. Label the face with the closest match if the score clears THRESHOLD,
     otherwise "Unknown".
"""

from pathlib import Path                          # OS-independent filesystem paths.
import cv2                                         # OpenCV: webcam + drawing.
import numpy as np                                 # NumPy: vector math + loading the .npz database.
import onnxruntime as ort                          # ONNX Runtime: lets us query available compute providers.
from insightface.app import FaceAnalysis           # InsightFace face detection + recognition.

BASE_DIR = Path(__file__).resolve().parent         # Folder this script lives in.
DB_FILE = BASE_DIR / "faces_db.npz"                # Path to the prebuilt face database.
THRESHOLD = 0.60                                   # Min cosine similarity to accept a match (else "Unknown").

# Auto-select GPU only if CUDA is really available
available_providers = ort.get_available_providers()  # List of execution backends ONNX Runtime can use here.

if "CUDAExecutionProvider" in available_providers:  # If a CUDA GPU backend is present...
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # ...prefer GPU, fall back to CPU.
    ctx_id = 0                                     # ctx_id 0 → first GPU.
else:                                              # Otherwise...
    providers = ["CPUExecutionProvider"]           # ...CPU only.
    ctx_id = -1                                    # ctx_id -1 → CPU.

print("Using providers:", providers)               # Report the chosen backend(s).

app = FaceAnalysis(                                # Build the face-analysis engine...
    name="buffalo_l",                              # ...with the buffalo_l model pack...
    allowed_modules=["detection", "recognition"],  # ...only detection + recognition (faster).
    providers=providers                            # ...on the selected backend(s).
)

app.prepare(ctx_id=ctx_id, det_size=(640, 640))    # Initialise: choose device + detection input size.


def cosine_similarity(a, b):                       # Similarity of two embedding vectors (1=identical).
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # dot(a,b) / (|a|·|b|).


def load_database():                               # Load names + embeddings from the .npz file.
    if not DB_FILE.exists():                       # If the database hasn't been built yet...
        raise FileNotFoundError(                   # ...fail with a helpful message.
            f"Database not found: {DB_FILE}\n"
            "Run build_database.py first."
        )

    data = np.load(DB_FILE, allow_pickle=False)    # Load the archive (allow_pickle=False is safer).

    names = data["names"]                          # Array of person labels.
    embeddings = data["embeddings"]                # Matching array of averaged embeddings.

    return names, embeddings                       # Hand both arrays back.


def recognize_face(face_embedding, names, known_embeddings):  # Match one face against the database.
    best_name = "Unknown"                          # Best candidate name so far.
    best_score = -1                                # Best similarity so far (start below any real score).

    for name, known_embedding in zip(names, known_embeddings):  # Compare against every known person...
        score = cosine_similarity(face_embedding, known_embedding)  # ...via cosine similarity.

        if score > best_score:                     # Keep the highest-scoring candidate.
            best_score = score
            best_name = str(name)

    if best_score >= THRESHOLD:                    # If the best match is confident enough...
        return best_name, best_score              # ...return the name + score.

    return "Unknown", best_score                  # Otherwise it's an unknown face.


names, known_embeddings = load_database()          # Load the database once at startup.

cap = cv2.VideoCapture(0)                          # Open the default webcam.

if not cap.isOpened():                             # Abort if it failed.
    raise RuntimeError("Could not open webcam.")

while True:                                         # Main loop.
    ret, frame = cap.read()                        # Grab a frame.

    if not ret:                                    # Stop if the grab failed.
        print("Could not read frame from webcam.")
        break

    faces = app.get(frame)                         # Detect + embed every face in the frame.

    for face in faces:                             # For each detected face...
        x1, y1, x2, y2 = face.bbox.astype(int)     # Bounding-box corners as ints.

        name, score = recognize_face(face.embedding, names, known_embeddings)  # Identify it.

        label = f"{name} {score:.2f}"              # Build a "<name> <score>" label.

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green box around the face.

        cv2.putText(                               # Draw the label above the box.
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)          # Show the annotated frame.

    if cv2.waitKey(1) & 0xFF == ord("q"):          # 'q' → quit.
        break

cap.release()                                      # Release the webcam.
cv2.destroyAllWindows()                            # Close OpenCV windows.
