"""
build_database.py
==================
Builds the known-faces database used by the face-recognition part of the system.

What it does, step by step:
  1. Loads the InsightFace 'buffalo_l' model (face detection + recognition).
  2. Walks through every sub-folder inside 'known_faces/'. Each sub-folder name
     is treated as a person's name and must contain one or more photos of them.
  3. For every photo it detects the largest face and extracts its 512-d
     "embedding" (a numeric fingerprint of the face).
  4. It averages all embeddings belonging to the same person into one vector.
  5. Finally it saves all names + averaged embeddings into 'faces_db.npz'
     (a compressed NumPy archive) for fast lookup at runtime.
"""

from pathlib import Path                       # Path: object-oriented, OS-independent filesystem paths.
import cv2                                      # OpenCV: used here only to read image files from disk.
import numpy as np                              # NumPy: array math (averaging embeddings, saving the DB).
from insightface.app import FaceAnalysis        # FaceAnalysis: high-level wrapper that runs the face models.

BASE_DIR = Path(__file__).resolve().parent      # Absolute path of the folder this script lives in.
KNOWN_DIR = BASE_DIR / "known_faces"            # Folder that holds one sub-folder of photos per person.
DB_FILE = BASE_DIR / "faces_db.npz"             # Output file where the finished database will be written.

app = FaceAnalysis(                             # Create the face-analysis engine...
    name="buffalo_l",                           # ...using the pre-trained 'buffalo_l' model pack.
    allowed_modules=["detection", "recognition"],  # Only load detection + recognition (skip age/gender → faster).
    providers=["CPUExecutionProvider"]          # Run inference on the CPU (no GPU required).
)

app.prepare(ctx_id=-1, det_size=(640, 640))     # Initialise the model: ctx_id=-1 → CPU; det_size → detection input size.


def get_largest_face(img):                      # Helper: return the biggest face found in an image (or None).
    faces = app.get(img)                        # Run detection + recognition; returns a list of Face objects.

    if not faces:                               # If the list is empty (no face detected)...
        return None                             # ...signal "nothing found" to the caller.

    return max(                                 # Otherwise pick the single largest face...
        faces,                                  # ...from all detected faces...
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])  # ...by bounding-box area: width × height.
    )                                           # bbox = [x1, y1, x2, y2]; (x2-x1)=width, (y2-y1)=height.


names = []                                      # Will collect each person's folder name (their label).
embeddings = []                                 # Will collect each person's averaged face embedding.

if not KNOWN_DIR.exists():                       # Guard: make sure the input folder actually exists.
    print(f"Folder not found: {KNOWN_DIR}")     # Tell the user the expected folder is missing.
    exit()                                      # Stop the program — there is nothing to process.

for person_dir in KNOWN_DIR.iterdir():           # Loop over every entry inside 'known_faces/'.
    if not person_dir.is_dir():                  # Skip anything that is not a directory (e.g. stray files).
        continue                                 # Move on to the next entry.

    person_embeddings = []                       # Holds all face embeddings for THIS one person.

    for img_path in person_dir.glob("*.*"):      # Loop over every file (any extension) in the person's folder.
        img = cv2.imread(str(img_path))          # Load the image from disk into a NumPy/BGR array.

        if img is None:                          # cv2.imread returns None if the file is not a valid image.
            print(f"Cannot read image: {img_path}")  # Warn and...
            continue                             # ...skip this unreadable file.

        face = get_largest_face(img)             # Detect the most prominent face in the photo.

        if face is None:                         # If no face was detected in this photo...
            print(f"No face found in: {img_path}")  # ...warn the user and...
            continue                             # ...skip it.

        person_embeddings.append(face.embedding) # Store this photo's 512-dim face embedding.

    if person_embeddings:                        # If we collected at least one valid embedding for this person...
        avg_embedding = np.mean(person_embeddings, axis=0)  # ...average them element-wise into one robust vector.

        names.append(person_dir.name)            # Record the person's name (= the folder name).
        embeddings.append(avg_embedding)         # Record their averaged embedding alongside the name.

        print(f"Saved {person_dir.name}: {len(person_embeddings)} photos")  # Progress report.

if not embeddings:                               # After processing everyone: if nothing was collected...
    print("No faces found. Database not created.")  # ...inform the user no DB was produced.
else:                                            # Otherwise we have data to save:
    np.savez_compressed(                         # Save multiple arrays into one compressed .npz file.
        DB_FILE,                                 # Destination path.
        names=np.array(names),                   # Array of person names (the labels).
        embeddings=np.array(embeddings)          # Matching array of averaged embeddings (the fingerprints).
    )

    print(f"Database saved to: {DB_FILE}")       # Confirm success and where the file landed.
