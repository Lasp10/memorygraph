import uuid
import json
import logging
import numpy as np
from pathlib import Path

from PIL import Image

from db.database import get_connection

logger = logging.getLogger(__name__)

THUMBNAILS_DIR = Path(__file__).parent.parent / "thumbnails"
MODELS_DIR = Path(__file__).parent.parent / "models"

_detector = None
_embedder = None

DETECT_CONFIDENCE = 0.5


def _load_models():
    global _detector, _embedder
    import cv2

    if _detector is None:
        _detector = cv2.dnn.readNetFromCaffe(
            str(MODELS_DIR / "deploy.prototxt"),
            str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"),
        )
    if _embedder is None:
        _embedder = cv2.dnn.readNetFromTorch(str(MODELS_DIR / "openface_nn4.small2.v1.t7"))
    return _detector, _embedder


def _detect_faces(cv2, bgr: np.ndarray, detector) -> list[tuple[int, int, int, int]]:
    """Returns list of (top, right, bottom, left) boxes in original image coordinates."""
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < DETECT_CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        left, top, right, bottom = (int(v) for v in box.astype(int))
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        if right - left < 20 or bottom - top < 20:
            continue
        boxes.append((top, right, bottom, left))
    return boxes


def _embed_face(cv2, bgr: np.ndarray, box: tuple[int, int, int, int], embedder) -> np.ndarray:
    top, right, bottom, left = box
    face = bgr[top:bottom, left:right]
    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(face_blob)
    vec = embedder.forward().flatten()
    return vec.astype(np.float64)


def detect_and_embed(limit: int = 0) -> dict:
    import cv2

    detector, embedder = _load_models()

    with get_connection() as conn:
        query = "SELECT id, filepath FROM media WHERE processed = 0"
        if limit:
            query += f" LIMIT {limit}"
        rows = list(conn.execute(query))

    processed = 0
    faces_found = 0

    for row in rows:
        media_id = row["id"]
        filepath = row["filepath"]
        try:
            with Image.open(filepath) as img:
                rgb = np.array(img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            boxes = _detect_faces(cv2, bgr, detector)

            with get_connection() as conn:
                for box in boxes:
                    enc = _embed_face(cv2, bgr, box, embedder)
                    top, right, bottom, left = box
                    face_id = str(uuid.uuid4())
                    bbox = {"top": top, "right": right, "bottom": bottom, "left": left}
                    conn.execute(
                        """INSERT INTO faces (id, media_id, bbox_json, embedding_blob, person_id, confirmed)
                           VALUES (?, ?, ?, ?, NULL, 0)""",
                        (face_id, media_id, json.dumps(bbox), enc.tobytes()),
                    )
                    faces_found += 1

                conn.execute("UPDATE media SET processed = 1 WHERE id = ?", (media_id,))

            processed += 1
        except Exception as e:
            logger.error(f"Face detection failed for {filepath}: {e}")
            with get_connection() as conn:
                conn.execute("UPDATE media SET processed = 1 WHERE id = ?", (media_id,))

    return {"processed": processed, "faces_found": faces_found}


def cluster_faces() -> dict:
    from sklearn.cluster import DBSCAN

    with get_connection() as conn:
        face_rows = list(conn.execute(
            "SELECT id, media_id, embedding_blob FROM faces WHERE person_id IS NULL"
        ))

    if not face_rows:
        return {"clusters": 0, "noise": 0}

    face_ids = [r["id"] for r in face_rows]
    embeddings = np.array([
        np.frombuffer(r["embedding_blob"], dtype=np.float64)
        for r in face_rows
    ])

    db = DBSCAN(eps=0.7, min_samples=2, metric="euclidean").fit(embeddings)
    labels = db.labels_

    # Map cluster label → person id
    cluster_to_person: dict[int, str] = {}
    noise_count = 0
    person_counter = 1

    with get_connection() as conn:
        # Clear existing unconfirmed people assignments before re-clustering
        conn.execute("UPDATE faces SET person_id = NULL WHERE confirmed = 0")
        conn.execute("DELETE FROM people WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE confirmed = 1 AND person_id IS NOT NULL)")

        for face_id, label in zip(face_ids, labels):
            if label == -1:
                # Noise: singleton cluster
                person_id = str(uuid.uuid4())
                person_name = f"Person {person_counter}"
                person_counter += 1
                noise_count += 1
                conn.execute(
                    "INSERT OR IGNORE INTO people (id, name, confirmed_name, representative_face_id) VALUES (?, ?, NULL, ?)",
                    (person_id, person_name, face_id),
                )
                conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
            else:
                if label not in cluster_to_person:
                    person_id = str(uuid.uuid4())
                    person_name = f"Person {person_counter}"
                    person_counter += 1
                    cluster_to_person[label] = person_id
                    conn.execute(
                        "INSERT OR IGNORE INTO people (id, name, confirmed_name, representative_face_id) VALUES (?, ?, NULL, ?)",
                        (person_id, person_name, face_id),
                    )
                conn.execute(
                    "UPDATE faces SET person_id = ? WHERE id = ?",
                    (cluster_to_person[label], face_id),
                )

    return {"clusters": len(cluster_to_person), "noise": noise_count}


def assign_unconfirmed_to_confirmed(threshold: float = 0.75) -> int:
    """After user confirms some names, assign remaining unconfirmed faces to nearest confirmed person."""
    with get_connection() as conn:
        confirmed_people = list(conn.execute(
            "SELECT p.id FROM people p WHERE p.confirmed_name IS NOT NULL"
        ))
        if not confirmed_people:
            return 0

        confirmed_ids = [r["id"] for r in confirmed_people]
        confirmed_faces: list[tuple[str, np.ndarray]] = []
        for pid in confirmed_ids:
            rows = list(conn.execute(
                "SELECT embedding_blob FROM faces WHERE person_id = ? AND confirmed = 1",
                (pid,),
            ))
            if rows:
                embeddings = np.array([
                    np.frombuffer(r["embedding_blob"], dtype=np.float64) for r in rows
                ])
                centroid = embeddings.mean(axis=0)
                confirmed_faces.append((pid, centroid))

        unconfirmed = list(conn.execute(
            "SELECT id, embedding_blob FROM faces WHERE confirmed = 0"
        ))

    reassigned = 0
    updates = []
    for face in unconfirmed:
        enc = np.frombuffer(face["embedding_blob"], dtype=np.float64)
        best_pid, best_dist = None, float("inf")
        for pid, centroid in confirmed_faces:
            dist = float(np.linalg.norm(enc - centroid))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        if best_pid and best_dist < threshold:
            updates.append((best_pid, face["id"]))
            reassigned += 1

    if updates:
        with get_connection() as conn:
            conn.executemany("UPDATE faces SET person_id = ? WHERE id = ?", updates)

    return reassigned


def get_people_with_faces() -> list[dict]:
    with get_connection() as conn:
        people = list(conn.execute("SELECT * FROM people"))
        result = []
        for person in people:
            face_rows = list(conn.execute(
                """SELECT f.id, f.media_id, m.thumbnail_path, f.bbox_json
                   FROM faces f JOIN media m ON f.media_id = m.id
                   WHERE f.person_id = ? LIMIT 20""",
                (person["id"],),
            ))
            photo_count = conn.execute(
                "SELECT COUNT(DISTINCT media_id) FROM faces WHERE person_id = ?",
                (person["id"],),
            ).fetchone()[0]

            result.append({
                "id": person["id"],
                "name": person["confirmed_name"] or person["name"],
                "confirmed_name": person["confirmed_name"],
                "photo_count": photo_count,
                "representative_face_id": person["representative_face_id"],
                "faces": [
                    {
                        "id": f["id"],
                        "media_id": f["media_id"],
                        "thumbnail_path": f["thumbnail_path"],
                        "bbox": json.loads(f["bbox_json"]) if f["bbox_json"] else None,
                    }
                    for f in face_rows
                ],
            })
    return result


def confirm_person_name(person_id: str, name: str) -> bool:
    with get_connection() as conn:
        conn.execute(
            "UPDATE people SET name = ?, confirmed_name = ? WHERE id = ?",
            (name, name, person_id),
        )
        conn.execute(
            "UPDATE faces SET confirmed = 1 WHERE person_id = ?",
            (person_id,),
        )
    assign_unconfirmed_to_confirmed()
    return True
