import uuid
import json
import logging
import threading
import numpy as np
from pathlib import Path

from PIL import Image

from db.database import get_connection

logger = logging.getLogger(__name__)

THUMBNAILS_DIR = Path(__file__).parent.parent / "thumbnails"
FACE_THUMBNAILS_DIR = THUMBNAILS_DIR / "faces"
MODELS_DIR = Path(__file__).parent.parent / "models"
FACE_THUMB_SIZE = (200, 200)
FACE_CROP_MARGIN = 0.25  # extra context around the tight detection box

_detector = None
_embedder = None
_model_lock = threading.Lock()

DETECT_CONFIDENCE = 0.5


def _load_models():
    global _detector, _embedder
    import cv2

    with _model_lock:
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


def _save_face_thumbnail(rgb: np.ndarray, box: tuple[int, int, int, int], dest: Path):
    """Crop a face (with a little context margin) out of the full image and save it
    as its own small thumbnail, so each person's card shows their face — not the
    whole shared photo, which looks like duplicates when several people appear
    in the same group photo."""
    top, right, bottom, left = box
    h, w = rgb.shape[:2]
    bw, bh = right - left, bottom - top
    mx, my = int(bw * FACE_CROP_MARGIN), int(bh * FACE_CROP_MARGIN)
    top2, bottom2 = max(0, top - my), min(h, bottom + my)
    left2, right2 = max(0, left - mx), min(w, right + mx)
    crop = rgb[top2:bottom2, left2:right2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).resize(FACE_THUMB_SIZE, Image.LANCZOS).save(dest, "JPEG", quality=85)


def backfill_face_thumbnails() -> int:
    """Generate face-crop thumbnails for faces detected before this feature existed."""
    with get_connection() as conn:
        rows = list(conn.execute(
            """SELECT f.id, f.bbox_json, m.filepath FROM faces f
               JOIN media m ON f.media_id = m.id
               WHERE f.face_thumbnail_path IS NULL"""
        ))

    updates = []
    for row in rows:
        try:
            bbox = json.loads(row["bbox_json"])
            box = (bbox["top"], bbox["right"], bbox["bottom"], bbox["left"])
            with Image.open(row["filepath"]) as img:
                rgb = np.array(img.convert("RGB"))
            dest = FACE_THUMBNAILS_DIR / f"{row['id']}.jpg"
            _save_face_thumbnail(rgb, box, dest)
            updates.append((str(dest.relative_to(THUMBNAILS_DIR.parent)), row["id"]))
        except Exception as e:
            logger.error(f"Face thumbnail backfill failed for {row['id']}: {e}")

    if updates:
        with get_connection() as conn:
            conn.executemany("UPDATE faces SET face_thumbnail_path = ? WHERE id = ?", updates)

    return len(updates)


def detect_and_embed(limit: int = 0) -> dict:
    import cv2

    detector, embedder = _load_models()

    with get_connection() as conn:
        query = "SELECT id, filepath FROM media WHERE processed = 0"
        params: tuple = ()
        if limit:
            query += " LIMIT ?"
            params = (limit,)
        rows = list(conn.execute(query, params))

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
                    face_thumb_dest = FACE_THUMBNAILS_DIR / f"{face_id}.jpg"
                    _save_face_thumbnail(rgb, box, face_thumb_dest)
                    face_thumb_rel = str(face_thumb_dest.relative_to(THUMBNAILS_DIR.parent))
                    conn.execute(
                        """INSERT INTO faces (id, media_id, bbox_json, embedding_blob, person_id, confirmed, face_thumbnail_path)
                           VALUES (?, ?, ?, ?, NULL, 0, ?)""",
                        (face_id, media_id, json.dumps(bbox), enc.tobytes(), face_thumb_rel),
                    )
                    faces_found += 1

                conn.execute("UPDATE media SET processed = 1 WHERE id = ?", (media_id,))

            processed += 1
        except Exception as e:
            logger.error(f"Face detection failed for {filepath}: {e}")
            with get_connection() as conn:
                conn.execute(
                    "UPDATE media SET processed = 1, processing_error = ? WHERE id = ?",
                    (str(e), media_id),
                )

    return {"processed": processed, "faces_found": faces_found}


def cluster_faces() -> dict:
    from sklearn.cluster import DBSCAN

    with get_connection() as conn:
        # confirmed = 0 covers both never-clustered faces (person_id IS NULL) and
        # previously-clustered-but-unconfirmed faces, matching the reset UPDATE below.
        face_rows = list(conn.execute(
            "SELECT id, media_id, embedding_blob FROM faces WHERE confirmed = 0"
        ))

    if not face_rows:
        return {"clusters": 0, "noise": 0}

    face_ids = [r["id"] for r in face_rows]
    embeddings = np.array([
        np.frombuffer(r["embedding_blob"], dtype=np.float64)
        for r in face_rows
    ])

    # eps=0.7 was too loose for this embedding space: a handful of borderline
    # pairs bridged transitively and chained hundreds of distinct people into a
    # single cluster. eps=0.45 keeps cluster sizes sane (empirically verified
    # against this library — cluster sizes stay <=14 up to this point, then
    # runaway chaining kicks in at eps=0.5+).
    db = DBSCAN(eps=0.45, min_samples=2, metric="euclidean").fit(embeddings)
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


def assign_unconfirmed_to_confirmed(threshold: float = 0.45, margin: float = 0.1, min_samples: int = 2) -> int:
    """After user confirms some names, assign remaining unconfirmed faces to nearest confirmed person.

    Requires a centroid built from at least `min_samples` confirmed faces (a single
    face is too noisy to trust), and requires the best match to beat the second-best
    candidate by `margin` — otherwise an ambiguous face is left unassigned rather than
    guessed, which is what let faces jump onto the wrong newly-named person.
    """
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
            if len(rows) >= min_samples:
                embeddings = np.array([
                    np.frombuffer(r["embedding_blob"], dtype=np.float64) for r in rows
                ])
                centroid = embeddings.mean(axis=0)
                confirmed_faces.append((pid, centroid))

        # Only consider true orphans and singleton (noise) clusters — never pull a face
        # out of an established, distinct unconfirmed person cluster.
        unconfirmed = list(conn.execute(
            """SELECT id, embedding_blob FROM faces
               WHERE confirmed = 0
                 AND (person_id IS NULL
                      OR person_id IN (
                          SELECT person_id FROM faces
                          WHERE confirmed = 0 AND person_id IS NOT NULL
                          GROUP BY person_id HAVING COUNT(*) = 1
                      ))"""
        ))

    reassigned = 0
    updates = []
    for face in unconfirmed:
        enc = np.frombuffer(face["embedding_blob"], dtype=np.float64)
        dists = sorted(
            (float(np.linalg.norm(enc - centroid)), pid) for pid, centroid in confirmed_faces
        )
        if not dists:
            continue
        best_dist, best_pid = dists[0]
        second_dist = dists[1][0] if len(dists) > 1 else float("inf")
        if best_dist < threshold and (second_dist - best_dist) >= margin:
            updates.append((best_pid, face["id"]))
            reassigned += 1

    if updates:
        with get_connection() as conn:
            # Mark as confirmed so a later re-clustering pass doesn't reset person_id
            # to NULL and undo the assignment.
            conn.executemany(
                "UPDATE faces SET person_id = ?, confirmed = 1 WHERE id = ?", updates
            )

    return reassigned


def get_people_with_faces() -> list[dict]:
    with get_connection() as conn:
        people = list(conn.execute("SELECT * FROM people WHERE hidden = 0 OR hidden IS NULL"))
        result = []
        for person in people:
            # Excludes faces from media flagged hidden_from_people (e.g. the original
            # training library), so those photos never surface in the People tab.
            face_rows = list(conn.execute(
                """SELECT f.id, f.media_id, m.thumbnail_path, f.bbox_json, f.face_thumbnail_path
                   FROM faces f JOIN media m ON f.media_id = m.id
                   WHERE f.person_id = ? AND (m.hidden_from_people = 0 OR m.hidden_from_people IS NULL)
                   LIMIT 20""",
                (person["id"],),
            ))
            photo_count = conn.execute(
                """SELECT COUNT(DISTINCT f.media_id) FROM faces f JOIN media m ON f.media_id = m.id
                   WHERE f.person_id = ? AND (m.hidden_from_people = 0 OR m.hidden_from_people IS NULL)""",
                (person["id"],),
            ).fetchone()[0]

            if photo_count == 0:
                # Every face for this person comes from hidden media — nothing to show.
                continue

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
                        # Prefer the cropped face thumbnail so cards for different
                        # people in the same group photo don't look identical.
                        "thumbnail_path": f["face_thumbnail_path"] or f["thumbnail_path"],
                        "full_photo_thumbnail_path": f["thumbnail_path"],
                        "bbox": json.loads(f["bbox_json"]) if f["bbox_json"] else None,
                    }
                    for f in face_rows
                ],
            })
    return result


def set_person_hidden(person_id: str, hidden: bool) -> bool:
    with get_connection() as conn:
        conn.execute(
            "UPDATE people SET hidden = ? WHERE id = ?",
            (1 if hidden else 0, person_id),
        )
    return True


def confirm_person_name(person_id: str, name: str) -> bool:
    name = name.strip()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM people WHERE LOWER(confirmed_name) = LOWER(?) AND id != ?",
            (name, person_id),
        ).fetchone()
        if existing:
            # A person with this name is already confirmed — merge into it instead
            # of creating a second, independently-growing "person" with the same name.
            target_id = existing["id"]
            conn.execute(
                "UPDATE faces SET person_id = ?, confirmed = 1 WHERE person_id = ?",
                (target_id, person_id),
            )
            conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        else:
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
