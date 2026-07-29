import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from db.database import get_connection

logger = logging.getLogger(__name__)

SCENE_LABELS = [
    "birthday party", "Christmas", "Thanksgiving", "wedding", "graduation",
    "beach", "mountains", "park", "backyard", "living room", "kitchen",
    "school", "church", "restaurant", "vacation", "camping",
    "baby", "child", "teenager", "adult", "elderly person",
    "snow", "summer", "autumn", "spring",
    "sports", "swimming", "playing", "cooking", "opening gifts",
    "group photo", "portrait", "candid", "formal", "casual",
    "1960s", "1970s", "1980s", "1990s", "2000s", "2010s",
]

_model = None
_chroma_client = None
_chroma_collection = None
_model_lock = threading.Lock()
_chroma_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("clip-ViT-B-32")
            logger.info("CLIP model loaded")
        return _model


def get_chroma_collection():
    global _chroma_client, _chroma_collection
    with _chroma_lock:
        if _chroma_collection is None:
            import chromadb
            db_dir = Path(__file__).parent.parent / "chroma_db"
            db_dir.mkdir(exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(db_dir))
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="media_embeddings",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection ready")
        return _chroma_collection


def tag_scenes(limit: int = 0) -> dict:
    model = get_model()
    collection = get_chroma_collection()

    with get_connection() as conn:
        query = "SELECT id, filepath, date_taken FROM media WHERE tags IS NULL"
        params: tuple = ()
        if limit:
            query += " LIMIT ?"
            params = (limit,)
        rows = list(conn.execute(query, params))

    if not rows:
        return {"tagged": 0}

    # Pre-encode all labels once
    label_embeddings = model.encode(SCENE_LABELS, convert_to_numpy=True, normalize_embeddings=True)

    tagged = 0
    batch_ids: list[str] = []
    batch_embeddings: list[list[float]] = []
    batch_metadatas: list[dict] = []
    tag_updates: list[tuple[str, str]] = []

    for row in rows:
        media_id = row["id"]
        filepath = row["filepath"]
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                img_embedding = model.encode(img_rgb, convert_to_numpy=True, normalize_embeddings=True)

            # Cosine similarity (both normalized so dot product works)
            scores = label_embeddings @ img_embedding
            top_indices = np.argsort(scores)[::-1]
            tags = [SCENE_LABELS[i] for i in top_indices if scores[i] > 0.22][:3]

            batch_ids.append(media_id)
            batch_embeddings.append(img_embedding.tolist())
            batch_metadatas.append({
                "media_id": media_id,
                "tags": ",".join(tags),
                "date_taken": row["date_taken"] or "",
            })
            tag_updates.append((json.dumps(tags), media_id))

            tagged += 1
            logger.info(f"Tagged {filepath}: {tags}")

        except Exception as e:
            logger.error(f"Scene tagging failed for {filepath}: {e}")
            tag_updates.append(("[]", media_id))

    if batch_ids:
        collection.upsert(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)

    if tag_updates:
        with get_connection() as conn:
            conn.executemany("UPDATE media SET tags = ? WHERE id = ?", tag_updates)

    return {"tagged": tagged}


def encode_query(query_text: str) -> list[float]:
    model = get_model()
    embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()
