import json
import logging
import difflib
from collections import defaultdict

from db.database import get_connection
from pipeline.graph import get_graph, get_media_for_person, get_media_for_tag, get_media_for_period
from pipeline.scene import get_chroma_collection, encode_query

logger = logging.getLogger(__name__)

DECADE_ALIASES = {
    "60s": "1960s", "70s": "1970s", "80s": "1980s", "90s": "1990s",
    "2000s": "2000s", "2010s": "2010s",
    "sixties": "1960s", "seventies": "1970s", "eighties": "1980s",
    "nineties": "1990s",
}

SCENE_KEYWORDS = [
    "birthday party", "christmas", "thanksgiving", "wedding", "graduation",
    "beach", "mountains", "park", "backyard", "living room", "kitchen",
    "school", "church", "restaurant", "vacation", "camping",
    "baby", "child", "teenager", "adult", "elderly",
    "snow", "summer", "autumn", "spring",
    "sports", "swimming", "playing", "cooking",
    "group photo", "portrait", "candid", "formal", "casual",
]


def _get_all_people() -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT id, name, confirmed_name FROM people")]


def _get_media_details(media_ids: list[str]) -> list[dict]:
    if not media_ids:
        return []
    placeholders = ",".join("?" * len(media_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id, filepath, thumbnail_path, date_taken, tags FROM media WHERE id IN ({placeholders})",
            media_ids,
        ).fetchall()
    result = []
    for row in rows:
        tags = json.loads(row["tags"]) if row["tags"] else []
        # Find people in this media
        G = get_graph()
        people_in_media = []
        for pred in G.predecessors(row["id"]):
            node_data = G.nodes.get(pred, {})
            if node_data.get("node_type") == "Person":
                people_in_media.append(node_data.get("name", pred))

        result.append({
            "id": row["id"],
            "filepath": row["filepath"],
            "thumbnail_path": row["thumbnail_path"],
            "date_taken": row["date_taken"],
            "tags": tags,
            "people": people_in_media,
        })
    return result


def _fuzzy_match_person(query: str, people: list[dict]) -> list[str]:
    matched_ids = []
    query_lower = query.lower()
    names = [(p["id"], p["confirmed_name"] or p["name"]) for p in people]
    for pid, name in names:
        if name.lower() in query_lower or query_lower in name.lower():
            matched_ids.append(pid)
            continue
        # Fuzzy match word by word
        words = query_lower.split()
        name_words = name.lower().split()
        for word in words:
            matches = difflib.get_close_matches(word, name_words, n=1, cutoff=0.8)
            if matches:
                matched_ids.append(pid)
                break
    return matched_ids


def _extract_period(query: str) -> list[str]:
    query_lower = query.lower()
    periods = []
    for alias, canonical in DECADE_ALIASES.items():
        if alias in query_lower:
            periods.append(canonical)
    # Also check for explicit "1980s" style references
    import re
    for match in re.findall(r"\b(19[5-9]0s|20[012]0s)\b", query_lower):
        periods.append(match)
    return list(set(periods))


def _extract_scene_keywords(query: str) -> list[str]:
    query_lower = query.lower()
    found = []
    for kw in SCENE_KEYWORDS:
        if kw in query_lower:
            found.append(kw)
    return found


def search(query: str, top_k: int = 20) -> list[dict]:
    graph_media_ids: dict[str, int] = defaultdict(int)  # media_id → score boost

    people = _get_all_people()

    # Graph traversal: person matches
    matched_person_ids = _fuzzy_match_person(query, people)
    for pid in matched_person_ids:
        for mid in get_media_for_person(pid):
            graph_media_ids[mid] += 3

    # Graph traversal: scene/tag matches
    for tag in _extract_scene_keywords(query):
        for mid in get_media_for_tag(tag):
            graph_media_ids[mid] += 2

    # Graph traversal: time period matches
    for period in _extract_period(query):
        for mid in get_media_for_period(period):
            graph_media_ids[mid] += 2

    # Vector search via ChromaDB
    vector_scores: dict[str, float] = {}
    try:
        collection = get_chroma_collection()
        query_embedding = encode_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(collection.count(), 1)),
            include=["distances", "metadatas"],
        )
        if results["ids"] and results["ids"][0]:
            for mid, dist in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance → similarity
                similarity = 1.0 - dist
                vector_scores[mid] = similarity
    except Exception as e:
        logger.error(f"Vector search failed: {e}")

    # Union and rank
    all_ids = set(graph_media_ids.keys()) | set(vector_scores.keys())

    def score(mid: str) -> float:
        graph_boost = graph_media_ids.get(mid, 0)
        vector_sim = vector_scores.get(mid, 0.0)
        # Graph hits first, then vector similarity as tiebreaker
        return graph_boost * 10 + vector_sim

    ranked = sorted(all_ids, key=score, reverse=True)[:top_k]
    return _get_media_details(ranked)
