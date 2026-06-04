import json
import uuid
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import networkx as nx

from db.database import get_connection

logger = logging.getLogger(__name__)

_graph: nx.DiGraph | None = None

DECADE_MAP = {
    str(d): f"{d}s"
    for d in range(1950, 2030, 10)
}

PLACE_LABELS = {
    "beach", "mountains", "park", "backyard", "living room", "kitchen",
    "school", "church", "restaurant", "camping",
}

SEASON_FROM_MONTH = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
    11: "Autumn", 12: "Winter",
}


def _get_decade(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        year = int(date_str[:4])
        decade = (year // 10) * 10
        return f"{decade}s"
    except Exception:
        return None


def _get_season_year(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str[:10])
        season = SEASON_FROM_MONTH.get(dt.month, "")
        return f"{season} {dt.year}" if season else str(dt.year)
    except Exception:
        return None


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:10])
    except Exception:
        return None


def build_graph() -> nx.DiGraph:
    global _graph
    G = nx.DiGraph()

    with get_connection() as conn:
        media_rows = list(conn.execute("SELECT * FROM media"))
        face_rows = list(conn.execute("SELECT * FROM faces WHERE person_id IS NOT NULL"))
        people_rows = list(conn.execute("SELECT * FROM people"))
        event_rows = list(conn.execute("SELECT * FROM events"))
        event_media_rows = list(conn.execute("SELECT * FROM event_media"))

    # Add Person nodes
    for person in people_rows:
        G.add_node(
            person["id"],
            node_type="Person",
            name=person["confirmed_name"] or person["name"],
        )

    # Add Media nodes
    media_by_id = {}
    for m in media_rows:
        tags = json.loads(m["tags"]) if m["tags"] else []
        G.add_node(
            m["id"],
            node_type="Media",
            filepath=m["filepath"],
            thumbnail_path=m["thumbnail_path"],
            date_taken=m["date_taken"],
            tags=tags,
        )
        media_by_id[m["id"]] = dict(m)

    # APPEARS_IN edges
    person_media: dict[str, list[str]] = defaultdict(list)
    for face in face_rows:
        if face["person_id"] and face["media_id"]:
            G.add_edge(face["person_id"], face["media_id"], edge_type="APPEARS_IN")
            person_media[face["person_id"]].append(face["media_id"])

    # CO-APPEARS_WITH edges (two people in same photo)
    media_people: dict[str, list[str]] = defaultdict(list)
    for face in face_rows:
        if face["person_id"]:
            media_people[face["media_id"]].append(face["person_id"])

    seen_pairs: set[tuple] = set()
    for media_id, pids in media_people.items():
        for i, p1 in enumerate(pids):
            for p2 in pids[i + 1:]:
                pair = tuple(sorted([p1, p2]))
                if pair not in seen_pairs:
                    G.add_edge(p1, p2, edge_type="CO-APPEARS_WITH")
                    G.add_edge(p2, p1, edge_type="CO-APPEARS_WITH")
                    seen_pairs.add(pair)

    # Auto-cluster events if none exist
    existing_event_ids = {r["event_id"] for r in event_media_rows}
    if not existing_event_ids:
        _auto_cluster_events(G, media_by_id, media_people)
    else:
        # Load existing events and edges
        for ev in event_rows:
            G.add_node(ev["id"], node_type="Event", name=ev["name"], place=ev["place"])
        for em in event_media_rows:
            G.add_edge(em["media_id"], em["event_id"], edge_type="PART_OF")

    # Refresh event nodes in case we just created them
    with get_connection() as conn:
        for ev in conn.execute("SELECT * FROM events"):
            if ev["id"] not in G:
                G.add_node(ev["id"], node_type="Event", name=ev["name"], place=ev["place"])
        for em in conn.execute("SELECT * FROM event_media"):
            G.add_edge(em["media_id"], em["event_id"], edge_type="PART_OF")
            # OCCURRED_AT / OCCURRED_IN edges
            ev_data = G.nodes.get(em["event_id"], {})
            place = ev_data.get("place")
            if place:
                place_node = f"place:{place}"
                G.add_node(place_node, node_type="Place", name=place)
                G.add_edge(em["event_id"], place_node, edge_type="OCCURRED_AT")

    _graph = G
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _auto_cluster_events(G: nx.DiGraph, media_by_id: dict, media_people: dict):
    """Group media into events by date proximity + shared people."""
    dated_media = [
        (mid, _parse_date(media_by_id[mid]["date_taken"]))
        for mid in media_by_id
        if media_by_id[mid]["date_taken"]
    ]
    dated_media.sort(key=lambda x: x[1] or datetime.min)

    clusters: list[list[str]] = []
    used = set()

    for i, (mid1, dt1) in enumerate(dated_media):
        if mid1 in used:
            continue
        cluster = [mid1]
        people1 = set(media_people.get(mid1, []))
        for j, (mid2, dt2) in enumerate(dated_media[i + 1:], i + 1):
            if mid2 in used:
                continue
            people2 = set(media_people.get(mid2, []))
            shared = people1 & people2
            if dt1 and dt2 and abs((dt2 - dt1).days) <= 3 and (shared or not people1):
                cluster.append(mid2)
                people1 |= people2
                used.add(mid2)
        used.add(mid1)
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Undated media with no date: group by shared people only
    undated = [mid for mid in media_by_id if not media_by_id[mid]["date_taken"] and mid not in used]
    people_to_media: dict[frozenset, list[str]] = defaultdict(list)
    for mid in undated:
        pset = frozenset(media_people.get(mid, []))
        if pset:
            people_to_media[pset].append(mid)
    for pset, mids in people_to_media.items():
        if len(mids) >= 2:
            clusters.append(mids)

    event_counter = 1
    with get_connection() as conn:
        for cluster in clusters:
            event_id = str(uuid.uuid4())
            # Auto-name
            people_in_cluster: set[str] = set()
            for mid in cluster:
                people_in_cluster.update(media_people.get(mid, []))

            person_names = []
            for pid in list(people_in_cluster)[:2]:
                row = conn.execute("SELECT confirmed_name, name FROM people WHERE id = ?", (pid,)).fetchone()
                if row:
                    person_names.append(row["confirmed_name"] or row["name"])

            tags_in_cluster = []
            for mid in cluster:
                tags = json.loads(media_by_id[mid].get("tags") or "[]")
                tags_in_cluster.extend(tags)

            place = next((t for t in tags_in_cluster if t in PLACE_LABELS), None)

            dates = [_parse_date(media_by_id[mid]["date_taken"]) for mid in cluster]
            dates = [d for d in dates if d]
            date_start = min(dates).date().isoformat() if dates else None
            date_end = max(dates).date().isoformat() if dates else None

            season_year = _get_season_year(date_start) if date_start else None

            if person_names and season_year:
                name = f"{' and '.join(person_names)}, {season_year}"
            elif person_names:
                name = f"{' and '.join(person_names)}"
            elif season_year:
                name = f"Event, {season_year}"
            else:
                name = f"Event {event_counter}"
            event_counter += 1

            conn.execute(
                "INSERT OR IGNORE INTO events (id, name, date_start, date_end, place) VALUES (?, ?, ?, ?, ?)",
                (event_id, name, date_start, date_end, place),
            )
            for mid in cluster:
                conn.execute(
                    "INSERT OR IGNORE INTO event_media (event_id, media_id) VALUES (?, ?)",
                    (event_id, mid),
                )

            G.add_node(event_id, node_type="Event", name=name, place=place)
            for mid in cluster:
                G.add_edge(mid, event_id, edge_type="PART_OF")

            if place:
                place_node = f"place:{place}"
                G.add_node(place_node, node_type="Place", name=place)
                G.add_edge(event_id, place_node, edge_type="OCCURRED_AT")

            decade = _get_decade(date_start)
            if decade:
                G.add_node(f"period:{decade}", node_type="TimePeriod", name=decade)
                G.add_edge(event_id, f"period:{decade}", edge_type="OCCURRED_IN")


def get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def graph_to_dict() -> dict:
    G = get_graph()
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({"id": node_id, **data})
    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append({"source": src, "target": dst, **data})
    return {"nodes": nodes, "edges": edges}


def get_media_for_person(person_id: str) -> list[str]:
    G = get_graph()
    return [
        n for n in G.successors(person_id)
        if G.nodes[n].get("node_type") == "Media"
    ]


def get_media_for_tag(tag: str) -> list[str]:
    G = get_graph()
    return [
        n for n, data in G.nodes(data=True)
        if data.get("node_type") == "Media" and tag in data.get("tags", [])
    ]


def get_media_for_period(period: str) -> list[str]:
    G = get_graph()
    period_node = f"period:{period}"
    if period_node not in G:
        return []
    # Events in this period
    events = [src for src, dst, data in G.edges(data=True)
               if dst == period_node and data.get("edge_type") == "OCCURRED_IN"]
    media_ids = []
    for event_id in events:
        for src, dst, data in G.edges(data=True):
            if dst == event_id and data.get("edge_type") == "PART_OF":
                media_ids.append(src)
    return media_ids
