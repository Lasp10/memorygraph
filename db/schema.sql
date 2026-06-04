CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    filepath TEXT NOT NULL,
    thumbnail_path TEXT,
    media_type TEXT CHECK(media_type IN ('image', 'video')),
    date_taken TEXT,
    date_ingested TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
    tags TEXT,
    ocr_text TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id TEXT PRIMARY KEY,
    media_id TEXT REFERENCES media(id),
    bbox_json TEXT,
    embedding_blob BLOB,
    person_id TEXT,
    confirmed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    confirmed_name TEXT,
    representative_face_id TEXT REFERENCES faces(id)
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    name TEXT,
    date_start TEXT,
    date_end TEXT,
    place TEXT
);

CREATE TABLE IF NOT EXISTS event_media (
    event_id TEXT REFERENCES events(id),
    media_id TEXT REFERENCES media(id),
    PRIMARY KEY (event_id, media_id)
);
