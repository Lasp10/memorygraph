import os
import uuid
import shutil
import hashlib
import zipfile
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ExifTags

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

from db.database import get_connection

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".heic", ".heif"}
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
THUMBNAILS_DIR = Path(__file__).parent.parent / "thumbnails"
THUMBNAIL_SIZE = (300, 300)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_exif_date(image: Image.Image) -> str | None:
    try:
        exif_data = image.getexif()
        if not exif_data:
            return None
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        date_tag = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        if date_tag and date_tag in exif_data:
            raw = exif_data[date_tag]
            # Format: "YYYY:MM:DD HH:MM:SS"
            return raw.replace(":", "-", 2).replace(" ", "T")
    except Exception:
        pass
    return None


def _make_thumbnail(src: Path, dest: Path):
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        img.save(dest, "JPEG", quality=85)


def ingest_folder(folder_path: str) -> dict:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Not a valid directory: {folder_path}")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    ingested = 0
    skipped = 0

    with get_connection() as conn:
        existing = {row["filepath"] for row in conn.execute("SELECT filepath FROM media")}

        for src in sorted(folder.rglob("*")):
            if "__MACOSX" in src.parts or src.name.startswith("._"):
                continue
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            if not src.is_file():
                continue

            try:
                file_hash = _file_hash(src)
                dest = UPLOADS_DIR / f"{file_hash}{src.suffix.lower()}"
                thumb_path = THUMBNAILS_DIR / f"{file_hash}.jpg"

                if str(dest) in existing:
                    skipped += 1
                    continue

                shutil.copy2(src, dest)

                with Image.open(dest) as img:
                    date_taken = _extract_exif_date(img)
                    _make_thumbnail(dest, thumb_path)

                media_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT OR IGNORE INTO media
                       (id, filepath, thumbnail_path, media_type, date_taken, date_ingested, processed)
                       VALUES (?, ?, ?, 'image', ?, ?, 0)""",
                    (
                        media_id,
                        str(dest),
                        str(thumb_path.relative_to(THUMBNAILS_DIR.parent)),
                        date_taken,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                existing.add(str(dest))
                ingested += 1
                logger.info(f"Ingested: {src.name}")

            except Exception as e:
                logger.error(f"Failed to ingest {src}: {e}")
                skipped += 1

    return {"ingested": ingested, "skipped": skipped}


def ingest_uploaded_files(files: list[tuple[str, bytes]]) -> dict:
    extract_dir = UPLOADS_DIR / "files_upload_tmp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        for filename, content in files:
            dest = extract_dir / Path(filename).name
            dest.write_bytes(content)
        return ingest_folder(str(extract_dir))
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _safe_extract(zf: zipfile.ZipFile, extract_dir: Path):
    extract_root = extract_dir.resolve()
    for member in zf.namelist():
        member_path = (extract_dir / member).resolve()
        if member_path != extract_root and extract_root not in member_path.parents:
            raise ValueError(f"Unsafe zip member path: {member}")
    zf.extractall(extract_dir)


def ingest_zip(zip_path: Path) -> dict:
    extract_dir = UPLOADS_DIR / "zip_extract_tmp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract(zf, extract_dir)
        return ingest_folder(str(extract_dir))
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()
