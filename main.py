import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from db.database import init_db
from pipeline import ingest, faces, scene, ocr, graph, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
THUMBNAILS_DIR = BASE_DIR / "thumbnails"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB

app = FastAPI(title="MemoryGraph", version="0.1.0")

# Initialize DB on startup
@app.on_event("startup")
async def startup():
    init_db()
    UPLOADS_DIR.mkdir(exist_ok=True)
    THUMBNAILS_DIR.mkdir(exist_ok=True)
    logger.info("MemoryGraph ready")


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


# ── Ingest ────────────────────────────────────────────────────────────────────

class FolderIngestRequest(BaseModel):
    folder_path: str


@app.post("/api/ingest")
async def api_ingest(request: FolderIngestRequest):
    try:
        result = ingest.ingest_folder(request.folder_path)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/zip")
async def api_ingest_zip(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "upload.zip").name
    tmp_path = UPLOADS_DIR / f"upload_{safe_name}"
    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Uploaded zip exceeds max size")
        tmp_path.write_bytes(content)
        result = ingest.ingest_zip(tmp_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Zip ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/files")
async def api_ingest_files(files: list[UploadFile] = File(...)):
    try:
        file_data = []
        for f in files:
            content = await f.read(MAX_UPLOAD_SIZE + 1)
            if len(content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail=f"{f.filename} exceeds max upload size")
            file_data.append((f.filename, content))
        result = ingest.ingest_uploaded_files(file_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Process: full pipeline ────────────────────────────────────────────────────

@app.post("/api/process/all")
async def api_process_all():
    results = {}
    try:
        results["faces"] = faces.detect_and_embed()
        results["clustering"] = faces.cluster_faces()
    except Exception as e:
        logger.error(f"Face pipeline error: {e}")
        results["faces_error"] = str(e)

    try:
        results["scenes"] = scene.tag_scenes()
    except Exception as e:
        logger.error(f"Scene pipeline error: {e}")
        results["scenes_error"] = str(e)

    try:
        results["ocr"] = ocr.run_ocr()
    except Exception as e:
        logger.error(f"OCR pipeline error: {e}")
        results["ocr_error"] = str(e)

    try:
        graph.build_graph()
        results["graph"] = "built"
    except Exception as e:
        logger.error(f"Graph build error: {e}")
        results["graph_error"] = str(e)

    return results


# ── Face endpoints ────────────────────────────────────────────────────────────

@app.post("/api/process/faces")
async def api_process_faces():
    try:
        detection = faces.detect_and_embed()
        clustering = faces.cluster_faces()
        graph.build_graph()
        return {**detection, **clustering}
    except Exception as e:
        logger.error(f"Face processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/face-thumbnails")
async def api_backfill_face_thumbnails():
    try:
        count = faces.backfill_face_thumbnails()
        return {"generated": count}
    except Exception as e:
        logger.error(f"Face thumbnail backfill error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/people")
async def api_get_people():
    try:
        return faces.get_people_with_faces()
    except Exception as e:
        logger.error(f"Get people error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class NameRequest(BaseModel):
    name: str


@app.post("/api/people/{person_id}/name")
async def api_name_person(person_id: str, request: NameRequest):
    try:
        faces.confirm_person_name(person_id, request.name)
        graph.build_graph()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Name person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/people/{person_id}/hide")
async def api_hide_person(person_id: str):
    try:
        faces.set_person_hidden(person_id, True)
        graph.build_graph()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Hide person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Scene endpoints ───────────────────────────────────────────────────────────

@app.post("/api/process/scenes")
async def api_process_scenes():
    try:
        result = scene.tag_scenes()
        graph.build_graph()
        return result
    except Exception as e:
        logger.error(f"Scene processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Graph endpoint ────────────────────────────────────────────────────────────

@app.get("/api/graph")
async def api_get_graph():
    try:
        return graph.graph_to_dict()
    except Exception as e:
        logger.error(f"Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Search endpoint ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str


@app.post("/api/search")
async def api_search(request: SearchRequest):
    try:
        results = search.search(request.query)
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats endpoint ────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    from db.database import get_connection
    with get_connection() as conn:
        media_count = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
        processed_count = conn.execute("SELECT COUNT(*) FROM media WHERE processed=1").fetchone()[0]
        people_count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return {
        "total_media": media_count,
        "processed": processed_count,
        "people": people_count,
        "events": event_count,
    }
