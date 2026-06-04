# MemoryGraph

Local photo search engine that builds a semantic graph from an image library. Upload photos and search them by describing what's in them (faces, scenes, text, dates) instead of by filename.

Extracts structured data from each image (face encodings, OCR text, scene classification, EXIF metadata), combines it into a sentence embedding, stores everything in SQLite + ChromaDB, and exposes a FastAPI search endpoint. Query with natural language, get back ranked results by cosine similarity.

---

## System Requirements

- **Python 3.11+**
- **Tesseract OCR binary** (for date extraction from scanned prints):
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: Download from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- ~300MB disk space for the CLIP model (downloaded automatically on first run)

---

## Setup

```bash
bash setup.sh
```


This will:
1. Create a Python virtual environment
2. Install `cmake` + `dlib` (required by the face recognition library)
3. Install all Python dependencies
4. Download the CLIP model (~300MB, one time)
5. Initialize the SQLite database

---

## Running

```bash
source venv/bin/activate
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

---

## Usage

1. **Ingest**: Enter a local folder path containing JPG/PNG/TIFF photos, or drop a `.zip` file. Click **Process All** to run the full pipeline.
2. **People**: Review auto-detected face clusters. Type a real name for each person and click **Save** to confirm. Confirmed names immediately improve search.
3. **Search**: Type natural language queries in the search bar. Try:
   - `grandma at Christmas`
   - `beach photos from the 80s`
   - `birthday party with kids`
   - `wedding 1975`

---

## Architecture

The pipeline runs in 6 sequential stages:

| Stage | What it does |
|---|---|
| **1 · Ingest** | Walks a folder or zip, detects image types, generates thumbnails, extracts EXIF dates, stores metadata in SQLite |
| **2 · Faces** | Detects face bounding boxes with dlib, generates 128-dim embeddings, clusters people with DBSCAN |
| **3 · Scenes** | Encodes each image with CLIP (`clip-ViT-B-32`), scores against 40 scene labels, stores top-3 tags + embedding in ChromaDB |
| **4 · OCR** | Runs Tesseract on each image + bottom 15% strip, extracts dates from printed lab labels |
| **5 · Graph** | Builds a NetworkX knowledge graph: Person → Media → Event → Place/TimePeriod nodes |
| **6 · Search** | Combines CLIP vector search (ChromaDB) with graph traversal (person/tag/decade matching) for natural language queries |

---

## Privacy

**All processing is local. No data ever leaves your machine.**

There are no cloud API calls anywhere in this codebase. Face recognition, image understanding, OCR, and search all run fully on your hardware. Your family photos never touch a third-party server.
