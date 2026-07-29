import re
import logging
from pathlib import Path

from PIL import Image

from db.database import get_connection

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    # MM/DD/YY or MM/DD/YYYY
    (r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", "mdy"),
    # Month YYYY e.g. "Jan 1985" or "January 1985"
    (r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
     r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
     r"[\s,]+(\d{4})\b", "month_year"),
    # YYYY standalone
    (r"\b(19[5-9]\d|20[0-2]\d)\b", "year_only"),
]

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _parse_date(text: str) -> str | None:
    for pattern, fmt in DATE_PATTERNS:
        if fmt == "year_only":
            # Iterate all matches and skip ones that look like prices, percentages,
            # phone numbers, or zip codes rather than a printed date.
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start, end = m.span(1)
                before = text[start - 1:start]
                after = text[end:end + 1]
                if before in "$." or before.isdigit() or after.isdigit() or after == "%":
                    continue
                return f"{m.group(1)}-01-01"
            continue

        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        try:
            if fmt == "mdy":
                month, day, year = int(m.group(1)), int(m.group(2)), m.group(3)
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    continue
                if len(year) == 2:
                    year = ("19" if int(year) > 30 else "20") + year
                return f"{year}-{month:02d}-{day:02d}"
            elif fmt == "month_year":
                month_str = m.group(1)[:3].lower()
                month = MONTH_MAP.get(month_str, "01")
                year = m.group(2)
                return f"{year}-{month}-01"
        except Exception:
            continue
    return None


def _run_ocr(image: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except ImportError:
        logger.warning("pytesseract not installed, skipping OCR")
        return ""
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


def run_ocr(limit: int = 0) -> dict:
    with get_connection() as conn:
        query = "SELECT id, filepath, date_taken FROM media WHERE ocr_text IS NULL"
        params: tuple = ()
        if limit:
            query += " LIMIT ?"
            params = (limit,)
        rows = list(conn.execute(query, params))

    processed = 0
    dates_found = 0

    for row in rows:
        media_id = row["id"]
        filepath = row["filepath"]
        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                full_text = _run_ocr(img)

                # Also OCR bottom 15%
                w, h = img.size
                bottom_strip = img.crop((0, int(h * 0.85), w, h))
                bottom_text = _run_ocr(bottom_strip)

            combined_text = full_text + "\n" + bottom_text
            extracted_date = _parse_date(combined_text)

            with get_connection() as conn:
                if extracted_date and not row["date_taken"]:
                    conn.execute(
                        "UPDATE media SET ocr_text = ?, date_taken = ? WHERE id = ?",
                        (combined_text.strip(), extracted_date, media_id),
                    )
                    dates_found += 1
                else:
                    conn.execute(
                        "UPDATE media SET ocr_text = ? WHERE id = ?",
                        (combined_text.strip(), media_id),
                    )

            processed += 1
        except Exception as e:
            logger.error(f"OCR pipeline failed for {filepath}: {e}")
            with get_connection() as conn:
                conn.execute("UPDATE media SET ocr_text = '' WHERE id = ?", (media_id,))

    return {"processed": processed, "dates_found": dates_found}
