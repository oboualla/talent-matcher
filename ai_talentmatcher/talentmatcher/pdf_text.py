"""Step 1: turn an uploaded PDF into plain text with PyMuPDF."""

import pymupdf


class PdfTextError(Exception):
    """The PDF cannot be opened, or it contains no extractable text."""


# Fewer meaningful characters than this means "probably a scanned/image PDF".
MIN_TEXT_CHARS = 30


def extract_text(pdf_bytes: bytes) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # corrupted file, not a PDF, ...
        raise PdfTextError(f"The file could not be opened as a PDF ({exc}).") from exc

    with doc:
        if doc.needs_pass:
            raise PdfTextError("The PDF is password-protected.")
        # sort=True orders text blocks top-to-bottom, left-to-right, which keeps
        # multi-column CVs more readable for the model.
        pages = [page.get_text("text", sort=True) for page in doc]

    text = "\n".join(pages).strip()
    if len("".join(text.split())) < MIN_TEXT_CHARS:
        raise PdfTextError(
            "No extractable text was found. Only text-based PDFs are supported "
            "(scanned documents and images are not)."
        )
    return text
