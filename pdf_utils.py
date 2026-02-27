import os
import io
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image


# --------------------------------------------------
# Extract Text Only
# --------------------------------------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


# --------------------------------------------------
# Extract Text + Diagram Descriptions (Local Build Only)
# --------------------------------------------------

def extract_text_and_images_from_pdf(file, client=None):

    # 1️⃣ Extract normal text
    full_text = extract_text_from_pdf(file)

    # If no client passed → skip vision
    if client is None:
        return full_text

    file.seek(0)

    try:
        images = convert_from_bytes(file.read())
    except:
        return full_text  # Fallback if pdf2image fails

    diagram_descriptions = ""

    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[
                    {
                        "mime_type": "image/png",
                        "data": image_bytes,
                    },
                    "Describe this diagram in detail. Explain all components, relationships, labels, and flow.",
                ],
            )

            if response.text:
                diagram_descriptions += "\n\n[DIAGRAM DESCRIPTION]\n"
                diagram_descriptions += response.text

        except:
            continue

    return full_text + "\n" + diagram_descriptions


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks