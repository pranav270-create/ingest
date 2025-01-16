import io
import pandas as pd
from docx import Document as DocxDocument
from PIL import Image
import fitz


def convert_to_pdf(file_content: bytes, file_extension: str) -> bytes:
    pdf_bytes = io.BytesIO()
    if file_extension in [".xlsx", ".xls"]:
        # Convert Excel to PDF
        df = pd.read_excel(io.BytesIO(file_content))
        df.to_pdf(pdf_bytes)
    elif file_extension in [".docx", ".doc"]:
        # Convert Word to PDF
        doc = DocxDocument(io.BytesIO(file_content))
        # This is a placeholder. You'll need a library like python-docx2pdf for actual conversion
        # For now, we'll just extract text
        full_text = "\n".join([para.text for para in doc.paragraphs])
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 50), full_text)
        pdf.save(pdf_bytes)
        pdf.close()
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        # Convert Image to PDF
        image = Image.open(io.BytesIO(file_content))
        pdf = fitz.open()
        page = pdf.new_page(width=image.width, height=image.height)
        page.insert_image(page.rect, stream=file_content)
        pdf.save(pdf_bytes)
        pdf.close()
    else:
        # Assume it's already a PDF
        return file_content
    return pdf_bytes.getvalue()
