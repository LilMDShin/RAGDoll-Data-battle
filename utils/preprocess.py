import os
import io
import re
import json
import base64
import PyPDF2
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup


def sanitize_filename(filename: str) -> str:
    """
    Replace characters that are invalid on typical filesystems with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)


def extract_text_from_page(page, pdf_filename, page_number):
    """
    Extract text from a PDF page, including OCR on images if present.
    """
    page_text = ""
    try:
        # Process images on the page
        if hasattr(page, 'images'):
            for image_file_object in page.images:
                image_data = image_file_object.data
                try:
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    print(f"Erreur lors du décodage de l'image sur la page {page_number} de {pdf_filename}: {e}")
                    continue
                caption = pytesseract.image_to_string(image)
                if caption:
                    page_text += caption + "\n"
                else:
                    print("Aucun texte généré pour cette image.")
        # Extract text from the page
        text = page.extract_text()
        if text:
            page_text += text + "\n"
    except Exception as e:
        print(f"Erreur lors du traitement de la page {page_number} de {pdf_filename}: {e}")
    return page_text

def process_pdf_to_json(pdf_path, save_folder):
     # Assume extract_text_from_page is defined elsewhere
    filename = os.path.basename(pdf_path)
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(save_folder, json_filename)
    
    if os.path.exists(json_path):
        return f"Skipping PDF {filename} as JSON already exists: {json_path}" , 1

    pages_data = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file, strict=False)
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = extract_text_from_page(page, filename, page_number)
                if page_text.strip():
                    pages_data.append({
                        "page": page_number,
                        "text": page_text.strip()
                    })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=4)
        return f"JSON file saved: {json_path}", 0
    except Exception as e:
        return f"Error processing {pdf_path}: {e}" , 1
    
def process_html_to_json(html_path, save_folder):
    # Assume extract_text_from_html_file is defined elsewhere
    filename = os.path.basename(html_path)
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(save_folder, json_filename)
    
    if os.path.exists(json_path):
        return f"Skipping HTML {filename} as JSON already exists: {json_path}", 1
        

    print(f"Processing {filename} (HTML)...")
    try:
        extracted_text = extract_text_from_html_file(html_path)
        pages_data = []
        if extracted_text.strip():
            pages_data.append({
                "page": 1,
                "text": extracted_text.strip()
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=4)
        return f"JSON file saved: {json_path}", 0
    except Exception as e:
        return f"Error processing {html_path}: {e}", 1

def process_pdf_and_save_json(pdf_folder, save_folder):
    """
    Process all PDF files in the given folder, extract text from each page,
    and save the results to a JSON file with the same name as the PDF.
    If the JSON file already exists, skip processing.
    """
    for root, _, files in os.walk(pdf_folder, topdown=False):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(save_folder, json_filename)
                if os.path.exists(json_path):
                    print(f"Skipping PDF {filename} as JSON already exists: {json_path}")
                    continue

                print(f"Traitement de {filename}...")
                pages_data = []
                try:
                    with open(pdf_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file, strict=False)
                        for page_number, page in enumerate(pdf_reader.pages, start=1):
                            page_text = extract_text_from_page(page, filename, page_number)
                            if page_text.strip():
                                pages_data.append({
                                    "page": page_number,
                                    "text": page_text.strip()
                                })
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(pages_data, f, ensure_ascii=False, indent=4)
                    print(f"Fichier JSON sauvegardé : {json_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de {pdf_path} : {e}")


def extract_text_from_html_file(html_path):
    """
    Extract text from an HTML file using BeautifulSoup and process inline images (data URI) via OCR.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture de {html_path}: {e}")
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the main text from the HTML
    text = soup.get_text(separator="\n").strip()

    # Process inline images (if any) that are embedded as data URI
    image_text = ""
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and src.startswith("data:image"):
            try:
                header, encoded = src.split(",", 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text:
                    image_text += ocr_text + "\n"
            except Exception as e:
                print(f"Erreur lors du traitement de l'image inline dans {html_path}: {e}")

    return text + "\n" + image_text


def process_html_and_save_json(html_folder, save_folder):
    """
    Process all HTML files in the given folder, extract their text (and inline image text via OCR),
    and save the results to a JSON file with the same name as the HTML.
    If the JSON file already exists, skip processing.
    """
    for root, _, files in os.walk(html_folder, topdown=False):
        for filename in files:
            if filename.lower().endswith(".html"):
                html_path = os.path.join(root, filename)
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(save_folder, json_filename)
                if os.path.exists(json_path):
                    print(f"Skipping HTML {filename} as JSON already exists: {json_path}")
                    continue

                print(f"Traitement de {filename} (HTML)...")
                try:
                    extracted_text = extract_text_from_html_file(html_path)
                    pages_data = []
                    if extracted_text.strip():
                        pages_data.append({
                            "page": 1,
                            "text": extracted_text.strip()
                        })
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(pages_data, f, ensure_ascii=False, indent=4)
                    print(f"Fichier JSON sauvegardé : {json_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de {html_path} : {e}")


if __name__ == "__main__":
    input_folder = "data"  # Folder containing your PDF and HTML files
    output_folder = "json_output"  # Folder to save the JSON files

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Traitement des fichiers PDF...")
    process_pdf_and_save_json(input_folder, output_folder)

    print("Traitement des fichiers HTML...")
    process_html_and_save_json(input_folder, output_folder)
