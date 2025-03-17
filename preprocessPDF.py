# pdf_to_json.py
import os
import io
import json
import PyPDF2
import pytesseract
from PIL import Image

def extract_text_from_page(page, pdf_filename, page_number):
    """
    Extrait le texte d'une page, en traitant également les images via OCR.
    """
    page_text = ""
    try:
        # Traitement des images de la page
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
        # Extraction du texte de la page
        text = page.extract_text()
        if text:
            page_text += text + "\n"
    except Exception as e:
        print(f"Erreur lors du traitement de la page {page_number} de {pdf_filename}: {e}")
    return page_text

def process_pdf_and_save_json(pdf_folder, save_folder):
    """
    Parcourt tous les PDF d'un dossier, extrait le texte de chaque page,
    et sauvegarde dans un fichier JSON par PDF avec le nom du PDF.
    Chaque JSON contient une liste de dictionnaires avec le numéro de page et le texte extrait.
    """
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
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
                # Sauvegarder dans un fichier JSON nommé d'après le PDF
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(save_folder, json_filename)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(pages_data, f, ensure_ascii=False, indent=4)
                print(f"Fichier JSON sauvegardé : {json_path}")
            except Exception as e:
                print(f"Erreur lors du traitement de {pdf_path} : {e}")

if __name__ == "__main__":
    pdf_folder = "data"          # Dossier contenant vos PDF
    save_folder = "json_output"  # Dossier de sauvegarde des fichiers JSON
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    process_pdf_and_save_json(pdf_folder, save_folder)
