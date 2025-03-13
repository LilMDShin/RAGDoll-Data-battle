import PyPDF2
import os
import io
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            # Process each image on the page
            if hasattr(page, 'images'):
                for image_file_object in page.images:
                    image_data = image_file_object.data
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        # Conversion en RGB pour s'assurer que l'image a trois canaux
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                    except Exception as e:
                        print(f"Erreur lors du décodage de l'image: {e}")
                        continue
                    caption = image_to_string(image)
                    if caption is not None:
                        text += caption + "\n"
                    else:
                        print("Aucun texte généré pour cette image.")
            # Extraire le texte de la page
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def image_to_string(image):
    """
    Génère une transcription textuelle de l'image à l'aide du modèle TrOCR.
    Retourne toujours une chaîne de caractères (même vide en cas d'échec).
    """
    try:
        # Charger le processor et le modèle
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        if decoded and len(decoded) > 0:
            return decoded[0]
        else:
            return ""
    except Exception as e:
        print(f"Erreur lors de la génération du texte depuis l'image : {e}")
        return ""

def split_text(text, chunk_size=500, overlap=100):
    """
    Découpe un texte en morceaux de taille définie avec un chevauchement.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def preprocess_pdf(pdf_folder, all_chunks):
    # Parcourir tous les PDF du dossier
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Traitement de {filename}...")
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)
            all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    text = extract_text_from_pdf("data/drive-download-20250311T094842Z-001/EQE Exams/02-Paper D/Archives/1990_paperD_pt1_en.pdf")
    with open("reponse_generée.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("La réponse générée a été sauvegardée dans le fichier 'reponse_generée.txt'")
