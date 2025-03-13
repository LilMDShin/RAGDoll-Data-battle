import os
import io
import json
import faiss
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

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

def split_text(text, chunk_size=500, overlap=100):
    """
    Découpe un texte en morceaux (chunks) de taille définie avec chevauchement.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def preprocess_pdf(pdf_folder):
    """
    Parcourt tous les PDF d'un dossier, extrait leur contenu et découpe le texte en chunks.
    Chaque chunk est un dictionnaire contenant le texte extrait, le nom du PDF et le numéro de page.
    """
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Traitement de {filename}...")
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    for page_number, page in enumerate(pdf_reader.pages, start=1):
                        page_text = extract_text_from_page(page, filename, page_number)
                        if page_text.strip():
                            # Découpage du texte de la page en chunks
                            chunks = split_text(page_text)
                            for chunk in chunks:
                                chunk_dict = {
                                    "text": chunk,
                                    "pdf": filename,
                                    "page": page_number
                                }
                                all_chunks.append(chunk_dict)
            except Exception as e:
                print(f"Erreur lors de la lecture de {pdf_path}: {e}")
    return all_chunks

def build_index(chunks, embedding_model):
    """
    Calcule les embeddings pour chaque chunk (texte uniquement) et construit un index FAISS.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    pdf_folder = "data"  # Dossier contenant vos PDF
    print("Prétraitement des PDF...")
    all_chunks = preprocess_pdf(pdf_folder)
    print(f"Nombre de chunks extraits: {len(all_chunks)}")
    
    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Construction de l'index FAISS...")
    index = build_index(all_chunks, embedding_model)
    
    save_folder = "saved_index"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Sauvegarde de l'index FAISS
    index_path = os.path.join(save_folder, "faiss_index.index")
    faiss.write_index(index, index_path)
    print(f"Index FAISS sauvegardé dans {index_path}")
    
    # Sauvegarde des chunks avec leurs métadonnées
    chunks_path = os.path.join(save_folder, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"Chunks sauvegardés dans {chunks_path}")
