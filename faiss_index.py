# faiss_index.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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

def load_pages_and_create_chunks(json_folder):
    """
    Charge tous les fichiers JSON du dossier, contenant les pages extraites,
    et crée des chunks pour chaque page.
    Chaque chunk est un dictionnaire contenant le texte extrait, le nom du PDF et le numéro de page.
    Le nom du PDF est dérivé du nom du fichier JSON.
    """
    all_chunks = []
    for filename in os.listdir(json_folder):
        if filename.lower().endswith(".json"):
            # On déduit le nom du PDF à partir du nom du fichier JSON
            pdf_name = os.path.splitext(filename)[0] + ".pdf"
            json_path = os.path.join(json_folder, filename)
            with open(json_path, "r", encoding="utf-8") as f:
                pages = json.load(f)
            for page in pages:
                page_number = page["page"]
                text = page["text"]
                # Découper le texte en chunks
                chunks = split_text(text)
                for chunk in chunks:
                    if chunk.strip():  # Ignorer les chunks vides
                        all_chunks.append({
                            "text": chunk.strip(),
                            "pdf": pdf_name,
                            "page": page_number
                        })
    return all_chunks

def build_index(chunks, embedding_model):
    """
    Calcule les embeddings pour chaque chunk et construit un index FAISS.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    json_folder = "json_output"  # Dossier contenant les fichiers JSON générés par pdf_to_json.py
    print("Chargement des pages et création des chunks...")
    all_chunks = load_pages_and_create_chunks(json_folder)
    print(f"Nombre de chunks créés: {len(all_chunks)}")

    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Construction de l'index FAISS...")
    index = build_index(all_chunks, embedding_model)

    save_folder = "saved_index"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    index_path = os.path.join(save_folder, "faiss_index.index")
    faiss.write_index(index, index_path)
    print(f"Index FAISS sauvegardé dans {index_path}")

    chunks_path = os.path.join(save_folder, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"Chunks sauvegardés dans {chunks_path}")
