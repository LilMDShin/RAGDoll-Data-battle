# update_faiss_index.py
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

def load_pages_and_create_chunks(json_folder):
    """
    Charge tous les fichiers JSON du dossier, contenant des pages extraites,
    et crée des chunks pour chaque page.
    Chaque chunk est un dictionnaire contenant le texte extrait, le nom du PDF et le numéro de page.
    Le nom du PDF est déduit du nom du fichier JSON.
    """
    all_chunks = []
    for filename in os.listdir(json_folder):
        if filename.lower().endswith(".json"):
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

def update_index(new_json_folder, saved_index_folder, embedding_model):
    """
    Met à jour l'index FAISS et le fichier JSON des chunks en ajoutant
    de nouvelles données provenant d'un dossier de fichiers JSON.
    """
    index_path = os.path.join(saved_index_folder, "faiss_index.index")
    chunks_path = os.path.join(saved_index_folder, "chunks.json")
    
    # Charger l'index FAISS existant
    index = faiss.read_index(index_path)
    
    # Charger les chunks existants
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    
    # Charger et créer les nouveaux chunks à partir des nouveaux fichiers JSON
    new_chunks = load_pages_and_create_chunks(new_json_folder)
    if not new_chunks:
        print("Aucune nouvelle donnée trouvée.")
        return
    print(f"Nombre de nouveaux chunks créés : {len(new_chunks)}")
    
    # Calculer les embeddings pour les nouveaux chunks
    new_texts = [chunk["text"] for chunk in new_chunks]
    new_embeddings = embedding_model.encode(new_texts, show_progress_bar=True)
    new_embeddings = np.array(new_embeddings).astype("float32")
    
    # Ajouter les nouveaux embeddings à l'index existant
    index.add(new_embeddings)
    
    # Mettre à jour la liste des chunks
    all_chunks.extend(new_chunks)
    
    # Sauvegarder l'index FAISS mis à jour
    faiss.write_index(index, index_path)
    print(f"Index FAISS mis à jour et sauvegardé dans {index_path}")
    
    # Sauvegarder les chunks mis à jour
    with open(chunks_path, "w", encoding="utf-8") as f:
         json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"Chunks mis à jour et sauvegardés dans {chunks_path}")

if __name__ == "__main__":
    # Dossiers utilisés
    saved_index_folder = "saved_index"   # Dossier contenant l'index FAISS et le fichier chunks.json existants
    new_json_folder = "new_json"           # Dossier contenant les nouveaux fichiers JSON à ajouter

    print("Chargement du modèle d'embeddings...")
    # Utilisez le même modèle que lors de la création initiale de l'index
    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
    
    print("Mise à jour de l'index FAISS avec les nouvelles données...")
    update_index(new_json_folder, saved_index_folder, embedding_model)

    #embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
#
    #print("Construction de l'index FAISS...")
    #index = build_index(all_chunks, embedding_model)
#
    #save_folder = "saved_index"
    #if not os.path.exists(save_folder):
    #    os.makedirs(save_folder)
#
    #index_path = os.path.join(save_folder, "faiss_index.index")
    #faiss.write_index(index, index_path)
    #print(f"Index FAISS sauvegardé dans {index_path}")
#
    #chunks_path = os.path.join(save_folder, "chunks.json")
    #with open(chunks_path, "w", encoding="utf-8") as f:
    #    json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    #print(f"Chunks sauvegardés dans {chunks_path}")
