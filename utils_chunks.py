import os
import json
import faiss
import numpy as np

def load_index_and_chunks(save_folder):
    """
    Charge l'index FAISS et les chunks sauvegardés.
    """
    index_path = os.path.join(save_folder, "faiss_index.index")
    chunks_path = os.path.join(save_folder, "chunks.json")
    
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve_chunks(query, index, chunks, embedding_model, k=5):
    """
    Recherche les k chunks les plus pertinents pour une requête donnée.
    """
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved