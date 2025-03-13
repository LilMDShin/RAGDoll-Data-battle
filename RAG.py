import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from preprocessPDF import preprocess_pdf

def build_index(chunks, embedding_model):
    """
    Calcule les embeddings des morceaux et construit un index FAISS.
    """
    # Calcul des embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Création de l'index FAISS (IndexFlatL2 pour la distance euclidienne)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, index, chunks, embedding_model, k=5):
    """
    Recherche les k morceaux les plus pertinents par rapport à la requête.
    """
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    # Récupérer les morceaux correspondants
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def generate_answer(query, retrieved_chunks, generator):
    """
    Construit un prompt en combinant le contexte récupéré et la requête,
    puis utilise le générateur pour produire une réponse.
    """
    context = " ".join(retrieved_chunks)
    prompt = f"Contexte: {context}\n\nRépondez en français a partir du context a cette question: {query}\n\nRéponse:"
    generated = generator(prompt, max_new_tokens=200, num_return_sequences=1)
    return generated[0]['generated_text']

#TODO: Add IA for Images To Text https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending

if __name__ == "__main__":
    # Dossier contenant les fichiers PDF
    pdf_folder = "data"  # Assurez-vous que ce dossier existe et contient vos PDF
    all_chunks = []
    
    # Chargement du modèle d'embeddings
    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = preprocess_pdf(pdf_folder, all_chunks)
    
    # Construction de l'index FAISS avec tous les morceaux extraits
    print("Construction de l'index FAISS...")
    index, _ = build_index(all_chunks, embedding_model)
    
    # Chargement du générateur de texte (ici GPT-2)
    print("Chargement du générateur de texte...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    generator = pipe = pipeline(
    "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )   
    
    # Exemple de requête
    query = "Quel est le contenu principal des documents ?"
    print(f"Requête : {query}")
    
    # Récupération des morceaux les plus pertinents
    retrieved_chunks = retrieve_chunks(query, index, all_chunks, embedding_model)
    
    # Génération de la réponse en se basant sur le contexte récupéré
    answer = generate_answer(query, retrieved_chunks, generator)
    with open("reponse_generée.txt", "w", encoding="utf-8") as f:
        f.write(answer)
    print("La réponse générée a été sauvegardée dans le fichier 'reponse_generée.txt'")

    
    print("\nRéponse générée :")
    print(answer)
