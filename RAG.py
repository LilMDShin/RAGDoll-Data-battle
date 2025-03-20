import os
import json
import faiss
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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

if __name__ == "__main__":
    time_tot_s = time.time()

    save_folder = "saved_index"
    print("Chargement de l'index FAISS et des chunks sauvegardés...")
    index, chunks = load_index_and_chunks(save_folder)
    
    print("Chargement du modèle d'embeddings...")
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
    
    query = input("Entrez votre requête : ")
    print(f"Requête : {query}")
    
    print("Recherche des chunks pertinents...")
    retrieved_chunks = retrieve_chunks(query, index, chunks, embedding_model)
    
    # Affichage des chunks récupérés avec leurs métadonnées
    print("\nChunks récupérés :")
    for chunk in retrieved_chunks:
        print(f"[PDF: {chunk['pdf']} - Page: {chunk['page']}]")
        print(chunk['text'])
        print("-" * 50)
    
    # Vous pouvez ensuite construire un prompt et générer une réponse via votre générateur de texte dans ce fichier séparé.
    # Par exemple, décommentez et adaptez le code suivant :

    # To test
    # generator = pipeline(
    #     "text-generation",
    #     model="meta-llama/Llama-3.2-1B-Instruct", # model="ministral/Ministral-3b-instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct", # model="ministral/Ministral-3b-instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    context = "\n".join([f"[PDF: {chunk['pdf']} - Page: {chunk['page']}] {chunk['text']}" for chunk in retrieved_chunks])

    # Define your custom system prompt
    # TODO : Setup and try
    # system_prompt = "You are an expert in [domain]. Please provide detailed answers."
    system_prompt = "You are an expert in patent laws. You provide both detailed answers and your source."

    # prompt = f"System:\n{system_prompt}\nnContexte:\n{context}\n\nRépondez en français à la question suivante : {query}" # \n\nRéponse:"
    prompt = f"System:\n{system_prompt}\nnContexte:\n{context}\n\nAnswer the following question in the same language : {query}\n\nAnswer:" # \n\nRéponse:"

    time_gen_s = time.time()

    generated = generator(prompt, max_new_tokens=200, num_return_sequences=1)

    # Cut to keep only after Answer [Or define a unique identifier for it] to show only the model's response

    time_gen_e = time.time()

    time_tot_e = time.time()

    print("\nRéponse générée :")
    print(generated[0]['generated_text'])

    elapsed_time_gen = time_gen_e - time_gen_s

    elapsed_time_tot = time_tot_e - time_tot_s

    print(f"Elapsed time gen text : {elapsed_time_gen}")

    print(f"Total elapsed time : {elapsed_time_tot}")
