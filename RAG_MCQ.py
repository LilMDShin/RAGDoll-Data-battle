import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

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
    # load environment variables from .env file
    load_dotenv()

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
    
    context = "\n".join([f"[PDF: {chunk['pdf']} - Page: {chunk['page']}] {chunk['text']}" for chunk in retrieved_chunks])

    system_prompt = "You are an expert in patent laws and specialized in making MCQs. You provide both detailed answers and your source after the user answers (include the name of the source document)."

    client = InferenceClient(
        provider="nebius",
        api_key=os.environ.get("API_KEY"),
    )

    conv_history = [
            {
                "role": "system",
                "content": f"{system_prompt}\nYou always answer in the same language as the query.\nYou have access to the following context : {context}"
            }, 
            {
                "role": "user",
                "content": query
            }
        ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=conv_history,
        max_tokens=500,
        temperature=0,
        top_p=1
    )

    model_res = completion.choices[0].message.content

    print(model_res)

    user_res = input("Continuer ? 1 pour oui, 0 pour non : ")

    if (user_res.isdigit()):
        user_val = int(user_res)
    else:
        user_val = 0
    while (user_val == 1):
        conv_history.append({
            "role": "assistant",
            "content": model_res
        })
        query = input("Entrez votre requête : ")
        print(f"Requête : {query}")
        # Do not add additional context when user answers MCQ
        conv_history.append({
            "role": "user",
            "content": query
        })
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=conv_history,
            max_tokens=500,
            temperature=0,
            top_p=1
        )
        model_res = completion.choices[0].message.content
        print("\nRéponse générée :")
        print(model_res)
        user_res = input("Continuer ? 1 pour oui, 0 pour non : ")
        if (user_res.isdigit()):
            user_val = int(user_res)
        else:
            user_val = 0
