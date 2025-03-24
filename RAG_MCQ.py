from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from utils.chunks import load_index_and_chunks
from utils.chunks import retrieve_chunks

def RAG_conv(model_id, client_inference, conv_history):
    completion = client_inference.chat.completions.create(
        model=model_id,
        messages=conv_history,
        max_tokens=550,
        temperature=0,
        # top_p=1
        top_p=0.99
    )
    model_res = completion.choices[0].message.content
    return(model_res)

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
        # provider="hf-inference",
        provider="novita",
        # token = os.environ.get("API_KEY"),
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
    
    # To test
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    model_res = RAG_conv(model_id, client, conv_history)

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

        model_res = RAG_conv(model_id, client, conv_history)

        print(model_res)

        user_res = input("Continuer ? 1 pour oui, 0 pour non : ")
        if user_res.isdigit():
            user_val = int(user_res)
        else:
            user_val = 0
