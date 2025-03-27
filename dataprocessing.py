import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from utils.preprocess import process_pdf_and_save_json, process_html_and_save_json
from utils.faiss_index import load_pages_and_create_chunks, build_index
from utils.emissions_tracker import codecarbone_fr
@codecarbone_fr
def dataprocessing():
    input_folder = "data"  # Folder containing your PDF and HTML files
    output_json_folder = "json_output"  # Folder to save the JSON files

    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)

    print("Traitement des fichiers PDF...")
    process_pdf_and_save_json(input_folder, output_json_folder)

    print("Traitement des fichiers HTML...")
    process_html_and_save_json(input_folder, output_json_folder)

    all_chunks = load_pages_and_create_chunks(output_json_folder)

    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

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


if __name__ == "__main__:":
    dataprocessing()