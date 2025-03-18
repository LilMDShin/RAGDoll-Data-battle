import os
from pikepdf import Pdf

def repair_pdf_file(filepath : str):
    with Pdf.open(filepath, allow_overwriting_input=True) as pdf:
        pdf.save(filepath)

if __name__ == "__main__":
    data_folder = os.path.join("data")
    for root, _, files in os.walk(data_folder, topdown=False):
        for filename in files:
           filepath = os.path.join(root, filename)
           if filepath.lower().endswith(".pdf"):
               print(f"Traitement de : {filename}")
               repair_pdf_file(filepath)
