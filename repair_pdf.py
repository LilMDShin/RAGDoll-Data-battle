import os
from pikepdf import Pdf

def repair_pdf_file(filepath : str):
    with Pdf.open(filepath, allow_overwriting_input=True) as pdf:
        pdf.save(filepath)

if __name__ == "__main__":
    data_folder = os.path.join("data")
    for root, dirs, files in os.walk(data_folder, topdown=False):
        for name in files:
           filepath = os.path.join(root, name)
           if filepath.lower().endswith(".pdf"):
               print(f"Traitement de : {filepath}")
               repair_pdf_file(filepath)
