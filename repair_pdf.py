import os
from pikepdf import Pdf
from dotenv import load_dotenv

def repair_pdf_file(filepath : str):
    with Pdf.open(filepath, allow_overwriting_input=True) as pdf:
        pdf.save(filepath)


import ghostscript


def repair_pdf(input_pdf, output_pdf):
    """
    Repairs a PDF by reprocessing it with Ghostscript.

    Parameters:
        input_pdf (str): Path to the input PDF file.
        output_pdf (str): Path where the repaired PDF will be saved.
    """
    # Ghostscript arguments:
    # -sDEVICE=pdfwrite      : Use PDF writer device.
    # -dCompatibilityLevel=1.4: Set PDF compatibility level.
    # -dPDFSETTINGS=/prepress : Use prepress quality (you can adjust this setting).
    # -dNOPAUSE and -dBATCH   : Run Ghostscript without user interaction.
    # -dSAFER                : Restrict file operations for security.
    # -sOutputFile=...       : Specify the output file.
    args = [
        "gs",  # dummy value for the program name (required by ghostscript)
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/prepress",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        f"-sOutputFile={output_pdf}",
        input_pdf
    ]

    try:
        ghostscript.Ghostscript(*args)
        print(f"Repaired PDF saved to {output_pdf}")
    except Exception as e:
        print(f"Error repairing PDF: {e}")


# Example usage:
# repair_pdf("corrupted_document.pdf", "repaired_document.pdf")


if __name__ == "__main__":
    data_folder = os.path.join("data")
    load_dotenv()
    for root, _, files in os.walk(data_folder, topdown=False):
       for filename in files:
          filepath = os.path.join(root, filename)
          if filepath.lower().endswith(".pdf"):
              print(f"Traitement de : {filename}")
              repair_pdf(filepath, os.path.join("test", filepath))
