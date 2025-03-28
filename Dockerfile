FROM python:3.12.9

WORKDIR .

RUN apt update && apt install -y tesseract-ocr

RUN apt install -y ghostscript

RUN apt-get install -y libgl1-mesa-glx

COPY requirements.txt .

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy saved preprocess
COPY json_output /json_output
COPY saved_index /saved_index

# Creating directories to handle added docs
RUN mkdir input
RUN mkdir input/html
RUN mkdir input/json
RUN mkdir input/pdf

COPY utils /utils
COPY RAG_open_quests.py .
COPY RAG_MCQ.py .
COPY app.py .

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py"]