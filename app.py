from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st
from utils.chunks import load_index_and_chunks
from utils.chunks import retrieve_chunks
from utils.repair_pdf import repair_pdf
from utils.preprocess import process_pdf_to_json, process_html_to_json
from utils.faiss_index import update_index
from RAG_open_quests import RAG_conv
from dotenv import load_dotenv
import os
import tempfile
import shutil

# First thing to call
st.set_page_config(page_title="RAG Model", layout="wide")

# Load env variables
load_dotenv()

@st.cache_resource
def load_embedding_model():
    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
    return embedding_model

@st.cache_resource
def load_chunks():
    save_folder = "saved_index"
    print("Chargement de l'index FAISS et des chunks sauvegardés...")
    index, chunks = load_index_and_chunks(save_folder)
    dict_chunks = {
        "index": index,
        "chunks": chunks
    }
    return dict_chunks

@st.cache_resource
def client_for_inference():
    client_inference = InferenceClient(
        provider="novita",
        # provider="hf-inference"
        # provider="nebius",
        token=os.environ.get("API_KEY")
    )
    return client_inference

# This is for the RAG for open questions
def rag_model(model_id, client_inference, query, dict_chunks):
    index = dict_chunks["index"]
    chunks = dict_chunks["chunks"]

    retrieved_chunks = retrieve_chunks(query, index, chunks, embedding_model)

    context = "\n".join([f"[PDF: {chunk['pdf']} - Page: {chunk['page']}] {chunk['text']}" for chunk in retrieved_chunks])

    # Define system prompt only if the conversation has just started
    if len(st.session_state.model_conv_history) == 0:
        if (st.session_state.RAG_type == "open_questions"):
            system_prompt = "You are an expert in patent laws. You provide both detailed answers and your source (include the name of the document)."
        elif (st.session_state.RAG_type == "create_questions"):
            system_prompt = "You are an expert in patent laws and making questions on the subject. You provide both detailed answers and your source after the user answers (include the name of the document)."
        st.session_state.model_conv_history.append({
            "role": "system",
            "content": f"Do not show this on your response.\n{system_prompt}\nYou always answer in the same language as the query and within 500 words sources included.\nYou have access to the following context : {context}"
        })
        st.session_state.model_conv_history.append({
            "role": "user",
            "content": query
        })
    else:
        # Add additional context
        st.session_state.model_conv_history.append({
            "role": "system",
            "content": f"Additional context : {context}"
        })
        # Add query
        st.session_state.model_conv_history.append({
            "role": "user",
            "content": query
        })
    model_res = RAG_conv(model_id, client_inference, st.session_state.model_conv_history)

    st.session_state.model_conv_history.append({
        "role": "assistant",
        "content": model_res
    })

    return model_res

def rag_MCQ(model_id, client_inference, query, dict_chunks):
    index = dict_chunks["index"]
    chunks = dict_chunks["chunks"]

    retrieved_chunks = retrieve_chunks(query, index, chunks, embedding_model)

    context = "\n".join([f"[PDF: {chunk['pdf']} - Page: {chunk['page']}] {chunk['text']}" for chunk in retrieved_chunks])

    # Define system prompt only if the conversation has just started
    if len(st.session_state.model_conv_history) == 0:
        system_prompt = "You are an expert in patent laws and specialized in making multiple choice questions on the subject (one or multiple good answers). You provide both detailed answers and your source after the user answers (include the name of the document)."
        st.session_state.model_conv_history.append({
            "role": "system",
            "content": f"Do not show this on your response.\n{system_prompt}\nYou always answer in the same language as the query and within 500 words sources included.\nYou have access to the following context : {context}"
        })
        st.session_state.model_conv_history.append({
            "role": "user",
            "content": query
        })
    else:
        # No additional context
        st.session_state.model_conv_history.append({
            "role": "user",
            "content": query
        })
    model_res = RAG_conv(model_id, client_inference, st.session_state.model_conv_history)

    st.session_state.model_conv_history.append({
        "role": "assistant",
        "content": model_res
    })

    return model_res

# Display conversation history
def display_conversations():
    for conversation in st.session_state.interface_conv_history:
        st.markdown(f"**User**: {conversation['user']}")
        st.markdown(f"**Model**: {conversation['model']}")

def submit():
    st.session_state.input = st.session_state.user_input
    st.session_state.user_input = ''

def reset_conv():
    st.session_state.interface_conv_history = []
    st.session_state.model_conv_history = []
    st.session_state.input = ''
    st.rerun()  # Rerun the script to reflect changes immediately

def init_session_states():
    # Session state for conversations for the model
    if 'model_conv_history' not in st.session_state:
        st.session_state.model_conv_history = []

    if 'model_id' not in st.session_state:
        st.session_state.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        # st.session_state.model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    if 'RAG_type' not in st.session_state:
        st.session_state.RAG_type = "open_questions"

    # Session state for the conversations for the UI
    if 'interface_conv_history' not in st.session_state:
        st.session_state.interface_conv_history = []

    if 'input' not in st.session_state:
        st.session_state.input = ''

    # Session state for file uploader toggles
    if 'show_pdf_uploader' not in st.session_state:
        st.session_state.show_pdf_uploader = False
    if 'show_html_uploader' not in st.session_state:
        st.session_state.show_html_uploader = False

init_session_states()

embedding_model = load_embedding_model()
dict_chunks = load_chunks()
client_inference = client_for_inference()

st.title("Retrieval-Augmented Generation (RAG) Base")

st.write(
    """
    This is a base Streamlit app for RAG. You can enter multiple queries 
    and view the corresponding responses.
    """
)

if st.session_state.RAG_type == "open_questions":
    st.write("Ready to answer any questions you may have.")
elif st.session_state.RAG_type == "MCQ":
    st.write(
        """
        Specialized in making MCQs.
        First, input a subject and then answer with only the corresponding letter or number.
        You can reset this conversation or open a new page if you want a MCQ for another subject.
        """
    )
elif st.session_state.RAG_type == "create_questions":
    st.write("Ready to create questions and give corrections on the answer.")

# Process input when the user submits
if st.session_state.input:
    # Call the RAG model based on the selected type
    if st.session_state.RAG_type == "open_questions" or st.session_state.RAG_type == "create_questions":
        model_response = rag_model(st.session_state.model_id, client_inference, st.session_state.input, dict_chunks)
    elif st.session_state.RAG_type == "MCQ":
        model_response = rag_MCQ(st.session_state.model_id, client_inference, st.session_state.input, dict_chunks)

    # Store the conversation in session state
    st.session_state.interface_conv_history.append({
        'user': st.session_state.input,
        'model': model_response,
    })
    
    display_conversations()

# Input box at the bottom
user_input = st.text_input("Your Input:", key="user_input", on_change=submit)

RAG_choices = {
    "Answer questions": "open_questions", 
    "MCQs": "MCQ",
    "Create questions": "create_questions"
}

col1, col2, col3 = st.columns(3)

with col1:
    btn1 = st.button('Reset Conversation')
with col2:
    btn2 = st.button('Add pdf')
with col3:
    btn3 = st.button('Add html')

# Button actions
if btn1:
    reset_conv()

if btn2:
    st.session_state.show_pdf_uploader = True

if btn3:
    st.session_state.show_html_uploader = True

# Display file uploaders based on session state flags
if st.session_state.show_pdf_uploader:
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")
    if pdf_file is not None:
        st.write("PDF file uploaded:", pdf_file.name)
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        # Define an output path for the repaired PDF
        output_path = os.path.join("input/pdf",pdf_file.name)
        # Repair the PDF using the provided function
        if not os.path.exists(output_path):
            repair_pdf(tmp_path, output_path)
            st.write(f"PDF repaired and saved at: {output_path}")
            output_str, err = process_pdf_to_json(output_path, "input/json")
            st.write(output_str)
            if err:
                st.error("Stop PDF processing")
            else : 
                st.write("Updating the FAISS index with the new data...")
                update_index("input/json", "saved_index", embedding_model)
                if os.path.exists("input/json"):
                    shutil.rmtree("input/json")
                os.makedirs("input/json")
                load_chunks.clear()
                dict_chunks = load_chunks()
                st.write("FAISS index updated with the new data...")
        else:
            st.error(f"PDF already exists at: {output_path}")
        st.session_state.show_pdf_uploader = False

if st.session_state.show_html_uploader:
    html_file = st.file_uploader("Upload an HTML file", type="html", key="html_uploader")
    if html_file is not None:
        st.write("HTML file uploaded:", html_file.name)
        # Define the output path for the HTML file
        output_path = os.path.join("input/html", html_file.name)
        # Check if the file already exists
        if not os.path.exists(output_path):
            # Save the uploaded HTML file directly to the folder
            with open(output_path, "wb") as f:
                f.write(html_file.read())
            st.write(f"HTML file saved at: {output_path}")
            output_str, err = process_html_to_json(output_path, "input/json")
            st.write(output_str)
            if err:
                st.error("Stop HTML processing")
            else : 
                st.write("Updating the FAISS index with the new data...")
                update_index("input/json", "saved_index", embedding_model)
                if os.path.exists("input/json"):
                    shutil.rmtree("input/json")
                os.makedirs("input/json")
                load_chunks.clear()
                dict_chunks = load_chunks()
                st.write("FAISS index updated with the new data...")
        else:
            st.error(f"HTML file already exists at: {output_path}")
        st.session_state.show_html_uploader = False

selCol1, _ = st.columns([1, 4])

with selCol1:
    RAG_option = st.selectbox(
        label="What functionality do you need?",
        options=list(RAG_choices.keys()),
        index=0
    )

if RAG_option:
    session_RAG_type = RAG_choices[RAG_option]
    if session_RAG_type != st.session_state.RAG_type:
        st.session_state.RAG_type = session_RAG_type
        reset_conv()

st.write("Currently selected:", RAG_option)
