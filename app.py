from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st
from utils.chunks import load_index_and_chunks, retrieve_chunks
from utils.repair_pdf import repair_pdf
from utils.preprocess import process_pdf_to_json, process_html_to_json
from utils.faiss_index import update_index
from RAG_open_quests import RAG_conv
from dotenv import load_dotenv
import os
import tempfile
import shutil

# ----------------------------
# Streamlit Setup & Environment
# ----------------------------
st.set_page_config(page_title="RAG Model", layout="wide")
load_dotenv()

# ----------------------------
# Resource Loading
# ----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L12-v2')

@st.cache_resource
def load_chunks():
    save_folder = "saved_index"
    index, chunks = load_index_and_chunks(save_folder)
    return {"index": index, "chunks": chunks}

@st.cache_resource
def client_for_inference():
    return InferenceClient(
        provider="novita",
        # provider="hf-inference"
        # provider="nebius",
        token=os.environ.get("API_KEY")
    )

# ----------------------------
# Helper Functions
# ----------------------------
def get_context(query, dict_chunks, embedding_model):
    index = dict_chunks["index"]
    chunks = dict_chunks["chunks"]
    retrieved_chunks = retrieve_chunks(query, index, chunks, embedding_model)
    # Build a string context from the retrieved chunks
    return "\n".join(
        [f"[PDF: {chunk['pdf']} - Page: {chunk['page']}] {chunk['text']}" for chunk in retrieved_chunks]
    )

def rag_generic(model_id, client_inference, query, dict_chunks, system_prompt, include_context_on_history=True):
    context = get_context(query, dict_chunks, embedding_model)
    # When starting a new conversation, add the system prompt with full context
    if not st.session_state.model_conv_history:
        # For at least Deutsch to work, need to specify in system prompt (note : English always seems to work)
        st.session_state.model_conv_history.append({
            "role": "system",
            "content": (
                f"Do not show this on your response.\n{system_prompt}\n"
                f"You always answer in the same language as the query and within 500 words (sources included).\n"
                f"You have access to the following context : {context}"
            )
        })
        st.session_state.model_conv_history.append({"role": "user", "content": query})
    else:
        # Optionally add additional context if conversation exists
        if include_context_on_history:
            st.session_state.model_conv_history.append({"role": "system", "content": f"Additional context : {context}"})
        st.session_state.model_conv_history.append({"role": "user", "content": query})
    
    model_res = RAG_conv(model_id, client_inference, st.session_state.model_conv_history)
    st.session_state.model_conv_history.append({"role": "assistant", "content": model_res})
    return model_res

def rag_model(model_id, client_inference, query, dict_chunks):
    if st.session_state.RAG_type == "open_questions":
        prompt = "You are an expert in patent laws. You provide both detailed answers and your source (include the name of the document)."
    elif st.session_state.RAG_type == "create_questions":
        prompt = (
            "You are an expert in patent laws and making questions on the subject. "
            "After the user answers, provide both detailed answers and your main sources (include the name of the document)."
        )
    return rag_generic(model_id, client_inference, query, dict_chunks, prompt, include_context_on_history=True)

def rag_MCQ(model_id, client_inference, query, dict_chunks):
    prompt = (
        "You are an expert in patent laws and specialized in making multiple choice questions (MCQ) on the subject "
        "(one or multiple correct answers). You give at least a set of 4 each time. After the user answers, provide both "
        "detailed answers and your main sources (include the name of the document)."
    )
    # For MCQs, we do not include additional context when conversation already exists
    return rag_generic(model_id, client_inference, query, dict_chunks, prompt, include_context_on_history=False)

def display_conversations():
    for conversation in st.session_state.interface_conv_history:
        st.markdown(f"**User**: {conversation['user']}")
        st.markdown(f"**Model**: {conversation['model']}")
    st.session_state.input = ''

def submit():
    st.session_state.input = st.session_state.user_input
    st.session_state.user_input = ''

def reset_conv():
    st.session_state.interface_conv_history = []
    st.session_state.model_conv_history = []
    st.session_state.input = ''

def init_session_states():
    if 'model_conv_history' not in st.session_state:
        st.session_state.model_conv_history = []
    if 'model_id' not in st.session_state:
        # st.session_state.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        st.session_state.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    if 'RAG_type' not in st.session_state:
        st.session_state.RAG_type = "open_questions"
    if 'interface_conv_history' not in st.session_state:
        st.session_state.interface_conv_history = []
    if 'input' not in st.session_state:
        st.session_state.input = ''
    if 'show_pdf_uploader' not in st.session_state:
        st.session_state.show_pdf_uploader = False
    if 'show_html_uploader' not in st.session_state:
        st.session_state.show_html_uploader = False
    if 'selBox_RAG_choice' not in st.session_state:
        st.session_state.selBox_RAG_choice = "Answer questions"
    # For now only to change the RAG's language output
    if 'language_choice' not in st.session_state:
        st.session_state.language_choice = "English"

def update_faiss_and_chunks():
    st.write("Updating the FAISS index with the new data...")
    update_index("input/json", "saved_index", embedding_model)
    if os.path.exists("input/json"):
        shutil.rmtree("input/json")
    os.makedirs("input/json")
    load_chunks.clear()
    new_chunks = load_chunks()
    st.write("FAISS index updated with the new data...")
    return new_chunks

# RAG functionality selection
def callback_selBox_RAG():
    reset_conv()
    option = st.session_state.selBox_choice
    st.session_state.RAG_type = RAG_choices[option]

# RAG response language selection
def callback_selBox_lang():
        # To test
        # reset_conv()
        st.session_state.language_choice = st.session_state.selBox_lang_choice

# ----------------------------
# Init
# ----------------------------
init_session_states()
embedding_model = load_embedding_model()
dict_chunks = load_chunks()
client_inference = client_for_inference()
# ----------------------------
# Tools and RAG Options
# ----------------------------
RAG_choices = {
    "Answer questions": "open_questions", 
    "MCQs": "MCQ",
    "Create questions": "create_questions"
}

languages = ["English", "French", "Deutsch"]

with st.sidebar:
    st.title("Tools")
    
    st.button("Reset Conversation", on_click=reset_conv)
    
    if st.button("Add PDF"):
        st.session_state.show_pdf_uploader = not st.session_state.show_pdf_uploader
    
    # PDF file uploader section
    if st.session_state.show_pdf_uploader:
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")
        if pdf_file is not None:
            st.write("PDF file uploaded:", pdf_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            output_path = os.path.join("input/pdf", pdf_file.name)
            if not os.path.exists(output_path):
                repair_pdf(tmp_path, output_path)
                st.write(f"PDF repaired and saved at: {output_path}")
                output_str, err = process_pdf_to_json(output_path, "input/json")
                st.write(output_str)
                if err:
                    st.error("Stop PDF processing")
                else:
                    dict_chunks = update_faiss_and_chunks()
                    reset_conv()
                    st.session_state.show_pdf_uploader = False
            else:
                st.error(f"PDF already exists at: {output_path}")
    
    if st.button("Add HTML"):
        st.session_state.show_html_uploader = not st.session_state.show_html_uploader

    # HTML file uploader section
    if st.session_state.show_html_uploader:
        html_file = st.file_uploader("Upload an HTML file", type="html", key="html_uploader")
        if html_file is not None:
            st.write("HTML file uploaded:", html_file.name)
            output_path = os.path.join("input/html", html_file.name)
            if not os.path.exists(output_path):
                with open(output_path, "wb") as f:
                    f.write(html_file.read())
                st.write(f"HTML file saved at: {output_path}")
                output_str, err = process_html_to_json(output_path, "input/json")
                st.write(output_str)
                if err:
                    st.error("Stop HTML processing")
                else:
                    dict_chunks = update_faiss_and_chunks()
                    reset_conv()
                    st.session_state.show_html_uploader = False
            else:
                st.error(f"HTML file already exists at: {output_path}")
    
    st.markdown("---")

    st.title("RAG options")

    selBoxCol, _ = st.columns([4, 1])

    with selBoxCol:
        lang_option = st.selectbox(
            label="Model response language",
            options=languages,
            index=0,
            key="selBox_lang_choice",
            on_change=callback_selBox_lang
        )

    with selBoxCol:
        RAG_option = st.selectbox(
            label="RAG functionality",
            options=list(RAG_choices.keys()),
            index=0,
            key="selBox_choice",
            on_change=callback_selBox_RAG
        )
    st.write("Currently selected:", RAG_option)

# ----------------------------
# Main App Logic
# ----------------------------
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
        Specialized in making MCQs.\n
        First, input a subject and then answer with only the corresponding letter or number.\n
        You can reset this conversation or open a new page if you want a MCQ for another subject.
        """
    )
elif st.session_state.RAG_type == "create_questions":
    st.write("Ready to create questions and give corrections on the answer.")

# Process input when the user submits a query
if st.session_state.input:
    if st.session_state.RAG_type in ["open_questions", "create_questions"]:
        model_response = rag_model(st.session_state.model_id, client_inference, st.session_state.input, dict_chunks)
    elif st.session_state.RAG_type == "MCQ":
        model_response = rag_MCQ(st.session_state.model_id, client_inference, st.session_state.input, dict_chunks)
    
    st.session_state.interface_conv_history.append({
        'user': st.session_state.input,
        'model': model_response,
    })
    display_conversations()
else:
    display_conversations()

# User input box
user_input = st.text_input("Your Input:", key="user_input", on_change=submit)

# ----------------------------
# File Uploaders and RAG Options
# ----------------------------
RAG_choices = {
    "Answer questions": "open_questions", 
    "MCQs": "MCQ",
    "Create questions": "create_questions"
}