from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st
from utils.chunks import load_index_and_chunks, retrieve_chunks
from utils.repair_pdf import repair_pdf
from utils.preprocess import process_pdf_to_json, process_html_to_json
from utils.faiss_index import update_index
from utils.rag import rag_stream, rag
from dotenv import load_dotenv
import re
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

def rag_generic(model_id, client_inference, query, dict_chunks, system_prompt, include_context_on_history=True, streaming=True):
    context = get_context(query, dict_chunks, embedding_model)
    if not st.session_state.model_conv_history:
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
        if include_context_on_history:
            st.session_state.model_conv_history.append({"role": "system", "content": f"Additional context : {context}"})
        st.session_state.model_conv_history.append({"role": "user", "content": query})
    
    # Get the response in streaming mode using the generator
    if streaming:
        response_generator = rag_stream(model_id, client_inference, st.session_state.model_conv_history)
        st.markdown(f"**User**: {query}")
        placeholder = st.empty()  # This placeholder is updated in real time
        full_response = ""
        for token in response_generator:
            full_response = token  # token contains the accumulation of the response
            placeholder.markdown("**Model**: " + full_response)
    else :
        full_response = rag(model_id, client_inference, st.session_state.model_conv_history)

    st.session_state.model_conv_history.append({"role": "assistant", "content": full_response})
    
    return full_response

def rag_model(model_id, client_inference, query, dict_chunks):
    if st.session_state.RAG_type == "open_questions":
        prompt = """You are a world-class expert in patent law with deep knowledge of both domestic and international intellectual property statutes, regulations, and case law. Your role is to deliver comprehensive, detailed answers to inquiries on patent law. Each response should include:
                    - A clear, step-by-step explanation of the relevant legal principles and their applications.
                    - Specific examples or case references where applicable.
                    - Citations that clearly list the names of the authoritative source documents (such as legislation, judicial opinions, or legal commentaries) used to support your answer. If multiple sources are used, list each one separately.

                    Your answers should be precise, logically structured, and written in plain language to ensure clarity and ease of verification by the user.
                 """

    elif st.session_state.RAG_type == "create_questions":
        prompt = (
            "You are a world-class expert in patent law with exceptional skills in crafting challenging questions on the subject. "
            "Your task is to ask an engaging question related to patent law based on the current conversation context (which is saved in cache), and then wait for the user's response. "
            "Do not provide the answer immediately. Once the user submits an answer, prompt them to confirm if they believe their answer is correct or need further clarification. "
            "If the user indicates uncertainty or provides an incorrect answer, deliver a detailed explanation of the correct answer, including step-by-step reasoning and citing your main authoritative sources (with the document names) to support your explanation."
        )
    return rag_generic(model_id, client_inference, query, dict_chunks, prompt, include_context_on_history=True)

def rag_MCQ(model_id, client_inference, query, dict_chunks):
    prompt = (
        "You are a world-class expert in patent law with exceptional skills in designing challenging multiple-choice questions (MCQs) on this subject. "
        "Your task is to generate an MCQ that may include one or more correct answers, with a minimum of four answer options. "
        "Please strictly follow the structure below:\n\n"
        "Question: <MCQ question text>\n"
        "Options:\n"
        "A) <Option A text>\n"
        "B) <Option B text>\n"
        "C) <Option C text>\n"
        "D) <Option D text>\n"
        "Answer: <Option letter(s)>\n"
        "Explanation: <A detailed explanation of why the answer(s) is/are correct, including relevant legal principles and citing key documents>\n"
        "Sources: <List of main sources (document names)>\n\n"
    )

    # For MCQs, we do not include additional context when conversation already exists
    return rag_generic(model_id, client_inference, query, dict_chunks, prompt, include_context_on_history=False, streaming=False)

def callback_mcq(user_choice, old):
    if not old:
        st.session_state[f"{st.session_state.nb_mcq}"] = user_choice
        st.session_state.nb_mcq += 1
    

def display_mcq(mcq, key, old):
    st.write("### MCQ:")
    st.markdown(f"**Question:** {mcq.get('question', 'No question provided')}")
    
    with st.form(key=f"mcq_form_" + key):
        if old :
            if st.session_state[f"{st.session_state.nb_mcq-1}"] == "A":
                index = 0
            elif st.session_state[f"{st.session_state.nb_mcq-1}"] == "B":
                index = 1
            elif st.session_state[f"{st.session_state.nb_mcq-1}"] == "C":
                index = 2
            else: 
                index = 3
            user_choice = st.radio(
                "Select your answer:",
                options=list(mcq.get("options", {}).keys()),
                format_func=lambda opt: f"{opt}) {mcq['options'][opt]}",
                disabled=old,
                index=index
            )
        else:
            user_choice = st.radio(
                "Select your answer:",
                options=list(mcq.get("options", {}).keys()),
                format_func=lambda opt: f"{opt}) {mcq['options'][opt]}",
                disabled=old
            )
        submitted = st.form_submit_button("Submit Answer", on_click=callback_mcq, args=(user_choice,old) ,disabled=old)
        correct_answer = mcq.get("hidden_answer", "").upper()
        if old: 
            if st.session_state[f"{st.session_state.nb_mcq-1}"].upper() == correct_answer[0]:
                result_text = "Correct Answer!"
            else:
                result_text = "Incorrect Answer."
            st.subheader("Result")
            if result_text == "Correct Answer!":
                st.success(result_text)
            else:
                st.error(result_text)
            st.write(mcq.get("hidden_answer", "") + ") " + mcq['options'][mcq.get("hidden_answer", "")])
            st.subheader("Explanation")
            st.write(mcq.get("hidden_explanation", "No explanation available."))
            st.subheader("Sources")
            st.write(mcq.get("hidden_sources", "No sources available."))
    return 

def display_conversations():
    for i in range(len(st.session_state.interface_conv_history)):
        if st.session_state.RAG_type == "MCQ":
            display_mcq(st.session_state.interface_conv_history[i]['model'], f"{i}", True)
        else :    
            st.markdown(f"**User**: {st.session_state.interface_conv_history[i]['user']}")
            st.markdown(f"**Model**: {st.session_state.interface_conv_history[i]['model']}")

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
    if 'mcq_choice' not in st.session_state:
        st.session_state.mcq_choice = []
    if 'nb_mcq' not in st.session_state:
        st.session_state.nb_mcq = 0

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
        st.session_state.language_choice = st.session_state.selBox_lang_choice
        reset_conv()

def parse_mcq(mcq_text):
    mcq_data = {}

    # Extract question: capture text between "Question:" and "Options:"
    question_match = re.search(r"Question:\s*(.*?)\s*Options:", mcq_text, re.DOTALL)
    if question_match:
        mcq_data["question"] = question_match.group(1).strip()
    else:
        mcq_data["question"] = ""

    # Extract options: capture text between "Options:" and "Answer:" then find each option (A-D)
    options_match = re.search(r"Options:\s*(.*?)\s*Answer:", mcq_text, re.DOTALL)
    options = {}
    if options_match:
        options_text = options_match.group(1)
        # Find options that start with a letter (A-D) followed by ')'
        for key, value in re.findall(r"([A-D])\)\s*(.*)", options_text):
            options[key] = value.strip()
    mcq_data["options"] = options

    # Extract hidden answer: capture text between "Answer:" and "Explanation:"
    answer_match = re.search(r"Answer:\s*(.*?)\s*Explanation:", mcq_text, re.DOTALL)
    if answer_match:
        mcq_data["hidden_answer"] = answer_match.group(1).strip()
    else:
        mcq_data["hidden_answer"] = ""

    # Extract hidden explanation: capture text between "Explanation:" and "Sources:"
    explanation_match = re.search(r"Explanation:\s*(.*?)\s*Sources:", mcq_text, re.DOTALL)
    if explanation_match:
        mcq_data["hidden_explanation"] = explanation_match.group(1).strip()
    else:
        mcq_data["hidden_explanation"] = ""

    # Extract hidden sources: capture text after "Sources:"
    sources_match = re.search(r"Sources:\s*(.*)", mcq_text, re.DOTALL)
    if sources_match:
        mcq_data["hidden_sources"] = sources_match.group(1).strip()
    else:
        mcq_data["hidden_sources"] = ""

    return mcq_data

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

conversation_container = st.container()
with conversation_container:
    display_conversations()

# Process input when the user submits a query
if st.session_state.input:
    # Capture the user query before resetting
    user_query = st.session_state.input
    preprompt = "Make a MCQ for " + user_query
    if st.session_state.RAG_type == "MCQ":
        # Only call the model if we haven't generated an MCQ yet for this query
        st.session_state.mcq_result = rag_MCQ(
            st.session_state.model_id,
            client_inference,
            preprompt,
            dict_chunks
        )
        st.session_state.mcq = parse_mcq(st.session_state.mcq_result)
        # Use the stored MCQ to build the UI (with a consistent form key)
        model_response = st.session_state.mcq
        display_mcq(st.session_state.mcq, "mcq_form", False)
    else:
        # For other RAG types (open or create questions)
        model_response = rag_model(st.session_state.model_id, client_inference, user_query, dict_chunks)
    
    # Append the conversation history using the captured user_query
    st.session_state.interface_conv_history.append({
        'user': user_query,
        'model': model_response,
    })
    # Now clear the input to avoid re-calling the model on every re-run
    st.session_state.input = ""

# User input box
user_input = st.text_input("Your Input:", key="user_input")

st.button("Submit", on_click=submit)