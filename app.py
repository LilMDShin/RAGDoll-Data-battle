import streamlit as st

# Placeholder for model functionality
def rag_model(input_text):
    # Placeholder for the actual RAG model implementation
    # For now, just return the same text back or a sample response.
    return f"This is a placeholder for query: {input_text}"

# Display conversation history
def display_conversations():
    for conversation in st.session_state.conversations:
        st.markdown(f"**User**: {conversation['user']}")
        st.markdown(f"**Model**: {conversation['model']}")

def submit():
    st.session_state.input = st.session_state.user_input
    st.session_state.user_input = ''

st.set_page_config(page_title="RAG Model", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) Base")

# Initialize session state for conversations if it doesn't exist yet
if 'conversations' not in st.session_state:
    st.session_state.conversations = []

if 'input' not in st.session_state:
    st.session_state.input = ''

# Process input when the user submits
if st.session_state.input:
    # Call the RAG model (placeholder for now)
    model_response = rag_model(st.session_state.input)

    # Store the conversation in session state
    st.session_state.conversations.append({
        'user': st.session_state.input,
        'model': model_response
    })

    display_conversations()

# Input box at the bottom
user_input = st.text_input("Your Input:", key="user_input", on_change=submit)
