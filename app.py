import streamlit as st

# Streamlit app header
st.title("Retrieval-Augmented Generation (RAG) - Base Interface")
st.write("This is a placeholder for the Retrieval-Augmented Generation system. The model will be integrated later.")

# Simple text input for user queries
query = st.text_input("Ask a question:")

# Display the query (you can replace this with model output later)
if query:
    st.write(f"You asked: {query}")
    st.write("Answer will be generated here once the model is integrated.")