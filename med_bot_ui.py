import streamlit as st
from connect_llm_with_memory import answer_question, embedding_model, db  # Import the chatbot logic and resources

@st.cache_resource
def get_cached_embedding_model():
    """Cache the embedding model as a resource."""
    return embedding_model

@st.cache_resource
def get_cached_vector_db():
    """Cache the vector database as a resource."""
    return db

@st.cache_data
def cached_answer_question(question):
    """Cache the chatbot's response for a given question."""
    return answer_question(question)

def main():
    st.title("Medical ChatBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load cached resources
    embedding_model = get_cached_embedding_model()
    vector_db = get_cached_vector_db()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Enter your question:")
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from the cached chatbot logic
        response = cached_answer_question(prompt)

        # Display assistant response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


