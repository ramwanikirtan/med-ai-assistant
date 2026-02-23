# Medical ChatBot

## Overview
The Medical ChatBot is a Streamlit-based application designed to provide evidence-based medical information to users. It leverages advanced natural language processing (NLP) techniques, including retrieval-augmented generation (RAG), to answer user queries with high accuracy and reliability. The chatbot integrates with a local FAISS vector database for efficient document retrieval and uses a pre-trained language model from OpenAI to generate responses.

---

## Features
1. **Interactive Chat Interface**: A user-friendly Streamlit interface for seamless interaction with the chatbot.
2. **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with language generation to provide accurate and contextually relevant answers.
3. **Local Vector Database**: Utilizes FAISS for efficient storage and retrieval of document embeddings.
4. **Caching Mechanism**: Implements Streamlit's `@st.cache_resource` and `@st.cache_data` to optimize performance by caching resources and responses.
5. **Evidence-Based Responses**: The chatbot strictly adheres to the provided context and avoids making unsupported claims or guesses.
6. **Source Attribution**: Includes detailed source information (e.g., document name and page number) for every response.

---

## Technologies Used

### 1. **Streamlit**
   - A Python-based framework for building interactive web applications.
   - Used to create the chatbot's user interface, enabling real-time interaction with users.

### 2. **LangChain**
   - A framework for building applications powered by large language models (LLMs).
   - Used to implement the retrieval-augmented generation (RAG) pipeline.

### 3. **OpenAI GPT-3.5-turbo**
   - A state-of-the-art large language model developed by OpenAI.
   - Used to generate natural language responses based on the retrieved context and user queries.

### 4. **HuggingFace Transformers**
   - Provides the `sentence-transformers/all-MiniLM-L6-v2` model for generating embeddings.
   - This model is lightweight and optimized for semantic similarity tasks, making it ideal for document retrieval.

### 5. **FAISS (Facebook AI Similarity Search)**
   - A library for efficient similarity search and clustering of dense vectors.
   - Used as the vector database to store and retrieve document embeddings.

### 6. **Python**
   - The primary programming language used for the project.
   - Libraries used include `streamlit`, `langchain`, `sentence-transformers`, and `faiss`.

### 7. **Environment Management**
   - The project is developed in a Conda environment named `genai` to manage dependencies and ensure compatibility.

---

## Project Structure
```
Medical ChatBot/
├── connect_llm_with_memory.py  # Core chatbot logic and RAG pipeline
├── med_bot_ui.py               # Streamlit-based user interface
├── data/                       # Directory for storing raw data
├── vectorstore/                # Directory for FAISS vector database
│   └── db_faiss/               # FAISS index files
```

---

## How It Works

1. **Document Embedding and Storage**:
   - The chatbot uses the `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace to generate embeddings for the documents.
   - These embeddings are stored in a FAISS vector database for efficient similarity search.

2. **Query Processing**:
   - When a user enters a question, the chatbot uses a custom `LocalMultiQueryRetriever` to generate multiple query variations.
   - These queries are used to retrieve relevant documents from the FAISS vector database.

3. **Response Generation**:
   - The retrieved documents are passed to a pre-trained OpenAI language model (`gpt-3.5-turbo`) along with a custom prompt.
   - The model generates a structured response that includes the answer, supporting evidence, limitations, and confidence level.

4. **Streamlit Integration**:
   - The `med_bot_ui.py` file provides a simple and interactive chat interface for users to interact with the chatbot.
   - User queries and chatbot responses are displayed in a chat-like format.

5. **Caching**:
   - The application uses Streamlit's `@st.cache_resource` to cache the embedding model and vector database.
   - Responses to user queries are cached using `@st.cache_data` to improve performance for repeated questions.

---

## How to Run the Application

1. **Install Dependencies**:
   Ensure you have Python installed on your system. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Vector Database**:
   - Place your documents in the `data/` directory.
   - Use the FAISS library to generate embeddings and store them in the `vectorstore/db_faiss/` directory.

3. **Run the Streamlit Application**:
   Start the Streamlit app by running the following command in your terminal:
   ```bash
   streamlit run med_bot_ui.py
   ```

4. **Interact with the ChatBot**:
   - Open the provided local URL in your browser (e.g., `http://localhost:8501`).
   - Enter your medical questions in the chat input box and receive evidence-based answers.

---

## Future Improvements

1. **Enhanced Data Sources**:
   - Integrate additional medical datasets to improve the chatbot's knowledge base.
   - Add support for real-time data updates from trusted medical APIs.

2. **Advanced Query Understanding**:
   - Implement more sophisticated query understanding techniques to handle complex medical questions.
   - Add support for multi-turn conversations to maintain context across multiple queries.

3. **User Authentication**:
   - Add user authentication to provide personalized experiences and save user history.

4. **Mobile Optimization**:
   - Optimize the Streamlit app for mobile devices to improve accessibility.

5. **Improved Caching**:
   - Implement a more robust caching mechanism to handle larger datasets and improve response times.

6. **Integration with External Tools**:
   - Add integration with external tools like electronic health records (EHR) systems for more personalized responses.

7. **Deployment**:
   - Deploy the application to a cloud platform (e.g., AWS, Azure, or Google Cloud) for wider accessibility.

---

## Contributing
Contributions are welcome! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the interactive web app framework.
- [LangChain](https://www.langchain.com/) for the retrieval-augmented generation pipeline.
- [HuggingFace](https://huggingface.co/) for the pre-trained embedding model.
- [FAISS](https://faiss.ai/) for the vector database.
- [OpenAI](https://openai.com/) for the GPT-3.5-turbo language model.