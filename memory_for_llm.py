from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. load raw pdfs

def load_pdf(data):
    loader = DirectoryLoader(
        path=data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,  # Faster loading
        silent_errors=True        # Don't crash on one bad file
    )
    documents = loader.load()
    return documents

documents = load_pdf("data/")



# 2. create chunks

def create_chunks(loaded_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(loaded_data)
    return chunks

chunks = create_chunks(documents)


# 3.create embeddings for chunks

def get_embeddings_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embeddings_model()


# 4. STORE EMBEDDINGS IN FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(chunks,embedding_model) 
db.save_local(DB_FAISS_PATH)