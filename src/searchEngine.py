from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

# Load PDF and extract text
pdf_path = "../Documents/llm.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_chunks = [doc.page_content for doc in docs]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS
vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)

print("Vectors stored successfully!")


query = "Who Is This Book For?"
retrieved_docs = vector_store.similarity_search(query, k=1)  # Get top 2 matches

for i, doc in enumerate(retrieved_docs):
    print(f"Match {i+1}: {doc.page_content[:500]}")