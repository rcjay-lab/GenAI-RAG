import os
# Disable Telemetry Posting for Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
#from LoadProperties import LoadProperties
from langchain_community.embeddings import OllamaEmbeddings

# load and split documents
pdf_loader = PyPDFDirectoryLoader("./pdf-docs" )
loaders = [pdf_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_documents = text_splitter.split_documents(documents)
print(f"Total number of documents: {len(all_documents)}")
vector_db = Chroma.from_documents(documents=all_documents, embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True), collection_name="jay-rag2", persist_directory = "./chromadb")
