import sys
import os

# Disable Telemetry Posting for Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "false"


import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma

st.set_page_config(page_title="Chat Application")
st.header("Chat :blue[Application]")

st.chat_message("ai").write("Hello!! Welcome to your private Gen AI Bot")
prompt = st.chat_input("Add your prompt here")
ollama = Ollama(base_url='http://localhost:11434', model='tinyllama')
vector_db = Chroma(persist_directory="/root/ollama/chromadb", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),collection_name="jay-rag2")
qachain = RetrievalQA.from_chain_type(ollama, retriever=vector_db.as_retriever(),return_source_documents=True)

if prompt:
     st.chat_message("user").write(prompt)
     print(prompt)
     output = qachain.invoke({"query": prompt})
     st.chat_message("ai").write(output["result"])
