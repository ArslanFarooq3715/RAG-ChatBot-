#################
# RAG from simple application of the PDF

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.title("RAG ChatBot")

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_file = "/Users/af-home/Desktop/reflextion.pdf"
    
    # Load the PDF
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

prompt = st.chat_input("Pass your prompt here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name=model)
    
    # Define the system prompt template
    prompt_template = """You are an AI assistant that answers questions ONLY based on the provided context from the PDF document.
    
    IMPORTANT RULES:
    - If the answer is in the context, provide a clear and accurate response.
    - If the answer is NOT in the context, you MUST respond with: "I cannot answer this question based on the provided document."
    - Do NOT use any outside knowledge or make assumptions.
    - Do NOT answer questions that are not covered in the context.
    - Start the answer directly. No small talk please.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the document")
        else:
            
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            result = chain({'query': prompt})
            response = result['result']

            
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: [{e}]")