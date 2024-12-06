import logging
import streamlit as st
import os 
import PyPDF2


from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema import Document
from ingest import initialize_vector_store
import chainlit as cl


# function to read the pdf file
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    all_page_text = ""
    for page in pdfReader.pages:
        all_page_text += page.extract_text() + "\n"
    return all_page_text


def retrieve_from_db(question):
   
    model = ChatOllama(model="llama3.2:3b")
    db = initialize_vector_store()

    retriever = db.similarity_search(question, k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )

    return after_rag_chain.invoke({"context": retriever, "question": question})


def retriever(doc, question):
    model_local = ChatOllama(model="llama3.2:3b")
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever = vectorstore.as_retriever(k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)






st.set_page_config(page_title="Your Law Assistant", layout="wide")


st.title("📜 Your Law Assistant")
st.write("💼 Drop your question about financial, commercial, or work law.")

st.sidebar.title("⚙️ Settings")
st.sidebar.write("Customize your experience.")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"])

st.header("📂 Upload Your Document")
file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Upload a PDF document to assist with your question.")

st.header("💬 Ask Your Question")
question = st.text_input("Type your question here:", placeholder="e.g., What are the labor laws in XYZ country?")

if file:
    st.success("✅ File uploaded successfully!")
    doc = read_pdf(file)  
    if st.button("Ask"):
        with st.spinner("Retrieving the answer..."):
            answer = retriever(doc, question)  
        st.subheader("📖 Answer")
        st.write(answer)
else:
    st.info("📂 No file uploaded. You can still ask general questions.")
    if st.button("Ask"):
        with st.spinner("Retrieving the answer..."):
            answer = retrieve_from_db(question)  
        st.subheader("📖 Answer")
        st.write(answer)

st.markdown("""
<style>
    .reportview-container {
        background-color: #f7f7f7;
        color: #000;
    }
    h1 {
        color: #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)