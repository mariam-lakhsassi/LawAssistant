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
    # get the model
    model = ChatOllama(model="llama3.2:3b")
    # initialize the vector store
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





# Chainlit integration
@cl.on_chat_start
async def chat_start():
    await cl.Message(content="Welcome! Please upload a PDF file to start or directly ask a question.").send()

""" @cl.file_upload(name="Upload a PDF", accept=["application/pdf"])
async def on_file_upload(file: cl.File):
    # Save the file content to the session
    file_path = f"./uploaded_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read and process the file
    doc_text = read_pdf(file_path)
    cl.user_session.set("doc", doc_text)

    await cl.Message(content=f"File '{file.name}' uploaded successfully! You can now ask questions based on this document.").send() """

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content

    try:
        answer = retrieve_from_db(question)
        logging.debug(f"Answer generated: {answer}") 
    except Exception as e:
        print(f"Error during retrieval: {e}")

    # Send the answer back to the user
    await cl.Message(content=answer).send()
"""     doc = cl.user_session.get("doc")

    # Decide whether to retrieve from the uploaded document or the database
    if doc:
        answer = retriever(doc, question)
        else """