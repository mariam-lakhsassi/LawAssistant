# LawAssistant
this project is a law assistant wich is a chatbot that can answer questions related to the moroccan law more precisely: financial law, commercial law and labor law.
this law assistant can:
    Provide accurate answers to common legal questions
    Assist in preparing simple legal documents
    Help users understand their rights and obligations under Moroccan law

## the pipline of our project:
### Data Collection:
we used existing pdf files of moroccan laws and court decisions in the different gouvernemental resources:
  adala.justice.gov.ma
  uriscassation.cspj.ma
  juricaf.org
  cg.gov.ma
### Data Preprocessing
we used pdfplumber to extract text from PDFs and langchain.text_splitter to split large legal documents into smaller, manageable chunks
### Embedding Creation
Each chunk was embedded to create vector representations using the embedding model: mxbai-embed-large:latest
### Vector Database Creation
We used Chroma to create and persist the vector database
### Retrieval and Answer Generation
to retrieve the most relevant chunks of text from the vector database to the user's querry we used Chroma's similarity_search
to generate the answer to the user's querry we used the llama2.7:7b model
### Streamlit Interface
we developed a streamlit based user interface that allows:
  Uploading PDF files.
  Typing general legal questions.
  Viewing responses directly in the browser.

## How to Use
**Install requirements**:
   ```bash
   pip install -r requirements.txt
```
Place your legal documents in the ./documents directory
**Run the ingestion script to embed texts and create the vector database**:
   ```bash
   python ingest.py
```
**Start the chatbot application**:
   ```bash
   streamlit run LLM.py
```
