.. LawAssistant documentation master file, created by
   sphinx-quickstart on Tue Dec  3 16:03:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LawAssistant documentation
==========================

**supervised by:** M.MASROUR

**Realised by:** Lakhsassi Mariam and Jhabli Hassna


the link to our github repositry: `<https://github.com/mariam-lakhsassi/LawAssistant.git>`_.

=================
Introduction
=================

Our law assistant can:

* Provide accurate answers to common legal questions about 
* Assist in preparing simple legal documents
* Help users understand their rights and obligations under Moroccan law

=================
The Pipeline of Our Project
=================

*Data Collection:*

We used existing PDF files of Moroccan laws and court decisions from the following government resources:

* adala.justice.gov.ma
* uriscassation.cspj.ma
* juricaf.org
* cg.gov.ma

*Data Preprocessing:*

We used pdfplumber to extract text from PDFs and langchain.text_splitter to split large legal documents into smaller, manageable chunks.

*Embedding Creation:*

Each chunk was embedded to create vector representations using the embedding model: mxbai-embed-large:latest.

*Vector Database Creation:*

We used Chroma to create and persist the vector database.

*Retrieval and Answer Generation:*

We used Chroma's similarity_search to retrieve the most relevant chunks of text from the vector database for the user's query. 

The answer to the user's query is generated using the llama2.7:7b model.

*Streamlit Interface:*

We developed a Streamlit-based user interface that allows:

* Uploading PDF files.
* Typing general legal questions.
* Viewing responses directly in the browser.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

