import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

worddoc = "./docs/test_rag.docx"


def load_doc_to_db():
    docs = []
    
    # load document
    loader = Docx2txtLoader(worddoc)
    docs.extend(loader.load())
    print("=============================")
    print(docs)

    # split docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)
    print("=============================")
    print(document_chunks)

    print("=============================")
    print("init chroma")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory='./db',
    )
    print("=============================")
    print("done chroma")



load_doc_to_db()