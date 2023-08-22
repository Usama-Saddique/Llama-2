import pinecone
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Now we will define a function to load the document
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents

extracted_data = load_pdf_file(data='data/')


# Now we will define a function to break document into chunks
# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))
