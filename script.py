from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import timeit
import sys
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'c117f68d-32ef-444e-a34f-12ad1ce2bd59')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

#Downloading the Embeddings from Hugging Face

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


embeddings = download_hugging_face_embeddings()

#Initializing Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="llama2"



query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))
#Since, we already have an index, that's why we are loading that index from pinecone
docsearch=Pinecone.from_existing_index(index_name, embeddings)

query = "What is Hiplink"

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0})
qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 1}),return_source_documents=False, chain_type_kwargs=chain_type_kwargs)


user_input=input(f"Input Prompt:")
result=qa({"query": user_input})
print("Response : ", result["result"])

