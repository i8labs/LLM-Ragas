from dotenv import load_dotenv
load_dotenv()

from os import system
system("clear")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Retrieve Data from a PDF
def get_docs():
    # Load the PDF using PDFPlumberLoader
    loader = PDFPlumberLoader('ok.pdf')  # Change this path to the location of your PDF
    docs = loader.load()

    # Split the documents into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()  # Initialize OpenAI embeddings
    vectorStore = FAISS.from_documents(docs, embedding=embedding)  # Create FAISS vector store
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    # Create document chain using the prompt and LLM
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Create a retriever using the vector store
    retriever = vectorStore.as_retriever()

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Load the documents from the PDF
docs = get_docs()

# Create the vector store from the split documents
vectorStore = create_vector_store(docs)

# Create the chain for retrieving answers from the vector store
chain = create_chain(vectorStore)

# Get a response to a specific question based on the PDF content
response = chain.invoke({
    "input": "What are Paras's Hobbies",
})

# Print the response
print(response)
