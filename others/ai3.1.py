from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Initialize conversation history
conversation_history = []

# Retrieve Data from PDF
def get_docs():
    loader = PyPDFLoader('ok.pdf')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Previous Conversations: {history}
    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Load documents, create vector store and chain
docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

# Function to handle conversation
def ask_question(question):
    global conversation_history
    # Prepare the history string
    history = "\n".join(conversation_history)
    
    # Invoke the chain with the question and history
    response = chain.invoke({
        "input": question,
        "history": history
    })
    
    # Store the question and response in history
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"AI: {response['output']}")
    
    return response['output']

# Example usage
response = ask_question("What is LCEL?")
print(response)

# You can continue asking questions
response = ask_question("Can you explain more about its applications?")
print(response)
