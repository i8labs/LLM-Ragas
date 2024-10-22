from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Initialize an empty list to store the conversation history
conversation_history = []

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
    Answer the user's question based on the conversation history.
    Conversation history: {history}
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

# Function to append the chat history and get a response from the chain
def ask_question(chain, input_question):
    # Append the new input to the conversation history
    global conversation_history
    
    # Combine the conversation history into a single string (with a separator for clarity)
    history_str = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in conversation_history])
    
    # Invoke the chain with the conversation history and the new input
    response = chain.invoke({
        "input": input_question,
        "history": history_str
    })
    
    # Save the input and response to the conversation history
    conversation_history.append({
        "user": input_question,
        "ai": response['answer']  # Assuming response['output'] holds the model's answer
    })

    return response

# Load the documents from the PDF
docs = get_docs()

# Create the vector store from the split documents
vectorStore = create_vector_store(docs)

# Create the chain for retrieving answers from the vector store
chain = create_chain(vectorStore)

# Example: Ask a question and get a response
response = ask_question(chain, "who is Paras?")
print(response)
print("--------------------------------------------------")
# Ask another question with the history included
response = ask_question(chain, "What was the last question i asked you about?")
print(response)
