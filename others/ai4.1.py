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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Retrieve Data
def get_docs():
    loader = PDFPlumberLoader("path/to/your/ok.pdf")
    docs = loader.load()

    if isinstance(docs[0], str):  # If docs contain plain text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        splitDocs = text_splitter.split_documents(docs)
    else:  # If docs are already Page objects
        splitDocs = docs

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
    Answer the user's question.
    Context: {context}
    Question: {input}
    Previous Conversation:
    {chat_history}
    """)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = {}
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Example usage
session_id = "example_session"

response1 = conversational_rag_chain.invoke({
    "input": "What is LCEL?",
    "session_id": session_id
})

print(response1)

response2 = conversational_rag_chain.invoke({
    "input": "Can you explain it further?",
    "session_id": session_id
})

print(response2)
