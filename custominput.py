from dotenv import load_dotenv
load_dotenv()
from os import system
system("clear")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader,TextLoader  # Using PyPDFLoader for PDF extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback

# Initialize chat history
chat_history = []

# Predefined set of questions
predefined_questions = [
    "What is the current capital of Himachal Pradesh?",
    "What was the summer capital of British India?",
    "Which temple was Shimla most popular for during the 19th century?",
    "Who constructed the first British summer home in Shimla?",
    "In what year was the Kalka-Shimla railway line constructed?",
]

# Retrieve Data from PDF
def get_docs():
    # loader = PyPDFLoader('ok.pdf')  # Load the PDF file
    loader = TextLoader('nyc.txt')  # Load the PDF file
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

    prompt_template = ChatPromptTemplate.from_template("""
    Answer the asked questions based on the context provided. Here is the conversation history:
    {history}
    
    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template
    
    )

    retriever = vectorStore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def ask_question(chain, question):
    # Prepare the conversation history for the prompt
    history_text = "\n".join(chat_history)
    with get_openai_callback() as cb:
        response = chain.invoke({
            "input": question,
            "history": history_text  # Pass the chat history to the model
        })
    print("--------------------------------------")    
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")    
    print("--------------------------------------")    
    
    # Store the question and response in chat history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {response}")

    return response

# Load documents and create vector store and chain
docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

# Only ask predefined questions
answers_list = []
for question in predefined_questions:
    answer = ask_question(chain, question)
    answers_list.append(answer['answer'])

# Display answers in a list format
print("\nAnswers:")
for i, ans in enumerate(answers_list, 1):
    print(f"{i}. {ans}")


print(answers_list)