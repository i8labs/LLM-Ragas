from dotenv import load_dotenv
load_dotenv()
from os import system
system("clear")

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
)

# Retrieve Data from txt
def get_docs():
    loader = TextLoader('nyc.txt')
    # loader = WebBaseLoader(web_path="https://hpshimla.nic.in/about-district/#:~:text=The%20Shimla%20back%20to%20the,civil%20servant%20Charles%20Pratt%20Kennedy.")
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

docs = get_docs()
vectorStore = create_vector_store(docs)

# loader = TextLoader("nyc.txt")
# index = VectorstoreIndexCreator().from_loaders([loader])


llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorStore.as_retriever(),
    return_source_documents=True,
)


question = "Whats is shiml's Altitude"
result = qa_chain({"query": question})
print(result['result'])