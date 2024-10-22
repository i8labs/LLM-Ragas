from dotenv import load_dotenv
load_dotenv()
from os import system
system("clear")

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import DirectoryLoader,TextLoader
loader = TextLoader("nyc.txt")
documents = loader.load()


# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
critic_llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings,
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=3, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})