from langchain_google_genai import ChatGoogleGenerativeAI
import os 

os.system("clear")
# Create an instance of the LLM, using the 'gemini-pro' model with a specified creativity level
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9, google_api_key="AIzaSyCFZhRAG5iELlbDdqOLKU2ggSk_AsOqjTQ")

print("Welcome! You can start chatting with the model. Type 'exit' to quit.\n")

while True:
    inp = input("You: ")
    if inp.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # Send the input to the LLM and get the response
    response = llm.invoke(inp)
    
    # Print the response content
    print("Model:", response.content)
    print("==============================================")