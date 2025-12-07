from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that finds the weather.Find the city where its sunny.",
    ),
    ("human", "which place is sunny today?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)