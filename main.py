from dotenv import load_dotenv

load_dotenv()
from langchain.agents import initialize_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq


tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching for {query}")
    return tavily.search(query)


# llm = ChatOllama(temperature=0,model="llama3:8b")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)
tools = [search]
agent = initialize_agent(tools=tools,llm=llm,handle_parsing_errors=True)

def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})

    # response = agent.invoke([
    #     HumanMessage(content="What is the weather in Tokyo?")
    # ])
    response = agent.invoke({"input": "What is the weather in Tokyo?"})
    print(response)


if __name__ == "__main__":
    main()
