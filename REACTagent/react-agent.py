from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)



search_tool = DuckDuckGoSearchRun()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Weather API key is missing."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": api_key,
                "units": "metric"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]

        return (
            f"Current weather in {city}: {description}, "
            f"{temperature}°C."
        )

    except requests.RequestException as e:
        return f"Weather request failed: {e}"
    except (KeyError, IndexError):
        return f"Unexpected weather response for {city}."

prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.

Available tools:
{tools}

Use this format:
Question: the question
Thought: decide what to do
Action: one of [{tool_names}]
Action Input: tool input
Observation: tool result
... repeat as needed
Thought: I have enough information
Final Answer: answer the question

Use get_weather for current weather.
Use duckduckgo_search for external factual information.
Use all necessary tools before answering multi-part questions.
Never invent observations.

Question: {input}
Thought:{agent_scratchpad}
""")

tools = [search_tool, get_weather]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_execution = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

response = agent_execution.invoke({
    "input": "What is the current weather in Surat and what is the capital of Gujarat?"
})

print("Final response:", response["output"])