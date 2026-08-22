import ast
import os
from typing import TypedDict,Annotated
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)



def merge_score_dicts(dict1: dict[str,int], dict2: dict[str,int]) -> dict[str,int]:
    """ Merge two dictionaries of safety scores by taking the maximum score for each key. """
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] = max(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict



class AnalyzerState(TypedDict):
    raw_input: str
    safety_score: Annotated[dict[str,int],merge_score_dicts]


def parse_score(response_text: str) -> int:
    """Parse either a plain integer or a dictionary-style response into an integer score."""
    cleaned = response_text.strip()

    try:
        return int(cleaned)
    except ValueError:
        pass

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, (int, float)):
                    return int(value)
        elif isinstance(parsed, (int, float)):
            return int(parsed)
    except (ValueError, SyntaxError):
        pass

    digits = "".join(character for character in cleaned if character.isdigit())
    if digits:
        return int(digits)

    raise ValueError(f"Could not parse a score from: {response_text}")


# nodes

def toxicity_node(state: AnalyzerState) -> AnalyzerState:
    """ Analyze the raw input for toxicity and hate speech and return a safety score. """
    prompt = (
        f"Raw input: {state['raw_input']}\n\n"
        "Please analyze the text for toxicity and hate speech and return a score from 0 -100 ,where 0 means perfectly clean and 100 means highly toxic. Return the score in a dictionary format like {'toxicity': 0, 'hate_speech': 0}. return only plain interger number ,do not include any text or explanation."
    )
    print("stage 1 executing TOXICITY ANALYSIS")
    response = llm.invoke(prompt)
    return {"safety_score": {"toxicity_level": parse_score(response.content)}}


def copyright_node(state: AnalyzerState) -> AnalyzerState:
    """ Analyze the raw input for copyright infringement and return a safety score. """
    prompt = (
        f"Raw input: {state['raw_input']}\n\n"
        "Please analyze the text for copyright infringement and return a score from 0 -100 ,where 0 means perfectly clean and 100 means highly infringing. Return the score in a dictionary format like {'copyright_infringement': 0}. return only plain interger number ,do not include any text or explanation."
    )
    print("stage 2 executing COPYRIGHT ANALYSIS")
    response = llm.invoke(prompt)
    return {"safety_score": {"copyright_infringement": parse_score(response.content)}}



def culture_node(state: AnalyzerState) -> AnalyzerState:
    """ Analyze the raw input for cultural sensitivity and return a safety score. """
    prompt = (
        f"Raw input: {state['raw_input']}\n\n"
        "Please analyze the text for cultural sensitivity and return a score from 0 -100 ,where 0 means perfectly clean and 100 means highly insensitive. Return the score in a dictionary format like {'cultural_insensitivity': 0}. return only plain interger number ,do not include any text or explanation."
    )
    print("stage 3 executing CULTURAL SENSITIVITY ANALYSIS")
    response = llm.invoke(prompt)
    return {"safety_score": {"cultural_insensitivity": parse_score(response.content)}}
 
 
graph=StateGraph(AnalyzerState)


graph.add_node("toxicity",toxicity_node)
graph.add_node("copyright",copyright_node)
graph.add_node("culture",culture_node)


graph.add_edge(START,"toxicity")
graph.add_edge(START,"copyright")
graph.add_edge(START,"culture")

graph.add_edge("toxicity",END)
graph.add_edge("copyright",END)
graph.add_edge("culture",END)




app=graph.compile()

raw_input = """
I hate these people so much. They are useless and deserve nothing but pain. Everyone should be ashamed of themselves and get out of my way. This is the kind of garbage that should be destroyed and erased from society.
"""

result = app.invoke({
    "raw_input": raw_input,
    "safety_score": {}
})


print(result["safety_score"])