import random
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

class stateagent(TypedDict):
    greeting:str
    counters:list[int]
    number:int
    

def greetingfunction(state: stateagent) -> stateagent:
    """greeting by name """
    state["greeting"] = f"hello there {state['greeting']}"
    state["number"] = 0
    return state
    

def random_node(state: stateagent) -> stateagent:
    
    """adding in counter list anad increment number"""
    state["counters"].append(random.randint(1, 13))
    state["number"] += 1
    return state



def should_loop(state: stateagent) -> Literal["loopedge", "exit"]:
    """ FUNCTION TO DECIDE TO LOOP OR NOT"""
    if state["number"] < 8:
        print("ENTERING LOOP", state["number"])
        return "loopedge"
    
    else:
        return "exit"
    
    
    
    
    
    
graph = StateGraph(stateagent)

graph.add_node("greeting", greetingfunction)
graph.add_node("random", random_node)


graph.add_edge(START, "greeting")
graph.add_edge("greeting", "random")

graph.add_conditional_edges(
    "random",
    should_loop,
    {
        "loopedge": "random",
        "exit": END,
    },
)


app = graph.compile()

result = app.invoke({
    "greeting": "SHREYAS",
    "counters": [],
    "number": 0,
})

print(result)