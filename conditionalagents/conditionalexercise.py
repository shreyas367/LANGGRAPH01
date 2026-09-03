from typing import Literal, TypedDict
from pathlib import Path

# from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph

class AgentState(TypedDict):
    number1:int
    operation1:str
    number2:int
    number3:int
    operation2:str
    number4:int
    finalNumber1:int
    finalNumber2:int
    
    
    
def addition_first(state: AgentState) -> AgentState:
    state["finalNumber1"] = state["number1"] + state["number2"]
    return state



def subtraction_first(state: AgentState) -> AgentState:
    state["finalNumber1"] = state["number1"] - state["number2"]
    return state



def addition_second(state: AgentState) -> AgentState:
    state["finalNumber2"] = state["number3"] + state["number4"]
    return state


def subtraction_second(state: AgentState) -> AgentState:
    state["finalNumber2"] = state["number3"] - state["number4"]
    return state


def decide_first(state: AgentState) -> Literal["additionedge", "subtractionedge"]:
    if state["operation1"] == "+":
        return "additionedge"
    if state["operation1"] == "-":
        return "subtractionedge"
    raise ValueError("operation1 must be '+' or '-'")


def decide_second(state: AgentState) -> Literal["additionedge", "subtractionedge"]:
    if state["operation2"] == "+":
        return "additionedge"
    if state["operation2"] == "-":
        return "subtractionedge"
    raise ValueError("operation2 must be '+' or '-'")




graph = StateGraph(AgentState)

# node adding
graph.add_node("addition_first", addition_first)
graph.add_node("subtraction_first", subtraction_first)
graph.add_node("addition_second", addition_second)
graph.add_node("subtraction_second", subtraction_second)

# route nodes
graph.add_node("first_route", lambda state: state)
graph.add_node("second_route", lambda state: state)





graph.add_edge(START, "first_route")

# conditional edge 1 added here
graph.add_conditional_edges(
    "first_route",
    decide_first,
    {
        "additionedge": "addition_first",
        "subtractionedge": "subtraction_first",
    }
)

graph.add_edge("addition_first", "second_route")
graph.add_edge("subtraction_first", "second_route")

# for second route conditional orute
graph.add_conditional_edges(
    "second_route",
    decide_second,
    {
        "additionedge": "addition_second",
        "subtractionedge": "subtraction_second",
    },
)

graph.add_edge("addition_second", END)
graph.add_edge("subtraction_second", END)

app = graph.compile()



# diagram_path = Path(__file__).with_name("conditionalexercise_graph.png")
# diagram_path.write_bytes(app.get_graph().draw_mermaid_png())
# print(app.get_graph().draw_mermaid())
# display(Image(filename=str(diagram_path)))
# print(f"Graph diagram saved to: {diagram_path.resolve()}")



initalstate1=AgentState(number1=33,number2=22,operation1="+",number3=88,number4=99,operation2="-")


response=app.invoke(initalstate1)
print(response)






