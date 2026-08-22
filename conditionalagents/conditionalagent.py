from typing import TypedDict
from langgraph.graph import StateGraph,START,END

class agentstate(TypedDict):
    number1:int
    operation:str
    number2:int
    finalNumber:int



def adder(state:agentstate) ->agentstate:
    """ this node is to add 2 number"""
    state['finalNumber']=state['number1']+state['number2']
    
    return state
    
    
def subtration(state:agentstate) ->agentstate:
    "subtrate the numbers"
    state['finalNumber']=state['number1']-state['number2']
    
    return state    



def decide_next_node(state:agentstate) -> str:
    """this node decide whether to add or subtract """
    if state['operation'] == "+":
        return "additionedge"

    if state['operation'] == "-":
        return "subtractionedge"
    

    
    
graph=StateGraph(agentstate)

graph.add_node("Add_node",adder)
graph.add_node("subtract_node",subtration)
graph.add_node("routernode",lambda state:state)




# edge adding
graph.add_edge(START,"routernode")


# conditional edge
graph.add_conditional_edges(
    "routernode",
    decide_next_node,
    {
        "additionedge": "Add_node",
        "subtractionedge": "subtract_node",
    }
)

graph.add_edge("Add_node",END)
graph.add_edge("subtract_node",END)

app=graph.compile()


initial_state1=agentstate(number1=12,operation="-",number2=22)

print(app.invoke(initial_state1))



    