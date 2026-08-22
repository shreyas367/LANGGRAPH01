import os
from typing import TypedDict


# create state

class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llmgroq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# create a node 

def editor_node(state: PipelineState) -> PipelineState:
    """You are a professional editor who cleans up grammar, removes typos, and refines the tone of the text for a YouTube video script."""
    prompt = (
        f"Raw input: {state['raw_input']}\n\n"
        "Please edit the text for grammar, typos, and tone, and provide the polished version."
    )
    print("Stage 1 executing")
    response = llmgroq.invoke(prompt)
    state["edited_text"] = response.content.strip()
    return state



def script_node(state :PipelineState ) -> PipelineState:
    """ Convert edited text into a YouTube script. """
    prompt=(
        f"Edited text: {state['edited_text']}\n\n"
        "Please convert the edited text into a script for a youtube video and return the script text."
    )
    print("stage 2 executing SCRIPT CONVERSION")
    response=llmgroq.invoke(prompt)
    return {"script_text":response.content.strip()}


def translator_node(state :PipelineState ) -> PipelineState:
    """ Translate script text into Hinglish (India). """
    prompt=(        
        f"Script text: {state['script_text']}\n\n"
        "Please translate the script text into the hinglish of India and return the final output."
    )
    print("stage 3 executing TRANSLATION")
    response=llmgroq.invoke(prompt)
    return {"final_output":response.content.strip()}


# build graph
from langgraph.graph import StateGraph, START, END



graph = StateGraph(PipelineState)



graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",script_node)
graph.add_node("translator",translator_node)

# edge added between nodes to define the flow of the pipeline
graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)


# compile graph
app = graph.compile()



# invoke pipeline with a sample input

result = app.invoke({
    "raw_input": """
Hey guys welcome back to the channel. So today we are going to talk about Artificial Intelligence and why everyone is suddenly talking about AI. A lot of students think that AI means only ChatGPT but actually that is not true. AI is a very big field and ChatGPT is just one application of it.

If you are a college student and you want to get into AI then don't get confused by seeing hundreds of roadmaps on YouTube. Start with Python first because it is one of the easiest languages to learn for AI. After that understand basic machine learning concepts like supervised learning, unsupervised learning and neural networks. Don't try to learn everything in one day because it will only make you frustrated.

Then once your basics are clear start learning Large Language Models. Learn what prompts are, how tokens work, what temperature means and how APIs are used. These concepts are very important if you want to build modern AI applications.

After that start building projects. Don't just watch tutorials. Build a chatbot, a PDF question answering system, a resume analyzer or even your own AI assistant. Projects teach you much more than watching videos.

One mistake that I made when I started learning programming was waiting until I felt completely ready. The truth is nobody ever feels completely ready. You learn by building and making mistakes. Every bug teaches you something new.

Finally, if your goal is getting an internship or a job then focus on consistency. Spend one or two hours every day coding instead of studying ten hours on one day and then doing nothing for the next week. Small consistent progress always beats random motivation.

So that's all for today's video. If you enjoyed it don't forget to like the video, subscribe to the channel and share it with your friends who are also interested in learning AI. Thank you for watching and I'll see you in the next video.
"""
})


print(result["final_output"])






# from IPython.display import Image, display

# display(Image(app.get_graph().draw_mermaid_png()))
