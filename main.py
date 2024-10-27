import gradio as gr

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

from State import State
from nodes import cal_final_score, check_depth, check_grammar, check_relevance, check_structure, generate_natural_language_feedback

load_dotenv()




# Initialize the LLMs
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
llm2 =ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="mixtral-8x7b-32768") 




# initialize graph
workflow = StateGraph(State)

# add nodes
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("check_structure", check_structure)
workflow.add_node("check_depth", check_depth)
workflow.add_node("cal_final_score", cal_final_score)

# define and add conditional edges
workflow.add_conditional_edges(
    "check_relevance",
    lambda x: "check_grammar" if x["relevance_score"] >= 0.5 else "cal_final_score",
)

workflow.add_conditional_edges(
    "check_grammar",
    lambda x: "check_structure" if x["grammar_score"] >= 0.6 else "cal_final_score",
)

workflow.add_conditional_edges(
    "check_structure",
    lambda x: "check_depth" if x["structure_score"] >= 0.7 else "cal_final_score",
)

workflow.add_conditional_edges(
    "check_depth",
    lambda x: "cal_final_score",
)

# entry point
workflow.set_entry_point("check_relevance")

# exit point
workflow.add_edge("cal_final_score", END)

# compile the graph
app = workflow.compile()


def grade_essay(essay: str) -> dict:
    """Grade the given essay and generate scores and feedback."""
    initial_state = State(
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0
    )

    # Sequentially apply grading functions
    initial_state = check_relevance(initial_state)
    initial_state = check_grammar(initial_state)
    initial_state = check_structure(initial_state)
    initial_state = check_depth(initial_state)
    initial_state = cal_final_score(initial_state)

    # Generate natural language feedback
    feedback = generate_natural_language_feedback(initial_state)
    
    return {
        "relevance_score": initial_state["relevance_score"],
        "grammar_score": initial_state["grammar_score"],
        "structure_score": initial_state["structure_score"],
        "depth_score": initial_state["depth_score"],
        "final_score": initial_state["final_score"],
        "feedback": feedback
    }




def generate_follow_up_response(state: State, question: str) -> str:
    """Generate a response to a follow-up question using llm2."""
    prompt = (
        f"The user has received feedback for their essay with the following scores:\n\n"
        f"Relevance: {state['relevance_score']:.2f}\n"
        f"Grammar: {state['grammar_score']:.2f}\n"
        f"Structure: {state['structure_score']:.2f}\n"
        f"Depth: {state['depth_score']:.2f}\n\n"
        f"Final Score: {state['final_score']:.2f}\n\n"
        f"The user asks: '{question}'\n\n"
        f"Please respond to their question based on the provided scores and feedback."
    )

    result = llm2.invoke(prompt)
    
    if hasattr(result, "content") and result.content:
        return result.content

    # Handle cases where result is a string or dictionary with 'generated_text' key
    if isinstance(result, str):
        return result
    elif isinstance(result, dict) and 'generated_text' in result:
        return result['generated_text']

    # Fallback for unexpected result structure
    return "Unexpected response structure. Could not generate feedback."






def gradio_interface(essay_text, follow_up_question="", feedback_provided=False):
    """Wrapper for the grading system to use with Gradio, handling initial grading and follow-up questions."""
    initial_state = State(
        essay=essay_text,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0
    )

    # Check if the feedback has already been provided
    if not feedback_provided:
        # Sequentially apply grading functions
        initial_state = check_relevance(initial_state)
        initial_state = check_grammar(initial_state)
        initial_state = check_structure(initial_state)
        initial_state = check_depth(initial_state)
        initial_state = cal_final_score(initial_state)

        # Generate natural language feedback
        feedback = generate_natural_language_feedback(initial_state)
        
        # Mark that feedback has now been provided
        feedback_provided = True
        follow_up_response = ""  # No follow-up response allowed until initial feedback is given
        
        # Enable the follow-up question box after feedback
        follow_up_interactive = True
    else:
        feedback = ""  # Feedback should only be generated once
        follow_up_interactive = True

    # If a follow-up question is provided and feedback is already generated, invoke llm2
    follow_up_response = ""
    if follow_up_question.strip() and feedback_provided:
        follow_up_response = generate_follow_up_response(initial_state, follow_up_question)

    return (
        initial_state["relevance_score"],
        initial_state["grammar_score"],
        initial_state["structure_score"],
        initial_state["depth_score"],
        initial_state["final_score"],
        feedback,
        follow_up_response,
        feedback_provided,  # Return the flag to track feedback state
        follow_up_interactive  # Return the interactivity state for follow-up question box
    )

# Set up Gradio app interface with the feedback_provided flag
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter your essay here...", label="Essay"),
        gr.Textbox(lines=2, placeholder="Ask a follow-up question about the feedback...", label="Follow-up Question"),  # Initially disabled
        gr.Checkbox(value=False, visible=False)  # Hidden flag to track feedback state
    ],
    outputs=[
        gr.Number(label="Relevance Score"),
        gr.Number(label="Grammar Score"),
        gr.Number(label="Structure Score"),
        gr.Number(label="Depth Score"),
        gr.Number(label="Final Score"),
        gr.Textbox(label="Feedback"),
        gr.Textbox(label="Follow-up Response"),
        gr.Checkbox(visible=False)  # Hidden flag to track feedback state
    ],
    title="Essay Grading System",
    description="Enter an essay to receive feedback on relevance, grammar, structure, depth, and an overall score. Follow-up questions can only be asked after feedback is provided."
)

if __name__ == "__main__":
    iface.launch(share=True)


