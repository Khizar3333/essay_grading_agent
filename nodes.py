from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from State import State

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import re


load_dotenv()


llm2 =ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="mixtral-8x7b-32768") 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def generate_natural_language_feedback(state: State) -> str:
    """Generate natural language feedback for the essay using llm2."""
    prompt = (
        f"The essay was graded, and the results are as follows:\n\n"
        f"Relevance: {state['relevance_score']:.2f}\n"
        f"Grammar: {state['grammar_score']:.2f}\n"
        f"Structure: {state['structure_score']:.2f}\n"
        f"Depth: {state['depth_score']:.2f}\n\n"
        f"Based on these scores, summarize the strengths and weaknesses of the essay "
        f"in a conversational and encouraging tone in 3 lines and also tell the final score along with other scores. Provide suggestions for improvement."
        f"only give the feedback related to essay dont answer anything else but if user ask a follow up question related to the essay give a response"
    )

    result = llm2.invoke(prompt)
    response=result.content

    if hasattr(result, "content") and result.content:
        return result.content  # Return content if valid

    # Handle cases where result is a string or dictionary with 'generated_text' key
    if isinstance(result, str):
        return result
    elif isinstance(result, dict) and 'generated_text' in result:
        return result['generated_text']

    # Fallback for unexpected result structure
    return "Unexpected response structure. Could not generate feedback."


def score(content: str) -> float:
    """Extract the numeric score from the LLM's response."""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract score from: {content}")

def check_relevance(state: State) -> State:
    """Check the relevance of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevance of the following essay to the given topic. "
        "Provide a relevance score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm2.invoke(prompt.format(essay=state["essay"]))
    try:
        state["relevance_score"] = score(result.content)
    except ValueError as e:
        print(f"Error in check_relevance: {e}")
        state["relevance_score"] = 0.0
    return state

def check_grammar(state: State) -> State:
    """Check the grammar of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the grammar of the following essay. "
        "Provide a grammar score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm2.invoke(prompt.format(essay=state["essay"]))
    try:
        state["grammar_score"] = score(result.content)
    except ValueError as e:
        print(f"Error in check_grammar: {e}")
        state["grammar_score"] = 0.0
    return state

def check_structure(state: State) -> State:
    """Check the structure of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the structure of the following essay. "
        "Provide a structure score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm2.invoke(prompt.format(essay=state["essay"]))
    try:
        state["structure_score"] = score(result.content)
    except ValueError as e:
        print(f"Error in check_structure: {e}")
        state["structure_score"] = 0.0
    return state

def check_depth(state: State) -> State:
    """Check the depth of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the depth of the following essay. "
        "Provide a depth score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )    
    result = llm2.invoke(prompt.format(essay=state["essay"]))
    try:
        state["depth_score"] = score(result.content)
    except ValueError as e:
        print(f"Error in check_depth: {e}")
        state["depth_score"] = 0.0
    return state

def cal_final_score(state: State) -> State:
    """Calculate the final score based on individual component scores."""
    state["final_score"] = (
        state["relevance_score"] * 0.3 +
        state["grammar_score"] * 0.2 +
        state["structure_score"] * 0.2 +
        state["depth_score"] * 0.3
    )
    return state