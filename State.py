from typing import TypedDict

class State(TypedDict):
    """Represents the state of the essay grading process."""
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float