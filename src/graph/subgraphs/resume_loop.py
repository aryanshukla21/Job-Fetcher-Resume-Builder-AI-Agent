"""
Subgraph: Resume Generation Loop
Encapsulates the cyclic generation, scoring, and refinement process.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from src.schemas.state import AgentState

from src.nodes.context_assembler import context_assembler
from src.nodes.resume_drafter import resume_drafter
from src.nodes.ats_scorer import ats_scorer
from src.graph.edges import route_ats

def build_resume_loop() -> StateGraph:
    """Builds the encapsulated loop for iterative drafting."""
    builder = StateGraph(AgentState)
    
    builder.add_node("context_assembler", context_assembler)
    builder.add_node("resume_drafter", resume_drafter)
    builder.add_node("ats_scorer", ats_scorer)
    
    builder.add_edge(START, "context_assembler")
    builder.add_edge("context_assembler", "resume_drafter")
    builder.add_edge("resume_drafter", "ats_scorer")
    
    # Cyclic conditional edge
    builder.add_conditional_edges(
        "ats_scorer",
        route_ats,
        {
            "exit_loop": END,
            "context_assembler": "context_assembler"
        }
    )
    
    return builder

# Compiled subgraph ready to be added as a single node in the main graph
resume_loop_graph = build_resume_loop().compile()