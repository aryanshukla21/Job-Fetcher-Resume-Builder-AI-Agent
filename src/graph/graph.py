"""
Top-level LangGraph compilation and orchestration.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from src.config import settings
from src.schemas.state import AgentState

# Nodes
from src.nodes.context_builder import context_builder
from src.nodes.store_context import store_context
from src.nodes.job_fetcher import job_fetcher
from src.nodes.relevance_scorer import relevance_scorer
from src.nodes.append_jobs_to_db import append_jobs_to_db
from src.nodes.job_hitl import job_hitl
from src.nodes.spawn_resume_worker import spawn_resume_worker
from src.nodes.archive_job import archive_job
from src.nodes.select_top2 import select_top2
from src.nodes.resume_hitl import resume_hitl
from src.nodes.retry_planner import retry_planner
from src.nodes.render_resume import render_resume

# Edges & Subgraph
from src.graph.edges import route_job_decision, route_resume_decision
from src.graph.subgraphs.resume_loop import resume_loop_graph


def build_main_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # 1. Register Main Graph Nodes
    builder.add_node("context_builder", context_builder)
    builder.add_node("store_context", store_context)
    builder.add_node("job_fetcher", job_fetcher)
    builder.add_node("relevance_scorer", relevance_scorer)
    
    # Alias the DB sync node to enforce persistence checkpoints
    builder.add_node("db_sync_fetch", append_jobs_to_db)
    builder.add_node("db_sync_archive", append_jobs_to_db)
    builder.add_node("db_sync_approve", append_jobs_to_db)
    builder.add_node("db_sync_final", append_jobs_to_db)
    
    builder.add_node("job_hitl", job_hitl)
    builder.add_node("spawn_resume_worker", spawn_resume_worker)
    builder.add_node("archive_job", archive_job)
    
    # Register the Subgraph as a single node
    builder.add_node("resume_loop", resume_loop_graph)
    
    builder.add_node("select_top2", select_top2)
    builder.add_node("resume_hitl", resume_hitl)
    builder.add_node("retry_planner", retry_planner)
    builder.add_node("render_resume", render_resume)

    # 2. Linear Fetching Pipeline
    builder.add_edge(START, "context_builder")
    builder.add_edge("context_builder", "store_context")
    builder.add_edge("store_context", "job_fetcher")
    builder.add_edge("job_fetcher", "relevance_scorer")
    builder.add_edge("relevance_scorer", "db_sync_fetch")
    builder.add_edge("db_sync_fetch", "job_hitl")

    # 3. Job Routing
    builder.add_conditional_edges(
        "job_hitl",
        route_job_decision,
        {
            "spawn_resume_worker": "spawn_resume_worker",
            "archive_job": "archive_job",
            "end": END
        }
    )

    # 4. Branches
    builder.add_edge("archive_job", "db_sync_archive")
    builder.add_edge("db_sync_archive", END)

    builder.add_edge("spawn_resume_worker", "db_sync_approve")
    builder.add_edge("db_sync_approve", "resume_loop")
    
    # Exiting the Subgraph
    builder.add_edge("resume_loop", "select_top2")
    builder.add_edge("select_top2", "resume_hitl")
    
    # 5. Resume Routing
    builder.add_conditional_edges(
        "resume_hitl",
        route_resume_decision,
        {
            "render_resume": "render_resume",
            "retry_planner": "retry_planner",
            "end": END
        }
    )

    builder.add_edge("retry_planner", "resume_loop")
    builder.add_edge("render_resume", "db_sync_final")
    builder.add_edge("db_sync_final", END)

    return builder

def compile_agent():
    """
    Compiles the main StateGraph with PostgreSQL checkpointer.
    """
    pool = ConnectionPool(conninfo=settings.state_database_url, max_size=10)
    checkpointer = PostgresSaver(pool)
    
    builder = build_main_graph()
    
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["job_hitl", "resume_hitl"]
    )