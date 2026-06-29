"""
Execution entry point and CLI test harness for the AI Job Agent.
Manages the StateGraph thread and handles HITL interrupts.
"""
from __future__ import annotations

import sys
import uuid
from typing import Any

from src.db.job_store import init_db as init_job_db
# Assuming user_store exists as designed previously
# from src.db.user_store import upsert_user, get_user 
from src.graph.graph import compile_agent
from src.schemas.models import UserConfig, WorkType

def setup_test_user() -> uuid.UUID:
    """
    Mocks a test user in the PostgreSQL database if one does not exist.
    In production, this is handled by a frontend signup flow.
    """
    test_user_id = uuid.uuid5(uuid.NAMESPACE_DNS, "test-user")
    
    # Example config injection
    config = UserConfig(
        github_username="octocat",
        base_resume_path="tests/fixtures/base_resume.docx",
        target_roles=["Senior Software Engineer", "Backend Engineer"],
        manual_skills=["Python", "PostgreSQL", "LangChain", "AWS"],
        salary_min=120000,
        salary_max=180000,
        work_type=WorkType.REMOTE,
        location="usa",
        max_iterations=5,
        ats_threshold=0.85,
        platforms=["linkedin", "indeed"]
    )
    
    # upsert_user(test_user_id, config)
    # return test_user_id
    
    # For CLI purposes without the DB fully wired, we just return the config directly
    return test_user_id, config

def run_agent_loop() -> None:
    """
    Executes the LangGraph state machine and handles CLI-based HITL interrupts.
    """
    print("Initializing Database...")
    init_job_db()
    
    print("Compiling Graph...")
    agent = compile_agent()
    
    user_id, user_config = setup_test_user()
    
    thread_config = {"configurable": {"thread_id": str(user_id)}}
    
    # Inject initial state
    initial_state = {
        "user_config": user_config
    }
    
    print(f"\n[STARTING RUN] Thread ID: {user_id}")
    
    # Standard LangGraph execution loop
    for event in agent.stream(initial_state, thread_config, stream_mode="updates"):
        for node_name, state_update in event.items():
            print(f"\n--- Output from node: {node_name} ---")
            if "last_error" in state_update and state_update["last_error"]:
                print(f"[CRITICAL ERROR] {state_update['last_error']}")
                sys.exit(1)
    
    # Infinite loop to handle interrupts
    while True:
        state = agent.get_state(thread_config)
        
        # If there are no pending tasks, the graph has reached END
        if not state.next:
            print("\n[FINISHED] Graph execution completed.")
            break
            
        print(f"\n[INTERRUPT] Graph paused at node(s): {state.next}")
        
        # Determine which HITL node paused the graph
        if "job_hitl" in state.next:
            print("\n>>> JOB APPROVAL REQUIRED")
            decision = input("Decision (approved/rejected): ").strip().lower()
            if decision not in ["approved", "rejected"]:
                print("Invalid input. Defaulting to rejected.")
                decision = "rejected"
            
            # Update state with the user's decision
            agent.update_state(thread_config, {"user_decision": decision})
            
        elif "resume_hitl" in state.next:
            print("\n>>> RESUME APPROVAL REQUIRED")
            decision = input("Decision (approved/retry/cancel): ").strip().lower()
            
            feedback = None
            if decision == "retry":
                feedback = input("Provide retry feedback for the planner: ").strip()
                agent.update_state(thread_config, {
                    "user_decision": decision, 
                    "rejection_feedback": feedback
                })
            elif decision in ["approved", "cancel"]:
                agent.update_state(thread_config, {"user_decision": decision})
            else:
                print("Invalid input. Defaulting to cancel.")
                agent.update_state(thread_config, {"user_decision": "cancel"})
                
        else:
            # Unhandled interrupt
            print(f"[WARN] Unknown interrupt at {state.next}. Forcing continue.")
            
        # Resume graph execution
        print("\n[RESUMING GRAPH]")
        for event in agent.stream(None, thread_config, stream_mode="updates"):
            for node_name, state_update in event.items():
                print(f"\n--- Output from node: {node_name} ---")

if __name__ == "__main__":
    try:
        run_agent_loop()
    except KeyboardInterrupt:
        print("\n[EXIT] Execution aborted by user.")
        sys.exit(0)