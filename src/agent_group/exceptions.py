# src/agent_group/exceptions.py

class PhaseFailure(Exception):
    """Custom exception for when a phase in the CollaborativeAgentGroup fails after all retries."""
    pass
