"""KANM: research software for Laputa-style epistemic networks and extensions."""

__version__ = "0.10.0"

from .bayes import ExactJointState, TrustDistribution, update_belief, update_log_odds
from .collaborative import CollaborativeAction, CollaborativePlan, optimal_collaborative_plan
from .model import AgentSpec, EdgeSpec, Simulation, SimulationResult
from .testimony import UrnTaskOutcome, simulate_urn_task
from .trust import TrustState

__all__ = [
    "AgentSpec",
    "CollaborativeAction",
    "CollaborativePlan",
    "EdgeSpec",
    "ExactJointState",
    "Simulation",
    "SimulationResult",
    "TrustDistribution",
    "TrustState",
    "UrnTaskOutcome",
    "update_belief",
    "update_log_odds",
    "simulate_urn_task",
    "optimal_collaborative_plan",
]
