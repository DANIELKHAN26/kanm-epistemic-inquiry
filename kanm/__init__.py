"""KANM: research software for Laputa-style epistemic networks and extensions."""

__version__ = "0.11.0"

from .bayes import ExactJointState, TrustDistribution, update_belief, update_log_odds
from .collaborative import CollaborativeAction, CollaborativePlan, optimal_collaborative_plan
from .model import AgentSpec, EdgeSpec, Simulation, SimulationResult
from .model_uncertainty import ProvenanceModelSet
from .robust_planning import robust_collaborative_plan, value_of_provenance_information
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
    "ProvenanceModelSet",
    "TrustDistribution",
    "TrustState",
    "UrnTaskOutcome",
    "update_belief",
    "update_log_odds",
    "simulate_urn_task",
    "optimal_collaborative_plan",
    "robust_collaborative_plan",
    "value_of_provenance_information",
]
