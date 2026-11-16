"""KANM: research software for Laputa-style epistemic networks and extensions."""

__version__ = "0.5.0"

from .bayes import ExactJointState, TrustDistribution, update_belief, update_log_odds
from .model import AgentSpec, EdgeSpec, Simulation, SimulationResult

__all__ = [
    "AgentSpec",
    "EdgeSpec",
    "ExactJointState",
    "Simulation",
    "SimulationResult",
    "TrustDistribution",
    "update_belief",
    "update_log_odds",
]
