"""Small worked example of evidence-aware role and action selection."""

from kanm.collaborative import CollaborativeAction, optimal_collaborative_plan
from kanm.dependence import ProvenanceNetwork, corroborating_relay, relay, symmetric_primary


def build_example():
    network = ProvenanceNetwork()
    network.add(symmetric_primary("pilot", 0.72, "Initial study"))
    network.add(relay("summary", "pilot", 0.98, "A derivative literature summary"))
    network.add(symmetric_primary("replication", 0.78, "Independent replication"))
    network.add(corroborating_relay(
        "audit", "pilot", fidelity=0.98, own_reliability=0.90, mix=0.65,
        description="Partly independent methods audit",
    ))
    actions = (
        CollaborativeAction("pilot", "pilot", "initial_study", cost=0.15),
        CollaborativeAction("summary", "summary", "literature_summary", cost=0.05,
                            requires=("pilot",)),
        CollaborativeAction("replication", "replication", "independent_replication",
                            cost=0.70, requires=("pilot",)),
        CollaborativeAction("audit", "audit", "methods_audit", cost=0.35,
                            requires=("pilot",)),
    )
    utilities = {
        "adopt": {True: 1.0, False: -2.0},
        "reject": {True: 0.0, False: 0.0},
    }
    return network, actions, utilities


def main():
    network, actions, utilities = build_example()
    action_by_id = {action.action_id: action for action in actions}
    for mode in ("joint", "independent"):
        plan = optimal_collaborative_plan(
            0.5, network, actions, utilities, horizon=3,
            cost_weight=0.5, inference=mode,
        )
        follow_up = optimal_collaborative_plan(
            0.5, network, actions, utilities, horizon=2,
            cost_weight=0.5, initial_observations={"pilot": True}, inference=mode,
        )
        first_role = action_by_id[plan.first_action].role_id if plan.first_action else "stop"
        follow_up_role = (
            action_by_id[follow_up.first_action].role_id if follow_up.first_action else "stop"
        )
        print(
            f"{mode:>11} policy: first role={first_role}; "
            f"after positive pilot={follow_up_role}; "
            f"states={plan.evaluated_states}"
        )


if __name__ == "__main__":
    main()
