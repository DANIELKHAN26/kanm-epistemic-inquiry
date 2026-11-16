"""Cost-aware inquiry planning: information, decision value, stopping, scope drift."""

from kanm.inquiry import (
    QueryAction,
    expected_information_gain,
    expected_value_of_sample_information,
    optimal_sequential_plan,
    question_relevance,
    select_action,
    select_action_by_value,
    select_action_or_stop,
    update_hypothesis_posterior,
)


prior = {"policy_works": 0.5, "policy_fails": 0.5}
actions = [
    QueryAction("pilot-study", "goal", "methods-team", ("works", "fails"),
                {"policy_works": (0.85, 0.15), "policy_fails": (0.15, 0.85)}, cost=3.0),
    QueryAction("small-survey", "goal", "survey-team", ("works", "fails"),
                {"policy_works": (0.7, 0.3), "policy_fails": (0.3, 0.7)}, cost=0.5),
]
for action in actions:
    print(action.action_id, "EIG (nats) =", round(expected_information_gain(prior, action), 4))

selected, scores = select_action(prior, actions, cost_weight=0.1)
print("information-based choice:", selected.action_id, {k: round(v, 4) for k, v in scores.items()})
print("posterior after 'works':", update_hypothesis_posterior(prior, selected, "works"))

goal_utility = {
    "adopt": {"policy_works": 10.0, "policy_fails": 0.0},
    "reject": {"policy_works": 0.0, "policy_fails": 7.0},
}
best, value_scores = select_action_by_value(prior, actions, goal_utility, cost_weight=0.1)
print("decision-value choice:", best.action_id, {k: round(v, 4) for k, v in value_scores.items()})
print("pilot-study EVSI:", round(expected_value_of_sample_information(prior, actions[0], goal_utility), 4))

# Myopic versus exact sequential planning: one weak test cannot change the
# decision, but two can (non-concavity of the value of information).
weak = QueryAction("weak-test", "goal", "lab", ("yes", "no"),
                   {"T": (0.55, 0.45), "F": (0.45, 0.55)}, cost=1.0)
utilities = {"adopt": {"T": 1.0, "F": -3.0}, "reject": {"T": 0.0, "F": 0.0}}
myopic, _ = select_action_or_stop({"T": 0.8, "F": 0.2}, [weak], utilities, cost_weight=0.005)
plan = optimal_sequential_plan({"T": 0.8, "F": 0.2}, [weak], utilities, horizon=3,
                               cost_weight=0.005)
print("myopic rule:", myopic and myopic.action_id, "| 3-step plan first action:",
      plan.first_action, "| value of planning:", round(plan.value_of_planning, 4))

# Scope drift: a proxy question can be perfectly informative about itself and
# carry no information about the original goal.
states = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
goal = {"a": "adopt", "b": "adopt", "c": "reject", "d": "reject"}
proxy = {"a": "x", "b": "y", "c": "x", "d": "y"}
proxy_test = QueryAction("proxy", "sub", "h", ("x", "y"),
                         {h: (1.0, 0.0) if proxy[h] == "x" else (0.0, 1.0) for h in states})
print("proxy question relevance:", question_relevance(states, proxy_test, goal, proxy))
