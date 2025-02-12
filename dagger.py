import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

import mdp as mdp
import imitation_learning as il

GRID_ROWS, GRID_COLS = mdp.GRID_ROWS, mdp.GRID_COLS
ACTIONS = mdp.ACTIONS
GOAL_STATE = mdp.goal_state

def dagger_algorithm(expert_policy, N_values):
    state_space = [(x, y) for x in range(GRID_ROWS) for y in range(GRID_COLS)]
    learned_policy = {state: random.choice(ACTIONS) for state in state_space if state != GOAL_STATE}
    learned_policy[GOAL_STATE] = None
    D = []
    accuracies = []
    max_N = max(N_values)

    for i in range(1, max_N + 1):
        trajectory = il.simulate_episode_fixed(learned_policy)
        D.extend([(state, expert_policy[state]) for (state, _, _) in trajectory if state != GOAL_STATE])

        X_train = [list(state) for state, _ in D]
        y_train = [ACTIONS.index(action) for _, action in D]
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        for state in state_space:
            if state != GOAL_STATE:
                action_idx = clf.predict([list(state)])[0]
                learned_policy[state] = ACTIONS[action_idx]

        if i in N_values:
            accuracy = sum(1 for state in state_space if learned_policy[state] == expert_policy[state]) / len(state_space)
            accuracies.append(accuracy)

    return accuracies

V_vi_095, policy_vi_095 = mdp.value_iteration(gamma=0.95)
expert_policy = {(x, y): policy_vi_095[x, y] for x in range(GRID_ROWS) for y in range(GRID_COLS)}

N_values = [5, 10, 20, 30, 40, 50]
accuracies = dagger_algorithm(expert_policy, N_values)

plt.figure(figsize=(10, 6))
plt.plot(N_values, accuracies, marker='o')
plt.title('DAgger Algorithm: Accuracy vs Number of Iterations')
plt.xlabel('Number of Iterations (N)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
