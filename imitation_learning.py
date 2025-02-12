import numpy as np
import random
import mdp

def simulate_episode_fixed(policy, gamma=0.95, start_state=mdp.start_state):
    state = start_state
    rewards = mdp.initialize_rewards()
    episode = []
    while True:
        action = policy.get(state)
        if action is None:
            break
        transitions = mdp.get_transitions(state, action)
        probs, next_states = zip(*transitions)
        next_state = random.choices(next_states, weights=probs, k=1)[0]
        reward = int(rewards[next_state])
        episode.append((state, action, reward))
        if next_state == mdp.goal_state:
            episode.append((mdp.goal_state, None, mdp.REWARD_GOAL))
            break
        state = next_state
    return episode

example_policy = {
    (0, 0): 'right', (0, 1): 'right', (0, 2): 'right', (0, 3): 'down',
    (1, 0): 'right', (1, 1): 'right', (1, 2): 'right', (1, 3): 'down',
    (2, 0): 'right', (2, 1): 'right', (2, 2): 'right', (2, 3): 'down',
    (3, 0): 'right', (3, 1): 'right', (3, 2): 'right', (3, 3): None
}

episode = simulate_episode_fixed(example_policy)

print("\nSimulated Episode:")
for step in episode:
    print(f"State: {step[0]}, Action: {step[1]}, Reward: {step[2]}")
