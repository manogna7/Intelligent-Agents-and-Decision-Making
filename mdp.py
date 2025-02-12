import numpy as np

GRID_ROWS, GRID_COLS = 4, 4
start_state = (0, 0)
goal_state = (3, 3)

water_cells = [(1, 1), (2, 2)]
wildfire_cells = [(0, 3), (3, 0)]

REWARD_GOAL = 100
REWARD_WATER = -5
REWARD_WILDFIRE = -10
REWARD_DEFAULT = -1 # default penalty for every step to encourage faster solutions

SUCCESS_PROB = 0.8
SLIDE_PROB = 0.1

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_EFFECTS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

def initialize_rewards():
    rewards = np.full((GRID_ROWS, GRID_COLS), REWARD_DEFAULT)
    for (x, y) in water_cells:
        rewards[x, y] = REWARD_WATER
    for (x, y) in wildfire_cells:
        rewards[x, y] = REWARD_WILDFIRE
    rewards[goal_state] = REWARD_GOAL
    return rewards

def is_valid_state(x, y):
    return 0 <= x < GRID_ROWS and 0 <= y < GRID_COLS

def get_transitions(state, action):
    if state == goal_state:
        return [(1.0, state)]
    
    transitions = []
    x, y = state
    dx, dy = ACTION_EFFECTS[action]
    
    next_x, next_y = x + dx, y + dy
    if is_valid_state(next_x, next_y):
        transitions.append((SUCCESS_PROB, (next_x, next_y)))
    else:
        transitions.append((SUCCESS_PROB, state))
    
    for slide_action in get_slide_actions(action):
        slide_dx, slide_dy = ACTION_EFFECTS[slide_action]
        slide_x, slide_y = x + slide_dx, y + slide_dy
        if is_valid_state(slide_x, slide_y):
            transitions.append((SLIDE_PROB, (slide_x, slide_y)))
        else:
            transitions.append((SLIDE_PROB, state))
    
    return transitions

def get_slide_actions(action):
    return ['left', 'right'] if action in ['up', 'down'] else ['up', 'down']

def print_grid(rewards):
    print("Gridworld Rewards:")
    for row in rewards:
        print("\t".join(f"{val:4}" for val in row))
    print()

def setup_environment():
    rewards = initialize_rewards()
    print_grid(rewards)
    print(f"Example transitions from {start_state}:")
    for action in ACTIONS:
        print(f"  Action '{action}':")
        transitions = get_transitions(start_state, action)
        for prob, next_state in transitions:
            print(f"    Moves to {next_state} with probability {prob:.1f}")
    return rewards

def policy_evaluation(policy, gamma, theta=1e-6):
    rewards = initialize_rewards()
    V = np.zeros((GRID_ROWS, GRID_COLS))
    
    while True:
        delta = 0
        for x in range(GRID_ROWS):
            for y in range(GRID_COLS):
                state = (x, y)
                if state == goal_state:
                    continue
                action = policy[x, y]
                new_v = 0
                for prob, next_state in get_transitions(state, action):
                    nx, ny = next_state
                    new_v += prob * (rewards[nx, ny] + gamma * V[nx, ny])
                delta = max(delta, abs(new_v - V[x, y]))
                V[x, y] = new_v
        if delta < theta:
            break
    return V

def policy_iteration(gamma, max_iter=50):
    rewards = initialize_rewards()
    policy = np.random.choice(ACTIONS, (GRID_ROWS, GRID_COLS))
    policy[goal_state] = 'G'
    
    for _ in range(max_iter):
        V = policy_evaluation(policy, gamma)
        policy_stable = True
        
        for x in range(GRID_ROWS):
            for y in range(GRID_COLS):
                state = (x, y)
                if state == goal_state:
                    continue
                old_action = policy[x, y]
                best_action, max_value = None, -np.inf
                
                for action in ACTIONS:
                    action_value = 0
                    for prob, next_state in get_transitions(state, action):
                        nx, ny = next_state
                        action_value += prob * (rewards[nx, ny] + gamma * V[nx, ny])
                    if action_value > max_value:
                        best_action = action
                        max_value = action_value
                
                policy[x, y] = best_action
                if best_action != old_action:
                    policy_stable = False
        if policy_stable:
            break
    return V, policy

def value_iteration(gamma, theta=1e-6):
    rewards = initialize_rewards()
    V = np.zeros((GRID_ROWS, GRID_COLS))
    
    while True:
        delta = 0
        for x in range(GRID_ROWS):
            for y in range(GRID_COLS):
                state = (x, y)
                if state == goal_state:
                    continue
                max_value = -np.inf
                for action in ACTIONS:
                    action_value = 0
                    for prob, next_state in get_transitions(state, action):
                        nx, ny = next_state
                        action_value += prob * (rewards[nx, ny] + gamma * V[nx, ny])
                    max_value = max(max_value, action_value)
                delta = max(delta, abs(max_value - V[x, y]))
                V[x, y] = max_value
        if delta < theta:
            break
    
    policy = np.full((GRID_ROWS, GRID_COLS), None)
    for x in range(GRID_ROWS):
        for y in range(GRID_COLS):
            state = (x, y)
            if state == goal_state:
                policy[x, y] = 'G'
                continue
            best_action, max_value = None, -np.inf
            for action in ACTIONS:
                action_value = 0
                for prob, next_state in get_transitions(state, action):
                    nx, ny = next_state
                    action_value += prob * (rewards[nx, ny] + gamma * V[nx, ny])
                if action_value > max_value:
                    best_action = action
                    max_value = action_value
            policy[x, y] = best_action
    return V, policy

if __name__ == "__main__":
    setup_environment()
    
    for gamma in [0.3, 0.95]:
        V_vi, policy_vi = value_iteration(gamma)
        print(f"\nValue Iteration Results (γ = {gamma}):")
        print("Policy:")
        for row in policy_vi:
            print("\t".join([str(a) if a != 'G' else 'G' for a in row]))
            np.set_printoptions(suppress=True)
        print("\nValue Function:")
        print(V_vi.round(2))
    
    V_pi, policy_pi = policy_iteration(0.95)
    print("\nPolicy Iteration Results (γ = 0.95):")
    print("Policy:")
    for row in policy_pi:
        print("\t".join([str(a) if a != 'G' else 'G' for a in row]))
    print("\nValue Function:")
    print(V_pi.round(2))
