import random


states = ["S1", "S2", "S3", "S4", "S5", "G"]

# 2 x 3 GridWorld
grid = [
    ["S1", "S2", "S3"],
    ["S4", "S5", "G"]
]

coordinates = {
    "S1": (0, 0),
    "S2": (0, 1),
    "S3": (0, 2),
    "S4": (1, 0),
    "S5": (1, 1),
    "G":  (1, 2)
}

actions = ["U", "D", "L", "R"]

gamma = 0.9


def step(state, action):

    if state == "G":
        return "G", 0, True

    row, col = coordinates[state]

    movement = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1)
    }

    dr, dc = movement[action]

    new_row = row + dr
    new_col = col + dc

    # Check whether the movement is valid
    valid = False

    for s, (r, c) in coordinates.items():
        if r == new_row and c == new_col:
            next_state = s
            valid = True
            break

    # Invalid movement: remain in same state
    if not valid:
        next_state = state

    # Reward
    if next_state == "G":
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done




# Given policy
evaluated_policy = {
    "S1": "R",
    "S2": "R",
    "S3": "L",
    "S4": "R",
    "S5": "R"
}

V = {state: 0.0 for state in states}

print("\nPOLICY EVALUATION")
print("-" * 70)

for iteration in range(1, 6):

    new_V = V.copy()
    delta = 0

    for state in states:

        if state == "G":
            new_V[state] = 0
            continue

        action = evaluated_policy[state]

        next_state, reward, done = step(state, action)

        if done:
            value = reward
        else:
            value = reward + gamma * V[next_state]

        new_V[state] = value

        delta = max(delta, abs(value - V[state]))

    V = new_V

    print(
        f"Iteration {iteration}: "
        f"Delta = {delta:.4f}, "
        f"Values = {V}"
    )

V_eval = {state: 0.0 for state in states}

threshold = 0.001
iteration = 0

while True:

    iteration += 1

    new_V = V_eval.copy()
    delta = 0

    for state in states:

        if state == "G":
            new_V[state] = 0
            continue

        action = evaluated_policy[state]

        next_state, reward, done = step(state, action)

        if done:
            value = reward
        else:
            value = reward + gamma * V_eval[next_state]

        new_V[state] = value

        delta = max(delta, abs(value - V_eval[state]))

    V_eval = new_V

    if delta < threshold:
        break

print("\nPolicy Evaluation Converged")
print("Iterations:", iteration)
print("Final Values:")

for state in states:
    print(state, round(V_eval[state], 4))



V = {state: 0.0 for state in states}

print("\nVALUE ITERATION")
print("-" * 70)

for iteration in range(1, 100):

    new_V = V.copy()
    delta = 0

    for state in states:

        if state == "G":
            new_V[state] = 0
            continue

        action_values = []

        for action in actions:

            next_state, reward, done = step(state, action)

            if done:
                value = reward
            else:
                value = reward + gamma * V[next_state]

            action_values.append(value)

        best_value = max(action_values)

        new_V[state] = best_value

        delta = max(
            delta,
            abs(best_value - V[state])
        )

    V = new_V

    print(
        f"Iteration {iteration}: "
        f"Delta = {delta:.4f}, "
        f"Values = {V}"
    )

    if delta < 0.001:
        break



optimal_policy = {}

for state in states:

    if state == "G":
        optimal_policy[state] = "-"
        continue

    best_action = None
    best_value = float("-inf")

    for action in actions:

        next_state, reward, done = step(state, action)

        if done:
            value = reward
        else:
            value = reward + gamma * V[next_state]

        # Tie-breaking preference: R, D, L, U
        if value > best_value:
            best_value = value
            best_action = action

    optimal_policy[state] = best_action


print("\nOPTIMAL POLICY")
print("-" * 40)

for state in states:
    print(
        state,
        "->",
        optimal_policy[state],
        " Value = ",
        round(V[state], 4)
    )




arrow = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→",
    "-": "G"
}

print("\nOPTIMAL POLICY GRID")
print("-" * 40)

for row in grid:

    line = ""

    for state in row:

        if state == "G":
            line += "[ G ] "
        else:
            line += (
                "[ " +
                state +
                " " +
                arrow[optimal_policy[state]] +
                " ] "
            )

    print(line)



def run_policy(policy_type, max_steps=50):

    state = "S1"
    path = [state]
    total_reward = 0

    for step_number in range(max_steps):

        if state == "G":
            return path, step_number, total_reward, True

        if policy_type == "random":

            action = random.choice(actions)

        elif policy_type == "evaluated":

            action = evaluated_policy[state]

        elif policy_type == "optimal":

            action = optimal_policy[state]

        next_state, reward, done = step(
            state,
            action
        )

        total_reward += reward

        state = next_state
        path.append(state)

        if done:
            return (
                path,
                step_number + 1,
                total_reward,
                True
            )

    return (
        path,
        max_steps,
        total_reward,
        False
    )




random.seed(42)

random_path, random_steps, random_reward, random_goal = \
    run_policy("random")

print("\nRANDOM POLICY")
print("Path:", " -> ".join(random_path))
print("Steps:", random_steps)
print("Total Reward:", random_reward)
print("Goal Reached:", random_goal)




evaluated_path, evaluated_steps, evaluated_reward, evaluated_goal = \
    run_policy("evaluated")

print("\nEVALUATED POLICY")
print("Path:", " -> ".join(evaluated_path))
print("Steps:", evaluated_steps)
print("Total Reward:", evaluated_reward)
print("Goal Reached:", evaluated_goal)




optimal_path, optimal_steps, optimal_reward, optimal_goal = \
    run_policy("optimal")

print("\nOPTIMAL POLICY")
print("Path:", " -> ".join(optimal_path))
print("Steps:", optimal_steps)
print("Total Reward:", optimal_reward)
print("Goal Reached:", optimal_goal)